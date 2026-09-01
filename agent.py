"""
agent.py
========
The orchestration loop. Each function below is one box in the
"What is Agentic RAG?" diagram. run_agentic_rag() wires them together
exactly the way the diagram's arrows do, including the retry loop —
capped by MAX_RETRIES so "No -> retry" can never spin forever.

This module doesn't know about Streamlit; it only depends on
llm_providers.LLMRouter and retrieval.VectorStore / tavily_search, so
it can be tested or reused outside the GUI.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from llm_providers import LLMRouter
from retrieval import VectorStore, tavily_search

MAX_RETRIES = 3
VALID_SOURCES = ("vector_db", "web_search")


@dataclass
class TraceStep:
    label: str
    detail: str
    provider: str | None = None
    fell_back: bool = False


@dataclass
class AgentResult:
    answer: str
    trace: list[TraceStep] = field(default_factory=list)
    attempts: int = 0


def _ask(router: LLMRouter, prompt: str, system: str, trace: list[TraceStep], label: str, max_tokens: int = 512) -> str:
    result = router.call(prompt, system=system, max_tokens=max_tokens)
    trace.append(TraceStep(label=label, detail=result.text.strip(), provider=result.provider_used, fell_back=result.fell_back))
    return result.text.strip()


def rewrite_query(router: LLMRouter, query: str, trace: list[TraceStep]) -> str:
    system = "You rewrite user questions to be clearer and more specific, in one sentence. Reply with only the rewritten question."
    rewritten = _ask(router, query, system, trace, "Rewrite query")
    return rewritten or query


def decide_if_retrieval_needed(router: LLMRouter, query: str, has_kb: bool, has_web: bool, trace: list[TraceStep]) -> bool:
    if not has_kb and not has_web:
        trace.append(TraceStep("Need retrieval?", "no sources configured -> answering from the model directly"))
        return False
    system = (
        "Decide if answering this question requires looking up external information "
        "(facts, current events, specifics not in general knowledge). Reply with only 'yes' or 'no'."
    )
    answer = _ask(router, query, system, trace, "Need retrieval?", max_tokens=150)
    return answer.lower().startswith("y")


def choose_source(router: LLMRouter, query: str, has_kb: bool, has_web: bool, trace: list[TraceStep]) -> str:
    if has_kb and not has_web:
        trace.append(TraceStep("Choose source", "only the document knowledge base is configured"))
        return "vector_db"
    if has_web and not has_kb:
        trace.append(TraceStep("Choose source", "only web search is configured"))
        return "web_search"
    system = (
        "Choose the best source to answer this question: 'vector_db' (uploaded documents) "
        "or 'web_search' (live internet search). Reply with only one of those two words."
    )
    answer = _ask(router, query, system, trace, "Choose source", max_tokens=150).lower()
    for s in VALID_SOURCES:
        if s.replace("_", "") in answer.replace("_", "").replace(" ", ""):
            return s
    return "vector_db"


def retrieve(
    source: str,
    query: str,
    vector_store: VectorStore,
    gemini_provider,
    tavily_key: str | None,
    deep_scrape: bool,
    trace: list[TraceStep],
) -> str:
    if source == "vector_db":
        hits = vector_store.search(query, k=4, gemini_provider=gemini_provider)
        if not hits:
            trace.append(TraceStep("Retrieve", "no relevant chunks found in the document knowledge base"))
            return ""
        context = "\n\n".join(f"[{h.source}] {h.text}" for h in hits)
        trace.append(TraceStep("Retrieve", f"{len(hits)} chunk(s) from documents:\n{context[:600]}"))
        return context

    if source == "web_search":
        if not tavily_key:
            trace.append(TraceStep("Retrieve", "web search requested but no Tavily key configured"))
            return ""
        results = tavily_search(tavily_key, query, max_results=4, deep_scrape=deep_scrape)
        if not results:
            trace.append(TraceStep("Retrieve", "Tavily search returned no results"))
            return ""
        context = "\n\n".join(f"[{r['title']}]({r['url']}) {r['content']}" for r in results)
        mode = "scraped pages" if deep_scrape else "search snippets"
        trace.append(TraceStep("Retrieve", f"{len(results)} {mode} from Tavily:\n{context[:600]}"))
        return context

    return ""


def generate_answer(router: LLMRouter, query: str, context: str, trace: list[TraceStep]) -> str:
    if context:
        system = (
            "Answer the user's question using ONLY the context provided. If the context doesn't "
            "contain the answer, say so honestly instead of guessing. Be concise."
        )
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
    else:
        system = "Answer the user's question directly and concisely, using your own knowledge."
        prompt = query
    return _ask(router, prompt, system, trace, "Generate answer", max_tokens=700)


def check_answer(router: LLMRouter, query: str, context: str, answer: str, trace: list[TraceStep]) -> bool:
    system = (
        "You are reviewing an AI-generated answer for correctness and groundedness. "
        "If context was provided, the answer must be supported by it. Reply with only 'yes' or 'no'."
    )
    prompt = f"Question: {query}\n\nContext:\n{context or '(none provided)'}\n\nAnswer to review: {answer}"
    verdict = _ask(router, prompt, system, trace, "Self-check", max_tokens=150)
    return verdict.lower().startswith("y")


def run_agentic_rag(
    query: str,
    router: LLMRouter,
    vector_store: VectorStore,
    gemini_provider=None,
    tavily_key: str | None = None,
    deep_scrape: bool = False,
) -> AgentResult:
    trace: list[TraceStep] = []
    has_kb = not vector_store.is_empty()
    has_web = bool(tavily_key)

    current_query = query
    attempts = 0
    last_answer = ""

    while attempts < MAX_RETRIES:
        attempts += 1
        trace.append(TraceStep(f"--- attempt {attempts} ---", ""))

        current_query = rewrite_query(router, current_query, trace)
        needs_retrieval = decide_if_retrieval_needed(router, current_query, has_kb, has_web, trace)

        context = ""
        if needs_retrieval:
            source = choose_source(router, current_query, has_kb, has_web, trace)
            context = retrieve(source, current_query, vector_store, gemini_provider, tavily_key, deep_scrape, trace)

        last_answer = generate_answer(router, current_query, context, trace)

        if check_answer(router, current_query, context, last_answer, trace):
            return AgentResult(answer=last_answer, trace=trace, attempts=attempts)

    trace.append(TraceStep("Max retries reached", "returning the last answer generated"))
    return AgentResult(answer=last_answer, trace=trace, attempts=attempts)