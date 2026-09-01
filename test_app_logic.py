"""
tests/test_app_logic.py
========================
Tests everything that is UNDER THIS APP'S CONTROL: fallback routing,
error diagnosis, chunking, the vector store, and the full agent
orchestration loop. None of these hit a real network — they use fake
providers — so they'll keep passing even if Gemini/Grok/Tavily change
their APIs again. That's the boundary: this suite guarantees the
app's own logic is correct; it can't guarantee a third-party API
won't rename a model tomorrow (nothing can — that's why the sidebar
has "Test connection" buttons for a live, immediate check).

Run with:
    pip install pytest
    pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from llm_providers import LLMRouter, ProviderError, _diagnose
from retrieval import VectorStore, chunk_text
from agent import run_agentic_rag, MAX_RETRIES


# Fakes — stand in for GeminiProvider without any network call

class FakeProvider:
    def __init__(self, name, behavior):
        self.name = name
        self.behavior = behavior  # callable(prompt, system, max_tokens) -> str, or raises

    def generate(self, prompt, system=None, max_tokens=1024):
        return self.behavior(prompt, system, max_tokens)


def always_fails(exc):
    def _f(prompt, system, max_tokens):
        raise ProviderError("fake", exc)
    return _f


# LLMRouter fallback behavior
def test_primary_success_no_fallback():
    router = LLMRouter(FakeProvider("p", lambda p, s, m: "hi"), FakeProvider("s", lambda p, s, m: "unused"))
    result = router.call("hello")
    assert result.text == "hi"
    assert result.provider_used == "p"
    assert result.fell_back is False


def test_primary_fails_secondary_succeeds():
    primary = FakeProvider("p", always_fails(RuntimeError("boom")))
    secondary = FakeProvider("s", lambda p, s, m: "backup answer")
    router = LLMRouter(primary, secondary)
    result = router.call("hello")
    assert result.text == "backup answer"
    assert result.provider_used == "s"
    assert result.fell_back is True
    assert "boom" in result.primary_error


def test_both_fail_raises_combined_error():
    primary = FakeProvider("p", always_fails(RuntimeError("primary down")))
    secondary = FakeProvider("s", always_fails(RuntimeError("secondary down")))
    router = LLMRouter(primary, secondary)
    with pytest.raises(RuntimeError) as exc_info:
        router.call("hello")
    msg = str(exc_info.value)
    assert "primary down" in msg
    assert "secondary down" in msg


def test_no_secondary_configured_raises_immediately():
    primary = FakeProvider("p", always_fails(RuntimeError("down")))
    router = LLMRouter(primary, None)
    with pytest.raises(RuntimeError):
        router.call("hello")


def test_fallback_is_single_shot_not_a_loop():
    """Regression test for the explicit 'no loop' requirement: each provider
    is called exactly once per router.call(), never retried."""
    calls = {"p": 0, "s": 0}

    def failing(name):
        def _f(prompt, system, max_tokens):
            calls[name] += 1
            raise ProviderError(name, RuntimeError("fail"))
        return _f

    router = LLMRouter(FakeProvider("p", failing("p")), FakeProvider("s", failing("s")))
    with pytest.raises(RuntimeError):
        router.call("hello")
    assert calls["p"] == 1
    assert calls["s"] == 1


# Error diagnosis
@pytest.mark.parametrize("message,expected_keyword", [
    ("Incorrect API key provided", "wrong or unset"),
    ("403 permission-denied: spending limit reached", "quota"),
    ("404 model not found, no longer available", "model"),
    ("429 rate limit exceeded", "Rate limited"),
    ("Connection timeout", "reach"),
])
def test_diagnose_gives_actionable_hints(message, expected_keyword):
    hint = _diagnose(RuntimeError(message))
    assert expected_keyword.lower() in hint.lower()


def test_diagnose_returns_empty_for_unknown_errors():
    assert _diagnose(RuntimeError("some totally novel failure")) == ""


# Gemini "thinking" token-budget self-heal (bounded, same-provider only)
def test_gemini_incomplete_response_retries_once_with_bigger_budget():
    """Regression test for Gemini 3.x's thinking tokens eating the whole
    max_output_tokens budget on short prompts. generate() should retry
    ONCE, same provider, with a larger budget — not loop, not cross
    providers."""
    import llm_providers

    class FakeInteraction:
        def __init__(self, output_text, status):
            self.output_text = output_text
            self.status = status

    calls = []

    class FakeInteractionsAPI:
        def create(self, model, input, system_instruction, generation_config):
            calls.append(generation_config["max_output_tokens"])
            if generation_config["max_output_tokens"] < 1000:
                return FakeInteraction(output_text=None, status="incomplete")
            return FakeInteraction(output_text="the real answer", status="completed")

    class FakeModels:
        def embed_content(self, model, contents):
            raise NotImplementedError

    class FakeClient:
        def __init__(self, api_key):
            self.interactions = FakeInteractionsAPI()
            self.models = FakeModels()

    class FakeGenaiModule:
        Client = FakeClient

    import sys
    sys.modules["google"] = type(sys)("google")
    sys.modules["google.genai"] = FakeGenaiModule
    sys.modules["google"].genai = FakeGenaiModule

    provider = llm_providers.GeminiProvider("fake-key", "gemini-3.6-flash")
    result = provider.generate("hello", max_tokens=50)

    assert result == "the real answer"
    assert calls == [50, 4096]  # exactly one retry, with a bigger budget

    del sys.modules["google.genai"]
    del sys.modules["google"]


def test_gemini_gives_up_after_one_retry_if_still_incomplete():
    """If the retry ALSO comes back incomplete, generate() must raise
    (not retry again) — the self-heal is bounded to a single attempt."""
    import llm_providers

    class FakeInteraction:
        output_text = None
        status = "incomplete"

    call_count = {"n": 0}

    class FakeInteractionsAPI:
        def create(self, **kwargs):
            call_count["n"] += 1
            return FakeInteraction()

    class FakeClient:
        def __init__(self, api_key):
            self.interactions = FakeInteractionsAPI()

    class FakeGenaiModule:
        Client = FakeClient

    import sys
    sys.modules["google"] = type(sys)("google")
    sys.modules["google.genai"] = FakeGenaiModule
    sys.modules["google"].genai = FakeGenaiModule

    provider = llm_providers.GeminiProvider("fake-key", "gemini-3.6-flash")
    with pytest.raises(ProviderError):
        provider.generate("hello", max_tokens=50)
    assert call_count["n"] == 2  # original attempt + exactly one retry

    del sys.modules["google.genai"]
    del sys.modules["google"]


# Document chunking + vector store (TF-IDF fallback path, no API key needed)
def test_chunk_text_respects_size_and_overlap():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=200, overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_text_handles_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_vector_store_retrieves_relevant_chunk():
    vs = VectorStore()
    vs.add_document("Agentic RAG lets an agent decide when to retrieve information.", source="a.txt")
    vs.add_document("The Eiffel Tower is located in Paris, France.", source="b.txt")

    hits = vs.search("What is Agentic RAG?", k=1)
    assert len(hits) == 1
    assert hits[0].source == "a.txt"

    hits2 = vs.search("Where is the Eiffel Tower?", k=1)
    assert hits2[0].source == "b.txt"


def test_vector_store_empty_returns_no_hits():
    vs = VectorStore()
    assert vs.search("anything") == []
    assert vs.is_empty() is True


# Full agent orchestration loop
class ScriptedLLM:
    """Deterministic mock LLM driven by (system-keyword -> answer) rules,
    so the agent loop can be tested end-to-end with no network access."""
    name = "scripted"

    def __init__(self, answer_map, fail_self_check_times=0):
        self.answer_map = answer_map
        self.fail_self_check_times = fail_self_check_times
        self._self_check_calls = 0

    def generate(self, prompt, system=None, max_tokens=1024):
        s = (system or "").lower()
        for keyword, answer in self.answer_map.items():
            if keyword in s:
                if keyword == "reviewing an ai-generated answer" and self._self_check_calls < self.fail_self_check_times:
                    self._self_check_calls += 1
                    return "no"
                return answer
        return "default response"


def test_agent_happy_path_with_documents():
    llm = ScriptedLLM({
        "rewrite": "What is Agentic RAG? (rewritten)",
        "requires looking up": "yes",
        "choose the best source": "vector_db",
        "answer the user": "Agentic RAG decides when and where to retrieve.",
        "reviewing an ai-generated answer": "yes",
    })
    router = LLMRouter(llm, None)
    vs = VectorStore()
    vs.add_document("Agentic RAG lets an agent decide when to retrieve and from where.", source="notes.txt")

    result = run_agentic_rag("What is Agentic RAG?", router, vs, gemini_provider=None, tavily_key=None)
    assert result.answer == "Agentic RAG decides when and where to retrieve."
    assert result.attempts == 1
    labels = [s.label for s in result.trace]
    assert "Retrieve" in labels
    assert "Self-check" in labels


def test_agent_skips_retrieval_when_not_needed():
    llm = ScriptedLLM({
        "rewrite": "2+2? (rewritten)",
        "requires looking up": "no",
        "answer the user": "4",
        "reviewing an ai-generated answer": "yes",
    })
    router = LLMRouter(llm, None)
    vs = VectorStore()  # empty, and also retrieval isn't needed

    result = run_agentic_rag("What is 2+2?", router, vs, gemini_provider=None, tavily_key=None)
    assert result.answer == "4"
    assert not any(s.label == "Retrieve" for s in result.trace)


def test_agent_retries_on_failed_self_check_then_succeeds():
    llm = ScriptedLLM({
        "rewrite": "query (rewritten)",
        "requires looking up": "no",
        "answer the user": "an answer",
        "reviewing an ai-generated answer": "yes",
    }, fail_self_check_times=1)
    router = LLMRouter(llm, None)
    vs = VectorStore()

    result = run_agentic_rag("some question", router, vs, gemini_provider=None, tavily_key=None)
    assert result.attempts == 2  # failed once, succeeded on retry


def test_agent_retry_loop_is_capped_and_terminates():
    """Regression test: a self-check that always says 'no' must not loop
    forever — it has to stop at MAX_RETRIES."""
    llm = ScriptedLLM({
        "rewrite": "query (rewritten)",
        "requires looking up": "no",
        "answer the user": "an answer",
        "reviewing an ai-generated answer": "no",  # always fails self-check
    })
    router = LLMRouter(llm, None)
    vs = VectorStore()

    result = run_agentic_rag("some question", router, vs, gemini_provider=None, tavily_key=None)
    assert result.attempts == MAX_RETRIES
    assert result.answer  # still returns the last attempt's answer, doesn't crash


def test_agent_falls_back_to_secondary_llm_mid_run():
    failing_primary = FakeProvider("p", always_fails(RuntimeError("primary down")))
    working_secondary = ScriptedLLM({
        "rewrite": "query (rewritten)",
        "requires looking up": "no",
        "answer the user": "fallback answer",
        "reviewing an ai-generated answer": "yes",
    })
    router = LLMRouter(failing_primary, working_secondary)
    vs = VectorStore()

    result = run_agentic_rag("some question", router, vs, gemini_provider=None, tavily_key=None)
    assert result.answer == "fallback answer"
    # every step in the trace should show the fallback provider was used
    provider_steps = [s for s in result.trace if s.provider]
    assert all(s.fell_back for s in provider_steps)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))