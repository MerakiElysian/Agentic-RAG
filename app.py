"""
app.py
======
Streamlit front end for the Agentic RAG assistant.

Run with:
    streamlit run app.py

Configure API keys either as environment variables (GEMINI_API_KEY,
XAI_API_KEY, TAVILY_API_KEY) or by typing them into the sidebar —
sidebar values win if both are present. Keys are kept only in
st.session_state for this browser session; nothing is written to disk.
"""

import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from llm_providers import GeminiProvider, LLMRouter
from retrieval import VectorStore, extract_text
from agent import run_agentic_rag


st.set_page_config(page_title="Agentic RAG", page_icon=None, layout="wide")

for key in ("GEMINI_API_KEY", "XAI_API_KEY", "TAVILY_API_KEY"):
    if key in st.secrets:
        os.environ.setdefault(key, st.secrets[key])
# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "trace", "attempts"}
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

# Sidebar — configuration
with st.sidebar:
    st.header("API keys")
    gemini_key = st.text_input("Gemini API key", value=os.environ.get("GEMINI_API_KEY", ""), type="password").strip()
    tavily_key = st.text_input("Tavily API key", value=os.environ.get("TAVILY_API_KEY", ""), type="password").strip()

    st.divider()
    st.header("Model")
    gemini_model = st.text_input("Gemini model", value="gemini-3-flash-preview")
    gemini_thinking = st.selectbox(
        "Gemini thinking level", ["low", "medium", "high"], index=0,
        help="Gemini 3.x models reason internally before answering, and that reasoning is billed against "
             "the token budget. 'low' keeps this app's short decision prompts from running out of budget "
             "before producing visible text.",
    )

    if st.button("Test Gemini", disabled=not gemini_key, use_container_width=True):
        try:
            reply = GeminiProvider(gemini_key, gemini_model, gemini_thinking).test_connection()
            st.success(f"Gemini OK: '{reply}'")
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.header("Knowledge base")
    uploaded = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded and st.button("Add to knowledge base"):
        gemini_for_embed = GeminiProvider(gemini_key, gemini_model, gemini_thinking) if gemini_key else None
        added = 0
        for f in uploaded:
            if f.name in st.session_state.ingested_files:
                continue
            try:
                text = extract_text(f.read(), f.name)
                n = st.session_state.vector_store.add_document(text, source=f.name, gemini_provider=gemini_for_embed)
                st.session_state.ingested_files.append(f.name)
                added += n
            except Exception as e:
                st.error(f"Couldn't process {f.name}: {e}")
        if added:
            st.success(f"Added {added} chunk(s). Embedding mode: {st.session_state.vector_store.mode}")

    if st.session_state.ingested_files:
        st.caption("In knowledge base: " + ", ".join(st.session_state.ingested_files))
        if st.button("Clear knowledge base"):
            st.session_state.vector_store = VectorStore()
            st.session_state.ingested_files = []
            st.rerun()

    st.divider()
    st.header("Web search")
    enable_web = st.checkbox("Allow web search (Tavily)", value=bool(tavily_key))
    deep_scrape = st.checkbox("Fetch full page content (scrape), not just snippets", value=False)

# Build the router for this run
def build_router():
    gemini = GeminiProvider(gemini_key, gemini_model, gemini_thinking) if gemini_key else None
    return LLMRouter(gemini), gemini

# Main chat UI
st.title("Agentic RAG assistant")
st.caption("The agent decides whether to retrieve, which source to use, and checks its own answer before replying.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander(f"Agent trace ({msg['attempts']} attempt(s))"):
                for step in msg["trace"]:
                    if step.detail:
                        provider_note = f"  ·  via **{step.provider}**" if step.provider else ""
                        st.markdown(f"**{step.label}**{provider_note}\n\n{step.detail}")
                    else:
                        st.markdown(f"**{step.label}**")

query = st.chat_input("Ask a question…")

if query:
    if not gemini_key:
        st.error("Add your Gemini API key in the sidebar or in .env before asking a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        router, gemini_provider = build_router()
        active_tavily_key = tavily_key if enable_web else None

        with st.chat_message("assistant"):
            with st.spinner("Working through the agent loop…"):
                try:
                    result = run_agentic_rag(
                        query=query,
                        router=router,
                        vector_store=st.session_state.vector_store,
                        gemini_provider=gemini_provider,
                        tavily_key=active_tavily_key,
                        deep_scrape=deep_scrape,
                    )
                    st.markdown(result.answer)
                    with st.expander(f"Agent trace ({result.attempts} attempt(s))"):
                        for step in result.trace:
                            if step.detail:
                                provider_note = f"  ·  via **{step.provider}**" if step.provider else ""
                                st.markdown(f"**{step.label}**{provider_note}\n\n{step.detail}")
                            else:
                                st.markdown(f"**{step.label}**")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.answer,
                        "trace": result.trace,
                        "attempts": result.attempts,
                    })
                except Exception as e:
                    st.error(f"The agent couldn't complete this request: {e}")