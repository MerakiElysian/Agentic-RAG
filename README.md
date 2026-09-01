# Agentic RAG assistant (Streamlit)

A working agentic RAG system: the agent decides *whether* to retrieve,
*where* to retrieve from (your documents vs. live web search), and
*checks its own answer* before showing it to you — with a visible
trace of every decision, powered by Google Gemini, and an automated
test suite covering the app's logic.

## What it does

- **Agentic Workflow.** Gemini orchestrates the full pipeline: rewriting
  the query, deciding if retrieval is required, choosing the source,
  generating the response, and performing a self-check verification.
- **Real retrieval.** Upload PDF / DOCX / TXT files; the app chunks
  and embeds them using Gemini embeddings (`gemini-embedding-001`), with
  a local TF-IDF fallback if offline.
- **Web search + scraping via Tavily**, matching the "Vector
  Databases / Tools and API / Google SERP / Tavily Search" source
  step in the architecture diagram. Turn on "fetch full page content"
  to have Tavily return scraped page text instead of just snippets.
- **Full agent trace.** Every chat response has an expandable trace
  showing the rewritten query, the retrieval decision, which source
  was chosen, what was retrieved, the LLM used, and the self-check verdict.
- **Test-connection diagnostics.** Sidebar button makes one live,
  minimal call to Gemini so a bad key or network issue surfaces
  immediately with a plain-English diagnosis — not after a failed
  chat turn.

## Setup

```bash
pip install -r requirements.txt
```

Get API keys:
- Gemini: https://aistudio.google.com/apikey
- Tavily (web search): https://app.tavily.com

Add them to `.env` (or copy `.env.example` to `.env`):

```bash
GEMINI_API_KEY=your_gemini_key_here
TAVILY_API_KEY=your_tavily_key_here
```

...or paste them into the sidebar once the app is running.

## Run it

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).
Click **Test Gemini** in the sidebar first — a green confirmation
means Gemini is correctly wired up before you spend a turn on a real question.

## Run the test suite

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

All 22 automated tests run without any API key and without touching the network.

## Using the app

1. In the sidebar or `.env`, add your Gemini API key and use **Test Gemini** to confirm.
2. Optionally upload documents and click **Add to knowledge base**.
3. Optionally add a Tavily key and enable web search.
4. Ask a question in the chat box. Expand **Agent trace** under any
   response to see exactly how the agent got there.

## File structure

```
app.py                Streamlit UI — chat, sidebar, session state, connection test
agent.py              The orchestration loop (mirrors the diagram's boxes)
llm_providers.py      Gemini wrapper & error diagnosis
retrieval.py          Document parsing/chunking/embeddings, Tavily search
test_app_logic.py     Automated tests (no API key / network required)
requirements.txt
requirements-dev.txt
```

