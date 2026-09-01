"""
retrieval.py
============
Everything the agent can retrieve FROM:

1. A local vector store built from uploaded PDF / DOCX / TXT files.
   Embeddings use Gemini's embedding model (or local TF-IDF if offline).

2. Tavily web search — also used for "scraping" a specific URL via
   Tavily's include_raw_content option, so no separate scraper
   dependency is needed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import io
import re

import numpy as np


# Document parsing
def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == "docx":
        import docx
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in document.paragraphs)
    if ext == "txt":
        return file_bytes.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: .{ext}")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


# Vector store
@dataclass
class Chunk:
    text: str
    source: str


@dataclass
class VectorStore:
    chunks: list[Chunk] = field(default_factory=list)
    vectors: np.ndarray | None = None
    mode: str = "none"          # "gemini" or "tfidf" or "none"
    _tfidf = None                # sklearn vectorizer, only used in tfidf mode

    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def add_document(self, text: str, source: str, gemini_provider=None) -> int:
        new_chunks = [Chunk(text=c, source=source) for c in chunk_text(text)]
        if not new_chunks:
            return 0
        self.chunks.extend(new_chunks)
        self._reembed(gemini_provider)
        return len(new_chunks)

    def _reembed(self, gemini_provider) -> None:
        texts = [c.text for c in self.chunks]
        if gemini_provider is not None:
            try:
                embeddings = gemini_provider.embed(texts)
                self.vectors = np.array(embeddings, dtype=np.float32)
                self.mode = "gemini"
                return
            except Exception:
                pass  # fall through to TF-IDF if Gemini embeddings fail
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(stop_words="english")
        self.vectors = self._tfidf.fit_transform(texts).toarray().astype(np.float32)
        self.mode = "tfidf"

    def search(self, query: str, k: int = 4, gemini_provider=None) -> list[Chunk]:
        if self.is_empty() or self.vectors is None:
            return []
        if self.mode == "gemini" and gemini_provider is not None:
            try:
                q_vec = np.array(gemini_provider.embed([query])[0], dtype=np.float32)
            except Exception:
                return self._tfidf_query(query, k)
        elif self.mode == "tfidf" and self._tfidf is not None:
            q_vec = self._tfidf.transform([query]).toarray().astype(np.float32)[0]
        else:
            return []

        sims = _cosine_sim(self.vectors, q_vec)
        top_idx = np.argsort(-sims)[:k]
        return [self.chunks[i] for i in top_idx if sims[i] > 0]

    def _tfidf_query(self, query: str, k: int) -> list[Chunk]:
        if self._tfidf is None:
            return []
        q_vec = self._tfidf.transform([query]).toarray().astype(np.float32)[0]
        sims = _cosine_sim(self.vectors, q_vec)
        top_idx = np.argsort(-sims)[:k]
        return [self.chunks[i] for i in top_idx if sims[i] > 0]


def _cosine_sim(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    norm_m = np.linalg.norm(matrix, axis=1) + 1e-8
    norm_v = np.linalg.norm(vector) + 1e-8
    return (matrix @ vector) / (norm_m * norm_v)


# Web search / scrape via Tavily
def tavily_search(api_key: str, query: str, max_results: int = 4, deep_scrape: bool = False) -> list[dict]:
    """
    Returns a list of {title, url, content} dicts. When deep_scrape=True,
    Tavily also fetches and returns the raw page content for each result
    (this is the "scraping" step in the architecture).
    """
    from tavily import TavilyClient
    client = TavilyClient(api_key=api_key)
    resp = client.search(
        query=query,
        max_results=max_results,
        include_raw_content=deep_scrape,
    )
    results = []
    for r in resp.get("results", []):
        content = r.get("raw_content") or r.get("content") or ""
        results.append({"title": r.get("title", ""), "url": r.get("url", ""), "content": content[:2000]})
    return results
