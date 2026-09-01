"""
llm_providers.py
=================
Wrapper around Gemini (Google) that exposes a clean interface:
generate(prompt, system, max_tokens) -> str
"""

from __future__ import annotations
from dataclasses import dataclass


class ProviderError(Exception):
    """Raised when a single provider call fails. Carries a human-readable
    hint alongside the raw error so the UI can show something actionable
    instead of a bare stack trace."""
    def __init__(self, provider: str, original: Exception):
        self.provider = provider
        self.original = original
        self.hint = _diagnose(original)
        msg = f"[{provider}] {type(original).__name__}: {original}"
        if self.hint:
            msg += f"\n  -> {self.hint}"
        super().__init__(msg)


def _diagnose(exc: Exception) -> str:
    """Best-effort, string-matching diagnosis of common API failure modes,
    so errors point at a fix instead of just a stack trace."""
    text = f"{type(exc).__name__} {exc}".lower()
    if any(k in text for k in ["401", "unauthorized", "incorrect api key", "invalid api key", "invalid-argument"]) \
            and "api key" in text:
        return "The API key looks wrong or unset — check it was pasted in full, with no extra spaces."
    if any(k in text for k in ["403", "permission-denied", "permissiondenied"]):
        if any(k in text for k in ["quota", "credit", "spending limit", "billing"]):
            return "The account is out of quota/credits or has hit its spending limit — check billing on the provider's console."
        return "The key doesn't have permission for this model/endpoint — check the key's scopes on the provider's console."
    if any(k in text for k in ["404", "not found", "no longer available"]):
        return "The model name isn't recognized by the API anymore — check the provider's current model list and update the model field in the sidebar."
    if any(k in text for k in ["429", "rate limit", "resource_exhausted", "resourceexhausted"]):
        return "Rate limited — the provider is asking you to slow down or upgrade your tier."
    if any(k in text for k in ["timeout", "connection", "network", "unreachable"]):
        return "Couldn't reach the provider's API — check your network connection."
    return ""


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-3-flash-preview", thinking_level: str = "low"):
        self.api_key = api_key
        self.model = model
        self.thinking_level = thinking_level

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 1024) -> str:
        try:
            return self._generate_once(prompt, system, max_tokens)
        except ProviderError as first_err:
            if "incomplete" in str(first_err).lower() and max_tokens < 4096:
                try:
                    return self._generate_once(prompt, system, max_tokens=4096)
                except ProviderError:
                    raise first_err
            raise

    def _generate_once(self, prompt: str, system: str | None, max_tokens: int) -> str:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            thinking_budget = 0 if self.thinking_level == "low" else (1024 if self.thinking_level == "medium" else 4096)
            config = types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            )
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            text = getattr(response, "text", None)
            if text:
                return text.strip()
            raise RuntimeError("Gemini returned an empty response.")
        except Exception as e:
            raise ProviderError(self.name, e) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
            )
            return [list(e.values) for e in result.embeddings]
        except Exception as e:
            raise ProviderError(self.name, e) from e

    def test_connection(self) -> str:
        """Minimal live call used by the sidebar 'Test connection' button."""
        text = self.generate("Reply with exactly: ok", max_tokens=100)
        return text.strip()



@dataclass
class FallbackResult:
    text: str
    provider_used: str
    fell_back: bool
    primary_error: str | None = None


class LLMRouter:
    """
    Wraps a primary + secondary provider. call() tries primary once,
    falls back to secondary once on any failure, then gives up — no loop.
    """

    def __init__(self, primary, secondary=None):
        self.primary = primary
        self.secondary = secondary

    def call(self, prompt: str, system: str | None = None, max_tokens: int = 1024) -> FallbackResult:
        try:
            text = self.primary.generate(prompt, system=system, max_tokens=max_tokens)
            return FallbackResult(text=text, provider_used=self.primary.name, fell_back=False)
        except ProviderError as primary_err:
            if self.secondary is None:
                raise RuntimeError(
                    f"{self.primary.name} failed and no fallback provider is configured: {primary_err}"
                ) from primary_err
            try:
                text = self.secondary.generate(prompt, system=system, max_tokens=max_tokens)
                return FallbackResult(
                    text=text,
                    provider_used=self.secondary.name,
                    fell_back=True,
                    primary_error=str(primary_err),
                )
            except ProviderError as secondary_err:
                raise RuntimeError(
                    f"Both providers failed.\n"
                    f"Primary ({self.primary.name}): {primary_err}\n"
                    f"Secondary ({self.secondary.name}): {secondary_err}"
                ) from secondary_err