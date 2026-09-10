# ---------------------------------------------------
# Codoctopus — Ollama list_models
# ---------------------------------------------------

from __future__ import annotations

import httpx
import pytest

from codoctopus.llm.base import ProviderError
from codoctopus.llm.providers.ollama import list_models


def _mock_client_factory(handler):
    """
    Real AsyncClient backed by MockTransport, no real network call. Captures
    the real class before patching — patching httpx.AsyncClient replaces the
    shared module attribute, so a factory that itself called httpx.AsyncClient(...)
    would recurse into its own patched replacement.
    """
    real_async_client = httpx.AsyncClient

    def factory(**kw):
        return real_async_client(transport=httpx.MockTransport(handler), **kw)

    return factory


async def test_list_models_returns_locally_pulled_model_names_sorted(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b"}, {"name": "llama3.1:latest"}]})

    monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))

    assert await list_models() == ["llama3.1:latest", "qwen2.5:7b"]


async def test_list_models_uses_base_url_as_the_host_override(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"models": []})

    monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))

    await list_models(base_url="http://custom-host:1234")

    assert seen["url"] == "http://custom-host:1234/api/tags"


async def test_list_models_raises_providererror_on_an_http_failure(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))

    with pytest.raises(ProviderError, match="Ollama returned 500"):
        await list_models()
