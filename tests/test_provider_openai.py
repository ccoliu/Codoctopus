# ---------------------------------------------------
# Codoctopus — OpenAIProvider construction
#
# Only covers client wiring (no network call): this is what lets the same
# adapter also reach an OpenAI-compatible local server (LM Studio, vLLM, ...)
# by pointing base_url at it instead of api.openai.com.
# ---------------------------------------------------

from __future__ import annotations

from types import SimpleNamespace

from codoctopus.llm.providers.openai import OpenAIProvider, list_models


def test_default_construction_uses_the_real_openai_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAIProvider(model="gpt-4.1")
    assert str(provider._client.base_url).rstrip("/") == "https://api.openai.com/v1"


def test_base_url_points_the_client_at_a_local_server():
    provider = OpenAIProvider(model="local-model", base_url="http://localhost:1234/v1")
    assert str(provider._client.base_url).rstrip("/") == "http://localhost:1234/v1"


def test_base_url_without_an_api_key_still_constructs(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider(model="local-model", base_url="http://localhost:1234/v1")
    assert provider._client.api_key == "not-needed"


def test_explicit_api_key_is_preserved_alongside_base_url():
    provider = OpenAIProvider(
        model="local-model", api_key="sk-real", base_url="http://localhost:1234/v1"
    )
    assert provider._client.api_key == "sk-real"


class _FakeAsyncPage:
    """Minimal stand-in for openai's AsyncPage: only needs to support `async for`."""

    def __init__(self, items):
        self._items = items

    async def __aiter__(self):
        for item in self._items:
            yield item


async def test_list_models_returns_ids_sorted(monkeypatch):
    async def fake_list(self, **kw):
        return _FakeAsyncPage([SimpleNamespace(id="gpt-4o"), SimpleNamespace(id="gpt-4.1")])

    monkeypatch.setattr("openai.resources.models.AsyncModels.list", fake_list)

    assert await list_models(api_key="sk-test") == ["gpt-4.1", "gpt-4o"]
