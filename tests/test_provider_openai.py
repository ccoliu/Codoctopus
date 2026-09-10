# ---------------------------------------------------
# Codoctopus — OpenAIProvider construction
#
# Only covers client wiring (no network call): this is what lets the same
# adapter also reach an OpenAI-compatible local server (LM Studio, vLLM, ...)
# by pointing base_url at it instead of api.openai.com.
# ---------------------------------------------------

from __future__ import annotations

from types import SimpleNamespace

import pytest

from codoctopus.llm.base import ProviderError
from codoctopus.llm.providers.openai import GatewayProvider, OpenAIProvider, list_models


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


# --- default model resolution against a custom gateway ----------------------
#
# A base_url with no explicit model means some other OpenAI-*compatible*
# server (LM Studio, vLLM, ...) — real OpenAI's "gpt-4.1" default is never
# going to be loaded there, so the gateway's own first listed model is used
# instead, resolved lazily on the first actual call.


def _fake_completion(model: str = "gpt-4.1"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi", tool_calls=None), finish_reason="stop")],
        model=model,
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


async def test_base_url_without_a_model_uses_one_of_the_gateways_own_models(monkeypatch):
    from codoctopus.llm.types import Message

    # list_models() alphabetizes its results (right for a GUI dropdown to
    # browse) — the point under test is only that a *real* model came from
    # the gateway's own catalog, not OpenAI's "gpt-4.1" default.
    async def fake_models_list(self, **kw):
        return _FakeAsyncPage([SimpleNamespace(id="qwen/qwen3.5-9b"), SimpleNamespace(id="gemma-4-e4b")])

    captured: dict = {}

    async def fake_create(self, **kwargs):
        captured["model"] = kwargs["model"]
        return _fake_completion(kwargs["model"])

    monkeypatch.setattr("openai.resources.models.AsyncModels.list", fake_models_list)
    monkeypatch.setattr(
        "openai.resources.chat.completions.completions.AsyncCompletions.create", fake_create
    )

    provider = OpenAIProvider(base_url="http://localhost:1234/v1")
    await provider.complete([Message.user("hello")])

    assert captured["model"] in {"qwen/qwen3.5-9b", "gemma-4-e4b"}
    assert provider.model == captured["model"]


async def test_base_url_with_an_explicit_model_never_queries_the_models_list(monkeypatch):
    from codoctopus.llm.types import Message

    async def fail_if_called(self, **kw):
        raise AssertionError("should not list models when a model was given explicitly")

    async def fake_create(self, **kwargs):
        return _fake_completion(kwargs["model"])

    monkeypatch.setattr("openai.resources.models.AsyncModels.list", fail_if_called)
    monkeypatch.setattr(
        "openai.resources.chat.completions.completions.AsyncCompletions.create", fake_create
    )

    provider = OpenAIProvider(model="local-model", base_url="http://localhost:1234/v1")
    await provider.complete([Message.user("hello")])

    assert provider.model == "local-model"


async def test_no_base_url_never_queries_the_models_list_either(monkeypatch):
    from codoctopus.llm.types import Message

    async def fail_if_called(self, **kw):
        raise AssertionError("should not list models for the real OpenAI endpoint")

    async def fake_create(self, **kwargs):
        return _fake_completion(kwargs["model"])

    monkeypatch.setattr("openai.resources.models.AsyncModels.list", fail_if_called)
    monkeypatch.setattr(
        "openai.resources.chat.completions.completions.AsyncCompletions.create", fake_create
    )

    provider = OpenAIProvider(api_key="sk-test")
    await provider.complete([Message.user("hello")])

    assert provider.model == "gpt-4.1"


# --- GatewayProvider: a distinct identity from "openai" for local gateways --


def test_gateway_provider_requires_a_base_url():
    with pytest.raises(ProviderError, match="base_url"):
        GatewayProvider(model="qwen/qwen3.5-9b")


def test_gateway_provider_points_the_client_at_the_given_base_url():
    provider = GatewayProvider(model="qwen/qwen3.5-9b", base_url="http://localhost:1234/v1")
    assert str(provider._client.base_url).rstrip("/") == "http://localhost:1234/v1"
    assert provider.name == "gateway"


async def test_gateway_provider_also_resolves_a_default_model_from_its_own_catalog(monkeypatch):
    from codoctopus.llm.types import Message

    async def fake_models_list(self, **kw):
        return _FakeAsyncPage([SimpleNamespace(id="qwen/qwen3.5-9b")])

    captured: dict = {}

    async def fake_create(self, **kwargs):
        captured["model"] = kwargs["model"]
        return _fake_completion(kwargs["model"])

    monkeypatch.setattr("openai.resources.models.AsyncModels.list", fake_models_list)
    monkeypatch.setattr(
        "openai.resources.chat.completions.completions.AsyncCompletions.create", fake_create
    )

    provider = GatewayProvider(base_url="http://localhost:1234/v1")
    await provider.complete([Message.user("hello")])

    assert captured["model"] == "qwen/qwen3.5-9b"
