# ---------------------------------------------------
# Codoctopus — Anthropic list_models
# ---------------------------------------------------

from __future__ import annotations

from types import SimpleNamespace

from codoctopus.llm.providers.anthropic import list_models


class _FakeAsyncPage:
    """Minimal stand-in for anthropic's AsyncPage: only needs to support `async for`."""

    def __init__(self, items):
        self._items = items

    async def __aiter__(self):
        for item in self._items:
            yield item


async def test_list_models_returns_ids_sorted(monkeypatch):
    async def fake_list(self, **kw):
        return _FakeAsyncPage([SimpleNamespace(id="claude-sonnet-5"), SimpleNamespace(id="claude-opus-5")])

    monkeypatch.setattr("anthropic.resources.models.AsyncModels.list", fake_list)

    assert await list_models(api_key="sk-ant-test") == ["claude-opus-5", "claude-sonnet-5"]
