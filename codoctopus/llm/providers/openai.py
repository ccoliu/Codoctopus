# ---------------------------------------------------
# Codoctopus — OpenAI adapter
# ---------------------------------------------------

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from codoctopus.llm.base import Provider, ProviderError, ProviderNotInstalled
from codoctopus.llm.schema import to_strict_schema
from codoctopus.llm.types import Completion, Message, StopReason, ToolCall, ToolSpec, Usage

_STOP_REASONS = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
}


def _client_kwargs(api_key: str | None, base_url: str | None) -> dict[str, Any]:
    # base_url lets this adapter also talk to any OpenAI-compatible local
    # server (LM Studio, vLLM, etc); those don't check the key's value.
    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
        kwargs.setdefault("api_key", "not-needed")
    return kwargs


async def list_models(*, api_key: str | None = None, base_url: str | None = None, **_: Any) -> list[str]:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ProviderNotInstalled("openai", "openai") from exc
    client = AsyncOpenAI(**_client_kwargs(api_key, base_url))
    page = await client.models.list()
    return sorted([m.id async for m in page])


class OpenAIProvider(Provider):
    name = "openai"
    default_model = "gpt-4.1"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **options: Any,
    ) -> None:
        super().__init__(model, **options)
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ProviderNotInstalled(self.name, "openai") from exc
        self._client = AsyncOpenAI(**_client_kwargs(api_key, base_url))
        self._api_key = api_key
        self._base_url = base_url
        # A custom base_url means this is some other OpenAI-*compatible*
        # gateway (LM Studio, vLLM, ...), not the real OpenAI API — its own
        # catalog decides what "the default model" is, not OpenAI's
        # "gpt-4.1" (which such a gateway is never going to have loaded).
        # Resolving that needs a network call, so it's deferred to the first
        # actual request rather than done in __init__.
        self._needs_default_model = base_url is not None and model is None

    @property
    def supports_structured_output(self) -> bool:
        return True

    async def _resolve_default_model(self) -> None:
        models = await list_models(api_key=self._api_key, base_url=self._base_url)
        if not models:
            raise ProviderError(f"No models available at {self._base_url}")
        self.model = models[0]
        self._needs_default_model = False

    async def _complete(
        self,
        messages: list[Message],
        *,
        system: str | None,
        tools: list[ToolSpec] | None,
        output_schema: type[BaseModel] | None,
        max_tokens: int,
        **kwargs: Any,
    ) -> Completion:
        if self._needs_default_model:
            await self._resolve_default_model()

        payload: list[dict[str, Any]] = []
        if system:
            payload.append({"role": "system", "content": system})
        for message in messages:
            payload.extend(_to_openai(message))

        params: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "messages": payload,
        }
        if tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": {**t.input_schema, "additionalProperties": False},
                        "strict": True,
                    },
                }
                for t in tools
            ]
        if output_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__,
                    "schema": to_strict_schema(output_schema),
                    "strict": True,
                },
            }
        params.update(kwargs)

        response = await self._client.chat.completions.create(**params)
        choice = response.choices[0]
        text = choice.message.content or ""

        tool_calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=json.loads(c.function.arguments or "{}"))
            for c in (choice.message.tool_calls or [])
        ]

        parsed = output_schema.model_validate(json.loads(text)) if output_schema and text else None
        usage = response.usage
        return Completion(
            text=text,
            model=response.model,
            stop_reason=_STOP_REASONS.get(choice.finish_reason, StopReason.OTHER),
            tool_calls=tool_calls,
            parsed=parsed,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            raw=response,
        )


class GatewayProvider(OpenAIProvider):
    """
    An OpenAI-*compatible* self-hosted gateway (LM Studio, vLLM, a local
    proxy, ...) — registered separately from "openai" so configuring one
    isn't mislabeled as configuring the real OpenAI API (a model loaded in
    LM Studio is not "an OpenAI model"), and so a missing base_url is a
    clear error here instead of silently falling through to api.openai.com.
    """

    name = "gateway"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **options: Any,
    ) -> None:
        if not base_url:
            raise ProviderError(
                "The 'gateway' provider needs a base_url pointing at your OpenAI-compatible "
                "server (e.g. LM Studio, vLLM)."
            )
        super().__init__(model, api_key=api_key, base_url=base_url, **options)


def _to_openai(message: Message) -> list[dict[str, Any]]:
    """OpenAI wants one message per tool result, under a dedicated role."""
    if message.tool_results:
        return [
            {"role": "tool", "tool_call_id": r.call_id, "content": r.content}
            for r in message.tool_results
        ]

    if message.tool_calls:
        return [
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in message.tool_calls
                ],
            }
        ]

    return [{"role": message.role, "content": message.content or ""}]
