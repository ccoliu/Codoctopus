# ---------------------------------------------------
# Codoctopus — Anthropic adapter (default provider)
#
# Uses adaptive thinking and server-enforced structured output, so schema
# conformance is guaranteed by the API rather than by parsing prose.
# ---------------------------------------------------

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from codoctopus.llm.base import Provider, ProviderError, ProviderNotInstalled
from codoctopus.llm.schema import to_strict_schema
from codoctopus.llm.types import Completion, Message, StopReason, ToolCall, ToolSpec, Usage

#: Above this, the SDK wants streaming so the request cannot hit an HTTP timeout.
_STREAM_ABOVE = 16000

_STOP_REASONS = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
    "stop_sequence": StopReason.END_TURN,
}


async def list_models(*, api_key: str | None = None, **_: Any) -> list[str]:
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ProviderNotInstalled("anthropic", "anthropic") from exc
    client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()
    page = await client.models.list()
    return sorted([m.id async for m in page])


class AnthropicProvider(Provider):
    name = "anthropic"
    default_model = "claude-opus-5"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        effort: str | None = "high",
        thinking: bool = True,
        **options: Any,
    ) -> None:
        super().__init__(model, **options)
        self.effort = effort
        self.thinking = thinking
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ProviderNotInstalled(self.name, "anthropic") from exc
        # With no api_key the SDK resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
        # or an `ant auth login` profile — do not force a key here.
        self._client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()

    @property
    def supports_structured_output(self) -> bool:
        return True

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
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [_to_anthropic(m) for m in messages],
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [_to_anthropic_tool(t) for t in tools]
        if self.thinking:
            params["thinking"] = {"type": "adaptive"}

        output_config: dict[str, Any] = {}
        if self.effort:
            output_config["effort"] = self.effort
        if output_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": to_strict_schema(output_schema),
            }
        if output_config:
            params["output_config"] = output_config

        params.update(kwargs)

        if max_tokens > _STREAM_ABOVE:
            async with self._client.messages.stream(**params) as stream:
                response = await stream.get_final_message()
        else:
            response = await self._client.messages.create(**params)

        return _to_completion(response, output_schema)


def _to_anthropic(message: Message) -> dict[str, Any]:
    """Render one neutral Message as an Anthropic message dict."""
    if message.tool_results:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": r.call_id,
                    "content": r.content,
                    **({"is_error": True} if r.is_error else {}),
                }
                for r in message.tool_results
            ],
        }

    if message.tool_calls:
        blocks: list[dict[str, Any]] = []
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        blocks.extend(
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
            for c in message.tool_calls
        )
        return {"role": "assistant", "content": blocks}

    return {"role": message.role, "content": message.content or ""}


def _to_anthropic_tool(tool: ToolSpec) -> dict[str, Any]:
    schema = dict(tool.input_schema)
    schema.setdefault("additionalProperties", False)
    return {
        "name": tool.name,
        "description": tool.description,
        "strict": True,
        "input_schema": schema,
    }


def _to_completion(response: Any, output_schema: type[BaseModel] | None) -> Completion:
    if response.stop_reason == "refusal":
        detail = getattr(response, "stop_details", None)
        raise ProviderError(
            f"Anthropic declined the request"
            f"{f' ({detail.category})' if detail and getattr(detail, 'category', None) else ''}"
        )

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            # Tool input arrives as a parsed object; never string-match the raw JSON.
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))

    text = "".join(text_parts)
    parsed = None
    if output_schema is not None and text:
        parsed = output_schema.model_validate(json.loads(text))

    usage = response.usage
    return Completion(
        text=text,
        model=response.model,
        stop_reason=_STOP_REASONS.get(response.stop_reason, StopReason.OTHER),
        tool_calls=tool_calls,
        parsed=parsed,
        usage=Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        ),
        raw=response,
    )
