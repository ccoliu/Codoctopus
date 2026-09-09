# ---------------------------------------------------
# Codoctopus — OpenAI adapter
# ---------------------------------------------------

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from codoctopus.llm.base import Provider, ProviderNotInstalled
from codoctopus.llm.schema import to_strict_schema
from codoctopus.llm.types import Completion, Message, StopReason, ToolCall, ToolSpec, Usage

_STOP_REASONS = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
}


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
        # base_url lets this adapter also talk to any OpenAI-compatible local
        # server (LM Studio, vLLM, etc); those don't check the key's value.
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
            client_kwargs.setdefault("api_key", "not-needed")
        self._client = AsyncOpenAI(**client_kwargs)

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
