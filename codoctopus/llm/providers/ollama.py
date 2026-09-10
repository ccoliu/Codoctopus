# ---------------------------------------------------
# Codoctopus — Ollama adapter (local models)
#
# Talks to the Ollama HTTP API directly, so running fully offline needs no
# vendor SDK beyond httpx.
# ---------------------------------------------------

from __future__ import annotations

import json
import uuid
from typing import Any

from pydantic import BaseModel

from codoctopus.llm.base import Provider, ProviderError, ProviderNotInstalled
from codoctopus.llm.schema import to_strict_schema
from codoctopus.llm.types import Completion, Message, StopReason, ToolCall, ToolSpec, Usage

DEFAULT_HOST = "http://localhost:11434"


async def list_models(*, base_url: str | None = None, **_: Any) -> list[str]:
    """Whatever's actually pulled locally — Ollama needs no key, just a reachable host."""
    import httpx

    host = (base_url or DEFAULT_HOST).rstrip("/")
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{host}/api/tags")
        if response.status_code >= 400:
            raise ProviderError(f"Ollama returned {response.status_code}: {response.text[:300]}")
        data = response.json()
    return sorted(m["name"] for m in data.get("models", []))


class OllamaProvider(Provider):
    name = "ollama"
    default_model = "llama3.1"

    def __init__(
        self,
        model: str | None = None,
        *,
        host: str = DEFAULT_HOST,
        timeout: float = 300.0,
        **options: Any,
    ) -> None:
        super().__init__(model, **options)
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ProviderNotInstalled(self.name, "httpx") from exc
        self._httpx = httpx
        self.host = host.rstrip("/")
        self.timeout = timeout

    @property
    def supports_structured_output(self) -> bool:
        # Ollama constrains decoding to a JSON Schema passed as `format`.
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
            payload.extend(_to_ollama(message))

        body: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
        if output_schema is not None:
            body["format"] = to_strict_schema(output_schema)
        body.update(kwargs)

        async with self._httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.host}/api/chat", json=body)
            if response.status_code >= 400:
                raise ProviderError(f"Ollama returned {response.status_code}: {response.text[:300]}")
            data = response.json()

        reply = data.get("message", {})
        text = reply.get("content") or ""

        # Ollama does not issue call ids; mint one so results can be matched back.
        tool_calls = [
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                name=call["function"]["name"],
                arguments=_as_dict(call["function"].get("arguments")),
            )
            for call in reply.get("tool_calls", []) or []
        ]

        parsed = output_schema.model_validate(json.loads(text)) if output_schema and text else None
        return Completion(
            text=text,
            model=data.get("model", self.model),
            stop_reason=StopReason.TOOL_USE
            if tool_calls
            else (StopReason.MAX_TOKENS if data.get("done_reason") == "length" else StopReason.END_TURN),
            tool_calls=tool_calls,
            parsed=parsed,
            usage=Usage(
                input_tokens=data.get("prompt_eval_count", 0) or 0,
                output_tokens=data.get("eval_count", 0) or 0,
            ),
            raw=data,
        )


def _as_dict(arguments: Any) -> dict[str, Any]:
    """Arguments come back as an object from most models, a JSON string from some."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        return json.loads(arguments)
    return {}


def _to_ollama(message: Message) -> list[dict[str, Any]]:
    if message.tool_results:
        return [{"role": "tool", "content": r.content} for r in message.tool_results]

    entry: dict[str, Any] = {"role": message.role, "content": message.content or ""}
    if message.tool_calls:
        entry["tool_calls"] = [
            {"function": {"name": c.name, "arguments": c.arguments}} for c in message.tool_calls
        ]
    return [entry]
