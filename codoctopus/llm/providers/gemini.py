# ---------------------------------------------------
# Codoctopus — Google Gemini adapter
#
# Carries over the provider the `refactor` branch had hard-wired, now behind
# the same interface as every other one.
# ---------------------------------------------------

from __future__ import annotations

import base64
import json
import uuid
from typing import Any

from pydantic import BaseModel

from codoctopus.llm.base import Provider, ProviderNotInstalled
from codoctopus.llm.types import Completion, Message, StopReason, ToolCall, ToolSpec, Usage

def _sanitize_schema_for_gemini(node: Any) -> Any:
    """
    Gemini's FunctionDeclaration.parameters uses its own OpenAPI-subset
    schema dialect, narrower than standard JSON Schema — sending a key it
    doesn't recognize rejects the whole request outright (unlike OpenAI/
    Anthropic/Ollama, which use to_strict_schema's stricter-but-compatible
    shape fine). Tool schemas come through as plain dicts already (unlike a
    Plan's own output_schema, there's no Pydantic class left to hand the SDK
    instead), so this rewrites the ones known to trip Gemini as they're hit:
    - additionalProperties: no equivalent at all — dropped.
    - exclusiveMinimum/Maximum: no equivalent — approximated as an inclusive
      minimum/maximum instead of dropped outright, since for the numeric
      timeout-style fields this shows up on, off-by-the-boundary is harmless
      and closer to the original intent than losing the bound entirely.
    """
    if isinstance(node, dict):
        out = {k: _sanitize_schema_for_gemini(v) for k, v in node.items() if k != "additionalProperties"}
        if "exclusiveMinimum" in out:
            out.setdefault("minimum", out.pop("exclusiveMinimum"))
        if "exclusiveMaximum" in out:
            out.setdefault("maximum", out.pop("exclusiveMaximum"))
        return out
    if isinstance(node, list):
        return [_sanitize_schema_for_gemini(item) for item in node]
    return node


_STOP_REASONS = {
    "STOP": StopReason.END_TURN,
    "MAX_TOKENS": StopReason.MAX_TOKENS,
    "SAFETY": StopReason.REFUSAL,
    "PROHIBITED_CONTENT": StopReason.REFUSAL,
}


async def list_models(*, api_key: str | None = None, **_: Any) -> list[str]:
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ProviderNotInstalled("gemini", "google-genai") from exc
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    page = await client.aio.models.list()
    return sorted(
        [
            m.name.removeprefix("models/")
            async for m in page
            if m.name and "generateContent" in (m.supported_actions or [])
        ]
    )


class GeminiProvider(Provider):
    name = "gemini"
    default_model = "gemini-2.0-flash"

    def __init__(self, model: str | None = None, *, api_key: str | None = None, **options: Any) -> None:
        super().__init__(model, **options)
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ProviderNotInstalled(self.name, "google-genai") from exc
        self._types = genai_types
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

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
        types = self._types

        config: dict[str, Any] = {"max_output_tokens": max_tokens}
        if system:
            config["system_instruction"] = system
        if tools:
            config["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t.name,
                            description=t.description,
                            parameters=_sanitize_schema_for_gemini(t.input_schema),
                        )
                        for t in tools
                    ]
                )
            ]
        if output_schema is not None:
            config["response_mime_type"] = "application/json"
            # Gemini's response_schema is its own OpenAPI-subset dialect, not
            # the strict-JSON-Schema shape to_strict_schema produces for
            # OpenAI/Ollama (it rejects additionalProperties outright) — the
            # SDK converts a Pydantic model class to that dialect itself, so
            # hand it the class directly instead.
            config["response_schema"] = output_schema
        config.update(kwargs)

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=[_to_gemini(m, types) for m in messages],
            config=types.GenerateContentConfig(**config),
        )

        text = response.text or ""
        candidate = (response.candidates or [None])[0]
        # Walked by hand (not response.function_calls) so each call can carry
        # its own Part's thought_signature — required to replay a function
        # call in a later turn (see _to_gemini) once thinking is involved;
        # sending the conversation back without it is a 400, not just a
        # quality hit, despite the SDK docs' wording.
        response_parts = (candidate and candidate.content and candidate.content.parts) or []
        tool_calls = []
        for part in response_parts:
            if part.function_call is None:
                continue
            extra: dict[str, Any] = {}
            if part.thought_signature:
                extra["gemini_thought_signature"] = base64.b64encode(part.thought_signature).decode("ascii")
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    name=part.function_call.name,
                    arguments=dict(part.function_call.args or {}),
                    extra=extra,
                )
            )

        parsed = output_schema.model_validate(json.loads(text)) if output_schema and text else None
        usage = response.usage_metadata
        return Completion(
            text=text,
            model=self.model,
            stop_reason=_STOP_REASONS.get(
                str(getattr(candidate, "finish_reason", "")).rsplit(".", 1)[-1], StopReason.OTHER
            )
            if not tool_calls
            else StopReason.TOOL_USE,
            tool_calls=tool_calls,
            parsed=parsed,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                cache_read_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
            ),
            raw=response,
        )


def _to_gemini(message: Message, types: Any) -> Any:
    """Gemini calls the assistant role "model" and wraps results in function responses."""
    if message.tool_results:
        return types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name=r.call_id, response={"error" if r.is_error else "output": r.content}
                )
                for r in message.tool_results
            ],
        )

    parts: list[Any] = []
    if message.content:
        parts.append(types.Part.from_text(text=message.content))
    for c in message.tool_calls:
        signature = c.extra.get("gemini_thought_signature")
        if signature:
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(name=c.name, args=c.arguments),
                    thought_signature=base64.b64decode(signature),
                )
            )
        else:
            parts.append(types.Part.from_function_call(name=c.name, args=c.arguments))
    return types.Content(role="model" if message.role == "assistant" else "user", parts=parts)
