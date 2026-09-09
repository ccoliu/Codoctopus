# ---------------------------------------------------
# Codoctopus — Provider interface
#
# Every adapter implements `_complete`. `complete` wraps it so that providers
# without native structured output still return a validated object, via a
# prompt-and-retry fallback instead of v1's string slicing.
# ---------------------------------------------------

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError

from codoctopus.llm.schema import to_strict_schema
from codoctopus.llm.types import Completion, Message, ToolSpec

DEFAULT_MAX_TOKENS = 16000


class ProviderError(RuntimeError):
    """A provider call failed in a way the caller cannot retry around."""


class ProviderNotInstalled(ProviderError):
    """The vendor SDK for this provider is not installed."""

    def __init__(self, provider: str, package: str) -> None:
        super().__init__(
            f"Provider '{provider}' needs the '{package}' package. "
            f"Install it with: pip install codoctopus[{provider}]"
        )


class StructuredOutputError(ProviderError):
    """The model could not produce output matching the requested schema."""


class Provider(ABC):
    """
    A model behind a uniform interface.

    Adapters are constructed with a model id and keyword options; they must not
    require any Codoctopus type beyond those in `codoctopus.llm.types`.
    """

    #: Identifier used in model strings, e.g. the "anthropic" in "anthropic:claude-opus-5".
    name: str = ""

    #: Model used when the caller names a provider but no model.
    default_model: str = ""

    def __init__(self, model: str | None = None, **options: Any) -> None:
        self.model = model or self.default_model
        self.options = options

    # -- capabilities -------------------------------------------------

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def supports_structured_output(self) -> bool:
        """True when the provider constrains decoding to a schema server-side."""
        return False

    # -- the one method adapters implement ----------------------------

    @abstractmethod
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
        """Perform one call. `output_schema` is passed only if natively supported."""

    # -- public entry point -------------------------------------------

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        output_schema: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        schema_retries: int = 2,
        **kwargs: Any,
    ) -> Completion:
        if tools and not self.supports_tools:
            raise ProviderError(f"Provider '{self.name}' does not support tool use")

        # A single choke point for every adapter's vendor-SDK exceptions
        # (missing auth, bad request, rate limit, ...): none of them wrap
        # their own errors into ProviderError, so without this a caller like
        # the CLI — which only catches ProviderError — sees a raw traceback
        # from whichever SDK happens to be installed instead of one
        # consistent, catchable error type.
        try:
            if output_schema is None or self.supports_structured_output:
                return await self._complete(
                    messages,
                    system=system,
                    tools=tools,
                    output_schema=output_schema,
                    max_tokens=max_tokens,
                    **kwargs,
                )

            return await self._complete_via_prompt(
                messages,
                system=system,
                tools=tools,
                output_schema=output_schema,
                max_tokens=max_tokens,
                schema_retries=schema_retries,
                **kwargs,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Provider '{self.name}' call failed: {exc}") from exc

    # -- fallback for providers without native structured output ------

    async def _complete_via_prompt(
        self,
        messages: list[Message],
        *,
        system: str | None,
        tools: list[ToolSpec] | None,
        output_schema: type[BaseModel],
        max_tokens: int,
        schema_retries: int,
        **kwargs: Any,
    ) -> Completion:
        schema = to_strict_schema(output_schema)
        instruction = (
            "Reply with a single JSON object and nothing else — no prose, no code fence.\n"
            "It must validate against this JSON Schema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
        system = f"{system}\n\n{instruction}" if system else instruction

        attempt = list(messages)
        last_error: Exception | None = None

        for _ in range(schema_retries + 1):
            completion = await self._complete(
                attempt,
                system=system,
                tools=tools,
                output_schema=None,
                max_tokens=max_tokens,
                **kwargs,
            )
            # A tool call means the model is still working; schema comes later.
            if completion.wants_tools:
                return completion
            try:
                completion.parsed = output_schema.model_validate(_extract_json(completion.text))
                return completion
            except (ValidationError, ValueError) as exc:
                last_error = exc
                attempt = [
                    *attempt,
                    completion.to_message(),
                    Message.user(f"That did not match the schema: {exc}\nReturn corrected JSON only."),
                ]

        raise StructuredOutputError(
            f"Provider '{self.name}' did not produce schema-valid output after "
            f"{schema_retries + 1} attempts: {last_error}"
        )


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating a code fence or surrounding prose."""
    if not text or not text.strip():
        raise ValueError("model returned no text")

    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Last resort: the outermost {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"no JSON object found in model output: {text[:200]!r}")
