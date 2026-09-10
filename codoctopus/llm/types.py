# ---------------------------------------------------
# Codoctopus — Provider-neutral LLM types
#
# These types are the contract between the core and every provider adapter.
# Nothing here may import a vendor SDK: adapters translate to and from these
# shapes so the rest of Codoctopus never sees a vendor-specific object.
# ---------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ToolSpec:
    """A tool offered to the model. `input_schema` is a JSON Schema object."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A model's request to invoke a tool."""

    id: str
    name: str
    arguments: dict[str, Any]
    #: Provider-specific data a provider needs back verbatim on a later turn
    #: to replay this same call (e.g. Gemini's thought_signature) — opaque to
    #: everything else, and empty for providers that don't need it.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """The outcome of running a tool, fed back to the model."""

    call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """
    One turn of a conversation.

    An assistant turn may carry `tool_calls`; the turn answering it carries the
    matching `tool_results`. Adapters map this onto whatever the vendor expects
    (Anthropic puts results in a user turn, OpenAI uses a dedicated tool role).
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str | None = None, tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def results(cls, results: list[ToolResult]) -> Message:
        return cls(role="user", tool_results=results)


class StopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    OTHER = "other"


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass
class Completion:
    """
    One model response, normalised.

    `parsed` is populated only when the call requested structured output; it is
    an instance of the schema that was asked for, already validated.
    """

    text: str
    model: str
    stop_reason: StopReason = StopReason.END_TURN
    tool_calls: list[ToolCall] = field(default_factory=list)
    parsed: BaseModel | None = None
    usage: Usage = field(default_factory=Usage)
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    def to_message(self) -> Message:
        """Fold this completion back into the transcript as an assistant turn."""
        return Message(role="assistant", content=self.text or None, tool_calls=list(self.tool_calls))
