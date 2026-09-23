# ---------------------------------------------------
# Codoctopus — Agent runtime
#
# This is what makes v2's agents different from v1's: v1's `employee_work()`
# just handed a block of text to the next role in the chain. An Agent here
# can call tools and see their real results before it answers.
#
# The tool side is deliberately a narrow Protocol, not an import of
# `codoctopus.tools`. That keeps this module buildable and testable against a
# fake executor independently of whatever the tools package ends up looking
# like internally — the two only need to agree on ToolSpec / ToolCall /
# ToolResult, which already exist in codoctopus.llm.types.
# ---------------------------------------------------

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from codoctopus.llm.base import DEFAULT_MAX_TOKENS, Provider
from codoctopus.llm.types import (
    Message,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)


@runtime_checkable
class ToolExecutor(Protocol):
    """
    What an Agent needs from a tool collection.

    Implemented by `codoctopus.tools.ToolRegistry`. An Agent never inspects a
    Tool directly — it only ever sees the specs it can advertise to the model
    and the results that come back from running one.
    """

    def specs(self) -> list[ToolSpec]: ...

    async def execute(self, call: ToolCall) -> ToolResult: ...


class ToolTurnLimitExceeded(RuntimeError):
    """The agent kept calling tools past `max_tool_turns` without finishing."""

    def __init__(self, max_tool_turns: int, transcript: list[Message]) -> None:
        super().__init__(
            f"Agent did not produce a final answer within {max_tool_turns} tool-use turn(s)"
        )
        self.max_tool_turns = max_tool_turns
        self.transcript = transcript


@dataclass
class AgentResult:
    """What an Agent.run() call produced, plus enough to audit how it got there."""

    text: str
    parsed: BaseModel | None
    stop_reason: StopReason
    turns: int
    usage: Usage
    transcript: list[Message] = field(default_factory=list)


class Agent:
    """
    A role bound to a model, an optional toolset, and an optional output shape.

    One call to `run()` drives the full tool-use loop: ask the model, and if
    it asks for tools, execute them and go around again, until it answers
    without asking for a tool or `max_tool_turns` is exhausted.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        system: str | None = None,
        tools: ToolExecutor | None = None,
        output_schema: type[BaseModel] | None = None,
        max_tool_turns: int = 25,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.provider = provider
        self.system = system
        self.tools = tools
        self.output_schema = output_schema
        self.max_tool_turns = max_tool_turns
        self.max_tokens = max_tokens

    async def run(self, task: str) -> AgentResult:
        return await self.continue_(Message.user(task))

    async def continue_(self, *initial: Message) -> AgentResult:
        """Run the loop starting from an existing transcript, not just one user turn."""
        messages: list[Message] = list(initial)
        tool_specs = self.tools.specs() if self.tools is not None else None
        usage = Usage()

        for turn in range(1, self.max_tool_turns + 1):
            completion = await self.provider.complete(
                messages,
                system=self.system,
                tools=tool_specs,
                output_schema=self.output_schema,
                max_tokens=self.max_tokens,
            )
            usage = usage + completion.usage
            messages.append(completion.to_message())

            if not completion.wants_tools:
                return AgentResult(
                    text=completion.text,
                    parsed=completion.parsed,
                    stop_reason=completion.stop_reason,
                    turns=turn,
                    usage=usage,
                    transcript=messages,
                )

            results = await self._execute_all(completion.tool_calls)
            messages.append(Message.results(results))

        raise ToolTurnLimitExceeded(self.max_tool_turns, messages)

    # -- tool execution -------------------------------------------------

    async def _execute_all(self, calls: list[ToolCall]) -> list[ToolResult]:
        # Parallel tool_use blocks must come back as one batch of tool_results
        # in a single turn — never split across messages, or providers learn
        # to stop making parallel calls.
        return list(await asyncio.gather(*(self._execute_one(c) for c in calls)))

    async def _execute_one(self, call: ToolCall) -> ToolResult:
        if self.tools is None:
            return ToolResult(
                call_id=call.id,
                content=f"Error: no tools are available, but the model called '{call.name}'",
                is_error=True,
            )
        try:
            return await self.tools.execute(call)
        except Exception as exc:  # noqa: BLE001 - a broken tool must not crash the loop
            return ToolResult(call_id=call.id, content=f"Error: {exc}", is_error=True)
