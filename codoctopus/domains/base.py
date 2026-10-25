# ---------------------------------------------------
# Codoctopus — Domain interface
#
# A Domain is a pluggable set of conventions for one kind of task: what roles
# exist, which tools are available by default, how to nudge the planner, and
# how to check the final result actually worked. This is what replaces v1's
# Config.yaml, where CODE_MASTER / ANALYST / PRESENTER and the coding-only
# worksheet format were the only thing Codoctopus could ever produce.
#
# A Domain only ever touches the Planner and, after the fact, verify() — the
# Executor (codoctopus.runtime) never imports this module. PlanStep.role stays
# plain prompt text all the way through execution; a Domain's role *names*
# are resolved to that text inside make_plan(), before the Plan is returned.
# That's what keeps LocalExecutor (and the future CoworkifyExecutor) usable
# for any domain without either of them knowing domains exist.
#
# verify() takes a plain `dict[str, str]` of step results rather than a
# runtime.PlanResult on purpose: domains sit in core alongside planning and
# tools, and core must never import a runtime backend (codoctopus.runtime) —
# doing so here previously created an import cycle (planning -> domains ->
# runtime -> planning, since LocalExecutor itself needs Plan/validate_plan).
# Whoever calls an Executor and then wants to verify its PlanResult passes
# `result.step_results` through; that glue is the caller's job, not this
# module's.
# ---------------------------------------------------

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from codoctopus.tools.base import ToolContext


@dataclass
class VerifyResult:
    #: False only when the check ran and failed — a skipped check counts as
    #: passed, so it never fails a run on its own.
    passed: bool
    detail: str = ""
    #: The check couldn't run at all (no tests to run, files on a remote
    #: worker, ...) — distinct from running and passing.
    skipped: bool = False

    @property
    def status(self) -> str:
        if self.skipped:
            return "skipped"
        return "passed" if self.passed else "failed"

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "detail": self.detail}


async def verify_plan_result(domain: Domain, step_results: dict[str, str], workspace: Path | None) -> VerifyResult:
    """
    Run `domain`'s check after a successful execution. `workspace` is None
    when the files aren't on this machine (a Coworkify run executes on a
    remote worker), in which case there's nothing local to check.
    """
    if workspace is None:
        return VerifyResult(
            passed=True,
            skipped=True,
            detail="Skipped: this run's files live on the Coworkify worker, not on this machine.",
        )
    try:
        return await domain.verify(step_results, ToolContext(workspace=workspace))
    except Exception as exc:
        return VerifyResult(passed=False, detail=f"Verification itself raised an error: {exc}")


class Domain(ABC):
    """
    Subclasses set the four class attributes below; `verify()` is optional to
    override — the default is a no-op that always passes, for domains (like
    prose writing) that have no automatic way to check the result.
    """

    #: Short identifier, e.g. "coding". Recorded on the Plan that used this domain.
    name: ClassVar[str]

    #: Role name -> system prompt text. The planner is told these names and
    #: picks among them; make_plan() expands the name back to this text.
    roles: ClassVar[dict[str, str]] = {}

    #: Tool names to mention to the planner as available in this domain.
    default_tools: ClassVar[list[str]] = []

    #: Domain-specific guidance injected into the planner's system prompt —
    #: e.g. "always end with a step that runs the test suite".
    planner_hint: ClassVar[str] = ""

    async def verify(self, step_results: dict[str, str], ctx: ToolContext) -> VerifyResult:
        """Check whether a completed plan's step outputs actually satisfy this domain."""
        return VerifyResult(passed=True, skipped=True, detail="No verification defined for this domain.")
