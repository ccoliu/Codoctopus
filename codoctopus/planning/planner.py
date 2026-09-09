# ---------------------------------------------------
# Codoctopus — Planner
#
# Turns a goal into a validated Plan. Structured output (codoctopus.llm)
# guarantees the *shape* is right — field types, required keys — but DAG
# semantics (no cycles, dependencies exist, for_each rules) can't be expressed
# in a JSON Schema, so they're checked afterward and, on failure, fed back to
# the model for a corrected plan. Same retry-with-the-error shape as the LLM
# layer's own structured-output fallback (codoctopus.llm.base), just one
# level up, because a schema-valid Plan can still be a semantically broken one.
#
# A Domain, if given, only shapes two things here: the prompt (which roles and
# tools the planner is told about) and the returned Plan's `role` text (a
# recognized role name is expanded to that domain's full prompt). Nothing
# about domains reaches codoctopus.runtime — see codoctopus/domains/base.py.
# ---------------------------------------------------

from __future__ import annotations

from codoctopus.domains.base import Domain
from codoctopus.llm.base import Provider
from codoctopus.llm.types import Message
from codoctopus.planning.models import Plan
from codoctopus.planning.validate import PlanValidationError, validate_plan

_SYSTEM_PROMPT = """You are a planner. Break the user's goal into a directed \
acyclic graph of steps for other agents to execute.

Each step needs:
- a stable "key" used to reference it from other steps
- a "role": the system prompt the step's agent should run under
- a concrete "instruction" describing exactly what to do
- "depends_on": keys of steps that must finish first (omit for steps that can \
start immediately or run in parallel)
- "tools": names of tools the step's agent may use, if any

Use "for_each" only when a step's result is a list and later steps must run \
once per item in it; reference the current item in a dependent step's \
instruction with "{{item}}" or "{{item.<field>}}". Reference another \
already-completed step's output with "{{steps.<key>.result}}".

Keep the plan as small as it can be while still fully accomplishing the goal."""


class PlanningError(RuntimeError):
    """The planner could not produce a valid Plan within its retry budget."""


def _build_system_prompt(domain: Domain | None, available_tools: list[str] | None) -> str:
    lines = [_SYSTEM_PROMPT]

    if domain is not None:
        lines += ["", f"You are planning within the '{domain.name}' domain."]
        if domain.planner_hint:
            lines.append(domain.planner_hint)
        if domain.roles:
            lines.append(
                "Use one of these exact names in a step's \"role\" field when it fits — the full "
                "prompt behind each name will be filled in for you afterward. Write your own free-form "
                "role text only when none of these fit:"
            )
            lines.extend(f"- {name}" for name in domain.roles)

    # The tool universe a step may draw from comes from whatever the caller
    # actually wired up (its ToolRegistry) — not the domain, which only hints
    # at what's typical here. Naming any other tool would make the step fail
    # at execution time with an "unknown tool" error, so always spell out the
    # real, callable set rather than only listing it when a domain is set.
    domain_tools = domain.default_tools if domain else []
    tools = list(dict.fromkeys([*domain_tools, *(available_tools or [])]))
    if tools:
        lines.append(
            f"Tools you may name in a step's \"tools\" field (use these exact names, "
            f"omit the field for a step that needs none): {', '.join(tools)}"
        )

    return "\n".join(lines)


def _apply_domain(plan: Plan, domain: Domain) -> Plan:
    """Expand any step whose role names a domain role into that role's full prompt text."""
    resolved_steps = [
        step.model_copy(update={"role": domain.roles[step.role]}) if step.role in domain.roles else step
        for step in plan.steps
    ]
    return plan.model_copy(update={"steps": resolved_steps, "domain": domain.name})


async def make_plan(
    provider: Provider,
    goal: str,
    *,
    domain: Domain | None = None,
    available_tools: list[str] | None = None,
    max_retries: int = 2,
) -> Plan:
    """Ask `provider` to decompose `goal` into a validated Plan.

    `available_tools` should list every tool name the executor that will run
    this plan can actually resolve (e.g. a ToolRegistry's tool names) — the
    planner has no other way to know what a step may legally ask for.
    """
    system = _build_system_prompt(domain, available_tools)
    domain_line = f"\nDomain: {domain.name}" if domain else ""
    messages = [Message.user(f"Goal: {goal}{domain_line}")]
    last_error: Exception | None = None

    for _ in range(max_retries + 1):
        completion = await provider.complete(messages, system=system, output_schema=Plan)
        plan = completion.parsed
        assert plan is not None, "output_schema was requested; the LLM layer guarantees a parsed result"

        try:
            validate_plan(plan)
        except PlanValidationError as exc:
            last_error = exc
            messages = [
                *messages,
                completion.to_message(),
                Message.user(f"That plan is not valid: {exc}\nReturn a corrected plan."),
            ]
            continue

        return _apply_domain(plan, domain) if domain is not None else plan.model_copy(update={"domain": "general"})

    raise PlanningError(f"could not produce a valid plan after {max_retries + 1} attempt(s): {last_error}")
