# ---------------------------------------------------
# Codoctopus — Plan validation
#
# Pure functions, no LLM involved — the same DAG rules Coworkify enforces in
# schemas/workflow.py's WorkflowCreate validator, ported here so a Plan means
# the same thing regardless of which Executor eventually runs it. If these
# rules and Coworkify's ever diverge, a plan built for one executor could
# silently misbehave on the other.
# ---------------------------------------------------

from __future__ import annotations

import re

from codoctopus.planning.models import Plan, PlanStep

# Same pattern Coworkify uses to find step references in a payload.
_STEP_REF = re.compile(r"\{\{\s*steps\.([a-zA-Z0-9_]+)\.result(?:\.[a-zA-Z0-9_]+)*\s*\}\}")


class PlanValidationError(ValueError):
    """A Plan's dependency graph is not executable."""


def validate_plan(plan: Plan) -> None:
    """Raise PlanValidationError if `plan` could not be scheduled by any executor."""
    steps = plan.steps
    keys = [s.key for s in steps]
    if len(keys) != len(set(keys)):
        raise PlanValidationError("step keys must be unique")

    key_set = set(keys)
    by_key = {s.key: s for s in steps}

    _validate_dependencies_exist(steps, key_set)
    _validate_for_each_rules(steps, key_set, by_key)
    _validate_no_cycles(steps, key_set)


def _validate_dependencies_exist(steps: list[PlanStep], key_set: set[str]) -> None:
    for step in steps:
        for dep in step.depends_on:
            if dep not in key_set:
                raise PlanValidationError(f"step '{step.key}' depends on unknown step '{dep}'")
            if dep == step.key:
                raise PlanValidationError(f"step '{step.key}' cannot depend on itself")


def _validate_for_each_rules(steps: list[PlanStep], key_set: set[str], by_key: dict[str, PlanStep]) -> None:
    for step in steps:
        if step.for_each is not None and step.reduce_of is not None:
            raise PlanValidationError(f"step '{step.key}' cannot set both for_each and reduce_of")

        if step.reduce_of is not None:
            _validate_reduce_step(step, key_set, by_key)
            continue

        if step.for_each is None:
            # A regular step may not depend on a for_each step directly — the
            # value produced is a list of per-item results, not a single
            # result. Collecting them is what reduce_of is for.
            for dep in step.depends_on:
                if by_key[dep].for_each is not None:
                    raise PlanValidationError(
                        f"step '{step.key}' depends on '{dep}', a for_each step; to fan in its "
                        f"results, use reduce_of='{dep}' instead"
                    )
            continue

        if step.for_each not in key_set:
            raise PlanValidationError(f"step '{step.key}' has for_each pointing to unknown step '{step.for_each}'")
        if step.for_each == step.key:
            raise PlanValidationError(f"step '{step.key}' cannot for_each itself")
        if by_key[step.for_each].for_each is not None:
            raise PlanValidationError(
                f"step '{step.key}' has for_each on '{step.for_each}', which is itself a for_each "
                "step (nested expansion is not supported)"
            )
        # A for_each step's dependencies must all belong to the same group —
        # it cannot reach outside to a step that won't be expanded alongside it.
        for dep in step.depends_on:
            if by_key[dep].for_each != step.for_each:
                raise PlanValidationError(
                    f"step '{step.key}' is in the for_each='{step.for_each}' group; it can only "
                    f"depend on other steps in that group, not '{dep}'"
                )


def _validate_reduce_step(step: PlanStep, key_set: set[str], by_key: dict[str, PlanStep]) -> None:
    if step.reduce_of not in key_set:
        raise PlanValidationError(f"step '{step.key}' has reduce_of pointing to unknown step '{step.reduce_of}'")
    if step.reduce_of == step.key:
        raise PlanValidationError(f"step '{step.key}' cannot reduce itself")
    if by_key[step.reduce_of].for_each is None:
        raise PlanValidationError(
            f"step '{step.key}' has reduce_of on '{step.reduce_of}', which is not a for_each step; "
            "reduce_of can only point to a step with for_each set"
        )
    # Coworkify fills a reduce step's dependencies itself once the group has
    # expanded, and renders only {{items}} into it — never step references.
    if step.depends_on:
        raise PlanValidationError(
            f"step '{step.key}' is a reduce step; its dependencies are implied by reduce_of and "
            "depends_on must be left empty"
        )
    if _STEP_REF.search(step.role) or _STEP_REF.search(step.instruction):
        raise PlanValidationError(
            f"step '{step.key}' is a reduce step; use {{{{items}}}} to get the upstream results, "
            "not {{steps.<key>.result}}"
        )


def _validate_no_cycles(steps: list[PlanStep], key_set: set[str]) -> None:
    # for_each and reduce_of each imply a wait on the step they name, so they
    # count as edges too — a cycle through one would deadlock every executor.
    deps = {
        s.key: [*s.depends_on, *(k for k in (s.for_each, s.reduce_of) if k is not None)] for s in steps
    }
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {k: WHITE for k in key_set}

    def dfs(node: str, path: list[str]) -> None:
        color[node] = GRAY
        for dep in deps[node]:
            if color[dep] == GRAY:
                raise PlanValidationError("plan has a circular dependency: " + " -> ".join([*path, dep]))
            if color[dep] == WHITE:
                dfs(dep, [*path, dep])
        color[node] = BLACK

    for key in key_set:
        if color[key] == WHITE:
            dfs(key, [key])
