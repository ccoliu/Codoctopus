# ---------------------------------------------------
# Codoctopus — Plan validation tests
#
# Pure DAG-rule tests, no LLM involved. These mirror the cases Coworkify's
# WorkflowCreate validator already covers (tests/... in the coworkify repo),
# on purpose — the two are meant to agree on what counts as an executable plan.
# ---------------------------------------------------

from __future__ import annotations

import pytest

from codoctopus.planning.models import Plan, PlanStep
from codoctopus.planning.validate import PlanValidationError, validate_plan


def step(key: str, **kwargs) -> PlanStep:
    kwargs.setdefault("instruction", f"do {key}")
    return PlanStep(key=key, name=key, role="be helpful", **kwargs)


def plan(*steps: PlanStep) -> Plan:
    return Plan(goal="test", steps=list(steps))


# --- the happy paths ----------------------------------------------------


def test_a_linear_plan_is_valid():
    validate_plan(plan(step("a"), step("b", depends_on=["a"]), step("c", depends_on=["b"])))


def test_independent_steps_need_no_dependency():
    validate_plan(plan(step("a"), step("b")))


def test_a_for_each_group_may_depend_on_its_own_siblings():
    validate_plan(
        plan(
            step("search", instruction="find jobs"),
            step("tailor", for_each="search", depends_on=[]),
            step("apply", for_each="search", depends_on=["tailor"]),
        )
    )


# --- structural errors ---------------------------------------------------


def test_duplicate_keys_are_rejected():
    with pytest.raises(PlanValidationError, match="unique"):
        validate_plan(plan(step("a"), step("a")))


def test_a_dependency_on_an_unknown_step_is_rejected():
    with pytest.raises(PlanValidationError, match="unknown step 'nope'"):
        validate_plan(plan(step("a", depends_on=["nope"])))


def test_a_step_cannot_depend_on_itself():
    with pytest.raises(PlanValidationError, match="cannot depend on itself"):
        validate_plan(plan(step("a", depends_on=["a"])))


@pytest.mark.parametrize(
    "steps",
    [
        [step("a", depends_on=["b"]), step("b", depends_on=["a"])],
        [step("a", depends_on=["c"]), step("b", depends_on=["a"]), step("c", depends_on=["b"])],
    ],
    ids=["two-cycle", "three-cycle"],
)
def test_circular_dependencies_are_rejected(steps):
    with pytest.raises(PlanValidationError, match="circular dependency"):
        validate_plan(plan(*steps))


# --- for_each rules --------------------------------------------------------


def test_for_each_target_must_exist():
    with pytest.raises(PlanValidationError, match="unknown step 'nope'"):
        validate_plan(plan(step("a", for_each="nope")))


def test_a_step_cannot_for_each_itself():
    with pytest.raises(PlanValidationError, match="cannot for_each itself"):
        validate_plan(plan(step("a", for_each="a")))


def test_nested_for_each_is_rejected():
    with pytest.raises(PlanValidationError, match="nested expansion is not supported"):
        validate_plan(
            plan(
                step("search", instruction="find jobs"),
                step("outer", for_each="search"),
                step("inner", for_each="outer"),
            )
        )


def test_a_regular_step_cannot_depend_on_a_for_each_step_directly():
    with pytest.raises(PlanValidationError, match="use reduce_of='tailor' instead"):
        validate_plan(
            plan(
                step("search", instruction="find jobs"),
                step("tailor", for_each="search"),
                step("summarize", depends_on=["tailor"]),
            )
        )


def test_a_for_each_step_cannot_depend_on_a_step_outside_its_group():
    with pytest.raises(PlanValidationError, match="can only depend on other steps in that group"):
        validate_plan(
            plan(
                step("search", instruction="find jobs"),
                step("unrelated"),
                step("tailor", for_each="search", depends_on=["unrelated"]),
            )
        )


# --- reduce_of rules -------------------------------------------------------


def test_a_reduce_step_may_fan_in_a_for_each_group():
    validate_plan(
        plan(
            step("search", instruction="find jobs"),
            step("tailor", for_each="search"),
            step("summarize", reduce_of="tailor", instruction="summarise {{items}}"),
            step("report", depends_on=["summarize"]),
        )
    )


def test_reduce_target_must_exist():
    with pytest.raises(PlanValidationError, match="reduce_of pointing to unknown step 'nope'"):
        validate_plan(plan(step("a", reduce_of="nope")))


def test_a_step_cannot_reduce_itself():
    with pytest.raises(PlanValidationError, match="cannot reduce itself"):
        validate_plan(plan(step("a", reduce_of="a")))


def test_reduce_target_must_be_a_for_each_step():
    with pytest.raises(PlanValidationError, match="not a for_each step"):
        validate_plan(plan(step("search"), step("summarize", reduce_of="search")))


def test_a_step_cannot_both_for_each_and_reduce_of():
    with pytest.raises(PlanValidationError, match="both for_each and reduce_of"):
        validate_plan(
            plan(
                step("search"),
                step("tailor", for_each="search"),
                step("both", for_each="search", reduce_of="tailor"),
            )
        )


def test_a_reduce_step_cannot_declare_its_own_dependencies():
    with pytest.raises(PlanValidationError, match="depends_on must be left empty"):
        validate_plan(
            plan(
                step("search"),
                step("tailor", for_each="search"),
                step("summarize", reduce_of="tailor", depends_on=["search"]),
            )
        )


def test_a_reduce_step_must_use_items_not_step_references():
    with pytest.raises(PlanValidationError, match=r"use \{\{items\}\}"):
        validate_plan(
            plan(
                step("search"),
                step("tailor", for_each="search"),
                step("summarize", reduce_of="tailor", instruction="combine {{steps.search.result}}"),
            )
        )


def test_a_cycle_through_for_each_and_reduce_of_is_rejected():
    # search waits on summarize, which waits on the tailor group, which waits
    # on search: no step's depends_on alone shows the loop.
    with pytest.raises(PlanValidationError, match="circular dependency"):
        validate_plan(
            plan(
                step("search", depends_on=["summarize"]),
                step("tailor", for_each="search"),
                step("summarize", reduce_of="tailor"),
            )
        )


# --- parity with Coworkify ---------------------------------------------------
#
# One shared table of plans, checked against this validator always and against
# Coworkify's own StepsGraph whenever a coworkify checkout is importable (set
# COWORKIFY_PATH, or keep it as a sibling directory of this repo). A new rule
# on either side that the other lacks shows up as a failing case here.
#
# Deliberately left out, because the two are known to differ:
# - a cycle only visible through for_each/reduce_of edges: rejected here,
#   accepted by Coworkify (which then never finishes the workflow).
# - {{steps.<key>.result}} inside a for_each step: rejected by Coworkify,
#   rendered normally by LocalExecutor.

PARITY_CASES = [
    ("linear", [step("a"), step("b", depends_on=["a"])], True),
    ("for_each", [step("s"), step("t", for_each="s")], True),
    ("for_each-siblings", [step("s"), step("t", for_each="s"), step("u", for_each="s", depends_on=["t"])], True),
    ("reduce", [step("s"), step("t", for_each="s"), step("r", reduce_of="t")], True),
    ("after-reduce", [step("s"), step("t", for_each="s"), step("r", reduce_of="t"), step("z", depends_on=["r"])], True),
    ("duplicate-keys", [step("a"), step("a")], False),
    ("unknown-dep", [step("a", depends_on=["nope"])], False),
    ("self-dep", [step("a", depends_on=["a"])], False),
    ("cycle", [step("a", depends_on=["b"]), step("b", depends_on=["a"])], False),
    ("for_each-unknown", [step("a", for_each="nope")], False),
    ("for_each-self", [step("a", for_each="a")], False),
    ("for_each-nested", [step("s"), step("t", for_each="s"), step("u", for_each="t")], False),
    ("for_each-outside-dep", [step("s"), step("x"), step("t", for_each="s", depends_on=["x"])], False),
    ("plain-depends-on-for_each", [step("s"), step("t", for_each="s"), step("z", depends_on=["t"])], False),
    ("reduce-unknown", [step("r", reduce_of="nope")], False),
    ("reduce-self", [step("r", reduce_of="r")], False),
    ("reduce-of-plain-step", [step("s"), step("r", reduce_of="s")], False),
    ("reduce-and-for_each", [step("s"), step("t", for_each="s"), step("r", for_each="s", reduce_of="t")], False),
    ("reduce-with-depends_on", [step("s"), step("t", for_each="s"), step("r", reduce_of="t", depends_on=["s"])], False),
    (
        "reduce-with-step-ref",
        [step("s"), step("t", for_each="s"), step("r", reduce_of="t", instruction="{{steps.s.result}}")],
        False,
    ),
]


@pytest.mark.parametrize("steps,valid", [c[1:] for c in PARITY_CASES], ids=[c[0] for c in PARITY_CASES])
def test_parity_cases_against_this_validator(steps, valid):
    if valid:
        validate_plan(plan(*steps))
    else:
        with pytest.raises(PlanValidationError):
            validate_plan(plan(*steps))


def _coworkify_steps_graph():
    import os
    import sys
    from pathlib import Path

    root = os.environ.get("COWORKIFY_PATH") or str(Path(__file__).resolve().parents[2] / "coworkify")
    if not (Path(root) / "app" / "schemas" / "workflow.py").is_file():
        pytest.skip("no coworkify checkout found (set COWORKIFY_PATH)")
    sys.path.insert(0, root)
    try:
        from app.schemas.workflow import StepsGraph
    finally:
        sys.path.remove(root)
    return StepsGraph


@pytest.mark.parametrize("steps,valid", [c[1:] for c in PARITY_CASES], ids=[c[0] for c in PARITY_CASES])
def test_parity_cases_against_coworkify(steps, valid):
    from pydantic import ValidationError

    from codoctopus.runtime.coworkify import CoworkifyExecutor

    steps_graph = _coworkify_steps_graph()
    # Translated exactly as CoworkifyExecutor submits it, so the check covers
    # the passthrough too, not just the rules.
    workflow_steps = CoworkifyExecutor("http://coworkify", token="t")._to_workflow_steps(plan(*steps), "/tmp/ws")
    if valid:
        steps_graph(steps=workflow_steps)
    else:
        with pytest.raises(ValidationError):
            steps_graph(steps=workflow_steps)
