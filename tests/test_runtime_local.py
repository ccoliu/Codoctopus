from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from codoctopus.llm import Completion, Provider
from codoctopus.planning.models import Plan, PlanStep
from codoctopus.runtime import LocalExecutor


class ScriptedProvider(Provider):
    name = "scripted"
    default_model = "scripted-1"

    def __init__(self, responses: dict[str, str] | None = None, **options: Any) -> None:
        super().__init__("scripted-1", **options)
        self.responses = responses or {}
        self.calls: list[dict[str, Any]] = []

    async def _complete(self, messages, *, system, tools, output_schema, max_tokens, **kwargs) -> Completion:
        user_msg = messages[-1].content if messages and messages[-1].content else ""
        self.calls.append({"messages": list(messages), "system": system, "task": user_msg})
        
        reply = self.responses.get(user_msg, f"response for: {user_msg}")
        return Completion(text=reply, model=self.model)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.mark.anyio
async def test_linear_plan_execution(workspace: Path):
    provider = ScriptedProvider(
        responses={
            "step 1 instruction": "result 1",
            "step 2 using result 1": "result 2",
        }
    )
    plan = Plan(
        goal="run linear steps",
        steps=[
            PlanStep(key="s1", name="Step 1", role="role 1", instruction="step 1 instruction"),
            PlanStep(
                key="s2",
                name="Step 2",
                role="role 2",
                instruction="step 2 using {{steps.s1.result}}",
                depends_on=["s1"],
            ),
        ],
    )
    executor = LocalExecutor(provider, workspace=workspace)
    result = await executor.run(plan)

    assert result.status == "success"
    assert result.step_results["s1"] == "result 1"
    assert result.step_results["s2"] == "result 2"


@pytest.mark.anyio
async def test_parallel_independent_steps(workspace: Path):
    provider = ScriptedProvider(
        responses={
            "task A": "out A",
            "task B": "out B",
        }
    )
    plan = Plan(
        goal="run parallel steps",
        steps=[
            PlanStep(key="a", name="A", role="rA", instruction="task A"),
            PlanStep(key="b", name="B", role="rB", instruction="task B"),
        ],
    )
    executor = LocalExecutor(provider, workspace=workspace)
    result = await executor.run(plan)

    assert result.status == "success"
    assert result.step_results["a"] == "out A"
    assert result.step_results["b"] == "out B"


@pytest.mark.anyio
async def test_for_each_dynamic_expansion(workspace: Path):
    provider = ScriptedProvider(
        responses={
            "list items": '["apple", "banana"]',
            "process item: apple": "apple processed",
            "process item: banana": "banana processed",
        }
    )
    plan = Plan(
        goal="run for_each steps",
        steps=[
            PlanStep(key="producer", name="Producer", role="rp", instruction="list items"),
            PlanStep(
                key="consumer",
                name="Consumer",
                role="rc",
                instruction="process item: {{item}}",
                depends_on=[],
                for_each="producer",
            ),
        ],
    )
    executor = LocalExecutor(provider, workspace=workspace)
    result = await executor.run(plan)

    assert result.status == "success"
    assert result.step_results["producer"] == '["apple", "banana"]'
    assert result.step_results["consumer[0]"] == "apple processed"
    assert result.step_results["consumer[1]"] == "banana processed"


@pytest.mark.anyio
async def test_step_failure_stops_execution(workspace: Path):
    class FailingProvider(Provider):
        name = "failing"
        default_model = "failing-1"

        async def _complete(self, messages, *, system, tools, output_schema, max_tokens, **kwargs) -> Completion:
            raise RuntimeError("API connection failure")

    plan = Plan(
        goal="fail gracefully",
        steps=[
            PlanStep(key="s1", name="S1", role="r1", instruction="fail me"),
            PlanStep(key="s2", name="S2", role="r2", instruction="will not run", depends_on=["s1"]),
        ],
    )
    executor = LocalExecutor(FailingProvider("failing-1"), workspace=workspace)
    result = await executor.run(plan)

    assert result.status == "failed"
    assert "API connection failure" in (result.error or "")
    assert "s1" not in result.step_results
    assert "s2" not in result.step_results

@pytest.mark.anyio
async def test_on_event_reports_step_and_run_lifecycle(workspace: Path):
    provider = ScriptedProvider(responses={"step 1 instruction": "result 1"})
    plan = Plan(
        goal="run one step",
        steps=[PlanStep(key="s1", name="Step 1", role="role 1", instruction="step 1 instruction")],
    )
    events: list[tuple[str, dict]] = []
    executor = LocalExecutor(provider, workspace=workspace, on_event=lambda e, d: events.append((e, d)))

    result = await executor.run(plan)

    assert result.status == "success"
    assert events == [
        ("step_started", {"key": "s1"}),
        ("step_done", {"key": "s1", "result": "result 1"}),
        ("run_done", {"status": "success", "step_results": {"s1": "result 1"}}),
    ]


@pytest.mark.anyio
async def test_on_event_reports_one_step_per_for_each_item(workspace: Path):
    provider = ScriptedProvider(
        responses={"list items": '["a", "b"]', "process item: a": "a done", "process item: b": "b done"}
    )
    plan = Plan(
        goal="run for_each steps",
        steps=[
            PlanStep(key="producer", name="Producer", role="rp", instruction="list items"),
            PlanStep(
                key="consumer",
                name="Consumer",
                role="rc",
                instruction="process item: {{item}}",
                for_each="producer",
            ),
        ],
    )
    events: list[tuple[str, dict]] = []
    executor = LocalExecutor(provider, workspace=workspace, on_event=lambda e, d: events.append((e, d)))

    await executor.run(plan)

    consumer_keys = {d["key"] for e, d in events if e == "step_started" and d["key"].startswith("consumer")}
    assert consumer_keys == {"consumer[0]", "consumer[1]"}


@pytest.mark.anyio
async def test_on_event_reports_step_failed_and_run_done_failed(workspace: Path):
    class FailingProvider(Provider):
        name = "failing"
        default_model = "failing-1"

        async def _complete(self, messages, *, system, tools, output_schema, max_tokens, **kwargs) -> Completion:
            raise RuntimeError("boom")

    plan = Plan(goal="fail", steps=[PlanStep(key="s1", name="S1", role="r1", instruction="fail me")])
    events: list[tuple[str, dict]] = []
    executor = LocalExecutor(
        FailingProvider("failing-1"), workspace=workspace, on_event=lambda e, d: events.append((e, d))
    )

    result = await executor.run(plan)

    assert result.status == "failed"
    failed_events = [d for e, d in events if e == "step_failed" and d["key"] == "s1"]
    assert len(failed_events) == 1
    assert "boom" in failed_events[0]["error"]
    assert events[-1][0] == "run_done"
    assert events[-1][1]["status"] == "failed"


@pytest.mark.anyio
async def test_a_broken_event_handler_does_not_break_the_run(workspace: Path):
    provider = ScriptedProvider(responses={"step 1 instruction": "result 1"})
    plan = Plan(
        goal="run one step",
        steps=[PlanStep(key="s1", name="Step 1", role="role 1", instruction="step 1 instruction")],
    )

    def broken_handler(event, data):
        raise ValueError("listener bug")

    executor = LocalExecutor(provider, workspace=workspace, on_event=broken_handler)
    result = await executor.run(plan)

    assert result.status == "success"
    assert result.step_results["s1"] == "result 1"


@pytest.mark.anyio
async def test_on_event_accepts_an_async_handler(workspace: Path):
    provider = ScriptedProvider(responses={"step 1 instruction": "result 1"})
    plan = Plan(
        goal="run one step",
        steps=[PlanStep(key="s1", name="Step 1", role="role 1", instruction="step 1 instruction")],
    )
    events: list[str] = []

    async def async_handler(event, data):
        events.append(event)

    executor = LocalExecutor(provider, workspace=workspace, on_event=async_handler)
    await executor.run(plan)

    assert events == ["step_started", "step_done", "run_done"]


@pytest.mark.anyio
async def test_independent_step_survives_an_unrelated_failure(workspace: Path):
    class MixedProvider(Provider):
        name = "mixed"
        default_model = "mixed-1"

        async def _complete(self, messages, *, system, tools, output_schema, max_tokens, **kwargs) -> Completion:
            user_msg = messages[-1].content if messages and messages[-1].content else ""
            if user_msg == "fail me":
                raise RuntimeError("boom")
            await asyncio.sleep(0.05)  # still in flight when s1 fails
            return Completion(text="done: " + user_msg, model=self.model)

    plan = Plan(
        goal="independent branch survives",
        steps=[
            PlanStep(key="s1", name="S1", role="r1", instruction="fail me"),
            PlanStep(key="c", name="C", role="rc", instruction="independent work"),
        ],
    )
    executor = LocalExecutor(MixedProvider("mixed-1"), workspace=workspace)
    result = await executor.run(plan)

    assert result.status == "failed"
    assert result.step_results["c"] == "done: independent work"


@pytest.mark.anyio
async def test_dependent_step_is_reported_as_skipped_not_as_its_own_failure(workspace: Path):
    class FailingProvider(Provider):
        name = "failing"
        default_model = "failing-1"

        async def _complete(self, messages, *, system, tools, output_schema, max_tokens, **kwargs) -> Completion:
            raise RuntimeError("boom")

    plan = Plan(
        goal="cascade reporting",
        steps=[
            PlanStep(key="s1", name="S1", role="r1", instruction="fail me"),
            PlanStep(key="s2", name="S2", role="r2", instruction="downstream", depends_on=["s1"]),
        ],
    )
    events: list[tuple[str, dict]] = []
    executor = LocalExecutor(
        FailingProvider("failing-1"), workspace=workspace, on_event=lambda e, d: events.append((e, d))
    )
    result = await executor.run(plan)

    # The run's headline error is the real cause, not the cascaded one.
    assert result.error == "Provider 'failing' call failed: boom"

    s2_failed = next(d for e, d in events if e == "step_failed" and d["key"] == "s2")
    assert "skipped" in s2_failed["error"]
    assert "s1" in s2_failed["error"]


@pytest.mark.anyio
async def test_a_failing_for_each_item_is_reported_and_its_siblings_still_complete(workspace: Path):
    class MixedProvider(Provider):
        name = "mixed"
        default_model = "mixed-1"

        async def _complete(self, messages, *, system, tools, output_schema, max_tokens, **kwargs) -> Completion:
            user_msg = messages[-1].content if messages and messages[-1].content else ""
            if user_msg == "list items":
                return Completion(text='["bad", "good"]', model=self.model)
            if "bad" in user_msg:
                raise RuntimeError("item failed")
            await asyncio.sleep(0.05)  # still in flight when the "bad" item fails
            return Completion(text="done: " + user_msg, model=self.model)

    plan = Plan(
        goal="for_each sibling survives",
        steps=[
            PlanStep(key="producer", name="P", role="rp", instruction="list items"),
            PlanStep(
                key="consumer", name="C", role="rc", instruction="process item: {{item}}", for_each="producer"
            ),
        ],
    )
    events: list[tuple[str, dict]] = []
    executor = LocalExecutor(
        MixedProvider("mixed-1"), workspace=workspace, on_event=lambda e, d: events.append((e, d))
    )
    result = await executor.run(plan)

    assert result.status == "failed"
    # The failing item's own result never lands, but its sibling's does — a
    # for_each item failing must not orphan/discard its siblings' work.
    assert "consumer[0]" not in result.step_results
    assert result.step_results["consumer[1]"] == "done: process item: good"

    # The failing item gets its own step_failed event (not just the group's),
    # so the GUI doesn't show it stuck on "running" forever.
    item_failed = next(d for e, d in events if e == "step_failed" and d["key"] == "consumer[0]")
    assert "item failed" in item_failed["error"]
    # And step_done for the surviving sibling must land before run_done —
    # not arrive late as an orphaned background task after the run reported done.
    assert events.index(("step_done", {"key": "consumer[1]", "result": "done: process item: good"})) < next(
        i for i, (e, _) in enumerate(events) if e == "run_done"
    )


@pytest.mark.anyio
async def test_plans_with_cyclic_dependency(workspace: Path):
    provider = ScriptedProvider(
        responses={
            "instruction 1": "instruction 1",
            "instruction 2": "instruction 2",
        }
    )
    cyclic_plan = Plan(
        goal="fail gracefully",
        steps=[
            PlanStep(key="s1", name="S1", role="r1", instruction="instruction 1", depends_on=["s2"]),
            PlanStep(key="s2", name="S2", role="r2", instruction="instruction 2", depends_on=["s1"]),
        ],
    )
    
    executor = LocalExecutor(provider, workspace=workspace)
    result = await executor.run(cyclic_plan)

    assert result.status == "failed"
    assert "circular dependency" in (result.error or "")
    assert "s1" not in result.step_results
    assert "s2" not in result.step_results
    

