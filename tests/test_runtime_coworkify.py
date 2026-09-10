# ---------------------------------------------------
# Codoctopus — CoworkifyExecutor tests
#
# A small stateful fake server, wired in via httpx.MockTransport (same
# pattern as tests/test_tools_http.py), stands in for Coworkify: it accepts
# POST /workflows/, then GET /workflows/{id} and GET /tasks/{id}/result the
# way the real API would. This proves the HTTP contract without a running
# Coworkify instance.
# ---------------------------------------------------

from __future__ import annotations

import httpx
import pytest

from codoctopus.planning.models import Plan, PlanStep
from codoctopus.runtime.coworkify import CoworkifyExecutor, _normalize_step_key


def step(key: str, **kwargs) -> PlanStep:
    kwargs.setdefault("instruction", f"do {key}")
    return PlanStep(key=key, name=f"Step {key}", role="be helpful", **kwargs)


class FakeCoworkify:
    """
    Enough of Coworkify's API to drive CoworkifyExecutor: create a workflow,
    report "pending" a fixed number of times, then a final status, and answer
    per-task result lookups from a table the test sets up.
    """

    def __init__(self, *, pending_polls: int = 1, final_status: str = "success") -> None:
        self.pending_polls = pending_polls
        self.final_status = final_status
        self.created_body: dict | None = None
        self.created_schedule_body: dict | None = None
        self.polls = 0
        #: task_name (== the WorkflowStepCreate "name" sent in) -> (task_id, status, result, error)
        self.tasks: dict[str, tuple[str, str, dict | None, str | None]] = {}

    def add_task_result(self, name: str, *, status: str, output: str | None = None, error: str | None = None) -> None:
        task_id = f"task-{name}"
        result = {"output": output} if output is not None else None
        self.tasks[name] = (task_id, status, result, error)

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-token"

        if request.method == "POST" and request.url.path == "/workflows/":
            import json

            self.created_body = json.loads(request.content)
            steps = [
                {
                    "id": f"step-{s['key']}",
                    "task_id": self.tasks[s["name"]][0],
                    "task_name": s["name"],
                    "task_status": "pending",
                    "depends_on": [],
                }
                for s in self.created_body["steps"]
            ]
            return httpx.Response(
                201, json={"id": "wf-1", "name": self.created_body["name"], "status": "pending", "steps": steps}
            )

        if request.method == "POST" and request.url.path == "/schedules/":
            import json

            self.created_schedule_body = json.loads(request.content)
            return httpx.Response(
                201,
                json={
                    "id": "sched-1",
                    "name": self.created_schedule_body["name"],
                    "cron_expression": self.created_schedule_body["cron_expression"],
                    "steps": self.created_schedule_body["steps"],
                    "enabled": self.created_schedule_body["enabled"],
                    "last_run_at": None,
                    "next_run_at": "2026-09-11T09:00:00",
                    "created_at": "2026-09-10T00:00:00",
                    "updated_at": "2026-09-10T00:00:00",
                },
            )

        if request.method == "GET" and request.url.path == "/workflows/wf-1":
            self.polls += 1
            status = "pending" if self.polls <= self.pending_polls else self.final_status
            steps = [
                {"id": f"step-{name}", "task_id": tid, "task_name": name, "task_status": st, "depends_on": []}
                for name, (tid, st, _, _) in self.tasks.items()
            ]
            return httpx.Response(200, json={"id": "wf-1", "name": "x", "status": status, "steps": steps})

        for name, (task_id, status, result, error) in self.tasks.items():
            if request.method == "GET" and request.url.path == f"/tasks/{task_id}/result":
                return httpx.Response(
                    200, json={"task_id": task_id, "status": status, "result": result, "error_message": error}
                )

        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")


@pytest.fixture
def patch_client(monkeypatch):
    """Route CoworkifyExecutor's httpx.AsyncClient through a FakeCoworkify instance."""

    def apply(fake: FakeCoworkify):
        real_async_client = httpx.AsyncClient

        def make_client(**kwargs):
            return real_async_client(transport=httpx.MockTransport(fake.handler), **kwargs)

        monkeypatch.setattr("codoctopus.runtime.coworkify.httpx.AsyncClient", make_client)

    return apply


# --- key normalization ---------------------------------------------------


def test_normalize_step_key_passes_through_a_plain_name():
    assert _normalize_step_key("write") == "write"


def test_normalize_step_key_converts_coworkifys_expansion_format():
    assert _normalize_step_key("tailor [1]") == "tailor[0]"
    assert _normalize_step_key("tailor [3]") == "tailor[2]"


# --- workflow translation -------------------------------------------------


async def test_plan_steps_become_agent_step_tasks_keyed_by_plan_key(patch_client):
    fake = FakeCoworkify(pending_polls=0)
    fake.add_task_result("write", status="success", output="wrote it")
    patch_client(fake)

    executor = CoworkifyExecutor("http://coworkify.test", token="test-token", model="scripted:x")
    plan = Plan(goal="ship it", steps=[step("write")])

    await executor.run(plan)

    sent = fake.created_body["steps"][0]
    assert sent["name"] == "write"
    assert sent["task_type"] == "agent_step"
    workspace = sent["payload"].pop("workspace")
    assert workspace.startswith("/tmp/codoctopus-agent-steps/")
    assert sent["payload"] == {
        "role": "be helpful",
        "instruction": "do write",
        "model": "scripted:x",
        "tools": [],
        "step_name": "Step write",
    }


async def test_each_run_gets_its_own_workspace_so_concurrent_runs_cant_collide(patch_client):
    fake = FakeCoworkify(pending_polls=0)
    fake.add_task_result("write", status="success", output="wrote it")
    patch_client(fake)

    executor = CoworkifyExecutor("http://coworkify.test", token="test-token", model="scripted:x")
    plan = Plan(goal="ship it", steps=[step("write")])

    await executor.run(plan)
    first_workspace = fake.created_body["steps"][0]["payload"]["workspace"]

    await executor.run(plan)
    second_workspace = fake.created_body["steps"][0]["payload"]["workspace"]

    assert first_workspace != second_workspace


async def test_depends_on_and_for_each_pass_through_unchanged(patch_client):
    fake = FakeCoworkify(pending_polls=0)
    fake.add_task_result("search", status="success", output="[]")
    fake.add_task_result("apply", status="success", output="")
    patch_client(fake)

    executor = CoworkifyExecutor("http://coworkify.test", token="test-token")
    plan = Plan(
        goal="apply to jobs",
        steps=[step("search"), step("apply", for_each="search", depends_on=[])],
    )

    await executor.run(plan)

    by_key = {s["key"]: s for s in fake.created_body["steps"]}
    assert by_key["apply"]["for_each"] == "search"


# --- polling ---------------------------------------------------------------


async def test_polls_until_a_terminal_status(patch_client):
    fake = FakeCoworkify(pending_polls=2, final_status="success")
    fake.add_task_result("write", status="success", output="done")
    patch_client(fake)

    executor = CoworkifyExecutor("http://coworkify.test", token="test-token", poll_interval=0)
    plan = Plan(goal="ship it", steps=[step("write")])

    result = await executor.run(plan)

    assert fake.polls == 3
    assert result.status == "success"
    assert result.step_results == {"write": "done"}


async def test_times_out_if_the_workflow_never_finishes(patch_client):
    fake = FakeCoworkify(pending_polls=10**6)
    fake.add_task_result("write", status="pending")
    patch_client(fake)

    executor = CoworkifyExecutor("http://coworkify.test", token="test-token", poll_interval=0, timeout=0)
    plan = Plan(goal="ship it", steps=[step("write")])

    result = await executor.run(plan)

    assert result.status == "failed"
    assert "timed out" in result.error


# --- result collection ------------------------------------------------------


async def test_a_failed_step_is_reported_with_its_error(patch_client):
    fake = FakeCoworkify(pending_polls=0, final_status="failed")
    fake.add_task_result("write", status="success", output="wrote it")
    fake.add_task_result("test", status="failed", error="assertion failed")
    patch_client(fake)

    executor = CoworkifyExecutor("http://coworkify.test", token="test-token")
    plan = Plan(goal="ship it", steps=[step("write"), step("test", depends_on=["write"])])

    result = await executor.run(plan)

    assert result.status == "failed"
    assert result.step_results["write"] == "wrote it"
    assert "test" in result.error
    assert "assertion failed" in result.error


# --- an invalid plan never reaches the network ------------------------------


async def test_an_invalid_plan_is_rejected_before_any_request(patch_client):
    fake = FakeCoworkify()
    patch_client(fake)

    executor = CoworkifyExecutor("http://coworkify.test", token="test-token")
    plan = Plan(goal="loop", steps=[step("a", depends_on=["b"]), step("b", depends_on=["a"])])

    with pytest.raises(Exception, match="circular dependency"):
        await executor.run(plan)

    assert fake.created_body is None


# --- cron schedules ---------------------------------------------------------


async def test_create_schedule_posts_the_plan_as_a_workflow_template(patch_client):
    fake = FakeCoworkify()
    patch_client(fake)

    executor = CoworkifyExecutor("http://coworkify.test", token="test-token", model="scripted:x")
    plan = Plan(goal="daily digest", steps=[step("write"), step("send", depends_on=["write"])])

    result = await executor.create_schedule(plan, cron_expression="0 9 * * *", name="daily-digest")

    assert result["id"] == "sched-1"
    assert result["next_run_at"] == "2026-09-11T09:00:00"
    assert fake.created_schedule_body["name"] == "daily-digest"
    assert fake.created_schedule_body["cron_expression"] == "0 9 * * *"
    assert fake.created_schedule_body["enabled"] is True
    sent_steps = {s["key"]: s for s in fake.created_schedule_body["steps"]}
    assert sent_steps["send"]["depends_on"] == ["write"]
    assert sent_steps["write"]["payload"]["workspace"].startswith("/tmp/codoctopus-agent-steps/")
    # create_schedule never talks to /workflows/ — it only registers the template.
    assert fake.created_body is None


async def test_create_schedule_rejects_an_invalid_plan_before_any_request(patch_client):
    fake = FakeCoworkify()
    patch_client(fake)

    executor = CoworkifyExecutor("http://coworkify.test", token="test-token")
    plan = Plan(goal="loop", steps=[step("a", depends_on=["b"]), step("b", depends_on=["a"])])

    with pytest.raises(Exception, match="circular dependency"):
        await executor.create_schedule(plan, cron_expression="0 9 * * *", name="bad")

    assert fake.created_schedule_body is None


# --- from_settings -----------------------------------------------------


def test_from_settings_requires_coworkify_url(monkeypatch):
    from codoctopus.config import Settings

    with pytest.raises(ValueError, match="CODOCTOPUS_COWORKIFY_URL"):
        CoworkifyExecutor.from_settings(Settings(coworkify_url=""))


def test_from_settings_builds_from_configured_values():
    from codoctopus.config import Settings

    executor = CoworkifyExecutor.from_settings(
        Settings(coworkify_url="http://coworkify.internal", coworkify_token="tok", worker_model="scripted:x")
    )

    assert executor.base_url == "http://coworkify.internal"
    assert executor.token == "tok"
    assert executor.model == "scripted:x"
