# ---------------------------------------------------
# Codoctopus — GUI backend tests
#
# A scripted provider stands in for the model, same pattern as test_cli.py.
# TestClient (used as a context manager) keeps a real background event loop
# alive for the whole test, so RunManager's asyncio.create_task background
# runs actually progress between requests — no manual sleeping needed, the
# WebSocket read is itself what lets the run advance.
# ---------------------------------------------------

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from codoctopus.config import get_settings, reset_settings
from codoctopus.llm import Completion, Provider, StopReason, register_provider
from codoctopus.planning import Plan
from codoctopus.server import create_app
from codoctopus.server.runs import Run, RunManager, ScheduleError


class ScriptedProvider(Provider):
    """Plans with a fixed one-step Plan, then answers that step with fixed text."""

    name = "scripted"
    default_model = "scripted-1"

    @property
    def supports_structured_output(self) -> bool:
        return True

    async def _complete(self, messages, *, system, tools, output_schema, max_tokens, **kwargs) -> Completion:
        if output_schema is Plan:
            plan = Plan.model_validate(
                {
                    "goal": messages[0].content.split("Goal: ")[1].split("\n")[0],
                    "steps": [{"key": "write", "name": "write", "role": "coder", "instruction": "do the thing"}],
                }
            )
            return Completion(text=plan.model_dump_json(), model=self.model, parsed=plan)
        return Completion(text="step done", model=self.model, stop_reason=StopReason.END_TURN)


class BrokenPlannerProvider(Provider):
    name = "broken-planner"
    default_model = "x"

    @property
    def supports_structured_output(self) -> bool:
        return True

    async def _complete(self, messages, *, system, tools, output_schema, max_tokens, **kwargs) -> Completion:
        raise RuntimeError("planner exploded")


construction_calls: list[dict[str, Any]] = []
models_calls: list[dict[str, Any]] = []


def _make_scripted(model=None, **kw):
    construction_calls.append({"model": model, **kw})
    return ScriptedProvider(model, **kw)


async def _scripted_list_models(**kw):
    models_calls.append(kw)
    return ["scripted-1", "scripted-2"]


@pytest.fixture(autouse=True)
def _register_scripted(tmp_path, monkeypatch):
    construction_calls.clear()
    models_calls.clear()
    register_provider("scripted", _make_scripted, list_models=_scripted_list_models)
    register_provider("broken-planner", lambda model=None, **kw: BrokenPlannerProvider(model, **kw))
    monkeypatch.setenv("CODOCTOPUS_WORKSPACE", str(tmp_path))
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def _collect_stream(client: TestClient, run_id: str) -> list[dict[str, Any]]:
    events = []
    with client.websocket_connect(f"/api/runs/{run_id}/stream") as ws:
        while True:
            msg = ws.receive_json()
            events.append(msg)
            if msg["event"] == "run_done":
                break
    return events


def test_list_domains_includes_coding(client: TestClient):
    resp = client.get("/api/domains")
    assert resp.status_code == 200
    assert "coding" in resp.json()["domains"]


def test_list_providers_includes_the_registered_scripted_provider(client: TestClient):
    resp = client.get("/api/providers")
    assert resp.status_code == 200
    assert "scripted" in resp.json()["providers"]


def test_list_provider_models_returns_the_providers_models(client: TestClient):
    resp = client.post("/api/providers/scripted/models", json={"api_key": "secret"})
    assert resp.status_code == 200
    assert resp.json()["models"] == ["scripted-1", "scripted-2"]
    assert models_calls[-1] == {"api_key": "secret"}


def test_list_provider_models_omits_unset_credentials(client: TestClient):
    client.post("/api/providers/scripted/models", json={})
    assert models_calls[-1] == {}


def test_list_provider_models_for_an_unknown_provider_is_a_clean_400(client: TestClient):
    resp = client.post("/api/providers/nope/models", json={})
    assert resp.status_code == 400
    assert "Unknown provider 'nope'" in resp.json()["detail"]


def test_list_provider_models_for_a_provider_without_support_is_a_clean_400(client: TestClient):
    resp = client.post("/api/providers/broken-planner/models", json={})
    assert resp.status_code == 400
    assert "does not support listing models" in resp.json()["detail"]


def test_create_run_returns_immediately_with_a_run_id(client: TestClient):
    resp = client.post("/api/runs", json={"goal": "ship it", "model": "scripted:x"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["id"]
    assert body["goal"] == "ship it"
    assert body["status"] in ("planning", "running", "success")


def test_get_unknown_run_is_404(client: TestClient):
    resp = client.get("/api/runs/does-not-exist")
    assert resp.status_code == 404


def test_run_appears_in_the_list(client: TestClient):
    created = client.post("/api/runs", json={"goal": "ship it", "model": "scripted:x"}).json()
    listed = client.get("/api/runs").json()["runs"]
    assert any(r["id"] == created["id"] for r in listed)


def test_stream_reports_plan_then_step_then_run_done(client: TestClient):
    created = client.post(
        "/api/runs", json={"goal": "ship it", "model": "scripted:x", "worker_model": "scripted:x"}
    ).json()

    events = _collect_stream(client, created["id"])
    kinds = [e["event"] for e in events]

    assert kinds[0] == "snapshot"
    assert "plan_ready" in kinds
    assert kinds.index("plan_ready") < kinds.index("step_started")
    assert kinds.index("step_started") < kinds.index("step_done")
    assert kinds[-1] == "run_done"
    assert events[-1]["data"]["status"] == "success"
    assert events[-1]["data"]["step_results"]["write"] == "step done"


def test_get_run_after_completion_has_the_final_result(client: TestClient):
    created = client.post(
        "/api/runs", json={"goal": "ship it", "model": "scripted:x", "worker_model": "scripted:x"}
    ).json()
    _collect_stream(client, created["id"])

    detail = client.get(f"/api/runs/{created['id']}").json()
    assert detail["status"] == "success"
    assert detail["step_results"]["write"] == "step done"
    assert detail["plan"]["goal"] == "ship it"


def test_a_local_runs_workspace_path_is_reported(client: TestClient):
    # Each run's directory is named by its (opaque) id — this is the only way
    # to find out where a run actually wrote its files without hunting
    # through .codoctopus/workspace/ by hand.
    created = client.post(
        "/api/runs",
        json={"goal": "ship it", "model": "scripted:x", "worker_model": "scripted:x", "executor": "local"},
    ).json()

    events = _collect_stream(client, created["id"])
    plan_ready = next(e for e in events if e["event"] == "plan_ready")
    assert plan_ready["data"]["workspace"]
    assert created["id"] in plan_ready["data"]["workspace"]

    detail = client.get(f"/api/runs/{created['id']}").json()
    assert detail["workspace"] == plan_ready["data"]["workspace"]


def test_a_coworkify_runs_workspace_is_reported_as_none(client: TestClient, monkeypatch):
    # Its files live on whichever Coworkify worker executed it, not here —
    # reporting a local path would be actively misleading.
    monkeypatch.delenv("CODOCTOPUS_COWORKIFY_URL", raising=False)
    created = client.post(
        "/api/runs",
        json={"goal": "ship it", "model": "scripted:x", "executor": "coworkify"},
    ).json()

    events = _collect_stream(client, created["id"])
    plan_ready = next(e for e in events if e["event"] == "plan_ready")
    assert plan_ready["data"]["workspace"] is None

    detail = client.get(f"/api/runs/{created['id']}").json()
    assert detail["workspace"] is None


def test_connecting_after_the_run_finished_still_gets_the_full_history(client: TestClient):
    created = client.post(
        "/api/runs", json={"goal": "ship it", "model": "scripted:x", "worker_model": "scripted:x"}
    ).json()
    _collect_stream(client, created["id"])  # drain to completion once

    # A second connection after the run is over should replay history and
    # close on its own — not hang waiting for events that already happened.
    with client.websocket_connect(f"/api/runs/{created['id']}/stream") as ws:
        first = ws.receive_json()
        assert first["event"] == "snapshot"
        assert first["data"]["status"] == "success"


def test_credentials_from_the_request_reach_the_provider_constructor(client: TestClient):
    created = client.post(
        "/api/runs",
        json={
            "goal": "ship it",
            "model": "scripted:x",
            "worker_model": "scripted:y",
            "planner_api_key": "planner-secret",
            "worker_api_key": "worker-secret",
            "worker_base_url": "http://localhost:1234/v1",
        },
    ).json()
    _collect_stream(client, created["id"])

    by_model = {c["model"]: c for c in construction_calls}
    assert by_model["x"]["api_key"] == "planner-secret"
    assert "base_url" not in by_model["x"]
    assert by_model["y"]["api_key"] == "worker-secret"
    assert by_model["y"]["base_url"] == "http://localhost:1234/v1"


def test_omitted_credentials_are_not_sent_to_the_provider_at_all(client: TestClient):
    created = client.post("/api/runs", json={"goal": "ship it", "model": "scripted:z"}).json()
    _collect_stream(client, created["id"])

    call = next(c for c in construction_calls if c["model"] == "z")
    assert "api_key" not in call
    assert "base_url" not in call


def _fake_coworkify_schedule_transport(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/schedules/"
        body = json.loads(request.content)
        captured["body"] = body
        return httpx.Response(
            201,
            json={
                "id": "sched-1",
                "name": body["name"],
                "cron_expression": body["cron_expression"],
                "steps": body["steps"],
                "enabled": body["enabled"],
                "last_run_at": None,
                "next_run_at": "2026-09-11T09:00:00",
                "created_at": "2026-09-10T00:00:00",
                "updated_at": "2026-09-10T00:00:00",
            },
        )

    return httpx.MockTransport(handler)


def test_create_schedule_registers_the_runs_plan_as_a_coworkify_cron(monkeypatch):
    # manager captures Settings by reference at create_app() time, so
    # coworkify must be configured *before* building the app — reusing the
    # shared `client` fixture here would reconfigure too late to matter.
    monkeypatch.setenv("CODOCTOPUS_COWORKIFY_URL", "http://coworkify.test")
    monkeypatch.setenv("CODOCTOPUS_COWORKIFY_TOKEN", "test-token")
    reset_settings()

    captured: dict = {}
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        "codoctopus.runtime.coworkify.httpx.AsyncClient",
        lambda **kw: real_async_client(transport=_fake_coworkify_schedule_transport(captured), **kw),
    )

    with TestClient(create_app()) as client:
        # The run itself stays on the default local executor — scheduling
        # only needs its finished Plan, regardless of how it first ran.
        created = client.post("/api/runs", json={"goal": "daily digest", "model": "scripted:x"}).json()
        _collect_stream(client, created["id"])  # wait for planning to finish

        resp = client.post(
            f"/api/runs/{created['id']}/schedule", json={"name": "daily-digest", "cron_expression": "0 9 * * *"}
        )

    assert resp.status_code == 201
    assert resp.json()["next_run_at"] == "2026-09-11T09:00:00"
    assert captured["body"]["name"] == "daily-digest"
    assert captured["body"]["cron_expression"] == "0 9 * * *"
    assert captured["body"]["steps"][0]["key"] == "write"


def test_create_schedule_without_coworkify_configured_is_a_clean_400(client: TestClient, monkeypatch):
    monkeypatch.delenv("CODOCTOPUS_COWORKIFY_URL", raising=False)
    reset_settings()

    created = client.post("/api/runs", json={"goal": "daily digest", "model": "scripted:x"}).json()
    _collect_stream(client, created["id"])

    resp = client.post(
        f"/api/runs/{created['id']}/schedule", json={"name": "daily-digest", "cron_expression": "0 9 * * *"}
    )

    assert resp.status_code == 400
    assert "CODOCTOPUS_COWORKIFY_URL" in resp.json()["detail"]


async def test_create_schedule_before_planning_finished_is_a_clean_400(monkeypatch):
    # Racing a real run to catch it mid-planning is flaky (the scripted
    # provider often finishes before the next request lands) — a run with no
    # plan set is exercised directly against RunManager instead.
    monkeypatch.setenv("CODOCTOPUS_COWORKIFY_URL", "http://coworkify.test")
    monkeypatch.setenv("CODOCTOPUS_COWORKIFY_TOKEN", "test-token")
    reset_settings()

    manager = RunManager(get_settings())
    run = Run(id="r1", goal="daily digest", domain=None, executor="local")
    manager._runs["r1"] = run

    with pytest.raises(ScheduleError, match="no plan yet"):
        await manager.create_schedule("r1", cron_expression="0 9 * * *", name="daily-digest")


def test_create_schedule_for_an_unknown_run_is_404(client: TestClient):
    resp = client.post(
        "/api/runs/does-not-exist/schedule", json={"name": "x", "cron_expression": "0 9 * * *"}
    )
    assert resp.status_code == 404


def test_a_planning_failure_reports_run_done_failed(client: TestClient):
    created = client.post("/api/runs", json={"goal": "ship it", "model": "broken-planner:x"}).json()

    events = _collect_stream(client, created["id"])

    assert events[-1]["event"] == "run_done"
    assert events[-1]["data"]["status"] == "failed"
    assert "planner exploded" in events[-1]["data"]["error"]

    detail = client.get(f"/api/runs/{created['id']}").json()
    assert detail["status"] == "failed"
