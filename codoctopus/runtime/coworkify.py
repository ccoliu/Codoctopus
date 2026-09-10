# ---------------------------------------------------
# Codoctopus — Coworkify executor
#
# Translates a Plan into a Coworkify workflow, submits it over Coworkify's
# REST API, polls until it finishes, and reports back a PlanResult built from
# each task's actual output. This is the executor for M5: the same Plan that
# LocalExecutor runs in-process can instead run on Coworkify's Celery workers,
# with retries, priority, and persistence Coworkify already provides.
#
# Every step is submitted as task_type "agent_step" (see Coworkify's
# app/tasks/handler.py::handle_agent_step) — that handler is Gap C: it is
# what lets a Coworkify worker actually run a codoctopus.agents.Agent.
#
# Step-result correlation: a Coworkify WorkflowStepResponse only carries
# `task_name`, not the plan step's `key` — so `name` is set to `step.key`
# when building the workflow, and PlanStep.name goes along for the ride
# inside the payload (for dashboards) rather than as the correlation field.
# Coworkify expands a for_each step's copies as "<name> [<i+1>]" (1-indexed,
# space before the bracket); results are re-keyed to "<key>[<i>]" (0-indexed,
# no space) to match LocalExecutor's convention — a PlanResult.step_results
# dict should look the same regardless of which executor produced it.
# ---------------------------------------------------

from __future__ import annotations

import asyncio
import re
import time
import uuid
from typing import Any

import httpx

from codoctopus.config import Settings, get_settings
from codoctopus.planning.models import Plan
from codoctopus.planning.validate import validate_plan
from codoctopus.runtime.base import Executor, PlanResult

_TERMINAL_STATUSES = {"success", "failed"}
_EXPANDED_TASK_NAME = re.compile(r"^(?P<key>.+) \[(?P<index>\d+)\]$")


def _normalize_step_key(task_name: str) -> str:
    match = _EXPANDED_TASK_NAME.fullmatch(task_name)
    if not match:
        return task_name
    return f"{match.group('key')}[{int(match.group('index')) - 1}]"


class CoworkifyExecutor(Executor):
    def __init__(
        self,
        base_url: str,
        *,
        token: str,
        model: str = "anthropic:claude-opus-5",
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.model = model
        self.poll_interval = poll_interval
        self.timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> CoworkifyExecutor:
        settings = settings or get_settings()
        if not settings.uses_coworkify:
            raise ValueError("CODOCTOPUS_COWORKIFY_URL is not set")
        return cls(settings.coworkify_url, token=settings.coworkify_token, model=settings.worker_model)

    async def run(self, plan: Plan) -> PlanResult:
        validate_plan(plan)

        # Every agent_step task that gets a `workspace` in its payload shares
        # that exact directory (see coworkify's handler.py — it falls back to
        # one directory shared across *every* agent_step ever run when
        # nothing is passed). Scoping it to a fresh id per run.() call means
        # two concurrent (or sequential) runs writing files of the same name
        # no longer clobber each other.
        run_workspace = f"/tmp/codoctopus-agent-steps/{uuid.uuid4().hex}"

        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=30.0,
        ) as client:
            created = await client.post(
                "/workflows/",
                json={"name": plan.goal[:255] or "codoctopus plan", "steps": self._to_workflow_steps(plan, run_workspace)},
            )
            created.raise_for_status()
            workflow_id = created.json()["id"]

            workflow = await self._poll_until_done(client, workflow_id)
            if workflow is None:
                return PlanResult(
                    plan=plan,
                    status="failed",
                    error=f"timed out after {self.timeout}s waiting for workflow {workflow_id}",
                )

            step_results, error = await self._collect_results(client, workflow)

        return PlanResult(
            plan=plan,
            status="success" if workflow["status"] == "success" else "failed",
            step_results=step_results,
            error=error,
        )

    async def create_schedule(
        self, plan: Plan, *, cron_expression: str, name: str, enabled: bool = True
    ) -> dict[str, Any]:
        """
        Register `plan` as a recurring Coworkify schedule instead of running it
        once. Coworkify's Celery Beat re-creates a fresh workflow from the
        same steps template each time the cron fires — see its README's
        Schedules section.

        Unlike run(), every future firing shares one workspace (there is no
        per-invocation hook to give each a fresh one) — that's a deliberate
        trade-off: a cron job re-running the same task and landing in the
        same place each time is the expected shape, not a collision.
        """
        validate_plan(plan)
        workspace = f"/tmp/codoctopus-agent-steps/schedule-{uuid.uuid4().hex}"

        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=30.0,
        ) as client:
            response = await client.post(
                "/schedules/",
                json={
                    "name": name,
                    "cron_expression": cron_expression,
                    "steps": self._to_workflow_steps(plan, workspace),
                    "enabled": enabled,
                },
            )
            response.raise_for_status()
            return response.json()

    def _to_workflow_steps(self, plan: Plan, workspace: str) -> list[dict[str, Any]]:
        return [
            {
                "key": step.key,
                # Set to step.key, not step.name, so the polling response's
                # task_name can be mapped straight back to this step — see
                # the module docstring.
                "name": step.key,
                "task_type": "agent_step",
                "payload": {
                    "role": step.role,
                    "instruction": step.instruction,
                    "model": self.model,
                    "tools": step.tools,
                    "step_name": step.name,
                    "workspace": workspace,
                },
                "depends_on": step.depends_on,
                "for_each": step.for_each,
            }
            for step in plan.steps
        ]

    async def _poll_until_done(self, client: httpx.AsyncClient, workflow_id: str) -> dict[str, Any] | None:
        deadline = time.monotonic() + self.timeout
        while True:
            response = await client.get(f"/workflows/{workflow_id}")
            response.raise_for_status()
            workflow = response.json()
            if workflow["status"] in _TERMINAL_STATUSES:
                return workflow
            if time.monotonic() > deadline:
                return None
            await asyncio.sleep(self.poll_interval)

    async def _collect_results(
        self, client: httpx.AsyncClient, workflow: dict[str, Any]
    ) -> tuple[dict[str, str], str | None]:
        step_results: dict[str, str] = {}
        errors: list[str] = []

        for step in workflow["steps"]:
            response = await client.get(f"/tasks/{step['task_id']}/result")
            response.raise_for_status()
            data = response.json()
            key = _normalize_step_key(step["task_name"])

            if data["status"] == "success":
                step_results[key] = (data.get("result") or {}).get("output", "")
            elif data["status"] == "failed":
                errors.append(f"{key}: {data.get('error_message') or 'failed'}")

        return step_results, "; ".join(errors) if errors else None
