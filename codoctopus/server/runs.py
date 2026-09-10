# ---------------------------------------------------
# Codoctopus — Run orchestration for the GUI backend
#
# A Run wraps one make_plan + Executor.run() call with everything a browser
# needs to watch it happen: a status a poller can read, and a per-run event
# log that both a freshly-connecting WebSocket (as a snapshot) and a live one
# (as they're appended) can consume. Kept in memory — this is a local dev
# tool for one user watching their own runs, not a durable job queue.
# ---------------------------------------------------

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from codoctopus.config import Settings
from codoctopus.domains import get_domain
from codoctopus.llm import get_provider
from codoctopus.planning import Plan, make_plan
from codoctopus.runtime import CoworkifyExecutor, LocalExecutor
from codoctopus.tools.builtin import BUILTIN_TOOLS, build_tool_registry

RunStatus = Literal["planning", "running", "success", "failed"]


def _provider_kwargs(api_key: str | None, base_url: str | None) -> dict[str, str]:
    """
    Only pass through what's actually set. A provider that doesn't accept a
    given kwarg (e.g. Ollama has no api_key) just stores it in its unused
    **options — get_provider's own wrapping turns a real mismatch into a
    clean ProviderError instead of a raw TypeError either way.
    """
    kwargs: dict[str, str] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs

#: Pushed to a subscriber's queue right after "run_done" so its WebSocket
#: handler knows the run is over and it can close the connection.
DONE = object()


@dataclass
class Run:
    id: str
    goal: str
    domain: str | None
    executor: str
    status: RunStatus = "planning"
    created_at: float = field(default_factory=time.time)
    plan: Plan | None = None
    step_results: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "domain": self.domain,
            "executor": self.executor,
            "status": self.status,
            "created_at": self.created_at,
        }

    def to_detail(self) -> dict[str, Any]:
        return {
            **self.to_summary(),
            "plan": self.plan.model_dump() if self.plan else None,
            "step_results": self.step_results,
            "error": self.error,
        }


class RunManager:
    """Creates Runs, executes them in the background, and fans out their events."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._runs: dict[str, Run] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def list(self) -> list[Run]:
        return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    def history(self, run_id: str) -> list[dict[str, Any]]:
        return list(self._events.get(run_id, []))

    def create(
        self,
        goal: str,
        *,
        domain: str | None = None,
        model: str | None = None,
        worker_model: str | None = None,
        executor: str = "local",
        workspace: str | None = None,
        planner_api_key: str | None = None,
        planner_base_url: str | None = None,
        worker_api_key: str | None = None,
        worker_base_url: str | None = None,
    ) -> Run:
        run = Run(id=str(uuid.uuid4()), goal=goal, domain=domain, executor=executor)
        self._runs[run.id] = run
        self._events[run.id] = []
        asyncio.create_task(
            self._execute(
                run.id,
                domain=domain,
                model=model,
                worker_model=worker_model,
                workspace=workspace,
                planner_api_key=planner_api_key,
                planner_base_url=planner_base_url,
                worker_api_key=worker_api_key,
                worker_base_url=worker_base_url,
            )
        )
        return run

    async def subscribe(self, run_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(run_id, []).append(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        subs = self._subscribers.get(run_id)
        if subs and queue in subs:
            subs.remove(queue)

    async def _broadcast(self, run_id: str, event: str, data: dict[str, Any]) -> None:
        entry = {"event": event, "data": data}
        self._events[run_id].append(entry)
        for queue in list(self._subscribers.get(run_id, [])):
            await queue.put(entry)
        if event == "run_done":
            for queue in list(self._subscribers.get(run_id, [])):
                await queue.put(DONE)

    async def _execute(
        self,
        run_id: str,
        *,
        domain: str | None,
        model: str | None,
        worker_model: str | None,
        workspace: str | None,
        planner_api_key: str | None = None,
        planner_base_url: str | None = None,
        worker_api_key: str | None = None,
        worker_base_url: str | None = None,
    ) -> None:
        run = self._runs[run_id]
        try:
            domain_obj = get_domain(domain) if domain else None
            planner_provider = get_provider(
                model or self._settings.planner_model,
                **_provider_kwargs(planner_api_key, planner_base_url),
            )
            plan = await make_plan(
                planner_provider, run.goal, domain=domain_obj, available_tools=list(BUILTIN_TOOLS)
            )
            run.plan = plan
            run.status = "running"
            await self._broadcast(run_id, "plan_ready", {"plan": plan.model_dump()})

            ws_path = Path(workspace) if workspace else self._settings.workspace / run_id
            ws_path.mkdir(parents=True, exist_ok=True)

            if run.executor == "coworkify":
                # CoworkifyExecutor has no on_event hook (Coworkify's own DAG
                # scheduling runs the steps) — the GUI gets plan_ready above
                # and one run_done at the end instead of per-step updates.
                executor = CoworkifyExecutor.from_settings(self._settings)
                result = await executor.run(plan)
            else:
                worker_provider = get_provider(
                    worker_model or self._settings.worker_model,
                    **_provider_kwargs(worker_api_key, worker_base_url),
                )
                tools = build_tool_registry(ws_path)

                async def on_event(event: str, data: dict[str, Any]) -> None:
                    if event != "run_done":  # emitted once, below, for both executors alike
                        await self._broadcast(run_id, event, data)

                executor = LocalExecutor(worker_provider, workspace=ws_path, tools=tools, on_event=on_event)
                result = await executor.run(plan)

            run.status = result.status
            run.step_results = result.step_results
            run.error = result.error
            await self._broadcast(
                run_id,
                "run_done",
                {"status": result.status, "step_results": result.step_results, "error": result.error},
            )
        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
            await self._broadcast(
                run_id, "run_done", {"status": "failed", "step_results": run.step_results, "error": str(exc)}
            )
