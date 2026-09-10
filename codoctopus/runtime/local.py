# ---------------------------------------------------
# Codoctopus — Local Executor
#
# Runs a Plan directly in the current process using asyncio.
# Handles DAG scheduling, step dependency resolution, instruction templating,
# and for_each expansion with zero external infrastructure.
# ---------------------------------------------------

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any, Callable

from codoctopus.agents import Agent
from codoctopus.llm import Provider
from codoctopus.planning.models import Plan, PlanStep
from codoctopus.planning.validate import validate_plan
from codoctopus.runtime.base import Executor, PlanResult
from codoctopus.tools import Tool, ToolRegistry

#: Called as on_event(event, data) at each step/run lifecycle point — see
#: LocalExecutor.run's docstring for the event names and payload shapes. May
#: be sync or async (e.g. a GUI backend awaiting a WebSocket send); a run
#: proceeds even if a handler raises, since progress reporting must never be
#: able to break execution.
EventCallback = Callable[[str, dict[str, Any]], Any]


def _parse_list_output(raw: str) -> list[Any]:
    """Parse output into a list for for_each expansion."""
    raw = raw.strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # Fallback to lines
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _render_instruction(
    instruction: str,
    step_results: dict[str, str],
    *,
    item: Any | None = None,
) -> str:
    """Render {{steps.<key>.result}} and {{item}} / {{item.<field>}} in instruction."""
    rendered = instruction

    for key, result in step_results.items():
        rendered = rendered.replace(f"{{{{steps.{key}.result}}}}", str(result))

    if item is not None:
        rendered = rendered.replace("{{item}}", str(item))
        if isinstance(item, dict):
            for k, v in item.items():
                rendered = rendered.replace(f"{{{{item.{k}}}}}", str(v))

    return rendered


class LocalExecutor(Executor):
    """
    In-process executor for Plan DAGs using asyncio.

    - Resolves dependencies and executes ready steps concurrently.
    - Renders `{{steps.<key>.result}}` and `{{item}}` placeholders before running a step.
    - Dynamically expands `for_each` steps and records results as `<key>[<index>]`.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        workspace: Path,
        artifact_dir: Path | None = None,
        tools: ToolRegistry | list[Tool] | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.provider = provider
        self.workspace = Path(workspace)
        if isinstance(tools, ToolRegistry):
            self._registry = tools
        elif isinstance(tools, list):
            self._registry = ToolRegistry(tools, workspace=self.workspace)
        else:
            self._registry = None
        self._on_event = on_event

    def _get_tools_for_step(self, tool_names: list[str]) -> ToolRegistry | None:
        if not tool_names or not isinstance(self._registry, ToolRegistry):
            return None
        return self._registry.subset(tool_names)

    async def _emit(self, event: str, data: dict[str, Any]) -> None:
        if self._on_event is None:
            return
        try:
            result = self._on_event(event, data)
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass  # a broken progress listener must never take the run down with it

    async def run(self, plan: Plan) -> PlanResult:
        """
        Execute `plan`, reporting progress through `on_event` (if given) as:
        - ("step_started", {"key": ...})
        - ("step_done", {"key": ..., "result": ...})
        - ("step_failed", {"key": ..., "error": ...})
        - ("run_done", {"status": "success", "step_results": {...}})
        - ("run_done", {"status": "failed", "step_results": {...}, "error": ...})

        A for_each step reports one step_started/step_done per expanded item,
        keyed like its result (e.g. "consumer[0]"), not once for the group.
        """
        if not plan.steps:
            return PlanResult(plan=plan, status="failed", error="No steps in plan")

        try:
            validate_plan(plan)
        except Exception as exc:
            return PlanResult(plan=plan, status="failed", error=str(exc))

        step_results: dict[str, str] = {}
        loop = asyncio.get_running_loop()

        def _create_step_future() -> asyncio.Future[None]:
            fut = loop.create_future()
            fut.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
            return fut

        step_futures: dict[str, asyncio.Future[None]] = {
            s.key: _create_step_future() for s in plan.steps
        }

        async def execute_step(s: PlanStep) -> None:
            # 1) Wait for all upstream dependencies
            for dep in s.depends_on:
                await step_futures[dep]

            # 2) for_each dynamic expansion
            if s.for_each is not None:
                source_raw = step_results.get(s.for_each, "")
                items = _parse_list_output(source_raw)

                async def run_item(idx: int, item: Any) -> None:
                    item_key = f"{s.key}[{idx}]"
                    await self._emit("step_started", {"key": item_key})
                    rendered = _render_instruction(s.instruction, step_results, item=item)
                    tools = self._get_tools_for_step(s.tools)
                    agent = Agent(self.provider, system=s.role, tools=tools)
                    res = await agent.run(rendered)
                    step_results[item_key] = res.text
                    await self._emit("step_done", {"key": item_key, "result": res.text})

                await asyncio.gather(*(run_item(i, item) for i, item in enumerate(items)))
            else:
                # 3) Regular single step execution
                await self._emit("step_started", {"key": s.key})
                rendered = _render_instruction(s.instruction, step_results)
                tools = self._get_tools_for_step(s.tools)
                agent = Agent(self.provider, system=s.role, tools=tools)
                res = await agent.run(rendered)
                step_results[s.key] = res.text
                await self._emit("step_done", {"key": s.key, "result": res.text})

        async def step_wrapper(s: PlanStep) -> None:
            try:
                await execute_step(s)
                if not step_futures[s.key].done():
                    step_futures[s.key].set_result(None)
            except Exception as exc:
                if not step_futures[s.key].done():
                    step_futures[s.key].set_exception(exc)
                await self._emit("step_failed", {"key": s.key, "error": str(exc)})
                raise

        tasks = [asyncio.create_task(step_wrapper(s)) for s in plan.steps]

        try:
            await asyncio.gather(*tasks)
            await self._emit("run_done", {"status": "success", "step_results": step_results})
            return PlanResult(plan=plan, status="success", step_results=step_results)
        except Exception as exc:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await self._emit(
                "run_done", {"status": "failed", "step_results": step_results, "error": str(exc)}
            )
            return PlanResult(
                plan=plan,
                status="failed",
                step_results=step_results,
                error=str(exc),
            )