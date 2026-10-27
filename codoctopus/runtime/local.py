# ---------------------------------------------------
# Codoctopus — Local Executor
#
# Runs a Plan directly in the current process using asyncio.
# Handles DAG scheduling, step dependency resolution, instruction templating,
# for_each expansion, and reduce_of fan-in with zero external infrastructure.
# ---------------------------------------------------

from __future__ import annotations

import asyncio
import inspect
import json
import re
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


class _SkippedError(RuntimeError):
    """A step's dependency failed, so it never ran — distinct from the step's own work failing."""


_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_ITEMS_PLACEHOLDER = re.compile(r"\{\{\s*items\s*\}\}")


def _json_list(text: str) -> list[Any] | None:
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, list) else None


def _parse_list_output(raw: str) -> list[Any]:
    """
    Turn a for_each source step's free-form output into its list of items.

    Models routinely wrap a JSON list in a code fence or put a sentence before
    it, so those are unwrapped first. The JSON array has to start at the
    beginning of a line, though: a plain one-item-per-line list whose items
    happen to contain brackets (`test_x[1]`, `arr[0].py`) must still split by
    line, not be misread as the bracketed fragment.
    """
    raw = raw.strip()
    if not raw:
        return []

    fenced = _FENCE.search(raw)
    if fenced:
        raw = fenced.group(1).strip()

    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    if (data := _json_list(raw)) is not None:
        return data

    # A lead-in or trailing sentence around a JSON array: take the span from
    # the first line opening a "[" to the last line closing a "]".
    starts = [i for i, line in enumerate(lines) if line.startswith("[")]
    ends = [i for i, line in enumerate(lines) if line.endswith("]")]
    if starts and ends and ends[-1] >= starts[0]:
        if (data := _json_list("\n".join(lines[starts[0] : ends[-1] + 1]))) is not None:
            return data

    return lines


def _render_instruction(
    instruction: str,
    step_results: dict[str, str],
    *,
    item: Any | None = None,
    items: list[str] | None = None,
) -> str:
    """Render {{steps.<key>.result}}, {{item}} / {{item.<field>}} and {{items}} in instruction."""
    rendered = instruction

    for key, result in step_results.items():
        rendered = rendered.replace(f"{{{{steps.{key}.result}}}}", str(result))

    if item is not None:
        rendered = rendered.replace("{{item}}", str(item))
        if isinstance(item, dict):
            for k, v in item.items():
                rendered = rendered.replace(f"{{{{item.{k}}}}}", str(v))

    if items is not None:
        # JSON, not str(), so the agent sees a real list — same as Coworkify's
        # render_reduce_payloads.
        items_json = json.dumps(items, ensure_ascii=False)
        rendered = _ITEMS_PLACEHOLDER.sub(lambda _: items_json, rendered)

    return rendered


class LocalExecutor(Executor):
    """
    In-process executor for Plan DAGs using asyncio.

    - Resolves dependencies and executes ready steps concurrently.
    - Renders `{{steps.<key>.result}}` and `{{item}}` placeholders before running a step.
    - Dynamically expands `for_each` steps and records results as `<key>[<index>]`.
    - Runs a `reduce_of` step once its whole for_each group is done, with `{{items}}`.
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
        # for_each step key -> how many items it expanded into, so a reduce
        # step knows which "<key>[<i>]" results make up the group.
        expanded_counts: dict[str, int] = {}
        loop = asyncio.get_running_loop()

        def _create_step_future() -> asyncio.Future[None]:
            fut = loop.create_future()
            fut.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
            return fut

        step_futures: dict[str, asyncio.Future[None]] = {
            s.key: _create_step_future() for s in plan.steps
        }

        async def execute_step(s: PlanStep) -> None:
            # 1) Wait for all upstream dependencies, including the implied
            # ones: a for_each step needs its source's list, and a reduce step
            # needs its whole group. A dependency that failed raises here too
            # (awaiting a future with an exception set) — wrap it so the
            # resulting step_failed event reads as "skipped because X failed",
            # not as if this step's own work threw that error.
            implied = [k for k in (s.for_each, s.reduce_of) if k is not None]
            for dep in [*s.depends_on, *implied]:
                try:
                    await step_futures[dep]
                except Exception as exc:
                    raise _SkippedError(f"skipped: upstream step '{dep}' failed: {exc}") from exc

            # 2) for_each dynamic expansion
            if s.for_each is not None:
                source_raw = step_results.get(s.for_each, "")
                items = _parse_list_output(source_raw)
                expanded_counts[s.key] = len(items)

                async def run_item(idx: int, item: Any) -> None:
                    item_key = f"{s.key}[{idx}]"
                    await self._emit("step_started", {"key": item_key})
                    try:
                        rendered = _render_instruction(s.instruction, step_results, item=item)
                        tools = self._get_tools_for_step(s.tools)
                        agent = Agent(self.provider, system=s.role, tools=tools)
                        res = await agent.run(rendered)
                        step_results[item_key] = res.text
                        await self._emit("step_done", {"key": item_key, "result": res.text})
                    except Exception as exc:
                        await self._emit("step_failed", {"key": item_key, "error": str(exc)})
                        raise

                # return_exceptions=True: one item failing must not orphan its
                # siblings — they keep running and their results/events still
                # land, instead of finishing invisibly after the run already
                # reported done (see run()'s matching use of it, below).
                item_results = await asyncio.gather(
                    *(run_item(i, item) for i, item in enumerate(items)), return_exceptions=True
                )
                item_errors = [r for r in item_results if isinstance(r, BaseException)]
                if item_errors:
                    raise item_errors[0]
            else:
                # 3) Regular single step execution — a reduce step is one too,
                # just with its group's results passed in index order.
                await self._emit("step_started", {"key": s.key})
                group = None
                if s.reduce_of is not None:
                    count = expanded_counts.get(s.reduce_of, 0)
                    group = [step_results[f"{s.reduce_of}[{i}]"] for i in range(count)]
                rendered = _render_instruction(s.instruction, step_results, items=group)
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

        # return_exceptions=True: one step failing must not cancel unrelated
        # steps still in flight — only steps that actually depend on the
        # failure (directly or transitively) raise, via the _SkippedError
        # path above. Everything else is left to finish and its result kept,
        # matching Coworkify's executor (which only cancels real downstream
        # dependents, not the whole workflow).
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]

        if not errors:
            await self._emit("run_done", {"status": "success", "step_results": step_results})
            return PlanResult(plan=plan, status="success", step_results=step_results)

        # Prefer a genuine failure over a cascaded "skipped" one when
        # reporting the run's headline error — that's the one worth surfacing.
        root_errors = [e for e in errors if not isinstance(e, _SkippedError)]
        primary = root_errors[0] if root_errors else errors[0]
        await self._emit(
            "run_done", {"status": "failed", "step_results": step_results, "error": str(primary)}
        )
        return PlanResult(plan=plan, status="failed", step_results=step_results, error=str(primary))