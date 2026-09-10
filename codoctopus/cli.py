# ---------------------------------------------------
# Codoctopus — CLI
#
# `codoctopus run "<goal>"` is the entry point M3 promised: decompose a goal
# into a Plan and execute it, with zero infrastructure required by default
# (LocalExecutor). This file adds no logic of its own beyond argument parsing
# and output formatting — everything it calls is the same public API in
# codoctopus.planning / codoctopus.domains / codoctopus.runtime / codoctopus.tools
# that the test suite already exercises.
# ---------------------------------------------------

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from codoctopus import __version__
from codoctopus.config import get_settings
from codoctopus.domains import get_domain
from codoctopus.llm import ProviderError, get_provider
from codoctopus.planning import Plan, PlanningError, make_plan
from codoctopus.planning.validate import PlanValidationError
from codoctopus.runtime import CoworkifyExecutor, LocalExecutor, PlanResult
from codoctopus.tools.builtin import BUILTIN_TOOLS, build_tool_registry


def _print_plan(plan: Plan) -> None:
    print(f"Plan for: {plan.goal}")
    print(f"Domain: {plan.domain}")
    for step in plan.steps:
        extras = []
        if step.depends_on:
            extras.append(f"depends_on={step.depends_on}")
        if step.for_each:
            extras.append(f"for_each={step.for_each}")
        suffix = f"  ({', '.join(extras)})" if extras else ""
        print(f"  [{step.key}] {step.name}{suffix}")
        print(f"      {step.instruction}")


def _print_result(result: PlanResult) -> None:
    print(f"\nStatus: {result.status}")
    if result.error:
        print(f"Error: {result.error}")
    for key, text in result.step_results.items():
        print(f"\n--- {key} ---")
        print(text)


def _result_to_json(plan: Plan, result: PlanResult | None) -> str:
    payload: dict = {"plan": plan.model_dump()}
    if result is not None:
        payload.update(status=result.status, step_results=result.step_results, error=result.error)
    return json.dumps(payload, indent=2, ensure_ascii=False)


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    domain = get_domain(args.domain) if args.domain else None

    planner_provider = get_provider(args.model or settings.planner_model)
    plan = await make_plan(
        planner_provider, args.goal, domain=domain, available_tools=list(BUILTIN_TOOLS)
    )

    if not args.json:
        _print_plan(plan)

    if args.dry_run:
        if args.json:
            print(_result_to_json(plan, None))
        return 0

    workspace = Path(args.workspace) if args.workspace else settings.workspace
    workspace.mkdir(parents=True, exist_ok=True)

    if args.executor == "coworkify":
        executor = CoworkifyExecutor.from_settings(settings)
    else:
        worker_provider = get_provider(args.worker_model or settings.worker_model)
        executor = LocalExecutor(worker_provider, workspace=workspace, tools=build_tool_registry(workspace))

    result = await executor.run(plan)

    if args.json:
        print(_result_to_json(plan, result))
    else:
        _print_result(result)

    return 0 if result.status == "success" else 1


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "The 'serve' command needs the 'server' extra. Install it with: pip install codoctopus[server]",
            file=sys.stderr,
        )
        return 1

    from codoctopus.server import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codoctopus", description="Decompose a goal into a Plan and run it.")
    parser.add_argument("--version", action="version", version=f"codoctopus {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Plan and execute a goal")
    run_parser.add_argument("goal", help="What you want Codoctopus to accomplish")
    run_parser.add_argument("--domain", default=None, help="Domain pack to plan within, e.g. 'coding'")
    run_parser.add_argument(
        "--model", default=None, help="provider:model used for planning, e.g. anthropic:claude-opus-5"
    )
    run_parser.add_argument("--worker-model", default=None, help="provider:model used to run each step")
    run_parser.add_argument("--workspace", default=None, help="Directory agent tools read and write in")
    run_parser.add_argument(
        "--executor", choices=["local", "coworkify"], default="local", help="Where to run the plan"
    )
    run_parser.add_argument("--dry-run", action="store_true", help="Only plan; don't execute it")
    run_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text")

    serve_parser = subparsers.add_parser("serve", help="Run the GUI backend (HTTP + WebSocket API)")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Interface to bind to")
    serve_parser.add_argument("--port", type=int, default=8420, help="Port to bind to")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        return _serve(args)

    if args.command != "run":
        parser.print_help()
        return 1

    try:
        return asyncio.run(_run(args))
    except (ValueError, ProviderError, PlanningError, PlanValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
