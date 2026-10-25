# ---------------------------------------------------
# Codoctopus — CLI tests
#
# A scripted provider stands in for the model, same pattern as every other
# test file in this suite. Every invocation passes --model/--worker-model
# explicitly so these never touch codoctopus.config's env-derived defaults.
# ---------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from codoctopus.cli import build_parser, main
from codoctopus.config import reset_settings
from codoctopus.domains import Domain, VerifyResult, register_domain
from codoctopus.llm import Completion, Provider, StopReason, register_provider
from codoctopus.planning import Plan


class ScriptedProvider(Provider):
    """Plans with a fixed one-step Plan, then answers that step with fixed text."""

    name = "scripted"
    default_model = "scripted-1"

    def __init__(self, model: str | None = None, **options: Any) -> None:
        super().__init__(model, **options)
        self.calls: list[dict[str, Any]] = []

    @property
    def supports_structured_output(self) -> bool:
        return True

    async def _complete(self, messages, *, system, tools, output_schema, max_tokens, **kwargs) -> Completion:
        self.calls.append({"system": system, "tools": tools, "output_schema": output_schema})

        if output_schema is Plan:
            plan = Plan.model_validate(
                {
                    "goal": messages[0].content.split("Goal: ")[1].split("\n")[0],
                    "steps": [{"key": "write", "name": "write", "role": "coder", "instruction": "do the thing"}],
                }
            )
            return Completion(text=plan.model_dump_json(), model=self.model, parsed=plan)

        return Completion(text="step done", model=self.model, stop_reason=StopReason.END_TURN)


@pytest.fixture(autouse=True)
def _register_scripted():
    register_provider("scripted", lambda model=None, **kw: ScriptedProvider(model, **kw))


# --- argument parsing ----------------------------------------------------


def test_run_requires_a_goal():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])


def test_run_parses_a_goal_with_defaults():
    parser = build_parser()
    args = parser.parse_args(["run", "ship the feature"])

    assert args.goal == "ship the feature"
    assert args.domain is None
    assert args.executor == "local"
    assert args.dry_run is False
    assert args.json is False


def test_run_parses_all_overrides():
    parser = build_parser()
    args = parser.parse_args(
        [
            "run", "ship it",
            "--domain", "coding",
            "--model", "anthropic:claude-opus-5",
            "--worker-model", "anthropic:claude-sonnet-5",
            "--workspace", "/tmp/ws",
            "--executor", "coworkify",
            "--dry-run",
            "--json",
        ]
    )

    assert args.domain == "coding"
    assert args.model == "anthropic:claude-opus-5"
    assert args.worker_model == "anthropic:claude-sonnet-5"
    assert args.workspace == "/tmp/ws"
    assert args.executor == "coworkify"
    assert args.dry_run is True
    assert args.json is True


def test_no_subcommand_prints_help_and_exits_nonzero(capsys):
    exit_code = main([])

    assert exit_code == 1
    assert "usage" in capsys.readouterr().out.lower()


# --- run: dry-run (plan only, no execution) -------------------------------


def test_dry_run_plans_but_does_not_execute(capsys, tmp_path):
    exit_code = main(
        ["run", "ship the feature", "--model", "scripted:x", "--dry-run", "--workspace", str(tmp_path)]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Plan for: ship the feature" in out
    assert "[write]" in out
    assert "Status:" not in out, "dry-run must not execute the plan"


def test_dry_run_json_output_is_valid_and_has_no_result_fields(capsys, tmp_path):
    exit_code = main(["run", "ship it", "--model", "scripted:x", "--dry-run", "--json"])

    assert exit_code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["plan"]["goal"] == "ship it"
    assert "status" not in data


# --- run: full execution via LocalExecutor --------------------------------


def test_run_executes_the_plan_and_reports_success(capsys, tmp_path):
    exit_code = main(
        [
            "run", "ship the feature",
            "--model", "scripted:x",
            "--worker-model", "scripted:x",
            "--workspace", str(tmp_path),
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Status: success" in out
    assert "--- write ---" in out
    assert "step done" in out


def test_run_json_output_includes_the_step_results(capsys, tmp_path):
    exit_code = main(
        [
            "run", "ship it",
            "--model", "scripted:x",
            "--worker-model", "scripted:x",
            "--workspace", str(tmp_path),
            "--json",
        ]
    )

    assert exit_code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "success"
    assert data["step_results"]["write"] == "step done"


# --- domain verification (issue #11) -----------------------------------------


class _CheckDomain(Domain):
    name = "cli-check"
    verdict = VerifyResult(passed=True)

    async def verify(self, step_results, ctx):
        return self.verdict


def _run_with_check(verdict: VerifyResult, *extra: str) -> int:
    register_domain("cli-check", lambda: _CheckDomain())
    _CheckDomain.verdict = verdict
    return main(["run", "ship it", "--model", "scripted:x", "--worker-model", "scripted:x", "--domain", "cli-check", *extra])


def test_a_failing_domain_check_fails_the_cli_run(capsys, tmp_path):
    exit_code = _run_with_check(VerifyResult(passed=False, detail="2 tests failed"), "--workspace", str(tmp_path))

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "Status: success" in out
    assert "Verification: FAILED" in out
    assert "2 tests failed" in out


def test_a_passing_or_skipped_check_keeps_exit_code_zero(capsys, tmp_path):
    assert _run_with_check(VerifyResult(passed=True), "--workspace", str(tmp_path)) == 0
    assert _run_with_check(VerifyResult(passed=True, skipped=True), "--workspace", str(tmp_path)) == 0
    assert "Verification: SKIPPED" in capsys.readouterr().out


def test_json_output_includes_the_verification(capsys, tmp_path):
    _run_with_check(VerifyResult(passed=False, detail="nope"), "--workspace", str(tmp_path), "--json")

    data = json.loads(capsys.readouterr().out)
    assert data["verification"] == {"status": "failed", "detail": "nope"}


def test_each_cli_run_gets_its_own_workspace_by_default(capsys, tmp_path, monkeypatch):
    # The shared root holds every earlier run's files; verification (pytest)
    # over the root would pick up all of them.
    monkeypatch.setenv("CODOCTOPUS_WORKSPACE", str(tmp_path))
    reset_settings()
    try:
        _run_with_check(VerifyResult(passed=True), "--json")
        first = json.loads(capsys.readouterr().out)["workspace"]
        _run_with_check(VerifyResult(passed=True), "--json")
        second = json.loads(capsys.readouterr().out)["workspace"]
    finally:
        reset_settings()

    assert first != second
    assert Path(first).parent == tmp_path


# --- error handling: known failures print a clean message, not a traceback -


def test_unknown_domain_fails_cleanly(capsys, tmp_path):
    exit_code = main(
        ["run", "ship it", "--model", "scripted:x", "--domain", "no-such-domain", "--workspace", str(tmp_path)]
    )

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "Unknown domain" in err


def test_coworkify_executor_without_configuration_fails_cleanly(capsys, tmp_path):
    exit_code = main(
        [
            "run", "ship it",
            "--model", "scripted:x",
            "--executor", "coworkify",
            "--workspace", str(tmp_path),
        ]
    )

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "CODOCTOPUS_COWORKIFY_URL" in err
