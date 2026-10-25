# ---------------------------------------------------
# Codoctopus — CodingDomain tests
#
# verify() actually runs pytest, so this needs real fixture test files on
# disk rather than a mock — same reasoning as tests/test_tools_testing.py.
# The other half is proving CodingDomain is a real, working Domain: it
# resolves through the registry under the name planner.py records on a Plan,
# and its role names are the ones make_plan() will expand.
# ---------------------------------------------------

from __future__ import annotations

import pytest

from codoctopus.domains import get_domain
from codoctopus.domains.coding import CodingDomain
from codoctopus.tools.base import ToolContext


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


@pytest.fixture
def ctx(workspace):
    return ToolContext(workspace=workspace)


# --- registration and shape ------------------------------------------


def test_resolves_through_the_registry_under_its_own_name():
    domain = get_domain("coding")

    assert isinstance(domain, CodingDomain)
    assert domain.name == "coding", "must match the registry key, or Plan.domain and get_domain() disagree"


def test_declares_the_v1_roles():
    domain = CodingDomain()

    assert set(domain.roles) == {"CODE_MASTER", "REVERSE_DESCRIBER", "ANALYST", "PRESENTER"}
    assert all(isinstance(prompt, str) and prompt.strip() for prompt in domain.roles.values())


def test_declares_its_default_tools():
    domain = CodingDomain()

    assert domain.default_tools == ["read_file", "write_file", "run_tests"]


def test_planner_hint_points_at_running_the_test_suite():
    domain = CodingDomain()

    assert "run_tests" in domain.planner_hint


# --- verify() actually runs the test suite -----------------------------


async def test_verify_passes_when_the_suite_passes(workspace, ctx):
    (workspace / "test_ok.py").write_text("def test_it():\n    assert 1 + 1 == 2\n")

    result = await CodingDomain().verify(step_results={"write": "wrote the code"}, ctx=ctx)

    assert result.passed is True
    assert "PASSED" in result.detail


async def test_verify_fails_when_the_suite_fails_and_reports_why(workspace, ctx):
    (workspace / "test_bad.py").write_text("def test_it():\n    assert 1 + 1 == 3, 'math is broken'\n")

    result = await CodingDomain().verify(step_results={"write": "wrote the code"}, ctx=ctx)

    assert result.passed is False
    assert "math is broken" in result.detail


async def test_verify_attaches_the_last_step_output_on_failure(workspace, ctx):
    (workspace / "test_bad.py").write_text("def test_it():\n    assert False\n")

    result = await CodingDomain().verify(
        step_results={"write": "here is the code I wrote"}, ctx=ctx
    )

    assert "here is the code I wrote" in result.detail


async def test_verify_is_skipped_not_failed_when_there_are_no_tests(workspace, ctx):
    (workspace / "hello.py").write_text('print("hi")\n')

    result = await CodingDomain().verify(step_results={"write": "wrote hello.py"}, ctx=ctx)

    assert result.status == "skipped"
    assert result.passed is True


async def test_verify_does_not_crash_with_no_step_results(workspace, ctx):
    (workspace / "test_ok.py").write_text("def test_it():\n    assert True\n")

    result = await CodingDomain().verify(step_results={}, ctx=ctx)

    assert result.passed is True
