# ---------------------------------------------------
# Codoctopus — ResearchDomain tests
#
# verify() checks the real filesystem (no mock) — same reasoning as
# test_domains_coding.py: it's plain, fast, and there's nothing worth
# mocking about "did any file actually get written".
# ---------------------------------------------------

from __future__ import annotations

import pytest

from codoctopus.domains import get_domain
from codoctopus.domains.research import ResearchDomain
from codoctopus.tools.base import ToolContext


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


@pytest.fixture
def ctx(workspace):
    return ToolContext(workspace=workspace)


# --- registration and shape ------------------------------------------


def test_resolves_through_the_registry_under_its_own_name():
    domain = get_domain("research")

    assert isinstance(domain, ResearchDomain)
    assert domain.name == "research", "must match the registry key, or Plan.domain and get_domain() disagree"


def test_is_a_second_distinct_domain_from_coding():
    coding = get_domain("coding")
    research = get_domain("research")

    assert research.name != coding.name
    assert set(research.roles) != set(coding.roles)


def test_declares_its_roles():
    domain = ResearchDomain()

    assert set(domain.roles) == {"RESEARCHER", "SYNTHESIZER", "REVIEWER"}
    assert all(isinstance(prompt, str) and prompt.strip() for prompt in domain.roles.values())


def test_declares_its_default_tools():
    domain = ResearchDomain()

    assert domain.default_tools == ["http_request", "write_file", "read_file"]


def test_planner_hint_points_at_writing_a_report():
    domain = ResearchDomain()

    assert "write_file" in domain.planner_hint


# --- verify() checks a report actually got written ----------------------


async def test_verify_fails_when_nothing_was_written(ctx):
    result = await ResearchDomain().verify(step_results={"research": "found some stuff"}, ctx=ctx)

    assert result.passed is False
    assert "No file was written" in result.detail


async def test_verify_fails_when_the_report_is_too_short(workspace, ctx):
    (workspace / "report.md").write_text("TODO")

    result = await ResearchDomain().verify(step_results={}, ctx=ctx)

    assert result.passed is False
    assert "too little" in result.detail


async def test_verify_passes_with_a_real_report(workspace, ctx):
    (workspace / "report.md").write_text("# Findings\n\n" + "This is a real research report. " * 5)

    result = await ResearchDomain().verify(step_results={}, ctx=ctx)

    assert result.passed is True
    assert "report.md" in result.detail


async def test_verify_counts_bytes_across_multiple_files(workspace, ctx):
    (workspace / "notes.md").write_text("x" * 30)
    (workspace / "report.md").write_text("y" * 30)

    result = await ResearchDomain().verify(step_results={}, ctx=ctx)

    assert result.passed is True
    assert "notes.md" in result.detail
    assert "report.md" in result.detail
