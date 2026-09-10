# ---------------------------------------------------
# Codoctopus — Research domain
#
# The second domain pack (after coding), added to actually prove domains are
# pluggable rather than assert it once with a single example. No new tools:
# http_request (already SSRF-guarded) and write_file are enough to gather
# public sources and produce a report — a dedicated search tool is a natural
# follow-up but isn't required to demonstrate a working, distinct domain.
# ---------------------------------------------------

from __future__ import annotations

from codoctopus.domains.base import Domain, VerifyResult
from codoctopus.tools.base import ToolContext

#: Below this many total bytes written to the workspace, a "report" is
#: almost certainly a stub or a refusal, not real research output.
_MIN_REPORT_BYTES = 50


class ResearchDomain(Domain):
    name = "research"

    roles = {
        "RESEARCHER": (
            "You are a research assistant. You gather information relevant to the "
            "given topic, using http_request to fetch specific public pages (official "
            "documentation, Wikipedia, standards bodies, etc.) when you need a source "
            "beyond your own knowledge. You report what you found plainly, noting which "
            "claims came from a fetched source and which came from general knowledge."
        ),
        "SYNTHESIZER": (
            "You are a technical writer. Given research notes gathered by others, you "
            "organize them into a single clear, well-structured report — headings, a "
            "short summary up top, no duplicated points — and write it to a file."
        ),
        "REVIEWER": (
            "You review a research report for gaps, unsupported claims, and unclear "
            "writing. Report what you find as a short bullet list; if the report already "
            "looks solid, say so plainly instead of inventing issues."
        ),
    }

    default_tools = ["http_request", "write_file", "read_file"]

    planner_hint = (
        "A research goal typically needs: one or more RESEARCHER steps that gather "
        "information (http_request against specific public URLs, not guessed search "
        "queries — a URL the model already knows, like a Wikipedia or official docs "
        "page), a SYNTHESIZER step that writes the findings to a report file "
        "(tool: write_file), and usually a REVIEWER step over that file afterward."
    )

    async def verify(self, step_results: dict[str, str], ctx: ToolContext) -> VerifyResult:
        written = [p for p in ctx.workspace.rglob("*") if p.is_file()]
        if not written:
            return VerifyResult(passed=False, detail="No file was written to the workspace.")

        total_bytes = sum(p.stat().st_size for p in written)
        if total_bytes < _MIN_REPORT_BYTES:
            return VerifyResult(
                passed=False,
                detail=f"Only {total_bytes} byte(s) written across {len(written)} file(s) — "
                "too little to be a real report.",
            )

        names = ", ".join(sorted(p.name for p in written))
        return VerifyResult(passed=True, detail=f"Wrote {len(written)} file(s) ({names}), {total_bytes} bytes total.")
