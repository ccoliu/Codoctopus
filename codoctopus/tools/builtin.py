# ---------------------------------------------------
# Codoctopus — Built-in tool set
#
# The tool names a planner may put in a step's "tools" field when the caller
# doesn't supply its own registry. Shared by cli.py and codoctopus.server so
# both offer a plan-executing agent the same tools under the same names.
# ---------------------------------------------------

from __future__ import annotations

from pathlib import Path

from codoctopus.tools.base import Tool
from codoctopus.tools.filesystem import ListFilesTool, ReadFileTool, WriteFileTool
from codoctopus.tools.http import HttpRequestTool
from codoctopus.tools.registry import ToolRegistry
from codoctopus.tools.testing import RunTestsTool

BUILTIN_TOOLS: dict[str, type[Tool]] = {
    "read_file": ReadFileTool,
    "write_file": WriteFileTool,
    "list_files": ListFilesTool,
    "http_request": HttpRequestTool,
    "run_tests": RunTestsTool,
}


def build_tool_registry(workspace: Path) -> ToolRegistry:
    return ToolRegistry([cls() for cls in BUILTIN_TOOLS.values()], workspace=workspace)
