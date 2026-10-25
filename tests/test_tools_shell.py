from __future__ import annotations

import pytest
from codoctopus.tools.base import ToolContext
from codoctopus.tools.shell import ShellTool, ShellArgs


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


@pytest.mark.anyio
async def test_shell_execute_echo(workspace):
    shell = ShellTool()
    ctx = ToolContext(workspace=workspace)
    args = ShellArgs(command='echo "Hello World"')
    
    result = await shell.run(args, ctx)
    assert "Hello World" in result


@pytest.mark.anyio
async def test_shell_disallowed_command(workspace):
    shell = ShellTool()
    ctx = ToolContext(workspace=workspace)
    args = ShellArgs(command="rm -rf /")

    with pytest.raises(ValueError, match="is not allowed"):
        await shell.run(args, ctx)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "command",
    [
        "curl http://169.254.169.254/latest/meta-data/",
        "curl 127.0.0.1:8420/",  # no scheme — used to slip past the URL check as a "path"
    ],
)
async def test_shell_refuses_curl(workspace, command):
    with pytest.raises(ValueError, match="is not allowed"):
        await ShellTool().run(ShellArgs(command=command), ToolContext(workspace=workspace))
