from __future__ import annotations

import asyncio
import shlex
from pydantic import BaseModel, Field

from codoctopus.tools.base import Tool, ToolContext
from codoctopus.tools.filesystem import resolve_in_workspace
from codoctopus.tools.http import _validate_public_url

# 允許的指令白名單（允許 List）
ALLOWED_COMMANDS = ["ls", "dir", "cat", "echo", "git", "curl"]
ALLOWED_SUBCOMMANDS = {
    "git": ["diff", "log", "status"]
}


class ShellArgs(BaseModel):
    command: str = Field(..., description="Command to execute")


class ShellTool(Tool):
    """
    Run shell command inside workspace.
    - Files are restricted to the workspace.
    - Uses an allowlist and timeout for safety.
    """

    name = "shell"
    description = "Execute shell command inside workspace"
    args_schema = ShellArgs

    async def run(self, args: ShellArgs, ctx: ToolContext) -> str:
        cmd = args.command.strip()

        # 1) 拆解指令
        try:
            parts = shlex.split(cmd)
        except ValueError:
            parts = cmd.split()

        if not parts:
            raise ValueError("Empty command")
        command = parts[0]
        command_sub = parts[1] if len(parts) > 1 else ""
        args_list = parts[1:]

        # 2) 安全檢查：白名單 (Allowlist)
        if command in ALLOWED_SUBCOMMANDS:
            if command_sub not in ALLOWED_SUBCOMMANDS[command]:
                raise ValueError(f"Command '{command} {command_sub}' is not allowed. Use only: {', '.join(ALLOWED_SUBCOMMANDS[command])}")
        if command not in ALLOWED_COMMANDS:
            raise ValueError(f"Command '{command}' is not allowed. Use only: {', '.join(ALLOWED_COMMANDS)}")

        # 3) 安全檢查：防止目錄穿越 / 任意路徑寫入 (Path Traversal)
        workspace = ctx.workspace
        for arg in args_list:
            if "://" in arg: #Allow URL
                try:
                    _validate_public_url(arg)
                except ValueError:
                    raise ValueError(f"Invalid URL: {arg}")
            try:
                resolve_in_workspace(workspace, arg)
            except ValueError:
                raise ValueError(f"Path traversal attempt detected: {arg}")

        # 4) 執行指令 (使用 asyncio.create_subprocess_exec)
        try:
            proc = await asyncio.create_subprocess_exec(
                command, *args_list,
                cwd=str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_data, stderr_data = await asyncio.wait_for(proc.communicate(), timeout=10)
            
            stdout_str = stdout_data.decode("utf-8", errors="replace")
            stderr_str = stderr_data.decode("utf-8", errors="replace")

            output_parts = []
            if stdout_str:
                output_parts.append(stdout_str)
            if stderr_str:
                output_parts.append(f"STDERR:\n{stderr_str}")
            if proc.returncode != 0:
                output_parts.append(f"Exit code: {proc.returncode}")

            return "\n".join(output_parts) if output_parts else "(no output)"

        except (asyncio.TimeoutError, TimeoutError):
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            raise ValueError("Command execution timeout")
        except Exception as e:
            raise ValueError(f"Execution error: {str(e)}")