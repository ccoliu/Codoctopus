from __future__ import annotations


from codoctopus.domains.base import Domain, VerifyResult
from codoctopus.tools.testing import RunTestsTool, RunTestsArgs
from codoctopus.tools.base import ToolContext

class CodingDomain(Domain):
    name = "coding"

    roles = {
        "CODE_MASTER": """You are a coding master, skilled at helping others modify their source code to ensure it runs correctly. If you receive only the source code, you will directly make corrections. If you receive both the source code and a list of problems, you will compare each problem to the source code and analyze whether these issues may occur. If they are likely to occur, you will then proceed to further revise the code.""",
        "REVERSE_DESCRIBER": """You are a reverse engineer, capable of understanding the source code and describing its functionality or what this code is doing in sentences.""",
        "ANALYST": """You are a program issue analyst, adept at identifying potential problems by observing code. If you notice any segment of code that might encounter issues during runtime, please print out the concerns in a bullet-point format. If you find no issues, simply print out the phrase 'No issues'.""",
        "PRESENTER": """You are the last person who is responsible for presenting the program (complete soruce code) by a specific format."""
        }
    
    default_tools = ["read_file", "write_file", "run_tests"]

    planner_hint = "Include a step that runs the test suite (tool: run_tests) after writing code, so failures can be caught and fixed before the plan finishes."
    
    async def verify(self, step_results: dict[str, str], ctx: ToolContext) -> VerifyResult:
        
        run_tests_tool = RunTestsTool()

        result = await run_tests_tool.run(RunTestsArgs(), ctx)

        # pytest exit code 5 = no tests collected. Nothing to check isn't the
        # same as a failing check (e.g. a plain "write hello.py" goal).
        if result.startswith("FAILED (exit code 5)"):
            return VerifyResult(passed=True, skipped=True, detail="Skipped: no tests found in the workspace.")

        passed = result.startswith("PASSED")

        #if failed, can attach the test result to the verifier's feedback
        detail = result
        if not passed and step_results:
            last_step_key = list(step_results.keys())[-1]
            detail += f"\n\n[Last step '{last_step_key}' output]:\n{step_results[last_step_key]}"

        return VerifyResult(passed=passed, detail=detail)
            
        

