import subprocess
import sys
from agent.tools.tool_registry import Tool


class PythonREPLTool(Tool):
    name = "python_repl"
    description = "Execute Python code and return the output"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def execute(self, code: str = "", **kwargs) -> str:
        if not code:
            return "No code provided"
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=self.timeout,
            )
            output = result.stdout
            if result.stderr:
                output += "\nSTDERR: " + result.stderr
            return output.strip() or "No output"
        except subprocess.TimeoutExpired:
            return "Code execution timed out"
        except Exception as e:
            return f"Error: {str(e)}"
