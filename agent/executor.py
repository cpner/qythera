from typing import Dict, Any
import subprocess
import sys


class ToolExecutor:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def execute(self, tool, **kwargs) -> str:
        try:
            result = tool.execute(**kwargs)
            return str(result)
        except Exception as e:
            return f"Tool execution error: {str(e)}"

    def run_python(self, code: str) -> str:
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=self.timeout,
            )
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr
            return output or "No output"
        except subprocess.TimeoutExpired:
            return "Execution timed out"
        except Exception as e:
            return f"Error: {str(e)}"
