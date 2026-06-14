import subprocess, sys, tempfile, os

class CodeInterpreterTool:
    name = "code_interpreter"
    description = "Execute Python code in a sandboxed environment"

    def __init__(self, timeout=30, max_output=10000):
        self.timeout = timeout
        self.max_output = max_output

    def execute(self, code="", language="python", **kwargs):
        if language != "python":
            return f"Only Python is supported. Got: {language}"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            try:
                result = subprocess.run([sys.executable, f.name],
                    capture_output=True, text=True, timeout=self.timeout,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
                output = result.stdout
                if result.stderr: output += "\nSTDERR: " + result.stderr
                return output[:self.max_output] or "No output"
            except subprocess.TimeoutExpired:
                return "Execution timed out"
            finally:
                os.unlink(f.name)
