import math
from agent.tools.tool_registry import Tool


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate mathematical expressions safely"

    SAFE_NAMES = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "int": int, "float": float,
        "pi": math.pi, "e": math.e, "sqrt": math.sqrt,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log2": math.log2, "log10": math.log10,
        "floor": math.floor, "ceil": math.ceil, "factorial": math.factorial,
    }

    def execute(self, expression: str = "", **kwargs) -> str:
        if not expression:
            return "No expression provided"
        try:
            result = eval(expression, {"__builtins__": {}}, self.SAFE_NAMES)
            return str(result)
        except Exception as e:
            return f"Calculation error: {str(e)}"
