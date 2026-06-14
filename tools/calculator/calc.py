import math

class CalculatorTool:
    name = "calculator"
    description = "Perform mathematical calculations"

    SAFE = {
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow,
        "pi": math.pi, "e": math.e, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "log2": math.log2, "log10": math.log10,
        "floor": math.floor, "ceil": math.ceil, "factorial": math.factorial,
        "radians": math.radians, "degrees": math.degrees, "gcd": math.gcd,
    }

    def execute(self, expression="", **kwargs):
        try:
            result = eval(expression, {"__builtins__": {}}, self.SAFE)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
