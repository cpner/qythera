from typing import Callable, List, Optional
import json


class PlanExecuteAgent:
    def __init__(self, model_fn: Callable, tools: Optional[List] = None,
                 max_iterations: int = 10, verbose: bool = True):
        self.model_fn = model_fn
        self.tools = {t.name: t for t in (tools or [])}
        self.max_iterations = max_iterations
        self.verbose = verbose

    def create_plan(self, query: str) -> List[str]:
        prompt = (
            f"Create a step-by-step plan to answer: {query}\n"
            f"Return a JSON array of step strings."
        )
        response = self.model_fn(prompt)
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        return [f"Answer: {query}"]

    def execute_plan(self, plan: List[str]) -> str:
        results = []
        for i, step in enumerate(plan):
            if self.verbose:
                print(f"Step {i+1}: {step}")
            result = self.model_fn(f"Execute this step: {step}")
            results.append({"step": step, "result": result})
        return "\n\n".join(f"Step {r['step']}: {r['result']}" for r in results)

    def run(self, query: str) -> str:
        plan = self.create_plan(query)
        if self.verbose:
            print(f"Plan: {json.dumps(plan, indent=2)}")
        return self.execute_plan(plan)
