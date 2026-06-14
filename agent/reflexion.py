from typing import Callable, List, Optional


class ReflexionAgent:
    def __init__(self, model_fn: Callable, tools: Optional[List] = None,
                 max_reflections: int = 3, verbose: bool = True):
        self.model_fn = model_fn
        self.tools = {t.name: t for t in (tools or [])}
        self.max_reflections = max_reflections
        self.verbose = verbose

    def attempt(self, query: str, previous_attempts: List[dict] = None) -> str:
        context = ""
        if previous_attempts:
            context = "\n".join(
                f"Attempt {a['attempt']}: {a['response']}\nReflection: {a['reflection']}"
                for a in previous_attempts
            ) + "\n\n"
        prompt = f"{context}Question: {query}\nAnswer:"
        return self.model_fn(prompt)

    def reflect(self, query: str, response: str) -> str:
        prompt = (
            f"Reflect on your answer to improve it.\n"
            f"Question: {query}\n"
            f"Your Answer: {response}\n"
            f"What could be improved?"
        )
        return self.model_fn(prompt)

    def run(self, query: str) -> str:
        history = []
        response = None
        for i in range(self.max_reflections):
            response = self.attempt(query, history)
            if i < self.max_reflections - 1:
                reflection = self.reflect(query, response)
                history.append({"attempt": i + 1, "response": response, "reflection": reflection})
                if self.verbose:
                    print(f"Reflection {i+1}: {reflection}")
        return response
