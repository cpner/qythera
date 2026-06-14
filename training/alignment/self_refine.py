from typing import List, Dict, Callable


class SelfRefine:
    def __init__(self, model_fn: Callable, max_iterations: int = 3, temperature: float = 0.7):
        self.model_fn = model_fn
        self.max_iterations = max_iterations
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        return self.model_fn(prompt, temperature=self.temperature)

    def reflect(self, prompt: str, response: str) -> str:
        reflect_prompt = (
            f"Review your response to the following:\n"
            f"Question: {prompt}\n"
            f"Your Response: {response}\n\n"
            f"Identify any errors, inaccuracies, or areas for improvement:"
        )
        return self.generate(reflect_prompt)

    def refine(self, prompt: str, response: str, reflection: str) -> str:
        refine_prompt = (
            f"Question: {prompt}\n"
            f"Previous Response: {response}\n"
            f"Self-Reflection: {reflection}\n\n"
            f"Provide an improved response:"
        )
        return self.generate(refine_prompt)

    def run(self, prompt: str) -> Dict:
        history = []
        response = self.generate(prompt)
        history.append({"iteration": 0, "response": response, "reflection": None})

        for i in range(self.max_iterations):
            reflection = self.reflect(prompt, response)
            new_response = self.refine(prompt, response, reflection)
            history.append({"iteration": i + 1, "response": new_response, "reflection": reflection})
            if new_response == response:
                break
            response = new_response

        return {"final_response": response, "history": history, "iterations": len(history) - 1}
