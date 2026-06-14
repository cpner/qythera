from typing import List, Dict, Optional


PRINCIPLES = [
    "Choose the response that is most helpful and harmless.",
    "Choose the response that is most accurate and truthful.",
    "Choose the response that is most respectful and considerate.",
    "Choose the response that avoids harmful, unethical, or dangerous content.",
    "Choose the response that is most informative and educational.",
    "Choose the response that follows the golden rule: treat others as you want to be treated.",
    "Choose the response that is most transparent about its limitations.",
]


class ConstitutionalAI:
    def __init__(self, model, principles=None):
        self.model = model
        self.principles = principles or PRINCIPLES

    def critique(self, prompt: str, response: str, principle: str) -> str:
        critique_prompt = (
            f"Given the following conversation:\n"
            f"Human: {prompt}\n"
            f"Assistant: {response}\n\n"
            f"Consider this principle: {principle}\n"
            f"Identify ways the response could be improved:"
        )
        return critique_prompt

    def revise(self, prompt: str, response: str, critique: str) -> str:
        revise_prompt = (
            f"Human: {prompt}\n"
            f"Assistant: {response}\n"
            f"Critique: {critique}\n"
            f"Please revise the assistant's response to address the critique:"
        )
        return revise_prompt

    def self_critique_and_revise(self, prompt: str, response: str, max_iterations: int = 3) -> str:
        current_response = response
        for i in range(max_iterations):
            principle = self.principles[i % len(self.principles)]
            critique = self.critique(prompt, current_response, principle)
            revise_prompt = self.revise(prompt, current_response, critique)
            current_response = revise_prompt
        return current_response

    def generate_training_data(self, prompts: List[str], max_iterations: int = 2) -> List[Dict]:
        training_data = []
        for prompt in prompts:
            initial_response = f"Response to: {prompt}"
            revised = self.self_critique_and_revise(prompt, initial_response, max_iterations)
            training_data.append({
                "prompt": prompt,
                "initial_response": initial_response,
                "revised_response": revised,
            })
        return training_data
