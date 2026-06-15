"""Reasoning systems for Qythera. Pure Python + NumPy."""
import math
import re
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# ChainOfThought
# ---------------------------------------------------------------------------

class ChainOfThought:
    """Generate reasoning in <think>reasoning</think> format."""

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self.step_templates: List[str] = [
            "Let me break this down step by step.",
            "First, I need to identify the key components.",
            "Next, I'll analyze the relationships between them.",
            "Then, I'll apply the relevant principles.",
            "Finally, I'll synthesize the conclusion.",
        ]

    def reason(self, problem: str, generate_fn: Optional[Callable] = None) -> str:
        if generate_fn is not None:
            steps = []
            prompt = problem
            for i in range(self.max_steps):
                step = generate_fn(prompt)
                steps.append(step)
                if "FINAL ANSWER:" in step:
                    break
                prompt = problem + "\n\n" + "\n".join(steps)
            return self._format(steps)

        steps = self._heuristic_steps(problem)
        return self._format(steps)

    def _heuristic_steps(self, problem: str) -> List[str]:
        words = problem.split()
        steps = [f"The problem asks about: {' '.join(words[:20])}."]
        steps.append("Let me identify the core question and constraints.")
        if any(w in problem.lower() for w in ['calculate', 'compute', 'solve']):
            steps.append("This requires numerical computation.")
        elif any(w in problem.lower() for w in ['explain', 'why', 'how']):
            steps.append("This requires explanatory reasoning.")
        else:
            steps.append("This requires analysis and synthesis.")
        steps.append("Based on the analysis above, I can form a conclusion.")
        steps.append("FINAL ANSWER: [Based on the reasoning chain]")
        return steps

    def _format(self, steps: List[str]) -> str:
        inner = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
        return f"<think>\n{inner}\n</think>"


# ---------------------------------------------------------------------------
# SelfConsistency
# ---------------------------------------------------------------------------

class SelfConsistency:
    """Generate K samples and take majority vote."""

    def __init__(self, k: int = 5):
        self.k = k

    def reason(self, problem: str, generate_fn: Optional[Callable] = None) -> str:
        if generate_fn is not None:
            answers = []
            for _ in range(self.k):
                resp = generate_fn(problem)
                answer = self._extract_answer(resp)
                answers.append(answer)
            return self._majority_vote(answers)

        answers = self._heuristic_samples(problem)
        return self._majority_vote(answers)

    def _heuristic_samples(self, problem: str) -> List[str]:
        words = problem.lower().split()
        answers = []
        for i in range(self.k):
            rng = np.random.RandomState(i + hash(problem) % 10000)
            answer_words = rng.choice(words, size=min(3, len(words)), replace=True)
            answers.append(" ".join(answer_words))
        return answers

    def _extract_answer(self, text: str) -> str:
        patterns = [
            r'FINAL ANSWER:\s*(.+?)(?:\n|$)',
            r'Answer:\s*(.+?)(?:\n|$)',
            r'\*\*(.+?)\*\*',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return text.strip().split("\n")[-1].strip()

    def _majority_vote(self, answers: List[str]) -> str:
        if not answers:
            return ""
        counts = defaultdict(int)
        for a in answers:
            counts[a] += 1
        best = max(counts.items(), key=lambda x: x[1])
        return best[0]


# ---------------------------------------------------------------------------
# TreeOfThought
# ---------------------------------------------------------------------------

class TreeOfThought:
    """BFS over reasoning steps with scoring."""

    def __init__(self, breadth: int = 3, depth: int = 4):
        self.breadth = breadth
        self.depth = depth
        self.best_path: List[str] = []
        self.best_score = -math.inf

    def reason(self, problem: str, score_fn: Optional[Callable] = None) -> str:
        if score_fn is None:
            score_fn = self._heuristic_score

        queue: List[Tuple[List[str], float]] = [([], 0.0)]

        for step in range(self.depth):
            next_queue = []
            for path, path_score in queue:
                for b in range(self.breadth):
                    new_step = f"Step {step+1} attempt {b+1}"
                    new_path = path + [new_step]
                    score = score_fn(problem, new_path)
                    next_queue.append((new_path, path_score + score))

            if next_queue:
                next_queue.sort(key=lambda x: x[1], reverse=True)
                queue = next_queue[:self.breadth]

        for path, score in queue:
            if score > self.best_score:
                self.best_score = score
                self.best_path = path

        return self._format(problem, self.best_path)

    def _heuristic_score(self, problem: str, steps: List[str]) -> float:
        base = 1.0
        length_bonus = 0.1 * len(steps)
        return base + length_bonus + np.random.normal(0, 0.1)

    def _format(self, problem: str, steps: List[str]) -> str:
        inner = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
        return f"<think>Tree of Thought:\n{inner}\nBest score: {self.best_score:.2f}\n</think>"


# ---------------------------------------------------------------------------
# ReACTLoop
# ---------------------------------------------------------------------------

class ReACTLoop:
    """Alternate reasoning and action calls."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.actions: Dict[str, Callable] = {}

    def register_action(self, name: str, fn: Callable):
        self.actions[name] = fn

    def reason(self, problem: str, generate_fn: Optional[Callable] = None) -> str:
        trace = []
        observation = problem

        for i in range(self.max_iterations):
            if generate_fn:
                thought = generate_fn(f"Thought about: {observation}")
                action = self._parse_action(thought)
                if action:
                    name, args = action
                    result = self._execute_action(name, args)
                    trace.append({"thought": thought, "action": name, "observation": result})
                    observation = result
                else:
                    trace.append({"thought": thought, "action": None, "observation": None})
                    break
            else:
                thought, obs = self._heuristic_step(i, problem, observation)
                trace.append({"thought": thought, "action": "observe", "observation": obs})
                observation = obs
                if "DONE" in obs:
                    break

        return self._format(trace)

    def _heuristic_step(self, step: int, problem: str, observation: str) -> Tuple[str, str]:
        if step == 0:
            return f"Thinking about: {problem[:80]}", f"Analyzing: {problem[:60]}"
        elif step == 1:
            return "Let me gather more information.", f"Information gathered for: {problem[:40]}"
        elif step == 2:
            return "Now I can form a conclusion.", f"Conclusion: Based on analysis of '{problem[:30]}' DONE"
        return "DONE", "DONE"

    def _parse_action(self, text: str) -> Optional[Tuple[str, str]]:
        m = re.search(r'Action:\s*(\w+)\((.+?)\)', text)
        if m:
            return m.group(1), m.group(2)
        return None

    def _execute_action(self, name: str, args: str) -> str:
        if name in self.actions:
            try:
                return str(self.actions[name](args))
            except Exception as e:
                return f"Error: {e}"
        return f"Unknown action: {name}"

    def _format(self, trace: List[Dict[str, Any]]) -> str:
        lines = []
        for i, t in enumerate(trace):
            lines.append(f"--- ReAct Step {i+1} ---")
            lines.append(f"Thought: {t['thought']}")
            if t.get('action'):
                lines.append(f"Action: {t['action']}")
            if t.get('observation'):
                lines.append(f"Observation: {t['observation']}")
        return f"<think>\n" + "\n".join(lines) + "\n</think>"


# ---------------------------------------------------------------------------
# ConstitutionalCritique
# ---------------------------------------------------------------------------

class ConstitutionalCritique:
    """Critique -> revision loop per principle."""

    def __init__(self, principles: Optional[List[str]] = None):
        self.principles = principles or [
            "Is the response helpful and harmless?",
            "Does it avoid biased or discriminatory language?",
            "Is the information accurate and well-sourced?",
            "Does it respect user privacy and safety?",
        ]

    def reason(self, problem: str, generate_fn: Optional[Callable] = None) -> str:
        if generate_fn:
            draft = generate_fn(problem)
        else:
            draft = f"Initial response to: {problem}"

        for i, principle in enumerate(self.principles):
            if generate_fn:
                critique = generate_fn(f"Critique this response for principle '{principle}': {draft}")
                draft = generate_fn(f"Revise based on critique: {critique}\nOriginal: {draft}")
            else:
                critique = f"Reviewing for: {principle} -> looks acceptable"
                draft = f"[Revised {i+1}] {draft}"

        return self._format(draft, [])

    def _format(self, final: str, critiques: List[str]) -> str:
        return f"<think>\nConstitutional review:\n{final}\n</think>"


# ---------------------------------------------------------------------------
# ScratchpadTrainer
# ---------------------------------------------------------------------------

class ScratchpadTrainer:
    """Intermediate steps format for training."""

    def __init__(self, steps_per_example: int = 3):
        self.steps_per_example = steps_per_example
        self.training_data: List[Dict[str, Any]] = []

    def reason(self, problem: str, generate_fn: Optional[Callable] = None) -> str:
        if generate_fn:
            steps = []
            for i in range(self.steps_per_example):
                prompt = f"Problem: {problem}\nSteps so far: {'; '.join(steps)}\nNext step:"
                step = generate_fn(prompt)
                steps.append(step)
            return self._format(problem, steps)

        steps = self._heuristic_steps(problem)
        return self._format(problem, steps)

    def _heuristic_steps(self, problem: str) -> List[str]:
        words = problem.split()
        steps = []
        steps.append(f"Analyze: {problem[:60]}")
        steps.append(f"Key elements: {', '.join(words[:5])}")
        steps.append("Synthesize conclusion from analysis")
        return steps

    def _format(self, problem: str, steps: List[str]) -> str:
        scratchpad = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
        return f"<think>\nProblem: {problem[:80]}\n{scratchpad}\n</think>"

    def add_training_example(self, problem: str, steps: List[str], answer: str):
        self.training_data.append({
            "problem": problem,
            "steps": steps,
            "answer": answer,
        })

    def get_training_pairs(self) -> List[Tuple[str, str]]:
        pairs = []
        for ex in self.training_data:
            input_text = f"Problem: {ex['problem']}\nSteps:"
            output_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(ex['steps']))
            output_text += f"\nAnswer: {ex['answer']}"
            pairs.append((input_text, output_text))
        return pairs


class PAL:
    """Program-Aided Language: generate executable Python, run it, use stdout."""
    def __init__(self, model):
        self.model = model

    def reason(self, question: str) -> str:
        code = self.model.generate(f"Write Python code to solve: {question}")
        try:
            local_ns = {}
            exec(code, {"__builtins__": {}}, local_ns)
            result = local_ns.get("result", local_ns.get("answer", str(code)))
            return str(result)
        except Exception as e:
            return f"Error: {e}"


class SocraticMethod:
    """Chain of decomposed sub-questions."""
    def __init__(self, model, max_depth=3):
        self.model = model
        self.max_depth = max_depth

    def reason(self, question: str) -> str:
        sub_questions = self._decompose(question)
        answers = []
        for sq in sub_questions:
            answer = self.model.generate(f"Answer concisely: {sq}")
            answers.append(f"Q: {sq}\nA: {answer}")
        synthesis = self.model.generate(
            f"Original question: {question}\n"
            f"Sub-answers:\n" + "\n".join(answers) +
            "\nSynthesize a final answer."
        )
        return synthesis

    def _decompose(self, question: str) -> list:
        raw = self.model.generate(
            f"Break this question into {self.max_depth} simpler sub-questions, "
            f"one per line: {question}"
        )
        return [line.strip() for line in raw.strip().split("\n") if line.strip()][:self.max_depth]
