import json
import re
from typing import Optional, Callable, List


class ReActAgent:
    def __init__(self, model_fn: Callable, tools: Optional[List] = None,
                 max_iterations: int = 10, verbose: bool = True):
        self.model_fn = model_fn
        self.tools = {t.name: t for t in (tools or [])}
        self.max_iterations = max_iterations
        self.verbose = verbose

    def parse_action(self, text: str):
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        if action_match:
            tool_name = action_match.group(1)
            try:
                args_str = action_match.group(2).strip()
                if args_str:
                    args = json.loads("{" + args_str + "}")
                else:
                    args = {}
            except json.JSONDecodeError:
                args = {"input": args_str}
            return thought, tool_name, args
        return thought, None, None

    def run(self, query: str) -> str:
        tool_descriptions = "\n".join(
            f"- {name}: {tool.description}" for name, tool in self.tools.items()
        )

        system_prompt = f"""You are a helpful assistant that reasons step by step.

Available tools:
{tool_descriptions}

To use a tool, format your response as:
Thought: <your reasoning>
Action: tool_name(param="value")

When you have the final answer:
Thought: I now know the final answer
Final Answer: <your answer>"""

        history = [f"Question: {query}"]

        for i in range(self.max_iterations):
            prompt = system_prompt + "\n\n" + "\n".join(history)
            response = self.model_fn(prompt)
            history.append(response)

            if self.verbose:
                print(f"--- Iteration {i+1} ---")
                print(response)

            thought, tool_name, args = self.parse_action(response)

            if tool_name is None:
                final_answer_match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL)
                if final_answer_match:
                    return final_answer_match.group(1).strip()
                return response

            if tool_name in self.tools:
                observation = self.tools[tool_name].execute(**args)
            else:
                observation = f"Error: Unknown tool '{tool_name}'"

            history.append(f"Observation: {observation}")

        return "Maximum iterations reached without a final answer."

    def chat(self, query: str) -> str:
        return self.run(query)
