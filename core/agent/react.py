import json
import re
from typing import Callable, List, Dict, Optional


class Tool:
    """Base tool class."""
    def __init__(self, name, description, execute_fn):
        self.name = name
        self.description = description
        self.execute_fn = execute_fn

    def execute(self, **kwargs):
        return self.execute_fn(**kwargs)


class ReActAgent:
    """ReAct (Reason + Act) agent.
    
    Loop: Thought -> Action -> Observation -> repeat
    Uses the LLM to generate structured reasoning and tool calls.
    """
    
    def __init__(self, model_fn: Callable, tools: List[Tool] = None, max_iterations: int = 10):
        self.model_fn = model_fn
        self.tools = {t.name: t for t in (tools or [])}
        self.max_iterations = max_iterations

    def _format_tools(self):
        return "\n".join(f"- {name}: {tool.description}" for name, tool in self.tools.items())

    def _parse_action(self, text):
        thought = ""
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        action_match = re.search(r"Action:\s*(\w+)\((.+?)\)", text)
        if action_match:
            tool_name = action_match.group(1)
            args_str = action_match.group(2)
            try:
                args = dict(re.findall(r"(\w+)=[\'"](.*?)[\'"]", args_str))
            except:
                args = {"input": args_str}
            return thought, tool_name, args
        
        return thought, None, None

    def run(self, query: str) -> str:
        system = f"""You are a helpful assistant that reasons step by step.

Available tools:
{self._format_tools()}

To use a tool:
Thought: <your reasoning>
Action: tool_name(param="value")

When you have the final answer:
Thought: I now know the final answer
Final Answer: <your answer>"""

        history = [f"Question: {query}"]

        for i in range(self.max_iterations):
            prompt = system + "\n\n" + "\n".join(history)
            response = self.model_fn(prompt)
            history.append(response)

            thought, tool_name, args = self._parse_action(response)

            if tool_name is None:
                fa_match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL)
                if fa_match:
                    return fa_match.group(1).strip()
                return response

            if tool_name in self.tools:
                try:
                    observation = self.tools[tool_name].execute(**args)
                except Exception as e:
                    observation = f"Error: {e}"
            else:
                observation = f"Unknown tool: {tool_name}"

            history.append(f"Observation: {observation}")

        return "Maximum iterations reached."

    def chat(self, query: str) -> str:
        return self.run(query)
