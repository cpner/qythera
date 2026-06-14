from typing import List, Dict, Optional, Callable
import json


class BaseAgent:
    def __init__(self, model_fn: Callable, tools: Optional[List] = None,
                 max_iterations: int = 10, verbose: bool = True):
        self.model_fn = model_fn
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.tool_registry = {}
        for tool in tools:
            self.tool_registry[tool.name] = tool

    def think(self, prompt: str) -> str:
        return self.model_fn(prompt)

    def act(self, tool_name: str, **kwargs) -> str:
        if tool_name not in self.tool_registry:
            return f"Error: Unknown tool '{tool_name}'"
        tool = self.tool_registry[tool_name]
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return f"Error: {str(e)}"

    def get_tool_descriptions(self) -> str:
        descriptions = []
        for name, tool in self.tool_registry.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)
