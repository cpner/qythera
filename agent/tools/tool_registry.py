from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import inspect


class Tool(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def execute(self, **kwargs) -> str:
        pass

    def to_schema(self) -> Dict:
        return {"name": self.name, "description": self.description}


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def list_tools(self):
        return [t.to_schema() for t in self.tools.values()]

    def execute(self, name: str, **kwargs) -> str:
        tool = self.get(name)
        if not tool:
            return f"Unknown tool: {name}"
        return tool.execute(**kwargs)
