"""Agent modules: ToolCall, ToolRegistry, ReACTLoop, ReflexionAgent, MultiAgentDebate."""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    result: Any = None


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.schemas: Dict[str, Dict] = {}

    def register(self, name: str, func: Callable, schema: Optional[Dict] = None) -> None:
        self.tools[name] = func
        if schema is None:
            schema = {"type": "object", "properties": {}}
        self.schemas[name] = schema

    def call(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not registered")
        return self.tools[name](**arguments)

    def get_tool_descriptions(self) -> str:
        lines = []
        for name, schema in self.schemas.items():
            props = schema.get("properties", {})
            param_str = ", ".join(props.keys()) if props else "none"
            lines.append(f"- {name}({param_str})")
        return "\n".join(lines)


class ReACTLoop:
    def __init__(self, registry: ToolRegistry, max_steps: int = 10):
        self.registry = registry
        self.max_steps = max_steps

    def _reason(self, input_text: str, history: List[Dict]) -> str:
        if not history:
            return f"Thought: I need to process: {input_text}\nAction: none"
        last = history[-1]
        return f"Thought: The previous action gave: {last['result']}\nAction: none"

    def _parse_action(self, reasoning: str) -> Optional[Tuple[str, Dict]]:
        action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", reasoning)
        if action_match and action_match.group(1) != "none":
            name = action_match.group(1)
            args_str = action_match.group(2)
            args = {}
            if args_str:
                for pair in args_str.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        args[k.strip()] = v.strip().strip("'\"")
            return name, args
        return None

    def run(self, input_text: str) -> str:
        history: List[Dict] = []
        for step in range(self.max_steps):
            reasoning = self._reason(input_text, history)
            action = self._parse_action(reasoning)
            if action is None:
                return reasoning
            name, args = action
            try:
                result = self.registry.call(name, args)
            except Exception as e:
                result = f"Error: {e}"
            tool_call = ToolCall(name=name, arguments=args, result=result)
            history.append({"tool_call": tool_call, "result": result})
        return f"After {self.max_steps} steps, result: {history[-1]['result']}"


class ReflexionAgent:
    def __init__(self, registry: ToolRegistry, max_attempts: int = 5):
        self.registry = registry
        self.max_attempts = max_attempts
        self.failed_attempts: List[Dict] = []

    def _reflect(self, input_text: str) -> str:
        if not self.failed_attempts:
            return ""
        reflections = []
        for attempt in self.failed_attempts[-3:]:
            reflections.append(f"Previously tried {attempt['action']} with {attempt['args']}, got: {attempt['error']}")
        return "\n".join(reflections)

    def _plan(self, input_text: str, reflection: str) -> Tuple[str, Dict]:
        tools = list(self.registry.tools.keys())
        if reflection:
            for attempt in self.failed_attempts:
                if attempt["action"] in tools:
                    tools.remove(attempt["action"])
        if not tools:
            tools = list(self.registry.tools.keys())
        tool_name = tools[0]
        return tool_name, {"text": input_text}

    def run(self, input_text: str) -> str:
        for attempt in range(self.max_attempts):
            reflection = self._reflect(input_text)
            action, args = self._plan(input_text, reflection)
            try:
                result = self.registry.call(action, args)
                return f"Success on attempt {attempt + 1}: {result}"
            except Exception as e:
                self.failed_attempts.append({
                    "action": action,
                    "args": args,
                    "error": str(e),
                })
        return f"Failed after {self.max_attempts} attempts. Errors: {[a['error'] for a in self.failed_attempts]}"


class MultiAgentDebate:
    def __init__(self, num_agents: int = 3, max_rounds: int = 3):
        self.num_agents = num_agents
        self.max_rounds = max_rounds

    def _agent_respond(self, agent_id: int, input_text: str, history: List[str]) -> str:
        if not history:
            return f"Agent {agent_id}: My analysis of '{input_text}' is that it requires careful consideration."
        last = history[-1]
        if "agree" in last.lower():
            return f"Agent {agent_id}: I agree with the previous point. Additionally, we should consider edge cases."
        return f"Agent {agent_id}: I disagree. The previous analysis missed important nuances about '{input_text}'."

    def _judge(self, input_text: str, statements: List[str]) -> str:
        if not statements:
            return "No arguments presented."
        return f"Judgment on '{input_text}': After reviewing {len(statements)} arguments, the consensus is that the topic requires balanced analysis considering multiple perspectives."

    def run(self, input_text: str) -> str:
        all_statements: List[str] = []
        for round_num in range(self.max_rounds):
            round_statements = []
            for agent_id in range(self.num_agents):
                statement = self._agent_respond(agent_id, input_text, all_statements)
                round_statements.append(statement)
                all_statements.append(statement)
        return self._judge(input_text, all_statements)
