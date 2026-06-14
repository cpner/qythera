import pytest
from agent.react import ReActAgent
from agent.plan_execute import PlanExecuteAgent
from agent.tools.calculator import CalculatorTool


class TestReActAgent:
    def test_agent_creation(self):
        agent = ReActAgent(model_fn=lambda x: "Final Answer: test", tools=[CalculatorTool()])
        assert len(agent.tools) == 1

    def test_agent_run(self):
        agent = ReActAgent(
            model_fn=lambda x: "Thought: I can calculate this\nAction: calculator(expression=\"2+2\")\n",
            tools=[CalculatorTool()],
            max_iterations=1,
        )
        result = agent.run("What is 2+2?")
        assert "4" in result


class TestPlanExecute:
    def test_plan_creation(self):
        agent = PlanExecuteAgent(
            model_fn=lambda x: '["Step 1: Think", "Step 2: Answer"]',
            max_iterations=1,
        )
        plan = agent.create_plan("test")
        assert len(plan) == 2
