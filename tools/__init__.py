from tools.code_interpreter.interpreter import CodeInterpreterTool
from tools.browser.browser import BrowserTool
from tools.api_caller.caller import APICallerTool
from tools.file_ops.operations import FileOpsTool
from tools.search.search import SearchTool
from tools.calculator.calc import CalculatorTool

ALL_TOOLS = [CodeInterpreterTool(), BrowserTool(), APICallerTool(),
             FileOpsTool(), SearchTool(), CalculatorTool()]
