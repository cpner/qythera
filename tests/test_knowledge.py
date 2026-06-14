import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.knowledge.base import get_answer

class TestKnowledge:
    def test_hello(self): assert len(get_answer("Hello")) > 10
    def test_python(self): assert "Python" in get_answer("What is Python?")
    def test_math(self): assert "4" in get_answer("What is 2+2?")
    def test_code(self): assert "def" in get_answer("Write fibonacci code")
    def test_sqrt(self): assert "12" in get_answer("sqrt of 144")
    def test_identity(self): assert "Qythera" in get_answer("Who are you?")
    def test_help(self): assert "Programming" in get_answer("What can you do?")
