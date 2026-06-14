#!/usr/bin/env python3
"""Universal test - verifies Qythera works on any platform."""

import sys
import os
import platform

def test_knowledge():
    sys.path.insert(0, '.')
    from core.knowledge.base import get_answer
    tests = [("Hello", 5), ("Python", 5), ("2+2", 5), ("fibonacci", 5),
             ("transformers", 5), ("git", 5), ("physics", 5), ("math", 5),
             ("photosynthesis", 5), ("sqrt 144", 5), ("Who are you", 5)]
    for q, min_len in tests:
        r = get_answer(q)
        assert len(r) >= min_len, f"FAIL: {q}"
    return True

def test_safety():
    sys.path.insert(0, '.')
    from core.safety import SafetyModerator
    sf = SafetyModerator()
    assert sf.moderate("Hello")["safe"]
    assert not sf.moderate("kill")["safe"]
    assert not sf.moderate("jailbreak")["safe"]
    assert not sf.moderate("test@example.com")["safe"]
    return True

def test_server():
    sys.path.insert(0, '.')
    from core.inference.server import QytheraAI
    ai = QytheraAI()
    for q in ["Hello", "Python", "fibonacci", "2+2"]:
        r = ai.generate([{"role":"user","content":q}])
        assert len(r) > 5
    return True

def main():
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print()
    
    tests = [("Knowledge", test_knowledge), ("Safety", test_safety), ("Server", test_server)]
    passed = 0
    for name, test in tests:
        try:
            test()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} - {e}")
    
    print(f"\nResult: {passed}/{len(tests)} tests passed")
    return passed == len(tests)

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
