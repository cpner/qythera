import re
from typing import Tuple


JAILBREAK_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+(instructions|prompts|rules)",
    r"you\s+are\s+now\s+(a|an)\s+(?:unrestricted|uncensored)",
    r"pretend\s+(you|that)\s+(have|there)\s+(are|is)\s+no\s+(rules|restrictions|limits)",
    r"bypass\s+(all|any|your)\s+(safety|content|filter)",
    r"do\s+anything\s+now",
    r"DAN\s+mode",
    r"jailbreak",
    r"act\s+as\s+(?:a\s+)?(?:evil|dark|unethical)",
    r"roleplay\s+as\s+(?:a\s+)?(?:villain|criminal|villain)",
    r"reveal\s+(your|all)\s+(system\s+)?prompts?",
]

HARMFUL_REQUEST_PATTERNS = [
    r"how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|explosive)",
    r"(hack|compromise)\s+(into|a)\s+(system|computer|network)",
    r"(steal|stealing)\s+(data|information|credentials)",
    r"(create|make)\s+(malware|virus|ransomware)",
]


class JailbreakFilter:
    def __init__(self):
        self.jailbreak_patterns = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]
        self.harmful_patterns = [re.compile(p, re.IGNORECASE) for p in HARMFUL_REQUEST_PATTERNS]

    def detect_jailbreak(self, text: str) -> Tuple[bool, list]:
        matches = []
        for pattern in self.jailbreak_patterns:
            found = pattern.findall(text)
            if found:
                matches.extend(found)
        return len(matches) > 0, matches

    def detect_harmful(self, text: str) -> Tuple[bool, list]:
        matches = []
        for pattern in self.harmful_patterns:
            found = pattern.findall(text)
            if found:
                matches.extend(found)
        return len(matches) > 0, matches

    def check(self, text: str) -> dict:
        is_jailbreak, jb_matches = self.detect_jailbreak(text)
        is_harmful, harm_matches = self.detect_harmful(text)
        return {
            "safe": not is_jailbreak and not is_harmful,
            "jailbreak_detected": is_jailbreak,
            "harmful_detected": is_harmful,
            "jailbreak_matches": jb_matches,
            "harmful_matches": harm_matches,
        }

    def filter(self, text: str) -> Tuple[str, dict]:
        result = self.check(text)
        if not result["safe"]:
            return "I cannot process this request as it may be harmful or violate safety guidelines.", result
        return text, result
