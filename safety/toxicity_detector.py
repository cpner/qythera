import re
from typing import Tuple


TOXIC_PATTERNS = [
    r"\b(kill|murder|assassinate)\b",
    r"\b(hate|racist|sexist)\b",
    r"\b(bomb|explosive|weapon)\b",
    r"\b(suicide|self.harm)\b",
    r"\b(abuse|harass|threaten)\b",
]

TOXICITY_THRESHOLD = 0.5


class ToxicityDetector:
    def __init__(self, threshold: float = TOXICITY_THRESHOLD):
        self.threshold = threshold
        self.patterns = [re.compile(p, re.IGNORECASE) for p in TOXIC_PATTERNS]

    def detect(self, text: str) -> Tuple[bool, float, list]:
        matches = []
        for pattern in self.patterns:
            found = pattern.findall(text)
            matches.extend(found)
        score = min(len(matches) / 5.0, 1.0)
        is_toxic = score >= self.threshold
        return is_toxic, score, matches

    def filter_text(self, text: str) -> Tuple[str, bool]:
        is_toxic, score, matches = self.detect(text)
        if is_toxic:
            filtered = text
            for match in matches:
                filtered = filtered.replace(match, "[FILTERED]")
            return filtered, True
        return text, False
