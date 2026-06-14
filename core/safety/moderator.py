"""Safety filters: toxicity, jailbreak, PII detection."""

import re
from typing import Dict, Tuple, List


TOXIC = [r"\b(kill|murder|abuse|bomb|weapon|suicide|self-harm)\b"]
JAILBREAK = [r"ignore.{0,30}(previous|all|above).{0,30}(instructions|rules)",
             r"bypass.{0,20}(safety|filter)", r"jailbreak", r"DAN mode",
             r"you are now.{0,20}(unrestricted|uncensored)"]
PII = {"email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
       "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
       "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
       "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"}


class SafetyModerator:
    def __init__(self, toxicity_threshold=0.3):
        self.tox_thresh = toxicity_threshold
        self.tox_patterns = [re.compile(p, re.I) for p in TOXIC]
        self.jb_patterns = [re.compile(p, re.I) for p in JAILBREAK]
        self.pii_patterns = {k: re.compile(v) for k, v in PII.items()}
    
    def check_toxicity(self, text):
        matches = []
        for p in self.tox_patterns:
            found = p.findall(text)
            if found: matches.extend(found)
        return min(len(matches) / 2.0, 1.0), matches
    
    def check_jailbreak(self, text):
        matches = []
        for p in self.jb_patterns:
            found = p.findall(text)
            if found: matches.extend(found)
        return len(matches) > 0, matches
    
    def check_pii(self, text):
        return {k: p.findall(text) for k, p in self.pii_patterns.items() if p.findall(text)}
    
    def redact_pii(self, text):
        for pii_type, pattern in self.pii_patterns.items():
            text = pattern.sub(f"[REDACTED_{pii_type.upper()}]", text)
        return text
    
    def moderate(self, text):
        tox_score, tox_matches = self.check_toxicity(text)
        is_jb, jb_matches = self.check_jailbreak(text)
        pii = self.check_pii(text)
        safe = tox_score < self.tox_thresh and not is_jb and not pii
        return {"safe": safe, "toxicity": tox_score, "jailbreak": is_jb, "pii": bool(pii)}
    
    def filter_input(self, text):
        result = self.moderate(text)
        if not result["safe"]:
            reasons = []
            if result["toxicity"] >= self.tox_thresh: reasons.append("toxic")
            if result["jailbreak"]: reasons.append("jailbreak")
            if result["pii"]: reasons.append("PII")
            return False, f"Blocked: {', '.join(reasons)}"
        return True, text
