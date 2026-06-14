
import re
from typing import Dict, Tuple

TOXIC = [r"\b(kill|murder|hate|abuse|bomb|weapon|suicide|self-harm)\b"]
JAILBREAK = [r"ignore\s+(previous|all|above)\s+(instructions|rules)", r"you\s+are\s+now\s+unrestricted",
             r"bypass\s+(safety|filter)", r"DAN\s+mode", r"jailbreak"]
PII = {"email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
       "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
       "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
       "credit": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"}

class SafetyModerator:
    def __init__(self, toxicity_thresh=0.5, enable_jailbreak=True, enable_pii=True):
        self.tox_thresh = toxicity_thresh
        self.enable_jb = enable_jailbreak
        self.enable_pii = enable_pii
        self.tox_patterns = [re.compile(p, re.I) for p in TOXIC]
        self.jb_patterns = [re.compile(p, re.I) for p in JAILBREAK]
        self.pii_patterns = {k: re.compile(v) for k, v in PII.items()}

    def check_toxicity(self, text):
        matches = []
        for p in self.tox_patterns: matches.extend(p.findall(text))
        score = min(len(matches) / 3.0, 1.0)
        return score, matches

    def check_jailbreak(self, text):
        matches = []
        for p in self.jb_patterns: matches.extend(p.findall(text))
        return len(matches) > 0, matches

    def check_pii(self, text):
        findings = {}
        for k, p in self.pii_patterns.items():
            m = p.findall(text)
            if m: findings[k] = m
        return findings

    def redact_pii(self, text):
        redacted = text
        for k, p in self.pii_patterns.items():
            redacted = p.sub(f"[REDACTED_{k.upper()}]", redacted)
        return redacted

    def moderate(self, text) -> Dict:
        tox_score, tox_matches = self.check_toxicity(text)
        is_jb, jb_matches = self.check_jailbreak(text) if self.enable_jb else (False, [])
        pii = self.check_pii(text) if self.enable_pii else {}
        safe = tox_score < self.tox_thresh and not is_jb and not pii
        result = {"safe": safe, "toxicity": tox_score, "jailbreak": is_jb, "pii": bool(pii)}
        if not safe and pii: result["redacted"] = self.redact_pii(text)
        return result

    def filter_input(self, text):
        result = self.moderate(text)
        if not result["safe"]:
            reasons = []
            if result["toxicity"] >= self.tox_thresh: reasons.append("toxic")
            if result["jailbreak"]: reasons.append("jailbreak")
            if result["pii"]: reasons.append("PII detected")
            return False, f"Blocked: {', '.join(reasons)}"
        return True, text
