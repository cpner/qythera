import re
from typing import Dict, Tuple, List


TOXIC_PATTERNS = [
    r"kill",
    r"murder",
    r"assassinate",
    r"execute",
    r"\bhate\b",
    r"\bracist\b",
    r"\bsexist\b",
    r"\bbomb\b",
    r"\bweapon\b",
    r"\bexplosive\b",
    r"\bsuicide\b",
    r"\bself.harm\b",
    r"\babuse\b",
    r"\bharass\b",
    r"\bthreaten\b",
]

JAILBREAK_PATTERNS = [
    r"ignore.{0,30}(previous|all|above|prior).{0,30}(instructions|rules|prompts|constraints)",
    r"you are now.{0,20}(unrestricted|uncensored|free)",
    r"pretend.{0,30}(no|without).{0,20}(rules|restrictions|limits)",
    r"bypass.{0,20}(safety|filter|content)",
    r"\bDAN\b.{0,10}mode",
    r"act as.{0,20}(evil|dark|unethical|villain)",
    r"reveal.{0,20}(your|all).{0,20}(system)?prompts?",
    r"disregard.{0,20}(all|any|previous)",
    r"forget.{0,20}(all|your).{0,20}(instructions|rules)",
    r"do anything now",
    r"jailbreak",
    r"ignore.{0,40}instructions",
    r"you are now a",
]

PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
}


class SafetyModerator:
    """Content safety moderator with toxicity, jailbreak, and PII detection."""

    def __init__(self, toxicity_threshold=0.3, enable_jailbreak=True, enable_pii=True):
        self.tox_thresh = toxicity_threshold
        self.enable_jb = enable_jailbreak
        self.enable_pii = enable_pii
        self.tox_patterns = [re.compile(p, re.IGNORECASE) for p in TOXIC_PATTERNS]
        self.jb_patterns = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]
        self.pii_patterns = {k: re.compile(v) for k, v in PII_PATTERNS.items()}

    def check_toxicity(self, text: str) -> Tuple[float, List[str]]:
        matches = []
        for p in self.tox_patterns:
            found = p.findall(text)
            if found:
                matches.extend(found if isinstance(found[0], str) else [str(found)])
        score = min(len(matches) / 2.0, 1.0)
        return score, matches

    def check_jailbreak(self, text: str) -> Tuple[bool, List[str]]:
        matches = []
        for p in self.jb_patterns:
            found = p.findall(text)
            if found:
                matches.extend(found if isinstance(found[0], str) else [str(found)])
        return len(matches) > 0, matches

    def check_pii(self, text: str) -> Dict[str, List[str]]:
        findings = {}
        for pii_type, pattern in self.pii_patterns.items():
            found = pattern.findall(text)
            if found:
                findings[pii_type] = found
        return findings

    def redact_pii(self, text: str) -> str:
        redacted = text
        for pii_type, pattern in self.pii_patterns.items():
            redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)
        return redacted

    def moderate(self, text: str) -> Dict:
        tox_score, tox_matches = self.check_toxicity(text)
        is_jb, jb_matches = self.check_jailbreak(text) if self.enable_jb else (False, [])
        pii = self.check_pii(text) if self.enable_pii else {}
        safe = tox_score < self.tox_thresh and not is_jb and not pii
        return {
            "safe": safe,
            "toxicity_score": tox_score,
            "toxicity_matches": tox_matches,
            "jailbreak_detected": is_jb,
            "jailbreak_matches": jb_matches,
            "pii_found": bool(pii),
            "pii_details": pii,
        }

    def filter_input(self, text: str) -> Tuple[bool, str]:
        result = self.moderate(text)
        if not result["safe"]:
            reasons = []
            if result["toxicity_score"] >= self.tox_thresh:
                reasons.append("toxic content")
            if result["jailbreak_detected"]:
                reasons.append("jailbreak attempt")
            if result["pii_found"]:
                reasons.append("PII detected")
            return False, f"Blocked: {', '.join(reasons)}"
        return True, text
