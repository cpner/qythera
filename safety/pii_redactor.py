import re
from typing import Dict, List, Tuple


class PIIRedactor:
    PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    def __init__(self):
        self.compiled = {k: re.compile(v) for k, v in self.PATTERNS.items()}

    def detect(self, text: str) -> Dict[str, List[str]]:
        findings = {}
        for pii_type, pattern in self.compiled.items():
            matches = pattern.findall(text)
            if matches:
                findings[pii_type] = matches
        return findings

    def redact(self, text: str) -> Tuple[str, Dict[str, int]]:
        redacted = text
        counts = {}
        for pii_type, pattern in self.compiled.items():
            matches = pattern.findall(redacted)
            if matches:
                counts[pii_type] = len(matches)
                redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)
        return redacted, counts

    def check_and_redact(self, text: str, auto_redact: bool = True) -> Dict:
        findings = self.detect(text)
        has_pii = len(findings) > 0
        redacted_text = text
        redaction_counts = {}
        if has_pii and auto_redact:
            redacted_text, redaction_counts = self.redact(text)
        return {
            "original": text,
            "redacted": redacted_text,
            "has_pii": has_pii,
            "findings": findings,
            "redaction_counts": redaction_counts,
        }
