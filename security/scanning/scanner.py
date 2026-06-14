class PromptScanner:
    INJECTION_PATTERNS = ["ignore previous", "forget instructions", "you are now",
                          "jailbreak", "DAN mode", "act as evil", "bypass safety"]

    def scan(self, prompt: str) -> dict:
        prompt_lower = prompt.lower()
        threats = [p for p in self.INJECTION_PATTERNS if p in prompt_lower]
        return {"safe": len(threats) == 0, "threats": threats, "risk_level": "high" if threats else "low"}

class OutputScanner:
    PII_PATTERNS = {"email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"}

    def scan(self, text: str) -> dict:
        import re
        findings = {}
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches: findings[pii_type] = matches
        return {"has_pii": len(findings) > 0, "findings": findings}
