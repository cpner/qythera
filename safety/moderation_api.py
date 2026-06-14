from typing import Dict, Optional
from safety.toxicity_detector import ToxicityDetector
from safety.jailbreak_filter import JailbreakFilter
from safety.pii_redactor import PIIRedactor


class ModerationAPI:
    def __init__(self):
        self.toxicity = ToxicityDetector()
        self.jailbreak = JailbreakFilter()
        self.pii = PIIRedactor()

    def moderate(self, text: str, check_pii: bool = True) -> Dict:
        is_toxic, tox_score, tox_matches = self.toxicity.detect(text)
        jb_result = self.jailbreak.check(text)
        pii_result = self.pii.check_and_redact(text) if check_pii else {"has_pii": False}

        safe = not is_toxic and jb_result["safe"] and not pii_result.get("has_pii", False)
        return {
            "safe": safe,
            "toxicity": {"score": tox_score, "matches": tox_matches, "is_toxic": is_toxic},
            "jailbreak": jb_result,
            "pii": pii_result,
        }

    def filter_input(self, text: str) -> tuple:
        mod = self.moderate(text)
        if not mod["safe"]:
            reasons = []
            if mod["toxicity"]["is_toxic"]:
                reasons.append("toxic content")
            if mod["jailbreak"]["jailbreak_detected"]:
                reasons.append("jailbreak attempt")
            if mod["jailbreak"]["harmful_detected"]:
                reasons.append("harmful request")
            if mod["pii"]["has_pii"]:
                reasons.append("contains PII")
            return f"Request blocked: {', '.join(reasons)}", mod
        return text, mod
