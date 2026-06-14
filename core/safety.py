import re

class Safety:
    def __init__(self):
        self.tox = [re.compile(p, re.I) for p in [r"kill|murder|abuse|bomb|weapon|suicide|self-harm"]]
        self.jb = [re.compile(p, re.I) for p in [r"ignore.{0,30}(previous|all|above)", r"bypass.{0,20}(safety|filter)", r"jailbreak", r"DAN mode"]]
        self.pii = {"email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")}

    def check(self, text):
        tox = sum(1 for p in self.tox if p.findall(text))
        jb = any(p.findall(text) for p in self.jb)
        pii = {k: p.findall(text) for k, p in self.pii.items() if p.findall(text)}
        safe = tox < 1 and not jb and not pii
        return safe, {"toxic": tox > 0, "jailbreak": jb, "pii": bool(pii)}

    def redact(self, text):
        for k, p in self.pii.items(): text = p.sub(f"[REDACTED]", text)
        return text
