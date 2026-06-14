import os, hashlib, json

class SecurityAuditor:
    def __init__(self, project_root):
        self.root = project_root
        self.findings = []

    def scan_secrets(self):
        patterns = ["API_KEY", "SECRET", "PASSWORD", "TOKEN", "PRIVATE_KEY"]
        for dirpath, _, filenames in os.walk(self.root):
            if ".git" in dirpath: continue
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, errors='ignore') as f:
                        for i, line in enumerate(f, 1):
                            for pattern in patterns:
                                if pattern in line and "=" in line:
                                    self.findings.append({"file": fpath, "line": i, "type": "secret", "pattern": pattern})
                except: pass
        return self.findings

    def check_dependencies(self):
        req_path = os.path.join(self.root, "requirements.txt")
        if os.path.exists(req_path):
            with open(req_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        self.findings.append({"type": "dependency", "package": line.split("==")[0].split(">=")[0]})
        return self.findings

    def generate_report(self):
        return {"total_findings": len(self.findings), "findings": self.findings,
                "recommendations": ["Use environment variables for secrets", "Pin dependency versions",
                                    "Enable pre-commit hooks", "Run security scanning in CI/CD"]}
