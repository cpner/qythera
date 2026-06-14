import urllib.request, json

class APICallerTool:
    name = "api_caller"
    description = "Make HTTP API calls"

    def execute(self, url="", method="GET", headers=None, body=None, **kwargs):
        try:
            data = json.dumps(body).encode() if body else None
            req = urllib.request.Request(url, data=data, method=method,
                headers={"Content-Type": "application/json", **(headers or {})})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode()[:5000]
        except Exception as e:
            return f"API Error: {e}"
