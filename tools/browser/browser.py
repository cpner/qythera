import urllib.request, urllib.parse, json, re

class BrowserTool:
    name = "browser"
    description = "Fetch and parse web pages"

    def execute(self, url="", action="fetch", query="", **kwargs):
        if action == "fetch":
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Qythera/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    content = resp.read().decode('utf-8', errors='ignore')
                    text = re.sub(r'<[^>]+>', ' ', content)
                    text = re.sub(r'\s+', ' ', text).strip()
                    return text[:5000]
            except Exception as e:
                return f"Error: {e}"
        elif action == "search":
            try:
                url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
                req = urllib.request.Request(url, headers={"User-Agent": "Qythera/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    html = resp.read().decode()
                    results = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html)
                    return "\n".join(results[:5]) or "No results"
            except Exception as e:
                return f"Search error: {e}"
        return f"Unknown action: {action}"
