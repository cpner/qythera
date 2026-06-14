from agent.tools.tool_registry import Tool


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for information"

    def execute(self, query: str = "", **kwargs) -> str:
        try:
            import urllib.request
            import urllib.parse
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "Qythera/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                import json
                data = json.loads(resp.read().decode())
                results = data.get("AbstractText", "No results found")
                return results[:2000]
        except Exception as e:
            return f"Search error: {str(e)}"
