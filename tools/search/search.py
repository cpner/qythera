import urllib.request, urllib.parse, re

class SearchTool:
    name = "search"
    description = "Search the internet for information"

    def execute(self, query="", engine="duckduckgo", num_results=5, **kwargs):
        try:
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode()
            titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html)
            snippets = re.findall(r'class="result__snippet">(.*?)</[a-z]', html, re.DOTALL)
            results = []
            for i in range(min(num_results, len(titles))):
                title = re.sub(r'<[^>]+>', '', titles[i]).strip()
                snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
                results.append(f"{i+1}. {title}\n   {snippet}")
            return "\n\n".join(results) or "No results found"
        except Exception as e:
            return f"Search error: {e}"
