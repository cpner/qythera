import aiohttp
from typing import List, Dict, Optional, AsyncGenerator

class AsyncQytheraClient:
    def __init__(self, api_url="http://localhost:8000", api_key=None):
        self.api_url = api_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key: self.headers["Authorization"] = f"Bearer {api_key}"

    async def chat(self, messages: List[Dict], **kwargs) -> Dict:
        async with aiohttp.ClientSession() as session:
            payload = {"messages": messages, **kwargs}
            async with session.post(f"{self.api_url}/v1/chat/completions",
                                    json=payload, headers=self.headers) as resp:
                return await resp.json()

    async def chat_stream(self, messages: List[Dict], **kwargs) -> AsyncGenerator[str, None]:
        async with aiohttp.ClientSession() as session:
            payload = {**kwargs, "messages": messages, "stream": True}
            async with session.post(f"{self.api_url}/v1/chat/completions",
                                    json=payload, headers=self.headers) as resp:
                import json
                async for line in resp.content:
                    line = line.decode().strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]": break
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta: yield delta["content"]
