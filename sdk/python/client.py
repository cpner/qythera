import requests
from typing import List, Dict, Optional, Generator

class QytheraClient:
    def __init__(self, api_url="http://localhost:8000", api_key=None):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def chat(self, messages: List[Dict], model="vaelon-7b", temperature=0.7,
             max_tokens=2048, **kwargs) -> Dict:
        payload = {"model": model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, **kwargs}
        resp = requests.post(f"{self.api_url}/v1/chat/completions",
                             json=payload, headers=self.headers, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def chat_stream(self, messages: List[Dict], **kwargs) -> Generator[str, None, None]:
        payload = {**kwargs, "messages": messages, "stream": True}
        resp = requests.post(f"{self.api_url}/v1/chat/completions",
                             json=payload, headers=self.headers, stream=True, timeout=120)
        for line in resp.iter_lines():
            if line and line.startswith(b"data: "):
                data = line[6:]
                if data == b"[DONE]": break
                import json
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta:
                    yield delta["content"]

    def models(self) -> List[Dict]:
        resp = requests.get(f"{self.api_url}/v1/models", headers=self.headers)
        return resp.json().get("data", [])

    def health(self) -> bool:
        try:
            resp = requests.get(f"{self.api_url}/health", timeout=5)
            return resp.status_code == 200
        except: return False

    def generate(self, prompt: str, **kwargs) -> str:
        result = self.chat([{"role": "user", "content": prompt}], **kwargs)
        return result["choices"][0]["message"]["content"]

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = requests.post(f"{self.api_url}/v1/embeddings",
                             json={"input": texts}, headers=self.headers)
        return [item["embedding"] for item in resp.json().get("data", [])]
