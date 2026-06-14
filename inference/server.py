import os
import json
import time
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from vaelon.config import VaelonConfig
from vaelon.model import VaelonModel
from vaelon.tokenizer import VaelonTokenizer


class InferenceServer:
    def __init__(self, model_path: Optional[str] = None, host: str = "0.0.0.0",
                 port: int = 8000, device: str = "auto"):
        self.host = host
        self.port = port
        self.config = VaelonConfig.vaelon_7b()
        self.model = VaelonModel(self.config)
        self.tokenizer = VaelonTokenizer()
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() and HAS_TORCH else "cpu")
        else:
            self.device = torch.device(device)
        if HAS_TORCH:
            self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7,
                 top_k: int = 50, top_p: float = 0.9) -> str:
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=self.device)
        with torch.no_grad():
            output = self.model.generate(
                input_tensor, max_new_tokens=max_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p,
            )
        return self.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

    def stream_generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7):
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=self.device)
        generated = input_ids.copy()
        with torch.no_grad():
            for _ in range(max_tokens):
                tensor = torch.tensor([generated], device=self.device)
                output = self.model(tensor)
                next_logits = output.logits[:, -1, :] / max(temperature, 1e-7)
                next_token = torch.argmax(next_logits, dim=-1).item()
                generated.append(next_token)
                yield self.tokenizer.decode([next_token], skip_special_tokens=True)

    def start(self):
        server = HTTPServer((self.host, self.port), self._create_handler())
        print(f"Qythera Inference Server running on http://{self.host}:{self.port}")
        server.serve_forever()

    def _create_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))

                if self.path == "/v1/chat/completions":
                    messages = body.get("messages", [])
                    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                    prompt += "\nassistant:"
                    response_text = server.generate(
                        prompt, max_tokens=body.get("max_tokens", 512),
                        temperature=body.get("temperature", 0.7),
                    )
                    result = {
                        "choices": [{"message": {"role": "assistant", "content": response_text}}],
                        "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(response_text.split())},
                    }
                elif self.path == "/v1/models":
                    result = {"data": [{"id": "vaelon-7b", "object": "model"}]}
                else:
                    result = {"error": "Not found"}

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())

            def do_GET(self):
                if self.path == "/health":
                    result = {"status": "ok"}
                else:
                    result = {"error": "Not found"}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())

            def log_message(self, format, *args):
                pass

        return Handler


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    server = InferenceServer(model_path=args.model, host=args.host, port=args.port)
    server.start()
