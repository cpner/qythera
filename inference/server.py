
import json, time, os, sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.config import Config, InferenceConfig
from core.model import VaelonModel
from core.tokenizer import Tokenizer
from core.safety import SafetyModerator

class QytheraServer:
    def __init__(self, config=None):
        self.config = config or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VaelonModel(self.config.model)
        self.model.to(self.device).eval()
        self.tokenizer = Tokenizer()
        self.safety = SafetyModerator()
        self.request_count = 0
        self.start_time = time.time()

    def generate(self, messages, max_tokens=512, temperature=0.7, top_k=50, top_p=0.9):
        for m in messages:
            safe, result = self.safety.filter_input(m.get("content", ""))
            if not safe: return result

        prompt = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages) + "\n<|assistant|>\n"
        ids = self.tokenizer.encode(prompt, add_special=False)
        input_t = torch.tensor([ids], device=self.device)
        output = self.model.generate(input_t, max_new=max_tokens, temp=temperature, top_k=top_k, top_p=top_p)
        response = self.tokenizer.decode(output[0].tolist(), skip_special=True)
        response = response[len(prompt):] if prompt in response else response
        return response.strip()

    def stream_generate(self, messages, **kwargs):
        full = self.generate(messages, **kwargs)
        words = full.split()
        for i, word in enumerate(words):
            chunk = word + " " if i < len(words) - 1 else word
            yield json.dumps({"choices": [{"delta": {"content": chunk}}]}) + "\n"

server_instance = None

class Handler(BaseHTTPRequestHandler):
    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            uptime = time.time() - server_instance.start_time
            self._json({"status": "ok", "uptime": uptime, "requests": server_instance.request_count,
                        "model": "vaelon", "device": str(server_instance.device)})
        elif self.path == "/v1/models":
            self._json({"data": [{"id": "vaelon", "object": "model", "owned_by": "qythera"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        server_instance.request_count += 1

        if self.path == "/v1/chat/completions":
            messages = body.get("messages", [])
            if not messages:
                self._json({"error": "messages required"}, 400)
                return
            stream = body.get("stream", False)
            gen_args = {"max_tokens": body.get("max_tokens", 512),
                        "temperature": body.get("temperature", 0.7),
                        "top_k": body.get("top_k", 50),
                        "top_p": body.get("top_p", 0.9)}

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                for chunk in server_instance.stream_generate(messages, **gen_args):
                    self.wfile.write(f"data: {chunk}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                t0 = time.time()
                response = server_instance.generate(messages, **gen_args)
                latency = time.time() - t0
                self._json({
                    "id": f"chatcmpl-{int(time.time()*1000)}",
                    "object": "chat.completion",
                    "model": "vaelon",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": sum(len(m.get("content","").split()) for m in messages),
                              "completion_tokens": len(response.split()),
                              "total_tokens": sum(len(m.get("content","").split()) for m in messages) + len(response.split())},
                    "latency_ms": round(latency * 1000, 1),
                })
        elif self.path == "/v1/embeddings":
            texts = body.get("input", [])
            if isinstance(texts, str): texts = [texts]
            embeddings = [[0.0]*384 for _ in texts]
            self._json({"data": [{"embedding": e, "index": i} for i, e in enumerate(embeddings)]})
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]}")

def run_server(host="0.0.0.0", port=8000):
    global server_instance
    print(f"\n  Qythera Inference Server")
    print(f"  Host: {host}:{port}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Model: Vaelon")
    print(f"\n  API: http://{host}:{port}/v1/chat/completions")
    print(f"  Health: http://{host}:{port}/health\n")
    server_instance = QytheraServer()
    httpd = HTTPServer((host, port), Handler)
    try: httpd.serve_forever()
    except KeyboardInterrupt: print("\nServer stopped.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    run_server(args.host, args.port)
