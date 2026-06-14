import json, time, os, sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.model import VaelonModel, VaelonConfig
from core.tokenizer.bpe import BPETokenizer
from core.safety import SafetyModerator


class InferenceServer:
    """Qythera Inference Server with OpenAI-compatible API."""
    
    def __init__(self, config=None):
        self.config = config or VaelonConfig.small()
        self.model = VaelonModel(self.config)
        self.tokenizer = BPETokenizer()
        self.safety = SafetyModerator()
        self.request_count = 0
        self.start_time = time.time()
        print(f"  Model loaded: {sum(p.data.size for p in self.model.parameters()):,} parameters")

    def generate(self, messages, max_tokens=512, temperature=0.7, top_k=50, top_p=0.9):
        for m in messages:
            safe, result = self.safety.filter_input(m.get("content", ""))
            if not safe:
                return result

        prompt = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages) + "\n<|assistant|>\n"
        ids = self.tokenizer.encode(prompt, add_special=False)
        
        generated = VaelonModel.generate_ids(
            self.model, ids, max_new=max_tokens,
            temp=temperature, top_k=top_k, top_p=top_p
        )
        
        response = self.tokenizer.decode(generated, skip_special=True)
        # Remove prompt from response
        if prompt in response:
            response = response[len(prompt):]
        return response.strip()


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
                        "model": "vaelon", "backend": "numpy"})
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

            t0 = time.time()
            response = server_instance.generate(
                messages, max_tokens=body.get("max_tokens", 512),
                temperature=body.get("temperature", 0.7),
                top_k=body.get("top_k", 50),
                top_p=body.get("top_p", 0.9)
            )
            latency = time.time() - t0

            self._json({
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion",
                "model": "vaelon",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": len(response.split()), "total_tokens": len(response.split())},
                "latency_ms": round(latency * 1000, 1),
            })
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        pass


def run_server(host="0.0.0.0", port=8000):
    global server_instance
    print(f"\n  Qythera Inference Server")
    print(f"  Host: {host}:{port}")
    print(f"  Backend: Custom numpy autodiff engine")
    print(f"  API: http://{host}:{port}/v1/chat/completions")
    print(f"  Health: http://{host}:{port}/health\n")
    server_instance = InferenceServer()
    httpd = HTTPServer((host, port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    run_server(args.host, args.port)
