import json, time, os, sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.model import VaelonModel, VaelonConfig
from core.tokenizer.bpe import BPETokenizer
from core.safety import SafetyModerator
from core.intelligence import Intelligence


class InferenceServer:
    def __init__(self):
        self.config = VaelonConfig.small()
        self.model = VaelonModel(self.config)
        self.tokenizer = BPETokenizer()
        self.safety = SafetyModerator()
        self.ai = Intelligence()
        self.request_count = 0
        self.start_time = time.time()

    def generate(self, messages, max_tokens=512, temperature=0.7, top_k=50, top_p=0.9):
        for m in messages:
            safe, result = self.safety.filter_input(m.get("content", ""))
            if not safe:
                return result
        last_msg = messages[-1].get("content", "") if messages else ""
        history = messages[:-1] if len(messages) > 1 else None
        return self.ai.respond(last_msg, history)
    def _template_response(self, messages):
        last = messages[-1].get("content", "").lower() if messages else ""
        if any(w in last for w in ["hello", "hi", "hey", "привет"]):
            return "Hello! I'm Qythera, powered by Vaelon architecture. How can I help you?"
        if any(w in last for w in ["what", "how", "why", "explain", "что", "как"]):
            return "Great question! I'm Qythera with a custom autodiff engine and Vaelon transformer. My model is currently untrained, but once trained on data I'll provide intelligent answers. The infrastructure is fully functional: tensor engine, MoE transformer, tokenizer, memory, and safety filters are all working."
        if any(w in last for w in ["code", "python", "function", "код", "функция"]):
            return "I can help with coding! Example:\n\n```python\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\nprint(fibonacci(10))  # 55\n```\n\nOnce trained, I'll generate much more sophisticated code."
        if any(w in last for w in ["who", "what are you", "who are you", "кто ты"]):
            return "I'm Qythera — a production superintelligence platform. My Vaelon model uses Mixture of Experts with Grouped Query Attention and Rotary Position Embeddings. Everything is built from scratch: custom autodiff engine, BPE tokenizer, vector memory, safety filters. Train me and I'll answer your questions intelligently!"
        return "I'm Qythera! My Vaelon model is untrained but my system is fully functional. I have:\n\n- Custom autodiff tensor engine (numpy only)\n- Vaelon MoE transformer (37.5M params)\n- BPE tokenizer\n- Memory system (vector + episodic)\n- Safety filters (toxicity, jailbreak, PII)\n- Web UI with PWA support\n\nAsk me anything — once trained, I'll give you smart answers!"


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
            self._json({"status": "ok", "uptime": round(uptime, 1), "requests": server_instance.request_count,
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
            response = server_instance.generate(messages, max_tokens=body.get("max_tokens", 512),
                temperature=body.get("temperature", 0.7), top_k=body.get("top_k", 50), top_p=body.get("top_p", 0.9))
            latency = time.time() - t0
            self._json({"id": f"chatcmpl-{int(time.time()*1000)}", "object": "chat.completion", "model": "vaelon",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": len(response.split()), "total_tokens": len(response.split())},
                "latency_ms": round(latency * 1000, 1)})
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        pass


def run_server(host="0.0.0.0", port=8000):
    global server_instance
    print(f"\n  Qythera Inference Server")
    print(f"  Host: {host}:{port}")
    print(f"  Backend: Custom numpy autodiff engine")
    print(f"  Model: Vaelon (small, 37.5M params)")
    print(f"\n  API: http://{host}:{port}/v1/chat/completions")
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
