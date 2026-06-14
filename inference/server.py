import json, time, os, sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.llm import Tokenizer, SmallTransformer, Trainer, TRAINING_CORPUS
from core.safety import SafetyModerator


class QytheraAI:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.model_path = "models/vaelon_small"
        
        if os.path.exists(os.path.join(self.model_path, "model.npz")):
            print("  Loading trained model...")
            self.model = SmallTransformer.load(self.model_path)
            self.tokenizer.load(os.path.join(self.model_path, "tokenizer.json"))
            print(f"  Model loaded: {self.model.num_params:,} params")
        else:
            print("  Training new model (first run)...")
            self.model = SmallTransformer(
                vocab_size=self.tokenizer.vocab_size,
                d_model=64, n_heads=4, n_layers=2, d_ff=128
            )
            trainer = Trainer(self.model, self.tokenizer, lr=0.005)
            trainer.train_on_text(TRAINING_CORPUS, epochs=10, seq_len=64, verbose=True)
            os.makedirs(self.model_path, exist_ok=True)
            self.model.save(self.model_path)
            self.tokenizer.save(os.path.join(self.model_path, "tokenizer.json"))
            print(f"  Model trained and saved: {self.model.num_params:,} params")
        
        self.safety = SafetyModerator()
        self.request_count = 0
        self.start_time = time.time()
    
    def generate(self, messages, max_tokens=256, temperature=0.7):
        for m in messages:
            safe, result = self.safety.filter_input(m.get("content", ""))
            if not safe:
                return result
        
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT: "
        ids = self.tokenizer.encode(prompt)
        output = self.model.generate(ids, max_new=min(max_tokens, 200), temperature=temperature)
        response = self.tokenizer.decode(output)
        
        # Clean up response
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        if "USER:" in response:
            response = response.split("USER:")[0].strip()
        
        return response.strip() if response.strip() else "I'm still learning. Please try again."


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
                        "model": "vaelon", "params": server_instance.model.num_params, "backend": "numpy"})
        elif self.path == "/v1/models":
            self._json({"data": [{"id": "vaelon", "object": "model", "owned_by": "qythera",
                                  "params": server_instance.model.num_params}]})
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
            response = server_instance.generate(messages, max_tokens=body.get("max_tokens", 256),
                temperature=body.get("temperature", 0.7))
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
    print("\n  Qythera Inference Server")
    print(f"  Host: {host}:{port}")
    server_instance = QytheraAI()
    print(f"  Model: {server_instance.model.num_params:,} params")
    print(f"\n  API: http://{host}:{port}/v1/chat/completions")
    print(f"  Health: http://{host}:{port}/health\n")
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
