"""HTTP inference server with graceful shutdown."""

import json
import time
import os
import sys
import signal
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.knowledge.base import get_answer
from core.safety import SafetyModerator


class QytheraAI:
    def __init__(self):
        self.safety = SafetyModerator()
        self.request_count = 0
        self.start_time = time.time()
        print("  Knowledge base loaded")
        print("  Safety filters active")
    
    def generate(self, messages, **kwargs):
        for m in messages:
            safe, result = self.safety.filter_input(m.get("content", ""))
            if not safe:
                return result
        last_msg = messages[-1].get("content", "") if messages else ""
        return get_answer(last_msg)


server_instance = None
httpd = None


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
            self._json({"status": "ok", "uptime": round(time.time()-server_instance.start_time, 1),
                        "requests": server_instance.request_count, "model": "vaelon"})
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
            response = server_instance.generate(messages, **body)
            latency = time.time() - t0
            self._json({
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion",
                "model": "vaelon",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": len(response.split()), "total_tokens": len(response.split())},
                "latency_ms": round(latency*1000, 1)
            })
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        pass


def shutdown_handler(signum, frame):
    global httpd
    print("\n  Shutting down server...")
    if httpd:
        httpd.shutdown()
    print("  Server stopped.")
    sys.exit(0)


def run_server(host="0.0.0.0", port=8000):
    global server_instance, httpd
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    print("\n  Qythera Inference Server")
    print(f"  Host: {host}:{port}")
    server_instance = QytheraAI()
    print(f"\n  API: http://{host}:{port}/v1/chat/completions")
    print(f"  Health: http://{host}:{port}/health")
    print(f"\n  Press Ctrl+C to stop\n")
    
    httpd = HTTPServer((host, port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n  Server stopped.")
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Qythera Inference Server")
    p.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    p.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = p.parse_args()
    run_server(args.host, args.port)
