"""HTTP inference server with graceful shutdown and error handling."""

import json
import time
import os
import sys
import signal
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.knowledge.base import get_answer
from core.safety import SafetyModerator


class QytheraAI:
    def __init__(self):
        self.safety = SafetyModerator()
        self.request_count = 0
        self.start_time = time.time()
        print("  \033[32m✓\033[0m Knowledge base loaded")
        print("  \033[32m✓\033[0m Safety filters active")

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
    print("\n\033[33m  Shutting down server...\033[0m")
    if httpd:
        httpd.shutdown()
    print("\033[32m  Server stopped.\033[0m")
    sys.exit(0)


def find_free_port(start=8000, end=9000):
    """Find a free port in range."""
    for port in range(start, end):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", port))
            s.close()
            return port
        except OSError:
            continue
    return start


def run_server(host="0.0.0.0", port=8000):
    global server_instance, httpd

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Try to find a free port if requested port is busy
    actual_port = port
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.close()
    except OSError:
        print(f"\n  \033[33m⚠ Port {port} is busy, finding free port...\033[0m")
        actual_port = find_free_port(port + 1)
        print(f"  \033[32m✓ Using port {actual_port}\033[0m")

    print(f"\n  \033[35m╔══════════════════════════════════╗\033[0m")
    print(f"  \033[35m║      Qythera Inference Server     ║\033[0m")
    print(f"  \033[35m╚══════════════════════════════════╝\033[0m")
    print(f"  Host: {host}:{actual_port}")

    server_instance = QytheraAI()

    print(f"\n  \033[36mAPI:\033[0m    http://{host}:{actual_port}/v1/chat/completions")
    print(f"  \033[36mHealth:\033[0m  http://{host}:{actual_port}/health")
    print(f"\n  \033[33mPress Ctrl+C to stop\033[0m\n")

    try:
        httpd = HTTPServer((host, actual_port), Handler)
        httpd.serve_forever()
    except PermissionError:
        print(f"\n  \033[31mERROR: Permission denied on port {actual_port}\033[0m")
        print(f"  \033[33mSolutions:\033[0m")
        print(f"  1. Use a higher port: python -m core.inference.server --port 8080")
        print(f"  2. On restricted hosts: use a higher port")
        print(f"  3. On Linux: sudo python -m core.inference.server")
        sys.exit(1)
    except OSError as e:
        print(f"\n  \033[31mERROR: {e}\033[0m")
        print(f"  \033[33mTry: python -m core.inference.server --port 8080\033[0m")
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n  \033[32mServer stopped.\033[0m")
        if httpd:
            httpd.server_close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Qythera Inference Server")
    p.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    p.add_argument("--port", type=int, default=8000, help="Port (auto-finds free if busy)")
    args = p.parse_args()
    run_server(args.host, args.port)
