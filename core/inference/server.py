"""Qythera Inference Server - serves both API and web UI."""

import json
import time
import os
import sys
import signal
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.knowledge.base import get_answer
from core.safety import SafetyModerator

HTML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "web", "standalone.html")

server_instance = None
httpd = None


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


class Handler(BaseHTTPRequestHandler):
    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _html(self, content, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]

        # Serve web UI for root and common paths
        if path in ("/", "/index.html", "/index.php"):
            try:
                with open(HTML_PATH, "r", encoding="utf-8") as f:
                    self._html(f.read())
            except FileNotFoundError:
                self._html("<h1>Qythera</h1><p>Web UI not found. Run: python -m core.inference.server</p>")
            return

        if path == "/health":
            self._json({"status": "ok", "uptime": round(time.time()-server_instance.start_time, 1),
                        "requests": server_instance.request_count, "model": "vaelon"})
        elif path == "/v1/models":
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
    print("\n\033[33m  Shutting down...\033[0m")
    if httpd:
        httpd.shutdown()
    sys.exit(0)


def run_server(host="0.0.0.0", port=8000):
    global server_instance, httpd

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Find free port
    actual_port = port
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.close()
    except OSError:
        print(f"\n  \033[33mPort {port} busy, finding free port...\033[0m")
        for p in range(port+1, port+100):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, p))
                s.close()
                actual_port = p
                break
            except OSError:
                continue

    print(f"\n  \033[35m╔══════════════════════════════════╗\033[0m")
    print(f"  \033[35m║      Qythera Inference Server     ║\033[0m")
    print(f"  \033[35m╚══════════════════════════════════╝\033[0m")
    print(f"  Host: {host}:{actual_port}")

    server_instance = QytheraAI()

    print(f"\n  \033[36mOpen in browser:\033[0m  http://localhost:{actual_port}")
    print(f"  \033[36mAPI:\033[0m             http://localhost:{actual_port}/v1/chat/completions")
    print(f"  \033[36mHealth:\033[0m          http://localhost:{actual_port}/health")
    print(f"\n  \033[33mPress Ctrl+C to stop\033[0m\n")

    try:
        httpd = HTTPServer((host, actual_port), Handler)
        httpd.serve_forever()
    except PermissionError:
        print(f"\n  \033[31mERROR: Permission denied\033[0m")
        print(f"  Try: python -m core.inference.server --port 9000")
        sys.exit(1)
    except OSError as e:
        print(f"\n  \033[31mERROR: {e}\033[0m")
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
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    run_server(args.host, args.port)
