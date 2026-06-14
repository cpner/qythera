import os


def run_serve(model=None, port=8000, host="0.0.0.0"):
    print(f"\n  Starting Qythera Inference Server")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    if model:
        print(f"  Model: {model}\n")

    os.environ["VAELEN_MODEL_PATH"] = model or ""

    try:
        from inference.server import InferenceServer
        server = InferenceServer(model_path=model, host=host, port=port)
        server.start()
    except ImportError:
        print("Starting simple server...")
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "model": "vaelon"}).encode())

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                resp = {"choices": [{"message": {"role": "assistant", "content": "Qythera server running (basic mode)."}}]}
                self.wfile.write(json.dumps(resp).encode())

            def log_message(self, format, *args):
                pass

        server = HTTPServer((host, port), Handler)
        print(f"Server running at http://{host}:{port}")
        server.serve_forever()
