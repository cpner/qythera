"""Raw socket-based HTTP server for Qythera. Pure Python + NumPy."""
import json
import os
import signal
import socket
import sys
import time
import threading
import queue
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from qythera.model import Transformer, TransformerConfig
    from qythera.tokenizer import BPETokenizer
except Exception:
    Transformer = None
    TransformerConfig = None
    BPETokenizer = None

BASE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# RequestParser
# ---------------------------------------------------------------------------

class RequestParser:
    """Parse raw HTTP request, extract JSON body."""

    @staticmethod
    def parse(raw: bytes) -> Dict[str, Any]:
        try:
            text = raw.decode('utf-8', errors='replace')
            header_end = text.find('\r\n\r\n')
            if header_end == -1:
                header_end = text.find('\n\n')
                if header_end == -1:
                    return {"error": "malformed request", "headers": {}, "body": {}, "method": "", "path": ""}
                raw_headers = text[:header_end]
                body_text = text[header_end + 2:]
            else:
                raw_headers = text[:header_end]
                body_text = text[header_end + 4:]

            lines = raw_headers.split('\r\n')
            if not lines:
                lines = raw_headers.split('\n')

            request_line = lines[0] if lines else ""
            parts = request_line.split()
            method = parts[0] if len(parts) > 0 else ""
            path = parts[1] if len(parts) > 1 else "/"

            headers = {}
            for line in lines[1:]:
                if ':' in line:
                    k, v = line.split(':', 1)
                    headers[k.strip().lower()] = v.strip()

            body = {}
            if body_text.strip():
                try:
                    body = json.loads(body_text)
                except json.JSONDecodeError:
                    body = {"raw": body_text}

            return {"method": method, "path": path, "headers": headers, "body": body}
        except Exception as e:
            return {"error": str(e), "headers": {}, "body": {}, "method": "", "path": ""}


# ---------------------------------------------------------------------------
# ResponseBuilder
# ---------------------------------------------------------------------------

class ResponseBuilder:
    """Build HTTP responses with proper headers."""

    @staticmethod
    def json(data: Any, status: int = 200) -> bytes:
        body = json.dumps(data).encode('utf-8')
        headers = (
            f"HTTP/1.1 {status} {'OK' if status == 200 else 'Error'}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"Access-Control-Allow-Methods: GET,POST,OPTIONS\r\n"
            f"Access-Control-Allow-Headers: Content-Type\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        return headers.encode('utf-8') + body

    @staticmethod
    def html(content: str, status: int = 200) -> bytes:
        body = content.encode('utf-8')
        headers = (
            f"HTTP/1.1 {status} OK\r\n"
            f"Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        return headers.encode('utf-8') + body

    @staticmethod
    def sse(event: str, data: str) -> bytes:
        body = f"event: {event}\ndata: {data}\n\n".encode('utf-8')
        headers = (
            f"HTTP/1.1 200 OK\r\n"
            f"Content-Type: text/event-stream\r\n"
            f"Cache-Control: no-cache\r\n"
            f"Connection: keep-alive\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"\r\n"
        )
        return headers.encode('utf-8') + body

    @staticmethod
    def not_found() -> bytes:
        return ResponseBuilder.json({"error": "not found"}, 404)

    @staticmethod
    def bad_request(msg: str = "bad request") -> bytes:
        return ResponseBuilder.json({"error": msg}, 400)


# ---------------------------------------------------------------------------
# StreamingResponse
# ---------------------------------------------------------------------------

class StreamingResponse:
    """SSE format for streaming output."""

    def __init__(self, conn: socket.socket):
        self.conn = conn

    def send_event(self, event: str, data: str):
        payload = f"event: {event}\ndata: {data}\n\n".encode('utf-8')
        try:
            self.conn.sendall(payload)
        except Exception:
            pass

    def send_token(self, token: str):
        self.send_event("token", json.dumps({"token": token}))

    def send_done(self):
        self.send_event("done", json.dumps({"finished": True}))

    def send_error(self, msg: str):
        self.send_event("error", json.dumps({"error": msg}))


# ---------------------------------------------------------------------------
# HealthCheck
# ---------------------------------------------------------------------------

class HealthCheck:
    """Track uptime, memory, latency stats."""

    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.total_latency_ms = 0.0
        self.error_count = 0

    def record_request(self, latency_ms: float, error: bool = False):
        self.request_count += 1
        self.total_latency_ms += latency_ms
        if error:
            self.error_count += 1

    def get_stats(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        avg_latency = self.total_latency_ms / max(self.request_count, 1)
        return {
            "status": "ok",
            "uptime_seconds": round(uptime, 1),
            "requests": self.request_count,
            "errors": self.error_count,
            "avg_latency_ms": round(avg_latency, 2),
            "model": "vaelon",
        }


# ---------------------------------------------------------------------------
# BatchScheduler
# ---------------------------------------------------------------------------

class BatchScheduler:
    """Continuous batching with slot management."""

    def __init__(self, max_slots: int = 4):
        self.max_slots = max_slots
        self.active_slots: Dict[int, Dict[str, Any]] = {}
        self.request_queue: queue.Queue = queue.Queue()
        self._slot_id = 0

    def acquire_slot(self) -> Optional[int]:
        if len(self.active_slots) < self.max_slots:
            self._slot_id += 1
            sid = self._slot_id
            self.active_slots[sid] = {"created": time.time(), "status": "active"}
            return sid
        return None

    def release_slot(self, slot_id: int):
        self.active_slots.pop(slot_id, None)

    def add_request(self, request: Dict[str, Any]) -> int:
        self._slot_id += 1
        rid = self._slot_id
        self.request_queue.put({"id": rid, **request})
        return rid

    def get_stats(self) -> Dict[str, Any]:
        return {
            "active_slots": len(self.active_slots),
            "max_slots": self.max_slots,
            "queue_size": self.request_queue.qsize(),
        }


# ---------------------------------------------------------------------------
# RouteHandler
# ---------------------------------------------------------------------------

class RouteHandler:
    """Route HTTP requests to handlers."""

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.health = HealthCheck()
        self.batch = BatchScheduler()
        self.routes: Dict[Tuple[str, str], Callable] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.routes[("GET", "/")] = self._serve_ui
        self.routes[("GET", "/health")] = self._health_check
        self.routes[("GET", "/v1/models")] = self._list_models
        self.routes[("POST", "/v1/chat/completions")] = self._chat_completions

    def handle(self, request: Dict[str, Any]) -> bytes:
        method = request.get("method", "")
        path = request.get("path", "/").split("?")[0]

        if method == "OPTIONS":
            return ResponseBuilder.json({}, 200)

        handler = self.routes.get((method, path))
        if handler:
            return handler(request)

        if path.startswith("/icon-") and path.endswith(".png"):
            return self._serve_static(path.lstrip("/"), "image/png")

        return ResponseBuilder.not_found()

    def _serve_ui(self, request: Dict) -> bytes:
        ui_path = os.path.join(os.path.dirname(BASE), "web", "ui.html")
        try:
            with open(ui_path, 'r') as f:
                return ResponseBuilder.html(f.read())
        except FileNotFoundError:
            return ResponseBuilder.html("<h1>Qythera</h1><p>Server running.</p>")

    def _health_check(self, request: Dict) -> bytes:
        return ResponseBuilder.json(self.health.get_stats())

    def _list_models(self, request: Dict) -> bytes:
        return ResponseBuilder.json({"data": [{"id": "vaelon", "object": "model"}]})

    def _chat_completions(self, request: Dict) -> bytes:
        body = request.get("body", {})
        messages = body.get("messages", [])
        if not messages:
            return ResponseBuilder.bad_request("messages required")

        stream = body.get("stream", False)
        prompt = messages[-1].get("content", "")

        t0 = time.time()
        slot_id = self.batch.acquire_slot()

        try:
            if stream:
                return ResponseBuilder.json({
                    "id": f"c{int(time.time()*1000)}",
                    "object": "chat.completion",
                    "model": body.get("model", "vaelon"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                    "streaming": True,
                })

            response_text = self._generate_response(prompt)
            latency = time.time() - t0
            self.health.record_request(latency * 1000)

            return ResponseBuilder.json({
                "id": f"c{int(time.time()*1000)}",
                "object": "chat.completion",
                "model": body.get("model", "vaelon"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }],
                "latency_ms": round(latency * 1000, 1),
            })
        except Exception as e:
            self.health.record_request(0, error=True)
            return ResponseBuilder.json({"error": str(e)}, 500)
        finally:
            if slot_id is not None:
                self.batch.release_slot(slot_id)

    def _generate_response(self, prompt: str) -> str:
        if self.model is not None and self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(prompt)
                input_arr = np.array([tokens], dtype=np.int32) if hasattr(tokens, '__iter__') else np.array([[tokens]], dtype=np.int32)
                from qythera.tensor import Tensor
                inp = Tensor(input_arr)
                output = self.model.forward(inp)
                if hasattr(output, 'data'):
                    logits = output.data
                    if logits.ndim == 3:
                        last_logits = logits[0, -1, :]
                    else:
                        last_logits = logits.flatten()
                    token_id = int(np.argmax(last_logits))
                    return self.tokenizer.decode([token_id])
                return str(output)
            except Exception as e:
                return f"Model error: {e}"
        return f"Qythera received: {prompt}"

    def _serve_static(self, path: str, content_type: str) -> bytes:
        # Serve icons from the root icons/ directory
        icons_dir = os.path.join(os.path.dirname(os.path.dirname(BASE)), "icons")
        file_path = os.path.join(icons_dir, path)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = f.read()
            headers = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: {content_type}\r\n"
                f"Content-Length: {len(data)}\r\n"
                f"Access-Control-Allow-Origin: *\r\n"
                f"Cache-Control: public, max-age=86400\r\n"
                f"\r\n"
            )
            return headers.encode('utf-8') + data
        return ResponseBuilder.not_found()


# ---------------------------------------------------------------------------
# RawSocketHTTPServer
# ---------------------------------------------------------------------------

class RawSocketHTTPServer:
    """Socket-based HTTP/1.1 server."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000, model=None, tokenizer=None):
        self.host = host
        self.port = port
        self.handler = RouteHandler(model=model, tokenizer=tokenizer)
        self._server_socket: Optional[socket.socket] = None
        self._running = False

    def _find_available_port(self) -> int:
        p = self.port
        for offset in range(100):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.host, p + offset))
                s.close()
                return p + offset
            except OSError:
                continue
        raise RuntimeError(f"No available port in range {self.port}-{self.port + 99}")

    def start(self):
        port = self._find_available_port()
        self.port = port
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, port))
        self._server_socket.listen(16)
        self._server_socket.settimeout(1.0)
        self._running = True

        print(f"\n  Qythera: http://localhost:{port}")

        signal.signal(signal.SIGINT, lambda s, f: self.stop())
        signal.signal(signal.SIGTERM, lambda s, f: self.stop())

        while self._running:
            try:
                conn, addr = self._server_socket.accept()
                t = threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True)
                t.start()
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_client(self, conn: socket.socket, addr: Tuple):
        try:
            conn.settimeout(30)
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\r\n\r\n" in data or b"\n\n" in data:
                    if b"Content-Length" in data:
                        header_end = data.find(b"\r\n\r\n")
                        if header_end == -1:
                            header_end = data.find(b"\n\n")
                        header_text = data[:header_end].decode('utf-8', errors='replace')
                        content_length = 0
                        for line in header_text.split('\r\n'):
                            if line.lower().startswith('content-length:'):
                                content_length = int(line.split(':', 1)[1].strip())
                                break
                        expected = header_end + 4 + content_length
                        while len(data) < expected:
                            chunk = conn.recv(min(4096, expected - len(data)))
                            if not chunk:
                                break
                            data += chunk
                    break

            if not data:
                conn.close()
                return

            request = RequestParser.parse(data)
            response = self.handler.handle(request)
            conn.sendall(response)
        except Exception as e:
            try:
                conn.sendall(ResponseBuilder.json({"error": str(e)}, 500))
            except Exception:
                pass
        finally:
            conn.close()

    def stop(self):
        self._running = False
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
        print("\nStopped.")
        sys.exit(0)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qythera Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    args = parser.parse_args()

    model = None
    tokenizer = None

    if args.model_path and Transformer is not None:
        try:
            cfg = TransformerConfig()
            model = Transformer(cfg)
            if os.path.exists(args.model_path):
                from qythera.tensor import Tensor
                state = np.load(args.model_path, allow_pickle=True).item()
                for name, param in model.parameters():
                    if name in state:
                        param.data = np.array(state[name], dtype=np.float32).reshape(param.shape)
            print(f"  Loaded model from {args.model_path}")
        except Exception as e:
            print(f"  Warning: Could not load model: {e}")

    if args.tokenizer_path and BPETokenizer is not None:
        try:
            tokenizer = BPETokenizer()
            if os.path.exists(args.tokenizer_path):
                tokenizer.load(args.tokenizer_path)
            print(f"  Loaded tokenizer from {args.tokenizer_path}")
        except Exception as e:
            print(f"  Warning: Could not load tokenizer: {e}")

    server = RawSocketHTTPServer(
        host=args.host,
        port=args.port,
        model=model,
        tokenizer=tokenizer,
    )
    server.start()


if __name__ == "__main__":
    main()
