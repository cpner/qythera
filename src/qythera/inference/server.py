"""Raw socket-based HTTP server for Qythera. Pure Python + NumPy."""
import json
import os
import signal
import socket
import sys
import time
import threading
import queue
import hashlib
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from collections import deque

import numpy as np

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
# CircuitBreaker
# ---------------------------------------------------------------------------

class _CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker: open after N failures, half-open after cooldown."""

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30.0,
                 half_open_max: int = 1):
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._half_open_max = half_open_max
        self._state = _CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_attempts = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> _CircuitState:
        with self._lock:
            if self._state == _CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self._cooldown:
                    self._state = _CircuitState.HALF_OPEN
                    self._half_open_attempts = 0
            return self._state

    def record_success(self):
        with self._lock:
            if self._state == _CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._half_open_max:
                    self._state = _CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == _CircuitState.CLOSED:
                self._failure_count = 0

    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._state == _CircuitState.HALF_OPEN:
                self._state = _CircuitState.OPEN
                self._success_count = 0
            elif self._failure_count >= self._failure_threshold:
                self._state = _CircuitState.OPEN

    def allow_request(self) -> bool:
        state = self.state
        if state == _CircuitState.CLOSED:
            return True
        if state == _CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_attempts < self._half_open_max:
                    self._half_open_attempts += 1
                    return True
            return False
        return False

    def reset(self):
        with self._lock:
            self._state = _CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_attempts = 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self._failure_threshold,
            "cooldown_seconds": self._cooldown,
        }


# ---------------------------------------------------------------------------
# RequestDeduplication
# ---------------------------------------------------------------------------

class RequestDeduplication:
    """Cache identical prompts to avoid redundant inference."""

    def __init__(self, max_size: int = 128, ttl_seconds: float = 300.0):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_order: deque = deque()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, messages: List[Dict], model: str, temperature: float,
                  max_tokens: int) -> str:
        payload = json.dumps({
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get(self, messages: List[Dict], model: str, temperature: float,
            max_tokens: int) -> Optional[Any]:
        key = self._make_key(messages, model, temperature, max_tokens)
        with self._lock:
            if key in self._cache:
                value, ts = self._cache[key]
                if time.time() - ts < self._ttl:
                    self._hits += 1
                    return value
                else:
                    del self._cache[key]
                    self._access_order.remove(key)
            self._misses += 1
        return None

    def put(self, messages: List[Dict], model: str, temperature: float,
            max_tokens: int, value: Any):
        key = self._make_key(messages, model, temperature, max_tokens)
        with self._lock:
            if len(self._cache) >= self._max_size:
                oldest = self._access_order.popleft()
                self._cache.pop(oldest, None)
            self._cache[key] = (value, time.time())
            self._access_order.append(key)

    def invalidate(self, messages: List[Dict], model: str, temperature: float,
                   max_tokens: int):
        key = self._make_key(messages, model, temperature, max_tokens)
        with self._lock:
            self._cache.pop(key, None)
            try:
                self._access_order.remove(key)
            except ValueError:
                pass

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / max(total, 1), 3),
            }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class Metrics:
    """Track tokens/sec, TTFT, latency distribution."""

    def __init__(self, window_size: int = 1000):
        self._window_size = window_size
        self._latencies: deque = deque(maxlen=window_size)
        self._ttfts: deque = deque(maxlen=window_size)
        self._tokens_per_sec: deque = deque(maxlen=window_size)
        self._total_tokens = 0
        self._total_requests = 0
        self._total_latency_ms = 0.0
        self._start_time = time.time()
        self._lock = threading.Lock()

    def record_request(self, latency_ms: float, ttft_ms: Optional[float] = None,
                       tokens_generated: int = 0, generation_time_ms: float = 0.0):
        with self._lock:
            self._latencies.append(latency_ms)
            self._total_requests += 1
            self._total_latency_ms += latency_ms

            if ttft_ms is not None:
                self._ttfts.append(ttft_ms)

            if tokens_generated > 0 and generation_time_ms > 0:
                tps = tokens_generated / (generation_time_ms / 1000.0)
                self._tokens_per_sec.append(tps)
                self._total_tokens += tokens_generated

    def _percentile(self, data: deque, p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            uptime = time.time() - self._start_time
            avg_lat = self._total_latency_ms / max(self._total_requests, 1)
            avg_tps = statistics.mean(self._tokens_per_sec) if self._tokens_per_sec else 0.0

            return {
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "uptime_seconds": round(uptime, 1),
                "avg_latency_ms": round(avg_lat, 2),
                "p50_latency_ms": round(self._percentile(self._latencies, 0.5), 2),
                "p95_latency_ms": round(self._percentile(self._latencies, 0.95), 2),
                "p99_latency_ms": round(self._percentile(self._latencies, 0.99), 2),
                "avg_ttft_ms": round(statistics.mean(self._ttfts), 2) if self._ttfts else 0.0,
                "p50_ttft_ms": round(self._percentile(self._ttfts, 0.5), 2),
                "avg_tokens_per_sec": round(avg_tps, 2),
                "requests_per_second": round(self._total_requests / max(uptime, 1), 2),
            }

    def reset(self):
        with self._lock:
            self._latencies.clear()
            self._ttfts.clear()
            self._tokens_per_sec.clear()
            self._total_tokens = 0
            self._total_requests = 0
            self._total_latency_ms = 0.0
            self._start_time = time.time()


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
        self.circuit = CircuitBreaker()
        self.dedup = RequestDeduplication()
        self.metrics = Metrics()
        self.routes: Dict[Tuple[str, str], Callable] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.routes[("GET", "/")] = self._serve_ui
        self.routes[("GET", "/health")] = self._health_check
        self.routes[("GET", "/v1/models")] = self._list_models
        self.routes[("POST", "/v1/chat/completions")] = self._chat_completions
        self.routes[("GET", "/metrics")] = self._metrics

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

    def _metrics(self, request: Dict) -> bytes:
        return ResponseBuilder.json({
            "health": self.health.get_stats(),
            "circuit_breaker": self.circuit.get_stats(),
            "dedup": self.dedup.get_stats(),
            "metrics": self.metrics.get_stats(),
            "batch": self.batch.get_stats(),
        })

    def _list_models(self, request: Dict) -> bytes:
        return ResponseBuilder.json({"data": [{"id": "vaelon", "object": "model"}]})

    def _chat_completions(self, request: Dict) -> bytes:
        body = request.get("body", {})
        messages = body.get("messages", [])
        if not messages:
            return ResponseBuilder.bad_request("messages required")

        if not self.circuit.allow_request():
            self.circuit.record_failure()
            return ResponseBuilder.json({"error": "circuit breaker open, service degraded"}, 503)

        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 1.0)
        top_k = body.get("top_k", 50)

        model_name = body.get("model", "vaelon")

        if not stream and temperature > 0:
            cached = self.dedup.get(messages, model_name, temperature, max_tokens)
            if cached is not None:
                self.circuit.record_success()
                return ResponseBuilder.json(cached)

        conv_id = body.get("conversation_id", f"conv_{int(time.time()*1000)}")
        history = self._get_conversation(conv_id)
        for msg in messages:
            role = msg.get("role", "user")
            content_text = msg.get("content", "")
            if history and history[-1]["role"] == role and role == "user":
                history[-1]["content"] = content_text
            else:
                history.append({"role": role, "content": content_text})
        self._save_conversation(conv_id, history)

        t0 = time.time()
        slot_id = self.batch.acquire_slot()

        try:
            if stream:
                return self._stream_chat(conv_id, history, max_tokens,
                                         temperature, top_k, t0, slot_id, body)
            else:
                gen_start = time.time()
                response_text = self._generate_autoregressive(
                    history, max_tokens, temperature, top_k)
                gen_time_ms = (time.time() - gen_start) * 1000
                latency = time.time() - t0
                latency_ms = latency * 1000
                self.health.record_request(latency_ms)
                self.circuit.record_success()
                self.metrics.record_request(latency_ms, tokens_generated=len(response_text.split()), generation_time_ms=gen_time_ms)
                history.append({"role": "assistant", "content": response_text})
                self._save_conversation(conv_id, history)
                resp = {
                    "id": f"c{int(time.time()*1000)}",
                    "object": "chat.completion",
                    "model": model_name,
                    "conversation_id": conv_id,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }],
                    "latency_ms": round(latency_ms, 1),
                }
                if temperature > 0:
                    self.dedup.put(messages, model_name, temperature, max_tokens, resp)
                return ResponseBuilder.json(resp)
        except Exception as e:
            self.health.record_request(0, error=True)
            self.circuit.record_failure()
            return ResponseBuilder.json({"error": str(e)}, 500)
        finally:
            if slot_id is not None:
                self.batch.release_slot(slot_id)

    def _get_conversation(self, conv_id: str) -> List[Dict]:
        if not hasattr(self, '_conversations'):
            self._conversations = {}
        return self._conversations.setdefault(conv_id, [])

    def _save_conversation(self, conv_id: str, history: List[Dict]):
        if not hasattr(self, '_conversations'):
            self._conversations = {}
        self._conversations[conv_id] = history

    def _format_prompt(self, history: List[Dict]) -> str:
        parts = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def _generate_autoregressive(self, history: List[Dict], max_tokens: int = 256,
                                  temperature: float = 1.0, top_k: int = 50) -> str:
        if self.model is None or self.tokenizer is None:
            prompt = history[-1].get("content", "")
            return f"Qythera received: {prompt}"

        try:
            from qythera.tensor import Tensor
            full_prompt = self._format_prompt(history)
            tokens = self.tokenizer.encode(full_prompt)
            if isinstance(tokens, int):
                tokens = [tokens]

            generated_tokens = []
            for _ in range(max_tokens):
                input_arr = np.array([tokens], dtype=np.int32)
                inp = Tensor(input_arr)
                output = self.model.forward(inp)
                if not hasattr(output, 'data'):
                    break
                logits = output.data
                if logits.ndim == 3:
                    last_logits = logits[0, -1, :]
                else:
                    last_logits = logits.flatten()

                if temperature > 0:
                    scaled = last_logits / temperature
                    top_k_idx = np.argpartition(scaled, -top_k)[-top_k:]
                    mask = np.full_like(scaled, -1e9)
                    mask[top_k_idx] = scaled[top_k_idx]
                    exp_logits = np.exp(mask - mask.max())
                    probs = exp_logits / exp_logits.sum()
                    token_id = int(np.random.choice(len(probs), p=probs))
                else:
                    token_id = int(np.argmax(last_logits))

                if hasattr(self.tokenizer, 'eos_token_id') and                    token_id == self.tokenizer.eos_token_id:
                    break
                generated_tokens.append(token_id)
                tokens.append(token_id)

            if generated_tokens:
                return self.tokenizer.decode(generated_tokens)
            return ""
        except Exception as e:
            return f"Model error: {e}"

    def _stream_chat(self, conv_id, history, max_tokens, temperature, top_k,
                     t0, slot_id, body):
        from qythera.tensor import Tensor
        gen_id = f"c{int(time.time()*1000)}"

        try:
            conn = None
            full_prompt = self._format_prompt(history)
            tokens = self.tokenizer.encode(full_prompt)
            if isinstance(tokens, int):
                tokens = [tokens]

            generated_tokens = []
            for i in range(max_tokens):
                input_arr = np.array([tokens], dtype=np.int32)
                inp = Tensor(input_arr)
                output = self.model.forward(inp)
                if not hasattr(output, 'data'):
                    break
                logits = output.data
                if logits.ndim == 3:
                    last_logits = logits[0, -1, :]
                else:
                    last_logits = logits.flatten()

                if temperature > 0:
                    scaled = last_logits / temperature
                    top_k_idx = np.argpartition(scaled, -top_k)[-top_k:]
                    mask = np.full_like(scaled, -1e9)
                    mask[top_k_idx] = scaled[top_k_idx]
                    exp_logits = np.exp(mask - mask.max())
                    probs = exp_logits / exp_logits.sum()
                    token_id = int(np.random.choice(len(probs), p=probs))
                else:
                    token_id = int(np.argmax(last_logits))

                if hasattr(self.tokenizer, 'eos_token_id') and                    token_id == self.tokenizer.eos_token_id:
                    break
                generated_tokens.append(token_id)
                tokens.append(token_id)

                token_text = self.tokenizer.decode([token_id])
                chunk = json.dumps({
                    "id": gen_id,
                    "object": "chat.completion.chunk",
                    "model": body.get("model", "vaelon"),
                    "conversation_id": conv_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token_text},
                        "finish_reason": None,
                    }],
                })
                sse_payload = f"data: {chunk}\n\n".encode("utf-8")
                yield sse_payload

            done_chunk = json.dumps({
                "id": gen_id,
                "object": "chat.completion.chunk",
                "model": body.get("model", "vaelon"),
                "conversation_id": conv_id,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            })
            yield f"data: {done_chunk}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

            latency = time.time() - t0
            self.health.record_request(latency * 1000)
            if generated_tokens:
                full_response = self.tokenizer.decode(generated_tokens)
                history.append({"role": "assistant", "content": full_response})
                self._save_conversation(conv_id, history)
        except Exception as e:
            self.health.record_request(0, error=True)
            err = json.dumps({"error": str(e)})
            yield f"data: {err}\n\n".encode("utf-8")
        finally:
            if slot_id is not None:
                self.batch.release_slot(slot_id)

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
        if sys.platform != 'win32':
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
            if hasattr(response, '__iter__') and not isinstance(response, (bytes, dict)):
                header = (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/event-stream\r\n"
                    b"Cache-Control: no-cache\r\n"
                    b"Connection: keep-alive\r\n"
                    b"Access-Control-Allow-Origin: *\r\n"
                    b"Transfer-Encoding: chunked\r\n"
                    b"\r\n"
                )
                conn.sendall(header)
                for chunk in response:
                    size = f"{len(chunk):x}\r\n".encode("utf-8")
                    conn.sendall(size + chunk + b"\r\n")
                conn.sendall(b"0\r\n\r\n")
            else:
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
