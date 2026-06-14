import json, time, os, sys, signal, socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from core.knowledge import answer
from core.safety import Safety

BASE = os.path.dirname(__file__)
safety = Safety()
srv = None

class H(BaseHTTPRequestHandler):
    def _j(self, d, s=200):
        self.send_response(s); self.send_header("Content-Type","application/json")
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
        self.end_headers(); self.wfile.write(json.dumps(d).encode())
    def _h(self, c, s=200, ct="text/html"):
        self.send_response(s); self.send_header("Content-Type", ct)
        self.send_header("Access-Control-Allow-Origin","*")
        self.end_headers(); self.wfile.write(c.encode("utf-8") if isinstance(c, str) else c)
    def _f(self, path, ct):
        try:
            with open(path, "rb") as f: data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(data)
        except: self._j({"error":"not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
        self.end_headers()

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            try:
                with open(os.path.join(BASE, "ui.html")) as f: self._h(f.read())
            except: self._h("<h1>Qythera</h1><p>Server running.</p>")
        elif p == "/manifest.json":
            self._f(os.path.join(BASE, "manifest.json"), "application/json")
        elif p == "/sw.js":
            self._f(os.path.join(BASE, "sw.js"), "application/javascript")
        elif p.startswith("/icon-") and p.endswith(".png"):
            self._f(os.path.join(BASE, p.lstrip("/")), "image/png")
        elif p == "/health":
            self._j({"status":"ok","model":"vaelon","uptime":round(time.time()-srv.t,1),"requests":srv.rc})
        elif p == "/v1/models":
            self._j({"data":[{"id":"vaelon","object":"model"}]})
        else: self._j({"error":"not found"}, 404)

    def do_POST(self):
        ln = int(self.headers.get("Content-Length",0))
        body = json.loads(self.rfile.read(ln)) if ln else {}
        srv.rc += 1
        if self.path == "/v1/chat/completions":
            msgs = body.get("messages",[])
            if not msgs: self._j({"error":"messages required"},400); return
            t0 = time.time()
            safe, _ = safety.check(msgs[-1].get("content",""))
            resp = safety.redact(answer(msgs[-1].get("content",""))) if safe else "Content blocked by safety filter."
            lat = time.time()-t0
            self._j({"id":f"c{int(time.time()*1000)}","object":"chat.completion","model":"vaelon",
                "choices":[{"index":0,"message":{"role":"assistant","content":resp},"finish_reason":"stop"}],
                "latency_ms":round(lat*1000,1)})
        else: self._j({"error":"not found"},404)
    def log_message(self,*a): pass

def shutdown(s,f):
    global srv; print("\nStopped."); sys.exit(0)

def run(port=8000):
    global srv
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    p = port
    try:
        s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(("0.0.0.0",p)); s.close()
    except:
        for p2 in range(port, port+100):
            try: s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(("0.0.0.0",p2)); s.close(); p=p2; break
            except: continue
    print(f"\n  Qythera: http://localhost:{p}")
    srv = type('S',(),{'rc':0,'t':time.time()})()
    httpd = HTTPServer(("0.0.0.0",p), H)
    try: httpd.serve_forever()
    except: pass
    finally: print("Stopped.")

if __name__ == "__main__":
    import argparse; a=argparse.ArgumentParser(); a.add_argument("--port",type=int,default=8000); run(a.parse_args().port)
