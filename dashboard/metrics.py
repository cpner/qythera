import time, psutil, threading

class MetricsCollector:
    def __init__(self):
        self.metrics = {"requests": 0, "errors": 0, "latencies": [], "uptime": time.time()}

    def record_request(self, latency, error=False):
        self.metrics["requests"] += 1
        self.metrics["latencies"].append(latency)
        if error: self.metrics["errors"] += 1
        if len(self.metrics["latencies"]) > 10000:
            self.metrics["latencies"] = self.metrics["latencies"][-5000:]

    def get_stats(self):
        latencies = self.metrics["latencies"]
        return {
            "total_requests": self.metrics["requests"],
            "total_errors": self.metrics["errors"],
            "avg_latency": sum(latencies)/len(latencies) if latencies else 0,
            "p95_latency": sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0,
            "uptime_seconds": time.time() - self.metrics["uptime"],
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
        }
