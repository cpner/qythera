import time, json, os
from collections import defaultdict

class MetricsTracker:
    def __init__(self, storage_path="./metrics"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.counters = defaultdict(int)
        self.histograms = defaultdict(list)
        self.gauges = {}

    def increment(self, name, value=1):
        self.counters[name] += value

    def observe(self, name, value):
        self.histograms[name].append(value)
        if len(self.histograms[name]) > 10000:
            self.histograms[name] = self.histograms[name][-5000:]

    def gauge(self, name, value):
        self.gauges[name] = value

    def export(self):
        result = {"counters": dict(self.counters), "gauges": dict(self.gauges)}
        for name, values in self.histograms.items():
            result[name] = {"count": len(values), "mean": sum(values)/len(values),
                           "min": min(values), "max": max(values),
                           "p50": sorted(values)[len(values)//2] if values else 0,
                           "p95": sorted(values)[int(len(values)*0.95)] if values else 0}
        return result

    def save(self):
        with open(os.path.join(self.storage_path, "metrics.json"), "w") as f:
            json.dump(self.export(), f, indent=2)
