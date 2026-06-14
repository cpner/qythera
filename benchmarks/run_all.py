import time, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

class BenchmarkSuite:
    def __init__(self, client):
        self.client = client
        self.results = {}

    def benchmark_latency(self, prompt="Hello", n=10):
        times = []
        for _ in range(n):
            start = time.time()
            self.client.generate(prompt, max_tokens=50)
            times.append(time.time() - start)
        self.results["latency"] = {"mean": sum(times)/len(times), "min": min(times), "max": max(times)}

    def benchmark_throughput(self, prompt="Write a story", tokens=500, n=5):
        times = []
        for _ in range(n):
            start = time.time()
            self.client.generate(prompt, max_tokens=tokens)
            times.append(time.time() - start)
        avg_time = sum(times)/len(times)
        self.results["throughput"] = {"tokens_per_sec": tokens/avg_time, "avg_time": avg_time}

    def run_all(self):
        print("Running benchmarks...")
        self.benchmark_latency()
        self.benchmark_throughput()
        print(json.dumps(self.results, indent=2))
        return self.results

if __name__ == "__main__":
    from sdk.python.client import QytheraClient
    client = QytheraClient()
    suite = BenchmarkSuite(client)
    suite.run_all()
