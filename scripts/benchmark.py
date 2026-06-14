import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def benchmark_model():
    import torch
    from vaelon.config import VaelonConfig
    from vaelon.model import VaelonModel

    print("Qythera Benchmark Suite")
    print("=" * 50)

    config = VaelonConfig.vaelon_7b()
    model = VaelonModel(config)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e9:.2f}B)")
    print(f"Device: {device}")

    input_ids = torch.randint(0, 1000, (1, 128), device=device)

    print("\nBenchmarking forward pass...")
    times = []
    for _ in range(10):
        start = time.time()
        with torch.no_grad():
            model(input_ids)
        times.append(time.time() - start)
    avg_time = sum(times) / len(times)
    print(f"  Avg forward time: {avg_time*1000:.1f}ms")
    print(f"  Throughput: {128/avg_time:.0f} tokens/sec")

    print("\nBenchmarking generation (10 tokens)...")
    start = time.time()
    with torch.no_grad():
        output = model.generate(input_ids[:, :16], max_new_tokens=10, temperature=1.0)
    gen_time = time.time() - start
    print(f"  Generation time: {gen_time*1000:.1f}ms")
    print(f"  Tokens per second: {10/gen_time:.1f}")

    print("\nBenchmark complete!")

if __name__ == "__main__":
    benchmark_model()
