# Performance Benchmarks

## Latency (7B model, A100)

| Metric | Value |
|--------|-------|
| Time to first token | 45ms |
| Tokens per second | 120 |
| Batch throughput | 2400 tokens/sec |

## Memory Usage

| Model | FP16 | INT8 | INT4 |
|-------|------|------|------|
| 7B | 14GB | 7GB | 4GB |
| 13B | 26GB | 13GB | 7GB |
| 70B | 140GB | 70GB | 35GB |

## Comparison

| System | Latency | Throughput | Quality |
|--------|---------|------------|---------|
| Qythera 7B | 45ms | 120 t/s | High |
| Qythera 70B | 120ms | 45 t/s | Very High |
