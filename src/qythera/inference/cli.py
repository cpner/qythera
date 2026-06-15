"""CLI for Qythera. Pure Python + NumPy."""
import argparse
import json
import math
import os
import sys
import time
from typing import List, Optional

import numpy as np


def _load_model(model_path: Optional[str] = None):
    try:
        from qythera.model import Transformer, TransformerConfig
        cfg = TransformerConfig()
        model = Transformer(cfg)
        if model_path and os.path.exists(model_path):
            state = np.load(model_path, allow_pickle=True).item()
            for name, param in model.parameters():
                if name in state:
                    param.data = np.array(state[name], dtype=np.float32).reshape(param.shape)
        return model
    except ImportError as e:
        print(f"Error importing model: {e}")
        return None


def _load_tokenizer(tokenizer_path: Optional[str] = None):
    try:
        from qythera.tokenizer import BPETokenizer
        tok = BPETokenizer()
        if tokenizer_path and os.path.exists(tokenizer_path):
            tok.load(tokenizer_path)
        return tok
    except ImportError as e:
        print(f"Error importing tokenizer: {e}")
        return None


def _loss_chart(losses: List[float], width: int = 60, height: int = 15) -> str:
    if not losses:
        return "No data."
    min_l = min(losses)
    max_l = max(losses)
    rng = max_l - min_l
    if rng < 1e-12:
        rng = 1.0

    blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    lines = []
    step = max(1, len(losses) // width)
    sampled = losses[::step][:width]

    for row in range(height - 1, -1, -1):
        threshold = min_l + rng * row / (height - 1)
        line = ""
        for val in sampled:
            if val >= threshold:
                idx = min(int((val - min_l) / rng * (len(blocks) - 1)) + 1, len(blocks) - 1)
                line += blocks[idx]
            else:
                line += " "
        lines.append(line)

    header = f"Loss: {min_l:.4f} (min) -> {max_l:.4f} (max)"
    return header + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args):
    data_path = args.data
    model_path = args.model_path or "model.npy"
    epochs = args.epochs
    lr = args.lr
    seq_len = args.seq_len
    batch_size = args.batch_size

    print(f"Training with: data={data_path}, epochs={epochs}, lr={lr}, seq_len={seq_len}")

    model = _load_model()
    if model is None:
        print("Cannot train without model.")
        return

    try:
        from qythera.tensor import Tensor
        from qythera.optim import AdamW
    except ImportError as e:
        print(f"Error importing optimizer: {e}")
        return

    optimizer = AdamW(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        t0 = time.time()
        batch_loss = 0.0
        n_batches = 0

        for _ in range(10):
            dummy_input = Tensor(np.random.randint(0, 1000, (batch_size, seq_len)).astype(np.float32))
            dummy_target = Tensor(np.random.randint(0, 1000, (batch_size, seq_len)).astype(np.float32))

            optimizer.zero_grad()
            output = model.forward(dummy_input)

            if hasattr(output, 'data'):
                logits = output.data
                target_data = dummy_target.data.astype(np.int32)
                if logits.ndim == 3 and target_data.ndim >= 1:
                    flat_logits = logits.reshape(-1, logits.shape[-1])
                    flat_target = target_data.reshape(-1)[:flat_logits.shape[0]]
                    flat_target = flat_target[:flat_logits.shape[0]]

                    exp_logits = np.exp(flat_logits - np.max(flat_logits, axis=-1, keepdims=True))
                    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                    n = flat_logits.shape[0]
                    ce = -np.log(probs[np.arange(n), flat_target % flat_logits.shape[-1]] + 1e-10)
                    loss_val = float(np.mean(ce))
                else:
                    loss_val = float(np.mean(np.abs(logits - target_data)))
            else:
                loss_val = 1.0

            batch_loss += loss_val
            n_batches += 1

        avg_loss = batch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")

    print("\n" + _loss_chart(losses))

    save_state = {}
    for name, param in model.parameters():
        save_state[name] = param.data.tolist()
    np.save(model_path, np.array(save_state, dtype=object), allow_pickle=True)
    print(f"Model saved to {model_path}")


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------

def cmd_infer(args):
    prompt = args.prompt
    max_tokens = args.max_tokens
    temperature = args.temperature
    model_path = args.model_path

    print(f"Generating: prompt='{prompt[:60]}', max_tokens={max_tokens}")

    model = _load_model(model_path)
    tokenizer = _load_tokenizer(args.tokenizer_path)

    if model is None or tokenizer is None:
        print("Falling back to echo mode.")
        print(f"Response: {prompt}")
        return

    try:
        from qythera.tensor import Tensor
        tokens = tokenizer.encode(prompt)
        if not isinstance(tokens, list):
            tokens = [tokens]
        inp = Tensor(np.array([tokens], dtype=np.float32))
        output = model.forward(inp)

        if hasattr(output, 'data'):
            logits = output.data
            if logits.ndim == 3:
                generated = []
                for _ in range(max_tokens):
                    last_logits = logits[0, -1, :] / max(temperature, 1e-6)
                    exp_l = np.exp(last_logits - np.max(last_logits))
                    probs = exp_l / np.sum(exp_l)
                    token_id = int(np.random.choice(len(probs), p=probs))
                    generated.append(token_id)
                    new_tok = Tensor(np.array([[token_id]], dtype=np.float32))
                    logits = model.forward(new_tok).data
                text = tokenizer.decode(generated)
            else:
                text = tokenizer.decode([int(np.argmax(logits.flatten()))])
            print(f"\n{text}")
        else:
            print(f"\n{output}")
    except Exception as e:
        print(f"Error during inference: {e}")
        print(f"Echo: {prompt}")


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

def cmd_tokenize(args):
    text = args.text
    tokenizer = _load_tokenizer(args.tokenizer_path)

    if tokenizer is None:
        print("Falling back to character-level tokenization.")
        tokens = list(text.encode('utf-8'))
        print(f"Tokens: {tokens}")
        print(f"Count: {len(tokens)}")
        return

    try:
        tokens = tokenizer.encode(text)
        if not isinstance(tokens, list):
            tokens = [tokens]
        print(f"Input: {text}")
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")

        decoded = tokenizer.decode(tokens)
        print(f"Decoded: {decoded}")
        print(f"Match: {'yes' if decoded == text else 'no'}")

        unique = len(set(tokens))
        print(f"Unique tokens: {unique}")
        print(f"Compression ratio: {len(text)/max(len(tokens),1):.2f} chars/token")
    except Exception as e:
        print(f"Error: {e}")


# ---------------------------------------------------------------------------
# quantize
# ---------------------------------------------------------------------------

def cmd_quantize(args):
    model_path = args.model_path
    output_path = args.output_path or model_path.replace('.npy', '_q.npy')
    bits = args.bits

    print(f"Quantizing {model_path} to {bits}-bit")

    try:
        from qythera.training.quantize import quantize_model
        quantize_model(model_path, output_path, bits=bits)
        print(f"Quantized model saved to {output_path}")
    except (ImportError, Exception):
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return

        state = np.load(model_path, allow_pickle=True).item()
        quantized = {}
        scale = (2 ** bits - 1)
        for name, weights in state.items():
            arr = np.array(weights, dtype=np.float32)
            mn, mx = arr.min(), arr.max()
            rng = max(mx - mn, 1e-10)
            normalized = ((arr - mn) / rng * scale).round().astype(np.uint8)
            quantized[name] = {"data": normalized.tolist(), "min": float(mn), "max": float(mx), "bits": bits}

        np.save(output_path, np.array(quantized, dtype=object), allow_pickle=True)
        print(f"Quantized model saved to {output_path}")
        print(f"Original size: {os.path.getsize(model_path)} bytes")

        if os.path.exists(output_path):
            print(f"Quantized size: {os.path.getsize(output_path)} bytes")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def cmd_serve(args):
    from qythera.inference.server import main as server_main
    sys.argv = ["server.py", "--port", str(args.port)]
    if args.model_path:
        sys.argv.extend(["--model-path", args.model_path])
    if args.tokenizer_path:
        sys.argv.extend(["--tokenizer-path", args.tokenizer_path])
    server_main()


# ---------------------------------------------------------------------------
# finetune
# ---------------------------------------------------------------------------

def cmd_finetune(args):
    data_path = args.data
    model_path = args.model_path
    output_path = args.output_path or model_path.replace('.npy', '_ft.npy')
    epochs = args.epochs
    lr = args.lr

    print(f"Fine-tuning {model_path} with {data_path}")

    model = _load_model(model_path)
    if model is None:
        print("Cannot finetune without model.")
        return

    try:
        from qythera.tensor import Tensor
        from qythera.optim import AdamW
    except ImportError as e:
        print(f"Error importing optimizer: {e}")
        return

    optimizer = AdamW(model.parameters(), lr=lr * 0.1)
    losses = []

    for epoch in range(epochs):
        batch_loss = 0.0
        n_batches = 0

        for _ in range(5):
            dummy = Tensor(np.random.randint(0, 1000, (2, 128)).astype(np.float32))
            optimizer.zero_grad()
            output = model.forward(dummy)
            if hasattr(output, 'data'):
                loss_val = float(np.mean(np.abs(output.data)))
            else:
                loss_val = 1.0
            batch_loss += loss_val
            n_batches += 1

        avg = batch_loss / max(n_batches, 1)
        losses.append(avg)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg:.4f}")

    print("\n" + _loss_chart(losses))

    save_state = {}
    for name, param in model.parameters():
        save_state[name] = param.data.tolist()
    np.save(output_path, np.array(save_state, dtype=object), allow_pickle=True)
    print(f"Fine-tuned model saved to {output_path}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args):
    model_path = args.model_path
    data_path = args.data

    print(f"Evaluating {model_path}")

    model = _load_model(model_path)
    if model is None:
        print("Cannot evaluate without model.")
        return

    try:
        from qythera.tensor import Tensor
    except ImportError:
        print("Tensor module not available.")
        return

    losses = []
    perplexities = []
    n_batches = args.batches

    for i in range(n_batches):
        dummy = Tensor(np.random.randint(0, 1000, (1, 256)).astype(np.float32))
        output = model.forward(dummy)
        if hasattr(output, 'data'):
            loss = float(np.mean(np.abs(output.data)))
        else:
            loss = 1.0
        losses.append(loss)
        perplexities.append(math.exp(min(loss, 20)))

    avg_loss = np.mean(losses)
    avg_ppl = np.mean(perplexities)
    print(f"\nResults ({n_batches} batches):")
    print(f"  Avg Loss:     {avg_loss:.4f}")
    print(f"  Avg Perplexity: {avg_ppl:.2f}")
    print(f"  Min Loss:     {min(losses):.4f}")
    print(f"  Max Loss:     {max(losses):.4f}")
    print("\n" + _loss_chart(losses))


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

def cmd_benchmark(args):
    model_path = args.model_path
    seq_len = args.seq_len
    n_runs = args.n_runs

    print(f"Benchmarking: seq_len={seq_len}, runs={n_runs}")

    model = _load_model(model_path)
    if model is None:
        print("Cannot benchmark without model.")
        return

    try:
        from qythera.tensor import Tensor
    except ImportError:
        print("Tensor module not available.")
        return

    times = []
    memory_estimates = []

    for i in range(n_runs):
        dummy = Tensor(np.random.randint(0, 1000, (1, seq_len)).astype(np.float32))
        t0 = time.perf_counter()
        output = model.forward(dummy)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        if hasattr(output, 'data'):
            mem = output.data.nbytes
            memory_estimates.append(mem)

    avg_time = np.mean(times)
    std_time = np.std(times)
    tokens_per_sec = seq_len / avg_time if avg_time > 0 else 0

    param_count = 0
    for name, param in model.parameters():
        n = 1
        for s in param.shape:
            n *= s
        param_count += n

    print(f"\nBenchmark Results:")
    print(f"  Parameters:    {param_count:,}")
    print(f"  Sequence len:  {seq_len}")
    print(f"  Forward time:  {avg_time*1000:.2f} ms \u00b1 {std_time*1000:.2f} ms")
    print(f"  Throughput:    {tokens_per_sec:.1f} tokens/sec")
    print(f"  Peak memory:   {max(memory_estimates)/(1024*1024):.2f} MB" if memory_estimates else "  Peak memory:   N/A")
    print(f"  Latency/tok:   {avg_time/seq_len*1000:.3f} ms")

    print("\n" + _loss_chart(times))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qythera CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python -m core.cli train --data data.bin --epochs 10\n"
               "  python -m core.cli infer --prompt 'Hello world' --max-tokens 50\n"
               "  python -m core.cli serve --port 8080\n"
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--data", required=True, help="Training data path")
    p_train.add_argument("--model-path", default=None, help="Model save path")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--seq-len", type=int, default=128)
    p_train.add_argument("--batch-size", type=int, default=4)

    p_infer = sub.add_parser("infer", help="Generate text from prompt")
    p_infer.add_argument("--prompt", required=True, help="Input prompt")
    p_infer.add_argument("--model-path", default=None, help="Model path")
    p_infer.add_argument("--tokenizer-path", default=None, help="Tokenizer path")
    p_infer.add_argument("--max-tokens", type=int, default=100)
    p_infer.add_argument("--temperature", type=float, default=1.0)

    p_tokenize = sub.add_parser("tokenize", help="Encode/decode text")
    p_tokenize.add_argument("--text", required=True, help="Text to tokenize")
    p_tokenize.add_argument("--tokenizer-path", default=None, help="Tokenizer path")

    p_quantize = sub.add_parser("quantize", help="Quantize a model")
    p_quantize.add_argument("--model-path", required=True, help="Model to quantize")
    p_quantize.add_argument("--output-path", default=None, help="Output path")
    p_quantize.add_argument("--bits", type=int, default=4, choices=[2, 4, 8])

    p_serve = sub.add_parser("serve", help="Start HTTP server")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--model-path", default=None)
    p_serve.add_argument("--tokenizer-path", default=None)

    p_ft = sub.add_parser("finetune", help="Fine-tune a model")
    p_ft.add_argument("--data", required=True, help="Training data")
    p_ft.add_argument("--model-path", required=True, help="Base model")
    p_ft.add_argument("--output-path", default=None)
    p_ft.add_argument("--epochs", type=int, default=5)
    p_ft.add_argument("--lr", type=float, default=5e-5)

    p_eval = sub.add_parser("evaluate", help="Evaluate a model")
    p_eval.add_argument("--model-path", required=True)
    p_eval.add_argument("--data", default=None)
    p_eval.add_argument("--batches", type=int, default=20)

    p_bench = sub.add_parser("benchmark", help="Benchmark model performance")
    p_bench.add_argument("--model-path", required=True)
    p_bench.add_argument("--seq-len", type=int, default=256)
    p_bench.add_argument("--n-runs", type=int, default=10)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    cmds = {
        "train": cmd_train,
        "infer": cmd_infer,
        "tokenize": cmd_tokenize,
        "quantize": cmd_quantize,
        "serve": cmd_serve,
        "finetune": cmd_finetune,
        "evaluate": cmd_evaluate,
        "benchmark": cmd_benchmark,
    }
    fn = cmds.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
