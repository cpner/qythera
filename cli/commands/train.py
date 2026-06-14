import os
import subprocess


def run_train(config=None, gpus=1):
    print(f"\n  Starting Qythera Training")
    print(f"  GPUs: {gpus}")
    print(f"  Config: {config or 'default'}\n")

    script = os.path.join(os.path.dirname(__file__), "..", "..", "training", "pretrain", "train_pretrain.py")
    script = os.path.abspath(script)

    cmd = ["torchrun", "--nproc_per_node", str(gpus), "--master_port", "29500", script]

    if config:
        cmd.extend(["--config", config])

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("torchrun not found. Install PyTorch: pip install torch")
    except KeyboardInterrupt:
        print("\nTraining stopped.")
