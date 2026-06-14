import os
import sys
import hashlib

def download_model(name="vaelon-7b", output_dir="./models"):
    print(f"Downloading {name}...")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Note: Pre-trained weights not yet available.")
    print(f"Train your own model first: qythera train --config training/configs/7b_lora.yaml")
    print(f"Or use a compatible model from HuggingFace.")
    print(f"Output directory: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="vaelon-7b")
    parser.add_argument("--output", default="./models")
    args = parser.parse_args()
    download_model(args.name, args.output)
