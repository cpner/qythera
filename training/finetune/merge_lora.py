import os
import torch

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

from vaelon.config import VaelonConfig
from vaelon.model import VaelonModel


def merge_lora(base_model_path: str, lora_path: str, output_path: str):
    if not HAS_PEFT:
        print("peft not installed. Using direct weight merging.")
        config = VaelonConfig.vaelon_7b()
        model = VaelonModel(config)
        lora_weights = torch.load(os.path.join(lora_path, "adapter_model.bin"), map_location="cpu")
        base_state = model.state_dict()
        for key, value in lora_weights.items():
            base_key = key.replace("base_model.model.", "")
            if base_key in base_state:
                if "lora_A" in key:
                    base_state[base_key] += value
                elif "lora_B" in key:
                    base_state[base_key] += value
        model.load_state_dict(base_state)
        os.makedirs(output_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_path, "model.pt"))
        print(f"Merged model saved to {output_path}")
        return

    config = VaelonConfig.vaelon_7b()
    base_model = VaelonModel(config)
    merged = PeftModel.from_pretrained(base_model, lora_path)
    merged = merged.merge_and_unload()
    os.makedirs(output_path, exist_ok=True)
    torch.save(merged.state_dict(), os.path.join(output_path, "model.pt"))
    print(f"Merged model saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merge_lora(args.base, args.lora, args.output)
