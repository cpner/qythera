# Converter for sharegpt dataset format
from typing import List, Dict

def convert_sharegpt(samples: List[Dict]) -> List[Dict]:
    converted = []
    for sample in samples:
        messages = []
        if "sharegpt" == "alpaca":
            if sample.get("instruction"):
                messages.append({"role": "user", "content": sample["instruction"]})
            if sample.get("output"):
                messages.append({"role": "assistant", "content": sample["output"]})
        elif "sharegpt" == "sharegpt":
            for turn in sample.get("conversations", []):
                role = "user" if turn.get("from") == "human" else "assistant"
                messages.append({"role": role, "content": turn.get("value", "")})
        elif "sharegpt" == "dolly":
            if sample.get("instruction"):
                messages.append({"role": "user", "content": sample["instruction"]})
            if sample.get("response"):
                messages.append({"role": "assistant", "content": sample["response"]})
        if messages:
            converted.append({"messages": messages})
    return converted

def validate_sharegpt(sample: Dict) -> bool:
    return "messages" in sample or "instruction" in sample
