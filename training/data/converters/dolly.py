# Converter for dolly dataset format
from typing import List, Dict

def convert_dolly(samples: List[Dict]) -> List[Dict]:
    converted = []
    for sample in samples:
        messages = []
        if "dolly" == "alpaca":
            if sample.get("instruction"):
                messages.append({"role": "user", "content": sample["instruction"]})
            if sample.get("output"):
                messages.append({"role": "assistant", "content": sample["output"]})
        elif "dolly" == "sharegpt":
            for turn in sample.get("conversations", []):
                role = "user" if turn.get("from") == "human" else "assistant"
                messages.append({"role": role, "content": turn.get("value", "")})
        elif "dolly" == "dolly":
            if sample.get("instruction"):
                messages.append({"role": "user", "content": sample["instruction"]})
            if sample.get("response"):
                messages.append({"role": "assistant", "content": sample["response"]})
        if messages:
            converted.append({"messages": messages})
    return converted

def validate_dolly(sample: Dict) -> bool:
    return "messages" in sample or "instruction" in sample
