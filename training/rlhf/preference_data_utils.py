import json
import os
from typing import List, Dict, Optional


def load_preference_data(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return data


def format_preference_pair(chosen: List[Dict], rejected: List[Dict]) -> Dict:
    return {"chosen": chosen, "rejected": rejected}


def create_preference_dataset(conversations: List[List[Dict]], scoring_fn=None) -> List[Dict]:
    pairs = []
    for conv in conversations:
        if len(conv) < 2:
            continue
        pairs.append({
            "prompt": conv[:-1],
            "chosen": conv[-1].get("content", ""),
            "rejected": conv[-1].get("content", ""),
        })
    return pairs


def save_preference_data(data: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
