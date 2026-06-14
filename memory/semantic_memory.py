import json
import os
from typing import List, Dict, Optional


class SemanticMemory:
    def __init__(self, storage_path: str = "./memory/semantic"):
        self.storage_path = storage_path
        self.knowledge: List[Dict] = []
        os.makedirs(storage_path, exist_ok=True)
        self._load()

    def _load(self):
        path = os.path.join(self.storage_path, "knowledge.json")
        if os.path.exists(path):
            with open(path) as f:
                self.knowledge = json.load(f)

    def add_fact(self, subject: str, predicate: str, obj: str, source: str = "unknown"):
        self.knowledge.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "source": source,
        })
        self._save()

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        query_lower = query.lower()
        results = []
        for fact in self.knowledge:
            score = sum(1 for field in [fact["subject"], fact["predicate"], fact["object"]]
                       if query_lower in field.lower())
            if score > 0:
                results.append({"fact": fact, "relevance": score})
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]

    def _save(self):
        path = os.path.join(self.storage_path, "knowledge.json")
        with open(path, "w") as f:
            json.dump(self.knowledge, f, indent=2)

    def __len__(self):
        return len(self.knowledge)
