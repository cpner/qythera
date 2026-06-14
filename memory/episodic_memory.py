import json
import os
import time
from typing import List, Dict, Optional


class EpisodicMemory:
    def __init__(self, storage_path: str = "./memory/episodic"):
        self.storage_path = storage_path
        self.conversations: List[Dict] = []
        self.current_conversation: Optional[Dict] = None
        os.makedirs(storage_path, exist_ok=True)
        self._load_all()

    def _load_all(self):
        for fname in os.listdir(self.storage_path):
            if fname.endswith(".json"):
                with open(os.path.join(self.storage_path, fname)) as f:
                    self.conversations.append(json.load(f))

    def start_conversation(self, metadata: Optional[Dict] = None) -> str:
        conv_id = f"conv_{int(time.time() * 1000)}"
        self.current_conversation = {
            "id": conv_id,
            "messages": [],
            "metadata": metadata or {},
            "created_at": time.time(),
        }
        return conv_id

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        if not self.current_conversation:
            self.start_conversation()
        msg = {"role": role, "content": content, "timestamp": time.time()}
        if metadata:
            msg["metadata"] = metadata
        self.current_conversation["messages"].append(msg)

    def end_conversation(self):
        if self.current_conversation:
            self.conversations.append(self.current_conversation)
            path = os.path.join(self.storage_path, f"{self.current_conversation['id']}.json")
            with open(path, "w") as f:
                json.dump(self.current_conversation, f, indent=2)
            self.current_conversation = None

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        results = []
        query_lower = query.lower()
        for conv in self.conversations:
            score = 0
            for msg in conv.get("messages", []):
                if query_lower in msg.get("content", "").lower():
                    score += 1
            if score > 0:
                results.append({"conversation": conv, "relevance": score})
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]

    def get_recent(self, n: int = 10) -> List[Dict]:
        return self.conversations[-n:]

    def get_stats(self) -> Dict:
        total_msgs = sum(len(c.get("messages", [])) for c in self.conversations)
        return {
            "total_conversations": len(self.conversations),
            "total_messages": total_msgs,
        }
