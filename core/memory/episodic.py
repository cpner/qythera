import json, os, time
from typing import List, Dict, Optional


class EpisodicMemory:
    """Stores conversation history with timestamps and decay.
    
    Implements exponential forgetting: older memories decay over time.
    """
    
    def __init__(self, path="./episodic_memory", decay_rate=0.01):
        self.path = path
        self.decay_rate = decay_rate
        self.conversations = []
        self.current = None
        os.makedirs(path, exist_ok=True)
        for f in os.listdir(path):
            if f.endswith(".json"):
                with open(os.path.join(path, f)) as fh:
                    self.conversations.append(json.load(fh))

    def start(self) -> str:
        cid = f"conv_{int(time.time() * 1000)}"
        self.current = {"id": cid, "messages": [], "created": time.time()}
        return cid

    def add(self, role: str, content: str, metadata: Optional[Dict] = None):
        if self.current is None:
            self.start()
        msg = {"role": role, "content": content, "timestamp": time.time()}
        if metadata:
            msg["metadata"] = metadata
        self.current["messages"].append(msg)

    def end(self):
        if self.current:
            self.conversations.append(self.current)
            path = os.path.join(self.path, f"{self.current['id']}.json")
            with open(path, "w") as f:
                json.dump(self.current, f, indent=2)
            self.current = None

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search conversations with relevance scoring + time decay."""
        query_lower = query.lower()
        results = []
        now = time.time()
        
        for conv in self.conversations:
            score = 0
            for msg in conv.get("messages", []):
                if query_lower in msg.get("content", "").lower():
                    score += 1
            if score > 0:
                age = now - conv.get("created", now)
                decay = self.decay_rate * age
                adjusted_score = score * max(0.1, 1.0 - decay)
                results.append({"conversation": conv, "score": adjusted_score})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def recent(self, n: int = 10) -> List[Dict]:
        return self.conversations[-n:]

    def summarize(self) -> str:
        """Create a summary of all conversations."""
        total = len(self.conversations)
        total_msgs = sum(len(c.get("messages", [])) for c in self.conversations)
        return f"{total} conversations, {total_msgs} messages"
