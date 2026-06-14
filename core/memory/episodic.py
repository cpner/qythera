
import json, os, time
from typing import List, Dict, Optional

class EpisodicMemory:
    def __init__(self, path="./episodic"):
        self.path, self.convs = path, []
        os.makedirs(path, exist_ok=True)
        for f in os.listdir(path):
            if f.endswith(".json"):
                with open(os.path.join(path, f)) as fh: self.convs.append(json.load(fh))

    def start(self) -> str:
        cid = f"c{int(time.time()*1000)}"
        self.current = {"id": cid, "messages": [], "created": time.time()}
        return cid

    def add(self, role, content):
        if not hasattr(self, 'current'): self.start()
        self.current["messages"].append({"role": role, "content": content, "ts": time.time()})

    def end(self):
        if hasattr(self, 'current') and self.current:
            self.convs.append(self.current)
            with open(os.path.join(self.path, f"{self.current['id']}.json"), "w") as f:
                json.dump(self.current, f)
            self.current = None

    def search(self, query, k=5):
        q = query.lower()
        scored = [(c, sum(1 for m in c["messages"] if q in m["content"].lower())) for c in self.convs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, s in scored[:k] if s > 0]

    def recent(self, n=10): return self.convs[-n:]
