from typing import List, Dict


class ConversationSummarizer:
    def __init__(self, model_fn=None, max_summary_length: int = 500):
        self.model_fn = model_fn
        self.max_summary_length = max_summary_length

    def summarize(self, messages: List[Dict]) -> str:
        if not messages:
            return ""
        conversation_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
        )
        if self.model_fn:
            prompt = f"Summarize this conversation in {self.max_summary_length} words or less:\n\n{conversation_text}"
            return self.model_fn(prompt)
        parts = []
        for m in messages[-3:]:
            role = m.get("role", "user")
            content = m.get("content", "")[:100]
            parts.append(f"{role}: {content}")
        return " | ".join(parts)

    def extract_topics(self, messages: List[Dict]) -> List[str]:
        topics = set()
        for m in messages:
            content = m.get("content", "").lower()
            for word in ["python", "javascript", "help", "error", "code", "data", "model"]:
                if word in content:
                    topics.add(word)
        return list(topics)
