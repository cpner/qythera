from sdk.python.client import QytheraClient

client = QytheraClient("http://localhost:8000")

system = "You are an expert financial analyst. Provide detailed, accurate analysis."
messages = [{"role": "system", "content": system}]

queries = [
    "Analyze the current state of AI infrastructure market",
    "What are the key investment opportunities in GPU cloud computing?",
    "Summarize the main risks and opportunities",
]

for q in queries:
    messages.append({"role": "user", "content": q})
    resp = client.chat(messages)
    reply = resp["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    print(f"Q: {q}\nA: {reply}\n{'='*60}\n")
