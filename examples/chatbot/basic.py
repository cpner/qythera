from sdk.python.client import QytheraClient

client = QytheraClient("http://localhost:8000")

def chat():
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    print("Qythera Chat (type 'quit' to exit)\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in ("quit", "exit"): break
        if not user: continue
        messages.append({"role": "user", "content": user})
        resp = client.chat(messages)
        reply = resp["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": reply})
        print(f"\nVaelon: {reply}\n")

if __name__ == "__main__": chat()
