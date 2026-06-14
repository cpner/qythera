from sdk.python.client import QytheraClient

client = QytheraClient("http://localhost:8000")

messages = [{"role": "user", "content": "Tell me about quantum computing"}]
for chunk in client.chat_stream(messages):
    print(chunk, end="", flush=True)
print()
