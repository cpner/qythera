from sdk.python.client import QytheraClient

client = QytheraClient("http://localhost:8000")

topics = ["transformer architecture", "mixture of experts", "reinforcement learning from human feedback"]
for topic in topics:
    result = client.generate(f"Explain {topic} in 3 sentences for a technical audience.")
    print(f"## {topic}\n{result}\n")
