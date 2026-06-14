from sdk.python.client import QytheraClient

client = QytheraClient("http://localhost:8000")

prompt = "Write a Python function to merge two sorted lists"
result = client.generate(f"{prompt}\n\nProvide only the code, no explanation.")
print(result)
