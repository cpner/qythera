import math, re

FACTS = {
"python":"Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability with significant indentation. Used for web dev, data science, AI/ML, automation.",
"javascript":"JavaScript is a programming language for web development. Runs in browsers and servers via Node.js. Used for React, Vue, Angular, Express.",
"transformer":"Transformers use self-attention to process sequences. Introduced in Attention Is All You Need (2017). Powers GPT, BERT, LLaMA.",
"neural_network":"A neural network is a computing system inspired by biological neurons. Layers of nodes with weighted connections. Trained via backpropagation.",
"machine_learning":"Machine learning: systems learn patterns from data. Types: supervised, unsupervised, reinforcement learning.",
"deep_learning":"Deep learning uses neural networks with many layers. Frameworks: PyTorch, TensorFlow. Applications: vision, NLP, speech.",
"git":"Git is a distributed version control system. Commands: init, add, commit, push, pull, branch, merge, rebase.",
"docker":"Docker packages apps in containers. Commands: build, run, pull, push. Docker Compose for multi-container apps.",
"quantum":"Quantum computing uses qubits in superposition. Quantum entanglement links particles. Enables exponential speedup.",
"physics":"Newton's laws: F=ma, inertia, action-reaction. Conservation of energy and momentum. E=mc^2 from relativity.",
"math":"Algebra uses symbols for numbers. Calculus studies rates of change and accumulation. Statistics analyzes data.",
"ai":"Artificial Intelligence: systems performing tasks requiring human intelligence. Includes ML, DL, NLP, CV.",
"llm":"Large Language Models: neural networks trained on text. GPT, LLaMA, Claude, Gemini are examples.",
"attention":"Attention computes weighted importance of tokens. Multi-head runs parallel operations for different aspects.",
"cnn":"CNN uses filters to detect patterns. Great for images, speech, time series.",
"nlp":"Natural Language Processing: computers understanding human language. Tokenization, parsing, sentiment.",
"evolution":"Evolution by natural selection: organisms with favorable traits survive and reproduce more.",
"photosynthesis":"Photosynthesis: 6CO2 + 6H2O + light -> C6H12O6 + 6O2.",
"dna":"DNA stores genetic information as a double helix. Bases: A, T, G, C.",
"internet":"Internet connects computers via TCP/IP. HTTP requests fetch web pages.",
"blockchain":"Blockchain: distributed ledger. Each block hashes the previous. Trustless transactions.",
}

CODE = {
"sort":"def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    return quicksort([x for x in arr if x < pivot]) + [x for x in arr if x == pivot] + quicksort([x for x in arr if x > pivot])",
"fib":"def fib(n):\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a + b\n    return a",
"bsearch":"def bsearch(arr, t):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == t: return mid\n        elif arr[mid] < t: lo = mid+1\n        else: hi = mid-1\n    return -1",
"api":"from fastapi import FastAPI\napp = FastAPI()\n@app.get('/items/{id}')\ndef get(id): return {'id': id}",
"flask":"from flask import Flask\napp = Flask(__name__)\n@app.route('/')\ndef home(): return 'Hello!'\napp.run()",
"db":"import sqlite3\nconn = sqlite3.connect('db.sqlite')\nc = conn.cursor()\nc.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)')\nconn.commit()",
"ml":"from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import train_test_split\nX_tr, X_te, y_tr, y_te = train_test_split(X, y)\nmodel = RandomForestClassifier().fit(X_tr, y_tr)\nprint(f'Accuracy: {model.score(X_te, y_te):.2%}')",
}

def answer(q):
    q = q.lower().strip()
    cw = ['write','code','function','script','implement','create','build']
    if any(w in q for w in cw):
        for k, v in CODE.items():
            if k in q: return f"Here is a {k}:\n\n```python\n{v}\n```"
        if 'python' in q: return f"```python\n{CODE['sort']}\n```"
        if 'javascript' in q or 'js' in q: return "```javascript\nconst greet = (n) => `Hello, ${n}!`;\nconsole.log(greet('World'));\n```"
        return f"```python\n{CODE['fib']}\n```"
    if re.search(r'(\d+)\s*\+\s*(\d+)', q): m = re.search(r'(\d+)\s*\+\s*(\d+)', q); return f"{m.group(1)} + {m.group(2)} = {int(m.group(1)) + int(m.group(2))}"
    if re.search(r'(\d+)\s*\*\s*(\d+)', q): m = re.search(r'(\d+)\s*\*\s*(\d+)', q); return f"{m.group(1)} * {m.group(2)} = {int(m.group(1)) * int(m.group(2))}"
    if re.search(r'sqrt\s*(?:of\s*)?(\d+)', q, re.I): m = re.search(r'sqrt\s*(?:of\s*)?(\d+)', q, re.I); return f"sqrt({m.group(1)}) = {math.sqrt(int(m.group(1))):.4f}"
    if 'pi' in q: return f"Pi = {math.pi:.10f}"
    for k, v in FACTS.items():
        if k in q: return v
    if any(w in q for w in ['hello','hi','hey']): return "Hello! I am Qythera. Ask about programming, science, math, or code."
    if 'who are you' in q: return "I am Qythera, built from scratch. No external AI APIs."
    return "Ask me about programming, science, math, or request code."
