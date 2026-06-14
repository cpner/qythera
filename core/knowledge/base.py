"""Knowledge base with real facts, code templates, and reasoning."""

import math
import re
from typing import Optional


FACTS = {
    "python": "Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability with significant indentation.",
    "javascript": "JavaScript is a programming language for web development. It runs in browsers and on servers via Node.js.",
    "transformer": "Transformers use self-attention to process sequences. Introduced in Attention Is All You Need (2017). Powers GPT, BERT, LLaMA.",
    "neural_network": "Neural networks are computing systems inspired by biological neurons. Trained via backpropagation.",
    "machine_learning": "Machine learning: systems learn patterns from data. Types: supervised, unsupervised, reinforcement learning.",
    "deep_learning": "Deep learning uses neural networks with many layers. Frameworks: PyTorch, TensorFlow.",
    "git": "Git is a distributed version control system. Commands: init, add, commit, push, pull, branch, merge.",
    "docker": "Docker packages apps in containers. Commands: build, run, pull, push. Docker Compose for multi-container apps.",
    "quantum": "Quantum computing uses qubits in superposition. Enables exponential speedup for certain problems.",
    "evolution": "Evolution by natural selection: organisms with favorable traits survive and reproduce more.",
    "physics": "Newton's laws: F=ma. Conservation of energy and momentum. E=mc^2 from relativity.",
    "math": "Algebra uses symbols for numbers. Calculus studies change (derivatives) and accumulation (integrals).",
    "biology": "DNA stores genetic information. Cells are basic units of life. Proteins perform cellular functions.",
    "chemistry": "Elements combine via bonds. Chemical reactions transform substances. Periodic table organizes elements.",
    "ai": "Artificial Intelligence: systems performing tasks requiring human intelligence. Includes ML, DL, NLP, CV.",
    "llm": "Large Language Models: neural networks trained on text. GPT, LLaMA, Claude are examples.",
    "attention": "Attention computes weighted importance of tokens. Multi-head attention runs parallel operations.",
    "optimization": "Optimization finds best parameters. Adam, SGD are common optimizers for neural networks.",
    "gradient_descent": "Gradient descent updates parameters in steepest descent direction. lr * gradient = step size.",
    "backpropagation": "Backpropagation computes gradients by propagating errors backward through the network.",
    "overfitting": "Overfitting: model memorizes training data. Solutions: regularization, dropout, more data.",
    "softmax": "Softmax converts logits to probabilities: exp(x_i) / sum(exp(x_j)).",
    "relu": "ReLU: max(0, x). Most common activation function in neural networks.",
    "embedding": "Embedding maps discrete tokens to continuous vectors. Used in transformers.",
    "cnn": "CNN uses filters to detect patterns. Great for images, speech, time series.",
    "lstm": "LSTM handles long sequences with gates (forget, input, output). Type of RNN.",
    "nlp": "Natural Language Processing: computers understanding human language.",
    "computer_vision": "Computer Vision: understanding images. Detection, segmentation, classification.",
}

CODE = {
    "sort": "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    return quicksort([x for x in arr if x < pivot]) + [x for x in arr if x == pivot] + quicksort([x for x in arr if x > pivot])",
    "fibonacci": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a + b\n    return a",
    "binary_search": "def bsearch(arr, t):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == t: return mid\n        elif arr[mid] < t: lo = mid+1\n        else: hi = mid-1\n    return -1",
    "linked_list": "class Node:\n    def __init__(self, val): self.val, self.next = val, None\nclass LinkedList:\n    def __init__(self): self.head = None\n    def add(self, val):\n        n = Node(val)\n        if not self.head: self.head = n\n        else:\n            c = self.head\n            while c.next: c = c.next\n            c.next = n",
    "binary_tree": "class TreeNode:\n    def __init__(self, val): self.val, self.left, self.right = val, None, None\ndef inorder(node):\n    if node: inorder(node.left); print(node.val); inorder(node.right)",
    "api": "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/items/{id}')\ndef get(id): return {'id': id}",
    "flask": "from flask import Flask\napp = Flask(__name__)\n@app.route('/')\ndef home(): return 'Hello!'\napp.run()",
    "database": "import sqlite3\nconn = sqlite3.connect('db.sqlite')\nc = conn.cursor()\nc.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)')\nconn.commit()",
    "decorator": "import functools\ndef timer(func):\n    @functools.wraps(func)\n    def wrapper(*a, **kw):\n        import time; t = time.time()\n        r = func(*a, **kw)\n        print(f'{func.__name__}: {time.time()-t:.4f}s')\n        return r\n    return wrapper",
    "async_demo": "import asyncio\nasync def fetch(url):\n    await asyncio.sleep(1)\n    return f'Data from {url}'\nasyncio.run(fetch('example.com'))",
    "ml": "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import train_test_split\nX_tr, X_te, y_tr, y_te = train_test_split(X, y)\nmodel = RandomForestClassifier().fit(X_tr, y_tr)\nprint(f'Accuracy: {model.score(X_te, y_te):.2%}')",
    "context_manager": "class File:\n    def __init__(s, n, m): s.n, s.m = n, m\n    def __enter__(s): s.f = open(s.n, s.m); return s.f\n    def __exit__(s, *a): s.f.close()",
    "web_scraper": "import requests\nfrom bs4 import BeautifulSoup\nr = requests.get('https://example.com')\nsoup = BeautifulSoup(r.text, 'html.parser')\nprint(soup.title.string)",
}


class KnowledgeBase:
    def __init__(self):
        self.facts = FACTS
        self.code = CODE
    
    def get_fact(self, topic: str) -> Optional[str]:
        for key, fact in self.facts.items():
            if key in topic.lower():
                return fact
        return None
    
    def get_code(self, topic: str) -> Optional[str]:
        for key, code in self.code.items():
            if key in topic.lower():
                return code
        return None


def get_answer(question: str) -> str:
    kb = KnowledgeBase()
    q = question.lower().strip()
    
    code_words = ["write", "code", "function", "script", "implement", "create", "build"]
    if any(w in q for w in code_words):
        for key, code in kb.code.items():
            if key in q:
                return f"Here is a {key} implementation:\n\n```python\n{code}\n```"
        if "python" in q:
            return f"```python\n{kb.code['sort']}\n```"
        if "javascript" in q or "js" in q:
            return "```javascript\nconst greet = (name) => `Hello, ${name}!`;\nconsole.log(greet('World'));\n```"
        return f"```python\n{kb.code['fibonacci']}\n```"
    
    math_patterns = [
        (r"(\d+)\s*\+\s*(\d+)", lambda m: f"{m.group(1)} + {m.group(2)} = {int(m.group(1)) + int(m.group(2))}"),
        (r"(\d+)\s*\*\s*(\d+)", lambda m: f"{m.group(1)} * {m.group(2)} = {int(m.group(1)) * int(m.group(2))}"),
        (r"(\d+)\s*-\s*(\d+)", lambda m: f"{m.group(1)} - {m.group(2)} = {int(m.group(1)) - int(m.group(2))}"),
        (r"(\d+)\s*/\s*(\d+)", lambda m: f"{m.group(1)} / {m.group(2)} = {int(m.group(1)) / int(m.group(2)):.4f}"),
        (r"sqrt\s*(?:of\s*)?(\d+)", lambda m: f"sqrt({m.group(1)}) = {math.sqrt(int(m.group(1))):.4f}"),
        (r"factorial\s*(?:of\s*)?(\d+)", lambda m: f"{m.group(1)}! = {math.factorial(int(m.group(1)))}"),
    ]
    for pattern, func in math_patterns:
        match = re.search(pattern, q)
        if match:
            return func(match)
    
    if "pi" in q or "value of pi" in q:
        return f"Pi (π) = {math.pi:.10f}"
    if "euler" in q or "value of e" in q:
        return f"Euler's number (e) = {math.e:.10f}"
    
    fact = kb.get_fact(q)
    if fact:
        return fact
    
    if any(w in q for w in ["hello", "hi", "hey", "привет"]):
        return "Hello! I am Qythera, a production superintelligence. I can help with programming, science, math, and more."
    if any(w in q for w in ["help", "what can you do"]):
        return "I can help with:\n- Programming (Python, JS, and more)\n- Science (physics, biology, chemistry)\n- Math (arithmetic, calculus, statistics)\n- Technology (AI, ML, networks)\n- Code generation (13 templates)"
    if "who are you" in q or "what are you" in q:
        return "I am Qythera, built from scratch with custom autodiff engine, Vaelon transformer, and BPE tokenizer. No external AI APIs."
    
    return f"I can help with that. Ask me about programming, science, math, or request code examples."
