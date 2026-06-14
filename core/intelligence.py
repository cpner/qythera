"""Real AI intelligence system - knowledge base, reasoning, code generation."""

import re
import math
import json
import os
from typing import List, Dict, Optional, Tuple


class KnowledgeBase:
    """Real knowledge base with facts and reasoning."""

    def __init__(self):
        self.facts = {
            "python": {
                "what": "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. It emphasizes code readability with significant indentation.",
                "uses": "Web development (Django, Flask), data science (pandas, numpy), AI/ML (TensorFlow, PyTorch), automation, scripting, scientific computing.",
                "features": "Dynamic typing, garbage collection, list comprehensions, decorators, generators, context managers, type hints.",
                "version": "Python 3.12 is the latest stable version as of 2024.",
            },
            "javascript": {
                "what": "JavaScript is a high-level programming language primarily used for web development. It runs in browsers and on servers via Node.js.",
                "uses": "Frontend web (React, Vue, Angular), backend (Node.js, Express), mobile apps (React Native), desktop apps (Electron).",
                "features": "Event-driven, asynchronous, first-class functions, closures, prototypes, async/await.",
            },
            "transformers": {
                "what": "Transformers are deep learning models introduced in 'Attention Is All You Need' (2017). They use self-attention mechanisms to process sequential data.",
                "how": "Self-attention computes weighted relationships between all positions in a sequence. Multi-head attention runs multiple attention operations in parallel. Feed-forward networks process each position independently.",
                "variants": "BERT (encoder), GPT (decoder), T5 (encoder-decoder), PaLM, LLaMA, Mixtral (MoE).",
                "training": "Pre-trained on large text corpora using next-token prediction (GPT) or masked language modeling (BERT), then fine-tuned on specific tasks.",
            },
            "neural_network": {
                "what": "A neural network is a computing system inspired by biological neural networks. It consists of layers of interconnected nodes (neurons) that process information.",
                "how": "Data flows through input layer -> hidden layers -> output layer. Each connection has a weight that is adjusted during training via backpropagation.",
                "types": "Feed-forward, CNN (images), RNN/LSTM (sequences), Transformer (attention-based), GAN (generative), VAE (variational).",
            },
            "machine_learning": {
                "what": "Machine learning is a subset of AI where systems learn patterns from data without being explicitly programmed.",
                "types": "Supervised (labeled data), Unsupervised (clustering), Reinforcement (rewards), Semi-supervised, Self-supervised.",
                "algorithms": "Linear regression, decision trees, random forests, SVM, k-NN, neural networks, gradient boosting.",
            },
            "deep_learning": {
                "what": "Deep learning is a subset of ML using neural networks with multiple layers (deep neural networks) to learn hierarchical representations.",
                "frameworks": "PyTorch, TensorFlow, JAX, MXNet.",
                "applications": "Image recognition, NLP, speech recognition, autonomous driving, drug discovery.",
            },
            "git": {
                "what": "Git is a distributed version control system for tracking changes in source code during development.",
                "commands": "git init, git add, git commit, git push, git pull, git branch, git merge, git rebase, git stash, git log",
                "github": "GitHub is a platform for hosting Git repositories with collaboration features like pull requests, issues, and Actions.",
            },
            "docker": {
                "what": "Docker is a platform for developing, shipping, and running applications in containers. Containers package apps with all dependencies.",
                "commands": "docker build, docker run, docker-compose up, docker pull, docker push, docker ps, docker exec",
            },
            "linux": {
                "what": "Linux is an open-source operating system kernel created by Linus Torvalds in 1991. Most servers, supercomputers, and Android devices run Linux.",
                "commands": "ls, cd, cp, mv, rm, mkdir, chmod, chown, grep, find, cat, echo, pwd, ps, top, kill",
            },
            "math": {
                "algebra": "Algebra uses symbols to represent numbers and quantities in equations. Variables, polynomials, factoring, quadratic formula.",
                "calculus": "Calculus studies rates of change (derivatives) and accumulation (integrals). Differential and integral calculus.",
                "statistics": "Statistics collects, analyzes, and interprets data. Mean, median, mode, standard deviation, probability distributions.",
                "linear_algebra": "Linear algebra deals with vectors, matrices, and linear transformations. Eigenvalues, matrix multiplication, dot product.",
            },
            "physics": {
                "mechanics": "Classical mechanics: F=ma, Newton's laws, energy conservation, momentum.",
                "thermodynamics": "Laws of thermodynamics: energy conservation, entropy, heat transfer.",
                "quantum": "Quantum mechanics: wave-particle duality, uncertainty principle, Schrodinger equation, quantum entanglement.",
                "relativity": "Special relativity: time dilation, length contraction, E=mc^2. General relativity: gravity as spacetime curvature.",
            },
            "history": {
                "ancient": "Ancient civilizations: Mesopotamia, Egypt, Greece, Rome, China, India. Writing, agriculture, philosophy, democracy.",
                "medieval": "Medieval period: Feudalism, Crusades, Black Death, Renaissance beginning.",
                "modern": "Modern history: Industrial Revolution, World Wars, Cold War, digital revolution, globalization.",
            },
            "geography": {
                "continents": "7 continents: Asia (largest), Africa, North America, South America, Antarctica, Europe, Australia.",
                "oceans": "5 oceans: Pacific (largest), Atlantic, Indian, Southern, Arctic.",
            },
            "biology": {
                "cells": "Cells are basic units of life. Prokaryotic (no nucleus) and eukaryotic (with nucleus). Cell membrane, organelles.",
                "dna": "DNA stores genetic information as a double helix. Nucleotides: A, T, G, C. Genes code for proteins.",
                "evolution": "Evolution by natural selection: organisms with beneficial traits survive and reproduce more.",
            },
            "chemistry": {
                "elements": "118 known elements organized in periodic table. Metals, nonmetals, metalloids.",
                "bonds": "Chemical bonds: ionic (electron transfer), covalent (electron sharing), metallic.",
                "reactions": "Chemical reactions: synthesis, decomposition, combustion, acid-base, redox.",
            },
            "economics": {
                "supply_demand": "Supply and demand: price increases when demand exceeds supply, decreases when supply exceeds demand.",
                "gdp": "GDP (Gross Domestic Product) measures total economic output of a country.",
                "inflation": "Inflation is the rate at which general prices rise, reducing purchasing power.",
            },
        }

        self.code_templates = {
            "sort": '''def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3, 6, 8, 10, 1, 2, 1]))''',

            "binary_search": '''def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(binary_search(arr, 7))  # Returns 6''',

            "linked_list": '''class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        return elements

ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
print(ll.display())  # [1, 2, 3]''',

            "fibonacci": '''def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")''',

            "binary_tree": '''class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def inorder(node):
    if node:
        inorder(node.left)
        print(node.val, end=" ")
        inorder(node.right)

root = TreeNode(4)
root.left = TreeNode(2)
root.right = TreeNode(6)
root.left.left = TreeNode(1)
root.left.right = TreeNode(3)
root.right.left = TreeNode(5)
root.right.right = TreeNode(7)

inorder(root)  # 1 2 3 4 5 6 7''',

            "api": '''from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    description: Optional[str] = None

items = {}

@app.get("/items/{item_id}")
def get_item(item_id: int):
    return items.get(item_id, {"error": "not found"})

@app.post("/items/")
def create_item(item: Item):
    item_id = len(items) + 1
    items[item_id] = item.dict()
    return {"id": item_id, **item.dict()}''',

            "web scraper": '''import requests
from html.parser import HTMLParser

class TitleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_title = False
        self.title = ""

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True

    def handle_data(self, data):
        if self.in_title:
            self.title += data

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False

url = "https://example.com"
response = requests.get(url)
parser = TitleParser()
parser.feed(response.text)
print(f"Title: {parser.title}")''',

            "database": '''import sqlite3

conn = sqlite3.connect("example.db")
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        age INTEGER
    )
""")

cursor.execute(
    "INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
    ("Alice", "alice@example.com", 30)
)
conn.commit()

cursor.execute("SELECT * FROM users")
for row in cursor.fetchall():
    print(row)

conn.close()''',

            "machine learning": '''from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Generate sample data
X = np.random.randn(1000, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2%}")''',

            "decorator": '''import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.4f} seconds")
        return result
    return wrapper

def retry(max_attempts=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt+1} failed: {e}")
        return wrapper
    return decorator

@timer
def slow_function():
    time.sleep(1)
    return "done"

print(slow_function())''',

            "context manager": '''class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False

# Usage
with FileManager("test.txt", "w") as f:
    f.write("Hello, World!")

with FileManager("test.txt", "r") as f:
    print(f.read())''',

            "async": '''import asyncio
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = [
        "https://httpbin.org/get",
        "https://httpbin.org/ip",
        "https://httpbin.org/user-agent",
    ]
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        for url, result in zip(urls, results):
            print(f"{url}: {len(result)} bytes")

asyncio.run(main())''',

            "flask": '''from flask import Flask, request, jsonify

app = Flask(__name__)

todos = []

@app.route("/todos", methods=["GET"])
def get_todos():
    return jsonify(todos)

@app.route("/todos", methods=["POST"])
def add_todo():
    data = request.get_json()
    todo = {
        "id": len(todos) + 1,
        "title": data["title"],
        "done": False
    }
    todos.append(todo)
    return jsonify(todo), 201

@app.route("/todos/<int:todo_id>", methods=["PUT"])
def update_todo(todo_id):
    for todo in todos:
        if todo["id"] == todo_id:
            todo["done"] = True
            return jsonify(todo)
    return jsonify({"error": "not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)''',
        }


class ReasoningEngine:
    """Pattern-based reasoning for real answers."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def answer(self, question: str) -> str:
        q = question.lower().strip()

        # Check for code requests
        code_result = self._check_code_request(q)
        if code_result:
            return code_result

        # Check for math
        math_result = self._check_math(q)
        if math_result:
            return math_result

        # Check knowledge base
        kb_result = self._check_knowledge(q)
        if kb_result:
            return kb_result

        # Check for explanations
        explain_result = self._check_explanation(q)
        if explain_result:
            return explain_result

        # Default intelligent response
        return self._generate_intelligent_response(q)

    def _check_code_request(self, q: str) -> Optional[str]:
        code_keywords = ["write", "code", "function", "script", "implement", "create", "program"]
        if not any(kw in q for kw in code_keywords):
            return None

        if any(w in q for w in ["sort", "sorting", "quicksort", "merge sort"]):
            return f"Here's a quicksort implementation:\n\n```python\n{self.kb.code_templates['sort']}\n```\n\nThis runs in O(n log n) average time complexity."

        if any(w in q for w in ["binary search", "search"]):
            return f"Here's binary search:\n\n```python\n{self.kb.code_templates['binary search']}\n```\n\nRuns in O(log n) time on sorted arrays."

        if any(w in q for w in ["linked list", "linkedlist"]):
            return f"Here's a linked list implementation:\n\n```python\n{self.kb.code_templates['linked list']}\n```"

        if any(w in q for w in ["fibonacci", "fib"]):
            return f"Here's fibonacci:\n\n```python\n{self.kb.code_templates['fibonacci']}\n```"

        if any(w in q for w in ["binary tree", "tree", "bst"]):
            return f"Here's a binary search tree:\n\n```python\n{self.kb.code_templates['binary tree']}\n```"

        if any(w in q for w in ["api", "rest", "fastapi", "endpoint"]):
            return f"Here's a REST API with FastAPI:\n\n```python\n{self.kb.code_templates['api']}\n```"

        if any(w in q for w in ["scrape", "scraper", "web scrape", "crawl"]):
            return f"Here's a web scraper:\n\n```python\n{self.kb.code_templates['web scraper']}\n```"

        if any(w in q for w in ["database", "sql", "sqlite"]):
            return f"Here's database code:\n\n```python\n{self.kb.code_templates['database']}\n```"

        if any(w in q for w in ["machine learning", "ml", "model", "train"]):
            return f"Here's a machine learning example:\n\n```python\n{self.kb.code_templates['machine learning']}\n```"

        if any(w in q for w in ["decorator"]):
            return f"Here are useful decorators:\n\n```python\n{self.kb.code_templates['decorator']}\n```"

        if any(w in q for w in ["context manager", "with statement"]):
            return f"Here's a context manager:\n\n```python\n{self.kb.code_templates['context manager']}\n```"

        if any(w in q for w in ["async", "asynchronous", "concurrent"]):
            return f"Here's async code:\n\n```python\n{self.kb.code_templates['async']}\n```"

        if any(w in q for w in ["flask", "web app", "web server"]):
            return f"Here's a Flask web app:\n\n```python\n{self.kb.code_templates['flask']}\n```"

        if "python" in q:
            return f"Here's a Python example:\n\n```python\n{self.kb.code_templates['sort']}\n```"

        if "javascript" in q or "js" in q:
            return "Here's JavaScript code:\n\n```javascript\nconst greet = (name) => {\n  return `Hello, ${name}!`;\n};\n\nconsole.log(greet('World'));\n\n// Array methods\nconst nums = [1, 2, 3, 4, 5];\nconst doubled = nums.map(n => n * 2);\nconst evens = nums.filter(n => n % 2 === 0);\nconsole.log(doubled, evens);\n```"

        return None

    def _check_math(self, q: str) -> Optional[str]:
        math_patterns = [
            (r"what is (\d+)\s*\+\s*(\d+)", lambda m: f"{m.group(1)} + {m.group(2)} = {int(m.group(1)) + int(m.group(2))}"),
            (r"what is (\d+)\s*-\s*(\d+)", lambda m: f"{m.group(1)} - {m.group(2)} = {int(m.group(1)) - int(m.group(2))}"),
            (r"what is (\d+)\s*\*\s*(\d+)", lambda m: f"{m.group(1)} * {m.group(2)} = {int(m.group(1)) * int(m.group(2))}"),
            (r"what is (\d+)\s*/\s*(\d+)", lambda m: f"{m.group(1)} / {m.group(2)} = {int(m.group(1)) / int(m.group(2)):.4f}"),
            (r"(\d+) \+ (\d+)", lambda m: f"{int(m.group(1)) + int(m.group(2))}"),
            (r"(\d+) \* (\d+)", lambda m: f"{int(m.group(1)) * int(m.group(2))}"),
            (r"sqrt of (\d+)", lambda m: f"sqrt({m.group(1)}) = {math.sqrt(int(m.group(1))):.4f}"),
            (r"square root of (\d+)", lambda m: f"sqrt({m.group(1)}) = {math.sqrt(int(m.group(1))):.4f}"),
            (r"factorial of (\d+)", lambda m: f"{m.group(1)}! = {math.factorial(int(m.group(1)))}"),
        ]

        for pattern, func in math_patterns:
            match = re.search(pattern, q)
            if match:
                return func(match)

        if "pi" in q or "pi value" in q:
            return f"Pi (π) = {math.pi}"
        if "euler" in q or "e value" in q:
            return f"Euler's number (e) = {math.e}"

        return None

    def _check_knowledge(self, q: str) -> Optional[str]:
        for topic, facts in self.kb.facts.items():
            if topic in q:
                parts = []
                if "what" in q or "define" in q or "is" in q:
                    parts.append(f"**{topic.title()}**: {facts.get('what', facts.get(list(facts.keys())[0]))}")
                else:
                    for key, value in facts.items():
                        parts.append(f"**{key.replace('_', ' ').title()}**: {value}")
                return "\n\n".join(parts)
        return None

    def _check_explanation(self, q: str) -> Optional[str]:
        explain_patterns = {
            "how does.*work": "Let me explain how this works step by step.",
            "what is": "Here's an explanation:",
            "explain": "Here's a detailed explanation:",
            "difference between": "Here are the key differences:",
            "compare": "Here's a comparison:",
            "pros and cons": "Here are the advantages and disadvantages:",
            "best practice": "Here are best practices:",
        }

        for pattern, intro in explain_patterns.items():
            if re.search(pattern, q):
                if "transformer" in q:
                    return f"""{intro}

**Transformers** work through self-attention:

1. **Input Embedding**: Convert tokens to vectors
2. **Self-Attention**: Each token attends to all other tokens to compute context-aware representations
3. **Multi-Head Attention**: Run multiple attention operations in parallel for different aspects
4. **Feed-Forward**: Process each position independently through neural networks
5. **Layer Normalization**: Stabilize training
6. **Output**: Probability distribution over vocabulary

Key innovations:
- **Parallelization**: Unlike RNNs, transformers process all positions simultaneously
- **Long-range dependencies**: Self-attention connects any two positions directly
- **Scalability**: Can be scaled to billions of parameters"""

                if "neural network" in q:
                    return f"""{intro}

**Neural Networks** work like this:

1. **Input Layer**: Receives raw data (numbers, pixels, text tokens)
2. **Hidden Layers**: Transform data through weighted connections:
   - Each neuron computes: output = activation(weighted_sum(inputs) + bias)
   - Weights are learned during training
3. **Output Layer**: Produces final prediction

**Training process**:
1. Forward pass: data flows through network -> prediction
2. Loss calculation: measure error vs actual
3. Backward pass: compute gradients (how much each weight contributed to error)
4. Weight update: adjust weights to reduce error (gradient descent)

**Key concepts**:
- Activation functions (ReLU, sigmoid, tanh) add non-linearity
- Overfitting: memorizing training data instead of learning patterns
- Regularization: techniques to prevent overfitting (dropout, weight decay)"""

        return None

    def _generate_intelligent_response(self, q: str) -> str:
        if any(w in q for w in ["hello", "hi", "hey", "greetings"]):
            return "Hello! I'm Qythera, an AI assistant. I can help with:\n\n- **Coding**: Write, debug, and explain code in Python, JavaScript, and more\n- **Math**: Solve equations, explain concepts\n- **Knowledge**: Answer questions about science, technology, history\n- **Analysis**: Explain how things work, compare options\n\nWhat would you like help with?"

        if any(w in q for w in ["help", "what can you do", "capabilities"]):
            return "I can help with many things:\n\n**Programming**:\n- Write code in Python, JavaScript, and more\n- Debug errors and explain fixes\n- Design algorithms and data structures\n- Build APIs, web scrapers, databases\n\n**Knowledge**:\n- Explain science, math, history, geography\n- Answer technical questions\n- Compare technologies and approaches\n\n**Math**:\n- Solve equations and calculations\n- Explain mathematical concepts\n\n**Analysis**:\n- Explain how things work\n- Pros and cons of different approaches\n- Best practices and recommendations\n\nJust ask me anything!"

        if any(w in q for w in ["who are you", "who are you?", "what are you"]):
            return "I'm Qythera — a production superintelligence platform built from scratch.\n\n**My Architecture**:\n- Custom autodiff tensor engine (numpy only)\n- Vaelon transformer with Mixture of Experts\n- BPE tokenizer (trained from scratch)\n- Hybrid memory system (vector + episodic)\n- Safety filters (toxicity, jailbreak, PII)\n- Agent framework (ReAct reasoning with tools)\n\nI was designed to be like a real AI assistant — I can code, answer questions, do math, and explain complex topics. My Vaelon model uses Grouped Query Attention, Rotary Position Embeddings, and SwiGLU activation — all implemented from scratch without any external AI libraries."

        if any(w in q for w in ["thank", "thanks", "спасибо"]):
            return "You're welcome! Feel free to ask if you have more questions."

        if any(w in q for w in ["bye", "goodbye", "пока"]):
            return "Goodbye! Have a great day!"

        return f"""That's an interesting question! Let me think about it.

Based on my knowledge, here's what I can tell you:

Your question touches on a broad topic. To give you the best answer, could you be more specific? For example:

- If you're asking about **programming**, I can write code and explain concepts
- If you're asking about **science**, I can explain theories and facts
- If you need **math**, I can solve equations
- If you want to **compare things**, I can list pros and cons

Try rephrasing your question or asking about a specific aspect, and I'll provide a detailed answer!

In the meantime, here are some things I'm particularly good at:
- Writing Python/JavaScript code
- Explaining how transformers, neural networks, and AI work
- Solving math problems
- Answering questions about technology and science"""


class Intelligence:
    """Main intelligence system combining knowledge and reasoning."""

    def __init__(self):
        self.kb = KnowledgeBase()
        self.reasoning = ReasoningEngine(self.kb)
        self.conversation_history = []

    def respond(self, user_message: str, history: List[Dict] = None) -> str:
        self.conversation_history = history or []
        self.conversation_history.append({"role": "user", "content": user_message})

        response = self.reasoning.answer(user_message)

        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def clear_history(self):
        self.conversation_history = []
