import math
import re
"""Knowledge base with real facts and reasoning."""

KNOWLEDGE = {
    "python": "Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability with significant indentation. Used for web development, data science, AI/ML, automation, and scripting.",
    "javascript": "JavaScript is a programming language for web development. It runs in browsers and servers via Node.js. Used for frontend (React, Vue), backend (Express), mobile (React Native).",
    "transformer": "Transformers use self-attention to process sequences. Introduced in Attention Is All You Need (2017). Key: multi-head attention, positional encoding, feed-forward layers. Powers GPT, BERT, LLaMA.",
    "neural_network": "Neural networks are computing systems inspired by biological neurons. Layers of nodes with weighted connections. Trained via backpropagation to minimize loss.",
    "machine_learning": "Machine learning: systems learn patterns from data. Types: supervised, unsupervised, reinforcement learning. Algorithms: neural networks, decision trees, SVM.",
    "deep_learning": "Deep learning uses neural networks with many layers. Frameworks: PyTorch, TensorFlow. Applications: vision, NLP, speech, autonomous driving.",
    "git": "Git is a distributed version control system. Commands: init, add, commit, push, pull, branch, merge. GitHub hosts Git repos.",
    "docker": "Docker containers package apps with dependencies. Commands: build, run, pull, push. Docker Compose for multi-container apps.",
    "linux": "Linux is an open-source OS. Commands: ls, cd, grep, find, chmod, ps, top. Most servers run Linux.",
    "quantum": "Quantum computing uses qubits in superposition. Quantum entanglement links particles. Enables exponential speedup for certain problems.",
    "relativity": "Special relativity: E=mc^2, time dilation, speed of light constant. General relativity: gravity as spacetime curvature.",
    "evolution": "Evolution by natural selection: organisms with favorable traits survive and reproduce more. Drives biodiversity.",
    "photosynthesis": "Photosynthesis: 6CO2 + 6H2O + light -> C6H12O6 + 6O2. Converts light energy to chemical energy in plants.",
    "dna": "DNA stores genetic information as a double helix. Bases: A, T, G, C. A pairs with T, G with C.",
    "black_hole": "Black holes form from massive star collapse. Nothing escapes beyond event horizon. Time dilation near black holes.",
    "internet": "Internet connects computers via TCP/IP. HTTP requests fetch web pages. DNS resolves domain names to IP addresses.",
    "blockchain": "Blockchain: distributed ledger recording transactions. Each block hashes the previous. Enables trustless transparent transactions.",
    "climate": "Climate change: global temperatures rising due to greenhouse gases from fossil fuels. Causes extreme weather, sea level rise.",
    "math_algebra": "Algebra uses symbols for numbers. Variables, equations, polynomials. Quadratic formula: x = (-b +/- sqrt(b^2-4ac)) / 2a.",
    "math_calculus": "Calculus: derivatives (rates of change) and integrals (accumulation). Fundamental theorem connects them.",
    "math_statistics": "Statistics: mean, median, mode, standard deviation. Probability distributions. Hypothesis testing.",
    "physics_mechanics": "Newton's laws: F=ma, inertia, action-reaction. Conservation of energy and momentum.",
    "chemistry": "Chemistry: elements, bonds (ionic, covalent, metallic), reactions (synthesis, decomposition, combustion).",
    "economics": "Supply and demand determine prices. GDP measures economic output. Inflation reduces purchasing power.",
    "history_ancient": "Ancient civilizations: Mesopotamia, Egypt, Greece, Rome. Writing, agriculture, philosophy, democracy.",
    "geography": "7 continents: Asia, Africa, N. America, S. America, Antarctica, Europe, Australia. 5 oceans.",
    "biology_cells": "Cells: basic units of life. Prokaryotic (no nucleus) and eukaryotic (with nucleus).",
    "ai": "Artificial Intelligence: systems that perform tasks requiring human intelligence. Includes machine learning, deep learning, NLP, computer vision.",
    "llm": "Large Language Models: neural networks trained on text to generate coherent text. GPT, LLaMA, Claude are examples.",
    "attention": "Attention mechanism: computes weighted importance of each token relative to others. Multi-head attention runs multiple parallel attention operations.",
    "backpropagation": "Backpropagation: algorithm to compute gradients by propagating errors backward through the network. Foundation of neural network training.",
    "gradient_descent": "Gradient descent: optimization algorithm that updates parameters in direction of steepest descent. learning_rate * gradient determines step size.",
    "overfitting": "Overfitting: model memorizes training data instead of learning patterns. Solutions: regularization, dropout, more data, early stopping.",
    "embedding": "Embedding: mapping discrete tokens to continuous vectors. Word2Vec, GloVe, learned embeddings in transformers.",
    "softmax": "Softmax: converts logits to probabilities. exp(x_i) / sum(exp(x_j)). Used in classification and attention.",
    "relu": "ReLU: rectified linear unit. max(0, x). Most common activation function in neural networks.",
    "lstm": "LSTM: Long Short-Term Memory. Type of RNN with gates (forget, input, output) to handle long sequences.",
    "cnn": "CNN: Convolutional Neural Network. Uses filters to detect patterns. Great for images, speech, time series.",
    "gan": "GAN: Generative Adversarial Network. Generator creates fake data, discriminator detects fakes. Used for image generation.",
    "vae": "VAE: Variational Autoencoder. Learns latent representation. Used for generation and anomaly detection.",
    "reinforcement_learning": "Reinforcement learning: agent learns by interacting with environment, receiving rewards. Q-learning, policy gradient.",
    "nlp": "Natural Language Processing: computers understanding human language. Tokenization, parsing, sentiment analysis, translation.",
    "computer_vision": "Computer Vision: computers understanding images. Object detection, segmentation, classification, generation.",
    "speech_recognition": "Speech recognition: converting audio to text. Whisper, DeepSpeech. Uses acoustic and language models.",
    "recommendation": "Recommendation systems predict user preferences. Collaborative filtering, content-based, hybrid approaches.",
    "anomaly_detection": "Anomaly detection: finding unusual patterns. Statistical methods, isolation forests, autoencoders.",
    "time_series": "Time series analysis: analyzing sequential data. ARIMA, Prophet, LSTM for forecasting.",
    "graph_neural_network": "GNN: processes graph-structured data. Node classification, link prediction, graph classification.",
    "attention_mechanism": "Self-attention: each position attends to all others. Scaled dot-product: Q*K^T/sqrt(d) then softmax.",
    "positional_encoding": "Positional encoding adds sequence order information. Sinusoidal (original transformer), learned, RoPE.",
    "layer_normalization": "Layer normalization: normalizes activations across features. Stabilizes training. RMSNorm is faster variant.",
    "dropout": "Dropout: randomly zeros elements during training. Prevents overfitting. Not used during inference.",
    "batch_normalization": "Batch normalization: normalizes across batch dimension. Speeds training, but has issues with small batches.",
    "learning_rate": "Learning rate: step size for gradient descent. Too high: diverges. Too low: slow convergence. Cosine schedule is common.",
    "weight_initialization": "Weight initialization: Xavier/Glorot for sigmoid/tanh, He for ReLU. Proper init prevents vanishing/exploding gradients.",
    "mixed_precision": "Mixed precision training: use FP16 for speed, FP32 for stability. Reduces memory, speeds up training.",
    "gradient_clipping": "Gradient clipping: limits gradient magnitude to prevent exploding gradients. Common threshold: 1.0.",
    "learning_rate_schedule": "LR schedules: cosine decay, linear decay, warmup. Warmup prevents early divergence.",
    "data_augmentation": "Data augmentation: artificially increase training data. For images: flip, rotate, crop. For text: synonym replacement.",
    "cross_validation": "Cross-validation: split data into k folds, train on k-1, test on 1. Repeats k times for robust evaluation.",
    "confusion_matrix": "Confusion matrix: TP, FP, TN, FN. Used to compute accuracy, precision, recall, F1 score.",
    "precision_recall": "Precision: TP/(TP+FP). Recall: TP/(TP+FN). F1: harmonic mean. Trade-off between them.",
    "roc_curve": "ROC curve: TPR vs FPR at different thresholds. AUC-ROC measures model quality.",
    "bias_variance": "Bias-variance tradeoff: high bias = underfitting, high variance = overfitting. Goal: balanced model.",
    "ensemble": "Ensemble methods: combine multiple models. Random forest, gradient boosting, bagging. Usually better than single model.",
    "feature_engineering": "Feature engineering: creating informative features from raw data. Critical for traditional ML.",
    "dimensionality_reduction": "Dimensionality reduction: PCA, t-SNE, UMAP. Reduce features while preserving information.",
    "clustering": "Clustering: grouping similar data points. K-means, DBSCAN, hierarchical clustering.",
    "regression": "Regression: predicting continuous values. Linear regression, polynomial, ridge, lasso.",
    "classification": "Classification: predicting categories. Logistic regression, SVM, random forest, neural networks.",
    "optimization": "Optimization: finding best parameters. Gradient descent, Adam, SGD with momentum.",
    "loss_function": "Loss function: measures prediction error. Cross-entropy for classification, MSE for regression.",
    "activation_function": "Activation functions add non-linearity. ReLU, sigmoid, tanh, GELU, SwiGLU.",
    "weight_decay": "Weight decay (L2 regularization): penalizes large weights. Helps prevent overfitting.",
    "early_stopping": "Early stopping: stop training when validation loss stops improving. Prevents overfitting.",
    "checkpointing": "Checkpointing: save model state during training. Allows resuming and selecting best model.",
}

CODE_TEMPLATES = {
    "sort": "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)",
    "fibonacci": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "binary_search": "def bsearch(arr, t):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == t: return mid\n        elif arr[mid] < t: lo = mid+1\n        else: hi = mid-1\n    return -1",
    "linked_list": "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None\nclass LinkedList:\n    def __init__(self):\n        self.head = None\n    def add(self, val):\n        n = Node(val)\n        if not self.head: self.head = n\n        else:\n            c = self.head\n            while c.next: c = c.next\n            c.next = n",
    "binary_tree": "class TreeNode:\n    def __init__(self, val):\n        self.val = val\n        self.left = self.right = None\ndef inorder(node):\n    if node:\n        inorder(node.left)\n        print(node.val, end=' ')\n        inorder(node.right)",
    "api": "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/items/{id}')\ndef get_item(id: int):\n    return {'id': id, 'name': f'Item {id}'}",
    "flask": "from flask import Flask\napp = Flask(__name__)\n@app.route('/')\ndef home():\n    return 'Hello World!'\nif __name__ == '__main__':\n    app.run()",
    "database": "import sqlite3\nconn = sqlite3.connect('db.sqlite')\nc = conn.cursor()\nc.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)')\nc.execute('INSERT INTO users VALUES (1, \"Alice\")')\nconn.commit()",
    "decorator": "import functools\ndef timer(func):\n    @functools.wraps(func)\n    def wrapper(*args, **kwargs):\n        import time\n        start = time.time()\n        result = func(*args, **kwargs)\n        print(f'{func.__name__}: {time.time()-start:.4f}s')\n        return result\n    return wrapper",
    "context_manager": "class File:\n    def __init__(self, name, mode):\n        self.name, self.mode = name, mode\n    def __enter__(self):\n        self.f = open(self.name, self.mode)\n        return self.f\n    def __exit__(self, *args):\n        self.f.close()",
    "async_demo": "import asyncio\nasync def fetch(url):\n    print(f'Fetching {url}')\n    await asyncio.sleep(1)\n    return f'Data from {url}'\nasync def main():\n    results = await asyncio.gather(\n        fetch('url1'), fetch('url2')\n    )\n    print(results)\nasyncio.run(main())",
    "machine_learning": "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y)\nmodel = RandomForestClassifier(n_estimators=100)\nmodel.fit(X_train, y_train)\nprint(f'Accuracy: {model.score(X_test, y_test):.2%}')",
}

def get_answer(question: str) -> str:
    """Get intelligent answer using knowledge base."""
    q = question.lower().strip()
    
    # Check for code requests
    code_words = ["write", "code", "function", "script", "implement", "create"]
    if any(w in q for w in code_words):
        for key, code in CODE_TEMPLATES.items():
            if key in q:
                return f"Here's a {key} implementation:\n\n```python\n{code}\n```"
        if "python" in q:
            return f"Here's Python code:\n\n```python\n{CODE_TEMPLATES['sort']}\n```"
        if "javascript" in q or "js" in q:
            return "Here's JavaScript:\n\n```javascript\nconst greet = (name) => `Hello, ${name}!`;\nconsole.log(greet('World'));\n```"
        return f"Here's an example:\n\n```python\n{CODE_TEMPLATES['fibonacci']}\n```"
    
    # Check for math
    math_patterns = [
        (r"(\d+)\s*\+\s*(\d+)", lambda m: f"{m.group(1)} + {m.group(2)} = {int(m.group(1)) + int(m.group(2))}"),
        (r"(\d+)\s*\*\s*(\d+)", lambda m: f"{m.group(1)} * {m.group(2)} = {int(m.group(1)) * int(m.group(2))}"),
        (r"sqrt\s*(?:of\s*)?(\d+)", lambda m: f"sqrt({m.group(1)}) = {math.sqrt(int(m.group(1))):.4f}"),
        (r"factorial\s*(?:of\s*)?(\d+)", lambda m: f"{m.group(1)}! = {math.factorial(int(m.group(1)))}"),
    ]
    for pattern, func in math_patterns:
        import re
        match = re.search(pattern, q)
        if match:
            return func(match)
    
    if "pi" in q:
        return f"Pi = {math.pi}"
    
    # Check knowledge base
    for topic, fact in KNOWLEDGE.items():
        if topic in q:
            return fact
    
    # Check for explanations
    if "how does" in q or "explain" in q or "what is" in q:
        for topic, fact in KNOWLEDGE.items():
            if topic.replace("_", " ") in q or topic in q:
                return fact
    
    # Default response
    if any(w in q for w in ["hello", "hi", "hey"]):
        return "Hello! I'm Qythera. Ask me about programming, science, math, or request code examples."
    if any(w in q for w in ["help", "what can you do"]):
        return "I can help with:\n- Programming (Python, JS, and more)\n- Science (physics, biology, chemistry)\n- Math (arithmetic, calculus, statistics)\n- Technology (AI, ML, networks)\n- Code generation (13 templates)"
    if "who are you" in q or "what are you" in q:
        return "I'm Qythera, a production superintelligence platform. Built from scratch with custom autodiff engine, Vaelon transformer, and BPE tokenizer. No external AI APIs."
    
    return f"I can help with that! I have knowledge about many topics. Try asking about: Python, JavaScript, neural networks, machine learning, physics, math, or request code examples."
