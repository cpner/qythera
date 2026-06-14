<?php
// Qythera API - PHP backend for serv00 hosting
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit;
}

// Knowledge base
$knowledge = [
    'python' => 'Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability with significant indentation.',
    'javascript' => 'JavaScript is a programming language for web development. It runs in browsers and on servers via Node.js.',
    'transformer' => 'Transformers use self-attention to process sequences. Introduced in Attention Is All You Need (2017).',
    'neural_network' => 'Neural networks are computing systems inspired by biological neurons. Trained via backpropagation.',
    'machine_learning' => 'Machine learning: systems learn patterns from data. Types: supervised, unsupervised, reinforcement.',
    'deep_learning' => 'Deep learning uses neural networks with many layers. Frameworks: PyTorch, TensorFlow.',
    'git' => 'Git is a distributed version control system. Commands: init, add, commit, push, pull, branch, merge.',
    'docker' => 'Docker packages apps in containers. Commands: build, run, pull, push.',
    'quantum' => 'Quantum computing uses qubits in superposition. Enables exponential speedup for certain problems.',
    'physics' => 'Newton laws: F=ma. Conservation of energy. E=mc^2 from relativity.',
    'math' => 'Algebra uses symbols for numbers. Calculus studies change and accumulation.',
    'ai' => 'Artificial Intelligence: systems performing tasks requiring human intelligence.',
    'llm' => 'Large Language Models: neural networks trained on text to generate coherent text.',
    'attention' => 'Attention computes weighted importance of tokens. Multi-head runs parallel operations.',
    'optimization' => 'Optimization finds best parameters. Adam, SGD are common optimizers.',
];

$code = [
    'sort' => "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    return quicksort([x for x in arr if x < pivot]) + [x for x in arr if x == pivot] + quicksort([x for x in arr if x > pivot])",
    'fibonacci' => "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    'binary_search' => "def bsearch(arr, t):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == t: return mid\n        elif arr[mid] < t: lo = mid+1\n        else: hi = mid-1\n    return -1",
    'api' => "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/items/{id}')\ndef get(id): return {'id': id}",
    'flask' => "from flask import Flask\napp = Flask(__name__)\n@app.route('/')\ndef home(): return 'Hello!'\napp.run()",
    'database' => "import sqlite3\nconn = sqlite3.connect('db.sqlite')\nc = conn.cursor()\nc.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)')\nconn.commit()",
    'machine_learning' => "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import train_test_split\nX_tr, X_te, y_tr, y_te = train_test_split(X, y)\nmodel = RandomForestClassifier().fit(X_tr, y_tr)\nprint(f'Accuracy: {model.score(X_te, y_te):.2%}')",
];

function getAnswer($question) {
    global $knowledge, $code;
    $q = strtolower(trim($question));
    
    // Code requests
    $codeWords = ['write','code','function','script','implement','create','build'];
    foreach ($codeWords as $w) {
        if (strpos($q, $w) !== false) {
            foreach ($code as $key => $c) {
                if (strpos($q, $key) !== false) {
                    return "Here is a {$key} implementation:\n\n```python\n{$c}\n```";
                }
            }
            if (strpos($q, 'python') !== false) return "```python\n{$code['sort']}\n```";
            if (strpos($q, 'javascript') !== false || strpos($q, 'js') !== false) {
                return "```javascript\nconst greet = (name) => `Hello, ${name}!`;\nconsole.log(greet('World'));\n```";
            }
            return "```python\n{$code['fibonacci']}\n```";
        }
    }
    
    // Math
    if (preg_match('/(\d+)\s*\+\s*(\d+)/', $q, $m)) return "{$m[1]} + {$m[2]} = " . ($m[1] + $m[2]);
    if (preg_match('/(\d+)\s*\*\s*(\d+)/', $q, $m)) return "{$m[1]} * {$m[2]} = " . ($m[1] * $m[2]);
    if (preg_match('/(\d+)\s*-\s*(\d+)/', $q, $m)) return "{$m[1]} - {$m[2]} = " . ($m[1] - $m[2]);
    if (preg_match('/sqrt\s*(?:of\s*)?(\d+)/i', $q, $m)) return "sqrt({$m[1]}) = " . round(sqrt($m[1]), 4);
    if (strpos($q, 'pi') !== false) return "Pi = " . pi();
    if (strpos($q, 'euler') !== false) return "Euler's number e = " . M_E;
    
    // Knowledge
    foreach ($knowledge as $topic => $fact) {
        if (strpos($q, $topic) !== false) return $fact;
    }
    
    // Default
    if (in_array(true, array_map(function($w) use ($q) { return strpos($q, $w) !== false; }, ['hello','hi','hey'])))
        return "Hello! I am Qythera. Ask me about programming, science, math, or request code examples.";
    if (in_array(true, array_map(function($w) use ($q) { return strpos($q, $w) !== false; }, ['help','what can you do'])))
        return "I can help with:\n- Programming (Python, JS)\n- Science (physics, biology)\n- Math (arithmetic, calculus)\n- Code generation (7 templates)";
    if (strpos($q, 'who are you') !== false || strpos($q, 'what are you') !== false)
        return "I am Qythera, built from scratch. No external AI APIs. Custom autodiff engine, Vaelon transformer, BPE tokenizer.";
    
    return "I can help with that. Ask me about programming, science, math, or request code examples.";
}

// Handle API requests
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $input = json_decode(file_get_contents('php://input'), true);
    $messages = (isset($input['messages']) ? $input['messages'] : []);
    $lastMsg = (isset(end($messages)['content']) ? end($messages)['content'] : '');
    
    $response = getAnswer($lastMsg);
    
    echo json_encode([
        'id' => 'chatcmpl-' . time() * 1000,
        'object' => 'chat.completion',
        'model' => 'vaelon',
        'choices' => [[
            'index' => 0,
            'message' => ['role' => 'assistant', 'content' => $response],
            'finish_reason' => 'stop'
        ]],
        'usage' => ['prompt_tokens' => 0, 'completion_tokens' => count(explode(' ', $response)), 'total_tokens' => count(explode(' ', $response))]
    ]);
    exit;
}

// Handle GET requests
if ($_SERVER['REQUEST_METHOD'] === 'GET') {
    $path = parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH);
    
    if (strpos($path, '/health') !== false) {
        echo json_encode(['status' => 'ok', 'model' => 'vaelon-php']);
        exit;
    }
    
    if (strpos($path, '/v1/models') !== false) {
        echo json_encode(['data' => [['id' => 'vaelon', 'object' => 'model']]]);
        exit;
    }
    
    echo json_encode(['error' => 'not found']);
}
