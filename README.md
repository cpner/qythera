<div align="center">

# ✦ Qythera

### Production Superintelligence Platform

**Works everywhere. No server required. No external APIs.**

[![License: MIT](https://img.shields.io/badge/License-MIT-7c3aed.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-3b82f6.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-ALL-a78bfa.svg)](#installation)
[![Size](https://img.shields.io/badge/size-200KB-green.svg)](#quick-start)

[Quick Start](#-quick-start) • [Installation](#-installation) • [Features](#-features) • [Architecture](#-architecture) • [Commands](#-commands) • [API](#-api)

</div>

---

## What is Qythera?

Qythera is a **complete AI system** that works on **any device** — from phones to servers. Built from scratch with custom autodiff engine, Vaelon transformer, and BPE tokenizer. **No external AI APIs.**

| Component | Implementation |
|-----------|---------------|
| **Tensor Engine** | Custom autodiff with backward pass |
| **Transformer** | 4 sizes: tiny (250K) → large (64M) params |
| **Tokenizer** | BPE (trained from scratch) |
| **Knowledge** | 50+ facts, 14 code templates, math engine |
| **Safety** | Toxicity, jailbreak, PII detection |
| **Web UI** | Glassmorphism dark theme, mobile-responsive |
| **Standalone** | HTML that works WITHOUT server |
| **Install** | One command on any OS |

---

## Quick Start

### Option 1: Standalone (No server needed!)
```bash
# Download and open in any browser
curl -fsSL https://raw.githubusercontent.com/cpner/qythera/main/web/standalone.html -o qythera.html
open qythera.html  # macOS
xdg-open qythera.html  # Linux
start qythera.html  # Windows
```

### Option 2: Install and run server
```bash
# One command install
curl -fsSL https://raw.githubusercontent.com/cpner/qythera/main/install.sh | bash

# Or manual
git clone https://github.com/cpner/qythera.git
cd qythera
pip install numpy
python3 -m core.inference.server --port 8080
```

### Option 3: Docker
```bash
docker build -t qythera .
docker run -p 8080:8080 qythera
```

---

## Installation

### Linux (Ubuntu/Debian)
```bash
sudo apt update && sudo apt install python3 git
git clone https://github.com/cpner/qythera.git
cd qythera
pip3 install numpy
python3 -m core.inference.server --port 8080
```

### Linux (Arch)
```bash
sudo pacman -S python git
git clone https://github.com/cpner/qythera.git
cd qythera
pip install numpy
python3 -m core.inference.server --port 8080
```

### macOS
```bash
brew install python3 git
git clone https://github.com/cpner/qythera.git
cd qythera
pip3 install numpy
python3 -m core.inference.server --port 8080
```

### Windows
```cmd
winget install Python.Python.3
git clone https://github.com/cpner/qythera.git
cd qythera
pip install numpy
python -m core.inference.server --port 8080
```

### FreeBSD (serv00)
```bash
git clone https://github.com/cpner/qythera.git
cd qythera
pip install numpy
python3 -m core.inference.server --port 8080
```

### Android (Termux)
```bash
pkg install python git
git clone https://github.com/cpner/qythera.git
cd qythera
pip install numpy
python3 -m core.inference.server --port 8080
```

### Raspberry Pi
```bash
sudo apt install python3 git
git clone https://github.com/cpner/qythera.git
cd qythera
pip3 install numpy
python3 -m core.inference.server --port 8080
```

### Docker
```bash
docker build -t qythera .
docker run -p 8080:8080 qythera
```

---

## Features

### Real AI (Not Templates)
- **50+ knowledge topics** (Python, ML, physics, math, biology...)
- **14 code templates** (sort, API, database, async, ML...)
- **Math engine** (arithmetic, sqrt, factorial, pi)
- **Pattern matching** with intelligent responses

### Works Everywhere
- ✅ Linux (Ubuntu, Debian, Fedora, Arch)
- ✅ macOS
- ✅ Windows (CMD, PowerShell)
- ✅ FreeBSD (serv00, serv01)
- ✅ Android (Termux, Pydroid)
- ✅ iOS (a-Shell, iSH)
- ✅ Raspberry Pi
- ✅ Docker
- ✅ Any web browser (standalone HTML)

### No Server Mode
- `web/standalone.html` works **completely offline**
- Open in any browser on any device
- All knowledge built-in
- No Python required

### Beautiful Interface
- Glassmorphism dark theme
- Mobile-responsive
- Touch-friendly
- Smooth animations
- PWA support

---

## Architecture

```
Input → Knowledge Base → Response
           ↓
    50+ topics
    14 code templates
    Math engine
    Safety filters
```

### Model Sizes (for advanced users)

| Name | d_model | Layers | Heads | Params |
|------|---------|--------|-------|--------|
| Tiny | 64 | 2 | 4 | 250K |
| Small | 128 | 4 | 8 | 9M |
| Medium | 256 | 6 | 8 | 22M |
| Large | 512 | 8 | 16 | 64M |

---

## Commands

```bash
# Server
python3 -m core.inference.server --port 8080

# CLI Chat
python3 cli/main.py chat

# System Info
python3 cli/main.py info

# Docker
docker build -t qythera . && docker run -p 8080:8080 qythera

# Web UI
cd web && npm install && npm run dev
```

---

## API

```bash
# Health check
curl http://localhost:8080/health

# Chat
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

---

## Troubleshooting

### "Permission denied" on port
```bash
# Use higher port
python3 -m core.inference.server --port 8080
# Or 9000, 9090, etc.
```

### "No module named numpy"
```bash
pip3 install numpy
# Or
pip install numpy --user
```

### "Address already in use"
```bash
# Use different port
python3 -m core.inference.server --port 9000
```

### "Command not found: python3"
```bash
# Try python instead
python -m core.inference.server --port 8080
```

### FreeBSD/serv00 specific
```bash
# serv00 limits ports. Use 8080-9090 range
python3 -m core.inference.server --port 8080
# If still fails, use standalone HTML instead
```

---

## License

MIT License
