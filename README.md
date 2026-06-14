<div align="center">

# ✦ Qythera

### Production Superintelligence Platform

**Works on ANY device. No server required. No external APIs.**

[![License: MIT](https://img.shields.io/badge/License-MIT-7c3aed.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-3b82f6.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-ALL-a78bfa.svg)](#installation)

</div>

---

## Quick Start

### 1. Standalone (works everywhere, no server)
```bash
# Download and open in any browser
curl -fsSL https://raw.githubusercontent.com/cpner/qythera/main/web/standalone.html -o qythera.html
# Open qythera.html in any browser
```

### 2. Python server
```bash
git clone https://github.com/cpner/qythera.git
cd qythera && pip install numpy
python3 -m core.inference.server --port 8080
```

### 3. serv00 (PHP hosting)
```bash
git clone https://github.com/cpner/qythera.git
cp -r qythera/php/* ~/public_html/
# Open: https://yourname.serv00.net/index.html
```

---

## Installation by Platform

| Platform | Command |
|----------|---------|
| **Linux** | `curl -fsSL https://raw.githubusercontent.com/cpner/qythera/main/install.sh \| bash` |
| **macOS** | Same as Linux |
| **Windows** | Run `install.bat` or `install.ps1` |
| **FreeBSD/serv00** | `git clone ... && cp -r php/* ~/public_html/` |
| **Android (Termux)** | `pkg install python git && git clone ...` |
| **Docker** | `docker build -t qythera . && docker run -p 8080:8080 qythera` |
| **Any browser** | Open `web/standalone.html` |

---

## Features

- **50+ knowledge topics** (Python, ML, physics, math, biology...)
- **14 code templates** (sort, API, database, async, ML...)
- **Math engine** (arithmetic, sqrt, factorial, pi)
- **Safety filters** (toxicity, jailbreak, PII)
- **Glassmorphism dark theme** (mobile-responsive)
- **Standalone HTML** (works without server)
- **PHP backend** (for serv00 hosting)
- **Docker support**

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `Permission denied` | Use port 8080: `python3 -m core.inference.server --port 8080` |
| `No module named numpy` | `pip3 install numpy` |
| `Address already in use` | Use different port: `--port 9000` |
| `Command not found: python3` | Try `python -m core.inference.server` |
| `git pull fails` | `rm -rf core/ && git pull` |
| serv00 404 | Copy `php/` contents to `~/public_html/` |

---

## License

MIT
