.PHONY: install serve web chat test clean help

help:
	@echo "Qythera - Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make serve       - Start inference server"
	@echo "  make web         - Start web UI"
	@echo "  make chat        - Start CLI chat"
	@echo "  make test        - Run tests"
	@echo "  make clean       - Clean cache"

install:
	pip install numpy
	cd web && npm install 2>/dev/null || true

serve:
	python3 -m core.inference.server --port 8080

web:
	cd web && npm run dev

chat:
	python3 cli/main.py chat

test:
	python3 -m pytest tests/ -v 2>/dev/null || python3 -c "import sys; sys.path.insert(0,'.'); exec(open('tests/test_all.py').read())"

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
	find . -name "*.pyc" -delete 2>/dev/null
