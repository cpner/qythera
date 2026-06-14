.PHONY: install serve web chat test
install: pip install -e . && cd web && npm install
serve: python -m core.inference.server
web: cd web && npm install && npm run dev
chat: python cli/main.py chat
test: python -m pytest tests/ -v
