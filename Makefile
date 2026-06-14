
.PHONY: install serve web train test
install: pip install -e .
serve: python -m inference.server
web: cd web && npm install && npm run dev
train: python -m training.trainer
test: python -m pytest tests/ -v
lint: python -m ruff check core/ inference/ training/ cli/
