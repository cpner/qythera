.PHONY: install serve web train test
install: pip install -e . && cd web && npm install
serve: python -m inference.server
web: cd web && npm run dev
train: python -c "from training.trainer import Trainer; Trainer().train('data/training.json')"
test: python -m pytest tests/ -v
