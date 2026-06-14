.PHONY: install dev-backend dev-frontend docker-up train test lint format clean help

help:
	@echo "Qythera - Available commands:"
	@echo "  make install        - Install all dependencies"
	@echo "  make dev-backend   - Start backend in development mode"
	@echo "  make dev-frontend  - Start frontend in development mode"
	@echo "  make docker-up     - Start all services with Docker"
	@echo "  make train         - Start training with default config"
	@echo "  make test          - Run test suite"
	@echo "  make lint          - Run linter"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean build artifacts"

install:
	pip install -e .
	cd web && npm install

dev-backend:
	python3 -m inference.server --port 8000

dev-frontend:
	cd web && npm run dev

docker-up:
	docker compose -f infra/docker-compose.yml up -d

train:
	python3 training/pretrain/train_pretrain.py

test:
	pytest tests/ -v

lint:
	ruff check vaelon/ training/ inference/ memory/ agent/ safety/ cli/

format:
	ruff format vaelon/ training/ inference/ memory/ agent/ safety/ cli/

clean:
	rm -rf __pycache__ .pytest_cache outputs/ data/
	find . -name "*.pyc" -delete
