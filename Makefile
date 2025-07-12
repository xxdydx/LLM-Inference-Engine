.PHONY: help install install-dev test lint format clean run

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

test: ## Run tests
	pytest tests/ -v

lint: ## Run linting
	flake8 src/ main.py tests/
	mypy src/ main.py

format: ## Format code
	black src/ main.py tests/

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

run: ## Run the inference engine
	python main.py

setup: install-dev ## Setup development environment
	mkdir -p models
	@echo "Development environment setup complete!"
	@echo "Please place your ONNX model in models/gpt2.onnx" 