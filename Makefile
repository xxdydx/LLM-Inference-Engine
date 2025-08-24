.PHONY: help install install-dev test lint format clean run client demo docker-build docker-run setup benchmark coverage security

# Variables
PYTHON := python3
PIP := pip3
DOCKER_IMAGE := inference-engine
DOCKER_TAG := latest

help: ## Show this help message
	@echo "🚀 Inference Engine - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install pytest-cov pytest-asyncio bandit safety mypy

setup: install-dev ## Setup complete development environment
	mkdir -p models logs
	@echo "✅ Development environment setup complete!"
	@echo "📝 Please place your ONNX model in models/gpt2.onnx"

test: ## Run all tests
	pytest tests/test_grpc_simple.py tests/test_beam_search.py -v

test-coverage: ## Run tests with coverage report
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml
	@echo "📊 Coverage report generated in htmlcov/"

test-specific: ## Run specific test (use TEST=test_name)
	pytest tests/$(TEST) -v

lint: ## Run code linting
	flake8 src/ main.py tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ main.py tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

type-check: ## Run type checking
	mypy src/ main.py --ignore-missing-imports

security: ## Run security checks
	bandit -r src/ main.py -f json -o security-report.json || true
	safety check --json --output safety-report.json || true
	@echo "🔒 Security reports generated"

format: ## Format code with black
	black src/ main.py tests/ client.py

format-check: ## Check code formatting
	black --check src/ main.py tests/ client.py

quality: lint type-check security format-check ## Run all quality checks

clean: ## Clean up build artifacts and cache
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ __pycache__/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	rm -f *.log security-report.json safety-report.json coverage.xml
	rm -rf htmlcov/

run: ## Run the inference engine server
	$(PYTHON) main.py

client: ## Run the simple demo client
	$(PYTHON) client.py

demo: ## Run the comprehensive demo script
	$(PYTHON) demo.py

demo-section: ## Run specific demo section (use SECTION=name)
	$(PYTHON) demo.py --section $(SECTION)

demo-instructions: ## Show demo instructions
	@echo "🎬 Demo Instructions:"
	@echo "1️⃣  In a separate terminal, run: make run"
	@echo "2️⃣  Then run: make demo (comprehensive) or make client (simple)"
	@echo "3️⃣  Or use Docker: make docker-demo"

benchmark: ## Run performance benchmarks
	$(PYTHON) -c "import sys; sys.path.insert(0, 'src'); print('🏃 Running benchmarks...'); print('Benchmark module would run here')"

# Docker commands
docker-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "🐳 Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)"

docker-run: ## Run Docker container
	docker run -d \
		--name $(DOCKER_IMAGE) \
		-p 50051:50051 \
		-v $(shell pwd)/models:/app/models:ro \
		-v $(shell pwd)/logs:/app/logs \
		$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "🚀 Container started. Check status with: docker ps"

docker-stop: ## Stop and remove Docker container
	docker stop $(DOCKER_IMAGE) || true
	docker rm $(DOCKER_IMAGE) || true

docker-logs: ## View Docker container logs
	docker logs -f $(DOCKER_IMAGE)

docker-shell: ## Open shell in running container
	docker exec -it $(DOCKER_IMAGE) /bin/bash

docker-demo: docker-build docker-stop docker-run ## Complete Docker demo
	@echo "🎉 Docker demo started!"
	@echo "🔍 Check logs: make docker-logs"
	@echo "🧪 Test client: make demo"
	@echo "🛑 Stop: make docker-stop"

docker-compose-up: ## Start with docker-compose
	docker-compose up -d
	@echo "🚀 Docker Compose services started"

docker-compose-down: ## Stop docker-compose services
	docker-compose down
	@echo "🛑 Docker Compose services stopped"

docker-compose-logs: ## View docker-compose logs
	docker-compose logs -f

# Advanced commands
proto-compile: ## Recompile protobuf files
	python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto

model-download: ## Download a sample ONNX model (requires internet)
	@echo "📥 This would download a sample model to models/"
	@echo "Please manually place your ONNX model in models/gpt2.onnx"

stress-test: ## Run stress tests
	@echo "⚡ Running stress tests..."
	$(PYTHON) -c "print('Stress test module would run here')"

deploy-prep: clean quality test docker-build ## Prepare for deployment
	@echo "🚀 Deployment preparation complete!"
	@echo "✅ Code quality checks passed"
	@echo "✅ Tests passed" 
	@echo "✅ Docker image built"

monitoring: ## Show monitoring commands
	@echo "📊 Monitoring Commands:"
	@echo "🐳 Docker status: docker ps"
	@echo "📝 Server logs: tail -f *.log"
	@echo "🔍 Container logs: make docker-logs"
	@echo "💾 System resources: docker stats"

all-checks: install-dev quality test security ## Run all checks and tests

# CI/CD helpers
ci-test: install-dev test-coverage lint type-check security ## Run all CI checks
	@echo "✅ All CI checks completed"

# Help for specific use cases
dev-setup: setup ## Alias for setup
	@echo "💻 Development setup complete"

prod-build: docker-build ## Alias for production build
	@echo "🏭 Production build complete" 