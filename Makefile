# PSOD Development Makefile
# ========================
# Comprehensive development commands for PSOD project

.PHONY: help install install-dev install-all test test-cov test-fast test-slow test-integration \
        lint format type-check security clean build docs docs-serve docs-deploy \
        benchmark profile pre-commit docker-build docker-run docker-dev docker-test \
        notebooks release release-test check-deps update-deps dev-setup dev-check ci-local \
        validate all

# Default target
.DEFAULT_GOAL := help

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
YELLOW := \033[0;33m
NC := \033[0m # No Color

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy
BANDIT := bandit
DOCKER := docker
JUPYTER := jupyter

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "$(BLUE)PSOD Development Makefile$(NC)"
	@echo "$(BLUE)=========================$(NC)"
	@echo ""
	@echo "$(GREEN)Installation:$(NC)"
	@grep -E '^install[a-z-]*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@grep -E '^test[a-z-]*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Code Quality:$(NC)"
	@grep -E '^(lint|format|type-check|security|pre-commit):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Documentation:$(NC)"
	@grep -E '^docs[a-z-]*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Performance:$(NC)"
	@grep -E '^(benchmark|profile):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Docker:$(NC)"
	@grep -E '^docker[a-z-]*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Release:$(NC)"
	@grep -E '^release[a-z-]*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Utilities:$(NC)"
	@grep -E '^(clean|check-deps|update-deps|dev-setup|dev-check|ci-local|validate|all):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# Installation
# =============================================================================

install: ## Install package for production use
	@echo "$(BLUE)Installing PSOD...$(NC)"
	$(PIP) install -e .

install-dev: ## Install with development dependencies
	@echo "$(BLUE)Installing PSOD with development dependencies...$(NC)"
	$(PIP) install -e ".[dev,test,viz,docs]"
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

install-all: install-dev ## Install all dependencies including pre-commit hooks
	@echo "$(BLUE)Setting up pre-commit hooks...$(NC)"
	pre-commit install
	@echo "$(GREEN)✓ Pre-commit hooks installed$(NC)"

dev-setup: install-all ## Complete development environment setup
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@echo "Creating necessary directories..."
	mkdir -p logs
	mkdir -p data
	mkdir -p models
	mkdir -p reports
	@echo "$(GREEN)✓ Development environment ready!$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Run 'make test' to verify installation"
	@echo "  2. Run 'make dev-check' before committing"
	@echo "  3. See 'make help' for all available commands"

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTEST) tests/ -v --tb=short

test-fast: ## Run fast tests only (exclude slow tests)
	@echo "$(BLUE)Running fast tests...$(NC)"
	$(PYTEST) tests/ -v -m "not slow" --tb=short

test-slow: ## Run slow tests only
	@echo "$(BLUE)Running slow tests...$(NC)"
	$(PYTEST) tests/ -v -m "slow" --tb=short

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTEST) tests/test_integration.py -v --tb=short

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) tests/ -v \
		--cov=src/psod \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-fail-under=80
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	$(PYTEST) tests/ -v --looponfail

test-debug: ## Run tests with debugging enabled
	@echo "$(BLUE)Running tests with debugging...$(NC)"
	$(PYTEST) tests/ -v --pdb --pdbcls=IPython.terminal.debugger:Pdb

test-parallel: ## Run tests in parallel
	@echo "$(BLUE)Running tests in parallel...$(NC)"
	$(PYTEST) tests/ -v -n auto --dist loadscope

test-examples: ## Run example scripts to verify they work
	@echo "$(BLUE)Testing example scripts...$(NC)"
	$(PYTHON) examples/basic_usage.py
	@echo "$(GREEN)✓ Examples verified$(NC)"

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run all linters
	@echo "$(BLUE)Running linters...$(NC)"
	@echo "→ flake8..."
	$(FLAKE8) src/ tests/ examples/ --count --statistics
	@echo "→ pylint..."
	pylint src/psod/ --fail-under=8.0 || echo "$(YELLOW)Warning: pylint score below 8.0$(NC)"
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	$(BLACK) src/ tests/ examples/ benchmarks/
	$(ISORT) src/ tests/ examples/ benchmarks/
	@echo "$(GREEN)✓ Code formatted$(NC)"

format-check: ## Check code formatting without modifying
	@echo "$(BLUE)Checking code formatting...$(NC)"
	$(BLACK) --check src/ tests/ examples/ benchmarks/
	$(ISORT) --check-only src/ tests/ examples/ benchmarks/

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checker...$(NC)"
	$(MYPY) src/psod/ --ignore-missing-imports --check-untyped-defs || echo "$(YELLOW)Type check warnings$(NC)"

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	@echo "→ bandit..."
	$(BANDIT) -r src/psod/ -c .bandit -ll
	@echo "→ pip-audit..."
	pip-audit || echo "$(YELLOW)Audit warnings found$(NC)"
	@echo "$(GREEN)✓ Security scan complete$(NC)"

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

validate: lint type-check test ## Run all validation checks
	@echo "$(GREEN)✓ All validation checks passed!$(NC)"

# =============================================================================
# Documentation
# =============================================================================

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	cd docs && make clean && make html
	@echo "$(GREEN)✓ Documentation built in docs/_build/html/$(NC)"

docs-serve: docs ## Build and serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

docs-deploy: docs ## Build and deploy documentation to GitHub Pages
	@echo "$(BLUE)Deploying documentation...$(NC)"
	@echo "$(YELLOW)Use GitHub Actions for deployment$(NC)"

docs-coverage: ## Check documentation coverage
	@echo "$(BLUE)Checking documentation coverage...$(NC)"
	interrogate -v src/psod/ --fail-under 80

docs-linkcheck: ## Check documentation for broken links
	@echo "$(BLUE)Checking documentation links...$(NC)"
	cd docs && make linkcheck

# =============================================================================
# Performance & Benchmarking
# =============================================================================

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	cd benchmarks && $(PYTHON) run_benchmarks.py
	@echo "$(GREEN)✓ Benchmarks complete$(NC)"

benchmark-compare: ## Compare benchmark results
	@echo "$(BLUE)Comparing benchmark results...$(NC)"
	cd benchmarks && $(PYTHON) run_benchmarks.py --compare
	@echo "$(GREEN)✓ Comparison complete$(NC)"

profile: ## Profile code execution
	@echo "$(BLUE)Profiling code...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats examples/basic_usage.py
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
	@echo "$(GREEN)✓ Profile saved to profile.stats$(NC)"

profile-memory: ## Profile memory usage
	@echo "$(BLUE)Profiling memory usage...$(NC)"
	$(PYTHON) -m memory_profiler examples/basic_usage.py

# =============================================================================
# Docker
# =============================================================================

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	$(DOCKER) build -t psod:latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-build-dev: ## Build development Docker image
	@echo "$(BLUE)Building development Docker image...$(NC)"
	$(DOCKER) build -f docker/Dockerfile.dev -t psod:dev .
	@echo "$(GREEN)✓ Development Docker image built$(NC)"

docker-run: docker-build ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	$(DOCKER) run -it --rm psod:latest

docker-dev: docker-build-dev ## Run development Docker container
	@echo "$(BLUE)Running development Docker container...$(NC)"
	$(DOCKER) run -it --rm \
		-v $(PWD):/app \
		-p 8888:8888 \
		psod:dev

docker-test: docker-build ## Run tests in Docker
	@echo "$(BLUE)Running tests in Docker...$(NC)"
	$(DOCKER) run --rm psod:latest pytest tests/

docker-jupyter: docker-build-dev ## Run Jupyter in Docker
	@echo "$(BLUE)Starting Jupyter in Docker...$(NC)"
	$(DOCKER) run -it --rm \
		-v $(PWD):/app \
		-p 8888:8888 \
		psod:dev \
		jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root

docker-clean: ## Remove Docker images
	@echo "$(BLUE)Cleaning Docker images...$(NC)"
	$(DOCKER) rmi psod:latest psod:dev || true
	@echo "$(GREEN)✓ Docker images removed$(NC)"

# =============================================================================
# Utilities
# =============================================================================

clean: ## Clean build artifacts and caches
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf profile.stats
	rm -rf *.png *.jpg *.pdf
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

build: clean ## Build distribution packages
	@echo "$(BLUE)Building distribution packages...$(NC)"
	$(PYTHON) -m build
	twine check dist/*
	@echo "$(GREEN)✓ Distribution built in dist/$(NC)"

check-deps: ## Check for outdated dependencies
	@echo "$(BLUE)Checking dependencies...$(NC)"
	$(PIP) list --outdated
	@echo "$(GREEN)✓ Dependency check complete$(NC)"

update-deps: ## Update all dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e ".[dev,test,viz,docs]"
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

notebooks: ## Start Jupyter notebooks
	@echo "$(BLUE)Starting Jupyter notebooks...$(NC)"
	$(JUPYTER) notebook examples/notebooks/

ci-local: ## Run CI checks locally
	@echo "$(BLUE)Running CI checks locally...$(NC)"
	@echo "→ Format check..."
	@$(MAKE) format-check
	@echo "→ Linting..."
	@$(MAKE) lint
	@echo "→ Type checking..."
	@$(MAKE) type-check
	@echo "→ Security..."
	@$(MAKE) security
	@echo "→ Tests..."
	@$(MAKE) test-cov
	@echo "$(GREEN)✓ All CI checks passed!$(NC)"

dev-check: format lint type-check test ## Run all development checks
	@echo "$(GREEN)✓ All development checks passed!$(NC)"

# =============================================================================
# Release
# =============================================================================

release-test: clean build ## Build and upload to TestPyPI
	@echo "$(BLUE)Uploading to TestPyPI...$(NC)"
	twine upload --repository testpypi dist/*
	@echo "$(GREEN)✓ Uploaded to TestPyPI$(NC)"
	@echo "Test install with:"
	@echo "  pip install --index-url https://test.pypi.org/simple/ psod"

release: clean build ## Build and upload to PyPI (requires confirmation)
	@echo "$(RED)WARNING: This will upload to PyPI!$(NC)"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read dummy
	@echo "$(BLUE)Uploading to PyPI...$(NC)"
	twine upload dist/*
	@echo "$(GREEN)✓ Released to PyPI!$(NC)"

release-check: ## Check if ready for release
	@echo "$(BLUE)Checking release readiness...$(NC)"
	@echo "→ Running tests..."
	@$(MAKE) test-cov
	@echo "→ Building package..."
	@$(MAKE) build
	@echo "→ Checking package..."
	twine check dist/*
	@echo "$(GREEN)✓ Ready for release!$(NC)"

# =============================================================================
# Composite targets
# =============================================================================

all: clean install-dev test docs ## Clean, install, test, and build docs
	@echo "$(GREEN)✓ Full build complete!$(NC)"
