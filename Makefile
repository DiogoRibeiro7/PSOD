.PHONY: help install install-dev test test-cov lint format type-check clean build docs

help:
	@echo "Available commands:"
	@echo "  install      Install the package"
	@echo "  install-dev  Install with development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  type-check   Run type checking"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build distribution"
	@echo "  docs         Build documentation"
	# TODO: Add more commands for release, benchmarks, etc.

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,viz,docs]"
	# TODO: Install pre-commit hooks
	# pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=psod --cov-report=html --cov-report=term
	# TODO: Open coverage report in browser
	# @echo "Opening coverage report..."
	# @python -m webbrowser htmlcov/index.html

# TODO: Add test-slow for slow tests
# test-slow:
# 	pytest tests/ -v -m slow

# TODO: Add test-integration
# test-integration:
# 	pytest tests/integration/ -v

lint:
	flake8 src tests
	# TODO: Add ruff linting
	# ruff check src tests

format:
	black src tests examples
	isort src tests examples

type-check:
	# TODO: Enable once type hints are complete
	# mypy src

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

# TODO: Add docs building
# docs:
# 	cd docs && make clean && make html
# 	@echo "Documentation built in docs/_build/html"

# TODO: Add release command
# release: clean test
# 	# Bump version
# 	# Build distribution
# 	# Upload to PyPI
# 	@echo "TODO: Implement release process"

# TODO: Add benchmark command
# benchmark:
# 	python benchmarks/run_benchmarks.py

# TODO: Add profile command for profiling
# profile:
# 	python -m cProfile -o profile.stats examples/profile_example.py
# 	python -m pstats profile.stats

# TODO: Add security check
# security:
# 	bandit -r src/
# 	safety check

# TODO: Add Docker commands
# docker-build:
# 	docker build -t psod .
# 
# docker-run:
# 	docker run -it --rm psod

# TODO: Add notebook examples
# notebooks:
# 	jupyter notebook examples/notebooks/

# Development workflow commands
dev-check: format lint type-check test
	@echo "All development checks passed!"

# TODO: Add pre-commit command
# pre-commit:
# 	pre-commit run --all-files
