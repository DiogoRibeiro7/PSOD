# Contributing to PSOD

Thank you for your interest in contributing to PSOD! This document provides guidelines and instructions for contributing.

## Code of Conduct

# TODO: Create CODE_OF_CONDUCT.md based on Contributor Covenant
Please read and follow our Code of Conduct.

## How to Contribute

### Reporting Bugs

# TODO: Create bug report issue template in .github/ISSUE_TEMPLATE/
Before creating bug reports, please check existing issues. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- System information (OS, Python version, package versions)

### Suggesting Enhancements

# TODO: Create feature request issue template
Enhancement suggestions are tracked as GitHub issues. Include:

- A clear and descriptive title
- Detailed description of the proposed enhancement
- Use cases and examples
- Possible implementation approach

### Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Add or update tests as needed
5. Update documentation
6. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/outlier_pseudo_supervised.git
cd outlier_pseudo_supervised

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# TODO: Add pre-commit hook setup instructions
# pre-commit install
```

## Testing

# TODO: Implement comprehensive test suite
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=psod --cov-report=html

# Run specific test file
pytest tests/test_psod.py
```

## Code Style

# TODO: Set up black, flake8, and mypy configuration
We use:
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking

```bash
# Format code
black src/

# Check linting
flake8 src/

# Check types
mypy src/
```

## Documentation

# TODO: Set up Sphinx documentation
Documentation is built using Sphinx:

```bash
cd docs/
make html
```

## Commit Guidelines

# TODO: Add commitizen or similar tool for conventional commits
Follow conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or modifications
- `refactor:` Code refactoring
- `style:` Code style changes
- `chore:` Maintenance tasks

## Release Process

# TODO: Document release process and versioning strategy
Maintainers follow semantic versioning and release through GitHub Actions.
