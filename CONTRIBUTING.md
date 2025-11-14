# Contributing to PSOD

Thank you for your interest in contributing to PSOD (Pseudo-Supervised Outlier Detection)! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Familiarity with outlier detection and machine learning concepts

### First Time Contributors

If you're new to contributing, we recommend starting with:

1. Browse [Good First Issues](https://github.com/diogoribeiro7/PSOD/labels/good%20first%20issue)
2. Read our [Development Guide](DEVELOPMENT.md)
3. Join our [Discussions](https://github.com/diogoribeiro7/PSOD/discussions)

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check [existing issues](https://github.com/diogoribeiro7/PSOD/issues) to avoid duplicates.

**When submitting a bug report, include:**

- **Clear title**: Summarize the bug in one line
- **Description**: Detailed explanation of the issue
- **Steps to reproduce**: Step-by-step instructions
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: OS, Python version, PSOD version, dependency versions
- **Code sample**: Minimal reproducible example
- **Error messages**: Full error traceback if applicable

**Example bug report:**

```markdown
**Title**: PSOD fails with categorical columns containing missing values

**Description**: When fitting PSOD with a dataset that has categorical columns
with missing values, the fit() method raises a ValueError.

**Steps to Reproduce**:
1. Create DataFrame with categorical column containing NaN
2. Initialize PSOD with cat_columns parameter
3. Call fit() method

**Code Sample**:
\```python
import pandas as pd
from psod import PSOD

df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', None]})
detector = PSOD(cat_columns=['b'])
detector.fit(df)  # Raises ValueError
\```

**Environment**:
- OS: Ubuntu 22.04
- Python: 3.11.5
- PSOD: 0.1.0
- pandas: 2.0.3
```

### Suggesting Enhancements

**Enhancement suggestions should include:**

- **Clear title**: Summarize the enhancement
- **Motivation**: Why is this enhancement needed?
- **Description**: Detailed explanation of the proposed feature
- **Use cases**: Real-world scenarios where this would be useful
- **Alternatives**: Other solutions you've considered
- **Implementation**: (Optional) Possible implementation approach

### Security Issues

**Do NOT** report security vulnerabilities through public issues. Instead, email security concerns to: [your.email@example.com]

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub first, then clone your fork
git clone https://github.com/YOUR-USERNAME/PSOD.git
cd PSOD

# Add upstream remote
git remote add upstream https://github.com/diogoribeiro7/PSOD.git
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install development dependencies
make dev-setup

# Or manually:
pip install -e ".[dev,test,viz,docs]"
pre-commit install
```

### 3. Verify Installation

```bash
# Run tests
make test

# Check code style
make lint

# Build documentation
make docs
```

## Code Standards

### Style Guide

- **PEP 8**: Follow Python style guide
- **Line length**: Maximum 100 characters
- **Docstrings**: NumPy style
- **Type hints**: Use for public APIs
- **Import order**: stdlib → third-party → local

### Tools

We use automated tools to enforce code quality:

| Tool | Purpose | Config |
|------|---------|--------|
| Black | Code formatting | `pyproject.toml` |
| isort | Import sorting | `pyproject.toml` |
| flake8 | Linting | `pyproject.toml` |
| pylint | Static analysis | `pyproject.toml` |
| mypy | Type checking | `pyproject.toml` |
| bandit | Security | `pyproject.toml` |

### Running Quality Checks

```bash
# Auto-format code
make format

# Check formatting
make format-check

# Run linters
make lint

# Type checking
make type-check

# Security scan
make security

# Run all checks
make dev-check
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`:

```bash
# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

## Testing Guidelines

### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Fast tests only
make test-fast

# Specific test file
pytest tests/test_core.py -v

# Specific test function
pytest tests/test_core.py::test_psod_init -v
```

### Writing Tests

**Guidelines:**

- Write tests for all new features
- Update tests for bug fixes
- Aim for 80%+ code coverage
- Use descriptive test names
- Include docstrings in test functions
- Use fixtures for common setup

**Example:**

```python
import pytest
import pandas as pd
import numpy as np
from psod import PSOD


def test_psod_fit_predict_returns_correct_shape():
    """Test that fit_predict returns array of correct shape."""
    # Arrange
    n_samples = 100
    n_features = 5
    df = pd.DataFrame(np.random.randn(n_samples, n_features))
    detector = PSOD(random_seed=42)

    # Act
    scores = detector.fit_predict(df, return_class=False)

    # Assert
    assert len(scores) == n_samples
    assert isinstance(scores, np.ndarray)


@pytest.mark.slow
def test_psod_with_large_dataset():
    """Test PSOD with large dataset (marked as slow)."""
    df = pd.DataFrame(np.random.randn(10000, 50))
    detector = PSOD()
    scores = detector.fit_predict(df, return_class=False)
    assert len(scores) == 10000


@pytest.fixture
def sample_outlier_data():
    """Fixture providing sample data with outliers."""
    np.random.seed(42)
    normal = np.random.randn(95, 5)
    outliers = np.random.uniform(-10, 10, (5, 5))
    data = np.vstack([normal, outliers])
    return pd.DataFrame(data)


def test_psod_detects_outliers(sample_outlier_data):
    """Test that PSOD can detect obvious outliers."""
    detector = PSOD(stdevs_to_outlier=2.0, random_seed=42)
    labels = detector.fit_predict(sample_outlier_data, return_class=True)
    assert sum(labels) > 0  # At least some outliers detected
```

## Documentation

### Building Documentation

```bash
# Build HTML documentation
make docs

# Serve locally
make docs-serve

# Check for broken links
make docs-linkcheck
```

### Writing Docstrings

Use NumPy style docstrings:

```python
def detect_outliers(data, threshold=2.0, columns=None):
    """Detect outliers in dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing features
    threshold : float, default=2.0
        Number of standard deviations for outlier threshold
    columns : List[str], optional
        Specific columns to analyze. If None, use all columns

    Returns
    -------
    outliers : np.ndarray of shape (n_samples,)
        Boolean array indicating outliers (True) and inliers (False)

    Raises
    ------
    ValueError
        If data contains non-numeric columns without columns parameter

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2, 3, 100], 'b': [4, 5, 6, 7]})
    >>> outliers = detect_outliers(df, threshold=2.0)
    >>> print(outliers)
    [False False False True]

    Notes
    -----
    This function uses the Z-score method for outlier detection [1]_.

    References
    ----------
    .. [1] Author, "Title", Journal, Year.

    See Also
    --------
    PSOD : Main outlier detection class
    """
    pass
```

## Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code changes that neither fix bugs nor add features
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `build`: Changes to build system or dependencies
- `ci`: CI/CD changes
- `chore`: Other changes (maintenance, etc.)

### Examples

```bash
# Feature
git commit -m "feat(core): add support for custom transformers"

# Bug fix
git commit -m "fix(utils): resolve memory leak in save_model function"

# Documentation
git commit -m "docs(api): update PSOD class docstring with examples"

# Breaking change
git commit -m "feat(core)!: remove deprecated parameter

BREAKING CHANGE: Removed 'old_param' parameter, use 'new_param' instead"
```

## Pull Request Process

### 1. Create a Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name

# Or bugfix branch
git checkout -b fix/issue-description
```

### 2. Make Changes

- Write code following our style guide
- Add/update tests
- Update documentation
- Run quality checks: `make dev-check`

### 3. Commit Changes

```bash
# Stage changes
git add .

# Commit with conventional commits format
git commit -m "feat(core): add your feature"

# Push to your fork
git push origin feature/your-feature-name
```

### 4. Open Pull Request

1. Go to your fork on GitHub
2. Click "Compare & pull request"
3. Fill in PR template:
   - Clear title
   - Description of changes
   - Link to related issues
   - Screenshots (if applicable)
4. Request reviewers
5. Wait for CI checks to pass

### 5. Address Review Comments

```bash
# Make requested changes
# ... edit files ...

# Commit and push
git add .
git commit -m "refactor: address review comments"
git push
```

### PR Checklist

Before submitting, ensure:

- [ ] Code follows style guide
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
- [ ] Commits follow conventional commits
- [ ] CI checks passing
- [ ] No merge conflicts

## Release Process

*For maintainers only*

### Version Bumping

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Steps

1. **Update version** in `src/psod/__init__.py`

2. **Update CHANGELOG.md**

3. **Create release PR**

4. **Merge to main**

5. **Create and push tag**:
   ```bash
   git tag -a v1.2.0 -m "Release v1.2.0"
   git push origin v1.2.0
   ```

6. **GitHub Actions** handles the rest:
   - Runs tests
   - Builds package
   - Publishes to PyPI
   - Creates GitHub release

## Questions?

- **General questions**: [GitHub Discussions](https://github.com/diogoribeiro7/PSOD/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/diogoribeiro7/PSOD/issues)
- **Security issues**: Email [your.email@example.com]

## Recognition

Contributors will be:
- Listed in AUTHORS.md
- Mentioned in release notes
- Thanked in project README

Thank you for contributing to PSOD! 🎉
