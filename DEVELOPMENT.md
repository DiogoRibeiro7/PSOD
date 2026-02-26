# PSOD Development Guide

Complete guide for setting up and contributing to the PSOD project.

## 📋 Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Performance](#performance)
- [Release Process](#release-process)

## 🚀 Development Environment Setup

### Prerequisites

- Python 3.8+
- Git
- Make (optional but recommended)
- Docker (optional)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/diogoribeiro7/PSOD.git
   cd PSOD
   ```

2. **Set up development environment**
   ```bash
   # One-command setup (recommended)
   make dev-setup

   # Or manually:
   pip install -e ".[dev,test,viz,docs]"
   pre-commit install
   ```

3. **Verify installation**
   ```bash
   make test
   python -c "from psod import PSOD; print('Success!')"
   ```

### Development Tools

#### Using Make

The project includes a comprehensive Makefile with 50+ commands:

```bash
# See all available commands
make help

# Common commands
make install-dev      # Install with dev dependencies
make test             # Run tests
make test-cov         # Run tests with coverage
make format           # Format code
make lint             # Run linters
make docs             # Build documentation
make dev-check        # Run all checks before commit
```

#### Using Docker

```bash
# Start Jupyter development environment
make docker-dev

# Run tests in Docker
make docker-test

# Build documentation in Docker
docker-compose up docs
```

## 📁 Project Structure

```
PSOD/
├── src/psod/              # Source code
│   ├── __init__.py        # Package initialization
│   ├── core.py            # PSOD class implementation
│   ├── utils.py           # Utility functions
│   └── visualization.py   # Visualization functions
├── tests/                 # Test suite
│   ├── conftest.py        # Pytest configuration
│   ├── test_psod.py       # Core tests
│   ├── test_utils.py      # Utils tests
│   ├── test_visualization.py  # Viz tests
│   └── test_integration.py    # Integration tests
├── examples/              # Example scripts
│   ├── basic_usage.py
│   ├── advanced_usage.py
│   ├── time_series_example.py
│   ├── comparison_example.py
│   ├── cli_examples.py
│   └── notebooks/         # Jupyter notebooks
├── benchmarks/            # Performance benchmarks
│   ├── run_benchmarks.py
│   ├── datasets.py
│   ├── methods.py
│   └── metrics.py
├── docs/                  # Documentation
│   ├── conf.py
│   ├── index.rst
│   └── api/
├── .github/               # GitHub Actions
│   └── workflows/
├── docker/                # Docker configuration
│   ├── Dockerfile.dev
│   ├── docker-compose.yml
│   └── README.md
├── Makefile               # Development commands
├── pyproject.toml         # Modern Python config
├── setup.py               # Legacy setup
├── requirements.txt       # Dependencies
└── README.md              # Project readme
```

## 🔄 Development Workflow

### 1. Create a Branch

```bash
# Update main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name

# Or bugfix branch
git checkout -b fix/issue-description
```

### 2. Make Changes

Follow these guidelines:

```bash
# Format code automatically
make format

# Check code quality
make lint

# Run tests
make test

# Or run all checks at once
make dev-check
```

### 3. Commit Changes

We use conventional commits:

```bash
# Format: <type>(<scope>): <subject>
git commit -m "feat(core): add new parameter to PSOD"
git commit -m "fix(utils): resolve memory leak in save_model"
git commit -m "docs(api): update PSOD class docstring"
git commit -m "test(core): add tests for edge cases"
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

### 4. Push and Create PR

```bash
# Push branch
git push origin feature/your-feature-name

# Create PR on GitHub
# - Fill in PR template
# - Link related issues
# - Request reviewers
```

### 5. Address Review Comments

```bash
# Make changes
# ... edit files ...

# Format and test
make dev-check

# Commit and push
git add .
git commit -m "refactor: address review comments"
git push
```

## 📏 Code Standards

### Style Guide

- **Line length**: 100 characters
- **String quotes**: Double quotes preferred
- **Imports**: Organized by stdlib, third-party, local
- **Docstrings**: NumPy style
- **Type hints**: Use where beneficial

### Code Formatting

We use Black and isort:

```bash
# Auto-format code
make format

# Check formatting without changes
make format-check
```

**Black configuration** (pyproject.toml):
```toml
[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310', 'py311']
```

**isort configuration** (pyproject.toml):
```toml
[tool.isort]
profile = "black"
line_length = 100
```

### Linting

We use multiple linters:

```bash
# Run all linters
make lint

# Individual linters
flake8 src/ tests/
pylint src/psod/
mypy src/psod/
```

**Configuration files**:
- flake8: `.flake8` or `pyproject.toml`
- pylint: `pyproject.toml`
- mypy: `pyproject.toml`

### Type Hints

Use type hints for public APIs:

```python
from typing import Optional, List, Union
import pandas as pd
import numpy as np

def detect_outliers(
    data: pd.DataFrame,
    threshold: float = 2.0,
    columns: Optional[List[str]] = None
) -> np.ndarray:
    """Detect outliers in data.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    threshold : float, default=2.0
        Detection threshold
    columns : List[str], optional
        Columns to use

    Returns
    -------
    np.ndarray
        Boolean array of outliers
    """
    ...
```

## 🧪 Testing

### Running Tests

```bash
# All tests
make test

# Fast tests only (skip slow tests)
make test-fast

# Slow tests only
make test-slow

# With coverage
make test-cov

# Specific test file
pytest tests/test_core.py -v

# Specific test function
pytest tests/test_core.py::test_psod_init -v

# In parallel
make test-parallel

# With debugging
make test-debug
```

### Writing Tests

Use pytest:

```python
import pytest
import numpy as np
import pandas as pd
from psod import PSOD

def test_psod_basic():
    """Test basic PSOD functionality."""
    # Arrange
    df = pd.DataFrame(np.random.randn(100, 5))
    detector = PSOD(random_seed=42)

    # Act
    scores = detector.fit_predict(df, return_class=False)

    # Assert
    assert len(scores) == 100
    assert scores.min() >= 0
    assert scores.max() <= 1

@pytest.mark.slow
def test_psod_large_dataset():
    """Test with large dataset (slow)."""
    df = pd.DataFrame(np.random.randn(10000, 50))
    detector = PSOD()
    scores = detector.fit_predict(df, return_class=False)
    assert len(scores) == 10000

@pytest.fixture
def sample_data():
    """Fixture for test data."""
    return pd.DataFrame(np.random.randn(100, 5))

def test_with_fixture(sample_data):
    """Test using fixture."""
    detector = PSOD()
    scores = detector.fit_predict(sample_data, return_class=False)
    assert scores is not None
```

### Test Organization

- `tests/test_core.py` - Core PSOD functionality
- `tests/test_utils.py` - Utility functions
- `tests/test_visualization.py` - Visualization
- `tests/test_integration.py` - Integration tests
- `tests/conftest.py` - Shared fixtures

### Coverage Requirements

- Minimum coverage: 80%
- Check coverage: `make test-cov`
- View HTML report: `open htmlcov/index.html`

## 📚 Documentation

### Building Documentation

```bash
# Build HTML documentation
make docs

# Serve locally
make docs-serve
# Visit http://localhost:8000

# Check for broken links
make docs-linkcheck

# Check docstring coverage
make docs-coverage
```

### Writing Documentation

#### Docstrings

Use NumPy style:

```python
class PSOD:
    """Pseudo-Supervised Outlier Detection.

    PSOD detects outliers by treating each feature as a target
    and using prediction errors as outlier scores.

    Parameters
    ----------
    n_jobs : int, default=-1
        Number of parallel jobs
    cat_columns : List[str], optional
        Categorical column names
    stdevs_to_outlier : float, default=1.96
        Threshold in standard deviations

    Attributes
    ----------
    scores_ : np.ndarray
        Outlier scores after fitting
    outlier_classes_ : np.ndarray
        Binary outlier labels

    Examples
    --------
    >>> import pandas as pd
    >>> from psod import PSOD
    >>> df = pd.DataFrame({'a': [1, 2, 3, 100]})
    >>> detector = PSOD()
    >>> scores = detector.fit_predict(df, return_class=False)
    >>> print(scores)
    [0.1, 0.1, 0.1, 0.9]

    Notes
    -----
    This implementation uses a bagging approach with
    multiple regression models.

    References
    ----------
    .. [1] Author, "Title", Journal, Year.
    """
```

#### Markdown Files

- Use relative links
- Include code examples
- Add screenshots where helpful
- Keep lines under 100 chars

### Documentation Structure

```
docs/
├── index.rst              # Main page
├── installation.rst       # Installation guide
├── quickstart.rst         # Quick start guide
├── user_guide.rst         # User guide
├── api/                   # API reference
│   ├── core.rst
│   ├── utils.rst
│   └── visualization.rst
├── examples.rst           # Examples
├── contributing.rst       # Contributing guide
└── changelog.rst          # Changelog
```

## ⚡ Performance

### Benchmarking

```bash
# Run benchmarks
make benchmark

# Compare with baseline
make benchmark-compare

# Profile code
make profile

# Memory profiling
make profile-memory
```

### Performance Tips

1. **Use NumPy arrays** when possible
2. **Vectorize operations** instead of loops
3. **Cache expensive computations**
4. **Use n_jobs=-1** for parallelization
5. **Profile before optimizing**

### Profiling Example

```bash
# CPU profiling
python -m cProfile -o profile.stats examples/basic_usage.py
python -m pstats profile.stats
>>> sort cumulative
>>> stats 20

# Memory profiling
python -m memory_profiler examples/basic_usage.py

# Line profiling
kernprof -l -v examples/basic_usage.py
```

## 🚢 Release Process

### Version Bumping

We use semantic versioning (MAJOR.MINOR.PATCH):

- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Creating a Release

1. **Update version** in `src/psod/__init__.py`

2. **Update CHANGELOG.md**
   ```markdown
   ## [1.2.0] - 2025-01-14

   ### Added
   - New feature X
   - Support for Y

   ### Changed
   - Improved Z

   ### Fixed
   - Bug in W
   ```

3. **Run release checks**
   ```bash
   make release-check
   ```

4. **Create git tag**
   ```bash
   git tag -a v1.2.0 -m "Release version 1.2.0"
   git push origin v1.2.0
   ```

5. **GitHub Actions** will automatically:
   - Run tests
   - Build package
   - Publish to PyPI
   - Create GitHub release

### Manual Release

```bash
# Test on TestPyPI first
make release-test

# Release to PyPI
make release
```

## 🔧 Troubleshooting

### Common Issues

#### Import Errors

```bash
# Reinstall in editable mode
pip install -e ".[dev]"

# Check PYTHONPATH
echo $PYTHONPATH
export PYTHONPATH=/path/to/PSOD/src:$PYTHONPATH
```

#### Pre-commit Hook Failures

```bash
# Update hooks
pre-commit autoupdate

# Run manually
pre-commit run --all-files

# Skip hooks (emergency only)
git commit --no-verify
```

#### Test Failures

```bash
# Run with verbose output
pytest -vv

# Run with print statements
pytest -s

# Run single test with debugging
pytest tests/test_core.py::test_name --pdb
```

## 📞 Getting Help

- **Issues**: [GitHub Issues](https://github.com/diogoribeiro7/PSOD/issues)
- **Discussions**: [GitHub Discussions](https://github.com/diogoribeiro7/PSOD/discussions)
- **Email**: dfr@esmad.ipp.pt

## 📄 Additional Resources

- [Contributing Guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [CI/CD Setup](.github/CICD_SETUP.md)
- [Docker Guide](docker/README.md)
- [Examples](examples/README.md)

---

**Happy coding! 🎉**

