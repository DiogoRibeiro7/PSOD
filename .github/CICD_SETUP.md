# CI/CD Pipeline Setup Guide

This document provides comprehensive information about the PSOD project's CI/CD pipeline implementation using GitHub Actions.

## 📋 Table of Contents

- [Overview](#overview)
- [Workflows](#workflows)
- [Setup Instructions](#setup-instructions)
- [Configuration](#configuration)
- [Badges](#badges)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The PSOD project uses GitHub Actions for a comprehensive CI/CD pipeline that includes:

- ✅ **Continuous Integration**: Automated testing on multiple Python versions and operating systems
- 📝 **Code Quality**: Linting, formatting, and type checking
- 🔒 **Security**: Multiple security scanners and dependency checks
- 📚 **Documentation**: Automated documentation builds and deployment
- ⚡ **Performance**: Benchmark tracking and regression detection
- 🚀 **Deployment**: Automated releases to PyPI

## 🔄 Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Triggers:** Push, Pull Request to `main` and `develop` branches

**Jobs:**
- **Code Quality Checks**
  - Black formatting
  - isort import sorting
  - flake8 linting
  - pylint analysis
  - mypy type checking
  - bandit security scan

- **Dependency Security**
  - safety check
  - pip-audit

- **Test Matrix**
  - Python versions: 3.8, 3.9, 3.10, 3.11, 3.12
  - Operating systems: Ubuntu, Windows, macOS
  - Coverage reporting with Codecov

- **Integration Tests**
  - Cross-module integration testing

- **Examples Testing**
  - Validate all example scripts
  - Test CLI functionality

- **Package Build**
  - Build source and wheel distributions
  - Validate with twine

- **Installation Testing**
  - Test package installation on all platforms
  - Verify basic functionality

**Artifacts:**
- Test results (JUnit XML)
- Coverage reports (HTML, XML)
- Build distributions (wheel, tar.gz)
- Security scan reports

### 2. Documentation Workflow (`.github/workflows/docs.yml`)

**Triggers:** Push/PR to documentation files, scheduled

**Jobs:**
- **Build Documentation**
  - Sphinx HTML build
  - Link checking
  - Generate build summary

- **Test Notebooks**
  - Execute all Jupyter notebooks
  - Verify output consistency

- **Validate Examples**
  - Syntax validation
  - Code style checking

- **Deploy to GitHub Pages**
  - Automatic deployment on main branch
  - Hosted at: `https://<username>.github.io/PSOD/`

- **Documentation Coverage**
  - Check docstring coverage with interrogate
  - Require 80% minimum coverage

**Artifacts:**
- Built HTML documentation
- Documentation coverage badge

### 3. Release Workflow (`.github/workflows/release.yml`)

**Triggers:** Git tags `v*.*.*`, manual workflow dispatch

**Jobs:**
- **Validate Release**
  - Semantic version validation
  - PyPI version conflict check

- **Run Full Test Suite**
  - Calls CI workflow
  - Ensures all tests pass

- **Build Package**
  - Create source and wheel distributions
  - Verify with twine

- **Test Package**
  - Install and test on multiple platforms
  - Functionality verification

- **Publish to TestPyPI**
  - Test publication process
  - Verify installation from TestPyPI

- **Publish to PyPI**
  - Production release
  - Requires manual approval (environment protection)

- **Create GitHub Release**
  - Generate release notes from commits
  - Attach distribution files
  - Calculate checksums

- **Post-Release Tasks**
  - Create post-release checklist issue
  - Generate summary report

**Environment Variables Required:**
- `PYPI_API_TOKEN`: PyPI API token (stored in GitHub Secrets)
- `TESTPYPI_API_TOKEN`: TestPyPI API token

### 4. Security Workflow (`.github/workflows/security.yml`)

**Triggers:** Push, PR, weekly schedule (Mondays at 00:00 UTC)

**Jobs:**
- **Dependency Review**
  - Checks for vulnerable dependencies in PRs
  - License compliance verification

- **CodeQL Analysis**
  - Static code analysis
  - Security and quality queries

- **Bandit Scan**
  - Python security issue detection
  - SARIF output for GitHub Security tab

- **Safety Check**
  - Known vulnerability database check

- **Pip Audit**
  - Dependency vulnerability scanning
  - SBOM generation

- **Semgrep Scan**
  - Pattern-based security scanning
  - OWASP Top 10 checks

- **Trivy Scan**
  - Comprehensive vulnerability scanner
  - Critical, High, and Medium severity alerts

- **Secrets Scan**
  - Gitleaks for secret detection
  - TruffleHog for credential scanning

- **License Check**
  - Verify compatible licenses
  - Flag GPL-3.0, AGPL-3.0, SSPL

- **OpenSSF Scorecard**
  - Best practices scoring
  - Supply chain security assessment

**Artifacts:**
- Security scan reports (JSON, SARIF)
- License compliance report

### 5. Performance Workflow (`.github/workflows/performance.yml`)

**Triggers:** Push, PR, weekly schedule (Sundays at 00:00 UTC)

**Jobs:**
- **Benchmark Execution**
  - Run comprehensive benchmarks
  - Track execution time and memory usage

- **Performance Comparison**
  - Compare PR performance vs main branch
  - Alert on regressions > 10%

- **Memory Profiling**
  - memory_profiler integration
  - Track memory usage patterns

- **Scalability Testing**
  - Test with datasets: 100, 1K, 5K, 10K samples
  - Calculate complexity estimates

- **Collect and Visualize**
  - Generate performance dashboards
  - Create scalability plots

- **Performance Report**
  - Update gh-pages with latest results
  - Historical performance tracking

**Artifacts:**
- Benchmark results (JSON)
- Memory profiles
- Performance dashboards (PNG)
- Scalability reports

## ⚙️ Setup Instructions

### 1. Repository Setup

1. **Enable GitHub Actions**
   ```
   Settings → Actions → General → Allow all actions
   ```

2. **Set Branch Protection Rules**
   ```
   Settings → Branches → Add rule
   - Branch name pattern: main
   - Require status checks: ✓
   - Require branches to be up to date: ✓
   - Required status checks:
     - Code Quality Checks
     - Test Python 3.11 on ubuntu-latest
     - Build Package
   ```

3. **Configure GitHub Pages**
   ```
   Settings → Pages
   - Source: Deploy from a branch
   - Branch: gh-pages
   - Folder: / (root)
   ```

### 2. Secrets Configuration

Add the following secrets to your repository:

```
Settings → Secrets and variables → Actions → New repository secret
```

**Required Secrets:**

- `PYPI_API_TOKEN`
  - Go to https://pypi.org/manage/account/token/
  - Create new API token
  - Scope: "Entire account" or specific to PSOD project
  - Copy token and add to GitHub Secrets

- `TESTPYPI_API_TOKEN`
  - Go to https://test.pypi.org/manage/account/token/
  - Create new API token
  - Add to GitHub Secrets

- `CODECOV_TOKEN` (optional, for private repos)
  - Sign up at https://codecov.io
  - Add repository
  - Copy token from repository settings

### 3. Environment Protection

Create protected environments for deployment:

```
Settings → Environments → New environment
```

**Environment: `pypi`**
- Required reviewers: Add team members
- Wait timer: 0 minutes
- Deployment branches: Selected branches → `main`

**Environment: `testpypi`**
- No protection rules (optional)

### 4. Enable Pre-commit Hooks

Developers should install pre-commit hooks locally:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually (optional)
pre-commit run --all-files
```

### 5. Configure Codecov (Optional)

For coverage reporting:

1. Sign up at https://codecov.io
2. Add GitHub repository
3. Configure coverage settings:
   ```yaml
   # In codecov.yml (create at repository root)
   coverage:
     status:
       project:
         default:
           target: 80%
           threshold: 1%
   ```

## 🎨 Configuration

### Customizing Workflows

#### Modify Test Matrix

Edit `.github/workflows/ci.yml`:

```yaml
matrix:
  os: [ubuntu-latest, windows-latest, macos-latest]
  python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
```

#### Adjust Code Quality Thresholds

Edit `.github/workflows/ci.yml`:

```yaml
- name: Lint with pylint
  run: |
    pylint src/psod/ --fail-under=8.0  # Change threshold here
```

#### Configure Security Scanning

Edit `.github/workflows/security.yml`:

```yaml
- name: Run Bandit scan
  run: |
    bandit -r src/ -ll  # Change severity level: -ll, -l, or no flag
```

### Pre-commit Configuration

Edit `.pre-commit-config.yaml` to add/remove hooks:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        args: [--line-length=100]  # Customize line length
```

## 📊 Badges

Add status badges to your README:

```markdown
![CI](https://github.com/<username>/PSOD/workflows/CI/badge.svg)
![Documentation](https://github.com/<username>/PSOD/workflows/Documentation/badge.svg)
![Security](https://github.com/<username>/PSOD/workflows/Security%20Scanning/badge.svg)
[![codecov](https://codecov.io/gh/<username>/PSOD/branch/main/graph/badge.svg)](https://codecov.io/gh/<username>/PSOD)
[![PyPI version](https://badge.fury.io/py/psod.svg)](https://badge.fury.io/py/psod)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Pre-commit Hook Failures

**Problem:** Commits fail due to pre-commit hooks

**Solution:**
```bash
# Fix formatting issues automatically
black src/ tests/ examples/
isort src/ tests/ examples/

# Or skip hooks for emergency commits (not recommended)
git commit --no-verify -m "message"
```

#### 2. CI Test Failures

**Problem:** Tests pass locally but fail in CI

**Solution:**
- Check Python version compatibility
- Review platform-specific code (Windows vs Linux)
- Verify all dependencies are in requirements.txt
- Check for hardcoded paths

#### 3. Documentation Build Failures

**Problem:** Sphinx build fails

**Solution:**
```bash
# Test locally
cd docs
make clean
make html

# Check for:
# - Syntax errors in docstrings
# - Missing dependencies in docs/requirements.txt
# - Broken internal links
```

#### 4. PyPI Publication Failures

**Problem:** Release workflow fails at PyPI publication

**Solution:**
- Verify API tokens are correct and not expired
- Check version doesn't already exist on PyPI
- Ensure package name is available
- Review distribution files with `twine check dist/*`

#### 5. Security Scan False Positives

**Problem:** Security scanners report false positives

**Solution:**
```yaml
# Add exclusions in workflow
- name: Run Bandit scan
  run: |
    bandit -r src/ -ll --skip B101,B601  # Skip specific checks
```

Or create `bandit.yml`:
```yaml
skips:
  - B101  # assert_used
  - B601  # paramiko_calls
```

### Getting Help

- **GitHub Discussions**: https://github.com/<username>/PSOD/discussions
- **Issues**: https://github.com/<username>/PSOD/issues
- **Documentation**: https://psod.readthedocs.io

## 📝 Best Practices

### For Contributors

1. **Always run pre-commit hooks** before pushing
2. **Write tests** for new features
3. **Update documentation** for API changes
4. **Keep PRs focused** on single features/fixes
5. **Wait for CI** before requesting review

### For Maintainers

1. **Review security scan results** weekly
2. **Monitor performance benchmarks** for regressions
3. **Keep dependencies updated** regularly
4. **Tag releases** following semantic versioning
5. **Review and merge** dependabot PRs promptly

## 🔄 Workflow Diagram

```
┌─────────────┐
│   Push/PR   │
└──────┬──────┘
       │
       ├─────────────┬─────────────┬─────────────┬─────────────┐
       │             │             │             │             │
       ▼             ▼             ▼             ▼             ▼
┌────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│    CI      │ │   Docs   │ │ Security │ │   Perf   │ │  Release │
│  Testing   │ │  Build   │ │   Scan   │ │  Tests   │ │  (tags)  │
└──────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
       │            │            │            │            │
       └────────────┴────────────┴────────────┴────────────┘
                              │
                              ▼
                        ┌───────────┐
                        │  All Pass │
                        └─────┬─────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
              ┌──────────┐        ┌──────────┐
              │  Merge   │        │  Deploy  │
              │   to     │        │   Docs   │
              │  Main    │        │   PyPI   │
              └──────────┘        └──────────┘
```

## 📅 Maintenance Schedule

- **Daily**: Monitor CI failures
- **Weekly**: Review security scan results
- **Monthly**: Update dependencies
- **Quarterly**: Review and update workflows
- **Release**: Follow semantic versioning

---

**Last Updated:** 2025-01-14
**Version:** 1.0.0
