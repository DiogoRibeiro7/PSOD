# CI/CD Pipeline

This document describes the workflows that are currently active in the PSOD repository during the refactor. It deliberately avoids promising release, benchmark, or deployment behavior that is not presently enforced.

## CI workflow

`.github/workflows/ci.yml` runs on pushes and pull requests targeting `main` or `develop`.

The current **gating baseline** covers:

- Python 3.10, 3.11, and 3.12 tests on Ubuntu;
- Black formatting;
- isort import ordering;
- flake8 syntax and undefined-name checks;
- unit and integration tests;
- package build and `twine check`;
- wheel installation smoke tests.

The workflow also runs **advisory diagnostics** that currently use `continue-on-error` and therefore do not by themselves fail CI:

- pylint;
- mypy;
- the CI-local Bandit diagnostic;
- the CI-local dependency-audit diagnostic;
- some example-execution checks.

These advisory steps are technical debt to be made strict or removed during the tooling cleanup. A green CI run must not be described as proving that an advisory diagnostic passed when GitHub recorded it as non-gating.

The supported Python floor is Python 3.10 during this refactor.

## Security workflow

`.github/workflows/security.yml` runs on pushes, pull requests, a weekly schedule, and manual dispatch.

It currently includes:

- dependency review for pull requests;
- CodeQL;
- Bandit;
- `pip-audit` with JSON and CycloneDX JSON reports;
- Semgrep;
- Trivy with an enforced vulnerability scan plus SARIF where repository permissions allow upload;
- Gitleaks and TruffleHog secret scanning;
- dependency license reporting;
- an aggregate security gate that fails when a required scan fails.

Fork pull requests still run the Trivy vulnerability policy, but SARIF upload is skipped when the token cannot write security events.

Because this repository is treated as GPL-3.0-only, GPL-3.0 dependencies are not rejected merely for being GPL-3.0. The explicit denied dependency-license policy is currently limited to AGPL-3.0 and SSPL-family licenses.

## Documentation, performance, and release workflows

The repository also contains `docs.yml`, `performance.yml`, and `release.yml`. These are being audited separately during the repository cleanup.

The release workflow currently **builds and tests artifacts only**. PyPI publication is intentionally disabled because the `psod` distribution name is already used by the earlier PSOD project. A new distribution identity must be selected and reviewed before publication can be enabled.

## Local verification

A representative local check is:

```bash
python -m pip install -e ".[dev,test]"
black --check src tests examples benchmarks
isort --check-only src tests examples benchmarks
pytest tests
```

For security-oriented checks:

```bash
bandit -r src/psod -c .bandit -ll
pip-audit --desc
```

## Refactor policy

CI is treated as a scientific and engineering gate, not decoration. Repository documentation must distinguish checks that can fail a workflow from diagnostics that are currently advisory. Security artifacts or required jobs must not be silently dropped merely to obtain a green status.
