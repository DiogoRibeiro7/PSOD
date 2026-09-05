# CI/CD Pipeline

This document describes the workflows that are currently active in the PSOD repository during the refactor. It deliberately avoids promising release, benchmark, or deployment behavior that is not presently enforced.

## CI workflow

`.github/workflows/ci.yml` runs on pushes and pull requests targeting `main` or `develop`.

The current required baseline covers:

- Python 3.10, 3.11, and 3.12 on Ubuntu;
- Black formatting;
- isort import ordering;
- flake8 syntax and undefined-name checks;
- pylint, mypy, and Bandit diagnostics;
- unit and integration tests;
- example execution;
- package build and `twine check`;
- wheel installation smoke tests;
- coverage artifact generation;
- dependency auditing.

The supported Python floor is therefore Python 3.10 during this refactor. Older classifiers or setup instructions should not be treated as supported unless they are restored to the CI matrix in a later change.

## Security workflow

`.github/workflows/security.yml` runs on pushes, pull requests, a weekly schedule, and manual dispatch.

It currently includes:

- dependency review for pull requests;
- CodeQL;
- Bandit;
- `pip-audit` with JSON and CycloneDX JSON reports;
- Semgrep;
- Trivy, including SARIF upload and a human-readable report;
- Gitleaks and TruffleHog secret scanning;
- dependency license reporting;
- an aggregate security gate that fails when a required scan fails.

Because this repository is now treated as GPL-3.0-only, GPL-3.0 dependencies are not rejected merely for being GPL-3.0. The explicit denied dependency-license policy is currently limited to AGPL-3.0 and SSPL-family licenses.

Security artifacts include machine-readable audit output, Trivy SARIF, the Trivy text report, and dependency-license reports where those jobs execute.

## Documentation, performance, and release workflows

The repository also contains `docs.yml`, `performance.yml`, and `release.yml`. These files are being audited separately during the repository cleanup. Their presence does not imply that documentation deployment, benchmark publication, or PyPI publication is currently release-ready.

In particular, the `psod` distribution name is already occupied by the earlier PSOD project, so no release workflow should be interpreted as authorization to publish this refactor under that name.

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

CI is treated as a scientific and engineering gate, not decoration. A green workflow means the checks that actually ran passed; it must not be achieved by silently dropping artifacts, omitting failed jobs from aggregate status, or weakening the documented contract without an explicit repository change.
