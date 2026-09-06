# Contributing to PSOD

PSOD is currently undergoing a staged refactor. Contributions are welcome, but changes should preserve the refactor boundaries described in the README and PROVENANCE.md.

## Prerequisites

- Python 3.10 or newer
- Git
- A virtual environment

## Development setup

```bash
git clone https://github.com/DiogoRibeiro7/PSOD.git
cd PSOD
python -m venv .venv
python -m pip install -e ".[dev,test]"
pre-commit install
```

## Before opening a pull request

Run the relevant checks locally:

```bash
black --check src tests examples benchmarks
isort --check-only src tests examples benchmarks
pytest tests
```

The active CI matrix currently covers Python 3.10, 3.11, and 3.12 on Ubuntu. Some legacy linting and security steps remain advisory while the tooling stack is simplified; the GitHub workflow is the source of truth for which checks are gating.

## Scope discipline

Keep pull requests focused. In particular:

- provenance, licensing, packaging, API cleanup, scoring changes, and benchmarking should remain separate reviewable stages;
- do not restore claims that PSOD is a novel method;
- do not publish benchmark results without committed, reproducible evidence;
- do not enable PyPI publication under the `psod` distribution name. That name is already used by the earlier PSOD project.

## Pull requests

Use a short-lived branch from the current target branch and describe:

1. what changed;
2. what did not change;
3. how the change was validated;
4. whether it changes public behavior or only repository infrastructure.

Address review findings with additional commits rather than silently weakening CI or deleting checks.

## Release policy during the refactor

The release workflow currently builds and tests distribution artifacts only. PyPI publication is intentionally disabled until a new distribution identity is chosen and reviewed.

## Security

Do not report security vulnerabilities in a public issue. Contact the maintainer privately instead.

## Provenance

This repository is derived from the earlier PSOD implementation by Thomas Meißner. See PROVENANCE.md for the attribution and licensing policy that contributions must preserve.
