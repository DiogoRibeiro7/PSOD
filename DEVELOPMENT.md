# PSOD Development Guide

This guide describes the repository during the active refactor. The README and PROVENANCE.md define the project-level scientific and attribution constraints; this file focuses on development mechanics.

## Supported Python

The current supported floor is Python 3.10. CI tests Python 3.10, 3.11, and 3.12.

## Setup

```bash
git clone https://github.com/DiogoRibeiro7/PSOD.git
cd PSOD
python -m venv .venv
python -m pip install -e ".[dev,test]"
pre-commit install
```

## Core checks

```bash
black --check src tests examples benchmarks
isort --check-only src tests examples benchmarks
pytest tests
```

The current CI also runs package builds, wheel installation smoke tests, integration tests, examples, coverage generation, and security checks. See `.github/CICD_SETUP.md` for the current gating/advisory distinction.

## Repository structure

```text
src/psod/       installed package
tests/          test suite
examples/       runnable examples
benchmarks/     benchmark infrastructure
docs/           documentation source
.github/        automation and repository policy
docker/         container/development assets
```

`pyproject.toml` is the authoritative packaging configuration. There is no supported legacy `setup.py` or root `requirements.txt` packaging path.

## Refactor stages

Changes should remain separated into reviewable stages:

1. repository integrity, provenance, licensing, and CI;
2. packaging and dependency cleanup;
3. characterization tests and public API reduction;
4. core algorithm rewrite with cross-fitted residuals;
5. robust score calibration and thresholding;
6. property-based and invariant tests;
7. reproducible benchmark suite;
8. documentation and release preparation.

Do not mix intentional scoring changes into infrastructure or packaging PRs.

## Benchmark policy

Benchmark claims must be backed by committed, reproducible artifacts and a protocol that records datasets, configurations, repeated runs, and uncertainty. Illustrative tables are not empirical results.

## Release policy

The repository can build and test source/wheel artifacts, but PyPI publication is intentionally disabled during the refactor. The `psod` distribution name is already used by the earlier PSOD project, so a new distribution identity must be selected before publication is enabled.

## Provenance

The method and substantial API lineage predate this repository. Preserve the attribution and GPL-3.0-only policy documented in PROVENANCE.md.
