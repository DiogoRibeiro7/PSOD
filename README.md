# PSOD

Pseudo-supervised outlier detection for tabular data using feature-wise predictive residuals.

> **Project status:** active refactor. The repository is being cleaned and revalidated before any new release. The current refactor does not claim the PSOD method as novel.

## Method

For each numerical feature \(X_j\), PSOD predicts that feature from a subset of the remaining columns and uses the prediction residual as anomaly evidence. Feature-wise residuals are then aggregated into a row-level anomaly score.

Conceptually,

```text
X_j <- f_j(X_-j)
residual_ij = |x_ij - f_j(x_i,-j)|
row score_i = aggregate_j(residual_ij)
```

This makes the method especially relevant for **conditional or relationship-breaking anomalies**: observations whose individual values may be plausible, but whose multivariate relationships are unusual.

## Provenance

This repository is an extension and reimplementation of the earlier **PSOD (Pseudo-supervised outlier detection)** package by Thomas Meißner. That project predates this repository and already implemented the central feature-as-target residual idea together with much of the original parameter vocabulary.

Original project:

- Author: Thomas Meißner
- Repository: https://github.com/ThomasMeissnerDS/PSOD
- PyPI package: `psod`
- Declared license: GPL-3.0-only

The previous wording in this repository described PSOD as a "novel approach" and presented the package without this attribution. That wording was inaccurate and has been removed.

See [PROVENANCE.md](PROVENANCE.md) for the refactor and attribution policy.

## Current implementation

The current codebase includes:

- scikit-learn style estimator methods;
- configurable regression learners;
- numerical and categorical inputs;
- missing-value handling;
- transformations;
- feature contribution utilities;
- persistence helpers;
- benchmarking and visualization modules.

These capabilities are being reviewed individually during the refactor. Their presence should not be interpreted as a claim that every current interface or benchmark result is release-ready.

## Installation

The package is **not currently being advertised for PyPI installation** while the refactor is in progress. The `psod` distribution name is already used by the original project.

For development from source:

```bash
git clone https://github.com/DiogoRibeiro7/PSOD.git
cd PSOD
python -m pip install -e .
```

## Basic usage

```python
import pandas as pd

from psod import PSOD

X = pd.DataFrame(
    {
        "feature_1": [1.0, 2.0, 3.0, 4.0, 100.0],
        "feature_2": [10.0, 20.0, 30.0, 40.0, 1000.0],
    }
)

detector = PSOD(random_seed=42)
scores = detector.fit_predict(X, return_class=False)
```

The public API will be simplified during the refactor. Examples in this README are therefore intentionally minimal.

## Benchmarks

The repository contains benchmarking infrastructure, but previous README material included illustrative performance tables that were not backed by committed benchmark artifacts. Those figures should not be treated as empirical results.

A later refactor phase will rebuild the benchmark protocol around reproducible datasets, repeated runs, locked configurations, uncertainty estimates, and explicit tests of contextual versus global anomaly geometries.

## Refactor roadmap

The cleanup is being done in reviewable stages:

1. repository integrity, provenance, licensing and CI;
2. packaging and dependency cleanup;
3. public API reduction and module boundaries;
4. core algorithm rewrite with cross-fitted residuals;
5. robust score calibration and thresholding;
6. tests and property-based invariants;
7. reproducible benchmark suite;
8. documentation, examples and release preparation.

## License

This refactor treats the project as GPL-3.0-only because the earlier PSOD implementation on which it is based declares GPL-3.0-only. See [LICENSE](LICENSE) and [PROVENANCE.md](PROVENANCE.md).
