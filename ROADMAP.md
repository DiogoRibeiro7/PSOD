# PSOD Refactor Roadmap

PSOD is being rehabilitated in small, reviewable stages. The order matters: provenance, packaging integrity, behavioral characterization, and module boundaries come before any scientific redesign of the anomaly score.

## Current state

### 1. Baseline integrity — completed

- correct provenance and attribution;
- align licensing with the inherited GPL-3.0-only codebase;
- remove unsupported novelty and benchmark claims;
- repair CI and security scanning;
- disable publication under the already-occupied `psod` distribution name;
- align the supported Python floor with the tested matrix.

### 2. Packaging and repository hygiene — substantially completed

- remove dead packaging and dependency-management paths;
- install docs and containers from `pyproject.toml` rather than missing requirements files;
- remove committed generated reports and implementation-summary artifacts;
- separate optional visualization dependencies from the core runtime;
- remove unconditional `tqdm` progress output and the direct runtime dependency;
- validate the built package and core-only installation path in CI.

Remaining hygiene items are intentionally secondary to the current core refactor:

- audit stale documentation/example setup instructions;
- remove obsolete CODEOWNERS and workflow references to deleted packaging files;
- simplify redundant documentation dependency installation;
- remove the upstream PyPI link until a new distribution name is chosen;
- align remaining repository metadata before release.

## Current engineering priority

### 3. Characterization and module boundaries — in progress

Completed:

- characterize serial versus parallel equivalence;
- characterize refit state replacement;
- characterize consistency between fit scores and `predict` on the training frame;
- characterize caller-owned DataFrame immutability;
- remove unconditional progress-bar side effects from estimator calls;
- extract stateless DataFrame coercion, optional datetime conversion, and input validation into `src/psod/_input.py` while preserving the existing private estimator delegates.

Next, before changing scoring semantics:

1. extract stateful missing-value handling without changing fitted-imputer behavior;
2. extract numeric transformation and feature-filtering concerns;
3. isolate per-feature model fitting and prediction-error calculation;
4. isolate persistence and explanation utilities where doing so reduces `core.py` coupling;
5. document the intended public estimator surface and reduce accidental exports only after compatibility-sensitive internal boundaries are stable.

The acceptance criterion for this stage is structural: `PSOD` remains the public estimator facade, characterization tests stay green, and the algorithm still produces the same scores as the pre-redesign implementation.

## Scientific redesign

### 4. Core residual scoring redesign

Do not begin this stage until the behavior-preserving module split is stable.

- replace the current mixed in-sample residual scoring with cross-fitted, out-of-fold, or otherwise leakage-controlled residuals;
- define the training/scoring split explicitly so calibration residuals are not evaluated on the same observations used to fit the corresponding predictor;
- make fold assignment, per-feature random choices, and parallel execution reproducible;
- define behavior for small-sample and high-dimensional regimes before selecting a default cross-fitting strategy;
- preserve a comparison path to the characterized legacy score so scientific changes can be measured rather than inferred.

### 5. Robust calibration and aggregation

- evaluate robust feature-wise residual normalization such as median/MAD and empirical-tail alternatives;
- define score aggregation explicitly rather than inheriting it accidentally from implementation details;
- distinguish score calibration from binary threshold selection;
- expose feature-level residual contributions in a form consistent with the redesigned score;
- investigate whether categorical targets require classification rather than regression on encoded categories.

### 6. Invariants and edge cases

Add property/invariant coverage around the redesigned estimator:

- deterministic-seed invariants;
- duplicate-row behavior;
- affine scaling behavior;
- contamination-threshold monotonicity;
- constant and collinear columns;
- missing values and unseen categories;
- categorical level changes;
- high-dimensional and small-sample regimes;
- explicit leakage checks for training-score construction.

### 7. Reproducible scientific benchmarks

Benchmark the scientific question rather than producing a single leaderboard.

- stratify synthetic and real-data experiments by anomaly geometry;
- include global, local, dependency-breaking, and contextual anomalies;
- compare conditional predictive residual detection with geometric/density baselines;
- commit benchmark configurations and machine-readable results;
- repeat stochastic runs and report uncertainty;
- separate exploratory benchmark results from claims that are strong enough for documentation or publication;
- never restore unsupported "best method" claims without reproducible evidence.

## Release preparation

### 8. Package identity and release readiness

Release remains blocked until the package identity is resolved.

- choose and verify a non-conflicting distribution name; the Python import namespace may remain `psod` if appropriate;
- remove or replace references that imply the upstream `psod` PyPI project is this package;
- complete license-file hygiene, including the canonical GPL text if required by the chosen distribution process;
- validate documentation and examples from built artifacts;
- ensure release automation publishes only under the new verified distribution identity;
- prepare a release candidate only after the redesigned scoring method and reproducible benchmark evidence are stable.

## Deferred until justified

- broad dependency version churn unrelated to an active compatibility/security problem;
- performance claims before the benchmark redesign exists;
- large public API removals before internal compatibility boundaries are understood;
- bulk issue cleanup until generated/stale issues can be classified deterministically;
- publication to PyPI before the package identity conflict is resolved.
