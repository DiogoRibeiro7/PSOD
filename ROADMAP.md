# PSOD Refactor Roadmap

PSOD is being rehabilitated in small, reviewable stages. The order matters: repository integrity and reproducibility come before algorithm changes.

## Completed

### 1. Baseline integrity

- correct provenance and attribution;
- align licensing with the inherited GPL-3.0-only codebase;
- remove unsupported novelty and benchmark claims;
- repair CI and security scanning;
- disable publication under the already-occupied `psod` distribution name;
- align the supported Python floor with the tested matrix.

## In progress

### 2. Packaging and repository hygiene

- remove dead packaging and dependency-management paths;
- install docs and containers from `pyproject.toml` rather than missing requirements files;
- remove committed generated reports and implementation-summary artifacts;
- reduce runtime dependencies and separate optional visualization tooling;
- audit documentation and examples for stale setup instructions;
- decide on a distinct distribution name before any public release.

## Planned

### 3. Characterization and API boundaries

- add characterization tests for current behavior;
- document the actual public API;
- split validation, preprocessing, modelling, scoring, and persistence concerns;
- reduce accidental exports and duplicate interfaces.

### 4. Core scoring redesign

- replace mixed in-sample residual scoring with cross-fitted or out-of-fold residuals;
- distinguish training residual calibration from scoring new observations;
- make randomness and fold assignment reproducible.

### 5. Robust calibration

- evaluate robust feature-wise residual normalization;
- define aggregation and threshold semantics explicitly;
- expose feature-level score contributions.

### 6. Invariants and edge cases

- deterministic-seed invariants;
- duplicate-row and scale behavior;
- constant and collinear columns;
- missing values and unseen categories;
- high-dimensional and small-sample regimes.

### 7. Reproducible benchmarks

- benchmark by anomaly geometry rather than a single aggregate leaderboard;
- include global, local, dependency-breaking, and contextual anomalies;
- commit configurations and machine-readable results;
- report uncertainty and repeated-run variability;
- avoid unsupported best-method claims.

### 8. Release preparation

- choose a non-conflicting distribution name;
- validate documentation and examples from built artifacts;
- restore publication only after the package identity is settled;
- prepare a release candidate with reproducible benchmark evidence.
