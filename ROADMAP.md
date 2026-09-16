# PSOD Roadmap

PSOD is being rehabilitated in small, reviewable stages. The order is deliberate: provenance, packaging integrity, behavioral characterization, and internal module boundaries come before any scientific redesign of the anomaly score.

The active planning source of truth is `.project.json`. This roadmap explains the broader engineering and scientific sequence around those managed milestones.

## Current managed milestone

### M01 — Complete the behavior-preserving PSOD module split

Status: **WIP**

The goal is to reduce `core.py` coupling without changing score semantics or the public `PSOD` estimator facade.

| Acceptance criterion | Status |
| --- | --- |
| A1 — Extract stateful missing-value handling without changing fitted-imputer behavior | ✅ Done |
| A2 — Extract numeric transformation and feature-filtering concerns from the estimator facade | 🟡 In progress |
| A3 — Isolate per-feature model fitting and prediction-error calculation behind internal boundaries | ⬜ Pending |
| A4 — Reduce remaining coupling in persistence and explanation utilities while preserving the facade and characterization tests | ⬜ Pending |

A1 was completed by the stateful imputation extraction. A2 is the current refactor slice and must remain behavior-preserving.

### M01 exit gate

M01 is complete only when:

- `PSOD` remains the public estimator facade;
- characterization tests remain green;
- fit/predict score behavior remains equivalent to the pre-redesign implementation;
- stateful preprocessing remains reproducible and reusable at prediction time;
- per-feature fitting and prediction-error logic no longer dominate `core.py`;
- persistence/explanation coupling is reduced where this can be done without public API breakage.

Only after this gate is satisfied should the project change scoring semantics.

## Completed foundation

### 1. Baseline integrity — completed

- correct provenance and attribution;
- align licensing with the inherited GPL-3.0-only codebase;
- remove unsupported novelty and benchmark claims;
- repair CI and security scanning;
- disable publication under the already-occupied `psod` distribution name;
- align the supported Python floor with the tested matrix.

### 2. Packaging and repository hygiene — substantially completed

Completed work includes:

- remove dead packaging and dependency-management paths;
- install docs and containers from `pyproject.toml` rather than missing requirements files;
- remove committed generated reports and implementation-summary artifacts;
- separate optional visualization dependencies from the core runtime;
- remove unconditional `tqdm` progress output and the direct runtime dependency;
- validate the built package and core-only installation path in CI.

Remaining hygiene work is secondary to M01:

- audit stale documentation/example setup instructions;
- remove obsolete CODEOWNERS and workflow references to deleted packaging files;
- simplify redundant documentation dependency installation;
- remove the upstream PyPI link until a new distribution name is chosen;
- align remaining repository metadata before release.

### 3. Characterization baseline — completed enough to support refactoring

The behavior-preserving refactor is protected by characterization of:

- serial versus parallel equivalence;
- refit state replacement;
- consistency between fit scores and `predict` on the training frame;
- caller-owned DataFrame immutability;
- input coercion and optional datetime conversion;
- stateful missing-value handling and fitted-imputer reuse.

These tests are the scientific control for M01: refactoring may change structure, but not the characterized behavior.

## Next engineering milestones

### M02 — Leakage-controlled residual scoring

Do not begin M02 until M01 is complete.

The current score mixes model fitting and residual evaluation on overlapping observations. M02 will make the training/scoring separation explicit.

Planned work:

- implement cross-fitted, out-of-fold, or otherwise leakage-controlled residuals;
- define fold construction and per-feature fitting rules explicitly;
- make fold assignment, random feature choices, and parallel execution reproducible;
- define small-sample and high-dimensional fallbacks before selecting a default strategy;
- retain a comparison path to the characterized legacy score so scientific changes are measurable rather than inferred.

Exit gate:

- no training observation is scored by a predictor fitted on that same observation unless explicitly documented as a legacy comparison path;
- deterministic seeds reproduce fold structure and scores;
- legacy and redesigned scores can be compared on the same datasets.

### M03 — Robust calibration and aggregation

Planned work:

- evaluate robust feature-wise residual normalization such as median/MAD and empirical-tail alternatives;
- define aggregation explicitly rather than inheriting it from implementation details;
- separate continuous score calibration from binary threshold selection;
- expose feature-level residual contributions consistently with the redesigned score;
- investigate whether categorical targets require classification rather than regression on encoded categories.

Exit gate:

- residual normalization and aggregation are explicit, tested components;
- threshold selection is not conflated with score construction;
- explanation outputs correspond mathematically to the implemented score.

### M04 — Statistical invariants and edge cases

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

Exit gate:

- invariants have executable tests;
- known non-invariances are documented rather than silently assumed away.

### M05 — Reproducible scientific benchmarks

Benchmark the scientific question rather than producing a single leaderboard.

Planned work:

- stratify synthetic and real-data experiments by anomaly geometry;
- include global, local, dependency-breaking, and contextual anomalies;
- compare conditional predictive residual detection with geometric and density baselines;
- commit benchmark configurations and machine-readable results;
- repeat stochastic runs and report uncertainty;
- separate exploratory benchmark results from claims strong enough for documentation or publication;
- never restore unsupported “best method” claims without reproducible evidence.

Exit gate:

- benchmark configurations are versioned;
- stochastic uncertainty is reported;
- claims in documentation are traceable to committed evidence.

## Release preparation

### M06 — Package identity and release readiness

Release remains blocked until the package identity conflict is resolved.

Required work:

- choose and verify a non-conflicting distribution name; the Python import namespace may remain `psod` if appropriate;
- remove or replace references that imply the upstream `psod` PyPI project is this package;
- complete license-file hygiene, including canonical GPL text if required by the chosen distribution process;
- validate documentation and examples from built artifacts;
- ensure release automation publishes only under the new verified distribution identity;
- prepare a release candidate only after redesigned scoring and benchmark evidence are stable.

### 1.0 readiness criteria

A 1.0 release should mean methodological and engineering stability, not feature accumulation. The minimum bar is:

- leakage-controlled score construction is the documented default;
- score calibration and aggregation semantics are stable;
- the public estimator API has an explicit compatibility policy;
- statistical invariants and important edge cases are tested;
- reproducible benchmark evidence exists for the claims made in documentation;
- package identity is conflict-free;
- build, install, docs, security, and release workflows are reproducible;
- migration from the characterized legacy score is documented.

## Deferred until justified

- broad dependency version churn unrelated to an active compatibility or security problem;
- performance claims before the benchmark redesign exists;
- large public API removals before internal compatibility boundaries are understood;
- bulk issue cleanup until generated/stale issues can be classified deterministically;
- publication to PyPI before the package identity conflict is resolved;
- new anomaly-score variants that bypass the M01 → M05 validation sequence.
