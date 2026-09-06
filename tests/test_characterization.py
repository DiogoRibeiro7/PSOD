"""Characterization tests for behavior that must survive the core refactor."""

import numpy as np
import pandas as pd

from psod import PSOD


def _numeric_frame(n_rows: int = 24) -> pd.DataFrame:
    """Create a small deterministic frame with non-trivial feature relationships."""
    x = np.linspace(-2.0, 2.0, n_rows)
    return pd.DataFrame(
        {
            "a": x,
            "b": 1.7 * x + np.sin(x),
            "c": -0.8 * x + np.cos(2.0 * x),
            "d": x**2 + 0.3 * x,
        }
    )


def _detector(*, n_jobs: int) -> PSOD:
    """Build a deterministic detector suitable for characterization tests."""
    return PSOD(
        n_jobs=n_jobs,
        min_cols_chosen=1.0,
        max_cols_chosen=1.0,
        correlation_threshold=0.0,
        transform_algorithm="none",
        random_seed=17,
        contamination=0.1,
    )


def test_serial_and_parallel_fit_are_equivalent() -> None:
    """Parallel execution must not change scores or selected model inputs."""
    frame = _numeric_frame()
    serial = _detector(n_jobs=1)
    parallel = _detector(n_jobs=2)

    serial_scores = serial.fit_predict(frame)
    parallel_scores = parallel.fit_predict(frame)

    assert serial_scores.index.equals(parallel_scores.index)
    np.testing.assert_allclose(
        serial_scores.to_numpy(), parallel_scores.to_numpy(), rtol=0, atol=1e-12
    )
    assert serial.chosen_columns == parallel.chosen_columns
    assert serial.prediction_errors_ is not None
    assert parallel.prediction_errors_ is not None
    assert serial.prediction_errors_.keys() == parallel.prediction_errors_.keys()
    for column in serial.prediction_errors_:
        np.testing.assert_allclose(
            serial.prediction_errors_[column],
            parallel.prediction_errors_[column],
            rtol=0,
            atol=1e-12,
        )


def test_refit_replaces_fit_state_instead_of_accumulating_it() -> None:
    """Reusing an estimator must not retain regressors or metadata from the old fit."""
    detector = _detector(n_jobs=1)
    first = _numeric_frame()
    second = first.loc[:, ["a", "b", "c"]].rename(columns={"a": "x", "b": "y", "c": "z"})

    detector.fit_predict(first)
    detector.fit_predict(second)

    expected_columns = set(second.columns)
    assert detector.feature_names_ == list(second.columns)
    assert detector.n_features_in_ == len(second.columns)
    assert set(detector.regressors) == expected_columns
    assert set(detector.chosen_columns) == expected_columns
    assert detector.prediction_errors_ is not None
    assert set(detector.prediction_errors_) == expected_columns
    assert detector.cat_encoders == {}


def test_predict_on_training_frame_reproduces_fit_scores() -> None:
    """Predicting the training frame must reproduce the fitted anomaly scores."""
    frame = _numeric_frame()
    detector = _detector(n_jobs=1)

    fitted_scores = detector.fit_predict(frame)
    predicted_scores = detector.predict(frame)

    assert fitted_scores.index.equals(predicted_scores.index)
    np.testing.assert_allclose(
        fitted_scores.to_numpy(), predicted_scores.to_numpy(), rtol=0, atol=1e-12
    )


def test_fit_and_predict_do_not_mutate_caller_dataframe() -> None:
    """Estimator operations must leave the caller-owned DataFrame unchanged."""
    frame = _numeric_frame()
    original = frame.copy(deep=True)
    detector = _detector(n_jobs=1)

    detector.fit_predict(frame)
    pd.testing.assert_frame_equal(frame, original)

    detector.predict(frame)
    pd.testing.assert_frame_equal(frame, original)
