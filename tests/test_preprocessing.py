"""Focused tests for the extracted numeric preprocessing boundary."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from psod._preprocessing import (
    correlation_feature_selection,
    fit_transform_numeric_data,
    intersect_columns,
    remove_zero_variance,
    transform_numeric_data,
)


def test_box_cox_offsets_non_positive_values() -> None:
    """Box-Cox fitting preserves the legacy offset-and-warn behavior."""
    frame = pd.DataFrame(
        {
            "a": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    with pytest.warns(UserWarning, match="Box-Cox transformation requires positive values"):
        transformed, encoder = fit_transform_numeric_data(
            frame,
            algorithm="box-cox",
            random_seed=17,
        )

    assert isinstance(encoder, PowerTransformer)
    assert transformed.index.equals(frame.index)
    assert transformed.columns.equals(frame.columns)
    assert np.isfinite(transformed.to_numpy()).all()


def test_quantile_fit_state_is_reused_for_prediction_transform() -> None:
    """Quantile fitting returns reusable transform state with stable frame metadata."""
    frame = pd.DataFrame(
        {
            "a": np.linspace(-2.0, 2.0, 32),
            "b": np.linspace(3.0, 7.0, 32) ** 2,
        }
    )

    fitted, encoder = fit_transform_numeric_data(
        frame,
        algorithm="quantile",
        random_seed=23,
    )

    assert isinstance(encoder, QuantileTransformer)
    transformed = transform_numeric_data(
        frame,
        algorithm="quantile",
        encoder=encoder,
    )

    pd.testing.assert_frame_equal(fitted, transformed)


def test_unknown_fit_transform_algorithm_fails_fast() -> None:
    """The extracted helper keeps the estimator's unknown-algorithm error contract."""
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="Unknown transformation algorithm"):
        fit_transform_numeric_data(
            frame,
            algorithm="not-a-transform",
            random_seed=1,
        )


def test_feature_filter_helpers_preserve_legacy_semantics() -> None:
    """Zero-variance, correlation, and intersection helpers keep old behavior."""
    frame = pd.DataFrame(
        {
            "target": [0.0, 1.0, 2.0, 3.0],
            "strong": [0.0, 2.0, 4.0, 6.0],
            "weak": [0.0, 1.0, 0.0, 1.0],
            "constant": [5.0, 5.0, 5.0, 5.0],
            "label": ["a", "b", "c", "d"],
        }
    )

    assert remove_zero_variance(frame[["target", "strong", "constant"]]) == [
        "target",
        "strong",
    ]
    correlation_frame = frame.drop(columns=["constant"])
    assert correlation_feature_selection(correlation_frame, "target", threshold=0.9) == [
        "strong"
    ]
    assert intersect_columns(["z", "b", "b", "a"], ["b", "a", "q"]) == ["a", "b"]
