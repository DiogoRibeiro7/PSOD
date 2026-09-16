"""Behavior-preserving numeric preprocessing helpers for PSOD."""

from __future__ import annotations

import warnings
from typing import TypeAlias

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

NumericEncoder: TypeAlias = PowerTransformer | QuantileTransformer


def fit_transform_numeric_data(
    df: pd.DataFrame,
    *,
    algorithm: str | None,
    random_seed: int,
    existing_encoder: NumericEncoder | None = None,
) -> tuple[pd.DataFrame, NumericEncoder | None]:
    """Fit the configured numeric transform and return transformed data and state."""
    if algorithm == "logarithmic":
        df_min = df.min().min()
        offset = abs(df_min) + 1 if df_min <= 0 else 0
        return np.log1p(df + offset), existing_encoder  # type: ignore[no-any-return]

    if algorithm == "yeo-johnson":
        encoder = PowerTransformer(method="yeo-johnson")
        transformed = encoder.fit_transform(df)
        return pd.DataFrame(transformed, columns=df.columns, index=df.index), encoder

    if algorithm == "box-cox":
        df_min = df.min().min()
        if df_min <= 0:
            warnings.warn("Box-Cox transformation requires positive values. Adding offset.")
            df = df - df_min + 1
        encoder = PowerTransformer(method="box-cox")
        transformed = encoder.fit_transform(df)
        return pd.DataFrame(transformed, columns=df.columns, index=df.index), encoder

    if algorithm == "quantile":
        encoder = QuantileTransformer(output_distribution="normal", random_state=random_seed)
        transformed = encoder.fit_transform(df)
        return pd.DataFrame(transformed, columns=df.columns, index=df.index), encoder

    if algorithm in ["none", None]:
        return df, existing_encoder

    raise ValueError(f"Unknown transformation algorithm: {algorithm}")


def transform_numeric_data(
    df: pd.DataFrame,
    *,
    algorithm: str | None,
    encoder: NumericEncoder | None,
) -> pd.DataFrame:
    """Apply the fitted numeric transform while preserving legacy semantics."""
    if algorithm == "logarithmic":
        # Preserve the existing prediction behavior: the offset is recomputed
        # from the frame being transformed rather than stored during fitting.
        df_min = df.min().min()
        offset = abs(df_min) + 1 if df_min <= 0 else 0
        return np.log1p(df + offset)  # type: ignore[no-any-return]

    if algorithm in ["yeo-johnson", "box-cox", "quantile"] and encoder is not None:
        transformed = encoder.transform(df)
        return pd.DataFrame(transformed, columns=df.columns, index=df.index)

    return df


def remove_zero_variance(df: pd.DataFrame) -> list[str]:
    """Return columns with non-zero population variance."""
    variance = df.var(ddof=0)
    return variance[variance != 0].index.to_list()


def correlation_feature_selection(
    df: pd.DataFrame,
    target_col: str,
    *,
    threshold: float,
) -> list[str]:
    """Return numeric predictors whose absolute correlation exceeds the threshold."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    return [
        col
        for col in numerical_cols
        if col != target_col and abs(df[col].corr(df[target_col])) > threshold
    ]


def intersect_columns(left: list[str], right: list[str]) -> list[str]:
    """Return the sorted unique intersection used by the legacy estimator."""
    return np.intersect1d(left, right).tolist()  # type: ignore[no-any-return]
