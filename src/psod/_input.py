"""Internal input normalization and validation for the PSOD estimator."""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def convert_datetime_columns(df: pd.DataFrame, *, enabled: bool) -> pd.DataFrame:
    """Convert datetime columns to integer nanoseconds when explicitly enabled."""
    if not enabled:
        return df

    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    if len(datetime_cols) == 0:
        return df

    converted = df.copy()
    for col in datetime_cols:
        converted[col] = converted[col].astype("int64")

    logger.info("Converted datetime columns to numeric: %s", list(datetime_cols))
    return converted


def to_dataframe(
    X: Union[pd.DataFrame, np.ndarray],
    *,
    feature_names: Optional[List[str]],
) -> pd.DataFrame:
    """Return a DataFrame while preserving fitted feature names for ndarray input."""
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, np.ndarray):
        n_features = X.shape[1] if X.ndim > 1 else 1
        columns = feature_names or [f"feature_{i}" for i in range(n_features)]
        return pd.DataFrame(X, columns=columns)
    raise TypeError(f"Expected DataFrame or ndarray, got {type(X)}")


def validate_input(
    df: pd.DataFrame,
    *,
    is_training: bool,
    min_samples: int,
    cat_columns: Optional[List[str]],
    is_fitted: bool,
    feature_names: Optional[List[str]],
) -> None:
    """Validate estimator input without mutating data or fitted estimator state."""
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].unique().tolist()
        raise ValueError(f"Input DataFrame has duplicate column names: {dupes}")

    if is_training and len(df) < min_samples:
        raise ValueError(
            f"Input DataFrame has {len(df)} samples, but minimum required is {min_samples}"
        )

    numeric_df = df.select_dtypes(include=[np.number])
    if (not is_training) and (not numeric_df.empty) and np.isinf(numeric_df.to_numpy()).any():
        raise ValueError(
            "Input DataFrame contains infinite values. Replace or drop them before prediction."
        )

    if df.isnull().any().any():
        logger.warning("Missing values detected. Consider imputation before fitting.")

    if numeric_df.empty:
        raise ValueError("No numeric columns found in DataFrame")

    cat_cols_set = set(cat_columns or [])
    non_numeric_cols = [
        col for col in df.columns if col not in numeric_df.columns and col not in cat_cols_set
    ]
    if non_numeric_cols:
        logger.warning(
            "Non-numeric columns will be ignored unless specified in cat_columns: %s",
            non_numeric_cols,
        )

    if cat_columns:
        missing_cat_cols = set(cat_columns) - set(df.columns)
        if missing_cat_cols:
            raise ValueError(f"Categorical columns not found in DataFrame: {missing_cat_cols}")
        numeric_cat_cols = [col for col in cat_columns if col in numeric_df.columns]
        if numeric_cat_cols:
            raise ValueError(
                "Categorical columns must be non-numeric. "
                f"Found numeric categorical columns: {numeric_cat_cols}"
            )

    if is_fitted and feature_names:
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Features missing from input: {missing_features}")

    logger.debug("Input validation passed. DataFrame shape: %s", df.shape)
