"""
Utility functions for PSOD.
"""

import pickle
import json
import joblib
import gzip
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Optional, List, Tuple
import logging
from datetime import datetime
import os
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)


def save_model(model: Any, filepath: str, format: str = "pickle") -> None:
    """
    Save a fitted PSOD model to disk.

    Parameters
    ----------
    model : PSOD
        Fitted PSOD model to save.
    filepath : str
        Path where to save the model.
    format : str, default='pickle'
        Format to use for saving ('pickle', 'joblib', 'json').
    """
    # Create directory if it doesn't exist
    os.makedirs(
        os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True
    )

    # Prepare model metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_type": type(model).__name__,
        "format_version": "1.0",
        "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
    }

    # Add model-specific metadata if available
    if hasattr(model, "get_params"):
        metadata["parameters"] = model.get_params()

    try:
        if format.lower() == "pickle":
            # Save with pickle (default compression)
            with gzip.open(f"{filepath}.gz", "wb") as f:
                pickle.dump({"model": model, "metadata": metadata}, f)
            logger.info(f"Model saved to {filepath}.gz using pickle with compression")

        elif format.lower() == "joblib":
            # Save with joblib (better for sklearn objects)
            joblib.dump(
                {"model": model, "metadata": metadata}, f"{filepath}.joblib", compress=3
            )
            logger.info(
                f"Model saved to {filepath}.joblib using joblib with compression"
            )

        elif format.lower() == "json":
            # Save serializable parts as JSON (limited support)
            if hasattr(model, "to_dict"):
                model_dict = model.to_dict()
            else:
                # Try to extract serializable attributes
                model_dict = {}
                for attr in dir(model):
                    if not attr.startswith("_"):
                        try:
                            value = getattr(model, attr)
                            if isinstance(
                                value, (int, float, str, list, dict, bool, type(None))
                            ):
                                model_dict[attr] = value
                            elif isinstance(value, np.ndarray):
                                model_dict[attr] = value.tolist()
                        except:
                            continue

            save_data = {"model": model_dict, "metadata": metadata}
            with open(f"{filepath}.json", "w") as f:
                json.dump(save_data, f, indent=2, default=str)
            logger.info(f"Model saved to {filepath}.json using JSON")

        else:
            raise ValueError(
                f"Unsupported format: {format}. Use 'pickle', 'joblib', or 'json'"
            )

    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise


def load_model(filepath: str, format: str = "pickle") -> Any:
    """
    Load a fitted PSOD model from disk.

    Parameters
    ----------
    filepath : str
        Path to the saved model.
    format : str, default='pickle'
        Format used for saving ('pickle', 'joblib', 'json').

    Returns
    -------
    PSOD
        Loaded PSOD model.
    """
    try:
        if format.lower() == "pickle":
            # Try compressed file first, then uncompressed
            if os.path.exists(f"{filepath}.gz"):
                with gzip.open(f"{filepath}.gz", "rb") as f:
                    data = pickle.load(f)
            else:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)

        elif format.lower() == "joblib":
            if os.path.exists(f"{filepath}.joblib"):
                data = joblib.load(f"{filepath}.joblib")
            else:
                data = joblib.load(filepath)

        elif format.lower() == "json":
            if os.path.exists(f"{filepath}.json"):
                with open(f"{filepath}.json", "r") as f:
                    data = json.load(f)
            else:
                with open(filepath, "r") as f:
                    data = json.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Validate loaded model integrity
        if not isinstance(data, dict) or "model" not in data:
            raise ValueError("Invalid model file format")

        # Check version compatibility
        if "metadata" in data:
            metadata = data["metadata"]
            logger.info(f"Loaded model saved at {metadata.get('timestamp', 'unknown')}")
            logger.info(f"Model type: {metadata.get('model_type', 'unknown')}")

            # Warn about version differences
            saved_version = metadata.get("format_version", "0.0")
            if saved_version != "1.0":
                warnings.warn(
                    f"Model was saved with format version {saved_version}, "
                    "current version is 1.0. Compatibility issues may occur."
                )

        model = data["model"]
        logger.info(f"Model loaded successfully from {filepath}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model from {filepath}: {str(e)}")
        raise


def validate_dataframe(
    df: pd.DataFrame,
    check_missing: bool = True,
    check_dtypes: bool = True,
    min_samples: int = 10,
) -> Dict[str, Any]:
    """
    Validate input DataFrame for outlier detection.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    check_missing : bool, default=True
        Whether to check for missing values.
    check_dtypes : bool, default=True
        Whether to check data types.
    min_samples : int, default=10
        Minimum number of samples required.

    Returns
    -------
    Dict[str, Any]
        Validation results with warnings and errors.
    """
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "info": {},
        "recommendations": [],
    }

    # Check DataFrame shape
    n_samples, n_features = df.shape
    validation_results["info"]["n_samples"] = n_samples
    validation_results["info"]["n_features"] = n_features

    if n_samples < min_samples:
        validation_results["errors"].append(
            f"Insufficient samples: {n_samples} < {min_samples}"
        )
        validation_results["is_valid"] = False

    if n_features == 0:
        validation_results["errors"].append("No features in DataFrame")
        validation_results["is_valid"] = False

    # Check for missing values
    if check_missing:
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]

        if len(missing_cols) > 0:
            validation_results["info"]["missing_values"] = missing_cols.to_dict()
            total_missing = missing_counts.sum()
            missing_percentage = (total_missing / (n_samples * n_features)) * 100

            if missing_percentage > 50:
                validation_results["errors"].append(
                    f"Excessive missing values: {missing_percentage:.1f}%"
                )
                validation_results["is_valid"] = False
            elif missing_percentage > 10:
                validation_results["warnings"].append(
                    f"High proportion of missing values: {missing_percentage:.1f}%"
                )
                validation_results["recommendations"].append(
                    "Consider imputation or removing columns with excessive missing values"
                )

    # Check data types consistency
    if check_dtypes:
        dtypes_info = {}
        for dtype in df.dtypes.unique():
            cols = df.select_dtypes(include=[dtype]).columns.tolist()
            dtypes_info[str(dtype)] = cols

        validation_results["info"]["data_types"] = dtypes_info

        # Check for mixed types that might cause issues
        object_cols = df.select_dtypes(include=["object"]).columns
        if len(object_cols) > 0:
            validation_results["warnings"].append(
                f"Found {len(object_cols)} object columns that may need encoding"
            )
            validation_results["recommendations"].append(
                "Consider encoding categorical variables or removing non-numeric columns"
            )

    # Check for constant columns
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)

    if constant_cols:
        validation_results["warnings"].append(
            f"Found {len(constant_cols)} constant columns: {constant_cols}"
        )
        validation_results["recommendations"].append(
            "Consider removing constant columns as they don't provide information"
        )
        validation_results["info"]["constant_columns"] = constant_cols

    # Check for duplicate rows
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        duplicate_percentage = (n_duplicates / n_samples) * 100
        validation_results["info"]["duplicate_rows"] = n_duplicates

        if duplicate_percentage > 20:
            validation_results["warnings"].append(
                f"High number of duplicate rows: {n_duplicates} ({duplicate_percentage:.1f}%)"
            )
            validation_results["recommendations"].append(
                "Consider removing duplicate rows"
            )

    # Check for infinite values
    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_counts > 0:
            validation_results["errors"].append(f"Found {inf_counts} infinite values")
            validation_results["is_valid"] = False

    # Check feature correlation (high correlation might indicate redundancy)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr().abs()
        # Find pairs with correlation > 0.95
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append(
                        (
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j],
                        )
                    )

        if high_corr_pairs:
            validation_results["warnings"].append(
                f"Found {len(high_corr_pairs)} highly correlated feature pairs"
            )
            validation_results["info"]["high_correlation_pairs"] = high_corr_pairs
            validation_results["recommendations"].append(
                "Consider removing redundant features or applying dimensionality reduction"
            )

    return validation_results


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "drop",
    fill_value: Optional[Union[int, float, str]] = None,
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potential missing values.
    strategy : str, default='drop'
        Strategy to handle missing values ('drop', 'mean', 'median', 'mode', 'constant', 'knn').
    fill_value : Union[int, float, str], optional
        Value to use when strategy='constant'.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    df_processed = df.copy()

    if not df_processed.isnull().any().any():
        logger.info("No missing values found")
        return df_processed

    logger.info(f"Handling missing values using strategy: {strategy}")

    if strategy == "drop":
        # Drop rows with any missing values
        initial_shape = df_processed.shape
        df_processed = df_processed.dropna()
        final_shape = df_processed.shape
        logger.info(
            f"Dropped {initial_shape[0] - final_shape[0]} rows with missing values"
        )

    elif strategy in ["mean", "median", "mode", "constant"]:
        # Apply imputation column-wise
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(
            include=["object", "category"]
        ).columns

        # Handle numeric columns
        if len(numeric_cols) > 0:
            if strategy == "constant":
                fill_val = fill_value if fill_value is not None else 0
                imputer = SimpleImputer(strategy="constant", fill_value=fill_val)
            elif strategy == "mode":
                # SimpleImputer uses 'most_frequent' instead of 'mode'
                imputer = SimpleImputer(strategy="most_frequent")
            else:
                imputer = SimpleImputer(strategy=strategy)

            df_processed[numeric_cols] = imputer.fit_transform(
                df_processed[numeric_cols]
            )

        # Handle categorical columns
        if len(categorical_cols) > 0:
            if strategy in ["mean", "median"]:
                # Use mode for categorical columns
                cat_imputer = SimpleImputer(strategy="most_frequent")
            elif strategy == "constant":
                fill_val = fill_value if fill_value is not None else "missing"
                cat_imputer = SimpleImputer(strategy="constant", fill_value=fill_val)
            else:  # mode
                cat_imputer = SimpleImputer(strategy="most_frequent")

            df_processed[categorical_cols] = cat_imputer.fit_transform(
                df_processed[categorical_cols]
            )

    elif strategy == "knn":
        # Use KNN imputation for numeric columns only
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            knn_imputer = KNNImputer(n_neighbors=5)
            df_processed[numeric_cols] = knn_imputer.fit_transform(
                df_processed[numeric_cols]
            )

        # Use mode for categorical columns
        categorical_cols = df_processed.select_dtypes(
            include=["object", "category"]
        ).columns
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            df_processed[categorical_cols] = cat_imputer.fit_transform(
                df_processed[categorical_cols]
            )

    else:
        raise ValueError(
            f"Unsupported strategy: {strategy}. "
            "Use 'drop', 'mean', 'median', 'mode', 'constant', or 'knn'"
        )

    logger.info(f"Missing value handling completed. Shape: {df_processed.shape}")
    return df_processed


def calibrate_outlier_scores(
    scores: np.ndarray, contamination: float = 0.1
) -> np.ndarray:
    """
    Calibrate outlier scores to a specific contamination level.

    Parameters
    ----------
    scores : np.ndarray
        Raw outlier scores.
    contamination : float, default=0.1
        Expected proportion of outliers.

    Returns
    -------
    np.ndarray
        Calibrated outlier scores.
    """
    if not 0 < contamination < 1:
        raise ValueError("Contamination must be between 0 and 1")

    # Find threshold corresponding to contamination level
    threshold = np.percentile(scores, (1 - contamination) * 100)

    # Scale scores so that threshold becomes 0.5
    # This makes scores more interpretable as probabilities
    calibrated_scores = np.copy(scores)

    # Min-max scaling to [0, 1] range
    min_score = np.min(scores)
    max_score = np.max(scores)

    if max_score > min_score:
        calibrated_scores = (scores - min_score) / (max_score - min_score)

        # Adjust so that threshold maps to contamination level
        threshold_normalized = (threshold - min_score) / (max_score - min_score)
        if threshold_normalized != contamination:
            # Apply power transformation to match contamination
            if threshold_normalized > 0 and threshold_normalized < 1:
                power = np.log(contamination) / np.log(threshold_normalized)
                calibrated_scores = np.power(calibrated_scores, power)

    logger.info(f"Scores calibrated for contamination level: {contamination}")
    return calibrated_scores


def combine_outlier_scores(
    scores_list: List[np.ndarray],
    method: str = "average",
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Combine multiple outlier score arrays.

    Parameters
    ----------
    scores_list : list
        List of outlier score arrays.
    method : str, default='average'
        Method to combine scores:
        - 'average': Simple average of scores
        - 'median': Median of scores (robust to outliers)
        - 'maximum': Maximum score across methods
        - 'minimum': Minimum score across methods
        - 'weighted': Weighted average (requires weights)
        - 'rank_average': Average of rank-normalized scores
        - 'geometric_mean': Geometric mean of scores
    weights : np.ndarray, optional
        Weights for weighted combination (required for 'weighted' method).

    Returns
    -------
    np.ndarray
        Combined outlier scores.
    """
    if not scores_list:
        raise ValueError("Empty scores list provided")

    # Validate score compatibility
    n_samples = len(scores_list[0])
    for i, scores in enumerate(scores_list):
        if len(scores) != n_samples:
            raise ValueError(
                f"Score array {i} has different length: {len(scores)} vs {n_samples}"
            )

    scores_array = np.array(scores_list)

    if method == "average":
        combined_scores = np.mean(scores_array, axis=0)

    elif method == "median":
        combined_scores = np.median(scores_array, axis=0)

    elif method == "maximum":
        combined_scores = np.max(scores_array, axis=0)

    elif method == "minimum":
        combined_scores = np.min(scores_array, axis=0)

    elif method == "weighted":
        if weights is None:
            raise ValueError("Weights must be provided for weighted combination")
        if len(weights) != len(scores_list):
            raise ValueError("Number of weights must match number of score arrays")

        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        combined_scores = np.average(scores_array, axis=0, weights=weights)

    elif method == "rank_average":
        # Use rank-based combination for better robustness
        ranked_scores = np.zeros_like(scores_array)
        for i, scores in enumerate(scores_list):
            ranked_scores[i] = np.argsort(np.argsort(scores)) / (len(scores) - 1)
        combined_scores = np.mean(ranked_scores, axis=0)

    elif method == "geometric_mean":
        # Geometric mean (good for combining normalized scores)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        scores_array_safe = np.maximum(scores_array, epsilon)
        combined_scores = np.exp(np.mean(np.log(scores_array_safe), axis=0))

    else:
        raise ValueError(
            f"Unsupported method: {method}. "
            "Use 'average', 'median', 'maximum', 'minimum', 'weighted', "
            "'rank_average', or 'geometric_mean'"
        )

    logger.info(f"Combined {len(scores_list)} score arrays using method: {method}")
    return combined_scores


def evaluate_outlier_detection(
    y_true: np.ndarray, y_pred: np.ndarray, metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate outlier detection performance.

    Parameters
    ----------
    y_true : np.ndarray
        True outlier labels.
    y_pred : np.ndarray
        Predicted outlier labels or scores.
    metrics : list, optional
        List of metrics to compute.

    Returns
    -------
    Dict[str, float]
        Dictionary of metric names and values.
    """
    if metrics is None:
        metrics = [
            "precision",
            "recall",
            "f1",
            "auc_roc",
            "auc_pr",
            "average_precision",
        ]

    results = {}

    # Handle both binary labels and continuous scores
    is_binary_pred = np.array_equal(y_pred, y_pred.astype(bool))

    # Convert to binary if needed for some metrics
    if not is_binary_pred:
        y_pred_binary = (y_pred > np.median(y_pred)).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)

    # Ensure y_true is binary
    y_true_binary = y_true.astype(int)

    try:
        for metric in metrics:
            if metric.lower() == "precision":
                results["precision"] = precision_score(
                    y_true_binary, y_pred_binary, zero_division=0
                )

            elif metric.lower() == "recall":
                results["recall"] = recall_score(
                    y_true_binary, y_pred_binary, zero_division=0
                )

            elif metric.lower() == "f1":
                results["f1"] = f1_score(y_true_binary, y_pred_binary, zero_division=0)

            elif metric.lower() in ["auc_roc", "roc_auc"]:
                if not is_binary_pred:
                    results["auc_roc"] = roc_auc_score(y_true_binary, y_pred)
                else:
                    results["auc_roc"] = roc_auc_score(y_true_binary, y_pred_binary)

            elif metric.lower() in ["auc_pr", "pr_auc"]:
                if not is_binary_pred:
                    precision_vals, recall_vals, _ = precision_recall_curve(
                        y_true_binary, y_pred
                    )
                    results["auc_pr"] = np.trapz(precision_vals, recall_vals)
                else:
                    precision_vals, recall_vals, _ = precision_recall_curve(
                        y_true_binary, y_pred_binary
                    )
                    results["auc_pr"] = np.trapz(precision_vals, recall_vals)

            elif metric.lower() == "average_precision":
                if not is_binary_pred:
                    results["average_precision"] = average_precision_score(
                        y_true_binary, y_pred
                    )
                else:
                    results["average_precision"] = average_precision_score(
                        y_true_binary, y_pred_binary
                    )

            else:
                logger.warning(f"Unknown metric: {metric}")

    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        # Return partial results
        pass

    # Add summary statistics
    n_outliers_true = np.sum(y_true_binary)
    n_outliers_pred = np.sum(y_pred_binary)
    results["n_outliers_true"] = n_outliers_true
    results["n_outliers_pred"] = n_outliers_pred
    results["contamination_true"] = n_outliers_true / len(y_true_binary)
    results["contamination_pred"] = n_outliers_pred / len(y_pred_binary)

    logger.info(
        f"Computed {len([k for k in results.keys() if k not in ['n_outliers_true', 'n_outliers_pred', 'contamination_true', 'contamination_pred']])} metrics"
    )
    return results


def generate_outlier_data(
    n_samples: int = 1000,
    n_features: int = 10,
    contamination: float = 0.1,
    outlier_type: str = "global",
    random_state: Optional[int] = None,
    noise_level: float = 0.1,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic data with outliers for testing.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate.
    n_features : int, default=10
        Number of features.
    contamination : float, default=0.1
        Proportion of outliers.
    outlier_type : str, default='global'
        Type of outliers to generate:
        - 'global': Far from normal data in all dimensions
        - 'local': Outliers in local neighborhoods/subspaces
        - 'collective': Groups of samples that are outliers together
        - 'contextual': Context-dependent outliers
        - 'point': Random extreme values in random features
        - 'mixed': Combination of different outlier types
    random_state : int, optional
        Random state for reproducibility.
    noise_level : float, default=0.1
        Amount of noise to add to normal data.

    Returns
    -------
    tuple
        (X, y) where X is feature matrix and y is outlier labels (0=normal, 1=outlier).
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_outliers = int(n_samples * contamination)
    n_normal = n_samples - n_outliers

    # Generate normal data from multivariate normal distribution
    mean = np.zeros(n_features)
    # Create covariance matrix with some correlation
    cov = np.eye(n_features)
    for i in range(n_features):
        for j in range(
            i + 1, min(i + 3, n_features)
        ):  # Add correlation to nearby features
            cov[i, j] = cov[j, i] = 0.3

    X_normal = np.random.multivariate_normal(mean, cov, n_normal)

    # Generate outliers based on type
    if outlier_type == "global":
        # Global outliers: far from normal data in feature space
        outlier_mean = mean + 3 * np.ones(n_features)  # Shift mean
        outlier_cov = 0.5 * np.eye(n_features)  # Smaller variance
        X_outliers = np.random.multivariate_normal(
            outlier_mean, outlier_cov, n_outliers
        )

    elif outlier_type == "local":
        # Local outliers: outliers in subspaces
        X_outliers = np.random.multivariate_normal(mean, cov, n_outliers)
        # Make outliers in random subspaces
        for i in range(n_outliers):
            # Select random subset of features to be outlying
            outlier_features = np.random.choice(
                n_features,
                size=np.random.randint(1, max(2, n_features // 2)),
                replace=False,
            )
            X_outliers[i, outlier_features] += np.random.normal(
                0, 3, len(outlier_features)
            )

    elif outlier_type == "collective":
        # Collective outliers: groups that are outliers together
        # Generate clusters of outliers
        n_clusters = max(1, n_outliers // 5)
        cluster_size = n_outliers // n_clusters

        X_outliers = []
        for cluster in range(n_clusters):
            # Random cluster center away from normal data
            cluster_center = mean + np.random.normal(0, 4, n_features)
            cluster_cov = 0.2 * np.eye(n_features)

            actual_size = cluster_size
            if cluster == n_clusters - 1:  # Last cluster gets remaining samples
                actual_size = n_outliers - cluster * cluster_size

            cluster_samples = np.random.multivariate_normal(
                cluster_center, cluster_cov, actual_size
            )
            X_outliers.extend(cluster_samples)

        X_outliers = np.array(X_outliers)

    elif outlier_type == "contextual":
        # Contextual outliers: outliers based on context (e.g., time, location)
        # Generate outliers that look normal in some features but anomalous in context
        X_outliers = np.random.multivariate_normal(mean, cov, n_outliers)

        # Add a "context" feature (e.g., time or spatial dimension)
        # Make outliers have unusual relationships between features
        for i in range(n_outliers):
            # Reverse or distort the relationship between features
            if n_features >= 2:
                # Create anticorrelation where there should be correlation
                X_outliers[i, 1] = -X_outliers[i, 0] + np.random.normal(0, 0.5)

                # Add random distortions to other feature relationships
                for j in range(2, n_features):
                    if np.random.rand() > 0.5:
                        X_outliers[i, j] *= -1  # Flip sign
                        X_outliers[i, j] += np.random.normal(0, 2)

    elif outlier_type == "point":
        # Point anomalies: random extreme values in random features
        X_outliers = np.random.multivariate_normal(mean, cov, n_outliers)

        for i in range(n_outliers):
            # Select 1-3 random features to make extreme
            n_extreme_features = np.random.randint(1, min(4, n_features + 1))
            extreme_features = np.random.choice(
                n_features, size=n_extreme_features, replace=False
            )

            # Make selected features extremely large or small
            for feat_idx in extreme_features:
                direction = np.random.choice([-1, 1])
                X_outliers[i, feat_idx] = direction * np.random.uniform(4, 7)

    elif outlier_type == "mixed":
        # Mixed outliers: combination of different types
        outlier_types = ["global", "local", "point"]
        type_counts = np.random.multinomial(n_outliers, [1/3, 1/3, 1/3])

        X_outliers_list = []

        # Generate global outliers
        if type_counts[0] > 0:
            outlier_mean = mean + 3 * np.ones(n_features)
            outlier_cov = 0.5 * np.eye(n_features)
            X_global = np.random.multivariate_normal(
                outlier_mean, outlier_cov, type_counts[0]
            )
            X_outliers_list.append(X_global)

        # Generate local outliers
        if type_counts[1] > 0:
            X_local = np.random.multivariate_normal(mean, cov, type_counts[1])
            for i in range(type_counts[1]):
                outlier_features = np.random.choice(
                    n_features,
                    size=np.random.randint(1, max(2, n_features // 2)),
                    replace=False,
                )
                X_local[i, outlier_features] += np.random.normal(0, 3, len(outlier_features))
            X_outliers_list.append(X_local)

        # Generate point outliers
        if type_counts[2] > 0:
            X_point = np.random.multivariate_normal(mean, cov, type_counts[2])
            for i in range(type_counts[2]):
                extreme_features = np.random.choice(
                    n_features, size=np.random.randint(1, 3), replace=False
                )
                for feat_idx in extreme_features:
                    direction = np.random.choice([-1, 1])
                    X_point[i, feat_idx] = direction * np.random.uniform(4, 7)
            X_outliers_list.append(X_point)

        X_outliers = np.vstack(X_outliers_list) if X_outliers_list else np.array([])

    else:
        raise ValueError(
            f"Unsupported outlier_type: {outlier_type}. "
            "Use 'global', 'local', 'collective', 'contextual', 'point', or 'mixed'"
        )

    # Combine normal and outlier data
    X = np.vstack([X_normal, X_outliers])
    y = np.hstack([np.zeros(n_normal), np.ones(n_outliers)])

    # Shuffle the data
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)

    logger.info(
        f"Generated synthetic data: {n_samples} samples, {n_features} features, "
        f"{n_outliers} outliers ({contamination * 100:.1f}%), type: {outlier_type}"
    )

    return X_df, y


# Additional utility functions
def standardize_features(
    X: pd.DataFrame, scaler: Optional[StandardScaler] = None, fit: bool = True
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize features to zero mean and unit variance.

    Parameters
    ----------
    X : pd.DataFrame
        Input features.
    scaler : StandardScaler, optional
        Pre-fitted scaler to use.
    fit : bool, default=True
        Whether to fit the scaler.

    Returns
    -------
    tuple
        (Standardized DataFrame, fitted scaler)
    """
    if scaler is None:
        scaler = StandardScaler()

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_scaled = X.copy()

    if fit:
        X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    else:
        X_scaled[numeric_cols] = scaler.transform(X[numeric_cols])

    return X_scaled, scaler


def compute_feature_importance(
    X: pd.DataFrame, outlier_scores: np.ndarray, method: str = "correlation"
) -> pd.Series:
    """
    Compute feature importance for outlier detection.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    outlier_scores : np.ndarray
        Outlier scores.
    method : str, default='correlation'
        Method to compute importance ('correlation', 'mutual_info').

    Returns
    -------
    pd.Series
        Feature importance scores.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    importance_scores = {}

    if method == "correlation":
        for col in numeric_cols:
            corr = np.abs(np.corrcoef(X[col], outlier_scores)[0, 1])
            importance_scores[col] = corr if not np.isnan(corr) else 0

    elif method == "mutual_info":
        from sklearn.feature_selection import mutual_info_regression

        mi_scores = mutual_info_regression(X[numeric_cols], outlier_scores)
        importance_scores = dict(zip(numeric_cols, mi_scores))

    else:
        raise ValueError(f"Unsupported method: {method}")

    return pd.Series(importance_scores).sort_values(ascending=False)


def compute_detailed_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_scores: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute detailed evaluation metrics including confusion matrix components.

    Parameters
    ----------
    y_true : np.ndarray
        True outlier labels (0=normal, 1=outlier).
    y_pred : np.ndarray
        Predicted outlier labels.
    y_scores : np.ndarray, optional
        Outlier scores (for threshold-based metrics).

    Returns
    -------
    Dict[str, Any]
        Dictionary with detailed metrics including:
        - accuracy, precision, recall, f1, specificity
        - true_positives, true_negatives, false_positives, false_negatives
        - confusion_matrix
        - auc_roc, auc_pr (if scores provided)
    """
    from sklearn.metrics import confusion_matrix, accuracy_score

    y_true_binary = y_true.astype(int)
    y_pred_binary = y_pred.astype(int)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true_binary, y_pred_binary),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

    # Add threshold-based metrics if scores are provided
    if y_scores is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true_binary, y_scores)
            precision_vals, recall_vals, _ = precision_recall_curve(
                y_true_binary, y_scores
            )
            metrics["auc_pr"] = np.trapz(precision_vals, recall_vals)
            metrics["average_precision"] = average_precision_score(
                y_true_binary, y_scores
            )
        except Exception as e:
            logger.warning(f"Could not compute threshold-based metrics: {str(e)}")

    # Add derived metrics
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0
    metrics["true_positive_rate"] = metrics["recall"]
    metrics["true_negative_rate"] = metrics["specificity"]

    return metrics


def select_threshold(
    scores: np.ndarray,
    method: str = "percentile",
    contamination: Optional[float] = None,
    **kwargs,
) -> float:
    """
    Select optimal threshold for outlier detection.

    Parameters
    ----------
    scores : np.ndarray
        Outlier scores.
    method : str, default='percentile'
        Method to select threshold:
        - 'percentile': Based on contamination percentile
        - 'median': Median of scores
        - 'mad': Median + k * MAD (median absolute deviation)
        - 'iqr': Q3 + k * IQR (interquartile range)
        - 'std': Mean + k * std
    contamination : float, optional
        Expected proportion of outliers (required for 'percentile').
    **kwargs : dict
        Additional arguments:
        - k: Multiplier for 'mad', 'iqr', 'std' methods (default=1.5)

    Returns
    -------
    float
        Selected threshold value.
    """
    if method == "percentile":
        if contamination is None:
            raise ValueError("contamination must be provided for percentile method")
        if not 0 < contamination < 1:
            raise ValueError("contamination must be between 0 and 1")
        threshold = np.percentile(scores, (1 - contamination) * 100)

    elif method == "median":
        threshold = np.median(scores)

    elif method == "mad":
        k = kwargs.get("k", 1.5)
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        threshold = median + k * mad

    elif method == "iqr":
        k = kwargs.get("k", 1.5)
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        threshold = q3 + k * iqr

    elif method == "std":
        k = kwargs.get("k", 2.0)
        mean = np.mean(scores)
        std = np.std(scores)
        threshold = mean + k * std

    else:
        raise ValueError(
            f"Unsupported method: {method}. "
            "Use 'percentile', 'median', 'mad', 'iqr', or 'std'"
        )

    logger.info(f"Selected threshold: {threshold:.4f} using method: {method}")
    return threshold


def normalize_scores(
    scores: np.ndarray, method: str = "minmax", clip: bool = True
) -> np.ndarray:
    """
    Normalize outlier scores to [0, 1] range.

    Parameters
    ----------
    scores : np.ndarray
        Raw outlier scores.
    method : str, default='minmax'
        Normalization method:
        - 'minmax': Min-max scaling to [0, 1]
        - 'zscore': Z-score normalization then sigmoid
        - 'rank': Rank-based normalization
        - 'sigmoid': Sigmoid transformation
    clip : bool, default=True
        Whether to clip values to [0, 1] range.

    Returns
    -------
    np.ndarray
        Normalized scores in [0, 1] range.
    """
    scores = np.asarray(scores).copy()

    if method == "minmax":
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val > min_val:
            normalized = (scores - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(scores)

    elif method == "zscore":
        mean = np.mean(scores)
        std = np.std(scores)
        if std > 0:
            z_scores = (scores - mean) / std
            # Apply sigmoid to map to [0, 1]
            normalized = 1 / (1 + np.exp(-z_scores))
        else:
            normalized = np.ones_like(scores) * 0.5

    elif method == "rank":
        # Rank-based normalization
        ranks = np.argsort(np.argsort(scores))
        normalized = ranks / (len(scores) - 1) if len(scores) > 1 else np.zeros_like(scores)

    elif method == "sigmoid":
        # Center around median and scale
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        if mad > 0:
            scaled = (scores - median) / mad
            normalized = 1 / (1 + np.exp(-scaled))
        else:
            normalized = np.ones_like(scores) * 0.5

    else:
        raise ValueError(
            f"Unsupported method: {method}. "
            "Use 'minmax', 'zscore', 'rank', or 'sigmoid'"
        )

    if clip:
        normalized = np.clip(normalized, 0, 1)

    logger.debug(f"Normalized scores using method: {method}")
    return normalized


def remove_outliers(
    df: pd.DataFrame, outlier_mask: np.ndarray, strategy: str = "remove"
) -> pd.DataFrame:
    """
    Remove or treat outliers in DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    outlier_mask : np.ndarray
        Boolean mask where True indicates outlier.
    strategy : str, default='remove'
        Strategy to handle outliers:
        - 'remove': Remove outlier rows
        - 'cap': Cap outlier values at percentile limits
        - 'impute': Replace outliers with median/mean

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers handled.
    """
    df_clean = df.copy()

    if strategy == "remove":
        df_clean = df_clean[~outlier_mask]
        logger.info(f"Removed {outlier_mask.sum()} outlier rows")

    elif strategy == "cap":
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Cap at 1st and 99th percentiles
            lower = df_clean.loc[~outlier_mask, col].quantile(0.01)
            upper = df_clean.loc[~outlier_mask, col].quantile(0.99)
            df_clean.loc[outlier_mask, col] = df_clean.loc[outlier_mask, col].clip(
                lower, upper
            )
        logger.info(f"Capped {outlier_mask.sum()} outlier rows at percentile limits")

    elif strategy == "impute":
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Replace with median of normal data
            median_val = df_clean.loc[~outlier_mask, col].median()
            df_clean.loc[outlier_mask, col] = median_val
        logger.info(
            f"Imputed {outlier_mask.sum()} outlier rows with median values"
        )

    else:
        raise ValueError(
            f"Unsupported strategy: {strategy}. Use 'remove', 'cap', or 'impute'"
        )

    return df_clean


def create_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive feature summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with statistics for each feature.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    summary_data = []

    for col in numeric_cols:
        col_data = df[col].dropna()

        stats = {
            "feature": col,
            "dtype": str(df[col].dtype),
            "count": len(col_data),
            "missing": df[col].isnull().sum(),
            "missing_pct": (df[col].isnull().sum() / len(df)) * 100,
            "unique": df[col].nunique(),
            "mean": col_data.mean(),
            "std": col_data.std(),
            "min": col_data.min(),
            "25%": col_data.quantile(0.25),
            "50%": col_data.quantile(0.50),
            "75%": col_data.quantile(0.75),
            "max": col_data.max(),
            "skew": col_data.skew(),
            "kurtosis": col_data.kurtosis(),
        }

        summary_data.append(stats)

    summary_df = pd.DataFrame(summary_data)
    logger.info(f"Created feature summary for {len(numeric_cols)} numeric columns")

    return summary_df


def detect_feature_drift(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Detect feature drift between reference and current datasets.

    Parameters
    ----------
    df_reference : pd.DataFrame
        Reference (training) dataset.
    df_current : pd.DataFrame
        Current (test/production) dataset.
    threshold : float, default=0.1
        Threshold for drift detection (relative difference).

    Returns
    -------
    Dict[str, Any]
        Dictionary with drift analysis results.
    """
    numeric_cols = df_reference.select_dtypes(include=[np.number]).columns
    common_cols = list(set(numeric_cols) & set(df_current.columns))

    drift_results = {
        "drifted_features": [],
        "drift_scores": {},
        "statistics": {},
    }

    for col in common_cols:
        ref_mean = df_reference[col].mean()
        ref_std = df_reference[col].std()
        curr_mean = df_current[col].mean()
        curr_std = df_current[col].std()

        # Calculate relative difference in mean
        if ref_mean != 0:
            mean_drift = abs(curr_mean - ref_mean) / abs(ref_mean)
        else:
            mean_drift = abs(curr_mean - ref_mean)

        # Calculate relative difference in std
        if ref_std != 0:
            std_drift = abs(curr_std - ref_std) / abs(ref_std)
        else:
            std_drift = abs(curr_std - ref_std)

        # Combined drift score
        drift_score = (mean_drift + std_drift) / 2

        drift_results["drift_scores"][col] = drift_score
        drift_results["statistics"][col] = {
            "ref_mean": ref_mean,
            "curr_mean": curr_mean,
            "ref_std": ref_std,
            "curr_std": curr_std,
            "mean_drift": mean_drift,
            "std_drift": std_drift,
        }

        if drift_score > threshold:
            drift_results["drifted_features"].append(col)

    logger.info(
        f"Detected drift in {len(drift_results['drifted_features'])} out of {len(common_cols)} features"
    )

    return drift_results


# Export all functions for easy import
__all__ = [
    # Model persistence
    "save_model",
    "load_model",
    # Data validation and preprocessing
    "validate_dataframe",
    "handle_missing_values",
    "standardize_features",
    "remove_outliers",
    "create_feature_summary",
    "detect_feature_drift",
    # Outlier score processing
    "calibrate_outlier_scores",
    "combine_outlier_scores",
    "normalize_scores",
    "select_threshold",
    # Evaluation metrics
    "evaluate_outlier_detection",
    "compute_detailed_metrics",
    # Feature importance
    "compute_feature_importance",
    # Synthetic data generation
    "generate_outlier_data",
]
