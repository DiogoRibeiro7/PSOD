"""
Dataset generation and loading for benchmarking.
Provides various synthetic and real-world datasets with different outlier types.
"""

import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_classification

warnings.filterwarnings("ignore")


def generate_global_outliers(
    n_samples: int = 1000,
    n_features: int = 20,
    contamination: float = 0.1,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset with global outliers (far from all normal points).

    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_features : int
        Number of features
    contamination : float
        Proportion of outliers
    random_state : int, optional
        Random seed

    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels (0=normal, 1=outlier)
    """
    rng = np.random.RandomState(random_state)

    n_outliers = int(n_samples * contamination)
    n_normal = n_samples - n_outliers

    # Generate normal data (Gaussian)
    X_normal = rng.randn(n_normal, n_features)

    # Generate global outliers (uniform in wider range)
    X_outliers = rng.uniform(-10, 10, (n_outliers, n_features))

    # Combine
    X = np.vstack([X_normal, X_outliers])
    y = np.hstack([np.zeros(n_normal), np.ones(n_outliers)])

    # Shuffle
    idx = rng.permutation(n_samples)

    return X[idx], y[idx]


def generate_local_outliers(
    n_samples: int = 1000,
    n_features: int = 20,
    contamination: float = 0.1,
    n_clusters: int = 5,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset with local outliers (far from local neighborhood).

    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_features : int
        Number of features
    contamination : float
        Proportion of outliers
    n_clusters : int
        Number of clusters
    random_state : int, optional
        Random seed

    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels (0=normal, 1=outlier)
    """
    rng = np.random.RandomState(random_state)

    n_outliers = int(n_samples * contamination)
    n_normal = n_samples - n_outliers

    # Generate clustered normal data
    X_normal, _ = make_blobs(
        n_samples=n_normal,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=random_state,
    )

    # Generate local outliers (between clusters)
    cluster_centers = np.array(
        [X_normal[np.random.choice(len(X_normal))] for _ in range(n_clusters)]
    )

    X_outliers = []
    for _ in range(n_outliers):
        # Pick two random cluster centers
        c1, c2 = cluster_centers[rng.choice(n_clusters, 2, replace=False)]
        # Place outlier between them with some noise
        outlier = (c1 + c2) / 2 + rng.randn(n_features) * 0.5
        X_outliers.append(outlier)

    X_outliers = np.array(X_outliers)

    # Combine
    X = np.vstack([X_normal, X_outliers])
    y = np.hstack([np.zeros(n_normal), np.ones(n_outliers)])

    # Shuffle
    idx = rng.permutation(n_samples)

    return X[idx], y[idx]


def generate_collective_outliers(
    n_samples: int = 1000,
    n_features: int = 20,
    contamination: float = 0.1,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset with collective outliers (small cluster of outliers).

    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_features : int
        Number of features
    contamination : float
        Proportion of outliers
    random_state : int, optional
        Random seed

    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels (0=normal, 1=outlier)
    """
    rng = np.random.RandomState(random_state)

    n_outliers = int(n_samples * contamination)
    n_normal = n_samples - n_outliers

    # Generate normal data
    X_normal = rng.randn(n_normal, n_features)

    # Generate collective outliers (small cluster far from normal)
    outlier_center = rng.randn(n_features) * 10
    X_outliers = outlier_center + rng.randn(n_outliers, n_features) * 0.5

    # Combine
    X = np.vstack([X_normal, X_outliers])
    y = np.hstack([np.zeros(n_normal), np.ones(n_outliers)])

    # Shuffle
    idx = rng.permutation(n_samples)

    return X[idx], y[idx]


def generate_contextual_outliers(
    n_samples: int = 1000,
    n_features: int = 20,
    contamination: float = 0.1,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset with contextual outliers (outliers in specific context).

    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_features : int
        Number of features
    contamination : float
        Proportion of outliers
    random_state : int, optional
        Random seed

    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels (0=normal, 1=outlier)
    """
    rng = np.random.RandomState(random_state)

    n_outliers = int(n_samples * contamination)
    n_normal = n_samples - n_outliers

    # Generate normal data with correlation
    X_normal = rng.randn(n_normal, n_features)
    # Add correlation: feature 1 depends on feature 0
    X_normal[:, 1] = X_normal[:, 0] * 2 + rng.randn(n_normal) * 0.1

    # Generate contextual outliers (break correlation)
    X_outliers = rng.randn(n_outliers, n_features)
    # Break correlation
    X_outliers[:, 1] = -X_outliers[:, 0] * 2 + rng.randn(n_outliers) * 0.1

    # Combine
    X = np.vstack([X_normal, X_outliers])
    y = np.hstack([np.zeros(n_normal), np.ones(n_outliers)])

    # Shuffle
    idx = rng.permutation(n_samples)

    return X[idx], y[idx]


def generate_mixed_outliers(
    n_samples: int = 1000,
    n_features: int = 20,
    contamination: float = 0.1,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset with mixed outlier types.

    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_features : int
        Number of features
    contamination : float
        Proportion of outliers
    random_state : int, optional
        Random seed

    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels (0=normal, 1=outlier)
    """
    rng = np.random.RandomState(random_state)

    n_outliers = int(n_samples * contamination)
    n_normal = n_samples - n_outliers

    # Generate normal data
    X_normal = rng.randn(n_normal, n_features)

    # Split outliers into different types
    n_global = n_outliers // 3
    n_local = n_outliers // 3
    n_collective = n_outliers - n_global - n_local

    # Global outliers
    X_global = rng.uniform(-10, 10, (n_global, n_features))

    # Local outliers (between normal data)
    X_local = X_normal[rng.choice(n_normal, n_local)] + rng.randn(n_local, n_features) * 5

    # Collective outliers
    outlier_center = rng.randn(n_features) * 8
    X_collective = outlier_center + rng.randn(n_collective, n_features) * 0.5

    # Combine all outliers
    X_outliers = np.vstack([X_global, X_local, X_collective])

    # Combine with normal data
    X = np.vstack([X_normal, X_outliers])
    y = np.hstack([np.zeros(n_normal), np.ones(n_outliers)])

    # Shuffle
    idx = rng.permutation(n_samples)

    return X[idx], y[idx]


def generate_high_dimensional_outliers(
    n_samples: int = 1000,
    n_features: int = 100,
    contamination: float = 0.1,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate high-dimensional dataset with sparse outliers.

    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_features : int
        Number of features (should be high, e.g., 100+)
    contamination : float
        Proportion of outliers
    random_state : int, optional
        Random seed

    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels (0=normal, 1=outlier)
    """
    rng = np.random.RandomState(random_state)

    n_outliers = int(n_samples * contamination)
    n_normal = n_samples - n_outliers

    # Generate sparse normal data
    X_normal = rng.randn(n_normal, n_features) * 0.1
    # Add structure to a few dimensions
    X_normal[:, :5] = rng.randn(n_normal, 5)

    # Generate outliers (anomalous in many dimensions)
    X_outliers = rng.randn(n_outliers, n_features) * 0.1
    # Make outliers anomalous in random subset of dimensions
    for i in range(n_outliers):
        anomalous_dims = rng.choice(n_features, size=n_features // 5, replace=False)
        X_outliers[i, anomalous_dims] += rng.randn(len(anomalous_dims)) * 5

    # Combine
    X = np.vstack([X_normal, X_outliers])
    y = np.hstack([np.zeros(n_normal), np.ones(n_outliers)])

    # Shuffle
    idx = rng.permutation(n_samples)

    return X[idx], y[idx]


# Dataset catalog
DATASET_GENERATORS: Dict[str, Callable[..., Tuple[np.ndarray, np.ndarray]]] = {
    "global": generate_global_outliers,
    "local": generate_local_outliers,
    "collective": generate_collective_outliers,
    "contextual": generate_contextual_outliers,
    "mixed": generate_mixed_outliers,
    "high_dimensional": generate_high_dimensional_outliers,
}


def get_benchmark_datasets() -> Dict[str, Dict]:
    """
    Get catalog of benchmark datasets with configurations.

    Returns
    -------
    datasets : Dict[str, Dict]
        Dictionary mapping dataset names to configurations
    """
    datasets = {
        # Small datasets (for quick testing)
        "small_global": {
            "generator": "global",
            "n_samples": 500,
            "n_features": 10,
            "contamination": 0.1,
            "description": "Small dataset with global outliers",
        },
        "small_local": {
            "generator": "local",
            "n_samples": 500,
            "n_features": 10,
            "contamination": 0.1,
            "n_clusters": 3,
            "description": "Small dataset with local outliers",
        },
        # Medium datasets
        "medium_global": {
            "generator": "global",
            "n_samples": 2000,
            "n_features": 20,
            "contamination": 0.05,
            "description": "Medium dataset with global outliers",
        },
        "medium_mixed": {
            "generator": "mixed",
            "n_samples": 2000,
            "n_features": 20,
            "contamination": 0.05,
            "description": "Medium dataset with mixed outlier types",
        },
        # Large datasets
        "large_global": {
            "generator": "global",
            "n_samples": 10000,
            "n_features": 30,
            "contamination": 0.02,
            "description": "Large dataset with global outliers",
        },
        "large_mixed": {
            "generator": "mixed",
            "n_samples": 10000,
            "n_features": 30,
            "contamination": 0.02,
            "description": "Large dataset with mixed outliers",
        },
        # High-dimensional datasets
        "high_dim_small": {
            "generator": "high_dimensional",
            "n_samples": 1000,
            "n_features": 100,
            "contamination": 0.1,
            "description": "High-dimensional sparse outliers",
        },
        "high_dim_large": {
            "generator": "high_dimensional",
            "n_samples": 5000,
            "n_features": 200,
            "contamination": 0.05,
            "description": "Large high-dimensional sparse outliers",
        },
        # Different contamination levels
        "low_contamination": {
            "generator": "global",
            "n_samples": 2000,
            "n_features": 20,
            "contamination": 0.01,
            "description": "Very low contamination (1%)",
        },
        "high_contamination": {
            "generator": "global",
            "n_samples": 2000,
            "n_features": 20,
            "contamination": 0.20,
            "description": "High contamination (20%)",
        },
        # Specific outlier types
        "collective_only": {
            "generator": "collective",
            "n_samples": 2000,
            "n_features": 20,
            "contamination": 0.05,
            "description": "Only collective outliers",
        },
        "contextual_only": {
            "generator": "contextual",
            "n_samples": 2000,
            "n_features": 20,
            "contamination": 0.05,
            "description": "Only contextual outliers",
        },
    }

    return datasets


def generate_dataset(
    dataset_config: Dict, random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset from configuration.

    Parameters
    ----------
    dataset_config : Dict
        Dataset configuration with 'generator' and parameters
    random_state : int, optional
        Random seed

    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels (0=normal, 1=outlier)
    """
    config = dataset_config.copy()
    generator_name = config.pop("generator")
    config.pop("description", None)  # Remove description

    generator = DATASET_GENERATORS[generator_name]

    return generator(**config, random_state=random_state)


def generate_scalability_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate datasets for scalability testing.

    Returns
    -------
    datasets : Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary of datasets with varying sizes
    """
    sample_sizes = [100, 500, 1000, 2000, 5000, 10000]
    feature_sizes = [10, 20, 50]

    datasets = {}

    for n_samples in sample_sizes:
        for n_features in feature_sizes:
            name = f"n{n_samples}_d{n_features}"
            X, y = generate_global_outliers(
                n_samples=n_samples, n_features=n_features, contamination=0.05, random_state=42
            )
            datasets[name] = (X, y)

    return datasets
