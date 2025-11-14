"""
Shared pytest fixtures for PSOD test suite.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psod import PSOD
from psod import utils


@pytest.fixture(scope="session")
def random_seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def sample_numeric_data(random_seed):
    """Basic numeric dataset for testing."""
    np.random.seed(random_seed)
    return pd.DataFrame(
        {
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100) * 2,
            "feature_3": np.random.uniform(-1, 1, 100),
            "feature_4": np.random.exponential(2, 100),
        }
    )


@pytest.fixture
def sample_data_with_categorical(random_seed):
    """Dataset with both numeric and categorical features."""
    np.random.seed(random_seed)
    return pd.DataFrame(
        {
            "numeric_1": np.random.randn(150),
            "numeric_2": np.random.randn(150),
            "numeric_3": np.random.randn(150),
            "category_1": np.random.choice(["A", "B", "C"], 150),
            "category_2": np.random.choice(["X", "Y", "Z"], 150),
        }
    )


@pytest.fixture
def outlier_data(random_seed):
    """Dataset with known outliers."""
    np.random.seed(random_seed)
    # 95 normal samples
    normal_data = np.random.randn(95, 5)

    # 5 clear outliers
    outliers = np.array(
        [
            [10, 10, 10, 10, 10],  # All features high
            [-10, -10, -10, -10, -10],  # All features low
            [0, 20, 0, 0, 0],  # One feature extreme
            [15, 0, 0, 15, 0],  # Two features high
            [0, 0, -15, 0, -15],  # Two features low
        ]
    )

    data = np.vstack([normal_data, outliers])
    return pd.DataFrame(data, columns=["X1", "X2", "X3", "X4", "X5"])


@pytest.fixture
def time_series_data(random_seed):
    """Time series dataset with temporal outliers."""
    np.random.seed(random_seed)
    n_samples = 200

    # Create datetime index
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")

    # Normal time series with trend and seasonality
    trend = np.linspace(0, 10, n_samples)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 30)
    noise = np.random.randn(n_samples) * 0.5

    values = trend + seasonal + noise

    # Add temporal outliers
    outlier_indices = [50, 75, 120, 160, 185]
    values[outlier_indices] += np.random.choice([-10, 10], len(outlier_indices))

    return pd.DataFrame(
        {
            "timestamp": dates,
            "value": values,
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
        }
    )


@pytest.fixture
def high_dimensional_data(random_seed):
    """High-dimensional sparse dataset."""
    np.random.seed(random_seed)
    n_samples = 100
    n_features = 50  # High dimensional

    # Create sparse data
    data = np.random.randn(n_samples, n_features) * 0.1

    # Add structure to a few dimensions
    data[:, 0] = np.random.randn(n_samples)
    data[:, 5] = np.random.randn(n_samples) * 2
    data[:, 10] = np.random.uniform(-1, 1, n_samples)

    # Add outliers in multiple dimensions
    data[-5:, :10] += 5

    columns = [f"dim_{i}" for i in range(n_features)]
    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def missing_value_data(random_seed):
    """Dataset with missing values."""
    np.random.seed(random_seed)
    data = pd.DataFrame(
        {
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
            "feature_4": np.random.randn(100),
        }
    )

    # Introduce missing values (about 10% missing)
    mask = np.random.rand(*data.shape) < 0.1
    data[mask] = np.nan

    return data


@pytest.fixture
def small_dataset(random_seed):
    """Very small dataset for edge case testing."""
    np.random.seed(random_seed)
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": [4.0, 5.0, 6.0],
        }
    )


@pytest.fixture
def single_column_data(random_seed):
    """Single column dataset."""
    np.random.seed(random_seed)
    return pd.DataFrame({"only_feature": np.random.randn(50)})


@pytest.fixture
def correlated_features_data(random_seed):
    """Dataset with highly correlated features."""
    np.random.seed(random_seed)
    n_samples = 100

    base_feature = np.random.randn(n_samples)

    return pd.DataFrame(
        {
            "feature_1": base_feature,
            "feature_2": base_feature * 2 + np.random.randn(n_samples) * 0.1,  # High correlation
            "feature_3": base_feature * -1
            + np.random.randn(n_samples) * 0.1,  # Negative correlation
            "feature_4": np.random.randn(n_samples),  # Independent
            "feature_5": base_feature * 0.5
            + np.random.randn(n_samples) * 0.5,  # Medium correlation
        }
    )


@pytest.fixture
def fitted_psod_model(sample_numeric_data, random_seed):
    """Pre-fitted PSOD model for testing predict()."""
    psod = PSOD(random_seed=random_seed, contamination=0.1)
    psod.fit_predict(sample_numeric_data)
    return psod


@pytest.fixture
def sample_outlier_scores(random_seed):
    """Sample outlier scores for testing utilities."""
    np.random.seed(random_seed)
    scores = np.random.randn(100)
    scores[-10:] += 3  # Make last 10 outliers
    return scores


@pytest.fixture
def sample_predictions(random_seed):
    """Sample binary predictions."""
    np.random.seed(random_seed)
    y_true = np.zeros(100)
    y_true[-10:] = 1  # 10 true outliers

    y_pred = np.zeros(100)
    y_pred[-12:] = 1  # 12 predicted outliers (2 false positives, 0 false negatives)

    return y_true.astype(int), y_pred.astype(int)


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for workflows")
    config.addinivalue_line("markers", "performance: Performance benchmark tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "visualization: Tests for visualization functions")
    config.addinivalue_line("markers", "requires_deps: Tests requiring optional dependencies")


# Pytest collection hooks
def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location/name."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark performance tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

        # Mark visualization tests
        if "visualization" in item.nodeid:
            item.add_marker(pytest.mark.visualization)
            item.add_marker(pytest.mark.requires_deps)


# Helper functions for tests
@pytest.fixture
def assert_valid_outlier_scores():
    """Helper to validate outlier scores."""

    def _validate(scores, expected_length=None):
        assert scores is not None
        assert isinstance(scores, (pd.Series, np.ndarray))
        if expected_length is not None:
            assert len(scores) == expected_length
        # Scores should be numeric
        assert np.all(np.isfinite(scores))
        return True

    return _validate


@pytest.fixture
def assert_valid_outlier_labels():
    """Helper to validate outlier labels."""

    def _validate(labels, expected_length=None):
        assert labels is not None
        assert isinstance(labels, (pd.Series, np.ndarray))
        if expected_length is not None:
            assert len(labels) == expected_length
        # Labels should be 0 or 1
        assert set(np.unique(labels)).issubset({0, 1})
        return True

    return _validate


@pytest.fixture
def tmp_model_path(tmp_path):
    """Temporary path for model saving."""
    return tmp_path / "test_model"
