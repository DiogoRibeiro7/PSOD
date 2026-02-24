"""
Comprehensive unit tests for PSOD outlier detection core module.
Tests all core functionality, edge cases, and sklearn compatibility.
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psod import PSOD

# Try importing optional encoders
try:
    from category_encoders import OneHotEncoder, TargetEncoder

    HAS_ENCODERS = True
except ImportError:
    HAS_ENCODERS = False


# ============================================================================
# INITIALIZATION AND PARAMETER VALIDATION TESTS
# ============================================================================


@pytest.mark.unit
class TestPSODInitialization:
    """Tests for PSOD initialization and parameter validation."""

    def test_default_initialization(self):
        """Test PSOD initialization with default parameters."""
        psod = PSOD()
        assert psod.n_jobs == -1
        assert psod.min_cols_chosen == 0.5
        assert psod.max_cols_chosen == 1.0
        assert psod.correlation_threshold == 0.05
        assert psod.sample_frac == 1.0
        assert psod.flag_outlier_on == "both ends"
        assert psod.contamination == 0.1
        assert psod.missing_value_strategy == "mean"

    def test_custom_initialization(self):
        """Test PSOD initialization with custom parameters."""
        psod = PSOD(
            n_jobs=2,
            min_cols_chosen=0.3,
            max_cols_chosen=0.8,
            random_seed=123,
            contamination=0.1,
            missing_value_strategy="median",
        )
        assert psod.n_jobs == 2
        assert psod.min_cols_chosen == 0.3
        assert psod.max_cols_chosen == 0.8
        assert psod.random_seed == 123
        assert psod.contamination == 0.1
        assert psod.missing_value_strategy == "median"

    def test_invalid_min_cols_chosen(self):
        """Test invalid min_cols_chosen raises ValueError."""
        with pytest.raises(ValueError, match="min_cols_chosen must be between 0 and 1"):
            PSOD(min_cols_chosen=1.5)

        with pytest.raises(ValueError, match="min_cols_chosen must be between 0 and 1"):
            PSOD(min_cols_chosen=-0.1)

    def test_invalid_max_cols_chosen(self):
        """Test invalid max_cols_chosen raises ValueError."""
        with pytest.raises(ValueError, match="max_cols_chosen must be between 0 and 1"):
            PSOD(max_cols_chosen=1.5)

        with pytest.raises(ValueError, match="max_cols_chosen must be between 0 and 1"):
            PSOD(max_cols_chosen=-0.1)

    def test_min_greater_than_max(self):
        """Test min > max raises ValueError."""
        with pytest.raises(
            ValueError, match="min_cols_chosen cannot be higher than max_cols_chosen"
        ):
            PSOD(min_cols_chosen=0.8, max_cols_chosen=0.5)

    def test_invalid_flag_outlier_on(self):
        """Test invalid flag_outlier_on raises ValueError."""
        with pytest.raises(ValueError, match="flag_outlier_on must be one of"):
            PSOD(flag_outlier_on="invalid")

    def test_invalid_contamination(self):
        """Test invalid contamination parameter."""
        with pytest.raises(ValueError):
            PSOD(contamination=-0.1)

        with pytest.raises(ValueError):
            PSOD(contamination=1.5)

    def test_invalid_missing_value_strategy(self):
        """Test invalid missing value strategy."""
        with pytest.raises(ValueError):
            PSOD(missing_value_strategy="invalid_strategy")

    def test_invalid_transform_algorithm(self):
        """Test invalid transform_algorithm raises ValueError."""
        with pytest.raises(ValueError, match="transform_algorithm must be one of"):
            PSOD(transform_algorithm="invalid_transform")

    def test_init_logs_parameters(self, caplog):
        """Test that initialization logs parameters."""
        with caplog.at_level(logging.INFO):
            PSOD()

        assert any("Initialized PSOD with parameters" in rec.message for rec in caplog.records)

    def test_invalid_cat_columns_type(self):
        """Test invalid cat_columns type."""
        with pytest.raises(ValueError, match="cat_columns must be a list"):
            PSOD(cat_columns="not-a-list")

    def test_invalid_cat_columns_empty(self):
        """Test empty cat_columns list."""
        with pytest.raises(ValueError, match="cat_columns must be a non-empty list"):
            PSOD(cat_columns=[])

    def test_invalid_cat_columns_non_string(self):
        """Test cat_columns with non-string entries."""
        with pytest.raises(ValueError, match="cat_columns must contain only strings"):
            PSOD(cat_columns=["A", 1])

    def test_invalid_cat_columns_duplicates(self):
        """Test cat_columns with duplicate entries."""
        with pytest.raises(ValueError, match="cat_columns must not contain duplicate names"):
            PSOD(cat_columns=["A", "A"])

    def test_invalid_n_jobs_zero(self):
        """Test n_jobs cannot be zero."""
        with pytest.raises(ValueError, match="n_jobs cannot be 0"):
            PSOD(n_jobs=0)


@pytest.mark.unit
class TestPSODCategoricalValidation:
    """Tests for categorical column validation."""

    def test_numeric_cat_columns_rejected(self, sample_numeric_data):
        """Test that numeric columns cannot be declared categorical."""
        psod = PSOD(cat_columns=["feature_1"], random_seed=42)

        with pytest.raises(ValueError, match="Categorical columns must be non-numeric"):
            psod.fit_predict(sample_numeric_data)


# ============================================================================
# CORE FUNCTIONALITY TESTS
# ============================================================================


@pytest.mark.unit
class TestPSODFitPredict:
    """Tests for PSOD fit_predict functionality."""

    def test_fit_predict_basic(self, sample_numeric_data):
        """Test basic fit_predict functionality."""
        psod = PSOD(random_seed=42)
        scores = psod.fit_predict(sample_numeric_data)

        assert isinstance(scores, (pd.Series, np.ndarray))
        assert len(scores) == len(sample_numeric_data)
        assert all(scores >= 0)

        # Model should be marked as fitted
        assert psod._is_fitted

    def test_fit_predict_return_class(self, sample_numeric_data):
        """Test fit_predict with return_class=True."""
        psod = PSOD(random_seed=42, contamination=0.1)
        labels = psod.fit_predict(sample_numeric_data, return_class=True)

        assert isinstance(labels, (pd.Series, np.ndarray))
        assert len(labels) == len(sample_numeric_data)
        assert set(np.unique(labels)).issubset({0, 1})

        # Should have approximately 10% outliers
        outlier_ratio = np.sum(labels) / len(labels)
        assert 0.05 <= outlier_ratio <= 0.15

    def test_fit_predict_with_contamination(self, sample_numeric_data):
        """Test automatic threshold with contamination parameter."""
        psod = PSOD(contamination=0.05, random_seed=42)
        labels = psod.fit_predict(sample_numeric_data, return_class=True)

        outlier_ratio = np.sum(labels) / len(labels)
        # Should be close to 5%
        assert abs(outlier_ratio - 0.05) < 0.03

    @pytest.mark.skipif(not HAS_ENCODERS, reason="category_encoders not installed")
    def test_fit_predict_with_categorical(self, sample_data_with_categorical):
        """Test fit_predict with categorical columns."""
        cat_cols = ["category_1", "category_2"]

        psod = PSOD(cat_columns=cat_cols, random_seed=42, cat_encoder=OneHotEncoder)
        scores = psod.fit_predict(sample_data_with_categorical)

        assert isinstance(scores, (pd.Series, np.ndarray))
        assert len(scores) == len(sample_data_with_categorical)

    def test_fit_predict_known_outliers(self, outlier_data):
        """Test if known outliers are detected."""
        psod = PSOD(stdevs_to_outlier=2.0, random_seed=42, contamination=0.05)

        # Get outlier classes
        outlier_classes = psod.fit_predict(outlier_data, return_class=True)

        # Last 5 points should be detected as outliers
        assert sum(outlier_classes[-5:]) >= 3  # At least 3 out of 5 outliers detected
        assert sum(outlier_classes[:-5]) <= 5  # Few false positives in normal data

    def test_fit_predict_reproducibility(self, sample_numeric_data):
        """Test that results are reproducible with same seed."""
        psod1 = PSOD(random_seed=42)
        scores1 = psod1.fit_predict(sample_numeric_data)

        psod2 = PSOD(random_seed=42)
        scores2 = psod2.fit_predict(sample_numeric_data)

        assert np.allclose(scores1, scores2)

    def test_fit_predict_small_dataset(self, small_dataset):
        """Test fit_predict with very small dataset."""
        psod = PSOD(random_seed=42, min_samples=3)
        scores = psod.fit_predict(small_dataset)

        assert len(scores) == len(small_dataset)

    def test_fit_predict_single_column(self, single_column_data):
        """Test with single column data."""
        psod = PSOD(min_cols_chosen=1.0, max_cols_chosen=1.0, random_seed=42)

        with pytest.warns(UserWarning):
            scores = psod.fit_predict(single_column_data)

        assert len(scores) == len(single_column_data)

    def test_fit_predict_time_series_data(self, time_series_data):
        """Test fit_predict with time series data (drop timestamp)."""
        psod = PSOD(random_seed=42)
        numeric_data = time_series_data.drop(columns=["timestamp"])
        scores = psod.fit_predict(numeric_data)

        assert len(scores) == len(numeric_data)

    def test_fit_predict_high_dimensional_data(self, high_dimensional_data):
        """Test fit_predict with high-dimensional data."""
        psod = PSOD(random_seed=42)
        scores = psod.fit_predict(high_dimensional_data)

        assert len(scores) == len(high_dimensional_data)

    def test_fit_predict_logs(self, sample_numeric_data, caplog):
        """Test that fit_predict logs start and completion."""
        psod = PSOD(random_seed=42)

        with caplog.at_level(logging.INFO):
            psod.fit_predict(sample_numeric_data)

        messages = [rec.message for rec in caplog.records]
        assert any("Starting PSOD fit_predict" in msg for msg in messages)
        assert any("Fitting completed" in msg for msg in messages)


@pytest.mark.unit
class TestPSODPredict:
    """Tests for PSOD predict functionality."""

    def test_predict_on_new_data(self, sample_numeric_data):
        """Test prediction on new data."""
        psod = PSOD(random_seed=42)

        # Fit on first 80 rows
        train_data = sample_numeric_data.iloc[:80]
        psod.fit_predict(train_data)

        # Predict on last 20 rows
        test_data = sample_numeric_data.iloc[80:]
        scores = psod.predict(test_data)

        assert isinstance(scores, (pd.Series, np.ndarray))
        assert len(scores) == len(test_data)

    def test_predict_without_fit(self, sample_numeric_data):
        """Test that predict raises error when not fitted."""
        psod = PSOD()

        with pytest.raises(ValueError, match="Model must be fitted"):
            psod.predict(sample_numeric_data)

    def test_predict_return_class(self, sample_numeric_data):
        """Test predict with return_class=True."""
        psod = PSOD(random_seed=42, contamination=0.1)
        psod.fit_predict(sample_numeric_data)

        labels = psod.predict(sample_numeric_data, return_class=True)

        assert isinstance(labels, (pd.Series, np.ndarray))
        assert set(np.unique(labels)).issubset({0, 1})

    def test_predict_consistency(self, sample_numeric_data):
        """Test that predict on training data gives same results."""
        psod = PSOD(random_seed=42)
        train_scores = psod.fit_predict(sample_numeric_data)

        predict_scores = psod.predict(sample_numeric_data)

        # Should be very similar (might have small differences)
        assert np.corrcoef(train_scores, predict_scores)[0, 1] > 0.95

    def test_predict_missing_feature_raises(self, sample_numeric_data):
        """Test predict raises when required feature is missing."""
        psod = PSOD(random_seed=42)
        psod.fit_predict(sample_numeric_data)

        test_data = sample_numeric_data.drop(columns=[sample_numeric_data.columns[0]])

        with pytest.raises(ValueError, match="Features missing from input"):
            psod.predict(test_data)

    def test_predict_extra_feature_allows(self, sample_numeric_data):
        """Test predict allows extra features but uses trained ones."""
        psod = PSOD(random_seed=42)
        psod.fit_predict(sample_numeric_data)

        test_data = sample_numeric_data.copy()
        test_data["extra_feature"] = np.random.randn(len(test_data))

        scores = psod.predict(test_data)
        assert len(scores) == len(test_data)

    def test_predict_ignores_non_numeric_columns(self, sample_numeric_data):
        """Test non-numeric columns are ignored when not in cat_columns."""
        data = sample_numeric_data.copy()
        data["timestamp"] = pd.date_range("2020-01-01", periods=len(data), freq="D")

        psod = PSOD(random_seed=42)
        psod.fit_predict(data)

        # Prediction should succeed even if timestamp is dropped
        scores = psod.predict(sample_numeric_data)
        assert len(scores) == len(sample_numeric_data)

    def test_predict_logs(self, sample_numeric_data, caplog):
        """Test that predict logs start and completion."""
        psod = PSOD(random_seed=42)
        psod.fit_predict(sample_numeric_data)

        with caplog.at_level(logging.INFO):
            psod.predict(sample_numeric_data)

        messages = [rec.message for rec in caplog.records]
        assert any("Starting PSOD predict" in msg for msg in messages)
        assert any("Prediction completed" in msg for msg in messages)

    def test_predict_requires_datetime_when_converted(self, sample_numeric_data):
        """Test datetime columns become required when auto-converted."""
        data = sample_numeric_data.copy()
        data["timestamp"] = pd.date_range("2020-01-01", periods=len(data), freq="D")

        psod = PSOD(random_seed=42, auto_convert_datetime=True)
        psod.fit_predict(data)

        with pytest.raises(ValueError, match="Features missing from input"):
            psod.predict(sample_numeric_data)

    def test_predict_ignores_datetime_when_disabled(self, sample_numeric_data):
        """Test datetime columns are ignored when auto_convert_datetime is False."""
        data = sample_numeric_data.copy()
        data["timestamp"] = pd.date_range("2020-01-01", periods=len(data), freq="D")

        psod = PSOD(random_seed=42, auto_convert_datetime=False)
        psod.fit_predict(data)

        scores = psod.predict(sample_numeric_data)
        assert len(scores) == len(sample_numeric_data)


@pytest.mark.unit
class TestPSODScoreSamples:
    """Tests for score_samples method (sklearn compatibility)."""

    def test_score_samples_basic(self, sample_numeric_data, fitted_psod_model):
        """Test basic score_samples functionality."""
        scores = fitted_psod_model.score_samples(sample_numeric_data)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(sample_numeric_data)
        assert all(scores >= 0)

    def test_score_samples_without_fit(self, sample_numeric_data):
        """Test score_samples without fitting."""
        psod = PSOD()

        with pytest.raises(ValueError):
            psod.score_samples(sample_numeric_data)

    def test_score_samples_equals_predict(self, sample_numeric_data, fitted_psod_model):
        """Test that score_samples equals predict with return_class=False."""
        scores1 = fitted_psod_model.score_samples(sample_numeric_data)
        scores2 = fitted_psod_model.predict(sample_numeric_data, return_class=False)

        assert np.allclose(scores1, scores2)


# ============================================================================
# SKLEARN COMPATIBILITY TESTS
# ============================================================================


@pytest.mark.unit
class TestPSODSklearnCompatibility:
    """Tests for sklearn BaseEstimator compatibility."""

    def test_get_params(self):
        """Test get_params method."""
        psod = PSOD(n_jobs=2, min_cols_chosen=0.3, random_seed=42, contamination=0.1)

        params = psod.get_params()

        assert isinstance(params, dict)
        assert params["n_jobs"] == 2
        assert params["min_cols_chosen"] == 0.3
        assert params["random_seed"] == 42
        assert params["contamination"] == 0.1
        assert "missing_value_strategy" in params

    def test_set_params(self):
        """Test set_params method."""
        psod = PSOD()

        psod.set_params(n_jobs=4, random_seed=123)

        assert psod.n_jobs == 4
        assert psod.random_seed == 123

    def test_get_set_params_roundtrip(self):
        """Test get_params and set_params roundtrip."""
        psod1 = PSOD(n_jobs=2, contamination=0.15, random_seed=42)
        params = psod1.get_params()

        psod2 = PSOD()
        psod2.set_params(**params)

        assert psod2.get_params() == params


# ============================================================================
# MODEL PERSISTENCE TESTS
# ============================================================================


@pytest.mark.unit
class TestPSODPersistence:
    """Tests for model save/load functionality."""

    def test_save_load_model(self, sample_numeric_data, tmp_model_path):
        """Test model persistence with save_model and load_model."""
        psod = PSOD(random_seed=42, contamination=0.1)
        original_scores = psod.fit_predict(sample_numeric_data)

        # Save model
        psod.save_model(str(tmp_model_path))
        assert tmp_model_path.with_suffix(".pkl").exists()

        # Load model
        loaded_psod = PSOD.load_model(str(tmp_model_path))

        # Test predictions are the same
        loaded_scores = loaded_psod.predict(sample_numeric_data)
        assert np.allclose(original_scores, loaded_scores)

    def test_load_preserves_attributes(self, sample_numeric_data, tmp_model_path):
        """Test that loading preserves all model attributes."""
        psod = PSOD(random_seed=42, contamination=0.1, n_jobs=2, min_cols_chosen=0.3)
        psod.fit_predict(sample_numeric_data)

        psod.save_model(str(tmp_model_path))
        loaded_psod = PSOD.load_model(str(tmp_model_path))

        assert loaded_psod.random_seed == 42
        assert loaded_psod.contamination == 0.1
        assert loaded_psod.n_jobs == 2
        assert loaded_psod.min_cols_chosen == 0.3
        assert loaded_psod._is_fitted

    def test_pickle_compatibility(self, sample_numeric_data, tmp_path):
        """Test that model can be pickled directly."""
        psod = PSOD(random_seed=42)
        psod.fit_predict(sample_numeric_data)

        pickle_path = tmp_path / "psod.pkl"

        # Save with pickle
        with open(pickle_path, "wb") as f:
            pickle.dump(psod, f)

        # Load with pickle
        with open(pickle_path, "rb") as f:
            loaded_psod = pickle.load(f)

        # Test functionality
        scores = loaded_psod.predict(sample_numeric_data)
        assert len(scores) == len(sample_numeric_data)

    def test_save_load_logs(self, sample_numeric_data, tmp_model_path, caplog):
        """Test that save/load log their operations."""
        psod = PSOD(random_seed=42)
        psod.fit_predict(sample_numeric_data)

        with caplog.at_level(logging.INFO):
            psod.save_model(str(tmp_model_path))
            PSOD.load_model(str(tmp_model_path))

        messages = [rec.message for rec in caplog.records]
        assert any("Saving PSOD model" in msg for msg in messages)
        assert any("Loading PSOD model" in msg for msg in messages)


# ============================================================================
# FEATURE IMPORTANCE TESTS
# ============================================================================


@pytest.mark.unit
class TestPSODFeatureImportance:
    """Tests for feature importance calculation."""

    def test_get_feature_importance(self, sample_numeric_data):
        """Test feature importance calculation."""
        psod = PSOD(random_seed=42)
        psod.fit_predict(sample_numeric_data)

        importance = psod.get_feature_importance()

        assert importance is not None
        assert isinstance(importance, (pd.DataFrame, pd.Series, dict))

    def test_feature_importance_without_fit(self):
        """Test feature importance before fitting."""
        psod = PSOD()

        importance = psod.get_feature_importance()

        # Should return None or empty
        assert importance is None or len(importance) == 0

    def test_feature_importance_methods(self, sample_numeric_data):
        """Test different feature importance calculation methods."""
        psod = PSOD(random_seed=42)
        psod.fit_predict(sample_numeric_data)

        # Test correlation-based importance
        importance_corr = psod.get_feature_importance(method="correlation")
        assert importance_corr is not None

        # Test mutual information importance (if implemented)
        try:
            importance_mi = psod.get_feature_importance(method="mutual_info")
            assert importance_mi is not None
        except (ValueError, NotImplementedError):
            pass  # Method might not be implemented


# ============================================================================
# EXPLAIN OUTLIER TESTS
# ============================================================================


@pytest.mark.unit
class TestPSODExplainOutlier:
    """Tests for explain_outlier interpretability method."""

    def test_explain_outlier_basic(self, sample_numeric_data, fitted_psod_model):
        """Test basic outlier explanation."""
        explanation = fitted_psod_model.explain_outlier(sample_numeric_data, sample_idx=0)

        assert explanation is not None
        assert isinstance(explanation, (dict, pd.DataFrame, pd.Series))

    def test_explain_outlier_topk(self, sample_numeric_data, fitted_psod_model):
        """Test explanation with top-k features."""
        explanation = fitted_psod_model.explain_outlier(sample_numeric_data, sample_idx=0, top_k=3)

        assert explanation is not None

    def test_explain_outlier_invalid_index(self, sample_numeric_data, fitted_psod_model):
        """Test explain_outlier with invalid index."""
        with pytest.raises((IndexError, ValueError)):
            fitted_psod_model.explain_outlier(sample_numeric_data, sample_idx=10000)

    def test_explain_outlier_without_fit(self, sample_numeric_data):
        """Test explain_outlier without fitting."""
        psod = PSOD()

        with pytest.raises(ValueError):
            psod.explain_outlier(sample_numeric_data, sample_idx=0)


# ============================================================================
# MISSING VALUE HANDLING TESTS
# ============================================================================


@pytest.mark.unit
class TestPSODMissingValues:
    """Tests for missing value handling."""

    def test_missing_values_mean_strategy(self, missing_value_data):
        """Test mean imputation strategy."""
        psod = PSOD(missing_value_strategy="mean", random_seed=42)
        scores = psod.fit_predict(missing_value_data)

        assert len(scores) == len(missing_value_data)
        assert all(np.isfinite(scores))

    def test_missing_values_median_strategy(self, missing_value_data):
        """Test median imputation strategy."""
        psod = PSOD(missing_value_strategy="median", random_seed=42)
        scores = psod.fit_predict(missing_value_data)

        assert len(scores) == len(missing_value_data)
        assert all(np.isfinite(scores))

    def test_missing_values_mode_strategy(self, missing_value_data):
        """Test mode imputation strategy."""
        psod = PSOD(missing_value_strategy="mode", random_seed=42)
        scores = psod.fit_predict(missing_value_data)

        assert len(scores) == len(missing_value_data)

    def test_missing_values_knn_strategy(self, missing_value_data):
        """Test KNN imputation strategy."""
        try:
            psod = PSOD(missing_value_strategy="knn", random_seed=42)
            scores = psod.fit_predict(missing_value_data)

            assert len(scores) == len(missing_value_data)
        except (ValueError, ImportError):
            pytest.skip("KNN imputation not available")

    def test_missing_values_drop_strategy(self, missing_value_data):
        """Test drop strategy for missing values."""
        try:
            psod = PSOD(missing_value_strategy="drop", random_seed=42)
            scores = psod.fit_predict(missing_value_data)

            # Should have fewer samples after dropping
            assert len(scores) <= len(missing_value_data)
        except (ValueError, NotImplementedError):
            pytest.skip("Drop strategy not implemented")

    def test_missing_values_inf_are_imputed_mean(self):
        """Test that inf values are treated as missing and imputed."""
        data = pd.DataFrame(
            {
                "A": [1.0, 2.0, np.inf, 4.0, 5.0],
                "B": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        psod = PSOD(missing_value_strategy="mean", random_seed=42)
        scores = psod.fit_predict(data)

        assert len(scores) == len(data)
        assert all(np.isfinite(scores))

    def test_missing_values_imputer_persists(self, missing_value_data):
        """Test that imputer fitted during training is used in prediction."""
        psod = PSOD(missing_value_strategy="mean", random_seed=42)
        psod.fit_predict(missing_value_data)

        assert psod.imputer_ is not None

        # Introduce missing values in new data and ensure prediction works
        test_data = missing_value_data.copy()
        test_data.iloc[0, 0] = np.nan
        scores = psod.predict(test_data)

        assert len(scores) == len(test_data)

    def test_predict_inf_raises_without_imputation(self, sample_numeric_data):
        """Test predict raises on inf when missing_value_strategy=None."""
        psod = PSOD(missing_value_strategy=None, random_seed=42)
        psod.fit_predict(sample_numeric_data)

        test_data = sample_numeric_data.copy()
        test_data.iloc[0, 0] = np.inf

        with pytest.raises(ValueError, match="infinite values"):
            psod.predict(test_data)


# ============================================================================
# TRANSFORMATION ALGORITHM TESTS
# ============================================================================


@pytest.mark.unit
class TestPSODTransformations:
    """Tests for different transformation algorithms."""

    def test_logarithmic_transform(self, sample_numeric_data):
        """Test logarithmic transformation."""
        psod = PSOD(transform_algorithm="logarithmic", random_seed=42)
        scores = psod.fit_predict(sample_numeric_data)

        assert len(scores) == len(sample_numeric_data)

    def test_yeo_johnson_transform(self, sample_numeric_data):
        """Test Yeo-Johnson transformation."""
        psod = PSOD(transform_algorithm="yeo-johnson", random_seed=42)
        scores = psod.fit_predict(sample_numeric_data)

        assert len(scores) == len(sample_numeric_data)

    def test_no_transform(self, sample_numeric_data):
        """Test with no transformation."""
        psod = PSOD(transform_algorithm="none", random_seed=42)
        scores = psod.fit_predict(sample_numeric_data)

        assert len(scores) == len(sample_numeric_data)

    def test_transform_with_negative_values(self):
        """Test transformation with negative values."""
        data = pd.DataFrame(
            {
                "A": np.random.randn(100) - 5,  # Negative values
                "B": np.random.randn(100),
                "C": np.random.randn(100) + 5,
            }
        )

        psod = PSOD(transform_algorithm="logarithmic", random_seed=42)
        scores = psod.fit_predict(data)

        assert len(scores) == len(data)


# ============================================================================
# BASE LEARNER TESTS
# ============================================================================


@pytest.mark.unit
class TestPSODBaseLearners:
    """Tests with different base learners."""

    def test_linear_regression_learner(self, sample_numeric_data):
        """Test with LinearRegression (default)."""
        psod = PSOD(base_learner=LinearRegression, random_seed=42)
        scores = psod.fit_predict(sample_numeric_data)

        assert len(scores) == len(sample_numeric_data)

    def test_ridge_regression_learner(self, sample_numeric_data):
        """Test with Ridge regression."""
        psod = PSOD(base_learner=Ridge, random_seed=42)
        scores = psod.fit_predict(sample_numeric_data)

        assert len(scores) == len(sample_numeric_data)

    def test_random_forest_learner(self, sample_numeric_data):
        """Test with RandomForest regression."""
        psod = PSOD(base_learner=RandomForestRegressor, random_seed=42)
        scores = psod.fit_predict(sample_numeric_data)

        assert len(scores) == len(sample_numeric_data)

    def test_custom_learner_params(self, sample_numeric_data):
        """Test passing parameters to base learner."""
        psod = PSOD(base_learner=Ridge, learner_kwargs={"alpha": 0.5}, random_seed=42)
        scores = psod.fit_predict(sample_numeric_data)

        assert len(scores) == len(sample_numeric_data)


# ============================================================================
# FLAG OUTLIER OPTIONS TESTS
# ============================================================================


@pytest.mark.unit
class TestPSODFlagOutlierOptions:
    """Tests for different outlier flagging options."""

    def test_flag_both_ends(self, outlier_data):
        """Test flagging outliers on both ends."""
        psod = PSOD(flag_outlier_on="both ends", random_seed=42)
        outlier_classes = psod.fit_predict(outlier_data, return_class=True)

        assert len(outlier_classes) == len(outlier_data)

    def test_flag_high_end(self, outlier_data):
        """Test flagging only high-end outliers."""
        psod = PSOD(flag_outlier_on="high end", random_seed=42)
        outlier_classes = psod.fit_predict(outlier_data, return_class=True)

        assert len(outlier_classes) == len(outlier_data)

    def test_flag_low_end(self, outlier_data):
        """Test flagging only low-end outliers."""
        psod = PSOD(flag_outlier_on="low end", random_seed=42)
        outlier_classes = psod.fit_predict(outlier_data, return_class=True)

        assert len(outlier_classes) == len(outlier_data)


# ============================================================================
# PARALLEL PROCESSING TESTS
# ============================================================================


@pytest.mark.unit
class TestPSODParallelProcessing:
    """Tests for parallel processing functionality."""

    def test_serial_processing(self, sample_numeric_data):
        """Test with serial processing (n_jobs=1)."""
        psod = PSOD(n_jobs=1, random_seed=42)
        scores = psod.fit_predict(sample_numeric_data)

        assert len(scores) == len(sample_numeric_data)

    def test_parallel_processing(self, sample_numeric_data):
        """Test with parallel processing (n_jobs=-1)."""
        psod = PSOD(n_jobs=-1, random_seed=42)
        scores = psod.fit_predict(sample_numeric_data)

        assert len(scores) == len(sample_numeric_data)

    def test_parallel_vs_serial_consistency(self, sample_numeric_data):
        """Test that parallel and serial give similar results."""
        psod_serial = PSOD(n_jobs=1, random_seed=42)
        scores_serial = psod_serial.fit_predict(sample_numeric_data)

        psod_parallel = PSOD(n_jobs=-1, random_seed=42)
        scores_parallel = psod_parallel.fit_predict(sample_numeric_data)

        # Should be very similar (might have small differences)
        assert len(scores_serial) == len(scores_parallel)
        assert np.corrcoef(scores_serial, scores_parallel)[0, 1] > 0.95


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


@pytest.mark.unit
class TestPSODEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        empty_df = pd.DataFrame()
        psod = PSOD()

        with pytest.raises((ValueError, IndexError)):
            psod.fit_predict(empty_df)

        # Model should remain unfitted after failed fit
        assert not psod._is_fitted

    def test_single_sample(self):
        """Test with single sample."""
        single_sample = pd.DataFrame({"A": [1.0], "B": [2.0]})
        psod = PSOD()

        # Should handle gracefully or raise informative error
        try:
            scores = psod.fit_predict(single_sample)
            assert len(scores) == 1
        except ValueError:
            pass  # Expected

    def test_all_same_values(self):
        """Test with all identical values."""
        constant_data = pd.DataFrame({"A": np.ones(100), "B": np.ones(100), "C": np.ones(100)})
        psod = PSOD()

        scores = psod.fit_predict(constant_data)
        assert len(scores) == len(constant_data)

    def test_high_correlation_features(self, correlated_features_data):
        """Test with highly correlated features."""
        psod = PSOD(correlation_threshold=0.9, random_seed=42)
        scores = psod.fit_predict(correlated_features_data)

        assert len(scores) == len(correlated_features_data)

    def test_correlation_threshold_effect(self):
        """Test correlation threshold filters chosen columns."""
        np.random.seed(42)
        n = 200
        feature_1 = np.random.randn(n)
        feature_2 = feature_1 * 2 + np.random.randn(n) * 0.001  # Highly correlated
        feature_3 = np.random.randn(n)  # Uncorrelated

        data = pd.DataFrame(
            {
                "feature_1": feature_1,
                "feature_2": feature_2,
                "feature_3": feature_3,
            }
        )

        psod = PSOD(
            correlation_threshold=0.9,
            min_cols_chosen=1.0,
            max_cols_chosen=1.0,
            random_seed=42,
        )
        psod.fit_predict(data)

        chosen = psod.chosen_columns.get("feature_1", [])
        assert set(chosen) == {"feature_2"}

    def test_inf_values(self):
        """Test with infinite values."""
        data_with_inf = pd.DataFrame({"A": [1, 2, 3, np.inf, 5], "B": [1, 2, 3, 4, 5]})
        psod = PSOD()

        scores = psod.fit_predict(data_with_inf)
        assert len(scores) == len(data_with_inf)

    def test_duplicate_columns(self):
        """Test with duplicate column names."""
        data = pd.DataFrame(
            np.random.randn(10, 2),
            columns=["A", "A"],
        )
        psod = PSOD()

        with pytest.raises(ValueError, match="duplicate column names"):
            psod.fit_predict(data)


# ============================================================================
# CATEGORICAL ENCODER TESTS
# ============================================================================


@pytest.mark.skipif(not HAS_ENCODERS, reason="category_encoders not installed")
@pytest.mark.unit
class TestPSODCategoricalEncoders:
    """Tests for different categorical encoding options."""

    def test_target_encoder(self, sample_data_with_categorical):
        """Test with TargetEncoder."""
        cat_cols = ["category_1", "category_2"]

        psod = PSOD(cat_columns=cat_cols, cat_encoder=TargetEncoder, random_seed=42)
        scores = psod.fit_predict(sample_data_with_categorical)

        assert len(scores) == len(sample_data_with_categorical)

    def test_onehot_encoder(self, sample_data_with_categorical):
        """Test with OneHotEncoder."""
        cat_cols = ["category_1", "category_2"]

        psod = PSOD(cat_columns=cat_cols, cat_encoder=OneHotEncoder, random_seed=42)
        scores = psod.fit_predict(sample_data_with_categorical)

        assert len(scores) == len(sample_data_with_categorical)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
class TestPSODIntegration:
    """Integration tests for complete PSOD workflows."""

    def test_full_pipeline(self, sample_numeric_data, tmp_model_path):
        """Test complete outlier detection pipeline."""
        # 1. Initialize
        psod = PSOD(random_seed=42, contamination=0.1)

        # 2. Fit and predict
        scores = psod.fit_predict(sample_numeric_data)
        assert len(scores) == len(sample_numeric_data)

        # 3. Get feature importance
        importance = psod.get_feature_importance()
        assert importance is not None

        # 4. Explain outlier
        explanation = psod.explain_outlier(sample_numeric_data, sample_idx=0)
        assert explanation is not None

        # 5. Save model
        psod.save_model(str(tmp_model_path))
        # save_model adds .pkl extension
        assert tmp_model_path.with_suffix(".pkl").exists()

        # 6. Load model
        loaded_psod = PSOD.load_model(str(tmp_model_path))

        # 7. Predict with loaded model
        new_scores = loaded_psod.predict(sample_numeric_data)
        assert np.allclose(scores, new_scores)

    def test_cross_validation_workflow(self, sample_numeric_data):
        """Test PSOD in cross-validation scenario."""
        from sklearn.model_selection import train_test_split

        # Split data
        train_data, test_data = train_test_split(
            sample_numeric_data, test_size=0.2, random_state=42
        )

        # Fit on train
        psod = PSOD(random_seed=42, contamination=0.1)
        train_scores = psod.fit_predict(train_data)

        # Predict on test
        test_scores = psod.predict(test_data)

        assert len(train_scores) == len(train_data)
        assert len(test_scores) == len(test_data)

    def test_complete_workflow_with_categorical(self, sample_data_with_categorical):
        """Test complete workflow with categorical features."""
        pytest.importorskip("category_encoders")

        cat_cols = ["category_1", "category_2"]

        # Fit
        psod = PSOD(
            cat_columns=cat_cols, cat_encoder=OneHotEncoder, random_seed=42, contamination=0.1
        )
        scores = psod.fit_predict(sample_data_with_categorical)

        # Get predictions
        labels = psod.predict(sample_data_with_categorical, return_class=True)

        assert len(scores) == len(labels) == len(sample_data_with_categorical)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


@pytest.mark.performance
@pytest.mark.slow
class TestPSODPerformance:
    """Performance tests for PSOD."""

    def test_large_dataset_performance(self):
        """Test performance on large datasets."""
        np.random.seed(42)
        large_data = pd.DataFrame({f"feature_{i}": np.random.randn(10000) for i in range(20)})

        psod = PSOD(n_jobs=-1, random_seed=42)
        scores = psod.fit_predict(large_data)

        assert len(scores) == len(large_data)

    def test_high_dimensional_performance(self, high_dimensional_data):
        """Test performance on high-dimensional data."""
        psod = PSOD(n_jobs=-1, random_seed=42)
        scores = psod.fit_predict(high_dimensional_data)

        assert len(scores) == len(high_dimensional_data)

    def test_parallel_speedup(self):
        """Test that parallel processing is faster."""
        import time

        np.random.seed(42)
        data = pd.DataFrame({f"feature_{i}": np.random.randn(1000) for i in range(10)})

        # Serial
        psod_serial = PSOD(n_jobs=1, random_seed=42)
        start = time.time()
        psod_serial.fit_predict(data)
        serial_time = time.time() - start

        # Parallel
        psod_parallel = PSOD(n_jobs=-1, random_seed=42)
        start = time.time()
        psod_parallel.fit_predict(data)
        parallel_time = time.time() - start

        # Parallel should be faster or similar (within 2x)
        # Note: Small datasets might not show speedup due to overhead
        assert parallel_time <= serial_time * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
