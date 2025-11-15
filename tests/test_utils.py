"""
Tests for PSOD utility functions.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psod import utils


@pytest.mark.unit
class TestModelPersistence:
    """Tests for model save/load functionality."""

    def test_save_load_pickle(self, fitted_psod_model, tmp_model_path):
        """Test save and load with pickle format."""
        # Save model
        utils.save_model(fitted_psod_model, str(tmp_model_path), format="pickle")

        # Load model
        loaded_model = utils.load_model(str(tmp_model_path), format="pickle")

        # Verify model attributes
        assert loaded_model._is_fitted
        assert loaded_model.random_seed == fitted_psod_model.random_seed

    def test_save_load_joblib(self, fitted_psod_model, tmp_model_path):
        """Test save and load with joblib format."""
        # Save model
        utils.save_model(fitted_psod_model, str(tmp_model_path), format="joblib")

        # Load model
        loaded_model = utils.load_model(str(tmp_model_path), format="joblib")

        # Verify model
        assert loaded_model._is_fitted

    def test_save_invalid_format(self, fitted_psod_model, tmp_model_path):
        """Test save with invalid format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            utils.save_model(fitted_psod_model, str(tmp_model_path), format="invalid")

    def test_load_nonexistent_file(self):
        """Test load with non-existent file."""
        with pytest.raises(FileNotFoundError):
            utils.load_model("nonexistent_file.pkl", format="pickle")


@pytest.mark.unit
class TestDataValidation:
    """Tests for data validation utilities."""

    def test_validate_basic_dataframe(self, sample_numeric_data):
        """Test basic dataframe validation."""
        result = utils.validate_dataframe(sample_numeric_data, min_samples=10)

        assert result["is_valid"]
        assert len(result["errors"]) == 0
        assert result["info"]["n_samples"] == len(sample_numeric_data)
        assert result["info"]["n_features"] == len(sample_numeric_data.columns)

    def test_validate_insufficient_samples(self, small_dataset):
        """Test validation with insufficient samples."""
        result = utils.validate_dataframe(small_dataset, min_samples=10)

        assert not result["is_valid"]
        assert any("Insufficient samples" in error for error in result["errors"])

    def test_validate_missing_values(self, missing_value_data):
        """Test validation with missing values."""
        result = utils.validate_dataframe(missing_value_data, check_missing=True)

        assert "missing_values" in result["info"]
        # Should have warnings about missing values
        assert len(result["warnings"]) > 0 or len(result["recommendations"]) > 0

    def test_validate_with_constant_columns(self):
        """Test validation with constant columns."""
        data = pd.DataFrame(
            {
                "constant": [1] * 100,
                "variable": np.random.randn(100),
            }
        )

        result = utils.validate_dataframe(data)

        assert "constant_columns" in result["info"]
        assert "constant" in result["info"]["constant_columns"]

    def test_validate_with_duplicates(self):
        """Test validation with duplicate rows."""
        data = pd.DataFrame(
            {
                "A": [1, 2, 3, 1, 2],
                "B": [4, 5, 6, 4, 5],
            }
        )

        result = utils.validate_dataframe(data)

        assert "duplicate_rows" in result["info"]
        assert result["info"]["duplicate_rows"] == 2


@pytest.mark.unit
class TestMissingValueHandling:
    """Tests for missing value handling."""

    def test_handle_missing_drop(self, missing_value_data):
        """Test drop strategy."""
        result = utils.handle_missing_values(missing_value_data, strategy="drop")

        assert result.isnull().sum().sum() == 0
        assert len(result) < len(missing_value_data)

    def test_handle_missing_mean(self, missing_value_data):
        """Test mean imputation."""
        result = utils.handle_missing_values(missing_value_data, strategy="mean")

        assert result.isnull().sum().sum() == 0
        assert len(result) == len(missing_value_data)

    def test_handle_missing_median(self, missing_value_data):
        """Test median imputation."""
        result = utils.handle_missing_values(missing_value_data, strategy="median")

        assert result.isnull().sum().sum() == 0
        assert len(result) == len(missing_value_data)

    def test_handle_missing_mode(self, missing_value_data):
        """Test mode imputation."""
        result = utils.handle_missing_values(missing_value_data, strategy="mode")

        assert result.isnull().sum().sum() == 0
        assert len(result) == len(missing_value_data)

    def test_handle_missing_knn(self, missing_value_data):
        """Test KNN imputation."""
        result = utils.handle_missing_values(missing_value_data, strategy="knn")

        assert result.isnull().sum().sum() == 0
        assert len(result) == len(missing_value_data)

    def test_handle_missing_no_missing(self, sample_numeric_data):
        """Test handling when no missing values present."""
        result = utils.handle_missing_values(sample_numeric_data, strategy="mean")

        assert result.equals(sample_numeric_data)


@pytest.mark.unit
class TestScoreCalibration:
    """Tests for outlier score calibration."""

    def test_calibrate_basic(self, sample_outlier_scores):
        """Test basic score calibration."""
        calibrated = utils.calibrate_outlier_scores(sample_outlier_scores, contamination=0.1)

        assert len(calibrated) == len(sample_outlier_scores)
        assert np.min(calibrated) >= 0
        assert np.max(calibrated) <= 1

    def test_calibrate_invalid_contamination(self, sample_outlier_scores):
        """Test calibration with invalid contamination."""
        with pytest.raises(ValueError, match="Contamination must be between 0 and 1"):
            utils.calibrate_outlier_scores(sample_outlier_scores, contamination=1.5)

    def test_calibrate_preserves_order(self, sample_outlier_scores):
        """Test that calibration preserves rank order."""
        calibrated = utils.calibrate_outlier_scores(sample_outlier_scores, contamination=0.1)

        original_order = np.argsort(sample_outlier_scores)
        calibrated_order = np.argsort(calibrated)

        assert np.array_equal(original_order, calibrated_order)


@pytest.mark.unit
class TestScoreCombination:
    """Tests for combining multiple outlier scores."""

    @pytest.fixture
    def multiple_scores(self, random_seed):
        """Create multiple score arrays."""
        np.random.seed(random_seed)
        return [
            np.random.randn(100),
            np.random.randn(100),
            np.random.randn(100),
        ]

    def test_combine_average(self, multiple_scores):
        """Test average combination."""
        combined = utils.combine_outlier_scores(multiple_scores, method="average")

        assert len(combined) == len(multiple_scores[0])
        expected_avg = np.mean(multiple_scores, axis=0)
        assert np.allclose(combined, expected_avg)

    def test_combine_median(self, multiple_scores):
        """Test median combination."""
        combined = utils.combine_outlier_scores(multiple_scores, method="median")

        assert len(combined) == len(multiple_scores[0])

    def test_combine_maximum(self, multiple_scores):
        """Test maximum combination."""
        combined = utils.combine_outlier_scores(multiple_scores, method="maximum")

        assert len(combined) == len(multiple_scores[0])
        expected_max = np.max(multiple_scores, axis=0)
        assert np.allclose(combined, expected_max)

    def test_combine_weighted(self, multiple_scores):
        """Test weighted combination."""
        weights = np.array([0.5, 0.3, 0.2])
        combined = utils.combine_outlier_scores(multiple_scores, method="weighted", weights=weights)

        assert len(combined) == len(multiple_scores[0])

    def test_combine_weighted_no_weights(self, multiple_scores):
        """Test weighted combination without weights raises error."""
        with pytest.raises(ValueError, match="Weights must be provided"):
            utils.combine_outlier_scores(multiple_scores, method="weighted")

    def test_combine_rank_average(self, multiple_scores):
        """Test rank-based combination."""
        combined = utils.combine_outlier_scores(multiple_scores, method="rank_average")

        assert len(combined) == len(multiple_scores[0])
        assert np.min(combined) >= 0
        assert np.max(combined) <= 1

    def test_combine_empty_list(self):
        """Test combination with empty list."""
        with pytest.raises(ValueError, match="Empty scores list"):
            utils.combine_outlier_scores([])

    def test_combine_mismatched_lengths(self):
        """Test combination with mismatched lengths."""
        scores_list = [
            np.random.randn(100),
            np.random.randn(50),  # Different length
        ]

        with pytest.raises(ValueError, match="different length"):
            utils.combine_outlier_scores(scores_list)


@pytest.mark.unit
class TestEvaluationMetrics:
    """Tests for evaluation metrics."""

    def test_evaluate_basic(self, sample_predictions):
        """Test basic evaluation."""
        y_true, y_pred = sample_predictions
        metrics = utils.evaluate_outlier_detection(y_true, y_pred)

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "n_outliers_true" in metrics

    def test_evaluate_with_scores(self, sample_predictions, sample_outlier_scores):
        """Test evaluation with continuous scores."""
        y_true, y_pred = sample_predictions
        metrics = utils.evaluate_outlier_detection(y_true, sample_outlier_scores[:100])

        assert "auc_roc" in metrics or "roc_auc" in metrics

    def test_compute_detailed_metrics(self, sample_predictions):
        """Test detailed metrics computation."""
        y_true, y_pred = sample_predictions
        metrics = utils.compute_detailed_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "specificity" in metrics
        assert "confusion_matrix" in metrics
        assert "true_positives" in metrics
        assert "false_positives" in metrics


@pytest.mark.unit
class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""

    def test_generate_global_outliers(self):
        """Test generation of global outliers."""
        X, y = utils.generate_outlier_data(
            n_samples=100, n_features=5, contamination=0.1, outlier_type="global", random_state=42
        )

        assert X.shape == (100, 5)
        assert len(y) == 100
        assert np.sum(y) == 10  # 10% outliers

    def test_generate_local_outliers(self):
        """Test generation of local outliers."""
        X, y = utils.generate_outlier_data(
            n_samples=100, n_features=10, contamination=0.05, outlier_type="local", random_state=42
        )

        assert X.shape == (100, 10)
        assert np.sum(y) == 5  # 5% outliers

    def test_generate_collective_outliers(self):
        """Test generation of collective outliers."""
        X, y = utils.generate_outlier_data(
            n_samples=200,
            n_features=8,
            contamination=0.15,
            outlier_type="collective",
            random_state=42,
        )

        assert X.shape == (200, 8)
        assert np.sum(y) == 30  # 15% outliers

    def test_generate_all_outlier_types(self):
        """Test all outlier types generate valid data."""
        outlier_types = ["global", "local", "collective", "contextual", "point", "mixed"]

        for outlier_type in outlier_types:
            X, y = utils.generate_outlier_data(
                n_samples=100,
                n_features=5,
                contamination=0.1,
                outlier_type=outlier_type,
                random_state=42,
            )

            assert X.shape == (100, 5)
            assert len(y) == 100
            assert 0 < np.sum(y) <= 15  # Should have some outliers

    def test_generate_invalid_outlier_type(self):
        """Test generation with invalid outlier type."""
        with pytest.raises(ValueError, match="Unsupported outlier_type"):
            utils.generate_outlier_data(n_samples=100, n_features=5, outlier_type="invalid")


@pytest.mark.unit
class TestFeatureStandardization:
    """Tests for feature standardization."""

    def test_standardize_basic(self, sample_numeric_data):
        """Test basic standardization."""
        X_scaled, scaler = utils.standardize_features(sample_numeric_data, fit=True)

        # Check mean is approximately 0 and std is approximately 1
        means = X_scaled.mean()
        stds = X_scaled.std()

        assert all(abs(means) < 0.1)
        assert all(abs(stds - 1.0) < 0.1)

    def test_standardize_transform_only(self, sample_numeric_data):
        """Test standardization with pre-fitted scaler."""
        # First fit
        X_scaled, scaler = utils.standardize_features(sample_numeric_data, fit=True)

        # Then transform only
        X_transformed, _ = utils.standardize_features(sample_numeric_data, scaler=scaler, fit=False)

        assert X_scaled.equals(X_transformed)


@pytest.mark.unit
class TestFeatureImportance:
    """Tests for feature importance calculation."""

    def test_compute_importance_correlation(self, sample_numeric_data, sample_outlier_scores):
        """Test importance with correlation method."""
        importance = utils.compute_feature_importance(
            sample_numeric_data, sample_outlier_scores[:100], method="correlation"
        )

        assert isinstance(importance, pd.Series)
        assert len(importance) == len(sample_numeric_data.columns)
        assert all(importance >= 0)

    def test_compute_importance_mutual_info(self, sample_numeric_data, sample_outlier_scores):
        """Test importance with mutual information method."""
        importance = utils.compute_feature_importance(
            sample_numeric_data, sample_outlier_scores[:100], method="mutual_info"
        )

        assert isinstance(importance, pd.Series)
        assert len(importance) == len(sample_numeric_data.columns)
        assert all(importance >= 0)

    def test_compute_importance_invalid_method(self, sample_numeric_data, sample_outlier_scores):
        """Test importance with invalid method."""
        with pytest.raises(ValueError, match="Unsupported method"):
            utils.compute_feature_importance(
                sample_numeric_data, sample_outlier_scores[:100], method="invalid"
            )


@pytest.mark.unit
class TestThresholdSelection:
    """Tests for threshold selection utilities."""

    def test_select_threshold_percentile(self, sample_outlier_scores):
        """Test percentile-based threshold selection."""
        threshold = utils.select_threshold(
            sample_outlier_scores, method="percentile", contamination=0.1
        )

        assert isinstance(threshold, (float, np.floating))
        # About 10% should be above threshold
        outliers: int = int(np.sum(sample_outlier_scores > threshold))
        assert 8 <= outliers <= 12  # Allow some tolerance

    def test_select_threshold_median(self, sample_outlier_scores):
        """Test median-based threshold."""
        threshold = utils.select_threshold(sample_outlier_scores, method="median")

        assert threshold == np.median(sample_outlier_scores)

    def test_select_threshold_mad(self, sample_outlier_scores):
        """Test MAD-based threshold."""
        threshold = utils.select_threshold(sample_outlier_scores, method="mad", k=1.5)

        assert isinstance(threshold, (float, np.floating))

    def test_select_threshold_iqr(self, sample_outlier_scores):
        """Test IQR-based threshold."""
        threshold = utils.select_threshold(sample_outlier_scores, method="iqr", k=1.5)

        assert isinstance(threshold, (float, np.floating))

    def test_select_threshold_std(self, sample_outlier_scores):
        """Test std-based threshold."""
        threshold = utils.select_threshold(sample_outlier_scores, method="std", k=2.0)

        expected = np.mean(sample_outlier_scores) + 2.0 * np.std(sample_outlier_scores)
        assert np.isclose(threshold, expected)


@pytest.mark.unit
class TestScoreNormalization:
    """Tests for score normalization."""

    def test_normalize_minmax(self, sample_outlier_scores):
        """Test min-max normalization."""
        normalized = utils.normalize_scores(sample_outlier_scores, method="minmax")

        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0

    def test_normalize_zscore(self, sample_outlier_scores):
        """Test z-score normalization."""
        normalized = utils.normalize_scores(sample_outlier_scores, method="zscore")

        assert all(0 <= normalized) and all(normalized <= 1)

    def test_normalize_rank(self, sample_outlier_scores):
        """Test rank normalization."""
        normalized = utils.normalize_scores(sample_outlier_scores, method="rank")

        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0

    def test_normalize_sigmoid(self, sample_outlier_scores):
        """Test sigmoid normalization."""
        normalized = utils.normalize_scores(sample_outlier_scores, method="sigmoid")

        assert all(0 < normalized) and all(normalized < 1)

    def test_normalize_preserves_order(self, sample_outlier_scores):
        """Test that normalization preserves rank order."""
        normalized = utils.normalize_scores(sample_outlier_scores, method="minmax")

        original_order = np.argsort(sample_outlier_scores)
        normalized_order = np.argsort(normalized)

        assert np.array_equal(original_order, normalized_order)


@pytest.mark.unit
class TestDataProcessing:
    """Tests for data preprocessing utilities."""

    def test_remove_outliers_remove_strategy(self, outlier_data):
        """Test outlier removal strategy."""
        outlier_mask = np.zeros(len(outlier_data), dtype=bool)
        outlier_mask[-5:] = True  # Mark last 5 as outliers

        result = utils.remove_outliers(outlier_data, outlier_mask, strategy="remove")

        assert len(result) == len(outlier_data) - 5

    def test_remove_outliers_cap_strategy(self, outlier_data):
        """Test outlier capping strategy."""
        outlier_mask = np.zeros(len(outlier_data), dtype=bool)
        outlier_mask[-5:] = True

        result = utils.remove_outliers(outlier_data, outlier_mask, strategy="cap")

        assert len(result) == len(outlier_data)

    def test_remove_outliers_impute_strategy(self, outlier_data):
        """Test outlier imputation strategy."""
        outlier_mask = np.zeros(len(outlier_data), dtype=bool)
        outlier_mask[-5:] = True

        result = utils.remove_outliers(outlier_data, outlier_mask, strategy="impute")

        assert len(result) == len(outlier_data)

    def test_create_feature_summary(self, sample_numeric_data):
        """Test feature summary creation."""
        summary = utils.create_feature_summary(sample_numeric_data)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == len(sample_numeric_data.columns)
        assert "feature" in summary.columns
        assert "mean" in summary.columns
        assert "std" in summary.columns

    def test_detect_feature_drift(self, sample_numeric_data):
        """Test feature drift detection."""
        # Create slightly different dataset
        df_ref = sample_numeric_data
        df_curr = sample_numeric_data + np.random.randn(*sample_numeric_data.shape) * 0.1

        drift_results = utils.detect_feature_drift(df_ref, df_curr, threshold=0.1)

        assert "drifted_features" in drift_results
        assert "drift_scores" in drift_results
        assert isinstance(drift_results["drifted_features"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
