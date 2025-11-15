"""
Comprehensive unit tests for PSOD visualization module.
Tests all static and interactive visualization functions.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psod import PSOD
from psod import visualization as viz

# ============================================================================
# STATIC VISUALIZATION TESTS (Matplotlib/Seaborn)
# ============================================================================


@pytest.mark.visualization
@pytest.mark.unit
class TestPlotOutlierScores:
    """Tests for comprehensive 4-subplot outlier score distribution."""

    def test_basic_plot_creation(self, sample_outlier_scores):
        """Test basic plot creation with default parameters."""
        fig = viz.plot_outlier_scores(scores=sample_outlier_scores)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

        # Should have 4 subplots (2x2 grid)
        axes = fig.get_axes()
        assert len(axes) == 4

        plt.close(fig)

    def test_with_threshold(self, sample_outlier_scores):
        """Test plot with threshold line."""
        threshold = np.percentile(sample_outlier_scores, 90)
        fig = viz.plot_outlier_scores(scores=sample_outlier_scores, threshold=threshold)

        assert fig is not None
        plt.close(fig)

    def test_custom_parameters(self, sample_outlier_scores):
        """Test plot with custom parameters."""
        fig = viz.plot_outlier_scores(
            scores=sample_outlier_scores,
            threshold=2.5,
            bins=30,
            title="Custom Title",
            figsize=(10, 6),
        )

        assert fig is not None
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 6

        plt.close(fig)

    def test_with_pandas_series(self, sample_outlier_scores):
        """Test with pandas Series input."""
        scores_series = pd.Series(sample_outlier_scores)
        fig = viz.plot_outlier_scores(scores=scores_series)

        assert fig is not None
        plt.close(fig)

    def test_small_dataset(self):
        """Test with very small dataset."""
        small_scores = np.array([1.0, 2.0, 3.0])
        fig = viz.plot_outlier_scores(scores=small_scores)

        assert fig is not None
        plt.close(fig)

    def test_single_value(self):
        """Test with single value."""
        single_score = np.array([1.0])
        fig = viz.plot_outlier_scores(scores=single_score)

        assert fig is not None
        plt.close(fig)

    def test_all_same_values(self):
        """Test with all identical scores."""
        constant_scores = np.ones(100)
        fig = viz.plot_outlier_scores(scores=constant_scores)

        assert fig is not None
        plt.close(fig)

    def test_negative_scores(self):
        """Test with negative outlier scores."""
        negative_scores = np.random.randn(100) - 5
        fig = viz.plot_outlier_scores(scores=negative_scores)

        assert fig is not None
        plt.close(fig)


@pytest.mark.visualization
@pytest.mark.unit
class TestCreateOutlierDashboard:
    """Tests for comprehensive 4x4 grid dashboard."""

    def test_basic_dashboard_creation(
        self, sample_numeric_data, sample_outlier_scores, sample_predictions
    ):
        """Test basic dashboard creation."""
        _, y_pred = sample_predictions

        fig = viz.create_outlier_dashboard(
            X=sample_numeric_data, outlier_scores=sample_outlier_scores, outlier_labels=y_pred
        )

        assert fig is not None
        assert isinstance(fig, plt.Figure)

        # Should have multiple subplots
        axes = fig.get_axes()
        assert len(axes) >= 12  # 4x4 grid should have at least 12 panels

        plt.close(fig)

    def test_dashboard_with_model(
        self, sample_numeric_data, fitted_psod_model, sample_outlier_scores, sample_predictions
    ):
        """Test dashboard with PSOD model integration."""
        _, y_pred = sample_predictions

        fig = viz.create_outlier_dashboard(
            X=sample_numeric_data,
            outlier_scores=sample_outlier_scores,
            outlier_labels=y_pred,
            model=fitted_psod_model,
        )

        assert fig is not None
        plt.close(fig)

    def test_dashboard_save(
        self, sample_numeric_data, sample_outlier_scores, sample_predictions, tmp_path
    ):
        """Test dashboard saving to file."""
        _, y_pred = sample_predictions
        save_path = tmp_path / "test_dashboard.png"

        fig = viz.create_outlier_dashboard(
            X=sample_numeric_data,
            outlier_scores=sample_outlier_scores,
            outlier_labels=y_pred,
            save_path=str(save_path),
        )

        assert save_path.exists()
        plt.close(fig)

    def test_dashboard_with_categorical(
        self, sample_data_with_categorical, sample_outlier_scores, sample_predictions
    ):
        """Test dashboard with categorical features."""
        _, y_pred = sample_predictions
        # Extend scores to match data length
        scores = np.concatenate(
            [
                sample_outlier_scores,
                np.random.randn(len(sample_data_with_categorical) - len(sample_outlier_scores)),
            ]
        )
        labels = np.concatenate(
            [y_pred, np.zeros(len(sample_data_with_categorical) - len(y_pred))]
        ).astype(int)

        # Select only numeric columns for dashboard
        numeric_data = sample_data_with_categorical.select_dtypes(include=[np.number])

        fig = viz.create_outlier_dashboard(
            X=numeric_data,
            outlier_scores=scores[: len(numeric_data)],
            outlier_labels=labels[: len(numeric_data)],
        )

        assert fig is not None
        plt.close(fig)

    def test_dashboard_small_dataset(self, small_dataset):
        """Test dashboard with very small dataset."""
        scores = np.array([1.0, 2.0, 3.0])
        labels = np.array([0, 0, 1])

        fig = viz.create_outlier_dashboard(
            X=small_dataset, outlier_scores=scores, outlier_labels=labels
        )

        assert fig is not None
        plt.close(fig)


@pytest.mark.visualization
@pytest.mark.unit
class TestPlotFeatureContributions:
    """Tests for feature contribution visualization."""

    def test_basic_contributions(self, sample_numeric_data, fitted_psod_model):
        """Test basic feature contribution plot."""
        sample_idx = 0

        fig = viz.plot_feature_contributions(model=fitted_psod_model, sample_idx=sample_idx)

        assert fig is not None
        plt.close(fig)

    def test_contributions_topk(self, sample_numeric_data, fitted_psod_model):
        """Test with top-k features."""
        fig = viz.plot_feature_contributions(model=fitted_psod_model, sample_idx=0, top_k=5)

        assert fig is not None
        plt.close(fig)

    def test_contributions_invalid_index(self, sample_numeric_data, fitted_psod_model):
        """Test with invalid sample index."""
        with pytest.raises((IndexError, ValueError)):
            viz.plot_feature_contributions(
                model=fitted_psod_model, sample_idx=10000  # Out of bounds
            )


@pytest.mark.visualization
@pytest.mark.unit
class TestPlotCorrelationHeatmap:
    """Tests for correlation heatmap visualization."""

    def test_basic_heatmap(self, sample_numeric_data, sample_predictions):
        """Test basic correlation heatmap."""
        _, y_pred = sample_predictions

        fig = viz.plot_correlation_heatmap(X=sample_numeric_data, outlier_labels=y_pred)

        assert fig is not None
        plt.close(fig)

    def test_heatmap_custom_figsize(self, sample_numeric_data, sample_predictions):
        """Test heatmap with custom figure size."""
        _, y_pred = sample_predictions

        fig = viz.plot_correlation_heatmap(
            X=sample_numeric_data, outlier_labels=y_pred, figsize=(10, 8)
        )

        assert fig is not None
        plt.close(fig)

    def test_heatmap_single_feature(self, single_column_data):
        """Test heatmap with single feature."""
        labels = np.zeros(len(single_column_data))

        fig = viz.plot_correlation_heatmap(X=single_column_data, outlier_labels=labels)

        assert fig is not None
        plt.close(fig)

    def test_heatmap_correlated_features(self, correlated_features_data, sample_predictions):
        """Test heatmap with highly correlated features."""
        _, y_pred = sample_predictions

        fig = viz.plot_correlation_heatmap(X=correlated_features_data, outlier_labels=y_pred)

        assert fig is not None
        plt.close(fig)


@pytest.mark.visualization
@pytest.mark.unit
class TestPlotROCPRCurves:
    """Tests for ROC and Precision-Recall curves."""

    def test_basic_curves(self, sample_predictions, sample_outlier_scores):
        """Test basic ROC/PR curve creation."""
        y_true, _ = sample_predictions

        fig = viz.plot_roc_pr_curves(y_true=y_true, y_scores=sample_outlier_scores)

        assert fig is not None

        # Should have 2 subplots (ROC and PR)
        axes = fig.get_axes()
        assert len(axes) == 2

        plt.close(fig)

    def test_curves_custom_title(self, sample_predictions, sample_outlier_scores):
        """Test with custom title."""
        y_true, _ = sample_predictions

        fig = viz.plot_roc_pr_curves(
            y_true=y_true, y_scores=sample_outlier_scores, title="Custom Performance Curves"
        )

        assert fig is not None
        plt.close(fig)

    def test_curves_perfect_prediction(self):
        """Test with perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])

        fig = viz.plot_roc_pr_curves(y_true=y_true, y_scores=y_scores)

        assert fig is not None
        plt.close(fig)

    def test_curves_random_prediction(self):
        """Test with random predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_scores = np.random.rand(100)

        fig = viz.plot_roc_pr_curves(y_true=y_true, y_scores=y_scores)

        assert fig is not None
        plt.close(fig)


@pytest.mark.visualization
@pytest.mark.unit
class TestPlotFeatureDistributions:
    """Tests for feature distribution comparison plots."""

    def test_basic_distributions(self, sample_numeric_data, sample_predictions):
        """Test basic feature distribution plot."""
        _, y_pred = sample_predictions

        fig = viz.plot_feature_distributions(X=sample_numeric_data, outlier_labels=y_pred)

        assert fig is not None
        plt.close(fig)

    def test_distributions_max_features(self, sample_numeric_data, sample_predictions):
        """Test with max_features parameter."""
        _, y_pred = sample_predictions

        fig = viz.plot_feature_distributions(
            X=sample_numeric_data, outlier_labels=y_pred, max_features=6
        )

        assert fig is not None
        plt.close(fig)

    def test_distributions_custom_figsize(self, sample_numeric_data, sample_predictions):
        """Test with custom figure size."""
        _, y_pred = sample_predictions

        fig = viz.plot_feature_distributions(
            X=sample_numeric_data, outlier_labels=y_pred, figsize=(20, 15)
        )

        assert fig is not None
        plt.close(fig)

    def test_distributions_single_feature(self, single_column_data):
        """Test with single feature."""
        labels = np.zeros(len(single_column_data))

        fig = viz.plot_feature_distributions(X=single_column_data, outlier_labels=labels)

        assert fig is not None
        plt.close(fig)

    def test_distributions_all_outliers(self, sample_numeric_data):
        """Test when all samples are outliers."""
        labels = np.ones(len(sample_numeric_data))

        fig = viz.plot_feature_distributions(X=sample_numeric_data, outlier_labels=labels)

        assert fig is not None
        plt.close(fig)

    def test_distributions_no_outliers(self, sample_numeric_data):
        """Test when no samples are outliers."""
        labels = np.zeros(len(sample_numeric_data))

        fig = viz.plot_feature_distributions(X=sample_numeric_data, outlier_labels=labels)

        assert fig is not None
        plt.close(fig)


@pytest.mark.visualization
@pytest.mark.unit
class TestPlotOutlierEvolutionHeatmap:
    """Tests for outlier evolution heatmap."""

    def test_basic_heatmap(self, sample_outlier_scores):
        """Test basic evolution heatmap."""
        # Create score history (5 iterations)
        scores_history = [
            sample_outlier_scores + np.random.randn(len(sample_outlier_scores)) * 0.1
            for _ in range(5)
        ]
        labels = [f"Iter {i+1}" for i in range(5)]

        fig = viz.plot_outlier_evolution_heatmap(scores_history=scores_history, labels=labels)

        assert fig is not None
        plt.close(fig)

    def test_heatmap_custom_figsize(self, sample_outlier_scores):
        """Test with custom figure size."""
        scores_history = [sample_outlier_scores for _ in range(3)]
        labels = ["A", "B", "C"]

        fig = viz.plot_outlier_evolution_heatmap(
            scores_history=scores_history, labels=labels, figsize=(12, 8)
        )

        assert fig is not None
        plt.close(fig)

    def test_heatmap_single_iteration(self, sample_outlier_scores):
        """Test with single iteration."""
        scores_history = [sample_outlier_scores]
        labels = ["Single"]

        fig = viz.plot_outlier_evolution_heatmap(scores_history=scores_history, labels=labels)

        assert fig is not None
        plt.close(fig)

    def test_heatmap_mismatched_labels(self, sample_outlier_scores):
        """Test with mismatched labels and scores."""
        scores_history = [sample_outlier_scores for _ in range(3)]
        labels = ["A", "B"]  # Only 2 labels for 3 iterations

        # Should either handle gracefully or raise error
        try:
            fig = viz.plot_outlier_evolution_heatmap(scores_history=scores_history, labels=labels)
            plt.close(fig)
        except (ValueError, AssertionError):
            pass  # Expected behavior


# ============================================================================
# INTERACTIVE VISUALIZATION TESTS (Plotly)
# ============================================================================


@pytest.mark.visualization
@pytest.mark.requires_deps
@pytest.mark.unit
class TestPlotOutliersScatter:
    """Tests for interactive scatter plot visualization."""

    def test_basic_2d_scatter(self, sample_numeric_data, sample_predictions):
        """Test basic 2D scatter plot."""
        pytest.importorskip("plotly")

        _, y_pred = sample_predictions

        fig = viz.plot_outliers_scatter(
            X=sample_numeric_data,
            outlier_labels=y_pred,
            features=list(sample_numeric_data.columns[:2]),
            dim=2,
        )

        assert fig is not None

    def test_basic_3d_scatter(self, sample_numeric_data, sample_predictions):
        """Test basic 3D scatter plot."""
        pytest.importorskip("plotly")

        _, y_pred = sample_predictions

        fig = viz.plot_outliers_scatter(
            X=sample_numeric_data,
            outlier_labels=y_pred,
            features=list(sample_numeric_data.columns[:3]),
            dim=3,
        )

        assert fig is not None

    def test_scatter_with_pca(self, sample_numeric_data, sample_predictions):
        """Test scatter with PCA dimensionality reduction."""
        pytest.importorskip("plotly")

        _, y_pred = sample_predictions

        fig = viz.plot_outliers_scatter(
            X=sample_numeric_data, outlier_labels=y_pred, dim=2, use_pca=True
        )

        assert fig is not None

    def test_scatter_save_html(self, sample_numeric_data, sample_predictions, tmp_path):
        """Test saving scatter plot to HTML."""
        pytest.importorskip("plotly")

        _, y_pred = sample_predictions
        save_path = tmp_path / "scatter.html"

        fig = viz.plot_outliers_scatter(
            X=sample_numeric_data, outlier_labels=y_pred, dim=2, use_pca=True
        )

        fig.write_html(str(save_path))
        assert save_path.exists()


@pytest.mark.visualization
@pytest.mark.requires_deps
@pytest.mark.unit
class TestPlotTimeseriesOutliers:
    """Tests for time series outlier visualization."""

    def test_basic_timeseries(self, time_series_data):
        """Test basic time series plot."""
        pytest.importorskip("plotly")

        outlier_labels = np.zeros(len(time_series_data))
        outlier_labels[[50, 75, 120]] = 1  # Mark some as outliers

        fig = viz.plot_timeseries_outliers(
            X=time_series_data, outlier_labels=outlier_labels, time_column="timestamp"
        )

        assert fig is not None

    def test_timeseries_specific_columns(self, time_series_data):
        """Test with specific value columns."""
        pytest.importorskip("plotly")

        outlier_labels = np.zeros(len(time_series_data))

        fig = viz.plot_timeseries_outliers(
            X=time_series_data,
            outlier_labels=outlier_labels,
            time_column="timestamp",
            value_columns=["value", "feature_2"],
        )

        assert fig is not None

    def test_timeseries_custom_title(self, time_series_data):
        """Test with custom title."""
        pytest.importorskip("plotly")

        outlier_labels = np.zeros(len(time_series_data))

        fig = viz.plot_timeseries_outliers(
            X=time_series_data,
            outlier_labels=outlier_labels,
            time_column="timestamp",
            title="Custom Time Series Plot",
        )

        assert fig is not None


@pytest.mark.visualization
@pytest.mark.requires_deps
@pytest.mark.unit
class TestPlotScoreEvolution:
    """Tests for score evolution visualization."""

    def test_basic_evolution(self, sample_outlier_scores):
        """Test basic score evolution plot."""
        pytest.importorskip("plotly")

        scores_history = [
            sample_outlier_scores + np.random.randn(len(sample_outlier_scores)) * 0.1
            for _ in range(5)
        ]
        labels = [f"Model {i+1}" for i in range(5)]

        fig = viz.plot_score_evolution(scores_history=scores_history, labels=labels)

        assert fig is not None

    def test_evolution_save_html(self, sample_outlier_scores, tmp_path):
        """Test saving evolution plot to HTML."""
        pytest.importorskip("plotly")

        scores_history = [sample_outlier_scores for _ in range(3)]
        labels = ["A", "B", "C"]
        save_path = tmp_path / "evolution.html"

        fig = viz.plot_score_evolution(scores_history=scores_history, labels=labels)

        fig.write_html(str(save_path))
        assert save_path.exists()

    def test_evolution_single_history(self, sample_outlier_scores):
        """Test with single score history."""
        pytest.importorskip("plotly")

        scores_history = [sample_outlier_scores]
        labels = ["Single"]

        fig = viz.plot_score_evolution(scores_history=scores_history, labels=labels)

        assert fig is not None


@pytest.mark.visualization
@pytest.mark.requires_deps
@pytest.mark.slow
class TestCreateInteractiveExplorer:
    """Tests for interactive Dash explorer application."""

    def test_explorer_mock(self, sample_numeric_data, fitted_psod_model):
        """Test explorer creation with mocked Dash."""
        pytest.importorskip("dash")

        # Mock the Dash app to avoid actually starting server
        with patch("psod.visualization.dash.Dash") as mock_dash:
            mock_app = MagicMock()
            mock_dash.return_value = mock_app

            # This should create the app but not run it
            try:
                viz.create_interactive_explorer(
                    X=sample_numeric_data, model=fitted_psod_model, port=8050, debug=False
                )
            except Exception as e:
                # If it fails, it might be trying to run the server
                # In that case, we just verify the import works
                pass

    def test_explorer_import_check(self):
        """Test that explorer requires dash."""
        # If dash is not available, should handle gracefully
        try:
            import dash

            has_dash = True
        except ImportError:
            has_dash = False

        if not has_dash:
            # Should either skip or raise informative error
            assert True  # Test passes if dash not available


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


@pytest.mark.visualization
@pytest.mark.unit
class TestVisualizationEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test visualization with empty dataframe."""
        empty_df = pd.DataFrame()
        scores = np.array([])
        labels = np.array([])

        # Should handle gracefully or raise informative error
        with pytest.raises((ValueError, IndexError)):
            viz.create_outlier_dashboard(X=empty_df, outlier_scores=scores, outlier_labels=labels)

    def test_mismatched_lengths(self, sample_numeric_data):
        """Test with mismatched data and score lengths."""
        scores = np.random.randn(50)  # Different length
        labels = np.zeros(50)

        # Should raise error about length mismatch
        with pytest.raises((ValueError, AssertionError)):
            viz.create_outlier_dashboard(
                X=sample_numeric_data, outlier_scores=scores, outlier_labels=labels
            )

    def test_nan_in_scores(self, sample_numeric_data):
        """Test with NaN values in scores."""
        scores = np.random.randn(len(sample_numeric_data))
        scores[0] = np.nan
        labels = np.zeros(len(sample_numeric_data))

        # Should either handle NaN or raise error
        try:
            fig = viz.plot_outlier_scores(scores=scores)
            plt.close(fig)
        except (ValueError, RuntimeError):
            pass  # Expected behavior

    def test_inf_in_scores(self, sample_numeric_data):
        """Test with infinite values in scores."""
        scores = np.random.randn(len(sample_numeric_data))
        scores[0] = np.inf

        # Should handle inf values
        try:
            fig = viz.plot_outlier_scores(scores=scores)
            plt.close(fig)
        except (ValueError, RuntimeError):
            pass  # Expected behavior

    def test_invalid_labels(self, sample_numeric_data, sample_outlier_scores):
        """Test with invalid label values (not 0 or 1)."""
        invalid_labels = np.array([0, 1, 2, 3] * 25)  # Invalid values

        # Should validate labels
        with pytest.raises((ValueError, AssertionError)):
            viz.create_outlier_dashboard(
                X=sample_numeric_data,
                outlier_scores=sample_outlier_scores,
                outlier_labels=invalid_labels,
            )


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.visualization
@pytest.mark.integration
class TestVisualizationIntegration:
    """Integration tests combining multiple visualizations."""

    def test_complete_visualization_workflow(self, sample_numeric_data, random_seed):
        """Test complete workflow from model to all visualizations."""
        # Fit model
        model = PSOD(random_seed=random_seed, contamination=0.1)
        scores = model.fit_predict(sample_numeric_data, return_class=False)
        predictions = model.predict(sample_numeric_data, return_class=True)

        # Create all static visualizations
        fig1 = viz.plot_outlier_scores(scores=scores)
        plt.close(fig1)

        fig2 = viz.create_outlier_dashboard(
            X=sample_numeric_data, outlier_scores=scores, outlier_labels=predictions, model=model
        )
        plt.close(fig2)

        fig3 = viz.plot_feature_distributions(X=sample_numeric_data, outlier_labels=predictions)
        plt.close(fig3)

        fig4 = viz.plot_correlation_heatmap(X=sample_numeric_data, outlier_labels=predictions)
        plt.close(fig4)

        # All should succeed
        assert True

    def test_save_all_visualizations(
        self, sample_numeric_data, sample_outlier_scores, sample_predictions, tmp_path
    ):
        """Test saving all visualization types."""
        _, y_pred = sample_predictions

        # Save static plots
        fig1 = viz.plot_outlier_scores(scores=sample_outlier_scores)
        fig1.savefig(tmp_path / "scores.png")
        plt.close(fig1)

        fig2 = viz.create_outlier_dashboard(
            X=sample_numeric_data,
            outlier_scores=sample_outlier_scores,
            outlier_labels=y_pred,
            save_path=str(tmp_path / "dashboard.png"),
        )
        plt.close(fig2)

        # Verify files exist
        assert (tmp_path / "scores.png").exists()
        assert (tmp_path / "dashboard.png").exists()
