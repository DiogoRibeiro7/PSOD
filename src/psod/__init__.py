"""
PSOD submodule: Core classes and functions.

This module contains the main PSOD implementation and supporting utilities.
"""

from .core import PSOD
from .utils import (
    calibrate_outlier_scores,
    combine_outlier_scores,
    compute_feature_importance,
    evaluate_outlier_detection,
    generate_outlier_data,
    handle_missing_values,
    load_model,
    save_model,
    standardize_features,
    validate_dataframe,
)
from .visualization import (
    create_interactive_explorer,
    create_outlier_dashboard,
    plot_correlation_heatmap,
    plot_feature_contributions,
    plot_feature_distributions,
    plot_outlier_evolution_heatmap,
    plot_outlier_scores,
    plot_outliers_scatter,
    plot_roc_pr_curves,
    plot_score_evolution,
    plot_timeseries_outliers,
)

__all__ = [
    # Core
    "PSOD",
    # Utils
    "save_model",
    "load_model",
    "validate_dataframe",
    "handle_missing_values",
    "calibrate_outlier_scores",
    "combine_outlier_scores",
    "evaluate_outlier_detection",
    "generate_outlier_data",
    "standardize_features",
    "compute_feature_importance",
    # Visualization
    "plot_outlier_scores",
    "plot_feature_contributions",
    "plot_outliers_scatter",
    "plot_timeseries_outliers",
    "plot_correlation_heatmap",
    "plot_score_evolution",
    "plot_roc_pr_curves",
    "create_outlier_dashboard",
    "create_interactive_explorer",
    "plot_feature_distributions",
    "plot_outlier_evolution_heatmap",
]
