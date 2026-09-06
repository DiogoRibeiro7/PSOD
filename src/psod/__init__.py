"""Public package interface for PSOD."""

from importlib import import_module
from typing import Any

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

_VISUALIZATION_EXPORTS = {
    "create_interactive_explorer",
    "create_outlier_dashboard",
    "plot_correlation_heatmap",
    "plot_feature_contributions",
    "plot_feature_distributions",
    "plot_outlier_evolution_heatmap",
    "plot_outlier_scores",
    "plot_outliers_scatter",
    "plot_roc_pr_curves",
    "plot_score_evolution",
    "plot_timeseries_outliers",
}


def __getattr__(name: str) -> Any:
    """Load optional visualization helpers only when they are requested."""
    if name not in _VISUALIZATION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = import_module(".visualization", __name__)
    except ModuleNotFoundError as exc:
        if exc.name in {"matplotlib", "seaborn", "plotly"}:
            raise ImportError(
                f"{name} requires the optional visualization dependencies; "
                "install PSOD with the 'viz' extra"
            ) from exc
        raise

    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "PSOD",
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
    *_VISUALIZATION_EXPORTS,
]
