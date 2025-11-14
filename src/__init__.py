"""
PSOD: Pseudo-Supervised Outlier Detection

A flexible outlier detection library using pseudo-supervised learning approach.

This package provides a comprehensive outlier detection system that uses a pseudo-supervised
approach by treating each feature as a target variable and using prediction errors as
outlier scores. The library includes visualization tools, utility functions, and is
designed to be compatible with scikit-learn.

Key Features:
- Pseudo-supervised outlier detection using ensemble of regressors
- Support for both numerical and categorical features
- Multiple data transformation options
- Comprehensive visualization suite
- Feature importance analysis
- Model persistence and serialization
- Sklearn compatibility

Example Usage:
    >>> import pandas as pd
    >>> from psod import PSOD
    >>>
    >>> # Load your data
    >>> df = pd.read_csv('your_data.csv')
    >>>
    >>> # Initialize and fit PSOD
    >>> detector = PSOD(contamination=0.1)
    >>> outlier_scores = detector.fit_predict(df)
    >>>
    >>> # Get outlier labels
    >>> outlier_labels = detector.predict(df, return_class=True)
"""

import logging
from typing import List

# Add proper version management (using simple approach, can be enhanced with versioneer)
__version__ = "0.1.0"
__author__ = "Diogo Ribeiro"
__email__ = "dfr@esmad.ipp.pt"
__license__ = "MIT"
__description__ = (
    "Pseudo-Supervised Outlier Detection library with comprehensive analysis tools"
)

# Set up logging for the package
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Import main classes and functions for easier access
try:
    from .psod.core import PSOD

    _CORE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Core PSOD module not available: {e}")
    _CORE_AVAILABLE = False

try:
    from .psod.utils import (
        save_model,
        load_model,
        validate_dataframe,
        handle_missing_values,
        calibrate_outlier_scores,
        combine_outlier_scores,
        evaluate_outlier_detection,
        generate_outlier_data,
        standardize_features,
        compute_feature_importance,
    )

    _UTILS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Utils module not available: {e}")
    _UTILS_AVAILABLE = False

try:
    from .psod.visualization import (
        plot_outlier_scores,
        plot_feature_contributions,
        plot_outliers_scatter,
        plot_timeseries_outliers,
        plot_correlation_heatmap,
        plot_score_evolution,
        plot_roc_pr_curves,
        create_outlier_dashboard,
        create_interactive_explorer,
        plot_feature_distributions,
        plot_outlier_evolution_heatmap,
    )

    _VISUALIZATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Visualization module not available: {e}")
    _VISUALIZATION_AVAILABLE = False

# Define __all__ for explicit API
_all_exports: List[str] = []

# Core functionality
if _CORE_AVAILABLE:
    _all_exports.extend(
        [
            "PSOD",
        ]
    )

# Utility functions
if _UTILS_AVAILABLE:
    _all_exports.extend(
        [
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
        ]
    )

# Visualization functions
if _VISUALIZATION_AVAILABLE:
    _all_exports.extend(
        [
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
    )

__all__ = _all_exports


def get_version() -> str:
    """Get the package version."""
    return __version__


def get_package_info() -> dict:
    """Get comprehensive package information."""
    return {
        "name": "PSOD",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "description": __description__,
        "modules": {
            "core": _CORE_AVAILABLE,
            "utils": _UTILS_AVAILABLE,
            "visualization": _VISUALIZATION_AVAILABLE,
        },
        "available_functions": len(__all__),
    }


def check_dependencies() -> dict:
    """Check if all required dependencies are available."""
    dependencies = {"required": {}, "optional": {}, "missing": []}

    # Check required dependencies
    required_deps = [
        ("pandas", "Data manipulation and analysis"),
        ("numpy", "Numerical computing"),
        ("scikit-learn", "Machine learning utilities"),
        ("tqdm", "Progress bars"),
        ("joblib", "Parallel processing and model persistence"),
    ]

    for dep_name, description in required_deps:
        try:
            __import__(dep_name)
            dependencies["required"][dep_name] = {
                "available": True,
                "description": description,
            }
        except ImportError:
            dependencies["required"][dep_name] = {
                "available": False,
                "description": description,
            }
            dependencies["missing"].append(dep_name)

    # Check optional dependencies
    optional_deps = [
        ("matplotlib", "Static plotting for visualization module"),
        ("seaborn", "Statistical data visualization"),
        ("plotly", "Interactive plotting"),
        ("dash", "Interactive web applications"),
        ("category_encoders", "Categorical variable encoding"),
    ]

    for dep_name, description in optional_deps:
        try:
            __import__(dep_name)
            dependencies["optional"][dep_name] = {
                "available": True,
                "description": description,
            }
        except ImportError:
            dependencies["optional"][dep_name] = {
                "available": False,
                "description": description,
            }

    return dependencies


def setup_logging(level: str = "INFO", format_string: str = None) -> None:
    """
    Set up logging for the PSOD package.

    Parameters
    ----------
    level : str, default="INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    format_string : str, optional
        Custom format string for log messages.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.info(f"PSOD logging setup complete. Level: {level}")


def quick_start_example():
    """
    Print a quick start example for using PSOD.
    """
    example_code = """
# Quick Start Example for PSOD

import pandas as pd
from psod import PSOD, generate_outlier_data, plot_outlier_scores

# 1. Generate synthetic data (or load your own)
X, y_true = generate_outlier_data(
    n_samples=1000, 
    n_features=10, 
    contamination=0.1,
    random_state=42
)

# 2. Initialize PSOD detector
detector = PSOD(
    contamination=0.1,
    transform_algorithm="yeo-johnson",
    random_seed=42
)

# 3. Fit and get outlier scores
outlier_scores = detector.fit_predict(X)

# 4. Get binary outlier labels
outlier_labels = detector.predict(X, return_class=True)

# 5. Visualize results
plot_outlier_scores(outlier_scores, 
                   threshold=detector.get_outlier_threshold(),
                   title="PSOD Outlier Scores")

# 6. Get feature importance
feature_importance = detector.get_feature_importance()
print("Top 5 most important features:")
print(feature_importance.head())

# 7. Explain a specific outlier
outlier_idx = outlier_labels[outlier_labels == 1].index[0]
explanation = detector.explain_outlier(X, outlier_idx)
print(f"Explanation for outlier at index {outlier_idx}:")
print(explanation)

# 8. Save the trained model
detector.save_model("my_psod_model")

# 9. Later, load the model
loaded_detector = PSOD.load_model("my_psod_model")
"""
    print(example_code)


def list_available_functions():
    """List all available functions in the package."""
    info = {
        "Core Functions": [],
        "Utility Functions": [],
        "Visualization Functions": [],
    }

    if _CORE_AVAILABLE:
        info["Core Functions"].append("PSOD - Main outlier detection class")

    if _UTILS_AVAILABLE:
        utils_funcs = [
            "save_model - Save trained models to disk",
            "load_model - Load trained models from disk",
            "validate_dataframe - Comprehensive data validation",
            "handle_missing_values - Missing value imputation",
            "calibrate_outlier_scores - Score calibration",
            "combine_outlier_scores - Ensemble score combination",
            "evaluate_outlier_detection - Performance evaluation",
            "generate_outlier_data - Synthetic data generation",
            "standardize_features - Feature standardization",
            "compute_feature_importance - Feature importance calculation",
        ]
        info["Utility Functions"].extend(utils_funcs)

    if _VISUALIZATION_AVAILABLE:
        viz_funcs = [
            "plot_outlier_scores - Score distribution plots",
            "plot_feature_contributions - Feature importance plots",
            "plot_outliers_scatter - 2D/3D scatter plots",
            "plot_timeseries_outliers - Time series visualization",
            "plot_correlation_heatmap - Correlation analysis",
            "plot_score_evolution - Score evolution tracking",
            "plot_roc_pr_curves - ROC and PR curves",
            "create_outlier_dashboard - Comprehensive dashboard",
            "create_interactive_explorer - Interactive web app",
            "plot_feature_distributions - Feature distribution analysis",
            "plot_outlier_evolution_heatmap - Evolution heatmaps",
        ]
        info["Visualization Functions"].extend(viz_funcs)

    return info


# Package metadata for external tools
metadata = {
    "name": "psod",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "author_email": __email__,
    "license": __license__,
    "url": "https://github.com/diogoribeiro7/psod",  # Update with actual URL
    "keywords": [
        "outlier detection",
        "anomaly detection",
        "machine learning",
        "data science",
    ],
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    "python_requires": ">=3.8",
    "install_requires": [
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
        "joblib>=1.0.0",
        "category-encoders>=2.3.0",
    ],
    "extras_require": {
        "visualization": ["matplotlib>=3.3.0", "seaborn>=0.11.0", "plotly>=5.0.0"],
        "interactive": [
            "dash>=2.0.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
    },
}

# Expose metadata for setup.py or other build tools
__metadata__ = metadata

# Welcome message when importing
if _CORE_AVAILABLE and _UTILS_AVAILABLE and _VISUALIZATION_AVAILABLE:
    logger = logging.getLogger(__name__)
    logger.info(f"PSOD v{__version__} loaded successfully with all modules.")
elif not all([_CORE_AVAILABLE, _UTILS_AVAILABLE, _VISUALIZATION_AVAILABLE]):
    missing_modules = []
    if not _CORE_AVAILABLE:
        missing_modules.append("core")
    if not _UTILS_AVAILABLE:
        missing_modules.append("utils")
    if not _VISUALIZATION_AVAILABLE:
        missing_modules.append("visualization")

    logger = logging.getLogger(__name__)
    logger.warning(
        f"PSOD v{__version__} loaded with missing modules: {missing_modules}"
    )
