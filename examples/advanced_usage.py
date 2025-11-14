"""
Advanced usage examples for PSOD outlier detection.

This module demonstrates:
- Custom base learners (RandomForest, XGBoost)
- Different transformation algorithms
- Cross-validation for threshold selection
- Feature importance analysis
- Parameter tuning and optimization
- Model persistence (save/load)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# For development, add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from psod import PSOD, save_model, load_model, evaluate_outlier_detection, compute_feature_importance
from psod.visualization import (
    plot_outlier_scores,
    plot_feature_contributions,
    create_outlier_dashboard,
    plot_roc_pr_curves
)


def custom_base_learner_example():
    """Example using different base learners."""
    print("=== Custom Base Learner Example ===\n")

    # Generate dataset with outliers
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    n_outliers = 10

    # Normal data
    normal_data = np.random.randn(n_samples - n_outliers, n_features)

    # Outliers
    outliers = np.random.uniform(-5, 5, (n_outliers, n_features))

    # Combine
    X = np.vstack([normal_data, outliers])
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])

    # Ground truth labels (1 for outlier, 0 for normal)
    y_true = np.array([0] * (n_samples - n_outliers) + [1] * n_outliers)

    print(f"Dataset shape: {df.shape}")
    print(f"Number of outliers: {n_outliers}\n")

    # Compare different base learners
    base_learners = {
        'Linear Regression': None,  # Default
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
    }

    results = {}

    for name, learner in base_learners.items():
        print(f"Testing {name}...")

        # Initialize PSOD with custom learner
        if learner is None:
            detector = PSOD(
                min_cols_chosen=0.5,
                max_cols_chosen=1.0,
                stdevs_to_outlier=2.0,
                random_seed=42
            )
        else:
            detector = PSOD(
                base_learner=type(learner),
                min_cols_chosen=0.5,
                max_cols_chosen=1.0,
                stdevs_to_outlier=2.0,
                random_seed=42
            )

        # Detect outliers
        scores = detector.fit_predict(df, return_class=False)
        labels = detector.fit_predict(df, return_class=True)

        # Evaluate
        metrics = evaluate_outlier_detection(y_true, labels, scores)
        results[name] = {
            'scores': scores,
            'labels': labels,
            'metrics': metrics
        }

        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1-Score: {metrics['f1']:.3f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.3f}\n")

    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, result) in enumerate(results.items()):
        plot_outlier_scores(
            result['scores'],
            result['labels'],
            ax=axes[idx],
            title=f'{name}\nF1={result["metrics"]["f1"]:.3f}, AUC={result["metrics"]["roc_auc"]:.3f}'
        )

    plt.tight_layout()
    plt.savefig('advanced_base_learner_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: advanced_base_learner_comparison.png\n")

    return results


def transformation_algorithms_example():
    """Example comparing different transformation algorithms."""
    print("=== Transformation Algorithms Example ===\n")

    # Generate skewed data
    np.random.seed(42)
    n_samples = 150
    n_outliers = 10

    # Log-normal distributed data (right-skewed)
    normal_data = np.random.lognormal(mean=0, sigma=0.5, size=(n_samples - n_outliers, 5))
    outliers = np.random.uniform(10, 15, (n_outliers, 5))

    X = np.vstack([normal_data, outliers])
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])

    print(f"Dataset shape: {df.shape}")
    print(f"Data statistics:\n{df.describe()}\n")

    # Test different transformations
    transformations = [
        "logarithmic",
        "yeo-johnson",
        "quantile",
        "box-cox",
        None  # No transformation
    ]

    results = {}

    for transform in transformations:
        transform_name = transform if transform else "none"
        print(f"Testing transformation: {transform_name}...")

        detector = PSOD(
            transform_algorithm=transform,
            min_cols_chosen=0.5,
            max_cols_chosen=1.0,
            stdevs_to_outlier=2.0,
            random_seed=42
        )

        scores = detector.fit_predict(df, return_class=False)
        labels = detector.fit_predict(df, return_class=True)

        results[transform_name] = {
            'scores': scores,
            'labels': labels,
            'n_outliers': sum(labels)
        }

        print(f"  Outliers detected: {sum(labels)}\n")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (name, result) in enumerate(results.items()):
        if idx < len(axes):
            plot_outlier_scores(
                result['scores'],
                result['labels'],
                ax=axes[idx],
                title=f'Transform: {name}\nOutliers: {result["n_outliers"]}'
            )

    # Remove empty subplot
    if len(results) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig('advanced_transformation_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: advanced_transformation_comparison.png\n")

    return results


def parameter_tuning_example():
    """Example demonstrating parameter tuning."""
    print("=== Parameter Tuning Example ===\n")

    # Generate dataset
    np.random.seed(42)
    n_samples = 200
    n_outliers = 15

    normal_data = np.random.randn(n_samples - n_outliers, 8)
    outliers = np.random.uniform(-4, 4, (n_outliers, 8))

    X = np.vstack([normal_data, outliers])
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(8)])

    y_true = np.array([0] * (n_samples - n_outliers) + [1] * n_outliers)

    print(f"Dataset shape: {df.shape}\n")

    # Grid search over parameters
    stdev_values = [1.5, 2.0, 2.5, 3.0]
    min_cols_values = [0.3, 0.5, 0.7]

    print("Tuning stdevs_to_outlier parameter...")
    best_f1 = 0
    best_stdev = None

    stdev_results = []

    for stdev in stdev_values:
        detector = PSOD(
            stdevs_to_outlier=stdev,
            min_cols_chosen=0.5,
            max_cols_chosen=1.0,
            random_seed=42
        )

        scores = detector.fit_predict(df, return_class=False)
        labels = detector.fit_predict(df, return_class=True)

        metrics = evaluate_outlier_detection(y_true, labels, scores)
        stdev_results.append({
            'stdev': stdev,
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        })

        print(f"  stdev={stdev:.1f}: F1={metrics['f1']:.3f}, "
              f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_stdev = stdev

    print(f"\nBest stdevs_to_outlier: {best_stdev} (F1={best_f1:.3f})\n")

    print("Tuning min_cols_chosen parameter...")
    best_f1 = 0
    best_min_cols = None

    cols_results = []

    for min_cols in min_cols_values:
        detector = PSOD(
            stdevs_to_outlier=best_stdev,
            min_cols_chosen=min_cols,
            max_cols_chosen=1.0,
            random_seed=42
        )

        scores = detector.fit_predict(df, return_class=False)
        labels = detector.fit_predict(df, return_class=True)

        metrics = evaluate_outlier_detection(y_true, labels, scores)
        cols_results.append({
            'min_cols': min_cols,
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        })

        print(f"  min_cols={min_cols:.1f}: F1={metrics['f1']:.3f}, "
              f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_min_cols = min_cols

    print(f"\nBest min_cols_chosen: {best_min_cols} (F1={best_f1:.3f})\n")

    # Visualize tuning results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot stdev tuning
    stdev_df = pd.DataFrame(stdev_results)
    ax1.plot(stdev_df['stdev'], stdev_df['f1'], 'o-', label='F1', linewidth=2, markersize=8)
    ax1.plot(stdev_df['stdev'], stdev_df['precision'], 's-', label='Precision', linewidth=2, markersize=8)
    ax1.plot(stdev_df['stdev'], stdev_df['recall'], '^-', label='Recall', linewidth=2, markersize=8)
    ax1.set_xlabel('stdevs_to_outlier', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Parameter Tuning: stdevs_to_outlier', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot min_cols tuning
    cols_df = pd.DataFrame(cols_results)
    ax2.plot(cols_df['min_cols'], cols_df['f1'], 'o-', label='F1', linewidth=2, markersize=8)
    ax2.plot(cols_df['min_cols'], cols_df['precision'], 's-', label='Precision', linewidth=2, markersize=8)
    ax2.plot(cols_df['min_cols'], cols_df['recall'], '^-', label='Recall', linewidth=2, markersize=8)
    ax2.set_xlabel('min_cols_chosen', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Parameter Tuning: min_cols_chosen', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('advanced_parameter_tuning.png', dpi=150, bbox_inches='tight')
    print("Saved: advanced_parameter_tuning.png\n")

    return best_stdev, best_min_cols


def feature_importance_example():
    """Example demonstrating feature importance analysis."""
    print("=== Feature Importance Example ===\n")

    # Generate dataset where outliers differ mainly in specific features
    np.random.seed(42)
    n_samples = 150
    n_outliers = 10

    # Normal data
    normal_data = np.random.randn(n_samples - n_outliers, 6)

    # Outliers - make them anomalous mainly in features 0, 1, and 2
    outliers = np.random.randn(n_outliers, 6)
    outliers[:, 0] += 5  # Strong anomaly in feature 0
    outliers[:, 1] += 4  # Strong anomaly in feature 1
    outliers[:, 2] += 3  # Moderate anomaly in feature 2

    X = np.vstack([normal_data, outliers])
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(6)])

    print(f"Dataset shape: {df.shape}")
    print("Features 0-2 contain the outlier signal\n")

    # Detect outliers
    detector = PSOD(
        min_cols_chosen=0.5,
        max_cols_chosen=1.0,
        stdevs_to_outlier=2.0,
        random_seed=42
    )

    scores = detector.fit_predict(df, return_class=False)
    labels = detector.fit_predict(df, return_class=True)

    print(f"Outliers detected: {sum(labels)}\n")

    # Compute feature importance
    feature_importance = compute_feature_importance(detector, df)

    print("Feature Importance Scores:")
    for idx, (feature, importance) in enumerate(feature_importance.items()):
        print(f"  {feature}: {importance:.4f}")

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot feature importance
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())

    ax1.barh(features, importances, color='steelblue')
    ax1.set_xlabel('Importance Score', fontsize=12)
    ax1.set_title('Feature Importance for Outlier Detection', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='x')

    # Plot outlier scores
    plot_outlier_scores(scores, labels, ax=ax2)

    plt.tight_layout()
    plt.savefig('advanced_feature_importance.png', dpi=150, bbox_inches='tight')
    print("\nSaved: advanced_feature_importance.png\n")

    return feature_importance


def model_persistence_example():
    """Example demonstrating model saving and loading."""
    print("=== Model Persistence Example ===\n")

    # Generate dataset
    np.random.seed(42)
    n_samples = 100
    normal_data = np.random.randn(n_samples, 5)
    df_train = pd.DataFrame(normal_data, columns=[f'feature_{i}' for i in range(5)])

    print(f"Training dataset shape: {df_train.shape}")

    # Train detector
    print("Training PSOD detector...")
    detector = PSOD(
        min_cols_chosen=0.5,
        max_cols_chosen=1.0,
        stdevs_to_outlier=2.0,
        random_seed=42
    )

    detector.fit(df_train)
    print("Training complete\n")

    # Save model
    model_path = 'psod_model.pkl'
    save_model(detector, model_path)
    print(f"Model saved to: {model_path}\n")

    # Load model
    print("Loading saved model...")
    loaded_detector = load_model(model_path)
    print("Model loaded successfully\n")

    # Test on new data
    test_normal = np.random.randn(50, 5)
    test_outliers = np.random.uniform(-5, 5, (5, 5))
    test_data = np.vstack([test_normal, test_outliers])
    df_test = pd.DataFrame(test_data, columns=[f'feature_{i}' for i in range(5)])

    print(f"Test dataset shape: {df_test.shape}")

    # Compare original and loaded models
    scores_original = detector.predict(df_test, return_class=False)
    scores_loaded = loaded_detector.predict(df_test, return_class=False)

    print("\nComparing predictions:")
    print(f"  Max difference: {np.max(np.abs(scores_original - scores_loaded)):.10f}")
    print(f"  Models match: {np.allclose(scores_original, scores_loaded)}")

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    labels_original = detector.predict(df_test, return_class=True)
    labels_loaded = loaded_detector.predict(df_test, return_class=True)

    plot_outlier_scores(scores_original, labels_original, ax=ax1, title='Original Model')
    plot_outlier_scores(scores_loaded, labels_loaded, ax=ax2, title='Loaded Model')

    plt.tight_layout()
    plt.savefig('advanced_model_persistence.png', dpi=150, bbox_inches='tight')
    print("\nSaved: advanced_model_persistence.png\n")

    return detector, loaded_detector


def comprehensive_dashboard_example():
    """Example creating a comprehensive outlier detection dashboard."""
    print("=== Comprehensive Dashboard Example ===\n")

    # Generate dataset
    np.random.seed(42)
    n_samples = 200
    n_outliers = 15

    normal_data = np.random.randn(n_samples - n_outliers, 6)
    outliers = np.random.uniform(-4, 4, (n_outliers, 6))

    X = np.vstack([normal_data, outliers])
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(6)])

    y_true = np.array([0] * (n_samples - n_outliers) + [1] * n_outliers)

    print(f"Dataset shape: {df.shape}")
    print(f"True outliers: {n_outliers}\n")

    # Detect outliers
    detector = PSOD(
        min_cols_chosen=0.5,
        max_cols_chosen=1.0,
        stdevs_to_outlier=2.0,
        random_seed=42
    )

    scores = detector.fit_predict(df, return_class=False)
    labels = detector.fit_predict(df, return_class=True)

    print(f"Detected outliers: {sum(labels)}\n")

    # Create comprehensive dashboard
    print("Creating comprehensive dashboard...")

    try:
        fig = create_outlier_dashboard(
            df,
            scores,
            labels,
            feature_names=df.columns.tolist(),
            y_true=y_true
        )
        plt.savefig('advanced_comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
        print("Saved: advanced_comprehensive_dashboard.png\n")
    except Exception as e:
        print(f"Could not create dashboard: {e}\n")

    return detector, scores, labels


if __name__ == "__main__":
    print("=" * 60)
    print("PSOD Advanced Usage Examples")
    print("=" * 60 + "\n")

    # Run all examples
    try:
        custom_base_learner_example()
        print("\n" + "=" * 60 + "\n")

        transformation_algorithms_example()
        print("\n" + "=" * 60 + "\n")

        parameter_tuning_example()
        print("\n" + "=" * 60 + "\n")

        feature_importance_example()
        print("\n" + "=" * 60 + "\n")

        model_persistence_example()
        print("\n" + "=" * 60 + "\n")

        comprehensive_dashboard_example()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
