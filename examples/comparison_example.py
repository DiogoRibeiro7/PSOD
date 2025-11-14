"""
Comparison of PSOD with other outlier detection methods.

This module demonstrates:
- Comparison with Isolation Forest
- Comparison with Local Outlier Factor (LOF)
- Comparison with One-Class SVM
- Comparison with Elliptic Envelope
- Performance benchmarking
- Visual comparison of results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
import time
import warnings

warnings.filterwarnings("ignore")

# For development, add parent directory to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psod import PSOD, evaluate_outlier_detection, generate_outlier_data
from psod.visualization import plot_outlier_scores, plot_outliers_scatter, plot_roc_pr_curves


def generate_comparison_datasets():
    """Generate various datasets for comparison."""
    datasets = {}

    np.random.seed(42)

    # Dataset 1: Simple Gaussian with outliers
    n_samples = 300
    n_outliers = 30
    normal_data = np.random.randn(n_samples - n_outliers, 5)
    outliers = np.random.uniform(-5, 5, (n_outliers, 5))
    X1 = np.vstack([normal_data, outliers])
    y1 = np.array([0] * (n_samples - n_outliers) + [1] * n_outliers)
    datasets["Gaussian"] = (X1, y1)

    # Dataset 2: Clustered data with outliers
    from sklearn.datasets import make_blobs

    X2, _ = make_blobs(n_samples=270, n_features=5, centers=3, cluster_std=1.0, random_state=42)
    outliers2 = np.random.uniform(-10, 10, (30, 5))
    X2 = np.vstack([X2, outliers2])
    y2 = np.array([0] * 270 + [1] * 30)
    datasets["Clustered"] = (X2, y2)

    # Dataset 3: High-dimensional data
    n_samples = 250
    n_outliers = 25
    n_features = 20
    normal_data = np.random.randn(n_samples - n_outliers, n_features)
    outliers = np.random.uniform(-4, 4, (n_outliers, n_features))
    X3 = np.vstack([normal_data, outliers])
    y3 = np.array([0] * (n_samples - n_outliers) + [1] * n_outliers)
    datasets["High-Dimensional"] = (X3, y3)

    # Dataset 4: Imbalanced (few outliers)
    n_samples = 500
    n_outliers = 10
    normal_data = np.random.randn(n_samples - n_outliers, 5)
    outliers = np.random.uniform(-6, 6, (n_outliers, 5))
    X4 = np.vstack([normal_data, outliers])
    y4 = np.array([0] * (n_samples - n_outliers) + [1] * n_outliers)
    datasets["Imbalanced"] = (X4, y4)

    return datasets


def compare_all_methods():
    """Compare PSOD with other popular outlier detection methods."""
    print("=== Comprehensive Method Comparison ===\n")

    # Generate datasets
    datasets = generate_comparison_datasets()

    # Define methods
    methods = {
        "PSOD": lambda contamination: PSOD(
            min_cols_chosen=0.5,
            max_cols_chosen=1.0,
            stdevs_to_outlier=2.0,
            contamination=contamination,
            random_seed=42,
        ),
        "Isolation Forest": lambda contamination: IsolationForest(
            contamination=contamination, random_state=42, n_jobs=-1
        ),
        "LOF": lambda contamination: LocalOutlierFactor(
            contamination=contamination, novelty=True, n_jobs=-1
        ),
        "One-Class SVM": lambda contamination: OneClassSVM(nu=contamination, gamma="auto"),
        "Elliptic Envelope": lambda contamination: EllipticEnvelope(
            contamination=contamination, random_state=42
        ),
    }

    # Store results
    all_results = {}

    # Compare on each dataset
    for dataset_name, (X, y_true) in datasets.items():
        print(f"\nDataset: {dataset_name}")
        print(f"  Shape: {X.shape}")
        print(f"  Outliers: {sum(y_true)} ({100 * sum(y_true) / len(y_true):.1f}%)")

        contamination = sum(y_true) / len(y_true)
        dataset_results = {}

        for method_name, method_factory in methods.items():
            print(f"\n  Testing {method_name}...")

            try:
                # Create detector
                detector = method_factory(contamination)

                # Measure time
                start_time = time.time()

                if method_name == "PSOD":
                    # PSOD uses pandas DataFrame
                    df = pd.DataFrame(X)
                    scores = detector.fit_predict(df, return_class=False)
                    labels = detector.fit_predict(df, return_class=True)
                elif method_name == "LOF":
                    # LOF with novelty=True requires fit then predict
                    detector.fit(X)
                    scores = -detector.score_samples(X)  # Negative to make outliers positive
                    labels = detector.predict(X)
                    labels = np.where(labels == -1, 1, 0)  # Convert -1/1 to 0/1
                else:
                    # Sklearn methods
                    detector.fit(X)
                    labels = detector.predict(X)
                    labels = np.where(labels == -1, 1, 0)  # Convert -1/1 to 0/1

                    if hasattr(detector, "score_samples"):
                        scores = -detector.score_samples(X)  # Negative to make outliers positive
                    elif hasattr(detector, "decision_function"):
                        scores = -detector.decision_function(X)
                    else:
                        scores = labels.astype(float)

                elapsed_time = time.time() - start_time

                # Evaluate
                metrics = evaluate_outlier_detection(y_true, labels, scores)
                metrics["time"] = elapsed_time

                dataset_results[method_name] = {
                    "scores": scores,
                    "labels": labels,
                    "metrics": metrics,
                }

                print(f"    Precision: {metrics['precision']:.3f}")
                print(f"    Recall: {metrics['recall']:.3f}")
                print(f"    F1-Score: {metrics['f1']:.3f}")
                print(f"    ROC-AUC: {metrics['roc_auc']:.3f}")
                print(f"    Time: {elapsed_time:.3f}s")

            except Exception as e:
                print(f"    Error: {e}")
                dataset_results[method_name] = None

        all_results[dataset_name] = dataset_results

    return all_results, datasets


def visualize_comparison_results(all_results, datasets):
    """Visualize comparison results."""
    print("\n\nGenerating comparison visualizations...\n")

    # Create summary metrics table
    summary_data = []

    for dataset_name, results in all_results.items():
        for method_name, result in results.items():
            if result is not None:
                summary_data.append(
                    {
                        "Dataset": dataset_name,
                        "Method": method_name,
                        "Precision": result["metrics"]["precision"],
                        "Recall": result["metrics"]["recall"],
                        "F1-Score": result["metrics"]["f1"],
                        "ROC-AUC": result["metrics"]["roc_auc"],
                        "Time (s)": result["metrics"]["time"],
                    }
                )

    summary_df = pd.DataFrame(summary_data)

    # Print summary table
    print("Performance Summary:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")

    # Visualize metrics comparison
    metrics_to_plot = ["F1-Score", "ROC-AUC", "Precision", "Recall"]
    n_datasets = len(all_results)
    n_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(n_metrics, n_datasets, figsize=(5 * n_datasets, 4 * n_metrics))

    if n_datasets == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, metric in enumerate(metrics_to_plot):
        for col_idx, (dataset_name, results) in enumerate(all_results.items()):
            ax = axes[row_idx, col_idx]

            methods = []
            values = []

            for method_name, result in results.items():
                if result is not None:
                    methods.append(method_name)
                    values.append(result["metrics"][metric.lower().replace("-", "_")])

            # Create bar plot
            bars = ax.bar(range(len(methods)), values, color="steelblue", alpha=0.7)

            # Highlight PSOD
            if "PSOD" in methods:
                psod_idx = methods.index("PSOD")
                bars[psod_idx].set_color("orange")
                bars[psod_idx].set_alpha(1.0)

            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha="right")
            ax.set_ylabel(metric, fontsize=10)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis="y")

            if row_idx == 0:
                ax.set_title(dataset_name, fontsize=12, fontweight="bold")

            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.02,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.savefig("comparison_metrics.png", dpi=150, bbox_inches="tight")
    print("Saved: comparison_metrics.png\n")

    # Plot execution time comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    dataset_names = list(all_results.keys())
    method_names = list(all_results[dataset_names[0]].keys())

    x = np.arange(len(dataset_names))
    width = 0.15

    for idx, method_name in enumerate(method_names):
        times = []
        for dataset_name in dataset_names:
            result = all_results[dataset_name].get(method_name)
            if result is not None:
                times.append(result["metrics"]["time"])
            else:
                times.append(0)

        offset = width * (idx - len(method_names) / 2)
        ax.bar(x + offset, times, width, label=method_name, alpha=0.8)

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Execution Time (seconds)", fontsize=12)
    ax.set_title("Execution Time Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("comparison_time.png", dpi=150, bbox_inches="tight")
    print("Saved: comparison_time.png\n")


def visual_comparison_2d():
    """Visual comparison on 2D data for better interpretability."""
    print("=== Visual Comparison on 2D Data ===\n")

    # Generate 2D dataset
    np.random.seed(42)
    n_samples = 200
    n_outliers = 20

    # Create two clusters
    cluster1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    cluster2 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    normal_data = np.vstack([cluster1, cluster2])

    # Add outliers
    outliers = np.random.uniform(-6, 6, (n_outliers, 2))

    X = np.vstack([normal_data, outliers])
    y_true = np.array([0] * n_samples + [1] * n_outliers)

    print(f"Dataset shape: {X.shape}")
    print(f"Outliers: {n_outliers}\n")

    # Define methods
    contamination = n_outliers / len(X)

    methods = {
        "PSOD": PSOD(
            min_cols_chosen=0.5,
            max_cols_chosen=1.0,
            stdevs_to_outlier=2.0,
            contamination=contamination,
            random_seed=42,
        ),
        "Isolation Forest": IsolationForest(contamination=contamination, random_state=42),
        "LOF": LocalOutlierFactor(contamination=contamination, novelty=True),
        "One-Class SVM": OneClassSVM(nu=contamination, gamma="auto"),
    }

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (method_name, detector) in enumerate(methods.items()):
        print(f"Visualizing {method_name}...")

        # Fit and predict
        if method_name == "PSOD":
            df = pd.DataFrame(X, columns=["feature1", "feature2"])
            scores = detector.fit_predict(df, return_class=False)
            labels = detector.fit_predict(df, return_class=True)
        elif method_name == "LOF":
            detector.fit(X)
            scores = -detector.score_samples(X)
            labels = detector.predict(X)
            labels = np.where(labels == -1, 1, 0)
        else:
            detector.fit(X)
            labels = detector.predict(X)
            labels = np.where(labels == -1, 1, 0)

            if hasattr(detector, "score_samples"):
                scores = -detector.score_samples(X)
            elif hasattr(detector, "decision_function"):
                scores = -detector.decision_function(X)
            else:
                scores = labels.astype(float)

        # Evaluate
        metrics = evaluate_outlier_detection(y_true, labels, scores)

        # Plot
        ax = axes[idx]
        plot_outliers_scatter(
            X,
            labels,
            scores,
            feature_names=["Feature 1", "Feature 2"],
            ax=ax,
            title=f'{method_name}\nF1={metrics["f1"]:.3f}, AUC={metrics["roc_auc"]:.3f}',
        )

        print(f"  F1-Score: {metrics['f1']:.3f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.3f}\n")

    plt.tight_layout()
    plt.savefig("comparison_visual_2d.png", dpi=150, bbox_inches="tight")
    print("Saved: comparison_visual_2d.png\n")


def roc_pr_comparison():
    """Compare methods using ROC and PR curves."""
    print("=== ROC and PR Curves Comparison ===\n")

    # Generate dataset
    np.random.seed(42)
    n_samples = 300
    n_outliers = 30

    normal_data = np.random.randn(n_samples - n_outliers, 8)
    outliers = np.random.uniform(-5, 5, (n_outliers, 8))

    X = np.vstack([normal_data, outliers])
    y_true = np.array([0] * (n_samples - n_outliers) + [1] * n_outliers)

    print(f"Dataset shape: {X.shape}")
    print(f"Outliers: {n_outliers}\n")

    # Define methods
    contamination = n_outliers / len(X)

    methods_scores = {}

    # PSOD
    print("Running PSOD...")
    detector = PSOD(min_cols_chosen=0.5, max_cols_chosen=1.0, stdevs_to_outlier=2.0, random_seed=42)
    df = pd.DataFrame(X)
    methods_scores["PSOD"] = detector.fit_predict(df, return_class=False)

    # Isolation Forest
    print("Running Isolation Forest...")
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(X)
    methods_scores["Isolation Forest"] = -iso_forest.score_samples(X)

    # LOF
    print("Running LOF...")
    lof = LocalOutlierFactor(contamination=contamination, novelty=True)
    lof.fit(X)
    methods_scores["LOF"] = -lof.score_samples(X)

    # One-Class SVM
    print("Running One-Class SVM...")
    svm = OneClassSVM(nu=contamination, gamma="auto")
    svm.fit(X)
    methods_scores["One-Class SVM"] = -svm.decision_function(X)

    print("\nGenerating ROC and PR curves...\n")

    # Plot ROC and PR curves
    fig = plot_roc_pr_curves(y_true, methods_scores, figsize=(14, 6))

    plt.savefig("comparison_roc_pr_curves.png", dpi=150, bbox_inches="tight")
    print("Saved: comparison_roc_pr_curves.png\n")


def scalability_comparison():
    """Compare scalability of methods with increasing data size."""
    print("=== Scalability Comparison ===\n")

    sample_sizes = [100, 500, 1000, 2000, 5000]
    n_features = 10

    results = {"PSOD": [], "Isolation Forest": [], "LOF": [], "One-Class SVM": []}

    for n_samples in sample_sizes:
        print(f"Testing with {n_samples} samples...")

        # Generate data
        np.random.seed(42)
        n_outliers = int(n_samples * 0.1)
        normal_data = np.random.randn(n_samples - n_outliers, n_features)
        outliers = np.random.uniform(-5, 5, (n_outliers, n_features))
        X = np.vstack([normal_data, outliers])

        contamination = n_outliers / n_samples

        # Test each method
        for method_name in results.keys():
            try:
                start_time = time.time()

                if method_name == "PSOD":
                    detector = PSOD(
                        min_cols_chosen=0.5,
                        max_cols_chosen=1.0,
                        stdevs_to_outlier=2.0,
                        random_seed=42,
                    )
                    df = pd.DataFrame(X)
                    detector.fit_predict(df, return_class=True)

                elif method_name == "Isolation Forest":
                    detector = IsolationForest(
                        contamination=contamination, random_state=42, n_jobs=-1
                    )
                    detector.fit(X)
                    detector.predict(X)

                elif method_name == "LOF":
                    detector = LocalOutlierFactor(
                        contamination=contamination, novelty=True, n_jobs=-1
                    )
                    detector.fit(X)
                    detector.predict(X)

                elif method_name == "One-Class SVM":
                    detector = OneClassSVM(nu=contamination, gamma="auto")
                    detector.fit(X)
                    detector.predict(X)

                elapsed_time = time.time() - start_time
                results[method_name].append(elapsed_time)

                print(f"  {method_name}: {elapsed_time:.3f}s")

            except Exception as e:
                print(f"  {method_name}: Failed ({e})")
                results[method_name].append(None)

        print()

    # Plot scalability
    fig, ax = plt.subplots(figsize=(12, 6))

    for method_name, times in results.items():
        # Filter out None values
        valid_sizes = [size for size, time in zip(sample_sizes, times) if time is not None]
        valid_times = [time for time in times if time is not None]

        ax.plot(valid_sizes, valid_times, marker="o", linewidth=2, markersize=8, label=method_name)

    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_ylabel("Execution Time (seconds)", fontsize=12)
    ax.set_title("Scalability Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig("comparison_scalability.png", dpi=150, bbox_inches="tight")
    print("Saved: comparison_scalability.png\n")


if __name__ == "__main__":
    print("=" * 60)
    print("PSOD vs. Other Methods Comparison")
    print("=" * 60 + "\n")

    try:
        # Run comprehensive comparison
        all_results, datasets = compare_all_methods()
        visualize_comparison_results(all_results, datasets)

        print("\n" + "=" * 60 + "\n")

        # Visual comparison
        visual_comparison_2d()

        print("\n" + "=" * 60 + "\n")

        # ROC/PR curves
        roc_pr_comparison()

        print("\n" + "=" * 60 + "\n")

        # Scalability
        scalability_comparison()

        print("\n" + "=" * 60)
        print("Comparison examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()
