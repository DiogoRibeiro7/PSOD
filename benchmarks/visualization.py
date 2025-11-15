"""
Visualization utilities for benchmark results.
Creates comprehensive plots and charts for performance analysis.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


def plot_method_comparison(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create bar plot comparing methods on a metric.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results
    metric : str
        Metric to plot
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Aggregate by method
    summary = results_df.groupby("method")[metric].agg(["mean", "std"]).reset_index()
    summary = summary.sort_values("mean", ascending=False)

    # Create bar plot
    bars = ax.bar(
        summary["method"],
        summary["mean"],
        yerr=summary["std"],
        capsize=5,
        alpha=0.7,
        color=sns.color_palette("viridis", len(summary)),
    )

    # Highlight best method
    best_idx = int(summary["mean"].idxmax())
    bars[best_idx].set_color("red")
    bars[best_idx].set_alpha(1.0)

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title or f"Method Comparison - {metric.upper()}", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_performance_vs_time(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    time_metric: str = "total_time",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create scatter plot of performance vs computation time.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results
    metric : str
        Performance metric
    time_metric : str
        Time metric ('train_time', 'pred_time', or 'total_time')
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Aggregate by method
    summary = results_df.groupby("method")[[metric, time_metric]].mean().reset_index()

    # Create scatter plot
    for idx, row in summary.iterrows():
        ax.scatter(row[time_metric], row[metric], s=200, alpha=0.7, label=str(row["method"]))
        # Add method name
        ax.annotate(
            str(row["method"]),
            (float(row[time_metric]), float(row[metric])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel(f'{time_metric.replace("_", " ").title()} (seconds)', fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title("Performance vs Computation Time", fontsize=14, fontweight="bold")

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_multi_metric_comparison(
    results_df: pd.DataFrame, metrics: List[str], save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create heatmap comparing methods across multiple metrics.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results
    metrics : List[str]
        List of metrics to compare
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Aggregate by method
    summary = results_df.groupby("method")[metrics].mean()

    # Normalize to [0, 1] for each metric
    normalized = (summary - summary.min()) / (summary.max() - summary.min())

    # Create heatmap
    sns.heatmap(
        normalized.T,
        annot=summary.T,
        fmt=".3f",
        cmap="RdYlGn",
        center=0.5,
        ax=ax,
        cbar_kws={"label": "Normalized Score"},
    )

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Metric", fontsize=12)
    ax.set_title("Multi-Metric Method Comparison", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_scalability(
    scalability_results: Dict[str, Dict], save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create scalability plots showing time vs dataset size.

    Parameters
    ----------
    scalability_results : Dict[str, Dict]
        Scalability benchmark results
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Extract data
    methods = list(scalability_results.keys())
    colors = sns.color_palette("tab10", len(methods))

    # Plot 1: Training time vs sample size
    ax = axes[0]
    for method, color in zip(methods, colors):
        data = scalability_results[method]
        ax.plot(
            data["n_samples"],
            data["train_time"],
            marker="o",
            label=method,
            color=color,
            linewidth=2,
        )

    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_ylabel("Training Time (seconds)", fontsize=12)
    ax.set_title("Training Time Scalability", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Plot 2: Prediction time vs sample size
    ax = axes[1]
    for method, color in zip(methods, colors):
        data = scalability_results[method]
        ax.plot(
            data["n_samples"], data["pred_time"], marker="s", label=method, color=color, linewidth=2
        )

    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_ylabel("Prediction Time (seconds)", fontsize=12)
    ax.set_title("Prediction Time Scalability", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_dataset_comparison(
    results_df: pd.DataFrame, metric: str = "roc_auc", save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create grouped bar plot comparing methods across datasets.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results
    metric : str
        Metric to plot
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Pivot data
    pivot = results_df.pivot_table(values=metric, index="dataset", columns="method", aggfunc="mean")

    # Create grouped bar plot
    pivot.plot(kind="bar", ax=ax, width=0.8, alpha=0.8)

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(
        f"Method Performance Across Datasets - {metric.upper()}", fontsize=14, fontweight="bold"
    )
    ax.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curves(roc_data: Dict[str, Dict], save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curves for multiple methods.

    Parameters
    ----------
    roc_data : Dict[str, Dict]
        ROC curve data for each method
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = sns.color_palette("tab10", len(roc_data))

    for (method, data), color in zip(roc_data.items(), colors):
        ax.plot(
            data["fpr"],
            data["tpr"],
            label=f"{method} (AUC = {data['auc']:.3f})",
            color=color,
            linewidth=2,
        )

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_precision_recall_curves(
    pr_data: Dict[str, Dict], save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot precision-recall curves for multiple methods.

    Parameters
    ----------
    pr_data : Dict[str, Dict]
        PR curve data for each method
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = sns.color_palette("tab10", len(pr_data))

    for (method, data), color in zip(pr_data.items(), colors):
        ax.plot(
            data["recall"],
            data["precision"],
            label=f"{method} (AP = {data['auc']:.3f})",
            color=color,
            linewidth=2,
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_memory_usage(
    memory_results: Dict[str, List[float]], save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot memory usage comparison.

    Parameters
    ----------
    memory_results : Dict[str, List[float]]
        Memory usage data for each method
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(memory_results.keys())
    memory_means = [np.mean(memory_results[m]) for m in methods]
    memory_stds = [np.std(memory_results[m]) for m in methods]

    # Sort by mean memory
    sorted_indices = np.argsort(memory_means)
    methods = [methods[i] for i in sorted_indices]
    memory_means = [memory_means[i] for i in sorted_indices]
    memory_stds = [memory_stds[i] for i in sorted_indices]

    bars = ax.barh(
        methods,
        memory_means,
        xerr=memory_stds,
        alpha=0.7,
        color=sns.color_palette("viridis", len(methods)),
    )

    ax.set_xlabel("Memory Usage (MB)", fontsize=12)
    ax.set_ylabel("Method", fontsize=12)
    ax.set_title("Memory Usage Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, value in zip(bars, memory_means):
        ax.text(
            float(value),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f} MB",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_radar_chart(
    results_df: pd.DataFrame,
    metrics: List[str],
    methods: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create radar chart for multi-dimensional method comparison.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results
    metrics : List[str]
        List of metrics to plot
    methods : List[str], optional
        Methods to include (default: all)
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    from math import pi

    import matplotlib.pyplot as plt

    if methods is None:
        methods = results_df["method"].unique()[:5].tolist()  # Limit to 5 methods

    # Aggregate data
    summary = results_df[results_df["method"].isin(methods)].groupby("method")[metrics].mean()

    # Normalize to [0, 1]
    normalized = (summary - summary.min()) / (summary.max() - summary.min())

    # Setup radar chart
    num_vars = len(metrics)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    colors = sns.color_palette("tab10", len(methods))

    for (method, values), color in zip(normalized.iterrows(), colors):
        values = values.tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=method, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title("Multi-Metric Radar Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_benchmark_dashboard(
    results_df: pd.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive benchmark dashboard with multiple plots.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. ROC-AUC comparison
    ax1 = fig.add_subplot(gs[0, 0])
    summary = results_df.groupby("method")["roc_auc"].agg(["mean", "std"]).reset_index()
    summary = summary.sort_values("mean", ascending=False)
    ax1.barh(summary["method"], summary["mean"], xerr=summary["std"], alpha=0.7)
    ax1.set_xlabel("ROC-AUC")
    ax1.set_title("ROC-AUC Comparison", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # 2. Training time comparison
    ax2 = fig.add_subplot(gs[0, 1])
    summary = (
        results_df.groupby("method")["train_time"].mean().reset_index().sort_values("train_time")
    )
    ax2.barh(summary["method"], summary["train_time"], alpha=0.7, color="orange")
    ax2.set_xlabel("Training Time (s)")
    ax2.set_title("Training Time", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # 3. Performance vs Time scatter
    ax3 = fig.add_subplot(gs[0, 2])
    for method in results_df["method"].unique():
        data = results_df[results_df["method"] == method]
        ax3.scatter(
            data["total_time"].mean(), data["roc_auc"].mean(), s=100, alpha=0.7, label=method
        )
    ax3.set_xlabel("Total Time (s)")
    ax3.set_ylabel("ROC-AUC")
    ax3.set_title("Performance vs Time", fontweight="bold")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Multi-metric heatmap
    ax4 = fig.add_subplot(gs[1, :])
    metrics = ["roc_auc", "avg_precision", "precision", "recall", "f1_score"]
    available_metrics = [m for m in metrics if m in results_df.columns]
    if available_metrics:
        summary = results_df.groupby("method")[available_metrics].mean()
        normalized = (summary - summary.min()) / (summary.max() - summary.min())
        sns.heatmap(
            normalized.T,
            annot=summary.T,
            fmt=".3f",
            cmap="RdYlGn",
            center=0.5,
            ax=ax4,
            cbar_kws={"label": "Normalized Score"},
        )
        ax4.set_title("Multi-Metric Heatmap", fontweight="bold")

    # 5. Dataset performance
    if "dataset" in results_df.columns and len(results_df["dataset"].unique()) > 1:
        ax5 = fig.add_subplot(gs[2, :])
        pivot = results_df.pivot_table(values="roc_auc", index="dataset", columns="method")
        pivot.plot(kind="bar", ax=ax5, alpha=0.8)
        ax5.set_ylabel("ROC-AUC")
        ax5.set_title("Performance Across Datasets", fontweight="bold")
        ax5.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax5.grid(True, alpha=0.3, axis="y")
        ax5.tick_params(axis="x", rotation=45)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
