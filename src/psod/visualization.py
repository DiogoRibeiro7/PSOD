"""
Visualization functions for PSOD outlier detection.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Any
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings


def plot_outlier_scores(
    scores: Union[pd.Series, np.ndarray],
    threshold: Optional[float] = None,
    bins: int = 50,
    title: str = "Outlier Scores Distribution",
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Create comprehensive outlier score distribution plot with 4 subplots.

    Parameters
    ----------
    scores : Union[pd.Series, np.ndarray]
        Outlier scores to plot.
    threshold : float, optional
        Threshold line to mark outliers.
    bins : int, default=50
        Number of bins for histogram.
    title : str, default="Outlier Scores Distribution"
        Plot title.
    figsize : Tuple[int, int], default=(12, 8)
        Figure size.

    Returns
    -------
    plt.Figure
        Matplotlib figure object with 4 subplots:
        - Main histogram with colored outlier bins
        - Box plot
        - Q-Q plot against normal distribution
        - Cumulative distribution function
    """
    # Convert to numpy array for consistency
    if isinstance(scores, pd.Series):
        scores_array = scores.values
    else:
        scores_array = np.asarray(scores)

    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    # 1. Main histogram with colored outlier bins
    n, bins_edges, patches = ax1.hist(
        scores_array, bins=bins, alpha=0.7, color="skyblue", edgecolor="black"
    )

    if threshold is not None:
        # Color outlier bars differently
        for i, (patch, left, right) in enumerate(zip(patches, bins_edges[:-1], bins_edges[1:])):
            if right > threshold:
                patch.set_facecolor("red")
                patch.set_alpha(0.8)

        ax1.axvline(
            threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold: {threshold:.3f}"
        )
        outlier_pct = np.sum(scores_array > threshold) / len(scores_array) * 100
        ax1.text(
            0.02,
            0.98,
            f"Outliers: {outlier_pct:.1f}%",
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax1.set_xlabel("Outlier Score")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Score Distribution")
    ax1.grid(True, alpha=0.3)
    if threshold is not None:
        ax1.legend()

    # 2. Box plot
    bp = ax2.boxplot(
        scores_array,
        vert=True,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops=dict(
            marker="D", markerfacecolor="green", markersize=8, markeredgecolor="darkgreen"
        ),
    )
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][0].set_edgecolor("black")
    bp["medians"][0].set_color("red")
    bp["medians"][0].set_linewidth(2)

    if threshold is not None:
        ax2.axhline(
            threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold: {threshold:.3f}"
        )
        ax2.legend()

    ax2.set_ylabel("Outlier Score")
    ax2.set_title("Box Plot")
    ax2.set_xticklabels(["Scores"])
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Q-Q plot against normal distribution
    from scipy import stats

    stats.probplot(scores_array, dist="norm", plot=ax3)
    ax3.set_title("Q-Q Plot vs Normal Distribution")
    ax3.grid(True, alpha=0.3)
    ax3.get_lines()[0].set_markerfacecolor("skyblue")
    ax3.get_lines()[0].set_markeredgecolor("black")
    ax3.get_lines()[0].set_markersize(4)
    ax3.get_lines()[1].set_color("red")
    ax3.get_lines()[1].set_linewidth(2)

    # 4. Cumulative distribution
    sorted_scores = np.sort(scores_array)
    cumulative_pct = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
    ax4.plot(sorted_scores, cumulative_pct, linewidth=2, color="steelblue")

    if threshold is not None:
        threshold_pct = np.sum(scores_array <= threshold) / len(scores_array) * 100
        ax4.axvline(threshold, color="red", linestyle="--", linewidth=2)
        ax4.axhline(threshold_pct, color="red", linestyle="--", linewidth=2)
        ax4.plot(threshold, threshold_pct, "ro", markersize=10, label=f"{threshold_pct:.1f}%")
        ax4.text(
            threshold,
            threshold_pct + 5,
            f"{threshold_pct:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax4.set_xlabel("Outlier Score")
    ax4.set_ylabel("Cumulative Percentage")
    ax4.set_title("Cumulative Distribution Function")
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(sorted_scores[0], sorted_scores[-1])
    ax4.set_ylim(0, 105)
    if threshold is not None:
        ax4.legend()

    # Add summary statistics box
    stats_text = f"""Statistics:
Mean: {np.mean(scores_array):.3f}
Std: {np.std(scores_array):.3f}
Median: {np.median(scores_array):.3f}
IQR: {np.percentile(scores_array, 75) - np.percentile(scores_array, 25):.3f}
Min: {np.min(scores_array):.3f}
Max: {np.max(scores_array):.3f}"""

    fig.text(
        0.02,
        0.02,
        stats_text,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        verticalalignment="bottom",
    )

    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig


def plot_feature_contributions(model, sample_idx: int, top_k: int = 10) -> plt.Figure:
    """
    Plot feature contributions to outlier score for a specific sample.

    Parameters
    ----------
    model : PSOD
        Fitted PSOD model.
    sample_idx : int
        Index of sample to analyze.
    top_k : int, default=10
        Number of top features to show.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    # Extract feature-wise prediction errors
    if not hasattr(model, "feature_names_"):
        feature_names = [f"Feature_{i}" for i in range(model.n_features_)]
    else:
        feature_names = model.feature_names_

    # Calculate feature contributions (this assumes model has prediction errors stored)
    if hasattr(model, "prediction_errors_"):
        contributions = np.abs(model.prediction_errors_[sample_idx])
    else:
        # Fallback: use feature importance if available
        if hasattr(model, "feature_importances_"):
            contributions = model.feature_importances_
        else:
            # Generate mock contributions for demonstration
            warnings.warn(
                "Model doesn't have prediction errors or feature importances. Using random contributions."
            )
            contributions = np.random.rand(len(feature_names))

    # Get top-k features
    top_indices = np.argsort(contributions)[-top_k:][::-1]
    top_contributions = contributions[top_indices]
    top_features = [feature_names[i] for i in top_indices]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [
        "red" if contrib > np.mean(top_contributions) else "skyblue"
        for contrib in top_contributions
    ]

    bars = ax.barh(range(len(top_features)), top_contributions, color=colors)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_contributions)):
        ax.text(
            value + 0.01 * max(top_contributions),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left",
            va="center",
        )

    # Formatting
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel("Contribution to Outlier Score")
    ax.set_title(f"Feature Contributions for Sample {sample_idx}")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig


def plot_outliers_scatter(
    X: pd.DataFrame,
    outlier_labels: np.ndarray,
    features: Optional[List[str]] = None,
    dim: int = 2,
    use_pca: bool = False,
) -> go.Figure:
    """
    Create scatter plot highlighting outliers.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    outlier_labels : np.ndarray
        Binary outlier labels.
    features : List[str], optional
        Features to plot (if None, use PCA).
    dim : int, default=2
        Number of dimensions (2 or 3).
    use_pca : bool, default=False
        Whether to use PCA for dimensionality reduction.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    # Handle dimensionality reduction if needed
    if use_pca or features is None or len(features) < dim:
        pca = PCA(n_components=dim)
        X_plot = pca.fit_transform(X)
        feature_names = [f"PC{i+1}" for i in range(dim)]
        explained_var = pca.explained_variance_ratio_
        title_suffix = f" (PCA: {sum(explained_var):.1%} variance)"
    else:
        X_plot = X[features[:dim]].values
        feature_names = features[:dim]
        title_suffix = ""

    # Create DataFrame for plotting
    plot_data = pd.DataFrame(X_plot, columns=feature_names)
    plot_data["Outlier"] = outlier_labels.astype(bool)
    plot_data["Label"] = ["Outlier" if x else "Normal" for x in outlier_labels]

    # Create scatter plot
    if dim == 2:
        fig = px.scatter(
            plot_data,
            x=feature_names[0],
            y=feature_names[1],
            color="Label",
            color_discrete_map={"Normal": "blue", "Outlier": "red"},
            hover_data={"Outlier": False},
            title=f"2D Scatter Plot of Data Points{title_suffix}",
        )
    elif dim == 3:
        fig = px.scatter_3d(
            plot_data,
            x=feature_names[0],
            y=feature_names[1],
            z=feature_names[2],
            color="Label",
            color_discrete_map={"Normal": "blue", "Outlier": "red"},
            hover_data={"Outlier": False},
            title=f"3D Scatter Plot of Data Points{title_suffix}",
        )
    else:
        raise ValueError("dim must be 2 or 3")

    # Update layout
    fig.update_layout(
        showlegend=True, legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1)
    )

    return fig


def plot_timeseries_outliers(
    data: pd.DataFrame, outlier_labels: np.ndarray, time_column: str, value_columns: List[str]
) -> go.Figure:
    """
    Plot time series data with outliers highlighted.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with time series data.
    outlier_labels : np.ndarray
        Binary outlier labels.
    time_column : str
        Name of time column.
    value_columns : List[str]
        Names of value columns to plot.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    # Create subplots
    fig = sp.make_subplots(
        rows=len(value_columns),
        cols=1,
        shared_xaxes=True,
        subplot_titles=value_columns,
        vertical_spacing=0.05,
    )

    # Plot each value column
    for i, col in enumerate(value_columns):
        row = i + 1

        # Normal points
        normal_mask = outlier_labels == 0
        fig.add_trace(
            go.Scatter(
                x=data[time_column][normal_mask],
                y=data[col][normal_mask],
                mode="markers+lines",
                name=f"{col} (Normal)",
                line=dict(color="blue", width=1),
                marker=dict(size=3),
                showlegend=(i == 0),
            ),
            row=row,
            col=1,
        )

        # Outlier points
        outlier_mask = outlier_labels == 1
        if np.any(outlier_mask):
            fig.add_trace(
                go.Scatter(
                    x=data[time_column][outlier_mask],
                    y=data[col][outlier_mask],
                    mode="markers",
                    name=f"{col} (Outliers)",
                    marker=dict(color="red", size=8, symbol="x"),
                    showlegend=(i == 0),
                ),
                row=row,
                col=1,
            )

    # Update layout
    fig.update_layout(
        height=200 * len(value_columns),
        title="Time Series with Outliers Highlighted",
        showlegend=True,
    )

    # Add range slider to bottom subplot
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))

    return fig


def plot_correlation_heatmap(
    X: pd.DataFrame, outlier_labels: np.ndarray, figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot correlation heatmap with outlier statistics.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    outlier_labels : np.ndarray
        Binary outlier labels.
    figsize : Tuple[int, int], default=(12, 10)
        Figure size.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    # Calculate correlations
    corr_matrix = X.corr()

    # Calculate outlier percentage for each feature
    outlier_pcts = {}
    for col in X.columns:
        # For numerical features, calculate correlation with outlier labels
        outlier_corr = np.corrcoef(X[col], outlier_labels)[0, 1]
        outlier_pcts[col] = outlier_corr

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Main correlation heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax1,
        cbar_kws={"shrink": 0.5},
    )
    ax1.set_title("Feature Correlation Matrix")

    # Outlier correlation heatmap
    outlier_corr_df = pd.DataFrame.from_dict(
        outlier_pcts, orient="index", columns=["Outlier_Correlation"]
    )
    sns.heatmap(
        outlier_corr_df, annot=True, cmap="RdBu_r", center=0, ax=ax2, cbar_kws={"shrink": 0.5}
    )
    ax2.set_title("Feature Correlation with Outliers")
    ax2.set_xlabel("Correlation with Outlier Labels")

    plt.tight_layout()
    return fig


def plot_score_evolution(
    scores_history: List[np.ndarray], labels: Optional[List[str]] = None
) -> go.Figure:
    """
    Plot evolution of outlier scores across iterations or models.

    Parameters
    ----------
    scores_history : List[np.ndarray]
        List of score arrays from different iterations.
    labels : List[str], optional
        Labels for each iteration.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    if labels is None:
        labels = [f"Iteration {i+1}" for i in range(len(scores_history))]

    fig = go.Figure()

    n_samples = len(scores_history[0])
    sample_indices = np.arange(n_samples)

    # Calculate statistics for each iteration
    for i, (scores, label) in enumerate(zip(scores_history, labels)):
        # Add mean score line
        fig.add_trace(
            go.Scatter(
                x=sample_indices,
                y=scores,
                mode="lines+markers",
                name=label,
                line=dict(width=2),
                marker=dict(size=4),
            )
        )

    # Add confidence intervals for the last iteration
    if len(scores_history) > 1:
        last_scores = scores_history[-1]
        mean_scores = np.mean(scores_history, axis=0)
        std_scores = np.std(scores_history, axis=0)

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([sample_indices, sample_indices[::-1]]),
                y=np.concatenate([mean_scores + std_scores, (mean_scores - std_scores)[::-1]]),
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Confidence Interval",
                showlegend=True,
            )
        )

    # Highlight stable vs unstable outliers
    if len(scores_history) > 2:
        # Calculate coefficient of variation for each sample
        cv = np.std(scores_history, axis=0) / np.mean(scores_history, axis=0)
        unstable_mask = cv > np.percentile(cv, 90)  # Top 10% most variable

        if np.any(unstable_mask):
            fig.add_trace(
                go.Scatter(
                    x=sample_indices[unstable_mask],
                    y=scores_history[-1][unstable_mask],
                    mode="markers",
                    name="Unstable Outliers",
                    marker=dict(color="red", size=8, symbol="x"),
                )
            )

    fig.update_layout(
        title="Evolution of Outlier Scores",
        xaxis_title="Sample Index",
        yaxis_title="Outlier Score",
        hovermode="x unified",
    )

    return fig


def plot_roc_pr_curves(
    y_true: np.ndarray, y_scores: np.ndarray, title: str = "ROC and PR Curves"
) -> plt.Figure:
    """
    Plot ROC and Precision-Recall curves.

    Parameters
    ----------
    y_true : np.ndarray
        True outlier labels.
    y_scores : np.ndarray
        Outlier scores.
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Calculate PR curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random baseline")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Receiver Operating Characteristic")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Precision-Recall Curve
    baseline_precision = np.sum(y_true) / len(y_true)
    ax2.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    ax2.axhline(
        y=baseline_precision,
        color="navy",
        linestyle="--",
        lw=2,
        label=f"Random baseline ({baseline_precision:.3f})",
    )
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def create_outlier_dashboard(
    X: pd.DataFrame,
    outlier_scores: np.ndarray,
    outlier_labels: np.ndarray,
    model: Optional[Any] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create comprehensive static dashboard with multiple analysis panels.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    outlier_scores : np.ndarray
        Continuous outlier scores.
    outlier_labels : np.ndarray
        Binary outlier labels.
    model : Any, optional
        Fitted PSOD model (for feature importance).
    save_path : str, optional
        Path to save the dashboard image.

    Returns
    -------
    plt.Figure
        Matplotlib figure object with comprehensive dashboard.
    """
    # Create large figure with 4x4 grid
    fig = plt.figure(figsize=(24, 18))

    # Define grid layout (4 rows x 4 columns)
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    # Determine threshold from labels
    if np.any(outlier_labels):
        threshold = np.min(outlier_scores[outlier_labels == 1])
    else:
        threshold = np.percentile(outlier_scores, 95)

    outlier_mask = outlier_labels == 1
    normal_mask = outlier_labels == 0

    # 1. Score distribution (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    n, bins, patches = ax1.hist(
        outlier_scores, bins=40, alpha=0.7, color="skyblue", edgecolor="black"
    )
    for patch, left, right in zip(patches, bins[:-1], bins[1:]):
        if right > threshold:
            patch.set_facecolor("red")
    ax1.axvline(threshold, color="red", linestyle="--", linewidth=2)
    ax1.set_title("Score Distribution")
    ax1.set_xlabel("Outlier Score")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)

    # 2. Feature importance (top-center)
    ax2 = fig.add_subplot(gs[0, 1])
    if model and hasattr(model, "get_feature_importance"):
        importance_df = model.get_feature_importance()
        top_features = importance_df.head(10)
        ax2.barh(range(len(top_features)), top_features["importance"], color="steelblue")
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features["feature"])
        ax2.set_title("Feature Importance")
        ax2.set_xlabel("Importance Score")
    else:
        # Fallback: use variance as proxy
        numeric_cols = X.select_dtypes(include=[np.number]).columns[:10]
        importances = X[numeric_cols].var().sort_values(ascending=False)
        ax2.barh(range(len(importances)), importances.values, color="steelblue")
        ax2.set_yticks(range(len(importances)))
        ax2.set_yticklabels(importances.index)
        ax2.set_title("Feature Variance")
        ax2.set_xlabel("Variance")
    ax2.grid(True, alpha=0.3, axis="x")

    # 3. PCA projection (top-center-right)
    ax3 = fig.add_subplot(gs[0, 2])
    if X.shape[1] >= 2:
        pca = PCA(n_components=2)
        numeric_data = X.select_dtypes(include=[np.number])
        X_pca = pca.fit_transform(numeric_data)
        colors = ["red" if label else "blue" for label in outlier_labels]
        scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=20)
        ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax3.set_title("PCA Projection")
        ax3.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                label="Normal",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                label="Outlier",
                markersize=8,
            ),
        ]
        ax3.legend(handles=legend_elements, loc="best")

    # 4. Summary statistics (top-right corner)
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis("off")

    n_total = len(outlier_labels)
    n_outliers = np.sum(outlier_labels)
    outlier_rate = n_outliers / n_total * 100

    stats_text = f"""Summary Statistics

Total Samples: {n_total:,}
Outliers: {n_outliers:,} ({outlier_rate:.1f}%)
Normal: {n_total - n_outliers:,} ({100-outlier_rate:.1f}%)

Score Statistics:
Mean: {np.mean(outlier_scores):.4f}
Std: {np.std(outlier_scores):.4f}
Median: {np.median(outlier_scores):.4f}
Min: {np.min(outlier_scores):.4f}
Max: {np.max(outlier_scores):.4f}

Threshold: {threshold:.4f}"""

    ax4.text(
        0.05,
        0.95,
        stats_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    # 5. Feature correlations (second row, left half)
    ax5 = fig.add_subplot(gs[1, :2])
    numeric_df = X.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.iloc[:, :15].corr()  # Limit to first 15 features
        im = ax5.imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        ax5.set_xticks(range(len(corr_matrix.columns)))
        ax5.set_yticks(range(len(corr_matrix.columns)))
        ax5.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=8)
        ax5.set_yticklabels(corr_matrix.columns, fontsize=8)
        ax5.set_title("Feature Correlations")
        plt.colorbar(im, ax=ax5, shrink=0.8)

    # 6. Outlier vs Normal comparison (second row, right half)
    ax6 = fig.add_subplot(gs[1, 2:])
    if np.any(outlier_mask) and np.any(normal_mask):
        numeric_features = X.select_dtypes(include=[np.number]).columns[:8]  # Top 8 features
        normal_means = X[~outlier_labels][numeric_features].mean()
        outlier_means = X[outlier_labels][numeric_features].mean()

        x_pos = np.arange(len(numeric_features))
        width = 0.35

        ax6.bar(
            x_pos - width / 2, normal_means.values, width, label="Normal", alpha=0.8, color="blue"
        )
        ax6.bar(
            x_pos + width / 2, outlier_means.values, width, label="Outliers", alpha=0.8, color="red"
        )

        ax6.set_xlabel("Features")
        ax6.set_ylabel("Mean Value")
        ax6.set_title("Feature Means: Normal vs Outliers")
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(numeric_features, rotation=45, ha="right")
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis="y")

    # 7. Score evolution over sample index (third row, left half)
    ax7 = fig.add_subplot(gs[2, :2])
    colors = ["red" if label else "blue" for label in outlier_labels]
    ax7.scatter(range(len(outlier_scores)), outlier_scores, c=colors, alpha=0.6, s=15)
    ax7.axhline(threshold, color="red", linestyle="--", alpha=0.7, linewidth=2)
    ax7.set_xlabel("Sample Index")
    ax7.set_ylabel("Outlier Score")
    ax7.set_title("Outlier Scores by Sample Index")
    ax7.grid(True, alpha=0.3)

    # 8. Box plots by outlier status (third row, right half)
    ax8 = fig.add_subplot(gs[2, 2:])
    if np.any(outlier_mask) and np.any(normal_mask):
        score_data = [outlier_scores[~outlier_labels], outlier_scores[outlier_labels]]
        bp = ax8.boxplot(score_data, labels=["Normal", "Outliers"], patch_artist=True)
        bp["boxes"][0].set_facecolor("lightblue")
        bp["boxes"][1].set_facecolor("lightcoral")
        ax8.set_ylabel("Outlier Score")
        ax8.set_title("Score Distribution by Class")
        ax8.grid(True, alpha=0.3, axis="y")

    # 9-12. Feature distributions (bottom row)
    representative_features = X.select_dtypes(include=[np.number]).columns[:4]
    for i, col in enumerate(representative_features):
        ax = fig.add_subplot(gs[3, i])
        if np.any(normal_mask):
            ax.hist(
                X.loc[normal_mask, col],
                bins=20,
                alpha=0.6,
                label="Normal",
                color="blue",
                density=True,
            )

        if np.any(outlier_mask):
            ax.hist(
                X.loc[outlier_mask, col],
                bins=20,
                alpha=0.6,
                label="Outliers",
                color="red",
                density=True,
            )

        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"Distribution: {col}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Overall title
    fig.suptitle("PSOD Outlier Detection Dashboard", fontsize=20, y=0.98)

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"Dashboard saved to: {save_path}")

    return fig


def create_interactive_explorer(X: pd.DataFrame, model, port: int = 8050) -> None:
    """
    Create interactive Dash app for exploring outliers.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    model : PSOD
        Fitted PSOD model.
    port : int, default=8050
        Port to run the app.
    """
    try:
        import dash
        from dash import dcc, html, Input, Output
    except ImportError:
        print("Dash is not installed. Please install it using: pip install dash")
        return

    # Get outlier scores and predictions
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        print("Model doesn't have decision_function or predict_proba method")
        return

    # Create Dash application
    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            html.H1("Interactive Outlier Explorer"),
            html.Div(
                [
                    html.Label("Select Features for Visualization:"),
                    dcc.Dropdown(
                        id="feature-dropdown",
                        options=[{"label": col, "value": col} for col in X.columns],
                        value=X.columns[:2].tolist(),
                        multi=True,
                    ),
                ],
                style={"width": "48%", "display": "inline-block"},
            ),
            html.Div(
                [
                    html.Label("Outlier Threshold:"),
                    dcc.Slider(
                        id="threshold-slider",
                        min=np.min(scores),
                        max=np.max(scores),
                        value=np.percentile(scores, 95),
                        step=(np.max(scores) - np.min(scores)) / 100,
                        marks={
                            np.min(scores): f"{np.min(scores):.2f}",
                            np.percentile(scores, 50): f"{np.percentile(scores, 50):.2f}",
                            np.percentile(scores, 95): f"{np.percentile(scores, 95):.2f}",
                            np.max(scores): f"{np.max(scores):.2f}",
                        },
                    ),
                ],
                style={"width": "48%", "float": "right", "display": "inline-block"},
            ),
            dcc.Graph(id="scatter-plot"),
            html.Div([html.H3("Sample Details"), html.Div(id="sample-details")]),
            html.Div(
                [html.Button("Export Results", id="export-button"), html.Div(id="export-status")]
            ),
        ]
    )

    @app.callback(
        Output("scatter-plot", "figure"),
        [Input("feature-dropdown", "value"), Input("threshold-slider", "value")],
    )
    def update_scatter_plot(selected_features, threshold):
        if not selected_features or len(selected_features) < 2:
            return {"data": [], "layout": {"title": "Please select at least 2 features"}}

        # Create outlier labels based on threshold
        outlier_labels = scores > threshold

        # Create scatter plot
        fig = px.scatter(
            x=X[selected_features[0]],
            y=X[selected_features[1]],
            color=["Outlier" if x else "Normal" for x in outlier_labels],
            color_discrete_map={"Normal": "blue", "Outlier": "red"},
            title=f"Outlier Detection: {selected_features[0]} vs {selected_features[1]}",
        )

        return fig

    @app.callback(
        Output("sample-details", "children"),
        [Input("scatter-plot", "clickData"), Input("threshold-slider", "value")],
    )
    def display_sample_details(clickData, threshold):
        if clickData is None:
            return "Click on a point to see sample details"

        point_idx = clickData["points"][0]["pointIndex"]
        sample_score = scores[point_idx]
        is_outlier = sample_score > threshold

        sample_data = X.iloc[point_idx]

        details = html.Div(
            [
                html.P(f"Sample Index: {point_idx}"),
                html.P(f"Outlier Score: {sample_score:.4f}"),
                html.P(f"Is Outlier: {'Yes' if is_outlier else 'No'}"),
                html.H4("Feature Values:"),
                html.Ul([html.Li(f"{col}: {val:.4f}") for col, val in sample_data.items()]),
            ]
        )

        return details

    @app.callback(
        Output("export-status", "children"),
        [Input("export-button", "n_clicks"), Input("threshold-slider", "value")],
    )
    def export_results(n_clicks, threshold):
        if n_clicks is None:
            return ""

        # Export results
        outlier_labels = scores > threshold
        results_df = X.copy()
        results_df["outlier_score"] = scores
        results_df["is_outlier"] = outlier_labels

        results_df.to_csv("/mnt/user-data/outputs/outlier_results.csv", index=False)

        return html.P("Results exported to outlier_results.csv", style={"color": "green"})

    print(f"Starting Dash app on port {port}")
    print(f"Open http://localhost:{port} in your browser")
    app.run_server(debug=True, port=port)


# Additional utility functions
def plot_feature_distributions(
    X: pd.DataFrame,
    outlier_labels: np.ndarray,
    max_features: int = 12,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Plot distributions of features split by outlier/normal labels.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    outlier_labels : np.ndarray
        Binary outlier labels.
    max_features : int, default=12
        Maximum number of features to plot.
    figsize : Tuple[int, int], default=(15, 10)
        Figure size.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    n_features = min(len(X.columns), max_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

    outlier_mask = outlier_labels == 1
    normal_mask = outlier_labels == 0

    for i, col in enumerate(X.columns[:n_features]):
        ax = axes[i]

        if np.any(normal_mask):
            ax.hist(
                X.loc[normal_mask, col],
                bins=20,
                alpha=0.6,
                label="Normal",
                color="blue",
                density=True,
            )

        if np.any(outlier_mask):
            ax.hist(
                X.loc[outlier_mask, col],
                bins=20,
                alpha=0.6,
                label="Outliers",
                color="red",
                density=True,
            )

        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Feature Distributions: Normal vs Outliers")
    plt.tight_layout()
    return fig


def plot_outlier_evolution_heatmap(
    scores_history: List[np.ndarray],
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Create heatmap showing how outlier scores evolve across iterations.

    Parameters
    ----------
    scores_history : List[np.ndarray]
        List of score arrays from different iterations.
    labels : List[str], optional
        Labels for each iteration.
    figsize : Tuple[int, int], default=(12, 8)
        Figure size.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    if labels is None:
        labels = [f"Iteration {i+1}" for i in range(len(scores_history))]

    # Create matrix of scores
    scores_matrix = np.array(scores_history).T

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(scores_matrix, cmap="YlOrRd", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Sample Index")
    ax.set_xlabel("Iteration/Model")
    ax.set_title("Evolution of Outlier Scores Across Iterations")

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Outlier Score")

    plt.tight_layout()
    return fig


# Export all functions for easy import
__all__ = [
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
