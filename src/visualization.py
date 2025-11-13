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


def plot_outlier_scores(scores: Union[pd.Series, np.ndarray],
                       threshold: Optional[float] = None,
                       title: str = "Outlier Scores Distribution",
                       figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot distribution of outlier scores.
    
    Parameters
    ----------
    scores : Union[pd.Series, np.ndarray]
        Outlier scores to plot.
    threshold : float, optional
        Threshold line to mark outliers.
    title : str, default="Outlier Scores Distribution"
        Plot title.
    figsize : Tuple[int, int], default=(10, 6)
        Figure size.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    # Convert to numpy array for consistency
    if isinstance(scores, pd.Series):
        scores_array = scores.values
    else:
        scores_array = scores
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram of scores
    n_bins = min(50, len(scores_array) // 10)
    n, bins, patches = ax.hist(scores_array, bins=n_bins, alpha=0.7, 
                              color='skyblue', edgecolor='black', linewidth=0.5)
    
    # Color outliers differently if threshold is provided
    if threshold is not None:
        # Find bins that exceed threshold
        for i, (patch, bin_left, bin_right) in enumerate(zip(patches, bins[:-1], bins[1:])):
            if bin_right > threshold:
                patch.set_facecolor('red')
                patch.set_alpha(0.8)
        
        # Add threshold line
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                  label=f'Threshold: {threshold:.3f}')
        
        # Calculate outlier percentage
        outlier_pct = np.sum(scores_array > threshold) / len(scores_array) * 100
        ax.text(0.02, 0.98, f'Outliers: {outlier_pct:.1f}%', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add summary statistics
    stats_text = f'Mean: {np.mean(scores_array):.3f}\n'
    stats_text += f'Std: {np.std(scores_array):.3f}\n'
    stats_text += f'Min: {np.min(scores_array):.3f}\n'
    stats_text += f'Max: {np.max(scores_array):.3f}'
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Formatting
    ax.set_xlabel('Outlier Score')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if threshold is not None:
        ax.legend()
    
    plt.tight_layout()
    return fig


def plot_feature_contributions(model,
                             sample_idx: int,
                             top_k: int = 10) -> plt.Figure:
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
    if not hasattr(model, 'feature_names_'):
        feature_names = [f'Feature_{i}' for i in range(model.n_features_)]
    else:
        feature_names = model.feature_names_
    
    # Calculate feature contributions (this assumes model has prediction errors stored)
    if hasattr(model, 'prediction_errors_'):
        contributions = np.abs(model.prediction_errors_[sample_idx])
    else:
        # Fallback: use feature importance if available
        if hasattr(model, 'feature_importances_'):
            contributions = model.feature_importances_
        else:
            # Generate mock contributions for demonstration
            warnings.warn("Model doesn't have prediction errors or feature importances. Using random contributions.")
            contributions = np.random.rand(len(feature_names))
    
    # Get top-k features
    top_indices = np.argsort(contributions)[-top_k:][::-1]
    top_contributions = contributions[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red' if contrib > np.mean(top_contributions) else 'skyblue' 
              for contrib in top_contributions]
    
    bars = ax.barh(range(len(top_features)), top_contributions, color=colors)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_contributions)):
        ax.text(value + 0.01 * max(top_contributions), bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', ha='left', va='center')
    
    # Formatting
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Contribution to Outlier Score')
    ax.set_title(f'Feature Contributions for Sample {sample_idx}')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_outliers_scatter(X: pd.DataFrame,
                         outlier_labels: np.ndarray,
                         features: Optional[List[str]] = None,
                         dim: int = 2,
                         use_pca: bool = False) -> go.Figure:
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
        feature_names = [f'PC{i+1}' for i in range(dim)]
        explained_var = pca.explained_variance_ratio_
        title_suffix = f" (PCA: {sum(explained_var):.1%} variance)"
    else:
        X_plot = X[features[:dim]].values
        feature_names = features[:dim]
        title_suffix = ""
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame(X_plot, columns=feature_names)
    plot_data['Outlier'] = outlier_labels.astype(bool)
    plot_data['Label'] = ['Outlier' if x else 'Normal' for x in outlier_labels]
    
    # Create scatter plot
    if dim == 2:
        fig = px.scatter(plot_data, 
                        x=feature_names[0], 
                        y=feature_names[1],
                        color='Label',
                        color_discrete_map={'Normal': 'blue', 'Outlier': 'red'},
                        hover_data={'Outlier': False},
                        title=f'2D Scatter Plot of Data Points{title_suffix}')
    elif dim == 3:
        fig = px.scatter_3d(plot_data,
                           x=feature_names[0],
                           y=feature_names[1], 
                           z=feature_names[2],
                           color='Label',
                           color_discrete_map={'Normal': 'blue', 'Outlier': 'red'},
                           hover_data={'Outlier': False},
                           title=f'3D Scatter Plot of Data Points{title_suffix}')
    else:
        raise ValueError("dim must be 2 or 3")
    
    # Update layout
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1)
    )
    
    return fig


def plot_timeseries_outliers(data: pd.DataFrame,
                           outlier_labels: np.ndarray,
                           time_column: str,
                           value_columns: List[str]) -> go.Figure:
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
        rows=len(value_columns), cols=1,
        shared_xaxes=True,
        subplot_titles=value_columns,
        vertical_spacing=0.05
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
                mode='markers+lines',
                name=f'{col} (Normal)',
                line=dict(color='blue', width=1),
                marker=dict(size=3),
                showlegend=(i == 0)
            ),
            row=row, col=1
        )
        
        # Outlier points
        outlier_mask = outlier_labels == 1
        if np.any(outlier_mask):
            fig.add_trace(
                go.Scatter(
                    x=data[time_column][outlier_mask],
                    y=data[col][outlier_mask],
                    mode='markers',
                    name=f'{col} (Outliers)',
                    marker=dict(color='red', size=8, symbol='x'),
                    showlegend=(i == 0)
                ),
                row=row, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=200 * len(value_columns),
        title="Time Series with Outliers Highlighted",
        showlegend=True
    )
    
    # Add range slider to bottom subplot
    fig.update_layout(
        xaxis=dict(rangeslider=dict(visible=True), type="date")
    )
    
    return fig


def plot_correlation_heatmap(X: pd.DataFrame,
                           outlier_labels: np.ndarray,
                           figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
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
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, ax=ax1, cbar_kws={"shrink": .5})
    ax1.set_title('Feature Correlation Matrix')
    
    # Outlier correlation heatmap
    outlier_corr_df = pd.DataFrame.from_dict(outlier_pcts, orient='index', 
                                           columns=['Outlier_Correlation'])
    sns.heatmap(outlier_corr_df, annot=True, cmap='RdBu_r', center=0,
                ax=ax2, cbar_kws={"shrink": .5})
    ax2.set_title('Feature Correlation with Outliers')
    ax2.set_xlabel('Correlation with Outlier Labels')
    
    plt.tight_layout()
    return fig


def plot_score_evolution(scores_history: List[np.ndarray],
                        labels: Optional[List[str]] = None) -> go.Figure:
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
        labels = [f'Iteration {i+1}' for i in range(len(scores_history))]
    
    fig = go.Figure()
    
    n_samples = len(scores_history[0])
    sample_indices = np.arange(n_samples)
    
    # Calculate statistics for each iteration
    for i, (scores, label) in enumerate(zip(scores_history, labels)):
        # Add mean score line
        fig.add_trace(go.Scatter(
            x=sample_indices,
            y=scores,
            mode='lines+markers',
            name=label,
            line=dict(width=2),
            marker=dict(size=4)
        ))
    
    # Add confidence intervals for the last iteration
    if len(scores_history) > 1:
        last_scores = scores_history[-1]
        mean_scores = np.mean(scores_history, axis=0)
        std_scores = np.std(scores_history, axis=0)
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([sample_indices, sample_indices[::-1]]),
            y=np.concatenate([mean_scores + std_scores, 
                             (mean_scores - std_scores)[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True
        ))
    
    # Highlight stable vs unstable outliers
    if len(scores_history) > 2:
        # Calculate coefficient of variation for each sample
        cv = np.std(scores_history, axis=0) / np.mean(scores_history, axis=0)
        unstable_mask = cv > np.percentile(cv, 90)  # Top 10% most variable
        
        if np.any(unstable_mask):
            fig.add_trace(go.Scatter(
                x=sample_indices[unstable_mask],
                y=scores_history[-1][unstable_mask],
                mode='markers',
                name='Unstable Outliers',
                marker=dict(color='red', size=8, symbol='x')
            ))
    
    fig.update_layout(
        title='Evolution of Outlier Scores',
        xaxis_title='Sample Index',
        yaxis_title='Outlier Score',
        hovermode='x unified'
    )
    
    return fig


def plot_roc_pr_curves(y_true: np.ndarray,
                      y_scores: np.ndarray,
                      title: str = "ROC and PR Curves") -> plt.Figure:
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
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random baseline')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    baseline_precision = np.sum(y_true) / len(y_true)
    ax2.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    ax2.axhline(y=baseline_precision, color='navy', linestyle='--', lw=2,
                label=f'Random baseline ({baseline_precision:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def create_outlier_dashboard(X: pd.DataFrame,
                           outlier_scores: np.ndarray,
                           outlier_labels: np.ndarray) -> None:
    """
    Create comprehensive dashboard for outlier analysis.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    outlier_scores : np.ndarray
        Continuous outlier scores.
    outlier_labels : np.ndarray
        Binary outlier labels.
    """
    # Create multi-panel dashboard
    fig = plt.figure(figsize=(20, 15))
    
    # Define threshold from labels
    if np.any(outlier_labels):
        threshold = np.min(outlier_scores[outlier_labels == 1])
    else:
        threshold = np.percentile(outlier_scores, 95)
    
    # 1. Score distribution
    plt.subplot(3, 3, 1)
    plt.hist(outlier_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    if threshold is not None:
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Outlier Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.grid(True, alpha=0.3)
    
    # 2. Feature importance (if available)
    plt.subplot(3, 3, 2)
    if X.shape[1] <= 20:  # Only plot if reasonable number of features
        feature_importance = np.std(X, axis=0)  # Use std as proxy for importance
        top_features = np.argsort(feature_importance)[-10:]
        plt.barh(range(len(top_features)), feature_importance[top_features])
        plt.yticks(range(len(top_features)), X.columns[top_features])
        plt.xlabel('Feature Importance (Std)')
        plt.title('Top Feature Importances')
    else:
        plt.text(0.5, 0.5, f'Too many features ({X.shape[1]}) to display', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Importance')
    
    # 3. 2D scatter plot
    plt.subplot(3, 3, 3)
    if X.shape[1] >= 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        colors = ['red' if label else 'blue' for label in outlier_labels]
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        plt.title('PCA Projection')
    
    # 4. Correlation heatmap (subset of features)
    plt.subplot(3, 3, 4)
    if X.shape[1] <= 15:
        corr_matrix = X.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0, square=True)
        plt.title('Feature Correlations')
    else:
        # Show correlation with first few features
        subset_corr = X.iloc[:, :10].corr()
        sns.heatmap(subset_corr, cmap='coolwarm', center=0, square=True)
        plt.title('Feature Correlations (Subset)')
    
    # 5. Outlier vs normal statistics
    plt.subplot(3, 3, 5)
    outlier_mask = outlier_labels == 1
    normal_mask = outlier_labels == 0
    
    if np.any(outlier_mask) and np.any(normal_mask):
        outlier_means = X[outlier_mask].mean()
        normal_means = X[normal_mask].mean()
        
        # Show means for top features
        top_diff_features = np.argsort(np.abs(outlier_means - normal_means))[-10:]
        
        x_pos = np.arange(len(top_diff_features))
        plt.bar(x_pos - 0.2, normal_means.iloc[top_diff_features], 
                width=0.4, label='Normal', alpha=0.7)
        plt.bar(x_pos + 0.2, outlier_means.iloc[top_diff_features], 
                width=0.4, label='Outliers', alpha=0.7)
        plt.xticks(x_pos, X.columns[top_diff_features], rotation=45)
        plt.ylabel('Mean Value')
        plt.title('Feature Means: Normal vs Outliers')
        plt.legend()
    
    # 6. Summary statistics table
    plt.subplot(3, 3, 6)
    plt.axis('off')
    
    # Create summary statistics
    n_total = len(outlier_labels)
    n_outliers = np.sum(outlier_labels)
    outlier_rate = n_outliers / n_total * 100
    
    stats_text = f"""
    Summary Statistics:
    
    Total Samples: {n_total:,}
    Outliers: {n_outliers:,} ({outlier_rate:.1f}%)
    Normal: {n_total - n_outliers:,} ({100 - outlier_rate:.1f}%)
    
    Score Statistics:
    Mean: {np.mean(outlier_scores):.3f}
    Std: {np.std(outlier_scores):.3f}
    Min: {np.min(outlier_scores):.3f}
    Max: {np.max(outlier_scores):.3f}
    
    Threshold: {threshold:.3f}
    """
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.title('Summary Statistics')
    
    # 7. Score vs index plot
    plt.subplot(3, 3, 7)
    colors = ['red' if label else 'blue' for label in outlier_labels]
    plt.scatter(range(len(outlier_scores)), outlier_scores, c=colors, alpha=0.6)
    plt.xlabel('Sample Index')
    plt.ylabel('Outlier Score')
    plt.title('Outlier Scores by Sample')
    if threshold is not None:
        plt.axhline(threshold, color='red', linestyle='--', alpha=0.7)
    
    # 8. Feature distribution comparison
    plt.subplot(3, 3, 8)
    if X.shape[1] > 0:
        # Pick a representative feature
        feature_idx = np.argmax(X.std())
        feature_name = X.columns[feature_idx]
        
        if np.any(outlier_mask) and np.any(normal_mask):
            plt.hist(X.iloc[normal_mask, feature_idx], bins=20, alpha=0.6, 
                    label='Normal', color='blue')
            plt.hist(X.iloc[outlier_mask, feature_idx], bins=20, alpha=0.6, 
                    label='Outliers', color='red')
            plt.xlabel(feature_name)
            plt.ylabel('Frequency')
            plt.title(f'Distribution: {feature_name}')
            plt.legend()
    
    # 9. Box plot of scores
    plt.subplot(3, 3, 9)
    score_data = [outlier_scores[normal_mask], outlier_scores[outlier_mask]]
    labels = ['Normal', 'Outliers']
    plt.boxplot(score_data, labels=labels)
    plt.ylabel('Outlier Score')
    plt.title('Score Distribution by Class')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Outlier Detection Dashboard', fontsize=16)
    plt.tight_layout()
    
    # Save as HTML report
    plt.savefig('/mnt/user-data/outputs/outlier_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_interactive_explorer(X: pd.DataFrame,
                              model,
                              port: int = 8050) -> None:
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
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X)
    elif hasattr(model, 'predict_proba'):
        scores = model.predict_proba(X)[:, 1]
    else:
        print("Model doesn't have decision_function or predict_proba method")
        return
    
    # Create Dash application
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Interactive Outlier Explorer"),
        
        html.Div([
            html.Label("Select Features for Visualization:"),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': col, 'value': col} for col in X.columns],
                value=X.columns[:2].tolist(),
                multi=True
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Outlier Threshold:"),
            dcc.Slider(
                id='threshold-slider',
                min=np.min(scores),
                max=np.max(scores),
                value=np.percentile(scores, 95),
                step=(np.max(scores) - np.min(scores)) / 100,
                marks={
                    np.min(scores): f'{np.min(scores):.2f}',
                    np.percentile(scores, 50): f'{np.percentile(scores, 50):.2f}',
                    np.percentile(scores, 95): f'{np.percentile(scores, 95):.2f}',
                    np.max(scores): f'{np.max(scores):.2f}'
                }
            ),
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        
        dcc.Graph(id='scatter-plot'),
        
        html.Div([
            html.H3("Sample Details"),
            html.Div(id='sample-details')
        ]),
        
        html.Div([
            html.Button("Export Results", id="export-button"),
            html.Div(id="export-status")
        ])
    ])
    
    @app.callback(
        Output('scatter-plot', 'figure'),
        [Input('feature-dropdown', 'value'),
         Input('threshold-slider', 'value')]
    )
    def update_scatter_plot(selected_features, threshold):
        if not selected_features or len(selected_features) < 2:
            return {'data': [], 'layout': {'title': 'Please select at least 2 features'}}
        
        # Create outlier labels based on threshold
        outlier_labels = scores > threshold
        
        # Create scatter plot
        fig = px.scatter(
            x=X[selected_features[0]], 
            y=X[selected_features[1]],
            color=['Outlier' if x else 'Normal' for x in outlier_labels],
            color_discrete_map={'Normal': 'blue', 'Outlier': 'red'},
            title=f'Outlier Detection: {selected_features[0]} vs {selected_features[1]}'
        )
        
        return fig
    
    @app.callback(
        Output('sample-details', 'children'),
        [Input('scatter-plot', 'clickData'),
         Input('threshold-slider', 'value')]
    )
    def display_sample_details(clickData, threshold):
        if clickData is None:
            return "Click on a point to see sample details"
        
        point_idx = clickData['points'][0]['pointIndex']
        sample_score = scores[point_idx]
        is_outlier = sample_score > threshold
        
        sample_data = X.iloc[point_idx]
        
        details = html.Div([
            html.P(f"Sample Index: {point_idx}"),
            html.P(f"Outlier Score: {sample_score:.4f}"),
            html.P(f"Is Outlier: {'Yes' if is_outlier else 'No'}"),
            html.H4("Feature Values:"),
            html.Ul([html.Li(f"{col}: {val:.4f}") for col, val in sample_data.items()])
        ])
        
        return details
    
    @app.callback(
        Output('export-status', 'children'),
        [Input('export-button', 'n_clicks'),
         Input('threshold-slider', 'value')]
    )
    def export_results(n_clicks, threshold):
        if n_clicks is None:
            return ""
        
        # Export results
        outlier_labels = scores > threshold
        results_df = X.copy()
        results_df['outlier_score'] = scores
        results_df['is_outlier'] = outlier_labels
        
        results_df.to_csv('/mnt/user-data/outputs/outlier_results.csv', index=False)
        
        return html.P("Results exported to outlier_results.csv", style={'color': 'green'})
    
    print(f"Starting Dash app on port {port}")
    print(f"Open http://localhost:{port} in your browser")
    app.run_server(debug=True, port=port)


# Additional utility functions
def plot_feature_distributions(X: pd.DataFrame, 
                             outlier_labels: np.ndarray,
                             max_features: int = 12,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
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
            ax.hist(X.loc[normal_mask, col], bins=20, alpha=0.6, 
                   label='Normal', color='blue', density=True)
        
        if np.any(outlier_mask):
            ax.hist(X.loc[outlier_mask, col], bins=20, alpha=0.6, 
                   label='Outliers', color='red', density=True)
        
        ax.set_title(col)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distributions: Normal vs Outliers')
    plt.tight_layout()
    return fig


def plot_outlier_evolution_heatmap(scores_history: List[np.ndarray],
                                 labels: Optional[List[str]] = None,
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
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
        labels = [f'Iteration {i+1}' for i in range(len(scores_history))]
    
    # Create matrix of scores
    scores_matrix = np.array(scores_history).T
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(scores_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Sample Index')
    ax.set_xlabel('Iteration/Model')
    ax.set_title('Evolution of Outlier Scores Across Iterations')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Outlier Score')
    
    plt.tight_layout()
    return fig


# Export all functions for easy import
__all__ = [
    'plot_outlier_scores',
    'plot_feature_contributions', 
    'plot_outliers_scatter',
    'plot_timeseries_outliers',
    'plot_correlation_heatmap',
    'plot_score_evolution',
    'plot_roc_pr_curves',
    'create_outlier_dashboard',
    'create_interactive_explorer',
    'plot_feature_distributions',
    'plot_outlier_evolution_heatmap'
]
