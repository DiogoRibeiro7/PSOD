"""
Evaluation metrics and utilities for benchmarking outlier detection methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix
)
import warnings

warnings.filterwarnings('ignore')


def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    contamination: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0=normal, 1=outlier)
    y_scores : np.ndarray
        Outlier scores (higher = more anomalous)
    y_pred : np.ndarray, optional
        Binary predictions (0=normal, 1=outlier)
    contamination : float, optional
        Expected contamination level for threshold

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary of evaluation metrics
    """
    metrics = {}

    # ROC-AUC (ranking-based)
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
    except Exception as e:
        metrics['roc_auc'] = np.nan

    # Average Precision (PR-AUC)
    try:
        metrics['avg_precision'] = average_precision_score(y_true, y_scores)
    except Exception as e:
        metrics['avg_precision'] = np.nan

    # Precision at k (top k predictions)
    if contamination is not None:
        k = int(len(y_true) * contamination)
        top_k_indices = np.argsort(y_scores)[-k:]
        metrics['precision_at_k'] = np.sum(y_true[top_k_indices]) / k

    # Binary classification metrics
    if y_pred is not None:
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)

        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0

    return metrics


def precision_at_n(y_true: np.ndarray, y_scores: np.ndarray, n: int) -> float:
    """
    Compute precision at top n predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_scores : np.ndarray
        Outlier scores
    n : int
        Number of top predictions

    Returns
    -------
    precision : float
        Precision at n
    """
    top_n_indices = np.argsort(y_scores)[-n:]
    return np.sum(y_true[top_n_indices]) / n


def compute_precision_recall_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k_values: Optional[list] = None
) -> Dict[str, list]:
    """
    Compute precision and recall at different k values.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_scores : np.ndarray
        Outlier scores
    k_values : list, optional
        List of k values (as proportions of dataset size)

    Returns
    -------
    results : Dict[str, list]
        Dictionary with 'k', 'precision', 'recall' lists
    """
    if k_values is None:
        k_values = [0.01, 0.02, 0.05, 0.10, 0.20]

    n_samples = len(y_true)
    n_outliers = np.sum(y_true)

    results = {'k': [], 'precision': [], 'recall': []}

    for k_prop in k_values:
        k = max(1, int(n_samples * k_prop))
        top_k_indices = np.argsort(y_scores)[-k:]

        precision = np.sum(y_true[top_k_indices]) / k
        recall = np.sum(y_true[top_k_indices]) / n_outliers if n_outliers > 0 else 0

        results['k'].append(k_prop)
        results['precision'].append(precision)
        results['recall'].append(recall)

    return results


def compute_roc_curve_data(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute ROC curve data.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_scores : np.ndarray
        Outlier scores

    Returns
    -------
    data : Dict[str, np.ndarray]
        Dictionary with 'fpr', 'tpr', 'thresholds'
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc_score(y_true, y_scores)
    }


def compute_pr_curve_data(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute precision-recall curve data.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_scores : np.ndarray
        Outlier scores

    Returns
    -------
    data : Dict[str, np.ndarray]
        Dictionary with 'precision', 'recall', 'thresholds'
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    return {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'auc': average_precision_score(y_true, y_scores)
    }


def compute_ranking_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """
    Compute ranking-based metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_scores : np.ndarray
        Outlier scores

    Returns
    -------
    metrics : Dict[str, float]
        Ranking metrics
    """
    # Sort by scores (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_labels = y_true[sorted_indices]

    # NDCG (Normalized Discounted Cumulative Gain)
    n_outliers = np.sum(y_true)
    if n_outliers == 0:
        return {'ndcg': 0.0, 'map': 0.0}

    # DCG
    dcg = np.sum(sorted_labels / np.log2(np.arange(2, len(sorted_labels) + 2)))

    # IDCG (ideal DCG)
    ideal_labels = np.sort(y_true)[::-1]
    idcg = np.sum(ideal_labels / np.log2(np.arange(2, len(ideal_labels) + 2)))

    ndcg = dcg / idcg if idcg > 0 else 0

    # Mean Average Precision (MAP)
    precisions_at_k = []
    for k in range(1, len(sorted_labels) + 1):
        if sorted_labels[k-1] == 1:
            precision_at_k = np.sum(sorted_labels[:k]) / k
            precisions_at_k.append(precision_at_k)

    map_score = np.mean(precisions_at_k) if precisions_at_k else 0

    return {
        'ndcg': ndcg,
        'map': map_score
    }


def aggregate_metrics(results_list: list) -> pd.DataFrame:
    """
    Aggregate metrics from multiple runs.

    Parameters
    ----------
    results_list : list
        List of metric dictionaries

    Returns
    -------
    summary : pd.DataFrame
        Aggregated statistics (mean, std, min, max)
    """
    df = pd.DataFrame(results_list)

    summary = pd.DataFrame({
        'mean': df.mean(),
        'std': df.std(),
        'min': df.min(),
        'max': df.max(),
        'median': df.median()
    })

    return summary


def compare_methods(
    results_df: pd.DataFrame,
    metric: str = 'roc_auc',
    baseline: str = 'IsolationForest'
) -> pd.DataFrame:
    """
    Compare methods against baseline.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results
    metric : str
        Metric to compare
    baseline : str
        Baseline method name

    Returns
    -------
    comparison : pd.DataFrame
        Comparison results with improvements
    """
    if baseline not in results_df['method'].values:
        baseline = results_df['method'].values[0]

    baseline_score = results_df[results_df['method'] == baseline][metric].mean()

    comparison = results_df.groupby('method')[metric].agg(['mean', 'std']).reset_index()
    comparison['improvement'] = (comparison['mean'] - baseline_score) / baseline_score * 100
    comparison = comparison.sort_values('mean', ascending=False)

    return comparison


def statistical_test(
    results_df: pd.DataFrame,
    method1: str,
    method2: str,
    metric: str = 'roc_auc'
) -> Dict[str, float]:
    """
    Perform statistical significance test between two methods.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results
    method1 : str
        First method name
    method2 : str
        Second method name
    metric : str
        Metric to test

    Returns
    -------
    test_results : Dict[str, float]
        Test statistics and p-value
    """
    from scipy import stats

    scores1 = results_df[results_df['method'] == method1][metric].values
    scores2 = results_df[results_df['method'] == method2][metric].values

    # Paired t-test
    if len(scores1) == len(scores2) and len(scores1) > 1:
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
    else:
        t_stat, p_value = stats.ttest_ind(scores1, scores2)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_diff': np.mean(scores1) - np.mean(scores2)
    }
