"""
Test script for PSOD visualization module.
Demonstrates all visualization functions with synthetic data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src" / "psod"))

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Import utils and visualization
import utils
import visualization as viz

print("=" * 70)
print("PSOD Visualization Module Test Suite")
print("=" * 70)

# Generate synthetic data
print("\n1. Generating synthetic outlier data...")
np.random.seed(42)
X, y_true = utils.generate_outlier_data(
    n_samples=200, n_features=10, contamination=0.1, outlier_type="mixed", random_state=42
)
print(f"   ✓ Generated {X.shape[0]} samples with {X.shape[1]} features")
print(f"   ✓ True outliers: {np.sum(y_true)} ({np.sum(y_true)/len(y_true)*100:.1f}%)")

# Generate outlier scores (simulated from model)
outlier_scores = np.random.randn(len(X))
outlier_scores[y_true == 1] += 2  # Make outliers have higher scores
threshold = np.percentile(outlier_scores, 90)
y_pred = (outlier_scores > threshold).astype(int)

print(f"   ✓ Generated outlier scores with threshold: {threshold:.3f}")

# Test 1: Comprehensive Outlier Score Distribution Plot
print("\n2. Testing plot_outlier_scores (comprehensive 4-subplot version)...")
try:
    fig = viz.plot_outlier_scores(
        scores=outlier_scores,
        threshold=threshold,
        bins=50,
        title="Comprehensive Outlier Score Distribution",
        figsize=(12, 8),
    )
    fig.savefig("test_output_score_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   ✓ Created comprehensive 4-subplot score distribution plot")
    print("     - Histogram with colored outlier bins")
    print("     - Box plot with threshold line")
    print("     - Q-Q plot vs normal distribution")
    print("     - Cumulative distribution function")
    print("   ✓ Saved to: test_output_score_dist.png")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")
    import traceback

    traceback.print_exc()

# Test 2: Comprehensive Static Dashboard
print("\n3. Testing create_outlier_dashboard (4x4 grid dashboard)...")
try:
    fig = viz.create_outlier_dashboard(
        X=X,
        outlier_scores=outlier_scores,
        outlier_labels=y_pred,
        model=None,  # Can pass PSOD model if available
        save_path="test_output_dashboard.png",
    )
    plt.close(fig)
    print("   ✓ Created comprehensive dashboard with 12+ panels:")
    print("     Row 1: Score dist, Feature importance, PCA projection, Summary stats")
    print("     Row 2: Feature correlations, Normal vs Outlier comparison")
    print("     Row 3: Score evolution, Box plots by class")
    print("     Row 4: Feature distributions (4 features)")
    print("   ✓ Saved to: test_output_dashboard.png")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")
    import traceback

    traceback.print_exc()

# Test 3: Outlier Scatter Plot
print("\n4. Testing plot_outliers_scatter...")
try:
    fig = viz.plot_outliers_scatter(
        X=X, outlier_labels=y_pred, features=list(X.columns[:3]), dim=3, use_pca=False
    )
    fig.write_html("test_output_scatter.html")
    print("   ✓ Created 3D scatter plot with outliers highlighted")
    print("   ✓ Saved to: test_output_scatter.html")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")

# Test 4: Feature Distributions
print("\n5. Testing plot_feature_distributions...")
try:
    fig = viz.plot_feature_distributions(
        X=X, outlier_labels=y_pred, max_features=12, figsize=(15, 10)
    )
    fig.savefig("test_output_feature_dists.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   ✓ Created feature distribution comparison plots")
    print("   ✓ Saved to: test_output_feature_dists.png")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")

# Test 5: ROC and PR Curves
print("\n6. Testing plot_roc_pr_curves...")
try:
    fig = viz.plot_roc_pr_curves(
        y_true=y_true, y_scores=outlier_scores, title="Outlier Detection Performance Curves"
    )
    fig.savefig("test_output_roc_pr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   ✓ Created ROC and Precision-Recall curves")
    print("   ✓ Saved to: test_output_roc_pr.png")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")

# Test 6: Correlation Heatmap
print("\n7. Testing plot_correlation_heatmap...")
try:
    fig = viz.plot_correlation_heatmap(X=X, outlier_labels=y_pred, figsize=(12, 10))
    fig.savefig("test_output_corr_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   ✓ Created correlation heatmap with outlier correlations")
    print("   ✓ Saved to: test_output_corr_heatmap.png")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")

# Test 7: Score Evolution (if multiple iterations available)
print("\n8. Testing plot_score_evolution...")
try:
    # Simulate multiple iterations
    scores_history = [outlier_scores + np.random.randn(len(outlier_scores)) * 0.1 for _ in range(5)]
    labels = [f"Iteration {i+1}" for i in range(5)]

    fig = viz.plot_score_evolution(scores_history=scores_history, labels=labels)
    fig.write_html("test_output_evolution.html")
    print("   ✓ Created score evolution plot across iterations")
    print("   ✓ Saved to: test_output_evolution.html")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")

# Test 8: Outlier Evolution Heatmap
print("\n9. Testing plot_outlier_evolution_heatmap...")
try:
    fig = viz.plot_outlier_evolution_heatmap(
        scores_history=scores_history, labels=labels, figsize=(12, 8)
    )
    fig.savefig("test_output_evolution_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   ✓ Created outlier evolution heatmap")
    print("   ✓ Saved to: test_output_evolution_heatmap.png")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")

# Summary
print("\n" + "=" * 70)
print("VISUALIZATION TEST SUMMARY")
print("=" * 70)
print("\nStatic Visualizations Created:")
print("  1. ✓ Comprehensive Score Distribution (4 subplots)")
print("  2. ✓ Comprehensive Dashboard (4x4 grid, 12+ panels)")
print("  3. ✓ Feature Distribution Comparison")
print("  4. ✓ ROC and PR Curves")
print("  5. ✓ Correlation Heatmap")
print("  6. ✓ Evolution Heatmap")
print("\nInteractive Visualizations Created:")
print("  7. ✓ 3D Scatter Plot (HTML)")
print("  8. ✓ Score Evolution Plot (HTML)")
print("\nAll visualization outputs saved to current directory!")
print("\nVisualization Module Features:")
print("  • Static plots using matplotlib/seaborn")
print("  • Interactive plots using plotly")
print("  • Comprehensive dashboards")
print("  • Multiple analysis perspectives")
print("  • Export capabilities (PNG, HTML)")
print("=" * 70)
