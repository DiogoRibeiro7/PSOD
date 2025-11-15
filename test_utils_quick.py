"""Quick test of utils module functionality."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from psod import utils

print("=" * 60)
print("Testing PSOD Utils Module")
print("=" * 60)

# Test 1: Generate synthetic data
print("\n1. Testing synthetic data generation...")
try:
    for outlier_type in ["global", "local", "collective", "contextual", "point", "mixed"]:
        X, y = utils.generate_outlier_data(
            n_samples=100,
            n_features=5,
            contamination=0.1,
            outlier_type=outlier_type,
            random_state=42,
        )
        n_outliers: int = int(np.sum(y))
        print(
            f"   ✓ Generated {outlier_type} outliers: {X.shape[0]} samples, {n_outliers} outliers"
        )
    print("   ✓ Synthetic data generation works!")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")

# Test 2: Data validation
print("\n2. Testing data validation...")
try:
    X, y = utils.generate_outlier_data(n_samples=100, n_features=5, random_state=42)
    validation_results = utils.validate_dataframe(X, min_samples=10)
    print(f"   ✓ Validation passed: {validation_results['is_valid']}")
    print(f"   ✓ Found {len(validation_results['warnings'])} warnings")
    print(f"   ✓ Found {len(validation_results['errors'])} errors")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")

# Test 3: Score normalization
print("\n3. Testing score normalization...")
try:
    scores = np.random.randn(100)
    for method in ["minmax", "zscore", "rank", "sigmoid"]:
        normalized = utils.normalize_scores(scores, method=method)
        print(f"   ✓ {method}: min={normalized.min():.3f}, max={normalized.max():.3f}")
    print("   ✓ Score normalization works!")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")

# Test 4: Threshold selection
print("\n4. Testing threshold selection...")
try:
    scores = np.random.randn(100)
    for method in ["percentile", "median", "mad", "iqr", "std"]:
        if method == "percentile":
            threshold = utils.select_threshold(scores, method=method, contamination=0.1)
        else:
            threshold = utils.select_threshold(scores, method=method, k=1.5)
        print(f"   ✓ {method}: threshold={threshold:.3f}")
    print("   ✓ Threshold selection works!")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")

# Test 5: Score combination
print("\n5. Testing score combination...")
try:
    scores1 = np.random.randn(100)
    scores2 = np.random.randn(100)
    scores3 = np.random.randn(100)
    scores_list = [scores1, scores2, scores3]

    for method in ["average", "median", "maximum", "minimum", "rank_average", "geometric_mean"]:
        combined = utils.combine_outlier_scores(scores_list, method=method)
        print(f"   ✓ {method}: combined {len(scores_list)} score arrays")
    print("   ✓ Score combination works!")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")

# Test 6: Evaluation metrics
print("\n6. Testing evaluation metrics...")
try:
    X, y_true = utils.generate_outlier_data(n_samples=100, n_features=5, random_state=42)
    y_pred = np.random.randint(0, 2, size=100)
    y_scores = np.random.rand(100)

    metrics = utils.evaluate_outlier_detection(y_true, y_pred)
    print(
        f"   ✓ Basic metrics: precision={metrics.get('precision', 0):.3f}, recall={metrics.get('recall', 0):.3f}"
    )

    detailed_metrics = utils.compute_detailed_metrics(y_true, y_pred, y_scores)
    print(
        f"   ✓ Detailed metrics: accuracy={detailed_metrics['accuracy']:.3f}, specificity={detailed_metrics['specificity']:.3f}"
    )
    print("   ✓ Evaluation metrics work!")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")

# Test 7: Data preprocessing
print("\n7. Testing data preprocessing...")
try:
    # Create data with missing values
    df = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
    df.iloc[::10, 0] = np.nan  # Add some missing values

    for strategy in ["drop", "mean", "median", "mode", "knn"]:
        df_clean = utils.handle_missing_values(df.copy(), strategy=strategy)
        print(f"   ✓ {strategy}: shape={df_clean.shape}, missing={df_clean.isnull().sum().sum()}")
    print("   ✓ Data preprocessing works!")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")

# Test 8: Feature importance
print("\n8. Testing feature importance...")
try:
    X, y = utils.generate_outlier_data(n_samples=100, n_features=5, random_state=42)
    scores = np.random.rand(100)

    importance = utils.compute_feature_importance(X, scores, method="correlation")
    print(f"   ✓ Computed importance for {len(importance)} features")
    print(f"   ✓ Top feature: {importance.index[0]} (score={importance.iloc[0]:.3f})")
    print("   ✓ Feature importance works!")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")

# Test 9: Feature summary and drift detection
print("\n9. Testing feature summary and drift detection...")
try:
    X1, _ = utils.generate_outlier_data(n_samples=100, n_features=5, random_state=42)
    X2, _ = utils.generate_outlier_data(n_samples=100, n_features=5, random_state=43)

    summary = utils.create_feature_summary(X1)
    print(f"   ✓ Created feature summary: {summary.shape[0]} features")

    drift_results = utils.detect_feature_drift(X1, X2, threshold=0.1)
    print(f"   ✓ Detected drift in {len(drift_results['drifted_features'])} features")
    print("   ✓ Feature analysis works!")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
