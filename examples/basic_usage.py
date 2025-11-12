"""
Basic usage example for PSOD outlier detection.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from category_encoders import OneHotEncoder

# TODO: Update import once package is installed
# from psod import PSOD
# For now, assuming direct import
import sys
sys.path.append('../src')
from psod.core import PSOD


def basic_example():
    """Basic example of using PSOD for outlier detection."""
    print("=== Basic PSOD Example ===")
    
    # Create sample data with outliers
    np.random.seed(42)
    n_samples = 100
    n_outliers = 5
    
    # Generate normal data
    normal_data = np.random.randn(n_samples - n_outliers, 3)
    
    # Generate outliers
    outliers = np.random.uniform(-10, 10, (n_outliers, 3))
    
    # Combine data
    data = np.vstack([normal_data, outliers])
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
    
    # Add categorical feature
    df['category'] = np.random.choice(['A', 'B', 'C'], len(df))
    
    print(f"Data shape: {df.shape}")
    print(f"Data preview:\n{df.head()}")
    
    # Initialize PSOD
    detector = PSOD(
        cat_columns=['category'],
        min_cols_chosen=0.5,
        max_cols_chosen=1.0,
        stdevs_to_outlier=2.0,
        random_seed=42
    )
    
    # Detect outliers
    print("\nDetecting outliers...")
    outlier_scores = detector.fit_predict(df, return_class=False)
    outlier_labels = detector.fit_predict(df, return_class=True)
    
    print(f"\nOutlier scores summary:")
    print(f"Mean: {outlier_scores.mean():.4f}")
    print(f"Std: {outlier_scores.std():.4f}")
    print(f"Min: {outlier_scores.min():.4f}")
    print(f"Max: {outlier_scores.max():.4f}")
    
    print(f"\nNumber of outliers detected: {sum(outlier_labels)}")
    print(f"Outlier indices: {df.index[outlier_labels == 1].tolist()}")
    
    # TODO: Add visualization
    # from psod.visualization import plot_outlier_scores
    # plot_outlier_scores(outlier_scores, threshold=detector.pred_distribution_stats['mean_score'])


def advanced_example():
    """Advanced example with custom settings."""
    print("\n=== Advanced PSOD Example ===")
    
    # TODO: Implement advanced example with:
    # - Custom base learner (RandomForestRegressor)
    # - Different transformation algorithms
    # - Cross-validation for threshold selection
    # - Feature importance analysis
    pass


def time_series_example():
    """Example with time series data."""
    print("\n=== Time Series PSOD Example ===")
    
    # TODO: Implement time series example with:
    # - Temporal features
    # - Sliding window approach
    # - Trend and seasonality handling
    pass


def comparison_example():
    """Compare PSOD with other outlier detection methods."""
    print("\n=== Comparison Example ===")
    
    # TODO: Implement comparison with:
    # - Isolation Forest
    # - Local Outlier Factor
    # - One-Class SVM
    # - Performance metrics
    pass


def real_world_example():
    """Example using real-world dataset."""
    print("\n=== Real World Example ===")
    
    # TODO: Implement example with:
    # - Credit card fraud detection
    # - Network intrusion detection
    # - Manufacturing defect detection
    pass


if __name__ == "__main__":
    # Run basic example
    basic_example()
    
    # TODO: Uncomment when implemented
    # advanced_example()
    # time_series_example()
    # comparison_example()
    # real_world_example()
