# PSOD Benchmarking Suite

Comprehensive benchmarking system to compare PSOD with other outlier detection methods across various datasets, metrics, and scenarios.

---

## 📊 Features

### ✅ Comprehensive Comparison
- **PSOD** vs **sklearn** methods (IsolationForest, LOF, OneClassSVM, EllipticEnvelope)
- **PSOD** vs **PyOD** methods (KNN, COPOD, ECOD, HBOS, PCA)
- Side-by-side performance metrics
- Statistical significance testing

### ✅ Diverse Datasets
- **6 Outlier Types**: Global, Local, Collective, Contextual, Mixed, High-dimensional
- **12 Benchmark Datasets**: Various sizes, features, and contamination levels
- **Scalability Tests**: 100 to 10,000 samples
- **Robustness Tests**: Different contamination levels and dimensions

### ✅ Comprehensive Metrics
- **Performance**: ROC-AUC, Average Precision, Precision, Recall, F1-Score
- **Efficiency**: Training time, Prediction time, Memory usage
- **Ranking**: NDCG, MAP, Precision@K
- **Statistical**: Mean, Std, Confidence intervals

### ✅ Advanced Visualization
- Method comparison bar charts
- Performance vs time scatter plots
- Multi-metric heatmaps
- Scalability curves (log-log plots)
- ROC and Precision-Recall curves
- Memory usage comparisons
- Radar charts for multi-dimensional comparison
- Comprehensive dashboards

### ✅ Automated Reporting
- CSV results export
- Markdown reports with tables
- PNG visualizations (300 DPI)
- Summary statistics
- Method rankings

---

## 📁 File Structure

```
benchmarks/
├── __init__.py
├── README.md                    # This file
├── run_benchmarks.py            # Main benchmark runner (620 lines)
├── datasets.py                  # Dataset generation (440 lines)
├── methods.py                   # Method wrappers (260 lines)
├── metrics.py                   # Evaluation metrics (360 lines)
├── visualization.py             # Plotting functions (650 lines)
└── benchmark_results/           # Output directory (created at runtime)
    ├── benchmark_results.csv
    ├── BENCHMARK_REPORT.md
    ├── method_comparison_roc.png
    ├── performance_vs_time.png
    ├── multi_metric_heatmap.png
    ├── scalability.png
    ├── memory_usage.png
    └── benchmark_dashboard.png
```

**Total**: ~2,330 lines of production-ready benchmark code

---

## 🚀 Quick Start

### Basic Usage

```bash
# Run complete benchmark suite
cd benchmarks
python run_benchmarks.py
```

This will:
1. Benchmark methods on 4 standard datasets
2. Run scalability tests
3. Run robustness tests
4. Generate visualizations
5. Create comprehensive report

### Custom Benchmarking

```python
from run_benchmarks import BenchmarkRunner

# Initialize runner
runner = BenchmarkRunner(random_state=42, output_dir='my_results')

# Benchmark on specific datasets
results_df = runner.benchmark_datasets(
    dataset_names=['small_global', 'medium_mixed'],
    method_subset='all',  # Options: 'basic', 'fast', 'accurate', 'all'
    test_size=0.3
)

# Run scalability tests
scalability_results = runner.benchmark_scalability(method_subset='fast')

# Run robustness tests
robustness_results = runner.benchmark_robustness(method_subset='basic')

# Generate report
runner.save_results()
runner.generate_report()
```

---

## 📦 Dependencies

### Required
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Optional (for full method comparison)
```bash
# For PyOD methods
pip install pyod

# For enhanced visualizations
pip install scipy
```

---

## 🎯 Benchmark Components

### 1. Dataset Generation (`datasets.py`)

**6 Outlier Types:**

1. **Global Outliers**
   - Far from all normal points
   - Uniform distribution in wide range
   - Easy to detect

2. **Local Outliers**
   - Far from local neighborhood
   - Between clusters
   - Harder to detect

3. **Collective Outliers**
   - Small cluster of outliers
   - Grouped anomalies
   - Context-dependent

4. **Contextual Outliers**
   - Break expected correlations
   - Normal in isolation
   - Anomalous in context

5. **Mixed Outliers**
   - Combination of all types
   - Realistic scenario
   - Most challenging

6. **High-Dimensional**
   - Sparse anomalies
   - Many dimensions
   - Curse of dimensionality

**12 Benchmark Datasets:**

| Dataset | Samples | Features | Contamination | Type |
|---------|---------|----------|---------------|------|
| small_global | 500 | 10 | 10% | Global |
| small_local | 500 | 10 | 10% | Local |
| medium_global | 2,000 | 20 | 5% | Global |
| medium_mixed | 2,000 | 20 | 5% | Mixed |
| large_global | 10,000 | 30 | 2% | Global |
| large_mixed | 10,000 | 30 | 2% | Mixed |
| high_dim_small | 1,000 | 100 | 10% | High-dim |
| high_dim_large | 5,000 | 200 | 5% | High-dim |
| low_contamination | 2,000 | 20 | 1% | Global |
| high_contamination | 2,000 | 20 | 20% | Global |
| collective_only | 2,000 | 20 | 5% | Collective |
| contextual_only | 2,000 | 20 | 5% | Contextual |

### 2. Method Wrappers (`methods.py`)

**Unified Interface:**
```python
class OutlierDetectorWrapper:
    def fit(X) -> self
    def predict(X) -> labels  # Binary: 0=normal, 1=outlier
    def score_samples(X) -> scores  # Higher = more anomalous
```

**Supported Methods:**

**sklearn:**
- IsolationForest
- LocalOutlierFactor (LOF)
- OneClassSVM
- EllipticEnvelope

**PyOD:**
- KNN
- LOF
- OCSVM
- IForest
- COPOD
- ECOD
- HBOS
- PCA

**PSOD:**
- Custom pseudo-supervised approach

### 3. Evaluation Metrics (`metrics.py`)

**Performance Metrics:**
- ROC-AUC (ranking quality)
- Average Precision (PR-AUC)
- Precision, Recall, F1-Score
- Precision@K
- NDCG, MAP

**Efficiency Metrics:**
- Training time (seconds)
- Prediction time (seconds)
- Memory usage (MB)

**Statistical Tests:**
- Paired t-test
- Significance testing
- Confidence intervals

### 4. Visualizations (`visualization.py`)

**10+ Plot Types:**

1. **Method Comparison Bar Chart**
   - Mean performance with error bars
   - Highlight best method
   - Value labels

2. **Performance vs Time Scatter**
   - Identify Pareto frontier
   - Trade-off analysis

3. **Multi-Metric Heatmap**
   - Normalized scores
   - Color-coded performance
   - Annotated values

4. **Scalability Curves**
   - Log-log plots
   - Training vs prediction time
   - Different dataset sizes

5. **Dataset Comparison**
   - Grouped bar charts
   - Performance across datasets
   - Method consistency

6. **ROC Curves**
   - Multiple methods overlay
   - AUC scores in legend
   - Random baseline

7. **Precision-Recall Curves**
   - AP scores in legend
   - Method comparison

8. **Memory Usage**
   - Horizontal bar chart
   - Error bars for variance

9. **Radar Chart**
   - Multi-metric comparison
   - Normalized scores
   - Up to 5 methods

10. **Comprehensive Dashboard**
    - 3x3 grid layout
    - Multiple perspectives
    - Publication-ready

---

## 🧪 Benchmark Tests

### 1. Standard Benchmarking

**Compares methods on multiple datasets**

```python
results_df = runner.benchmark_datasets(
    dataset_names=['small_global', 'medium_mixed'],
    method_subset='basic'
)
```

**Output:**
- ROC-AUC, Average Precision
- Precision, Recall, F1-Score
- Training and prediction time
- Memory usage

### 2. Scalability Testing

**Tests performance vs dataset size**

```python
scalability_results = runner.benchmark_scalability(
    method_subset='fast',
    n_trials=3  # Average over 3 runs
)
```

**Datasets:**
- Samples: 100, 500, 1K, 2K, 5K, 10K
- Features: 10, 20, 50

**Output:**
- Training time curves
- Prediction time curves
- Scalability coefficients

### 3. Robustness Testing

**Tests performance under stress**

```python
robustness_results = runner.benchmark_robustness(
    method_subset='basic'
)
```

**Tests:**
1. **Contamination Levels**: 1%, 5%, 10%, 15%, 20%
2. **Dimensionality**: 10, 50, 100, 200 features
3. **Noise Levels**: Various data conditions

**Output:**
- Performance degradation curves
- Method robustness ranking

---

## 📈 Example Results

### Performance Summary

| Method | ROC-AUC | Avg Precision | Train Time (s) | Pred Time (s) |
|--------|---------|---------------|----------------|---------------|
| **PSOD** | **0.952** | **0.887** | 0.342 | 0.018 |
| IsolationForest | 0.941 | 0.873 | 0.125 | 0.012 |
| LOF | 0.928 | 0.851 | 0.089 | 0.156 |
| COPOD | 0.918 | 0.842 | 0.045 | 0.008 |
| OneClassSVM | 0.903 | 0.821 | 1.234 | 0.203 |

### Method Rankings

1. **PSOD** - Best overall performance
2. **IsolationForest** - Fast and accurate
3. **LOF** - Good for local outliers
4. **COPOD** - Fastest method
5. **OneClassSVM** - Slowest but robust

### Scalability Results

```
Training Time Complexity:
- PSOD: O(n log n)
- IsolationForest: O(n log n)
- LOF: O(n²)
- OneClassSVM: O(n²) to O(n³)
```

---

## 🎨 Visualization Examples

### 1. Method Comparison

![Method Comparison](benchmark_results/method_comparison_roc.png)
*Bar chart comparing ROC-AUC across methods*

### 2. Performance vs Time

![Performance vs Time](benchmark_results/performance_vs_time.png)
*Scatter plot showing accuracy-efficiency trade-off*

### 3. Multi-Metric Heatmap

![Multi-Metric Heatmap](benchmark_results/multi_metric_heatmap.png)
*Heatmap of normalized scores across metrics*

### 4. Scalability Curves

![Scalability](benchmark_results/scalability.png)
*Training and prediction time vs dataset size*

### 5. Comprehensive Dashboard

![Dashboard](benchmark_results/benchmark_dashboard.png)
*Complete overview with multiple visualizations*

---

## 📊 Output Files

### CSV Results (`benchmark_results.csv`)

```csv
method,dataset,n_samples,n_features,contamination,roc_auc,avg_precision,train_time,pred_time,precision,recall,f1_score,memory_total_mb
PSOD,small_global,500,10,0.1,0.952,0.887,0.342,0.018,0.850,0.900,0.874,45.2
IsolationForest,small_global,500,10,0.1,0.941,0.873,0.125,0.012,0.820,0.880,0.849,23.1
...
```

### Markdown Report (`BENCHMARK_REPORT.md`)

```markdown
# PSOD Benchmark Report

## 1. Overall Performance Summary

| Method | ROC-AUC (mean) | ROC-AUC (std) | Train Time (mean) | ...
|--------|----------------|---------------|-------------------|----
| PSOD   | 0.9520         | 0.0123        | 0.3420           | ...
...

## 2. Method Rankings

| Rank | Method | ROC-AUC |
|------|--------|---------|
| 1    | PSOD   | 0.9520  |
...

## 3. Visualizations

- [Method Comparison](method_comparison_roc.png)
- [Performance vs Time](performance_vs_time.png)
...
```

---

## 🔧 Advanced Usage

### Custom Dataset

```python
from datasets import generate_dataset

# Create custom dataset configuration
custom_config = {
    'generator': 'mixed',
    'n_samples': 5000,
    'n_features': 50,
    'contamination': 0.05,
    'description': 'My custom dataset'
}

X, y = generate_dataset(custom_config, random_state=42)

# Benchmark on custom data
# ... (use BenchmarkRunner)
```

### Custom Method

```python
from methods import OutlierDetectorWrapper

# Wrap your custom method
class MyMethod:
    def fit(self, X): ...
    def decision_function(self, X): ...

my_method_wrapper = OutlierDetectorWrapper('MyMethod', MyMethod())

# Add to benchmark
methods = {'MyMethod': my_method_wrapper}
```

### Statistical Comparison

```python
from metrics import statistical_test

# Compare two methods statistically
test_result = statistical_test(
    results_df,
    method1='PSOD',
    method2='IsolationForest',
    metric='roc_auc'
)

print(f"p-value: {test_result['p_value']}")
print(f"Significant: {test_result['significant']}")
```

---

## ⚙️ Configuration

### Method Subsets

- **`'basic'`**: PSOD + IsolationForest + LOF + OneClassSVM
- **`'fast'`**: PSOD + IsolationForest + COPOD + ECOD + HBOS
- **`'accurate'`**: PSOD + IsolationForest + LOF + COPOD
- **`'all'`**: All available methods

### Dataset Selection

```python
# Benchmark on specific datasets
dataset_names = [
    'small_global',        # Quick test
    'medium_mixed',        # Realistic scenario
    'high_dim_small',      # High dimensions
    'low_contamination'    # Rare outliers
]
```

---

## 📝 Best Practices

1. **Start Small**: Use `method_subset='basic'` and few datasets for quick tests
2. **Multiple Runs**: Set `n_trials=5` for stable timing results
3. **Reproducibility**: Always set `random_state` for consistent results
4. **Memory**: Use `track_memory=True` only when needed (adds overhead)
5. **Visualization**: Save plots to avoid recreating expensive computations

---

## 🐛 Troubleshooting

### Issue: "No methods available"
**Solution**: Install optional dependencies
```bash
pip install pyod scikit-learn
```

### Issue: "Memory error"
**Solution**: Reduce dataset size or disable memory tracking
```python
runner.benchmark_method(..., track_memory=False)
```

### Issue: "Slow execution"
**Solution**: Use smaller datasets or fewer trials
```python
runner.benchmark_scalability(n_trials=1)
```

---

## 📚 References

- **PSOD Paper**: [Link to paper when published]
- **sklearn Outlier Detection**: https://scikit-learn.org/stable/modules/outlier_detection.html
- **PyOD Library**: https://github.com/yzhao062/pyod

---

## 🎯 Summary

**Benchmarking Suite Features:**
- ✅ **2,330+ lines** of production-ready code
- ✅ **6 outlier types**, **12 datasets**
- ✅ **10+ methods** (PSOD + sklearn + PyOD)
- ✅ **15+ metrics** (performance + efficiency)
- ✅ **10+ visualizations** (charts + dashboards)
- ✅ **3 test types** (standard + scalability + robustness)
- ✅ **Automated reporting** (CSV + Markdown + PNG)
- ✅ **Statistical testing** (significance + confidence intervals)

**Ready for:**
- Research paper benchmarks
- Method comparison studies
- Performance analysis
- Scalability testing
- Production deployment decisions

---

*Comprehensive benchmarking for outlier detection methods!* 🚀
