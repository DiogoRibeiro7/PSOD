# PSOD Examples and Tutorials

This directory contains comprehensive examples and tutorials for using PSOD (Pseudo-Supervised Outlier Detection).

## 📁 Directory Structure

```
examples/
├── README.md                      # This file
├── basic_usage.py                 # Basic outlier detection workflow
├── advanced_usage.py              # Advanced techniques and parameter tuning
├── time_series_example.py         # Time series anomaly detection
├── comparison_example.py          # Comparison with other methods
├── cli_examples.py               # Command-line interface examples
└── notebooks/                     # Jupyter notebooks
    ├── 01_basic_tutorial.ipynb
    ├── 02_advanced_tutorial.ipynb
    └── 03_real_world_case_studies.ipynb
```

## 🚀 Quick Start

### Basic Usage

Start with the basic example to understand the core workflow:

```bash
python basic_usage.py
```

This demonstrates:
- Loading and preparing data
- Initializing PSOD detector
- Detecting outliers
- Visualizing results

### Advanced Usage

Explore advanced techniques with:

```bash
python advanced_usage.py
```

Topics covered:
- Custom base learners (Random Forest, Gradient Boosting)
- Different transformation algorithms
- Parameter tuning and optimization
- Feature importance analysis
- Model persistence (save/load)
- Comprehensive dashboards

### Time Series Analysis

Learn time series outlier detection:

```bash
python time_series_example.py
```

Includes:
- Temporal feature engineering
- Sliding window approach
- Seasonal decomposition
- Multivariate time series

### Method Comparison

Compare PSOD with other outlier detection methods:

```bash
python comparison_example.py
```

Compares with:
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- Elliptic Envelope

Includes performance metrics, visualizations, and scalability analysis.

## 📓 Jupyter Notebooks

Interactive tutorials for deeper learning:

### 1. Basic Tutorial (`01_basic_tutorial.ipynb`)

Perfect for beginners. Covers:
- Installation and setup
- Basic workflow
- Understanding outlier scores
- Visualization techniques
- Working with real data
- Model persistence

### 2. Advanced Tutorial (`02_advanced_tutorial.ipynb`)

For intermediate users. Includes:
- Custom base learners
- Systematic parameter tuning
- Ensemble methods
- Time series detection
- Interactive dashboards
- Performance optimization

### 3. Real-World Case Studies (`03_real_world_case_studies.ipynb`)

Practical applications:
- **Credit Card Fraud Detection**
  - Imbalanced datasets
  - Real-time detection
  - Business impact analysis

- **Network Intrusion Detection**
  - High-dimensional data
  - Attack pattern recognition
  - False alarm minimization

- **IoT Sensor Monitoring**
  - Multivariate time series
  - Equipment failure detection
  - Early warning systems

### Running Notebooks

```bash
cd notebooks
jupyter notebook
```

Or use JupyterLab:

```bash
jupyter lab
```

## 🖥️ Command-Line Interface

The CLI provides production-ready tools for batch processing:

### Detect Outliers

```bash
python cli_examples.py detect \
    --input data.csv \
    --output results.csv \
    --cat-columns "category,region" \
    --stdevs 2.5
```

### Train and Save Model

```bash
python cli_examples.py train \
    --input train_data.csv \
    --model fraud_detector.pkl \
    --cat-columns "merchant_category,card_type"
```

### Score New Data

```bash
python cli_examples.py score \
    --input new_transactions.csv \
    --model fraud_detector.pkl \
    --output scored_transactions.csv
```

### Generate HTML Report

```bash
python cli_examples.py report \
    --input data.csv \
    --output report.html
```

### Evaluate with Ground Truth

```bash
python cli_examples.py evaluate \
    --input labeled_data.csv \
    --labels is_fraud
```

### CLI Options

Common options for all commands:

- `--input, -i`: Input CSV file (required)
- `--cat-columns`: Comma-separated categorical column names
- `--min-cols`: Minimum fraction of columns to use (default: 0.5)
- `--max-cols`: Maximum fraction of columns to use (default: 1.0)
- `--stdevs`: Standard deviations for outlier threshold (default: 2.0)
- `--transform`: Transformation algorithm (default: 'logarithmic')
  - Options: 'logarithmic', 'yeo-johnson', 'quantile', 'box-cox', None
- `--contamination`: Expected outlier fraction (default: 0.1)
- `--seed`: Random seed for reproducibility (default: 42)

## 📊 Example Output

### Console Output

```
=== PSOD Detection Complete ===
Total samples: 10000
Outliers detected: 98 (0.98%)

Score Statistics:
  Mean: 0.0234
  Std:  0.0456
  Min:  0.0001
  Max:  0.8923

Top 5 Outliers:
  Sample 9523: score=0.8923
  Sample 8741: score=0.7654
  Sample 5432: score=0.6789
  ...
```

### Generated Visualizations

Each script generates PNG visualizations:
- `basic_outlier_scores.png` - Score distribution
- `basic_outliers_scatter.png` - 2D scatter plot
- `advanced_parameter_tuning.png` - Tuning results
- `comparison_metrics.png` - Method comparison
- And many more...

### HTML Reports

The CLI can generate comprehensive HTML reports with:
- Summary statistics
- Interactive visualizations
- Top outliers list
- Feature importance
- Detection parameters

## 🎯 Use Case Selection Guide

Choose the right example based on your needs:

| Use Case | Recommended Example |
|----------|---------------------|
| Learning PSOD basics | `basic_usage.py` or Notebook 01 |
| Production deployment | `cli_examples.py` |
| Parameter optimization | `advanced_usage.py` |
| Time series data | `time_series_example.py` |
| Method selection | `comparison_example.py` |
| Fraud detection | Notebook 03 (Case Study 1) |
| Network security | Notebook 03 (Case Study 2) |
| IoT monitoring | Notebook 03 (Case Study 3) |

## 🔧 Requirements

Install required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install category_encoders tqdm joblib

# Optional for notebooks
pip install jupyter ipykernel

# Optional for time series
pip install statsmodels

# Optional for comparison examples
pip install xgboost
```

Or install all at once:

```bash
pip install -r ../requirements.txt
```

## 📚 Learning Path

Recommended learning sequence:

1. **Start Here** → `basic_usage.py`
2. **Interactive Learning** → Notebook 01 (Basic Tutorial)
3. **Deep Dive** → `advanced_usage.py`
4. **Advanced Concepts** → Notebook 02 (Advanced Tutorial)
5. **Specific Applications**:
   - Time series → `time_series_example.py`
   - Comparisons → `comparison_example.py`
   - Real-world → Notebook 03 (Case Studies)
6. **Production Use** → `cli_examples.py`

## 💡 Tips and Best Practices

### Data Preparation

- **Handle missing values** before passing to PSOD
- **Specify categorical columns** explicitly
- **Scale features** if they have very different ranges (PSOD handles this internally)

### Parameter Tuning

- Start with **default parameters**
- Adjust `stdevs_to_outlier` to control sensitivity:
  - Lower (1.5-2.0) = more outliers detected
  - Higher (2.5-3.0) = fewer, more extreme outliers
- Use `contamination` to set expected outlier rate

### Performance Optimization

- Use `sample_frac < 1.0` for large datasets
- Reduce `max_cols_chosen` for high-dimensional data
- Set `n_jobs=-1` to use all CPU cores
- Choose simpler base learners (Linear) for speed

### Visualization

- Always visualize results before taking action
- Use score distributions to understand thresholds
- Check feature importance for interpretability
- Validate with domain knowledge

## 🐛 Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'psod'"**
- Solution: Ensure PSOD is installed or add parent directory to path

**"ValueError: could not convert string to float"**
- Solution: Specify categorical columns with `cat_columns` parameter

**Memory error on large datasets**
- Solution: Use `sample_frac` parameter or process in batches

**Poor detection performance**
- Solution: Try different transformation algorithms or tune parameters

## 📖 Additional Resources

- **Documentation**: See `../docs/` directory
- **API Reference**: Check docstrings in source code
- **Benchmarks**: See `../benchmarks/` directory
- **Tests**: See `../tests/` for usage examples

## 🤝 Contributing

Have a cool example or use case? Contributions are welcome!

1. Create your example script
2. Add documentation
3. Include sample output
4. Submit a pull request

## 📄 License

These examples are part of the PSOD project. See LICENSE file in the root directory.

## 📧 Support

For questions or issues:
- Check existing examples and notebooks
- Review documentation
- Open an issue on GitHub

---

**Happy Outlier Detecting! 🎯**
