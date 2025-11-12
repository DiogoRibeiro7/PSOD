# Outlier Detection using Pseudo-Supervised Learning (PSOD)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

PSOD (Pseudo-Supervised Outlier Detection) is a novel approach for detecting outliers in tabular data by treating each feature as a target variable and using the prediction errors from regression models as outlier scores.

## Key Features

- 🚀 **Flexible Architecture**: Supports any scikit-learn compatible regressor as base learner
- 📊 **Mixed Data Types**: Handles both numerical and categorical features
- 🔄 **Multiple Transformations**: Supports logarithmic, Yeo-Johnson, and no transformation
- 🎯 **Customizable Detection**: Configure outlier detection on low end, high end, or both

## Installation

```bash
# TODO: Add PyPI installation instructions once package is published
pip install outlier-pseudo-supervised
```

### From Source
```bash
git clone https://github.com/diogoribeiro7/outlier_pseudo_supervised.git
cd outlier_pseudo_supervised
pip install -e .
```

## Quick Start

```python
from psod import PSOD
import pandas as pd

# Load your data
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 100],
    'feature2': [10, 20, 30, 40, 1000],
    'category': ['A', 'B', 'A', 'B', 'A']
})

# Initialize PSOD
detector = PSOD(cat_columns=['category'])

# Detect outliers
outlier_scores = detector.fit_predict(df)
```

## Documentation

# TODO: Add link to full documentation once Sphinx docs are generated
Full documentation available at [https://outlier-pseudo-supervised.readthedocs.io](https://outlier-pseudo-supervised.readthedocs.io)

## Benchmarks

# TODO: Add comprehensive benchmark results comparing PSOD with other outlier detection methods
Performance comparisons coming soon.

## Contributing

# TODO: Create CONTRIBUTING.md with detailed contribution guidelines
We welcome contributions! Please see our Contributing Guide for details.

## Citation

# TODO: Add proper citation format once paper/preprint is published
If you use this software in your research, please cite:

```bibtex
@software{ribeiro2024psod,
  author = {Ribeiro, Diogo},
  title = {PSOD: Pseudo-Supervised Outlier Detection},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/diogoribeiro7/outlier_pseudo_supervised}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

Diogo Ribeiro - [@diogoribeiro7](https://github.com/diogoribeiro7)

Project Link: [https://github.com/diogoribeiro7/outlier_pseudo_supervised](https://github.com/diogoribeiro7/outlier_pseudo_supervised)
