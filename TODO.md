# PSOD Repository TODO List

This file consolidates all TODOs from the repository for easy tracking.

## High Priority

### Core Implementation (src/psod/core.py)
- [ ] Add logging throughout the class
- [ ] Add contamination parameter for automatic threshold selection
- [ ] Add feature_importances_ attribute
- [ ] Track fitting status properly
- [ ] Add __repr__ method for better debugging
- [ ] Add comprehensive input validation
- [ ] Handle missing values
- [ ] Add parallel processing option using joblib
- [ ] Calculate and store feature importances
- [ ] Add save_model method
- [ ] Add load_model class method
- [ ] Add get_feature_importance method
- [ ] Add score_samples method (similar to sklearn)

### Testing (tests/test_psod.py)
- [ ] Add fixture for time series data
- [ ] Add fixture for high-dimensional data
- [ ] Test invalid transform_algorithm
- [ ] Test that model is marked as fitted
- [ ] Test predict without fitting
- [ ] Test with RandomForest
- [ ] Test empty DataFrame
- [ ] Test missing value handling
- [ ] Test reproducibility
- [ ] Test different outlier flagging options
- [ ] Test categorical encoders
- [ ] Test correlation threshold effect
- [ ] Test save and load functionality
- [ ] Test feature importance
- [ ] Test parallel processing
- [ ] Add integration tests
- [ ] Add performance tests

### Documentation
- [ ] Add PyPI installation instructions once package is published
- [ ] Add link to full documentation once Sphinx docs are generated
- [ ] Add comprehensive benchmark results
- [ ] Create CONTRIBUTING.md with detailed contribution guidelines
- [ ] Create CODE_OF_CONDUCT.md
- [ ] Create bug report issue template
- [ ] Create feature request issue template
- [ ] Set up Sphinx documentation
- [ ] Add proper citation format once paper/preprint is published

## Medium Priority

### Utilities (src/psod/utils.py)
- [ ] Implement model serialization functions
- [ ] Implement different serialization formats
- [ ] Add compression options
- [ ] Save model metadata (version, timestamp, etc.)
- [ ] Implement loading for different formats
- [ ] Add version compatibility checking
- [ ] Validate loaded model integrity
- [ ] Add data validation utilities
- [ ] Add preprocessing utilities
- [ ] Add outlier score utilities
- [ ] Add ensemble utilities
- [ ] Add performance evaluation utilities
- [ ] Add data generation utilities for testing

### Visualization (src/psod/visualization.py)
- [ ] Implement basic outlier score visualization
- [ ] Implement feature contribution plot
- [ ] Implement 2D/3D scatter plot with outliers
- [ ] Implement time series outlier visualization
- [ ] Implement correlation heatmap with outlier indicators
- [ ] Implement outlier score evolution plot
- [ ] Implement ROC and PR curves
- [ ] Implement outlier summary dashboard
- [ ] Add interactive outlier explorer

### CI/CD (.github/workflows/ci.yml)
- [ ] Add schedule for nightly builds
- [ ] Add concurrency control
- [ ] Add ruff for modern linting
- [ ] Enable mypy type checking
- [ ] Add Python 3.12 once all dependencies support it
- [ ] Consider reducing OS matrix for faster CI
- [ ] Add slow test runs on schedule only
- [ ] Add codecov token as secret
- [ ] Add integration tests
- [ ] Add benchmark tests
- [ ] Add documentation build
- [ ] Add security scanning
- [ ] Add release workflow

### Build Configuration
- [ ] Update version management in setup.py
- [ ] Update author email with actual email
- [ ] Add more classifiers once package is more mature
- [ ] Add Python 3.12 support and testing
- [ ] Add pre-commit to dev dependencies
- [ ] Add 'docs' extra with documentation dependencies
- [ ] Add 'all' extra that includes everything
- [ ] Add documentation URL once docs are hosted
- [ ] Add project.scripts for CLI commands

## Low Priority

### Examples (examples/basic_usage.py)
- [ ] Update import once package is installed
- [ ] Add visualization to basic example
- [ ] Implement advanced example
- [ ] Implement time series example
- [ ] Implement comparison example
- [ ] Implement real-world example

### Benchmarks (benchmarks/run_benchmarks.py)
- [ ] Import outlier detection methods
- [ ] Import PSOD
- [ ] Implement different outlier types
- [ ] Add more metrics (precision, recall, F1)
- [ ] Add memory usage tracking
- [ ] Handle different prediction methods
- [ ] Add real-world datasets
- [ ] Uncomment method implementations
- [ ] Implement benchmark visualization
- [ ] Implement scalability tests
- [ ] Implement robustness tests
- [ ] Save as LaTeX table
- [ ] Generate markdown report
- [ ] Generate comprehensive report

### Development Tools
- [ ] Create .pre-commit-config.yaml
- [ ] Install pre-commit hooks
- [ ] Add ruff configuration
- [ ] Add bandit for security checks
- [ ] Enable mypy in Makefile
- [ ] Add docs building in Makefile
- [ ] Add release command in Makefile
- [ ] Add benchmark command in Makefile
- [ ] Add profile command in Makefile
- [ ] Add security check in Makefile
- [ ] Add Docker commands in Makefile
- [ ] Add notebook examples in Makefile

### Documentation Configuration (docs/)
- [ ] Use dynamic versioning
- [ ] Add more Sphinx extensions
- [ ] Configure autodoc
- [ ] Configure theme options
- [ ] Add logo and favicon
- [ ] Configure intersphinx mappings
- [ ] Add feature comparison table
- [ ] Add architecture diagram
- [ ] Add performance benchmarks
- [ ] Add links to GitHub, PyPI, documentation
- [ ] Add badges for build status, coverage, etc.

## Future Enhancements

### Algorithm Improvements
- [ ] Add Box-Cox transformation option
- [ ] Add quantile transformation option
- [ ] Support for sparse matrices
- [ ] Implement incremental/online learning
- [ ] Add early stopping for large datasets
- [ ] Add ensemble methods beyond single base learner
- [ ] Add support for time series data
- [ ] Add support for GPU acceleration
- [ ] Add interpretability features (SHAP integration)
- [ ] Create interactive visualization dashboard
- [ ] Add AutoML capabilities for hyperparameter tuning
- [ ] Support for multi-output outlier detection

### Repository Structure
- [ ] Move psod.py to src/psod/core.py
- [ ] Create proper package structure
- [ ] Add Docker support
- [ ] Add Jupyter notebook examples
- [ ] Create example datasets
- [ ] Add performance profiling tools
- [ ] Set up documentation hosting
- [ ] Create project website

## Notes

- TODOs are scattered throughout the codebase and marked with "TODO:" comments
- The GitHub Actions workflow will automatically create issues from these TODOs
- Priority levels are subjective and should be adjusted based on user needs
- Some TODOs depend on others (e.g., can't test save/load until it's implemented)
