Welcome to PSOD Documentation
============================

PSOD (Pseudo-Supervised Outlier Detection) is a flexible and powerful library for detecting outliers in tabular data using a pseudo-supervised learning approach.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   quickstart
   api/index
   examples/index
   theory
   benchmarks
   contributing
   changelog

Quick Example
------------

.. code-block:: python

   from psod import PSOD
   import pandas as pd

   # Load your data
   df = pd.read_csv('your_data.csv')

   # Initialize detector
   detector = PSOD(cat_columns=['category_column'])

   # Detect outliers
   outlier_scores = detector.fit_predict(df)

Features
--------

- **Flexible Architecture**: Use any scikit-learn regressor as base learner
- **Mixed Data Types**: Handle both numerical and categorical features
- **Multiple Transformations**: Support for various data transformations
- **Customizable Detection**: Configure detection thresholds and directions
- **Production Ready**: Model persistence and comprehensive error handling

.. TODO: Add feature comparison table
.. TODO: Add architecture diagram
.. TODO: Add performance benchmarks

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. TODO: Add links to GitHub, PyPI, documentation
.. TODO: Add badges for build status, coverage, etc.