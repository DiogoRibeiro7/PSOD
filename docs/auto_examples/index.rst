:orphan:

Examples and Tutorials
======================

This section contains examples and tutorials for using PSOD (Pseudo-Supervised Outlier Detection).

Quick Start
-----------

Run the basic example:

.. code-block:: bash

   python basic_usage.py

Other scripts:

- ``advanced_usage.py``: advanced techniques and parameter tuning
- ``time_series_example.py``: time series anomaly detection
- ``comparison_example.py``: comparisons with other methods
- ``cli_examples.py``: command-line interface examples

Notebooks
---------

Interactive notebooks are available under ``notebooks/``. To run them:

.. code-block:: bash

   cd notebooks
   jupyter notebook

Requirements
------------

.. code-block:: bash

   pip install pandas numpy scikit-learn matplotlib seaborn
   pip install category_encoders tqdm joblib
   pip install jupyter ipykernel  # optional for notebooks
   pip install statsmodels        # optional for time series
   pip install xgboost            # optional for comparison examples

Toctree
-------

.. toctree::
   :maxdepth: 1

   /auto_examples/basic_usage
   /auto_examples/time_series_example
   /auto_examples/advanced_usage
   /auto_examples/cli_examples
   /auto_examples/comparison_example
