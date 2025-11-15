"""
Outlier detection method wrappers for benchmarking.
Provides consistent interface for different outlier detection methods.
"""

import warnings
from typing import Any, Optional, cast

import numpy as np

warnings.filterwarnings("ignore")


class OutlierDetectorWrapper:
    """Base wrapper class for outlier detection methods."""

    def __init__(self, name: str, model: Any):
        self.name = name
        self.model = model
        self._fitted = False

    def fit(self, X: np.ndarray) -> "OutlierDetectorWrapper":
        """Fit the model."""
        self.model.fit(X)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outlier labels (0=normal, 1=outlier)."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        # Different methods have different prediction interfaces
        if hasattr(self.model, "predict"):
            predictions = self.model.predict(X)
            # Convert sklearn format (-1, 1) to (1, 0)
            if np.min(predictions) == -1:
                return cast(np.ndarray, (predictions == -1).astype(int))
            return cast(np.ndarray, predictions)
        else:
            raise NotImplementedError(f"Predict not available for {self.name}")

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get outlier scores."""
        if not self._fitted:
            raise ValueError("Model must be fitted before scoring")

        # Try different scoring methods
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            # Negative scores = outliers for some methods
            return cast(np.ndarray, -scores)
        elif hasattr(self.model, "score_samples"):
            scores_ = self.model.score_samples(X)
            # Negative scores = outliers
            return cast(np.ndarray, -scores_)
        elif hasattr(self.model, "predict_proba"):
            # For probabilistic methods
            proba = self.model.predict_proba(X)
            if proba.shape[1] == 2:
                return cast(np.ndarray, proba[:, 1])  # Outlier probability
            return cast(np.ndarray, proba)
        else:
            # Fallback to binary predictions
            return self.predict(X).astype(float)


def get_sklearn_methods(contamination: float = 0.1, random_state: Optional[int] = None):
    """
    Get sklearn-based outlier detection methods.

    Parameters
    ----------
    contamination : float
        Expected proportion of outliers
    random_state : int, optional
        Random seed

    Returns
    -------
    methods : dict
        Dictionary of method name -> wrapped model
    """
    methods = {}

    try:
        from sklearn.ensemble import IsolationForest

        methods["IsolationForest"] = OutlierDetectorWrapper(
            "IsolationForest",
            IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1),
        )
    except ImportError:
        pass

    try:
        from sklearn.neighbors import LocalOutlierFactor

        methods["LOF"] = OutlierDetectorWrapper(
            "LOF", LocalOutlierFactor(contamination=contamination, novelty=True, n_jobs=-1)
        )
    except ImportError:
        pass

    try:
        from sklearn.svm import OneClassSVM

        methods["OneClassSVM"] = OutlierDetectorWrapper(
            "OneClassSVM", OneClassSVM(nu=contamination)
        )
    except ImportError:
        pass

    try:
        from sklearn.covariance import EllipticEnvelope

        methods["EllipticEnvelope"] = OutlierDetectorWrapper(
            "EllipticEnvelope",
            EllipticEnvelope(contamination=contamination, random_state=random_state),
        )
    except ImportError:
        pass

    return methods


def get_pyod_methods(contamination: float = 0.1):
    """
    Get PyOD-based outlier detection methods.

    Parameters
    ----------
    contamination : float
        Expected proportion of outliers

    Returns
    -------
    methods : dict
        Dictionary of method name -> wrapped model
    """
    methods = {}

    try:
        from pyod.models.knn import KNN

        methods["KNN"] = OutlierDetectorWrapper("KNN", KNN(contamination=contamination))
    except ImportError:
        pass

    try:
        from pyod.models.lof import LOF as PyOD_LOF

        methods["PyOD_LOF"] = OutlierDetectorWrapper(
            "PyOD_LOF", PyOD_LOF(contamination=contamination)
        )
    except ImportError:
        pass

    try:
        from pyod.models.ocsvm import OCSVM

        methods["PyOD_OCSVM"] = OutlierDetectorWrapper(
            "PyOD_OCSVM", OCSVM(contamination=contamination)
        )
    except ImportError:
        pass

    try:
        from pyod.models.iforest import IForest

        methods["PyOD_IForest"] = OutlierDetectorWrapper(
            "PyOD_IForest", IForest(contamination=contamination)
        )
    except ImportError:
        pass

    try:
        from pyod.models.copod import COPOD

        methods["COPOD"] = OutlierDetectorWrapper("COPOD", COPOD(contamination=contamination))
    except ImportError:
        pass

    try:
        from pyod.models.ecod import ECOD

        methods["ECOD"] = OutlierDetectorWrapper("ECOD", ECOD(contamination=contamination))
    except ImportError:
        pass

    try:
        from pyod.models.hbos import HBOS

        methods["HBOS"] = OutlierDetectorWrapper("HBOS", HBOS(contamination=contamination))
    except ImportError:
        pass

    try:
        from pyod.models.pca import PCA as PyOD_PCA

        methods["PyOD_PCA"] = OutlierDetectorWrapper(
            "PyOD_PCA", PyOD_PCA(contamination=contamination)
        )
    except ImportError:
        pass

    return methods


def get_psod_method(contamination: Optional[float] = None, random_state: Optional[int] = None):
    """
    Get PSOD method.

    Parameters
    ----------
    contamination : float, optional
        Expected proportion of outliers
    random_state : int, optional
        Random seed

    Returns
    -------
    method : OutlierDetectorWrapper
        Wrapped PSOD model
    """
    try:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

        from psod import PSOD

        model = PSOD(contamination=contamination, random_seed=random_state, n_jobs=-1)

        return OutlierDetectorWrapper("PSOD", model)

    except ImportError as e:
        print(f"Warning: Could not import PSOD: {e}")
        return None


def get_all_methods(contamination: float = 0.1, random_state: Optional[int] = None):
    """
    Get all available outlier detection methods.

    Parameters
    ----------
    contamination : float
        Expected proportion of outliers
    random_state : int, optional
        Random seed

    Returns
    -------
    methods : dict
        Dictionary of method name -> wrapped model
    """
    methods = {}

    # Add PSOD
    psod = get_psod_method(contamination, random_state)
    if psod is not None:
        methods["PSOD"] = psod

    # Add sklearn methods
    sklearn_methods = get_sklearn_methods(contamination, random_state)
    methods.update(sklearn_methods)

    # Add PyOD methods
    pyod_methods = get_pyod_methods(contamination)
    methods.update(pyod_methods)

    return methods


def get_method_subset(
    methods: str = "fast", contamination: float = 0.1, random_state: Optional[int] = None
):
    """
    Get subset of methods based on category.

    Parameters
    ----------
    methods : str
        Category of methods: 'fast', 'accurate', 'all', or 'basic'
    contamination : float
        Expected proportion of outliers
    random_state : int, optional
        Random seed

    Returns
    -------
    methods_dict : dict
        Dictionary of selected methods
    """
    all_methods = get_all_methods(contamination, random_state)

    if methods == "all":
        return all_methods

    elif methods == "basic":
        # Basic sklearn methods + PSOD
        basic_names = ["PSOD", "IsolationForest", "LOF", "OneClassSVM"]
        return {k: v for k, v in all_methods.items() if k in basic_names}

    elif methods == "fast":
        # Fast methods suitable for large datasets
        fast_names = ["PSOD", "IsolationForest", "COPOD", "ECOD", "HBOS"]
        return {k: v for k, v in all_methods.items() if k in fast_names}

    elif methods == "accurate":
        # Methods known for high accuracy
        accurate_names = ["PSOD", "IsolationForest", "LOF", "COPOD"]
        return {k: v for k, v in all_methods.items() if k in accurate_names}

    else:
        return all_methods
