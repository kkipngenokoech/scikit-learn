import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

def test_issue_reproduction():
    # Set pandas output configuration
    set_config(transform_output="pandas")
    
    try:
        # Create a simple dataset
        X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
        
        # Create a calibrated classifier with isotonic regression
        base_clf = GaussianNB()
        calibrated_clf = CalibratedClassifierCV(base_clf, method='isotonic', cv=3)
        
        # Fit the classifier
        calibrated_clf.fit(X, y)
        
        # This should fail with the current code when isotonic regression returns pandas DataFrame
        probas = calibrated_clf.predict_proba(X)
        
        # If we get here, the bug is fixed
        assert isinstance(probas, np.ndarray)
        assert probas.shape == (100, 2)
    finally:
        # Reset config to avoid affecting other tests
        set_config(transform_output="default")