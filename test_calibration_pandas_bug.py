import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

def test_issue_reproduction():
    """Test that CalibratedClassifierCV with isotonic regression fails with pandas output config."""
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
    
    # Set pandas output configuration
    set_config(transform_output="pandas")
    
    try:
        # Create calibrated classifier with isotonic regression
        clf = CalibratedClassifierCV(
            estimator=GaussianNB(),
            method="isotonic",
            cv=3
        )
        
        # Fit the classifier
        clf.fit(X, y)
        
        # This should fail due to the pandas DataFrame output issue
        clf.predict_proba(X)
        
    finally:
        # Reset config to avoid affecting other tests
        set_config(transform_output="default")