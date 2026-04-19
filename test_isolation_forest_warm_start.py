import pytest
import numpy as np
from sklearn.ensemble import IsolationForest

def test_issue_reproduction():
    """Test that warm_start parameter should be exposed in IsolationForest.__init__()"""
    # This should work but currently fails because warm_start is not exposed
    X = np.random.RandomState(42).randn(100, 4)
    
    # Try to create IsolationForest with warm_start parameter
    # This will fail with TypeError on current code
    clf = IsolationForest(n_estimators=5, warm_start=True, random_state=42)
    
    # Fit initial model
    clf.fit(X)
    initial_estimators = len(clf.estimators_)
    
    # Increase n_estimators and refit - should add more trees
    clf.n_estimators = 10
    clf.fit(X)
    final_estimators = len(clf.estimators_)
    
    # With warm_start=True, we should have added trees incrementally
    assert final_estimators == 10
    assert final_estimators > initial_estimators