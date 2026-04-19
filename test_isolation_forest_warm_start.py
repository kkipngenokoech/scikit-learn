import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

def test_issue_reproduction():
    """Test that warm_start parameter should be exposed in IsolationForest.__init__()"""
    # Generate sample data
    X, _ = make_blobs(n_samples=100, centers=1, n_features=2, random_state=42)
    
    # Try to initialize IsolationForest with warm_start parameter
    # This should work if the parameter is properly exposed
    try:
        clf = IsolationForest(n_estimators=5, warm_start=True, random_state=42)
        # If we get here, the parameter is exposed (test should pass after fix)
        clf.fit(X)
        initial_estimators = len(clf.estimators_)
        
        # Increment n_estimators and refit to add more trees
        clf.n_estimators = 10
        clf.fit(X)
        final_estimators = len(clf.estimators_)
        
        # Should have added 5 more estimators
        assert final_estimators == 10
        assert final_estimators > initial_estimators
        
    except TypeError as e:
        # This is what happens currently - warm_start is not exposed
        if "warm_start" in str(e):
            # Demonstrate that it works when set manually (the workaround mentioned in issue)
            clf = IsolationForest(n_estimators=5, random_state=42)
            clf.warm_start = True  # Set it manually after initialization
            clf.fit(X)
            initial_estimators = len(clf.estimators_)
            
            # Increment n_estimators and refit
            clf.n_estimators = 10
            clf.fit(X)
            final_estimators = len(clf.estimators_)
            
            # This works, proving the functionality exists but isn't exposed
            assert final_estimators == 10
            
            # Fail the test because warm_start should be exposed in __init__
            raise AssertionError("warm_start parameter is not exposed in IsolationForest.__init__() but should be")
        else:
            raise