import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.utils._pprint import _EstimatorPrettyPrinter

def test_issue_reproduction():
    """Test that print_changed_only fails with vector values due to ambiguous array truth value."""
    
    class MockEstimator(BaseEstimator):
        def __init__(self, vector_param=None):
            self.vector_param = vector_param
    
    # Create an estimator with a numpy array parameter
    estimator = MockEstimator(vector_param=np.array([1, 2, 3]))
    
    # Enable print_changed_only which should trigger the bug
    sklearn.set_config(print_changed_only=True)
    
    try:
        # This should fail with "ValueError: The truth value of an array with more than one element is ambiguous"
        repr(estimator)
    finally:
        # Reset config
        sklearn.set_config(print_changed_only=False)