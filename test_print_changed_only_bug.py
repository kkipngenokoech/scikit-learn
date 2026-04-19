import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn import set_config

class DummyEstimator(BaseEstimator):
    def __init__(self, array_param=None):
        self.array_param = array_param

def test_issue_reproduction():
    # Enable print_changed_only which triggers the bug
    set_config(print_changed_only=True)
    
    # Create estimator with numpy array parameter
    arr = np.array([1, 2, 3])
    estimator = DummyEstimator(array_param=arr)
    
    # This should trigger the ValueError when comparing arrays
    repr(estimator)