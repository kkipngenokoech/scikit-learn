import numpy as np
import pytest
from sklearn.feature_selection.mutual_info_ import _estimate_mi

def test_issue_reproduction():
    # Create simple test data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    
    # Test with discrete_features as an array of indices
    discrete_features = np.array([0])  # First feature is discrete
    
    # This should trigger the bug: comparing array to string 'auto'
    # The comparison discrete_features == 'auto' will cause numpy warning/error
    with pytest.warns(FutureWarning, match="elementwise comparison failed"):
        _estimate_mi(X, y, discrete_features=discrete_features, discrete_target=True)