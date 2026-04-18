import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sklearn.feature_selection.mutual_info_ import (
    _estimate_mi,
    mutual_info_regression,
    mutual_info_classif
)
from sklearn.utils._testing import assert_array_equal


def test_estimate_mi_with_discrete_features_array():
    """Test _estimate_mi with discrete_features as array of indices."""
    # Create simple test data
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])
    y = np.array([0, 1, 0, 1])
    
    # Test with discrete_features as array of indices
    discrete_features = np.array([0, 2])  # First and third features are discrete
    mi_scores = _estimate_mi(X, y, discrete_features=discrete_features, 
                            discrete_target=True, n_neighbors=3)
    
    assert len(mi_scores) == X.shape[1]
    assert all(score >= 0 for score in mi_scores)  # MI should be non-negative


def test_estimate_mi_with_discrete_features_boolean_mask():
    """Test _estimate_mi with discrete_features as boolean mask."""
    # Create simple test data
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])
    y = np.array([0, 1, 0, 1])
    
    # Test with discrete_features as boolean mask
    discrete_features = np.array([True, False, True])  # First and third features are discrete
    mi_scores = _estimate_mi(X, y, discrete_features=discrete_features, 
                            discrete_target=True, n_neighbors=3)
    
    assert len(mi_scores) == X.shape[1]
    assert all(score >= 0 for score in mi_scores)  # MI should be non-negative


def test_estimate_mi_with_discrete_features_auto():
    """Test _estimate_mi with discrete_features='auto'."""
    # Create simple test data
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])
    y = np.array([0, 1, 0, 1])
    
    # Test with discrete_features='auto' (should work as before)
    mi_scores = _estimate_mi(X, y, discrete_features='auto', 
                            discrete_target=True, n_neighbors=3)
    
    assert len(mi_scores) == X.shape[1]
    assert all(score >= 0 for score in mi_scores)  # MI should be non-negative


def test_mutual_info_regression_with_discrete_features_array():
    """Test mutual_info_regression with discrete_features as array."""
    X = np.array([[1, 2.5, 3],
                  [4, 5.5, 6],
                  [7, 8.5, 9],
                  [10, 11.5, 12]], dtype=float)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Test with discrete_features as array of indices
    discrete_features = np.array([0, 2])  # First and third features are discrete
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    
    assert len(mi_scores) == X.shape[1]
    assert all(score >= 0 for score in mi_scores)


def test_mutual_info_classif_with_discrete_features_array():
    """Test mutual_info_classif with discrete_features as array."""
    X = np.array([[1, 2.5, 3],
                  [4, 5.5, 6],
                  [7, 8.5, 9],
                  [10, 11.5, 12]], dtype=float)
    y = np.array([0, 1, 0, 1])
    
    # Test with discrete_features as array of indices
    discrete_features = np.array([0, 2])  # First and third features are discrete
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    
    assert len(mi_scores) == X.shape[1]
    assert all(score >= 0 for score in mi_scores)


def test_discrete_features_comparison_no_warning():
    """Test that comparing discrete_features array to 'auto' doesn't raise warnings."""
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])
    y = np.array([0, 1, 0, 1])
    
    # This should not raise any numpy warnings about array comparison
    discrete_features = np.array([0, 2])
    
    # Should execute without warnings
    with pytest.warns(None) as warning_list:
        _estimate_mi(X, y, discrete_features=discrete_features, 
                    discrete_target=True, n_neighbors=3)
    
    # Filter out any warnings that are not related to array comparison
    array_comparison_warnings = [
        w for w in warning_list 
        if "comparison" in str(w.message).lower() or "array" in str(w.message).lower()
    ]
    assert len(array_comparison_warnings) == 0
