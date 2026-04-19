import numpy as np
import scipy.sparse as sp
import pytest

from sklearn.svm import SVC
from sklearn.datasets import make_classification


def test_svm_sparse_empty_support_vectors():
    """Test that SVM with sparse data handles empty support vectors correctly.
    
    This is a regression test for issue #14894 where a ZeroDivisionError
    was raised when support_vectors_ was empty in sparse fitting.
    """
    # Create a simple dataset where SVM might not find support vectors
    # This can happen with certain data distributions and parameters
    X = sp.csr_matrix([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([1, 1, 1, 1])  # All same class
    
    # Use parameters that might lead to no support vectors
    svm = SVC(kernel='linear', C=1e-10)  # Very small C
    
    # This should not raise a ZeroDivisionError
    try:
        svm.fit(X, y)
        # If we get here, the fix worked
        assert hasattr(svm, 'dual_coef_')
        # dual_coef_ should be a sparse matrix
        assert sp.issparse(svm.dual_coef_)
    except ZeroDivisionError:
        pytest.fail("ZeroDivisionError was raised when fitting SVM with sparse data")


def test_svm_sparse_normal_case():
    """Test that normal sparse SVM functionality still works."""
    # Create a dataset that should have support vectors
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, 
                              random_state=42)
    X_sparse = sp.csr_matrix(X)
    
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_sparse, y)
    
    # Should have support vectors and dual coefficients
    assert hasattr(svm, 'support_vectors_')
    assert hasattr(svm, 'dual_coef_')
    assert sp.issparse(svm.dual_coef_)
    assert svm.support_vectors_.shape[0] > 0
    assert svm.dual_coef_.shape[1] == svm.support_vectors_.shape[0]
