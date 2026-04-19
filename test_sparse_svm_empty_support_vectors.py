import numpy as np
import scipy.sparse as sp
from sklearn.svm import SVC
import pytest

def test_issue_reproduction():
    # Create sparse data that will result in empty support_vectors_
    # This happens when all samples are perfectly separable with large margin
    # or when C is very small
    X = sp.csr_matrix([[1, 0], [0, 1], [2, 0], [0, 2]])
    y = np.array([0, 1, 0, 1])
    
    # Use very small C to potentially get no support vectors
    # and linear kernel with sparse data to trigger _sparse_fit
    svm = SVC(kernel='linear', C=1e-10)
    
    # This should trigger the ZeroDivisionError in _sparse_fit
    # when support_vectors_ is empty
    svm.fit(X, y)