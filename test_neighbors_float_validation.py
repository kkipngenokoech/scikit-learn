import pytest
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs

def test_issue_reproduction():
    # Create sample data
    X, _ = make_blobs(n_samples=10, centers=2, random_state=42)
    
    # Fit the estimator
    neighbors = NearestNeighbors(n_neighbors=3)
    neighbors.fit(X)
    
    # This should either work (by casting to int) or raise a clear ValueError
    # Currently it raises a cryptic TypeError from Cython code
    with pytest.raises(TypeError, match="'float' object cannot be interpreted as an integer"):
        neighbors.kneighbors(X, n_neighbors=3.0)