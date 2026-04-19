import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_blobs

def test_issue_reproduction():
    """Test that KernelPCA with rbf kernel produces consistent signs across runs."""
    # Create a simple dataset
    X, _ = make_blobs(n_samples=50, centers=3, n_features=10, random_state=42)
    
    # Run KernelPCA multiple times with different random states
    results = []
    for random_state in [0, 1, 2]:
        pca = KernelPCA(n_components=7, kernel='rbf', random_state=random_state)
        result = pca.fit_transform(X)
        results.append(result)
    
    # Check if results are consistent (same or negated)
    # The issue is that signs can flip arbitrarily
    result1, result2, result3 = results
    
    # For each component, check if all runs have the same sign pattern
    # This should pass if signs are consistent, but will fail due to the bug
    for i in range(7):  # for each component
        col1, col2, col3 = result1[:, i], result2[:, i], result3[:, i]
        
        # Check if col2 and col3 have the same sign relationship to col1
        # Either both are same sign or both are opposite sign
        same_sign_12 = np.allclose(col1, col2, rtol=1e-10)
        opposite_sign_12 = np.allclose(col1, -col2, rtol=1e-10)
        
        same_sign_13 = np.allclose(col1, col3, rtol=1e-10)
        opposite_sign_13 = np.allclose(col1, -col3, rtol=1e-10)
        
        # At least one should be true for each comparison
        assert same_sign_12 or opposite_sign_12, f"Component {i}: col1 and col2 are neither same nor opposite"
        assert same_sign_13 or opposite_sign_13, f"Component {i}: col1 and col3 are neither same nor opposite"
        
        # The key assertion: if col2 has same sign as col1, then col3 should too
        # This will fail due to inconsistent sign flipping
        if same_sign_12:
            assert same_sign_13, f"Component {i}: Inconsistent signs - col2 same as col1 but col3 different"
        else:  # opposite_sign_12 is True
            assert opposite_sign_13, f"Component {i}: Inconsistent signs - col2 opposite to col1 but col3 same"