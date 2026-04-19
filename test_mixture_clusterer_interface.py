import numpy as np
from sklearn.mixture import GaussianMixture

def test_issue_reproduction():
    """Test that mixture models have clusterer-compatible interface."""
    # Create sample data
    X = np.array([[0, 0], [1, 1], [2, 2], [10, 10], [11, 11], [12, 12]])
    
    # Test 1: n_clusters parameter should work as alias for n_components
    try:
        gmm = GaussianMixture(n_clusters=2)
        assert False, "n_clusters parameter should be supported"
    except TypeError:
        pass  # Expected to fail on current code
    
    # Test 2: labels_ attribute should be available after fitting
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)
    
    # This should exist but doesn't in current implementation
    assert hasattr(gmm, 'labels_'), "Mixture model should store labels_ after fitting"
    assert gmm.labels_.shape == (X.shape[0],), "labels_ should have shape (n_samples,)"
    
    # Test 3: fit_predict method should exist
    gmm = GaussianMixture(n_components=2, random_state=42)
    assert hasattr(gmm, 'fit_predict'), "Mixture model should have fit_predict method"
    
    # fit_predict should return labels directly
    labels = gmm.fit_predict(X)
    assert labels.shape == (X.shape[0],), "fit_predict should return labels with shape (n_samples,)"