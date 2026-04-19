import numpy as np
from sklearn.mixture import GaussianMixture

def test_issue_reproduction():
    """Test that fit_predict and predict agree when n_init > 1."""
    # Create some simple 2D data with clear clusters
    np.random.seed(42)
    X = np.vstack([
        np.random.normal([0, 0], 0.5, (50, 2)),
        np.random.normal([3, 3], 0.5, (50, 2))
    ])
    
    # Create GaussianMixture with n_init > 1
    gm = GaussianMixture(n_components=2, n_init=5, random_state=42)
    
    # Get labels from fit_predict
    labels_fit_predict = gm.fit_predict(X)
    
    # Get labels from predict (after fitting)
    labels_predict = gm.predict(X)
    
    # These should be identical but they're not due to the bug
    assert np.array_equal(labels_fit_predict, labels_predict), \
        "fit_predict and predict should return identical labels when n_init > 1"