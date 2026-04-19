import numpy as np
from sklearn.cluster import AffinityPropagation
import warnings

def test_issue_reproduction():
    # Create a simple dataset that will not converge with very low max_iter
    # Use a small similarity matrix that would normally find clusters but
    # won't converge in just 1 iteration
    X = np.array([[0, 0], [1, 1], [2, 2]])
    
    # Use very low max_iter to force non-convergence
    ap = AffinityPropagation(max_iter=1, verbose=False)
    
    # Suppress convergence warnings for cleaner test output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ap.fit(X)
    
    # According to documentation, when algorithm doesn't converge:
    # - cluster_centers_indices_ should be empty array
    # - labels_ should be all -1
    
    # This assertion should pass but currently fails because
    # the algorithm returns actual clusters even when not converged
    assert len(ap.cluster_centers_indices_) == 0, f"Expected empty cluster_centers_indices_, got {ap.cluster_centers_indices_}"
    assert np.all(ap.labels_ == -1), f"Expected all labels to be -1, got {ap.labels_}"