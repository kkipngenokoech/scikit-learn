import numpy as np
from sklearn.cluster import AffinityPropagation

def test_issue_reproduction():
    # Create a simple dataset that will not converge with very few iterations
    # Use a small max_iter to force non-convergence
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    
    # Use very low max_iter to ensure non-convergence
    ap = AffinityPropagation(max_iter=1, verbose=False)
    ap.fit(X)
    
    # According to documentation, when algorithm does not converge:
    # - cluster_centers_indices_ should be empty array
    # - labels_ should be all -1
    
    # Check if it actually didn't converge by seeing if n_iter_ equals max_iter
    # This is a strong indicator of non-convergence
    if ap.n_iter_ == ap.max_iter:
        # Algorithm hit max_iter without converging
        assert len(ap.cluster_centers_indices_) == 0, f"Expected empty cluster centers, got {ap.cluster_centers_indices_}"
        assert all(label == -1 for label in ap.labels_), f"Expected all labels to be -1, got {ap.labels_}"
    else:
        # If it somehow converged in 1 iteration, that's unexpected but not the bug we're testing
        pass