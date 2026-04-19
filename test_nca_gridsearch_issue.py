import numpy as np
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.datasets import make_classification

def test_issue_reproduction():
    # Create simple dataset
    X, y = make_classification(n_samples=20, n_features=4, n_classes=2, random_state=42)
    
    # Test case 1: tol as int (should be accepted but currently fails)
    nca1 = NeighborhoodComponentsAnalysis(tol=1, random_state=42)
    nca1.fit(X, y)  # This should work but fails due to strict type checking
    
    # Test case 2: n_components as np.int64 (should be accepted but currently fails) 
    nca2 = NeighborhoodComponentsAnalysis(n_components=np.int64(2), random_state=42)
    nca2.fit(X, y)  # This should work but fails due to strict type checking