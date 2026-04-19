import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def test_issue_reproduction():
    """Test that VotingClassifier fails when weights are passed and an estimator is None."""
    X, y = load_iris(return_X_y=True)
    
    # Create a VotingClassifier with one None estimator
    voter = VotingClassifier(
        estimators=[('lr', LogisticRegression()), ('dt', None)],
        voting='hard'
    )
    
    # Create sample weights
    sample_weight = np.ones(len(y))
    
    # This should fail with AttributeError: 'NoneType' object has no attribute 'fit'
    # because the code doesn't check for None estimators when validating sample_weight support
    voter.fit(X, y, sample_weight=sample_weight)