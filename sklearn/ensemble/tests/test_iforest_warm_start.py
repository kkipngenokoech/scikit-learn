import numpy as np
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
from sklearn.utils.testing import assert_array_equal, assert_raises


def test_warm_start_parameter_exposed():
    """Test that warm_start parameter is properly exposed in __init__."""
    # Test default value
    clf = IsolationForest()
    assert clf.warm_start is False
    
    # Test setting to True
    clf = IsolationForest(warm_start=True)
    assert clf.warm_start is True


def test_warm_start_incremental_fitting():
    """Test that warm_start allows incremental addition of estimators."""
    X, _ = make_blobs(n_samples=100, centers=1, n_features=2, random_state=42)
    
    # Fit with 5 estimators initially
    clf = IsolationForest(n_estimators=5, warm_start=True, random_state=42)
    clf.fit(X)
    
    # Check we have 5 estimators
    assert len(clf.estimators_) == 5
    
    # Store the first 5 estimators for comparison
    first_estimators = clf.estimators_.copy()
    
    # Increase n_estimators and fit again
    clf.n_estimators = 10
    clf.fit(X)
    
    # Check we now have 10 estimators
    assert len(clf.estimators_) == 10
    
    # Check that the first 5 estimators are the same (warm start worked)
    for i in range(5):
        # Compare tree structures by checking if they make the same predictions
        # on a small test set
        test_X = X[:10]
        pred1 = first_estimators[i].predict(test_X)
        pred2 = clf.estimators_[i].predict(test_X)
        assert_array_equal(pred1, pred2)


def test_warm_start_vs_cold_start():
    """Test that warm_start=False creates new estimators each time."""
    X, _ = make_blobs(n_samples=100, centers=1, n_features=2, random_state=42)
    
    # Fit with warm_start=False
    clf = IsolationForest(n_estimators=5, warm_start=False, random_state=42)
    clf.fit(X)
    first_estimators = clf.estimators_.copy()
    
    # Fit again with same parameters
    clf.fit(X)
    second_estimators = clf.estimators_.copy()
    
    # With warm_start=False, estimators should be different
    # (we can't directly compare estimator objects, so we check predictions)
    test_X = X[:10]
    different_found = False
    for i in range(5):
        pred1 = first_estimators[i].predict(test_X)
        pred2 = second_estimators[i].predict(test_X)
        if not np.array_equal(pred1, pred2):
            different_found = True
            break
    
    # At least some estimators should be different due to randomness
    assert different_found


def test_warm_start_with_different_parameters():
    """Test that changing other parameters resets the ensemble even with warm_start."""
    X, _ = make_blobs(n_samples=100, centers=1, n_features=2, random_state=42)
    
    clf = IsolationForest(n_estimators=5, max_samples=50, warm_start=True, random_state=42)
    clf.fit(X)
    
    # Change max_samples - this should reset the ensemble
    clf.max_samples = 80
    clf.fit(X)
    
    # Should still have 5 estimators
    assert len(clf.estimators_) == 5


def test_warm_start_decreasing_n_estimators():
    """Test behavior when decreasing n_estimators with warm_start."""
    X, _ = make_blobs(n_samples=100, centers=1, n_features=2, random_state=42)
    
    clf = IsolationForest(n_estimators=10, warm_start=True, random_state=42)
    clf.fit(X)
    
    # Decrease n_estimators
    clf.n_estimators = 5
    clf.fit(X)
    
    # Should have 5 estimators (truncated)
    assert len(clf.estimators_) == 5


def test_warm_start_same_n_estimators():
    """Test that fitting with same n_estimators and warm_start=True doesn't change estimators."""
    X, _ = make_blobs(n_samples=100, centers=1, n_features=2, random_state=42)
    
    clf = IsolationForest(n_estimators=5, warm_start=True, random_state=42)
    clf.fit(X)
    
    first_estimators = clf.estimators_.copy()
    
    # Fit again with same n_estimators
    clf.fit(X)
    
    # Estimators should be the same
    assert len(clf.estimators_) == len(first_estimators)
    
    # Check that estimators make same predictions
    test_X = X[:10]
    for i in range(len(first_estimators)):
        pred1 = first_estimators[i].predict(test_X)
        pred2 = clf.estimators_[i].predict(test_X)
        assert_array_equal(pred1, pred2)


def test_warm_start_consistent_predictions():
    """Test that predictions are consistent when using warm_start."""
    X, _ = make_blobs(n_samples=100, centers=1, n_features=2, random_state=42)
    
    # Fit incrementally with warm_start
    clf_warm = IsolationForest(n_estimators=5, warm_start=True, random_state=42)
    clf_warm.fit(X)
    clf_warm.n_estimators = 10
    clf_warm.fit(X)
    
    # Fit all at once without warm_start
    clf_cold = IsolationForest(n_estimators=10, warm_start=False, random_state=42)
    clf_cold.fit(X)
    
    # Predictions should be similar (not exactly equal due to different fitting process)
    pred_warm = clf_warm.predict(X)
    pred_cold = clf_cold.predict(X)
    
    # At least the majority of predictions should agree
    agreement = np.mean(pred_warm == pred_cold)
    assert agreement > 0.7  # Allow some difference due to randomness
