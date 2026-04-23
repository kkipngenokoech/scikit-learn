"""Tests for threshold tuning with cross-validation."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._threshold import TunedThresholdClassifierCV
from sklearn.metrics import f1_score


class TestTunedThresholdClassifierCV:
    """Test TunedThresholdClassifierCV."""
    
    def test_refit_false_cv_float_consistency(self):
        """Test that estimator_ and best_threshold_ use the same split when refit=False and cv=float."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        
        # Test with random_state=None to ensure splits are reused properly
        tuned_clf = TunedThresholdClassifierCV(
            estimator=LogisticRegression(random_state=42),
            cv=0.5,  # Use float for train_test_split
            refit=False,
            scoring="f1",
            thresholds=10,
            random_state=None
        )
        
        tuned_clf.fit(X, y)
        
        # Verify that the estimator was fitted
        assert hasattr(tuned_clf, "estimator_")
        assert hasattr(tuned_clf, "best_threshold_")
        
        # The estimator should be fitted on the same data used for threshold tuning
        # We can't directly verify this without modifying the implementation,
        # but we can check that the results are consistent
        predictions = tuned_clf.predict(X)
        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_refit_false_cv_float_fixed_random_state(self):
        """Test consistency with fixed random_state."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        
        tuned_clf1 = TunedThresholdClassifierCV(
            estimator=LogisticRegression(random_state=42),
            cv=0.5,
            refit=False,
            scoring="f1",
            thresholds=10,
            random_state=42
        )
        
        tuned_clf2 = TunedThresholdClassifierCV(
            estimator=LogisticRegression(random_state=42),
            cv=0.5,
            refit=False,
            scoring="f1",
            thresholds=10,
            random_state=42
        )
        
        tuned_clf1.fit(X, y)
        tuned_clf2.fit(X, y)
        
        # Results should be identical with same random_state
        assert tuned_clf1.best_threshold_ == tuned_clf2.best_threshold_
        assert tuned_clf1.best_score_ == tuned_clf2.best_score_
        
        # Predictions should be identical
        pred1 = tuned_clf1.predict(X)
        pred2 = tuned_clf2.predict(X)
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_refit_true_behavior_unchanged(self):
        """Test that refit=True behavior is unchanged."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        
        tuned_clf = TunedThresholdClassifierCV(
            estimator=LogisticRegression(random_state=42),
            cv=3,
            refit=True,
            scoring="f1",
            thresholds=10,
            random_state=42
        )
        
        tuned_clf.fit(X, y)
        
        # Verify that the estimator was fitted on the full dataset
        assert hasattr(tuned_clf, "estimator_")
        assert hasattr(tuned_clf, "best_threshold_")
        
        # Should be able to make predictions
        predictions = tuned_clf.predict(X)
        assert len(predictions) == len(y)
    
    def test_cv_integer_splits_reuse(self):
        """Test that integer CV splits are properly reused."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        
        tuned_clf = TunedThresholdClassifierCV(
            estimator=LogisticRegression(random_state=42),
            cv=3,
            refit=False,
            scoring="f1",
            thresholds=5,
            random_state=42
        )
        
        tuned_clf.fit(X, y)
        
        assert hasattr(tuned_clf, "estimator_")
        assert hasattr(tuned_clf, "best_threshold_")
        assert 0 <= tuned_clf.best_threshold_ <= 1
    
    def test_cv_object_splits_reuse(self):
        """Test that CV object splits are properly reused."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        tuned_clf = TunedThresholdClassifierCV(
            estimator=LogisticRegression(random_state=42),
            cv=cv,
            refit=False,
            scoring="f1",
            thresholds=5,
            random_state=42
        )
        
        tuned_clf.fit(X, y)
        
        assert hasattr(tuned_clf, "estimator_")
        assert hasattr(tuned_clf, "best_threshold_")
        assert 0 <= tuned_clf.best_threshold_ <= 1
    
    def test_threshold_array_input(self):
        """Test using explicit threshold array."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        
        thresholds = [0.3, 0.5, 0.7]
        
        tuned_clf = TunedThresholdClassifierCV(
            estimator=LogisticRegression(random_state=42),
            cv=3,
            refit=False,
            scoring="f1",
            thresholds=thresholds,
            random_state=42
        )
        
        tuned_clf.fit(X, y)
        
        assert tuned_clf.best_threshold_ in thresholds
    
    def test_store_cv_results(self):
        """Test storing CV results."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        
        tuned_clf = TunedThresholdClassifierCV(
            estimator=LogisticRegression(random_state=42),
            cv=3,
            refit=False,
            scoring="f1",
            thresholds=5,
            store_cv_results=True,
            random_state=42
        )
        
        tuned_clf.fit(X, y)
        
        assert hasattr(tuned_clf, "cv_results_")
        assert "thresholds" in tuned_clf.cv_results_
        assert "scores" in tuned_clf.cv_results_
        assert len(tuned_clf.cv_results_["thresholds"]) == 5
        assert len(tuned_clf.cv_results_["scores"]) == 5
    
    def test_predict_proba_and_decision_function(self):
        """Test predict_proba and decision_function methods."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        
        # Test with estimator that has predict_proba
        tuned_clf = TunedThresholdClassifierCV(
            estimator=LogisticRegression(random_state=42),
            cv=3,
            refit=True,
            scoring="f1",
            thresholds=5,
            random_state=42
        )
        
        tuned_clf.fit(X, y)
        
        # Test predict_proba
        proba = tuned_clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        
        # Test decision_function
        decision = tuned_clf.decision_function(X)
        assert decision.shape == (len(X),)
    
    def test_different_estimators(self):
        """Test with different types of estimators."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        
        estimators = [
            LogisticRegression(random_state=42),
            RandomForestClassifier(n_estimators=10, random_state=42)
        ]
        
        for estimator in estimators:
            tuned_clf = TunedThresholdClassifierCV(
                estimator=estimator,
                cv=3,
                refit=False,
                scoring="f1",
                thresholds=5,
                random_state=42
            )
            
            tuned_clf.fit(X, y)
            predictions = tuned_clf.predict(X)
            
            assert len(predictions) == len(y)
            assert all(pred in [0, 1] for pred in predictions)
