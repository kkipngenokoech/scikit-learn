"""Threshold tuning with cross-validation."""

import numpy as np
from numbers import Integral, Real

from ..base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from ..metrics import check_scoring
from ..model_selection import check_cv
from ..utils import check_X_y
from ..utils._param_validation import HasMethods, Interval
from ..utils.validation import check_is_fitted


class TunedThresholdClassifierCV(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Tune the decision threshold using cross-validation.
    
    This meta-estimator optimizes the decision threshold of a binary classifier
    using cross-validation.
    
    Parameters
    ----------
    estimator : estimator instance
        The base estimator to fit on each fold.
        
    scoring : str, callable, default=None
        A single str or a callable to evaluate the predictions on the test set.
        
    response_method : {'auto', 'predict_proba', 'decision_function'}, default='auto'
        Methods by the classifier to get the response values.
        
    thresholds : int, array-like, default=100
        The number of thresholds to try or the explicit thresholds to use.
        
    cv : int, cross-validation generator, iterable, or float, default=None
        Determines the cross-validation splitting strategy.
        
    refit : bool, default=True
        Whether to refit the estimator on the entire dataset once the best
        threshold has been found.
        
    n_jobs : int, default=None
        Number of jobs to run in parallel.
        
    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
        
    store_cv_results : bool, default=False
        Whether to store the CV results.
        
    Attributes
    ----------
    estimator_ : estimator
        The fitted estimator used to make predictions.
        
    best_threshold_ : float
        The optimal threshold found during cross-validation.
        
    best_score_ : float
        The best score achieved during cross-validation.
        
    cv_results_ : dict
        Cross-validation results (if store_cv_results=True).
    """
    
    _parameter_constraints = {
        "estimator": [HasMethods(["fit"])],
        "scoring": [str, callable, None],
        "response_method": [str],
        "thresholds": [Integral, "array-like"],
        "cv": [Integral, Real, None, "cv_object"],
        "refit": ["boolean"],
        "n_jobs": [Integral, None],
        "random_state": ["random_state"],
        "store_cv_results": ["boolean"],
    }
    
    def __init__(
        self,
        estimator,
        *,
        scoring=None,
        response_method="auto",
        thresholds=100,
        cv=None,
        refit=True,
        n_jobs=None,
        random_state=None,
        store_cv_results=False,
    ):
        self.estimator = estimator
        self.scoring = scoring
        self.response_method = response_method
        self.thresholds = thresholds
        self.cv = cv
        self.refit = refit
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.store_cv_results = store_cv_results
    
    def _more_tags(self):
        return {
            "binary_only": True,
            "requires_positive_X": self.estimator._more_tags().get("requires_positive_X", False),
        }
    
    def fit(self, X, y, **params):
        """Fit the classifier and tune the decision threshold.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        y : array-like of shape (n_samples,)
            Target values.
            
        **params : dict
            Parameters passed to the underlying estimator's fit method.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self._fit(X, y, **params)
    
    def _fit(self, X, y, **params):
        """Fit the classifier and tune the decision threshold."""
        X, y = check_X_y(X, y)
        
        cv = check_cv(self.cv, y, classifier=True)
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        
        # Convert splits generator to list to allow reuse
        splits = list(cv.split(X, y))
        
        if not self.refit:
            # Use the first split for fitting the estimator when refit=False
            train_indices, _ = splits[0]
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X[train_indices], y[train_indices], **params)
        
        # Tune threshold using the same splits
        best_score = -np.inf
        best_threshold = 0.5
        cv_results = {"thresholds": [], "scores": []}
        
        # Generate thresholds
        if isinstance(self.thresholds, Integral):
            thresholds = np.linspace(0, 1, self.thresholds)
        else:
            thresholds = np.asarray(self.thresholds)
        
        for threshold in thresholds:
            scores = []
            for train_idx, test_idx in splits:
                # Fit estimator on training fold
                estimator_fold = clone(self.estimator)
                estimator_fold.fit(X[train_idx], y[train_idx], **params)
                
                # Get predictions on test fold
                if hasattr(estimator_fold, "predict_proba"):
                    y_score = estimator_fold.predict_proba(X[test_idx])[:, 1]
                elif hasattr(estimator_fold, "decision_function"):
                    y_score = estimator_fold.decision_function(X[test_idx])
                else:
                    raise ValueError("Estimator must have predict_proba or decision_function")
                
                # Apply threshold
                y_pred = (y_score >= threshold).astype(int)
                
                # Score the predictions
                score = scorer(estimator_fold, X[test_idx], y[test_idx])
                scores.append(score)
            
            mean_score = np.mean(scores)
            cv_results["thresholds"].append(threshold)
            cv_results["scores"].append(mean_score)
            
            if mean_score > best_score:
                best_score = mean_score
                best_threshold = threshold
        
        self.best_threshold_ = best_threshold
        self.best_score_ = best_score
        
        if self.store_cv_results:
            self.cv_results_ = cv_results
        
        if self.refit:
            # Refit on entire dataset
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y, **params)
        
        return self
    
    def predict(self, X):
        """Predict class labels using the tuned threshold.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        
        if hasattr(self.estimator_, "predict_proba"):
            y_score = self.estimator_.predict_proba(X)[:, 1]
        elif hasattr(self.estimator_, "decision_function"):
            y_score = self.estimator_.decision_function(X)
        else:
            raise ValueError("Estimator must have predict_proba or decision_function")
        
        return (y_score >= self.best_threshold_).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_proba : ndarray of shape (n_samples, 2)
            Predicted class probabilities.
        """
        check_is_fitted(self)
        return self.estimator_.predict_proba(X)
    
    def decision_function(self, X):
        """Compute the decision function.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_score : ndarray of shape (n_samples,)
            Decision function values.
        """
        check_is_fitted(self)
        return self.estimator_.decision_function(X)
