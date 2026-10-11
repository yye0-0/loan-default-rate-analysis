"""Expose the sklearn estimator contract without altering CatBoost training.

CatBoost 1.2.7 lacks the estimator tags required by sklearn 1.8 Pipeline.
The wrapped native estimator still performs every fit and prediction.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted


class SklearnCatBoostClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None, **fit_params):
        self.model_ = clone(self.estimator)
        self.model_.fit(X, y, sample_weight=sample_weight, **fit_params)
        self.classes_ = np.asarray(self.model_.classes_)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'model_')
        return self.model_.predict_proba(X)

    def predict(self, X):
        check_is_fitted(self, 'model_')
        return np.asarray(self.model_.predict(X)).reshape(-1)
