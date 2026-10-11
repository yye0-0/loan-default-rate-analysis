"""Regression: Pipeline must accept a native estimator without sklearn tags."""
import io
import sys
import unittest
from pathlib import Path
import joblib
import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from catboost_compat import SklearnCatBoostClassifier


class NativeWithoutTags:
    def get_params(self, deep=True):
        return {}

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array([0, 1])
        self.probability = np.average(y, weights=sample_weight)
        return self

    def predict_proba(self, X):
        return np.tile([1-self.probability, self.probability], (len(X), 1))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class CatBoostInterfaceTests(unittest.TestCase):
    def test_weighted_native_pipeline_prediction_and_save_reload(self):
        model = Pipeline([('scale', StandardScaler()),
                          ('model', SklearnCatBoostClassifier(NativeWithoutTags()))])
        X = np.array([[1., 2.], [2., 3.], [3., 4.]])
        self.assertTrue(is_classifier(model))
        clone(model).fit(X, [0, 0, 1])
        model.fit(X, [0, 0, 1], model__sample_weight=[1, 1, 8])
        np.testing.assert_allclose(model.predict_proba(X)[:, 1], .8)
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)
        np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))

    def test_prediction_before_training_rejected(self):
        with self.assertRaises(NotFittedError):
            SklearnCatBoostClassifier(NativeWithoutTags()).predict_proba(np.ones((2, 1)))
