"""Test kontraktu predykcji używanego przez aplikację."""

import numpy as np

from app.model_service import predict


class FixedModel:
    classes_ = np.arange(7)

    def predict_proba(self, X):
        assert X.shape == (1, 310)
        return np.array([[0.05, 0.10, 0.15, 0.20, 0.25, 0.15, 0.10]])


def test_nowy_wektor_jest_klasyfikowany_deterministycznie():
    artifact = {"model": FixedModel()}
    vector = np.linspace(-2.0, 2.0, 310)

    first = predict(artifact, vector)
    second = predict(artifact, vector)

    np.testing.assert_array_equal(first.classes, np.arange(7))
    np.testing.assert_allclose(first.probabilities, second.probabilities)
