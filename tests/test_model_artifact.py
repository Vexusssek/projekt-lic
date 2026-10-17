"""Testy metadanych wersjonowanego artefaktu."""

import numpy as np

from ml.model_artifact import (
    dataset_fingerprint,
)


def test_fingerprint_jest_deterministyczny_i_zalezny_od_danych():
    first = np.arange(12).reshape(3, 4)
    second = first.copy()
    assert dataset_fingerprint(first) == dataset_fingerprint(second)
    second[0, 0] = 99
    assert dataset_fingerprint(first) != dataset_fingerprint(second)
