"""Budowanie i weryfikacja metadanych artefaktu modelu."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version

import numpy as np


ARTIFACT_VERSION = 2
TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "joblib",
    "matplotlib",
    "streamlit",
)


def dataset_fingerprint(*arrays: np.ndarray) -> str:
    """Zwraca SHA-256 zależny od wartości, kształtów i typów tablic."""
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def library_versions() -> dict[str, str]:
    """Zwraca wersje bibliotek potrzebnych do odtworzenia treningu."""
    result = {}
    for package in TRACKED_PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result
