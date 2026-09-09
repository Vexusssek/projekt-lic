"""Walidacja artefaktów i wykonywanie predykcji poza warstwą interfejsu."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.config import EMOTION_MAP, N_BANDS, N_CHANNELS
from ml.model_artifact import ARTIFACT_VERSION


@dataclass(frozen=True)
class Prediction:
    probabilities: np.ndarray
    classes: np.ndarray


def validate_artifact(artifact: dict, path: Path, expected_feature_type: str) -> None:
    """Sprawdza kontrakt między treningiem a aplikacją."""
    if not isinstance(artifact, dict):
        raise ValueError(f"Plik {path.name} nie zawiera słownika artefaktu.")
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"Model {path.name} ma nieobsługiwaną wersję artefaktu. "
            "Uruchom trening ponownie z aktualnego kodu."
        )
    if artifact.get("feature_type") != expected_feature_type:
        raise ValueError(
            f"Model {path.name} ma typ cech {artifact.get('feature_type')!r}; "
            f"oczekiwano {expected_feature_type!r}."
        )
    if artifact.get("emotion_map") != EMOTION_MAP:
        raise ValueError(f"Model {path.name} ma niezgodne mapowanie klas emocji.")
    if artifact.get("selection_scheme") != "loso":
        raise ValueError(f"Model {path.name} nie został wybrany w walidacji LOSO.")

    model = artifact.get("model")
    expected_classes = np.arange(len(EMOTION_MAP))
    if model is None or not np.array_equal(getattr(model, "classes_", None), expected_classes):
        raise ValueError(f"Model {path.name} ma niezgodny zestaw klas.")
    expected_features = N_BANDS * N_CHANNELS
    if getattr(model, "n_features_in_", None) != expected_features:
        raise ValueError(
            f"Model {path.name} nie przyjmuje wymaganych {expected_features} cech."
        )


def predict(artifact: dict, vector: np.ndarray) -> Prediction:
    """Zwraca wynik klasyfikatora dla przygotowanego wektora cech."""
    model = artifact["model"]
    probabilities = np.asarray(
        model.predict_proba(np.asarray(vector).reshape(1, -1))[0], dtype=float
    )
    classes = np.asarray(model.classes_)
    if probabilities.shape != classes.shape or not np.isfinite(probabilities).all():
        raise ValueError("Model zwrócił nieprawidłowy rozkład wyników.")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
        raise ValueError("Wyniki klas nie sumują się do jedności.")
    return Prediction(
        probabilities=probabilities,
        classes=classes,
    )
