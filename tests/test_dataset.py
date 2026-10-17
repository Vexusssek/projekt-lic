"""Testy pipeline'u danych: mapowanie etykiet i agregacja okien czasowych."""

import numpy as np
import pytest

from ml.build_dataset import aggregate_trial_features
from ml.config import (
    EMOTION_MAP,
    N_TRIALS,
    TRIAL_EMOTIONS,
    trial_to_emotion,
    trial_to_label,
)


class TestLabelMapping:
    def test_liczba_prob(self):
        assert len(TRIAL_EMOTIONS) == N_TRIALS == 80

    def test_znane_etykiety_z_xlsx(self):
        # Wartości zweryfikowane ręcznie z emotion_label_and_stimuli_order.xlsx
        assert trial_to_emotion(1) == "Happy"
        assert trial_to_emotion(20) == "Happy"
        assert trial_to_emotion(21) == "Anger"
        assert trial_to_emotion(23) == "Fear"
        assert trial_to_emotion(42) == "Surprise"
        assert trial_to_emotion(61) == "Disgust"
        assert trial_to_emotion(80) == "Disgust"

    def test_rozklad_klas(self):
        # 6 emocji po 12 prób + Neutral 8 prób (tylko sesje 1-2)
        from collections import Counter

        counts = Counter(TRIAL_EMOTIONS)
        assert counts["Neutral"] == 8
        for emotion in ["Happy", "Sad", "Fear", "Disgust", "Surprise", "Anger"]:
            assert counts[emotion] == 12

    def test_etykiety_liczbowe(self):
        assert trial_to_label(1) == EMOTION_MAP["Happy"]
        assert trial_to_label(2) == EMOTION_MAP["Neutral"]

    def test_zly_numer_proby(self):
        with pytest.raises(ValueError):
            trial_to_emotion(0)
        with pytest.raises(ValueError):
            trial_to_emotion(81)


class TestAggregation:
    def test_srednia_po_oknach(self):
        trial = np.ones((10, 5, 62))
        trial[5:] = 3.0  # średnia = 2.0
        vec = aggregate_trial_features(trial)
        assert vec.shape == (310,)
        assert np.allclose(vec, 2.0)

    def test_jedno_okno(self):
        trial = np.random.default_rng(42).random((1, 5, 62))
        vec = aggregate_trial_features(trial)
        assert vec.shape == (310,)
        assert np.allclose(vec, trial[0].flatten())

    def test_ksztalt_2d(self):
        trial = np.random.default_rng(42).random((5, 62))
        vec = aggregate_trial_features(trial)
        assert vec.shape == (310,)
        assert np.allclose(vec, trial.flatten())

    def test_zly_ksztalt(self):
        with pytest.raises(ValueError):
            aggregate_trial_features(np.random.default_rng(42).random((10, 4, 62)))
        with pytest.raises(ValueError):
            aggregate_trial_features(np.random.default_rng(42).random((10, 5, 61)))


class TestModelSelection:
    def test_loso_criterion_selection(self, tmp_path):
        import joblib
        import pandas as pd
        from sklearn.dummy import DummyClassifier
        from ml.train import save_best_model

        models = {
            "Model_A": DummyClassifier(strategy="most_frequent"),
            "Model_B": DummyClassifier(strategy="prior"),
        }
        metrics_df = pd.DataFrame([
            {"model": "Model_A", "scheme": "subject_dependent", "accuracy": 0.50, "f1_macro": 0.48},
            {"model": "Model_B", "scheme": "subject_dependent", "accuracy": 0.35, "f1_macro": 0.33},
            {"model": "Model_A", "scheme": "loso", "accuracy": 0.20, "f1_macro": 0.18},
            {"model": "Model_B", "scheme": "loso", "accuracy": 0.25, "f1_macro": 0.24},
        ])

        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, 310))
        y = np.tile([0, 1], 10)
        save_path = tmp_path / "best_model.joblib"

        # 1. Selekcja z kryterium domyślnym "loso" -> musi wybrać Model_B
        save_best_model(models, metrics_df, X, y, model_path=save_path, criterion="loso")
        art = joblib.load(save_path)
        assert art["model_name"] == "Model_B"
        assert art["selection_scheme"] == "loso"
        assert art["selection_accuracy"] == 0.25

        # 2. Selekcja z kryterium "subject_dependent" -> musi wybrać Model_A
        save_best_model(models, metrics_df, X, y, model_path=save_path, criterion="subject_dependent")
        art_sd = joblib.load(save_path)
        assert art_sd["model_name"] == "Model_A"
        assert art_sd["selection_scheme"] == "subject_dependent"
        assert art_sd["selection_accuracy"] == 0.50

    def test_brak_wymaganego_schematu_blokuje_zapis(self, tmp_path):
        import pandas as pd
        from sklearn.dummy import DummyClassifier
        from ml.train import save_best_model

        metrics = pd.DataFrame([
            {"model": "Model", "scheme": "subject_dependent", "accuracy": 0.5, "f1_macro": 0.5}
        ])
        X = np.random.default_rng(42).normal(size=(20, 310))
        y = np.tile([0, 1], 10)
        with pytest.raises(ValueError, match="nie zawiera schematu 'loso'"):
            save_best_model(
                {"Model": DummyClassifier()},
                metrics,
                X,
                y,
                model_path=tmp_path / "model.joblib",
                criterion="loso",
            )


def test_stimulus_independent_nie_miesza_bodzcow():
    from ml.train import validation_split

    trials = np.repeat(np.arange(35), 3)
    y = np.repeat(np.tile(np.arange(7), 5), 3)
    subjects = np.tile(np.arange(3), 35)
    X = np.zeros((len(y), 1))
    cv, groups, _ = validation_split("stimulus_independent", subjects, trials)
    for train_index, test_index in cv.split(X, y, groups):
        assert set(trials[train_index]).isdisjoint(trials[test_index])


def test_kanaly_wykresu_maja_oczekiwane_nazwy():
    from ml.config import PLOT_CHANNELS, SEED_VII_CHANNELS

    assert [SEED_VII_CHANNELS[index] for index, _ in PLOT_CHANNELS] == [
        "FP1", "FC3", "CZ", "P3", "CB2"
    ]
