"""Testy ekstrakcji cech DE z sygnału EEG."""

import numpy as np
import pytest
import scipy.io

from ml.config import FEATURES_DIR, N_CHANNELS, PREPROCESSED_DIR, SAMPLING_RATE
from ml.feature_extraction import (
    WINDOW_SECONDS,
    differential_entropy,
    extract_de_features,
    lds_smooth,
)


class TestDifferentialEntropy:
    def test_znana_wariancja(self):
        # DE gaussowskiego szumu
        rng = np.random.default_rng(42)
        sigma = 2.0
        signal = rng.normal(0, sigma, size=(1, 200_000))
        expected = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
        de = differential_entropy(signal)
        assert de.shape == (1,)
        assert abs(de[0] - expected) < 0.01

    def test_sygnal_stalym(self):
        # Zerowa wariancja nie powoduje log(0)
        de = differential_entropy(np.ones((3, 100)))
        assert np.all(np.isfinite(de))


class TestExtractDeFeatures:
    def test_ksztalt_wyjscia(self):
        rng = np.random.default_rng(0)
        n_sec = 10
        signal = rng.normal(size=(N_CHANNELS, n_sec * SAMPLING_RATE))
        feats = extract_de_features(signal)
        assert feats.shape == (n_sec // WINDOW_SECONDS, 5, N_CHANNELS)
        assert np.all(np.isfinite(feats))

    def test_zla_liczba_kanalow(self):
        with pytest.raises(ValueError):
            extract_de_features(np.zeros((61, 2000)))

    def test_za_krotki_sygnal(self):
        with pytest.raises(ValueError):
            extract_de_features(np.zeros((N_CHANNELS, SAMPLING_RATE // 2)))


class TestLdsSmooth:
    def test_redukcja_wariancji_zachowana_srednia(self):
        # Stała + szum: wygładzanie zmniejsza wariancję, średnia niemal bez zmian
        rng = np.random.default_rng(42)
        seq = 3.0 + rng.normal(0, 1.0, size=(50, 5, 62))
        smoothed = lds_smooth(seq)
        assert smoothed.shape == seq.shape
        assert smoothed.var() < 0.5 * seq.var()
        assert abs(smoothed.mean() - seq.mean()) < 0.05

    def test_jedno_okno_bez_zmian(self):
        # Przypadek brzegowy (1, 5, 62): sekwencji nie da się wygładzić
        seq = np.random.default_rng(0).normal(size=(1, 5, 62))
        smoothed = lds_smooth(seq)
        np.testing.assert_array_equal(smoothed, seq)

    def test_smooth_w_extract(self):
        # extract_de_features(smooth=True) daje gładsze cechy niż bez wygładzania
        rng = np.random.default_rng(1)
        signal = rng.normal(size=(N_CHANNELS, 40 * SAMPLING_RATE))
        raw = extract_de_features(signal, smooth=False)
        smoothed = extract_de_features(signal, smooth=True)
        assert raw.shape == smoothed.shape
        # zmienność między oknami (różnice sąsiednich okien) maleje
        assert np.abs(np.diff(smoothed, axis=0)).mean() < np.abs(np.diff(raw, axis=0)).mean()


@pytest.mark.skipif(
    not (PREPROCESSED_DIR / "2.mat").exists() or not (FEATURES_DIR / "2.mat").exists(),
    reason="Wymaga danych SEED-VII (EEG_preprocessed i EEG_features)",
)
def test_zgodnosc_z_cechami_seed():
    """
    Test zgodności: DE liczone z sygnału preprocessed koreluje z cechami `de_n`
    z EEG_features dla tej samej próby (osoba 2, trial 1).

    Uwaga: cechy referencyjne SEED liczone są przez STFT, a nasz moduł przez
    filtrację Butterwortha — stąd próg korelacji 0.7 (dla osoby 2 uzyskujemy ~0.81).
    """
    signal = scipy.io.loadmat(str(PREPROCESSED_DIR / "2.mat"))["1"]
    ours = extract_de_features(signal)

    reference = scipy.io.loadmat(str(FEATURES_DIR / "2.mat"))["de_1"]

    n = min(ours.shape[0], reference.shape[0])
    corr = np.corrcoef(ours[:n].flatten(), reference[:n].flatten())[0, 1]
    assert corr > 0.7, f"Za niska korelacja z cechami SEED: {corr:.3f}"
