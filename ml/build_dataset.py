"""
Budowa zbioru danych do treningu modeli ML z cech EEG (SEED-VII).

Wczytuje cechy `de_LDS_1..80` z plików `EEG_features/1.mat .. 20.mat`
(kształt każdej próby: (okna, 5 pasm, 62 kanały), zmienna liczba okien),
uśrednia po oknach czasowych do wektora 310 cech (5×62) na próbę
i zapisuje macierz X, etykiety y, identyfikatory osób i numery prób do
`data/dataset.npz`.

Obsługuje dwa źródła cech:
  - features (domyślne): gotowe cechy `de_LDS_n` z `EEG_features/*.mat`,
  - preprocessed: cechy DE liczone własnym ekstraktorem (`ml/feature_extraction.py`,
    z wygładzaniem LDS) z sygnału `EEG_preprocessed/*.mat` — dzięki temu trening
    i aplikacja używają identycznego pipeline'u ekstrakcji (brak domain shift).

Uruchomienie:
    python -m ml.build_dataset                        # -> data/dataset.npz
    python -m ml.build_dataset --source preprocessed  # -> data/dataset_own.npz
"""

import argparse

import numpy as np
import scipy.io

from ml.config import (
    DATA_DIR,
    DATASET_PATH,
    FEATURES_DIR,
    N_BANDS,
    N_CHANNELS,
    N_SUBJECTS,
    N_TRIALS,
    OWN_DATASET_PATH,
    PREPROCESSED_DIR,
    trial_to_label,
)


def aggregate_trial_features(trial_data: np.ndarray) -> np.ndarray:
    """
    Agreguje cechy jednej próby do wektora o stałym wymiarze 310.

    Parametry:
        trial_data: tablica (okna, 5, 62) lub (5, 62) — cechy DE.

    Zwraca:
        wektor (310,) — średnia po oknach czasowych, spłaszczona (pasma × kanały).
    """
    trial_data = np.asarray(trial_data, dtype=np.float64)
    if trial_data.ndim == 2:
        trial_data = trial_data[np.newaxis, ...]
    if trial_data.ndim != 3 or trial_data.shape[1:] != (N_BANDS, N_CHANNELS):
        raise ValueError(
            f"Oczekiwano kształtu (okna, {N_BANDS}, {N_CHANNELS}), otrzymano: {trial_data.shape}"
        )
    return trial_data.mean(axis=0).flatten()


def load_subject(subject_id: int, feature_prefix: str = "de_LDS"):
    """
    Wczytuje wszystkie 80 prób jednej osoby i zwraca (X, y) dla tej osoby.

    Parametry:
        subject_id: numer osoby (1-20), odpowiada plikowi EEG_features/{id}.mat.
        feature_prefix: prefiks klucza cech w pliku .mat (domyślnie 'de_LDS').
    """
    mat_path = FEATURES_DIR / f"{subject_id}.mat"
    if not mat_path.exists():
        # Tolerancja na drobne różnice w nazwie pliku (np. '1f.mat' zamiast '1.mat')
        candidates = sorted(FEATURES_DIR.glob(f"{subject_id}[a-z].mat"))
        if not candidates:
            raise FileNotFoundError(f"Brak pliku cech dla osoby {subject_id} w {FEATURES_DIR}")
        mat_path = candidates[0]
    data = scipy.io.loadmat(str(mat_path))

    X, y = [], []
    for trial in range(1, N_TRIALS + 1):
        key = f"{feature_prefix}_{trial}"
        if key not in data:
            raise KeyError(f"Brak klucza '{key}' w pliku {mat_path}")
        X.append(aggregate_trial_features(data[key]))
        y.append(trial_to_label(trial))
    trials = np.arange(1, N_TRIALS + 1, dtype=np.int16)
    return np.array(X), np.array(y), trials


def load_subject_from_signal(subject_id: int):
    """
    Wczytuje 80 prób sygnału jednej osoby z EEG_preprocessed i liczy cechy DE
    własnym ekstraktorem (z wygładzaniem LDS). Zwraca (X, y) dla tej osoby.
    """
    # Import lokalny — unika zależności scipy.signal przy źródle 'features'
    from ml.feature_extraction import extract_de_features

    mat_path = PREPROCESSED_DIR / f"{subject_id}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Brak pliku sygnału dla osoby {subject_id} w {PREPROCESSED_DIR}")
    data = scipy.io.loadmat(str(mat_path))

    X, y = [], []
    for trial in range(1, N_TRIALS + 1):
        key = str(trial)
        if key not in data:
            raise KeyError(f"Brak klucza '{key}' w pliku {mat_path}")
        features = extract_de_features(data[key], smooth=True)
        X.append(aggregate_trial_features(features))
        y.append(trial_to_label(trial))
    trials = np.arange(1, N_TRIALS + 1, dtype=np.int16)
    return np.array(X), np.array(y), trials


def build_dataset(source: str = "features") -> dict:
    """
    Buduje pełny zbiór danych (wszystkie osoby) i zapisuje go do pliku .npz.

    Parametry:
        source: 'features' (gotowe cechy de_LDS) lub 'preprocessed'
                (cechy z własnego ekstraktora DE + LDS).
    """
    if source == "features":
        loader, out_path = load_subject, DATASET_PATH
    elif source == "preprocessed":
        loader, out_path = load_subject_from_signal, OWN_DATASET_PATH
    else:
        raise ValueError(f"Nieznane źródło danych: {source}")

    all_X, all_y, all_subjects, all_trials = [], [], [], []
    for subject_id in range(1, N_SUBJECTS + 1):
        X_s, y_s, trials_s = loader(subject_id)
        all_X.append(X_s)
        all_y.append(y_s)
        all_subjects.append(np.full(len(y_s), subject_id))
        all_trials.append(trials_s)
        print(f"Osoba {subject_id:2d}: {X_s.shape[0]} prób, {X_s.shape[1]} cech", flush=True)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    subjects = np.concatenate(all_subjects)
    trials = np.concatenate(all_trials)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, subjects=subjects, trials=trials)

    print(f"\nZapisano: {out_path}")
    print(f"X: {X.shape}, y: {y.shape}, subjects: {subjects.shape}, trials: {trials.shape}")
    print(f"Klasy: {np.unique(y)}, osoby: {len(np.unique(subjects))}")
    print(f"Rozkład klas: {np.bincount(y)}")
    return {"X": X, "y": y, "subjects": subjects, "trials": trials}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Budowa zbioru danych SEED-VII")
    parser.add_argument("--source", choices=["features", "preprocessed"],
                        default="features")
    args = parser.parse_args()
    build_dataset(source=args.source)
