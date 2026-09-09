"""
Funkcje pomocnicze aplikacji Streamlit: wczytywanie i walidacja plików .mat,
weryfikacja zgodności danych z modelem SEED-VII, detekcja formatu (cechy DE vs sygnał EEG),
przygotowanie wektora cech, wycinanie fragmentów sygnału, eksport do .mat oraz rozbijanie dużych plików.
"""

import io
import sys
from pathlib import Path

import numpy as np
import scipy.io

# Umożliwia import pakietu ml/ przy uruchamianiu przez `streamlit run app/...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.build_dataset import aggregate_trial_features
from ml.config import (
    DE_VALUE_MAX,
    DE_VALUE_MIN,
    EEG_MEAN_MAX_UV,
    EEG_STD_MAX_UV,
    EEG_STD_MIN_UV,
    N_BANDS,
    N_CHANNELS,
    SAMPLING_RATE,
    SEED_VII_CHANNELS,
)
from ml.feature_extraction import WINDOW_SECONDS, extract_de_features

SAMPLING_RATE_KEYS = ("fs", "srate", "sampling_rate", "sr")
CHANNEL_KEYS = ("channels", "clab", "channel_names", "chan_names")
METADATA_KEYS = set(SAMPLING_RATE_KEYS + CHANNEL_KEYS)


class MatFileError(ValueError):
    """Błąd walidacji wgranego pliku .mat (z czytelnym komunikatem dla użytkownika)."""


def validate_sampling_rate(fs_val) -> float:
    """Sprawdza, czy podana częstotliwość próbkowania wynosi 200 Hz (wymóg modelu SEED-VII)."""
    try:
        fs = float(np.squeeze(fs_val))
    except (ValueError, TypeError) as e:
        raise MatFileError(f"Nieprawidłowy format metadanych częstotliwości próbkowania: {fs_val}") from e
    if abs(fs - SAMPLING_RATE) > 1e-2:
        raise MatFileError(
            f"Niezgodna częstotliwość próbkowania: {fs:g} Hz. "
            f"Model i filtry pasmowe SEED-VII wymagają dokładnie {SAMPLING_RATE} Hz (dane preprocessed)."
        )
    return fs


def validate_channel_names(channels_val) -> list[str]:
    """Sprawdza, czy lista kanałów odpowiada układowi 62 kanałów SEED-VII (układ 10-20 Neuroscan)."""
    if isinstance(channels_val, np.ndarray):
        if channels_val.dtype.kind in ("U", "S", "O"):
            channels_list = [str(x).strip() for x in channels_val.flatten()]
        else:
            raise MatFileError("Metadane kanałów mają nieprawidłowy format tablicy.")
    elif isinstance(channels_val, (list, tuple)):
        channels_list = [str(x).strip() for x in channels_val]
    else:
        raise MatFileError(f"Nieobsługiwany typ metadanych kanałów: {type(channels_val)}")

    if len(channels_list) != N_CHANNELS:
        raise MatFileError(
            f"Nieprawidłowa liczba kanałów w metadanych: {len(channels_list)}. "
            f"Model SEED-VII wymaga dokładnie {N_CHANNELS} kanałów."
        )

    expected = [c.upper() for c in SEED_VII_CHANNELS]
    actual = [c.upper() for c in channels_list]
    mismatches = [(i + 1, exp, act) for i, (exp, act) in enumerate(zip(expected, actual)) if exp != act]
    if mismatches:
        err_details = ", ".join(f"poz. {pos}: oczekiwano '{exp}', otrzymano '{act}'" for pos, exp, act in mismatches[:3])
        if len(mismatches) > 3:
            err_details += f" (oraz {len(mismatches) - 3} innych)"
        raise MatFileError(
            f"Niezgodność kolejności lub nazw 62 kanałów EEG ({err_details}). "
            "Model wymaga układu elektrod 10-20 Neuroscan z bazy SEED-VII."
        )
    return channels_list


def validate_feature_values(array: np.ndarray, key: str = "") -> None:
    """Weryfikuje, czy tablica zawiera poprawne cechy Differential Entropy (DE), a nie PSD."""
    if key and (key.lower().startswith("psd_") or "_psd_" in key.lower()):
        raise MatFileError(
            f"Odrzucono tablicę '{key}': Wykryto cechy PSD (Power Spectral Density). "
            "Model SEED-VII został wytrenowany na cechach Differential Entropy (DE) z wygładzaniem LDS. "
            "Wybierz tablicę o prefiksie 'de_LDS_'."
        )

    if not np.isfinite(array).all():
        raise MatFileError(f"Tablica '{key or 'features'}' zawiera wartości NaN lub nieskończone.")

    mean_val = float(np.mean(array))
    max_val = float(np.max(array))

    if mean_val > DE_VALUE_MAX or max_val > 500.0:
        raise MatFileError(
            f"Wartości w tablicy '{key or 'features'}' (średnia: {mean_val:.1f}, maks: {max_val:.1f}) "
            f"przekraczają dopuszczalny zakres Differential Entropy dla EEG (oczekiwano średniej w przedziale {DE_VALUE_MIN}–{DE_VALUE_MAX}). "
            "Dane wskazują na gęstość widmową mocy (PSD) lub inną miarę zamiast cech DE."
        )
    if mean_val < DE_VALUE_MIN:
        raise MatFileError(
            f"Wartości w tablicy '{key or 'features'}' (średnia: {mean_val:.1f}) "
            f"są nienaturalnie niskie dla cech DE EEG (min: {DE_VALUE_MIN})."
        )


def validate_signal_properties(signal: np.ndarray, key: str = "") -> None:
    """Weryfikuje fizjologiczne statystyki sygnału EEG (62, N) – ochrona przed sygnałem raw / V / szumem."""
    if not np.isfinite(signal).all():
        raise MatFileError(f"Sygnał '{key or 'signal'}' zawiera wartości NaN lub nieskończone.")

    std_val = float(np.std(signal))
    mean_val = float(np.abs(np.mean(signal)))

    if std_val < EEG_STD_MIN_UV:
        raise MatFileError(
            f"Amplituda sygnału '{key or 'signal'}' jest nienaturalnie niska (std = {std_val:.4f} µV < {EEG_STD_MIN_UV} µV). "
            "Prawdopodobnie dane zapisano w woltach (V) zamiast mikrowoltów (µV) lub sygnał jest uszkodzony."
        )
    if std_val > EEG_STD_MAX_UV:
        raise MatFileError(
            f"Amplituda sygnału '{key or 'signal'}' jest nienaturalnie wysoka (std = {std_val:.1f} µV > {EEG_STD_MAX_UV} µV). "
            "Sygnał zawiera artefakty nasycenia przetwornika i nie spełnia wymogów preprocessingu SEED-VII."
        )
    if mean_val > EEG_MEAN_MAX_UV:
        raise MatFileError(
            f"Sygnał '{key or 'signal'}' wykazuje silny dryft stałoprądowy (|mean| = {mean_val:.1f} µV > {EEG_MEAN_MAX_UV} µV). "
            "Wymagany jest sygnał po filtracji pasmowoprzepustowej 0.3–50 Hz (konwencja EEG_preprocessed SEED-VII)."
        )


def load_mat(file) -> dict:
    """Wczytuje plik .mat, weryfikuje metadane (fs, channels) i zwraca słownik tablic."""
    try:
        data = scipy.io.loadmat(file)
    except Exception as e:
        raise MatFileError(f"Nie udało się wczytać pliku .mat: {e}") from e

    # Weryfikacja metadanych próbkowania, jeśli występują
    for fs_key in SAMPLING_RATE_KEYS:
        if fs_key in data:
            validate_sampling_rate(data[fs_key])

    # Weryfikacja metadanych kanałów, jeśli występują
    for ch_key in CHANNEL_KEYS:
        if ch_key in data:
            validate_channel_names(data[ch_key])

    arrays = {k: v for k, v in data.items()
              if not k.startswith("__")
              and k.lower() not in METADATA_KEYS
              and isinstance(v, np.ndarray)}
    if not arrays:
        raise MatFileError("Plik .mat nie zawiera żadnych tablic z danymi sygnałowymi ani cechami.")
    return arrays


def detect_format(array: np.ndarray, key: str = "") -> str:
    """
    Rozpoznaje format tablicy z walidacją numeryczną:
      - 'features': cechy DE (okna, 5, 62) lub (5, 62),
      - 'signal': sygnał EEG (62, N) @ 200 Hz.

    Wyjątki:
        MatFileError: format nierozpoznany lub dane niezgodne z wymogami SEED-VII.
    """
    if key and (key.lower().startswith("psd_") or "_psd_" in key.lower()):
        raise MatFileError(
            f"Odrzucono tablicę '{key}': Wykryto cechy PSD (Power Spectral Density). "
            "Model SEED-VII wymaga cech Differential Entropy (klucze 'de_LDS_*' lub 'de_*')."
        )

    array = np.asarray(array)

    # Detekcja cech okulograficznych (EYE_features z 33 parametrami)
    if (array.ndim == 2 and 33 in array.shape) or (array.ndim == 1 and len(array) == 33):
        raise MatFileError(
            f"Odrzucono tablicę '{key}': Wykryto 33 parametry (kształt {array.shape}), "
            "co wskazuje na dane okulograficzne (EYE_features — śledzenie wzroku). "
            "Model SEED-VII operuje wyłącznie na 62 kanałach EEG. "
            "Wybierz plik z cechami EEG (katalog EEG_features z cechami de_LDS_1..80) "
            "lub przetworzonym sygnałem (katalog EEG_preprocessed)."
        )

    # Cechy DE: (okna, 5, 62) lub (5, 62) wraz z orientacją transponowaną
    if (array.ndim == 3 and array.shape[1:] in ((N_BANDS, N_CHANNELS), (N_CHANNELS, N_BANDS))) or \
       (array.ndim == 2 and array.shape in ((N_BANDS, N_CHANNELS), (N_CHANNELS, N_BANDS))):
        validate_feature_values(array, key=key)
        return "features"

    # Sygnał EEG: (62, N) lub (N, 62) przy N >= 200 próbek
    if array.ndim == 2 and array.shape[0] == N_CHANNELS and array.shape[1] >= SAMPLING_RATE:
        validate_signal_properties(array, key=key)
        return "signal"
    if array.ndim == 2 and array.shape[1] == N_CHANNELS and array.shape[0] >= SAMPLING_RATE:
        validate_signal_properties(array.T, key=key)
        return "signal"
    raise MatFileError(
        f"Nierozpoznany format danych o kształcie {array.shape}. "
        f"Oczekiwano cech DE (okna, {N_BANDS}, {N_CHANNELS}) lub sygnału ({N_CHANNELS}, N >= {SAMPLING_RATE}) zgodnego z SEED-VII."
    )


def normalize_signal_shape(array: np.ndarray) -> np.ndarray:
    """Ujednolica orientację sygnału do formatu (62, N)."""
    array = np.asarray(array)
    if array.ndim == 2 and array.shape[1] == N_CHANNELS and array.shape[0] > N_CHANNELS:
        return array.T
    return array


def list_trials(arrays: dict) -> dict:
    """
    Zwraca słownik {etykieta_próby: (klucz, format)} dla tablic o rozpoznanym formacie.
    Akceptuje klucze cech `de_LDS_n` oraz sygnał 62-kanałowy.
    Jawnie odrzuca i blokuje PSD oraz dane okulograficzne (EYE_features).
    """
    keys_no_meta = [k for k in arrays if k.lower() not in METADATA_KEYS]
    if keys_no_meta and all(k.lower().startswith("psd_") or "_psd_" in k.lower() for k in keys_no_meta):
        raise MatFileError(
            "Wykryto wyłącznie tablice cech PSD (Power Spectral Density). "
            "Model SEED-VII został wytrenowany na cechach Differential Entropy (DE) z wygładzaniem LDS. "
            "Wymagane są tablice 'de_LDS_*' lub przetworzony sygnał EEG."
        )

    # Sprawdzenie czy cały plik nie pochodzi z modułu okulograficznego (33 cechy)
    if keys_no_meta and all(
        isinstance(arrays[k], np.ndarray) and (arrays[k].ndim == 2 and 33 in arrays[k].shape)
        for k in keys_no_meta
    ):
        raise MatFileError(
            "Wykryto plik z cechami okulograficznymi (EYE_features — 33 parametry śledzenia wzroku, np. średnica źrenicy). "
            "Model SEED-VII rozpoznaje emocje wyłącznie z 62-kanałowego EEG. "
            "Użyj pliku z katalogu EEG_features (zawierającego tablice de_LDS_1..80) "
            "lub z katalogu EEG_preprocessed (sygnał EEG 0.3–50 Hz)."
        )

    trials = {}
    def priority(k):
        if k.startswith("de_LDS_") or k.startswith("de_LDS"):
            return (0, k)
        if k.startswith("de_") or k.startswith("de"):
            return (1, k)
        return (2, k)

    seen_trials = set()
    for key in sorted(arrays, key=priority):
        if key.lower() in METADATA_KEYS or key.lower().startswith("psd_"):
            continue
        try:
            fmt = detect_format(arrays[key], key=key)
        except MatFileError:
            continue
        if fmt == "features" and not key.lower().startswith("de_lds_"):
            continue
        # Grupuj po numerze próby, żeby nie dublować de_n / de_LDS_n
        trial_id = key.split("_")[-1]
        if trial_id in seen_trials and key.startswith("de_"):
            continue
        seen_trials.add(trial_id)
        trials[f"{key} ({fmt})"] = (key, fmt)

    if not trials:
        eye_found = any(
            isinstance(v, np.ndarray) and (v.ndim == 2 and 33 in v.shape)
            for v in arrays.values()
        )
        if eye_found:
            raise MatFileError(
                "Wykryto plik z cechami okulograficznymi (EYE_features — 33 parametry śledzenia wzroku). "
                "Aplikacja rozpoznaje emocje wyłącznie z 62 kanałów EEG. "
                "Użyj pliku z katalogu EEG_features (cechy de_LDS_1..80) lub EEG_preprocessed (sygnał EEG)."
            )
        raise MatFileError(
            "Żadna tablica w pliku nie spełnia kryteriów formatu SEED-VII "
            f"(wymagane cechy DE (okna, {N_BANDS} pasm, {N_CHANNELS} kanałów) o wartościach {DE_VALUE_MIN}–{DE_VALUE_MAX} "
            f"lub odfiltrowany sygnał EEG ({N_CHANNELS} kanałów, min. {SAMPLING_RATE} próbek) @ {SAMPLING_RATE} Hz)."
        )
    return trials


def slice_signal(signal: np.ndarray, start_sec: float, end_sec: float,
                 fs: int = SAMPLING_RATE) -> np.ndarray:
    """
    Wycina fragment sygnału EEG w zadanym przedziale czasowym [start_sec, end_sec].

    Parametry:
        signal: tablica (62, N) lub (N, 62)
        start_sec: początek w sekundach
        end_sec: koniec w sekundach
        fs: częstotliwość próbkowania w Hz
    """
    signal = normalize_signal_shape(signal)
    if signal.ndim != 2 or signal.shape[0] != N_CHANNELS:
        raise MatFileError(f"Oczekiwano sygnału ({N_CHANNELS}, N), otrzymano {signal.shape}")

    total_samples = signal.shape[1]
    total_sec = total_samples / fs

    if start_sec < 0 or end_sec > total_sec or start_sec >= end_sec:
        raise MatFileError(
            f"Nieprawidłowy zakres czasowy: [{start_sec:.1f} s, {end_sec:.1f} s]. "
            f"Długość sygnału wynosi {total_sec:.1f} s."
        )

    start_idx = int(round(start_sec * fs))
    end_idx = int(round(end_sec * fs))
    sliced = signal[:, start_idx:end_idx]

    min_len = WINDOW_SECONDS * fs
    if sliced.shape[1] < min_len:
        raise MatFileError(
            f"Wycięty fragment jest za krótki ({sliced.shape[1] / fs:.1f} s). "
            f"Wymagane minimum to {WINDOW_SECONDS} s."
        )
    return sliced


def export_to_mat_bytes(array: np.ndarray, key_name: str = "signal") -> bytes:
    """
    Serializuje tablicę numpy do pliku .mat w pamięci RAM i zwraca bajty.
    """
    buf = io.BytesIO()
    scipy.io.savemat(buf, {key_name: np.asarray(array)})
    return buf.getvalue()


def split_mat_file(input_path: Path, output_dir: Path,
                   segment_length_sec: float | None = None,
                   fs: int = SAMPLING_RATE) -> list[Path]:
    """
    Rozbija duży plik .mat na mniejsze, pojedyncze pliki prób w folderze wyjściowym.

    - Jeśli plik zawiera wiele prób (np. klucze 1..80 lub de_LDS_1..80):
      zapisuje każdą próbę jako osobny plik (np. trial_1.mat).
    - Jeśli plik zawiera sygnał ciągły i podano segment_length_sec:
      tnie sygnał na równe kawałki i zapisuje jako segment_1.mat, segment_2.mat itd.

    Zwraca:
        Listę ścieżek do utworzonych plików .mat.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_mat(str(input_path))
    trials = list_trials(arrays)
    created_files = []

    if len(trials) > 1:
        # Plik wielopróbkowy (np. SEED-VII z 80 próbami)
        for label, (key, fmt) in trials.items():
            arr = arrays[key]
            out_file = output_dir / f"trial_{key}.mat"
            scipy.io.savemat(str(out_file), {key: arr})
            created_files.append(out_file)
    else:
        # Plik z jedną próbą / sygnałem ciągłym
        first_label = list(trials.keys())[0]
        key, fmt = trials[first_label]
        arr = arrays[key]

        if fmt == "signal" and segment_length_sec is not None and segment_length_sec > 0:
            arr = normalize_signal_shape(arr)
            seg_samples = int(round(segment_length_sec * fs))
            total_samples = arr.shape[1]
            n_segments = total_samples // seg_samples
            if n_segments < 1:
                out_file = output_dir / f"segment_1.mat"
                scipy.io.savemat(str(out_file), {"1": arr})
                created_files.append(out_file)
            else:
                for i in range(n_segments):
                    seg_data = arr[:, i * seg_samples:(i + 1) * seg_samples]
                    out_file = output_dir / f"segment_{i + 1}.mat"
                    scipy.io.savemat(str(out_file), {"1": seg_data})
                    created_files.append(out_file)
        else:
            out_file = output_dir / f"sample_{key}.mat"
            scipy.io.savemat(str(out_file), {key: arr})
            created_files.append(out_file)

    return created_files


def prepare_feature_vector(array: np.ndarray, fmt: str, smooth: bool = True):
    """
    Przygotowuje wektor 310 cech do predykcji.

    Parametry:
        smooth: dla sygnału — wygładzanie LDS cech DE (domyślnie włączone,
                spójne z pipeline'em treningowym `dataset_own.npz`).

    Zwraca:
        (wektor (310,), cechy (okna, 5, 62)) — cechy per okno do wizualizacji.

    Wyjątki:
        MatFileError: zła liczba kanałów / za krótki sygnał / wartości niespełniające kryteriów SEED-VII.
    """
    try:
        if fmt == "signal":
            array = normalize_signal_shape(array)
            validate_signal_properties(array, key="signal")
            min_len = WINDOW_SECONDS * SAMPLING_RATE
            if array.shape[1] < min_len:
                raise MatFileError(
                    f"Sygnał za krótki: {array.shape[1] / SAMPLING_RATE:.1f} s "
                    f"(minimum {WINDOW_SECONDS} s przy {SAMPLING_RATE} Hz)."
                )
            features = extract_de_features(array, smooth=smooth)
        else:
            features = np.asarray(array, dtype=np.float64)
            if features.ndim == 2:
                if features.shape == (N_CHANNELS, N_BANDS):
                    features = features.T
                features = features[np.newaxis, ...]
            if features.ndim == 3 and features.shape[1:] == (N_CHANNELS, N_BANDS):
                features = np.transpose(features, (0, 2, 1))
            validate_feature_values(features, key="features")
        vector = aggregate_trial_features(features)
    except MatFileError:
        raise
    except ValueError as e:
        raise MatFileError(str(e)) from e
    return vector, features
