"""
Ekstrakcja cech Differential Entropy (DE) z przetworzonego sygnału EEG.

Sygnał wejściowy: (62 kanały, N próbek) @ 200 Hz (jak w EEG_preprocessed).
Metodologia zgodna z SEED: sygnał dzielony na nienakładające się okna 4 s,
w każdym oknie filtracja pasmowa (5 pasm: delta/theta/alpha/beta/gamma)
i obliczenie DE przy założeniu rozkładu Gaussa:

    DE = 0.5 * ln(2 * pi * e * sigma^2)

Wynik: tablica (okna, 5 pasm, 62 kanały) — ten sam układ co cechy `de_n`
w EEG_features, dzięki czemu moduł jest współdzielony przez trening i aplikację.
"""

import numpy as np
from scipy.signal import butter, filtfilt

from ml.config import FREQ_BANDS, N_CHANNELS, SAMPLING_RATE

WINDOW_SECONDS = 4  # długość okna czasowego w sekundach (zgodna z SEED-VII:
# np. próba 1 ma 61 s sygnału i 15 okien cech -> 61 // 4 = 15)


def bandpass_filter(signal: np.ndarray, low: float, high: float,
                    fs: int = SAMPLING_RATE, order: int = 4) -> np.ndarray:
    """
    Filtracja pasmowoprzepustowa Butterwortha (zero-fazowa, filtfilt).

    Parametry:
        signal: tablica (kanały, próbki),
        low, high: granice pasma w Hz.
    """
    nyq = fs / 2
    high = min(high, nyq * 0.99)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=-1)


def differential_entropy(signal: np.ndarray) -> np.ndarray:
    """
    DE sygnału gaussowskiego: 0.5*ln(2*pi*e*sigma^2), liczona po ostatniej osi.
    """
    variance = np.var(signal, axis=-1)
    variance = np.maximum(variance, 1e-12)  # ochrona przed log(0)
    return 0.5 * np.log(2 * np.pi * np.e * variance)


def extract_de_features(signal: np.ndarray, fs: int = SAMPLING_RATE,
                        smooth: bool = False) -> np.ndarray:
    """
    Liczy cechy DE z sygnału EEG.

    Parametry:
        signal: tablica (62, N) — sygnał @ fs Hz.
        smooth: czy wygładzić sekwencję okien algorytmem LDS (`lds_smooth`).

    Zwraca:
        tablica (okna, 5, 62) — DE per okno 4 s, pasmo i kanał.

    Wyjątki:
        ValueError: zła liczba kanałów lub sygnał krótszy niż 1 okno.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] != N_CHANNELS:
        raise ValueError(
            f"Oczekiwano sygnału o kształcie ({N_CHANNELS}, N), otrzymano: {signal.shape}"
        )
    window_len = WINDOW_SECONDS * fs
    n_windows = signal.shape[1] // window_len
    if n_windows < 1:
        raise ValueError(
            f"Sygnał za krótki: {signal.shape[1]} próbek (< {window_len} = 1 okno {WINDOW_SECONDS} s @ {fs} Hz)"
        )

    # Filtracja całego sygnału raz na pasmo (stabilniejsza niż per okno)
    band_signals = [
        bandpass_filter(signal, low, high, fs) for (low, high) in FREQ_BANDS.values()
    ]

    features = np.empty((n_windows, len(FREQ_BANDS), N_CHANNELS))
    for w in range(n_windows):
        start, stop = w * window_len, (w + 1) * window_len
        for b, band_signal in enumerate(band_signals):
            features[w, b, :] = differential_entropy(band_signal[:, start:stop])

    if smooth:
        features = lds_smooth(features)
    return features


def lds_smooth(features: np.ndarray, q: float = 1e-3, r: float = 1e-2) -> np.ndarray:
    """
    Wygładzanie sekwencji cech w czasie liniowym systemem dynamicznym (LDS),
    tj. filtrem Kalmana z wygładzaniem wstecznym RTS.

    Model przestrzeni stanów (losowy spacer):
        x_t = x_{t-1} + w_t,  w_t ~ N(0, q)   (szum procesu)
        y_t = x_t + v_t,      v_t ~ N(0, r)   (szum pomiaru)

    Wygładzanie działa niezależnie dla każdej pary (pasmo, kanał) wzdłuż osi
    okien czasowych. Implementacja własna na podstawie równań podręcznikowych:
    Kalman (1960); Rauch, Tung & Striebel (1965); zastosowanie do cech EEG:
    Shi & Lu (2010), Zheng & Lu (2015).

    Parametry:
        features: tablica (okna, 5, 62) — sekwencja cech DE,
        q: wariancja szumu procesu (im mniejsza, tym mocniejsze wygładzanie),
        r: wariancja szumu pomiaru.

    Zwraca:
        tablicę (okna, 5, 62) — wygładzone cechy.
    """
    y = np.asarray(features, dtype=np.float64)
    T = y.shape[0]
    if T < 2:
        return y.copy()

    # --- filtr Kalmana (przejście w przód) ---
    x_f = np.empty_like(y)   # estymaty filtrowane
    p_f = np.empty(T)        # wariancje estymat (wspólne dla wszystkich cech)
    x_f[0], p_f[0] = y[0], 1.0
    for t in range(1, T):
        p_pred = p_f[t - 1] + q                          # predykcja wariancji
        k = p_pred / (p_pred + r)                        # wzmocnienie Kalmana
        x_f[t] = x_f[t - 1] + k * (y[t] - x_f[t - 1])    # korekcja
        p_f[t] = (1.0 - k) * p_pred

    # --- wygładzanie RTS (przejście w tył) ---
    x_s = x_f.copy()
    for t in range(T - 2, -1, -1):
        g = p_f[t] / (p_f[t] + q)                        # wzmocnienie wygładzania
        x_s[t] = x_f[t] + g * (x_s[t + 1] - x_f[t])
    return x_s
