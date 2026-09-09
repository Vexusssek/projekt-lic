"""
Konfiguracja projektu SEED-VII — rozpoznawanie emocji z sygnałów EEG.

Zawiera stałe współdzielone przez skrypty treningowe (ml/) i aplikację (app/):
mapę emocji, kolejność bodźców (trial -> emocja), pasma częstotliwości,
ścieżki do danych oraz ziarno losowości.
"""

import os
from pathlib import Path

# --- Ścieżki -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _find_data_source(rel_path: str) -> Path:
    """
    Wyszukuje surowe zbiory danych bez ich modyfikacji:
    1. Wskazane zmienną środowiskową SEED_VII_DATA_DIR
    2. Bezpośrednio w folderze projektu
    3. W katalogu nadrzędnym (np. gdy projekt jest podkatalogiem zbioru SEED-VII)
    """
    env_dir = os.environ.get("SEED_VII_DATA_DIR")
    if env_dir and (Path(env_dir) / rel_path).exists():
        return Path(env_dir) / rel_path
    if (PROJECT_ROOT / rel_path).exists():
        return PROJECT_ROOT / rel_path
    if (PROJECT_ROOT.parent / rel_path).exists():
        return PROJECT_ROOT.parent / rel_path
    return PROJECT_ROOT / rel_path


FEATURES_DIR = _find_data_source("EEG_features")
PREPROCESSED_DIR = _find_data_source("EEG_preprocessed")
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "dataset.npz"
# Dataset zbudowany własnym ekstraktorem DE z sygnału EEG_preprocessed
OWN_DATASET_PATH = DATA_DIR / "dataset_own.npz"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
# Model trenowany na cechach z własnego ekstraktora (ścieżka sygnałowa aplikacji)
OWN_MODEL_PATH = MODELS_DIR / "best_model_own_features.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- Powtarzalność -----------------------------------------------------------
RANDOM_STATE = 42

# --- Emocje (7 klas) ---------------------------------------------------------
EMOTION_MAP = {
    "Happy": 0,
    "Sad": 1,
    "Fear": 2,
    "Disgust": 3,
    "Surprise": 4,
    "Anger": 5,
    "Neutral": 6,
}
EMOTION_NAMES = {v: k for k, v in EMOTION_MAP.items()}

# Zunifikowane polskie nazwy i spójna paleta kolorów dla interfejsu oraz raportów PDF
EMOTION_POLISH = {
    "Happy": {"pl": "Radość", "color": "#dcb265"},
    "Sad": {"pl": "Smutek", "color": "#7c90b6"},
    "Fear": {"pl": "Strach", "color": "#9d8cb8"},
    "Disgust": {"pl": "Wstręt", "color": "#729e88"},
    "Surprise": {"pl": "Zaskoczenie", "color": "#d68d65"},
    "Anger": {"pl": "Gniew", "color": "#c86565"},
    "Neutral": {"pl": "Stan neutralny", "color": "#8a877d"},
}

# --- Mapowanie trial (1-80) -> emocja ----------------------------------------
# Kolejność bodźców z pliku emotion_label_and_stimuli_order.xlsx
# (4 sesje po 20 filmów; wiersze zweryfikowane ręcznie z arkuszem).
_SESSION_EMOTIONS = [
    # Session 1: trials 1-20
    ["Happy", "Neutral", "Disgust", "Sad", "Anger",
     "Anger", "Sad", "Disgust", "Neutral", "Happy",
     "Happy", "Neutral", "Disgust", "Sad", "Anger",
     "Anger", "Sad", "Disgust", "Neutral", "Happy"],
    # Session 2: trials 21-40
    ["Anger", "Sad", "Fear", "Neutral", "Surprise",
     "Surprise", "Neutral", "Fear", "Sad", "Anger",
     "Anger", "Sad", "Fear", "Neutral", "Surprise",
     "Surprise", "Neutral", "Fear", "Sad", "Anger"],
    # Session 3: trials 41-60
    ["Happy", "Surprise", "Disgust", "Fear", "Anger",
     "Anger", "Fear", "Disgust", "Surprise", "Happy",
     "Happy", "Surprise", "Disgust", "Fear", "Anger",
     "Anger", "Fear", "Disgust", "Surprise", "Happy"],
    # Session 4: trials 61-80
    ["Disgust", "Sad", "Fear", "Surprise", "Happy",
     "Happy", "Surprise", "Fear", "Sad", "Disgust",
     "Disgust", "Sad", "Fear", "Surprise", "Happy",
     "Happy", "Surprise", "Fear", "Sad", "Disgust"],
]

# Płaska lista 80 emocji: TRIAL_EMOTIONS[i] to emocja próby o numerze i+1.
TRIAL_EMOTIONS = [e for session in _SESSION_EMOTIONS for e in session]
assert len(TRIAL_EMOTIONS) == 80

N_TRIALS = 80
N_SUBJECTS = 20
N_CHANNELS = 62

# --- Pasma częstotliwości (Hz) — zgodne z metodologią SEED --------------------
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 14),
    "beta": (14, 31),
    "gamma": (31, 50),
}
BAND_NAMES = list(FREQ_BANDS.keys())
N_BANDS = len(FREQ_BANDS)

SAMPLING_RATE = 200  # Hz, sygnał z EEG_preprocessed

# Kolejność i nazewnictwo 62 kanałów wg czepka Neuroscan w SEED-VII (układ 10-20)
SEED_VII_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ",
    "O2", "CB2",
]
assert len(SEED_VII_CHANNELS) == N_CHANNELS

# Progi podstawowej kontroli wartości wejściowych
DE_VALUE_MIN = -20.0     # min logiczny zakres Differential Entropy dla EEG
DE_VALUE_MAX = 80.0      # max logiczny zakres DE (wartości > 100 to PSD lub surowa moc)
EEG_STD_MIN_UV = 0.5     # minimalne dopuszczalne std sygnału (poniżej to sygnał w woltach V lub płaski)
EEG_STD_MAX_UV = 1500.0  # maksymalne dopuszczalne std sygnału (powyżej to nasycenie przetwornika/artefakt)
EEG_MEAN_MAX_UV = 250.0  # maksymalny dryft stałoprądowy DC (preprocessed SEED-VII ma filtrację 0.3-50 Hz)

# Typ cech używany do treningu i predykcji
FEATURE_TYPE = "de_lds_mean"  # średnia cech de_LDS po oknach czasowych
OWN_FEATURE_TYPE = "de_own_lds_mean"  # cechy z własnego ekstraktora DE + wygładzanie LDS

# Kanały prezentowane na wykresach. Nazwy są zawsze pobierane z SEED_VII_CHANNELS.
PLOT_CHANNELS = (
    (0, "czołowy"),
    (16, "czołowo-centralny"),
    (27, "centralny"),
    (43, "ciemieniowy"),
    (61, "móżdżkowy"),
)


def trial_to_emotion(trial_number: int) -> str:
    """Zwraca nazwę emocji dla numeru próby (1-80)."""
    if not 1 <= trial_number <= N_TRIALS:
        raise ValueError(f"Numer próby musi być w zakresie 1-{N_TRIALS}, otrzymano: {trial_number}")
    return TRIAL_EMOTIONS[trial_number - 1]


def trial_to_label(trial_number: int) -> int:
    """Zwraca etykietę liczbową (0-6) dla numeru próby (1-80)."""
    return EMOTION_MAP[trial_to_emotion(trial_number)]
