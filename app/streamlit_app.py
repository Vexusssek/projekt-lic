"""Interfejs Streamlit do analizy danych EEG zgodnych ze zbiorem SEED-VII."""

from datetime import datetime
from html import escape
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pdf_report import generate_pdf_report
from app.model_service import Prediction, predict, validate_artifact
from app.utils import (
    MatFileError,
    export_to_mat_bytes,
    list_trials,
    load_mat,
    normalize_signal_shape,
    prepare_feature_vector,
    slice_signal,
)
from ml.config import (
    BAND_NAMES,
    BEST_MODEL_PATH,
    DATA_DIR,
    EMOTION_NAMES,
    EMOTION_POLISH,
    FEATURE_TYPE,
    FEATURES_DIR,
    OWN_FEATURE_TYPE,
    OWN_MODEL_PATH,
    PLOT_CHANNELS,
    PREPROCESSED_DIR,
    SAMPLING_RATE,
    SEED_VII_CHANNELS,
)


st.set_page_config(
    page_title="Rozpoznawanie Emocji EEG • SEED-VII",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def load_custom_css():
    """Wczytuje arkusz stylów CSS."""
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        st.html(css_path)


load_custom_css()


@st.cache_resource
def load_artifact(model_path: Path):
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def get_artifact_for_format(fmt: str):
    """Wczytuje artefakt zgodny z typem wejścia i sprawdza jego metadane."""
    model_path, feature_type = (
        (OWN_MODEL_PATH, OWN_FEATURE_TYPE) if fmt == "signal"
        else (BEST_MODEL_PATH, FEATURE_TYPE)
    )
    artifact = load_artifact(model_path)
    if artifact is None:
        raise MatFileError(f"Nie znaleziono modelu wymaganego dla tego wejścia: {model_path}")
    try:
        validate_artifact(artifact, model_path, feature_type)
    except ValueError as exc:
        raise MatFileError(str(exc)) from exc
    return artifact


def get_available_local_files() -> dict:
    files = {}
    for p in sorted(DATA_DIR.glob("*.mat")):
        files[f"data/{p.name}"] = p
    samples_dir = DATA_DIR / "samples"
    if samples_dir.exists():
        for p in sorted(samples_dir.rglob("*.mat")):
            rel = p.relative_to(DATA_DIR)
            files[f"data/{rel}"] = p
    if PREPROCESSED_DIR.exists():
        for p in sorted(PREPROCESSED_DIR.glob("*.mat")):
            files[f"EEG_preprocessed/{p.name} (~1.3 GB)"] = p
    if FEATURES_DIR.exists():
        for p in sorted(FEATURES_DIR.glob("*.mat")):
            files[f"EEG_features/{p.name}"] = p
    return files


def render_hero_screen():
    """Ekran 1: Gradientowa plansza na pełen ekran z tytułem i strzałką w dół."""
    st.markdown(
        '<div class="hero-fullscreen">'
        '<div class="hero-header-box">'
        '<h1 class="hero-main-title">Rozpoznawanie Emocji na podstawie sygnału EEG</h1>'
        '<div class="hero-subtitle">Interfejs afektywny SEED-VII z predykcją wieloklasową</div>'
        '</div>'
        '<a href="#source-section" class="scroll-arrow-anchor" onclick="const el=document.getElementById(\'source-section\'); if(el){el.scrollIntoView({behavior:\'smooth\'});} return true;">'
        '<div class="scroll-circle-btn" title="Przejdź do wyboru danych">↓</div>'
        '</a>'
        '</div>'
        '<div id="source-section"></div>',
        unsafe_allow_html=True,
    )


def render_tech_chips(fmt: str, current_data: np.ndarray) -> str:
    """Generuje spójny pasek metadanych technicznych sygnału lub cech."""
    chips = [
        ("Sygnał", "62 kanały (10-20)"),
        ("Założone próbkowanie", f"{SAMPLING_RATE} Hz"),
        ("Wymagane pasmo", "0.3–50 Hz"),
    ]
    if fmt == "signal":
        total_sec = current_data.shape[1] / SAMPLING_RATE
        chips.extend([
            ("Długość", f"{total_sec:.1f} s ({current_data.shape[1]} próbek)"),
            ("Wymagany format", "SEED-VII preprocessed"),
        ])
    else:
        chips.insert(0, ("Cechy", "Differential Entropy (de_LDS)"))
        n_windows = 1 if current_data.ndim == 2 else current_data.shape[0]
        chips.append(("Okna", f"{n_windows} segmentów"))

    items = "".join(
        f'<div class="tech-chip"><span class="tech-chip-key">{k}:</span>'
        f'<span class="tech-chip-val">{v}</span></div>'
        for k, v in chips
    )
    return f'<div class="tech-chips-grid">{items}</div>'


def show_prediction_results(result: Prediction):
    """Wyświetla najwyżej ocenioną klasę oraz rozkład wyników modelu."""
    proba = result.probabilities
    classes = result.classes
    best_idx = int(np.argmax(proba))
    conf = float(proba[best_idx])
    pred_eng = EMOTION_NAMES[int(classes[best_idx])]
    pred_info = EMOTION_POLISH.get(pred_eng, {"pl": pred_eng, "color": "#dcb265"})
    sorted_indices = np.argsort(proba)[::-1]
    distribution_rows = []
    for idx in sorted_indices:
        c = classes[idx]
        name_eng = EMOTION_NAMES[int(c)]
        info = EMOTION_POLISH.get(name_eng, {"pl": name_eng, "color": "#dcb265"})
        p = proba[idx]
        distribution_rows.append(
            f'<div class="distribution-row">'
            f'<div class="distribution-header">'
            f'<span class="distribution-label">{info["pl"]}</span>'
            f'<span class="distribution-value">{p:.1%}</span>'
            f'</div>'
            f'<div class="distribution-track">'
            f'<div class="distribution-bar" style="width: {p*100:.1f}%; background-color: {info["color"]};"></div>'
            f'</div>'
            f'</div>'
        )

    rows_html = "".join(distribution_rows)

    result_html = (
        f'<div class="result-card-unified">'
        f'<div class="kpi-header">'
        f'<div>'
        f'<div class="result-header-label">Najwyższy wynik modelu</div>'
        f'<div class="kpi-title-row">'
        f'<span class="kpi-emotion-title">{pred_info["pl"]}</span>'
        f'<span class="kpi-emotion-pct">— {conf:.1%}</span>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div class="kpi-divider"></div>'
        f'<div class="distribution-section">'
        f'<div class="result-header-label" style="margin-bottom: 1.1rem;">Rozkład wyników modelu</div>'
        f'{rows_html}'
        f'</div>'
        f'</div>'
    )
    st.markdown(result_html, unsafe_allow_html=True)
    st.caption(
        "Wartości procentowe opisują rozkład odpowiedzi modelu i nie są "
        "oszacowaniem prawdopodobieństwa, że rozpoznanie jest poprawne."
    )




def show_signal_visualizations(features: np.ndarray, raw_array: np.ndarray, fmt: str):
    """Wyświetla mapę cech DE i wybrane kanały sygnału."""
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    title_text = (
        "Wizualizacja cech DE i przebiegu falowego EEG"
        if fmt == "signal"
        else "Wizualizacja cech Differential Entropy (DE)"
    )
    st.markdown(f"<div class='section-title' style='font-size: 1.3rem; margin-bottom: 1rem;'>{title_text}</div>", unsafe_allow_html=True)

    mean_features = features.mean(axis=0)  # (5, 62)
    fig, ax = plt.subplots(figsize=(12, 3), facecolor="#1b1a17")
    ax.set_facecolor("#1b1a17")

    im = ax.imshow(mean_features, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks(range(len(BAND_NAMES)))
    ax.set_yticklabels([f"{b.upper()} ({BAND_NAMES[i]})" for i, b in enumerate(["δ", "θ", "α", "β", "γ"])], color="#a8a59b", fontsize=8)
    ax.set_xlabel("Kanał EEG (1–62)", color="#a8a59b", fontsize=8, labelpad=6)
    ax.tick_params(colors="#6e6b62", labelsize=8)
    for s in ax.spines.values():
        s.set_color((0.92, 0.9, 0.86, 0.08))

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Wartość DE", color="#a8a59b", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#6e6b62", labelcolor="#a8a59b", labelsize=7)
    cbar.outline.set_edgecolor((0.92, 0.9, 0.86, 0.08))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if fmt == "signal":
        n_show = min(raw_array.shape[1], 10 * SAMPLING_RATE)
        t = np.arange(n_show) / SAMPLING_RATE
        fig, ax = plt.subplots(figsize=(12, 3.8), facecolor="#1b1a17")
        ax.set_facecolor("#1b1a17")

        ch_indices = [index for index, _ in PLOT_CHANNELS]
        ch_names = [
            f"{SEED_VII_CHANNELS[index]} ({region})"
            for index, region in PLOT_CHANNELS
        ]
        colors = ["#9d9fc4", "#dcb265", "#729e88", "#d68d65", "#7c90b6"]

        for i, ch in enumerate(ch_indices):
            offset = i * 4 * np.std(raw_array[ch, :n_show])
            ax.plot(t, raw_array[ch, :n_show] + offset, lw=0.75, color=colors[i], label=ch_names[i])

        ax.set_xlabel("Czas [s]", color="#a8a59b", fontsize=8)
        ax.set_yticks([])
        ax.tick_params(colors="#6e6b62", labelsize=8)
        for s in ax.spines.values():
            s.set_color((0.92, 0.9, 0.86, 0.08))
        ax.grid(True, color=(0.92, 0.9, 0.86, 0.04), linestyle="--")
        ax.legend(loc="upper right", fontsize=8, facecolor="#23221e", edgecolor=(0.92, 0.9, 0.86, 0.08), labelcolor="#ebe8df")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

def reset_loaded_state():
    """Resetuje stan załadowanych danych i odświeża interfejs."""
    st.session_state.data_loaded = False
    st.session_state.loaded_arrays = None
    st.session_state.source_name = ""
    st.session_state.uploader_key += 1
    st.rerun()


def main():
    st.session_state.setdefault("data_loaded", False)
    st.session_state.setdefault("uploader_key", 0)

    # Ekran 1: Główny tytuł i strzałka
    render_hero_screen()

    # Jeśli dane NIE są wgrane, pokazujemy dwie funkcjonalne karty wgrywania z animacjami
    if not st.session_state.data_loaded:
        st.markdown(
            '<div class="section-wrapper">'
            '<div class="section-title">Wybierz sposób wczytania danych</div>'
            '<a href="#upload-section" class="scroll-arrow-anchor" onclick="const el=document.getElementById(\'upload-section\'); if(el){el.scrollIntoView({behavior:\'smooth\'});} return true;">'
            '<div class="scroll-circle-btn" title="Przejdź do wyboru plików">↓</div>'
            '</a>'
            '</div>'
            '<div id="upload-section"></div>'
            '<div style="background: rgba(157, 159, 196, 0.06); border: 1px solid rgba(157, 159, 196, 0.2); border-radius: 8px; padding: 0.9rem 1.2rem; margin-bottom: 1.5rem;">'
            '<div style="font-weight: 600; font-size: 0.95rem; color: #9d9fc4; margin-bottom: 0.3rem;">'
            'Wymagania wejściowe modelu SEED-VII'
            '</div>'
            '<div style="font-size: 0.85rem; color: #a8a59b; line-height: 1.45;">'
            'Model wymaga <b>62 kanałów</b> w kolejności SEED-VII, próbkowania <b>200 Hz</b> i sygnału '
            '<b>SEED-VII preprocessed 0.3–50 Hz</b> albo cech <b>de_LDS_*</b>. Metadane są sprawdzane, '
            'jeśli znajdują się w pliku. Gdy ich brakuje, aplikacja nie może potwierdzić próbkowania ani kolejności kanałów.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        col_b1, col_b2 = st.columns(2, gap="large")

        with col_b1:
            local_files = get_available_local_files()
            selected_label = st.selectbox(
                "Wybierz plik z dysku lokalnego:",
                list(local_files.keys()) + ["Wprowadź własną ścieżkę..."],
                key="disk_file_select",
            )
            if selected_label == "Wprowadź własną ścieżkę...":
                custom_p = st.text_input("Podaj ścieżkę do pliku .mat:", key="disk_custom_path")
                target_path = Path(custom_p) if custom_p and Path(custom_p).exists() else None
            else:
                target_path = local_files[selected_label]

            with st.container(key="disk_card_container"):
                target_display = escape(target_path.name) if target_path else 'Brak pliku'
                st.markdown(
                    f'<div class="source-card source-card-disk">'
                    f'<div class="source-card-title">Wgraj z dysku lokalnego</div>'
                    f'<div class="source-card-desc">Wybrany plik: <b>{target_display}</b></div>'
                    f'<div class="source-card-cta">KLIKNIJ TO OKNO, ABY WCZYTAĆ</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button("Wczytaj z dysku", width="stretch", key="btn_load_disk"):
                    if target_path and target_path.exists():
                        try:
                            arrays = load_mat(str(target_path))
                            st.session_state.loaded_arrays = arrays
                            st.session_state.source_name = target_path.name
                            st.session_state.data_loaded = True
                            st.rerun()
                        except MatFileError as e:
                            st.error(str(e))
                    else:
                        st.error("Wskazany plik nie istnieje.")

        with col_b2:
            st.markdown(
                '<div class="source-top-panel">'
                '<div class="source-top-label">Wgraj plik przez przeglądarkę:</div>'
                '<div class="source-top-info">Obsługa plików .mat (do 200 MB)</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            with st.container(key="browser_card_container"):
                st.markdown(
                    f'<div class="source-card source-card-browser">'
                    f'<div class="source-card-title">Wgraj przez przeglądarkę</div>'
                    f'<div class="source-card-desc">Wybierz plik z dowolnego folderu</div>'
                    f'<div class="source-card-cta">KLIKNIJ TO OKNO LUB UPUŚĆ PLIK TUTAJ</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                uploaded = st.file_uploader(
                    "Wybierz plik .mat",
                    type=["mat"],
                    key=f"uploader_{st.session_state.uploader_key}",
                    label_visibility="collapsed",
                )
                if uploaded is not None:
                    try:
                        arrays = load_mat(uploaded)
                        st.session_state.loaded_arrays = arrays
                        st.session_state.source_name = uploaded.name
                        st.session_state.data_loaded = True
                        st.rerun()
                    except MatFileError as e:
                        st.error(str(e))

    # Sekcja Wyników — gdy dane zostały wczytane
    else:
        arrays = st.session_state.loaded_arrays
        source_name = st.session_state.source_name

        try:
            trials = list_trials(arrays)
        except MatFileError as e:
            st.error(str(e))
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            if st.button("Wróć do wyboru pliku", key="err_back_btn"):
                reset_loaded_state()
            return

        trial_label = (
            st.selectbox("Wybierz próbę do analizy:", list(trials))
            if len(trials) > 1 else next(iter(trials))
        )
        key, fmt = trials[trial_label]
        model_data = arrays[key]
        display_data = model_data
        if fmt == "signal":
            model_data = normalize_signal_shape(model_data)
            display_data = model_data
            total_sec = model_data.shape[1] / SAMPLING_RATE
            if total_sec > 4.0:
                with st.expander("Wybierz fragment do wizualizacji i eksportu", expanded=False):
                    st.caption(
                        "Klasyfikacja zawsze wykorzystuje całą próbę, zgodnie ze sposobem "
                        "przygotowania danych treningowych."
                    )
                    start_sec, end_sec = st.slider(
                        "Zakres czasu [s]", 0.0, float(total_sec),
                        (0.0, float(total_sec)), step=0.5,
                    )
                    if (start_sec, end_sec) != (0.0, float(total_sec)):
                        try:
                            display_data = slice_signal(model_data, start_sec, end_sec)
                        except MatFileError as exc:
                            st.error(str(exc))
                            return
        chips_html = render_tech_chips(fmt, model_data)

        col_hdr_left, col_hdr_right = st.columns([3, 1], gap="medium")
        with col_hdr_left:
            st.markdown(
                f'<div style="margin: 2rem 0 0.5rem 0;">'
                f'<div class="section-title" style="margin: 0;">Raport Analizy Afektywnej</div>'
                f'<div style="font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.3rem;">'
                f'Plik: <b>{escape(source_name)}</b>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_hdr_right:
            st.markdown("<div style='margin-top: 2rem; display: flex; justify-content: flex-end;'>", unsafe_allow_html=True)
            if st.button("← WGRAJ INNY PLIK", key="top_reset_btn"):
                reset_loaded_state()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(chips_html, unsafe_allow_html=True)

        try:
            vector, features = prepare_feature_vector(model_data, fmt)
        except MatFileError as e:
            st.error(str(e))
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            if st.button("Wróć do wyboru pliku", key="prep_err_btn"):
                reset_loaded_state()
            return

        try:
            active_artifact = get_artifact_for_format(fmt)
            prediction = predict(active_artifact, vector)
        except ValueError as exc:
            st.error(str(exc))
            return

        show_prediction_results(prediction)

        # Generowanie raportu PDF w pamięci RAM
        pdf_bytes = generate_pdf_report(
            source_name=source_name,
            fmt=fmt,
            raw_array=display_data if fmt == "signal" else None,
            features=features,
            proba=prediction.probabilities,
            classes=prediction.classes,
            model_name=active_artifact.get("model_name", "nieznany"),
            selection_scheme=active_artifact.get("selection_scheme", "nieznany"),
            selection_metric=active_artifact.get("selection_metric", "nieznana"),
            selection_score=active_artifact.get("selection_score"),
        )
        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"Raport_SEED_VII_{Path(source_name).stem}_{timestamp_slug}.pdf"

        col_pdf, col_mat = st.columns(2)
        with col_pdf:
            st.download_button(
                label="POBIERZ RAPORT PDF (A4)",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                width="stretch",
            )
        with col_mat:
            st.download_button(
                label="POBIERZ ANALIZOWANĄ PRÓBKĘ (.MAT)",
                data=export_to_mat_bytes(display_data, key_name=key),
                file_name=f"{Path(source_name).stem}_{key}_sample.mat",
                mime="application/octet-stream",
                width="stretch",
            )

        # Wizualizacje falowe i spektrogram
        show_signal_visualizations(features, display_data, fmt)

        # Sekcja na samym dole: wczytanie kolejnego pliku
        st.markdown("<div class='reset-btn-container'>", unsafe_allow_html=True)
        if st.button("Wgraj kolejny plik", width="stretch", key="bottom_reset_btn"):
            reset_loaded_state()
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
