"""Generowanie dwustronicowego raportu PDF z analizy EEG."""

import io
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ml.config import (
    BAND_NAMES,
    EMOTION_NAMES,
    EMOTION_POLISH,
    PLOT_CHANNELS,
    SAMPLING_RATE,
    SEED_VII_CHANNELS,
)



def generate_pdf_report(
    source_name: str,
    fmt: str,
    raw_array: np.ndarray,
    features: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    model_name: str = "nieznany",
    selection_scheme: str = "nieznany",
    selection_metric: str = "nieznana",
    selection_score: float | None = None,
) -> bytes:
    """Zwraca raport A4 w postaci bajtów gotowych do pobrania."""
    buf = io.BytesIO()
    sorted_indices = np.argsort(proba)[::-1]
    best_idx = sorted_indices[0]
    best_class = classes[best_idx]
    best_name = EMOTION_NAMES[int(best_class)]
    best_pl = EMOTION_POLISH.get(best_name, {}).get("pl", best_name)
    best_conf = proba[best_idx]


    with PdfPages(buf) as pdf:
        # Strona 1: podsumowanie predykcji, rozkład klas i parametry.
        fig1 = plt.figure(figsize=(8.27, 11.69), facecolor="#ffffff")

        # Nagłówek
        fig1.text(0.08, 0.945, "RAPORT ANALIZY AFEKTYWNEJ EEG", fontsize=18, fontweight="bold", color="#18181b")
        fig1.text(0.08, 0.925, "Klasyfikacja stanów emocjonalnych na podstawie bazy SEED-VII (62 kanały)", fontsize=9.5, color="#71717a")

        # Linia pod nagłówkiem
        line_ax = fig1.add_axes([0.08, 0.91, 0.84, 0.002])
        line_ax.axhline(0, color="#e4e4e7", lw=1.5)
        line_ax.axis("off")

        # Metadane pliku (2 kolumny)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta_text_left = (
            f"Plik źródłowy: {source_name}\n"
            f"Data wygenerowania: {now_str}\n"
            f"Interpretowany typ: {'Sygnał EEG preprocessed' if fmt == 'signal' else 'Cechy Differential Entropy (de_LDS)'}"
        )
        meta_text_right = (
            f"Oczekiwany montaż: 62 kanały (10-20 Neuroscan)\n"
            f"Założone próbkowanie: {SAMPLING_RATE} Hz\n"
            f"Wymagane pasmo: 0.3–50.0 Hz"
        )
        fig1.text(0.08, 0.845, meta_text_left, fontsize=8.5, color="#3f3f46", linespacing=1.4)
        fig1.text(0.52, 0.845, meta_text_right, fontsize=8.5, color="#3f3f46", linespacing=1.4)

        # Ramka głównego KPI
        kpi_rect = plt.Rectangle((0.08, 0.72), 0.84, 0.10, facecolor="#f8fafc", edgecolor="#e2e8f0", lw=1.2, transform=fig1.transFigure)
        fig1.patches.append(kpi_rect)

        fig1.text(0.11, 0.785, "NAJWYŻEJ OCENIONA KLASA", fontsize=8.5, fontweight="bold", color="#64748b")
        fig1.text(0.11, 0.74, f"{best_pl}", fontsize=22, fontweight="bold", color="#0f172a")
        fig1.text(0.38, 0.742, f"—  {best_conf:.1%}", fontsize=18, fontweight="bold", color="#0f172a")

        # Sekcja rozkładu wyników modelu.
        fig1.text(0.08, 0.675, "Rozkład wyników modelu (7 stanów afektywnych)", fontsize=11, fontweight="bold", color="#18181b")

        # Wykres poziomy słupkowy
        ax_bars = fig1.add_axes([0.18, 0.38, 0.74, 0.27], facecolor="#ffffff")
        labels = [EMOTION_POLISH.get(EMOTION_NAMES[int(classes[i])], {}).get("pl", EMOTION_NAMES[int(classes[i])]) for i in sorted_indices[::-1]]
        values = [proba[i] for i in sorted_indices[::-1]]
        bar_colors = [EMOTION_POLISH.get(EMOTION_NAMES[int(classes[i])], {}).get("color", "#9d9fc4") for i in sorted_indices[::-1]]

        y_pos = np.arange(len(labels))
        bars = ax_bars.barh(y_pos, values, color=bar_colors, height=0.55, edgecolor="none")
        ax_bars.set_yticks(y_pos)
        ax_bars.set_yticklabels(labels, fontsize=9, color="#18181b", fontweight="normal")
        ax_bars.set_xlim(0, max(1.0, max(values) * 1.15))
        ax_bars.set_xlabel("Wynik modelu", fontsize=8, color="#71717a")
        ax_bars.tick_params(colors="#71717a", labelsize=8)
        ax_bars.spines["top"].set_visible(False)
        ax_bars.spines["right"].set_visible(False)
        ax_bars.spines["left"].set_color("#e4e4e7")
        ax_bars.spines["bottom"].set_color("#e4e4e7")
        ax_bars.grid(axis="x", color="#f4f4f5", linestyle="--")

        for bar, val in zip(bars, values):
            ax_bars.text(
                bar.get_width() + 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="#18181b",
            )

        # Tabela numeryczna pod wykresem
        table_data = [
            ["Emocja", "Kod klasy", "Wynik modelu", "Status"],
        ]
        for idx in sorted_indices:
            c = int(classes[idx])
            name_en = EMOTION_NAMES[c]
            name_pl = EMOTION_POLISH.get(name_en, {}).get("pl", name_en)
            p = proba[idx]
            status = "Dominująca" if idx == best_idx else "Alternatywna"
            table_data.append([name_pl, f"Klasa {c}", f"{p:.2%}", status])

        ax_table = fig1.add_axes([0.08, 0.10, 0.84, 0.22])
        ax_table.axis("off")
        tbl = ax_table.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            loc="center",
            cellLoc="left",
            colWidths=[0.35, 0.20, 0.25, 0.20],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.3)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#e4e4e7")
            if r == 0:
                cell.set_facecolor("#f4f4f5")
                cell.set_text_props(weight="bold", color="#18181b")
            else:
                cell.set_facecolor("#ffffff" if r % 2 == 1 else "#fafafa")
                cell.set_text_props(color="#27272a")

        # Stopka strony 1
        footer_model = f"Model: {model_name} • wybór: {selection_scheme.upper()}"
        if selection_score is not None:
            footer_model += f" ({selection_metric}={selection_score:.1%})"
        fig1.text(0.08, 0.04, f"SEED-VII Affective BCI Platform • {footer_model}", fontsize=7.5, color="#a1a1aa")
        fig1.text(0.92, 0.04, "Strona 1 / 2", fontsize=7.5, color="#a1a1aa", ha="right")
        pdf.savefig(fig1)
        plt.close(fig1)

        # Strona 2: heatmapa cech DE i przebiegi EEG.
        fig2 = plt.figure(figsize=(8.27, 11.69), facecolor="#ffffff")

        # Nagłówek strony 2
        fig2.text(0.08, 0.945, "WIZUALIZACJA CECH I SYGNAŁU EEG", fontsize=16, fontweight="bold", color="#18181b")
        fig2.text(0.08, 0.925, "Rozkład widmowy Differential Entropy (DE) oraz zapis falowy wybranych odprowadzeń", fontsize=9.5, color="#71717a")

        line_ax2 = fig2.add_axes([0.08, 0.91, 0.84, 0.002])
        line_ax2.axhline(0, color="#e4e4e7", lw=1.5)
        line_ax2.axis("off")

        # Mapa cieplna DE
        fig2.text(0.08, 0.87, "Mapa cieplna cech Differential Entropy (5 pasm częstotliwościowych × 62 kanały)", fontsize=10.5, fontweight="bold", color="#18181b")
        ax_heat = fig2.add_axes([0.08, 0.65, 0.84, 0.20], facecolor="#ffffff")
        mean_features = features.mean(axis=0)  # (5, 62)
        im = ax_heat.imshow(mean_features, aspect="auto", cmap="viridis", interpolation="nearest")
        ax_heat.set_yticks(range(len(BAND_NAMES)))
        ax_heat.set_yticklabels([f"{b.upper()} ({BAND_NAMES[i]})" for i, b in enumerate(["δ", "θ", "α", "β", "γ"])], fontsize=8, color="#27272a")
        ax_heat.set_xlabel("Numer kanału EEG (1–62, układ 10-20)", fontsize=8, color="#71717a", labelpad=4)
        ax_heat.tick_params(colors="#71717a", labelsize=8)
        for s in ax_heat.spines.values():
            s.set_color("#e4e4e7")
        cbar = fig2.colorbar(im, ax=ax_heat, pad=0.02)
        cbar.set_label("Wartość cechy DE", color="#71717a", fontsize=7.5)
        cbar.ax.yaxis.set_tick_params(color="#71717a", labelcolor="#71717a", labelsize=7)
        cbar.outline.set_edgecolor("#e4e4e7")

        # Wykres sygnału falowego (jeśli fmt == 'signal')
        if fmt == "signal" and raw_array is not None:
            fig2.text(0.08, 0.59, "Przebieg czasowy potencjałów bioelektrycznych (okno 10 s)", fontsize=10.5, fontweight="bold", color="#18181b")
            ax_sig = fig2.add_axes([0.08, 0.32, 0.84, 0.24], facecolor="#ffffff")
            n_show = min(raw_array.shape[1], 10 * SAMPLING_RATE)
            t = np.arange(n_show) / SAMPLING_RATE
            ch_indices = [index for index, _ in PLOT_CHANNELS]
            ch_names = [
                f"{SEED_VII_CHANNELS[index]} ({region})"
                for index, region in PLOT_CHANNELS
            ]
            sig_colors = ["#4338ca", "#b45309", "#047857", "#c2410c", "#1d4ed8"]

            for i, ch in enumerate(ch_indices):
                std_val = np.std(raw_array[ch, :n_show])
                offset = i * 4 * (std_val if std_val > 0 else 1.0)
                ax_sig.plot(t, raw_array[ch, :n_show] + offset, lw=0.8, color=sig_colors[i], label=ch_names[i])

            ax_sig.set_xlabel("Czas rejestracji [s]", color="#71717a", fontsize=8)
            ax_sig.set_yticks([])
            ax_sig.tick_params(colors="#71717a", labelsize=8)
            for s in ax_sig.spines.values():
                s.set_color("#e4e4e7")
            ax_sig.grid(True, color="#f4f4f5", linestyle="--")
            ax_sig.legend(loc="upper right", fontsize=7.5, facecolor="#ffffff", edgecolor="#e4e4e7")
        else:
            # Informacja o trybie samych cech DE
            box_info = plt.Rectangle((0.08, 0.44), 0.84, 0.12, facecolor="#f8fafc", edgecolor="#e2e8f0", lw=1.2, transform=fig2.transFigure)
            fig2.patches.append(box_info)
            fig2.text(0.11, 0.52, "Brak sygnału czasowego w pliku wejściowym", fontsize=9.5, fontweight="bold", color="#334155")
            fig2.text(0.11, 0.47, "Plik wejściowy zawiera gotowe cechy Differential Entropy (de_LDS). Przebieg czasowy fal\nmózgowych generowany jest wyłącznie dla plików sygnałowych z bazy EEG_preprocessed.", fontsize=8, color="#64748b")

        # Notatka naukowa i metodologiczna
        fig2.text(0.08, 0.25, "UWAGI METODOLOGICZNE I INTERPRETACJA NAUKOWA", fontsize=9.5, fontweight="bold", color="#18181b")
        notes_text = (
            f"1. Klasyfikator: {model_name}. Model wybrano według wyniku {selection_scheme.upper()}, który służy do oceny\n"
            "   generalizacji między osobami; nie gwarantuje poprawnego rozpoznania dla konkretnego użytkownika.\n"
            "2. Wygładzanie LDS ogranicza krótkotrwałe zmiany kolejnych wektorów Differential Entropy.\n"
            "3. Aplikacja sprawdza format i sposób przygotowania wejścia; przyjmuje także zgodne dane nowych osób.\n"
            "4. Wartości procentowe są wynikami modelu, a nie prawdopodobieństwem poprawności rozpoznania.\n"
            "5. Klasyfikacja wykorzystuje całą próbę; wybrany zakres czasu dotyczy wykresu i eksportu.\n"
            "6. Raport ma charakter badawczy i analityczny w ramach interfejsu mózg-komputer (Affective BCI)."
        )
        fig2.text(0.08, 0.12, notes_text, fontsize=7.2, color="#4b5563", linespacing=1.4)

        # Stopka strony 2
        fig2.text(0.08, 0.04, "SEED-VII Affective BCI Platform • Wygenerowano automatycznie", fontsize=7.5, color="#a1a1aa")
        fig2.text(0.92, 0.04, "Strona 2 / 2", fontsize=7.5, color="#a1a1aa", ha="right")
        pdf.savefig(fig2)
        plt.close(fig2)

    return buf.getvalue()
