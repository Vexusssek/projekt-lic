"""
Trening i porównanie modeli ML (SVM, Random Forest, XGBoost) — SEED-VII.

Obsługuje trzy schematy walidacji:
  - subject-dependent: stratyfikowana 5-krotna CV na całym zbiorze
    (dane tej samej osoby mogą trafić do treningu i testu),
  - loso (leave-one-subject-out): każda osoba kolejno jako zbiór testowy
    (ocena generalizacji na nowe osoby).
  - stimulus-independent: pięć podziałów grupowanych numerem bodźca
    (ocena generalizacji na materiały niewidziane podczas treningu).

Wyniki (metryki CSV/JSON, macierze pomyłek PNG) zapisywane są do `reports/`,
a model wybrany według LOSO do `ml/models/best_model.joblib`.

Uruchomienie:
    python -m ml.train                 # wszystkie schematy
    python -m ml.train --scheme loso   # leave-one-subject-out
    python -m ml.train --scheme both   # subject-dependent + LOSO
    python -m ml.train --features own  # trening na cechach z własnego ekstraktora
                                       # (data/dataset_own.npz -> best_model_own_features.joblib)
"""

import argparse
import json
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ml.config import (
    BEST_MODEL_PATH,
    DATASET_PATH,
    EMOTION_MAP,
    EMOTION_NAMES,
    FEATURE_TYPE,
    MODELS_DIR,
    N_BANDS,
    N_CHANNELS,
    N_TRIALS,
    OWN_DATASET_PATH,
    OWN_FEATURE_TYPE,
    OWN_MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
)
from ml.model_artifact import (
    ARTIFACT_VERSION,
    dataset_fingerprint,
    library_versions,
)

# Warianty cech: 'reference' = gotowe de_LDS z EEG_features,
# 'own' = cechy z własnego ekstraktora DE+LDS (dataset_own.npz)
FEATURE_VARIANTS = {
    "reference": {"dataset": DATASET_PATH, "model": BEST_MODEL_PATH,
                  "feature_type": FEATURE_TYPE, "tag": ""},
    "own": {"dataset": OWN_DATASET_PATH, "model": OWN_MODEL_PATH,
            "feature_type": OWN_FEATURE_TYPE, "tag": "_own"},
}

N_FOLDS = 5
BOOTSTRAP_ITERATIONS = 10_000
PRIMARY_METRIC = "f1_macro"


def build_models() -> dict:
    """Zwraca klasyfikatory porównywane w walidacji."""
    svm = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=10.0, gamma="scale")),
        ])
    calibration_cv = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    return {
        "SVM": CalibratedClassifierCV(estimator=svm, cv=calibration_cv, method="sigmoid"),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            objective="multi:softprob", eval_metric="mlogloss",
            tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def build_baselines() -> dict:
    """Zwraca punkty odniesienia, które nie uczestniczą w wyborze modelu."""
    return {
        "DummyMostFrequent": DummyClassifier(strategy="most_frequent"),
        "DummyStratified": DummyClassifier(
            strategy="stratified", random_state=RANDOM_STATE
        ),
    }


def load_dataset(dataset_path=DATASET_PATH):
    """Wczytuje cechy, etykiety, identyfikatory osób i bodźców."""
    with np.load(dataset_path) as data:
        X, y, subjects = data["X"], data["y"], data["subjects"]
        if "trials" in data.files:
            trials = data["trials"]
        else:
            if len(y) % N_TRIALS:
                raise ValueError("Nie można odtworzyć numerów prób ze starszego zbioru.")
            trials = np.tile(np.arange(1, N_TRIALS + 1), len(y) // N_TRIALS)
    validate_dataset(X, y, subjects, trials)
    return X, y, subjects, trials


def validate_dataset(X, y, subjects, trials) -> None:
    """Sprawdza spójność zbioru przed kosztowną walidacją modeli."""
    expected_features = N_BANDS * N_CHANNELS
    if X.ndim != 2 or X.shape[1] != expected_features:
        raise ValueError(f"Oczekiwano X o kształcie (N, {expected_features}), otrzymano {X.shape}.")
    if not (len(X) == len(y) == len(subjects) == len(trials)):
        raise ValueError("X, y, subjects i trials mają różne długości.")
    if not np.isfinite(X).all():
        raise ValueError("Zbiór cech zawiera NaN lub wartości nieskończone.")
    if not np.array_equal(np.unique(y), np.arange(len(EMOTION_MAP))):
        raise ValueError("Zbiór nie zawiera oczekiwanego zestawu siedmiu klas.")
    expected_trials = np.arange(1, N_TRIALS + 1)
    for subject in np.unique(subjects):
        if not np.array_equal(np.sort(trials[subjects == subject]), expected_trials):
            raise ValueError(f"Osoba {subject} nie ma kompletnego zestawu prób.")


def validation_split(scheme: str, subjects, trials):
    """Zwraca obiekt CV, grupy podziału i grupy używane w bootstrapie."""
    if scheme == "subject_dependent":
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        return cv, None, subjects
    if scheme == "loso":
        return LeaveOneGroupOut(), subjects, subjects
    if scheme == "stimulus_independent":
        cv = StratifiedGroupKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
        )
        return cv, trials, trials
    raise ValueError(f"Nieznany schemat walidacji: {scheme}")


def clustered_accuracy_ci(y_true, y_pred, groups) -> tuple[float, float]:
    """Wyznacza 95% bootstrap CI przez losowanie całych osób lub bodźców."""
    scores = np.array([
        accuracy_score(y_true[groups == group], y_pred[groups == group])
        for group in np.unique(groups)
    ])
    rng = np.random.default_rng(RANDOM_STATE)
    sampled = rng.choice(
        scores, size=(BOOTSTRAP_ITERATIONS, len(scores)), replace=True
    )
    return tuple(float(value) for value in np.quantile(sampled.mean(axis=1), (0.025, 0.975)))


def expected_calibration_error(y_true, y_pred, probabilities, bins: int = 10) -> float:
    """Oblicza ECE dla najwyższego wyniku klasyfikatora."""
    top_scores = probabilities.max(axis=1)
    correct = y_true == y_pred
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (top_scores > lower) & (top_scores <= upper)
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - top_scores[mask].mean())
    return float(error)


def save_confusion_matrix(cm: np.ndarray, title: str, path):
    """Rysuje i zapisuje macierz pomyłek (znormalizowaną po wierszach)."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    class_names = [EMOTION_NAMES[i] for i in range(len(EMOTION_MAP))]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Prawda")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_scheme(
    models: dict,
    X,
    y,
    subjects,
    trials,
    scheme: str,
    tag: str = "",
    baseline_names: set[str] | None = None,
) -> pd.DataFrame:
    """Oblicza predykcje out-of-fold i zapisuje kompletną diagnostykę."""
    baseline_names = baseline_names or set()
    classes = np.unique(y)
    cv, cv_groups, bootstrap_groups = validation_split(scheme, subjects, trials)
    results = []
    calibration = []

    for name, model in models.items():
        probabilities = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            groups=cv_groups,
            method="predict_proba",
            n_jobs=1,
        )
        predictions = classes[np.argmax(probabilities, axis=1)]
        accuracy = accuracy_score(y, predictions)
        f1_macro = f1_score(y, predictions, average="macro")
        ci_low, ci_high = clustered_accuracy_ci(y, predictions, bootstrap_groups)
        is_baseline = name in baseline_names
        results.append({
            "model": name,
            "scheme": scheme,
            "is_baseline": is_baseline,
            "accuracy": round(float(accuracy), 4),
            "accuracy_ci_low": round(ci_low, 4),
            "accuracy_ci_high": round(ci_high, 4),
            "f1_macro": round(float(f1_macro), 4),
        })

        top_scores = probabilities.max(axis=1)
        encoded = np.eye(len(classes))[y]
        calibration.append({
            "model": name,
            "scheme": scheme,
            "log_loss": round(float(log_loss(y, probabilities, labels=classes)), 4),
            "brier_multiclass": round(
                float(np.mean(np.sum((probabilities - encoded) ** 2, axis=1))), 4
            ),
            "ece_10_bins": round(
                expected_calibration_error(y, predictions, probabilities), 4
            ),
            "mean_top_score": round(float(top_scores.mean()), 4),
        })

        safe_name = name.lower().replace(" ", "_")
        oof = pd.DataFrame({
            "subject": subjects,
            "trial": trials,
            "y_true": y,
            "y_pred": predictions,
            "top_score": top_scores,
        })
        for column, class_code in enumerate(classes):
            oof[f"score_class_{class_code}"] = probabilities[:, column]
        oof.to_csv(
            REPORTS_DIR / f"oof_{scheme}{tag}_{safe_name}.csv", index=False
        )

        if scheme in {"loso", "stimulus_independent"}:
            grouping = subjects if scheme == "loso" else trials
            label = "subject" if scheme == "loso" else "trial"
            per_group = {
                int(group): accuracy_score(y[grouping == group], predictions[grouping == group])
                for group in np.unique(grouping)
            }
            pd.Series(per_group, name="accuracy").rename_axis(label).to_csv(
                REPORTS_DIR / f"accuracy_by_{label}{tag}_{safe_name}.csv"
            )

        if not is_baseline:
            cm = confusion_matrix(y, predictions, labels=classes)
            save_confusion_matrix(
                cm,
                f"{name} — {scheme}{tag} (accuracy={accuracy:.3f})",
                REPORTS_DIR / f"cm_{scheme}{tag}_{name}.png",
            )
        print(
            f"[{scheme}] {name}: accuracy={accuracy:.4f} "
            f"(95% CI {ci_low:.4f}–{ci_high:.4f}), F1-macro={f1_macro:.4f}",
            flush=True,
        )

    pd.DataFrame(calibration).to_csv(
        REPORTS_DIR / f"calibration_metrics{tag}_{scheme}.csv", index=False
    )
    return pd.DataFrame(results)


def save_best_model(
    models: dict,
    metrics_df: pd.DataFrame,
    X,
    y,
    subjects=None,
    trials=None,
    model_path=BEST_MODEL_PATH,
    feature_type=FEATURE_TYPE,
    criterion: str = "loso",
    selection_metric: str = PRIMARY_METRIC,
):
    """Trenuje na pełnym zbiorze model wybrany w wymaganym schemacie CV."""
    if criterion not in metrics_df["scheme"].values:
        raise ValueError(
            f"Nie zapisano modelu: bieżący przebieg nie zawiera schematu '{criterion}'."
        )
    if selection_metric not in metrics_df.columns:
        raise ValueError(f"Brak metryki wyboru '{selection_metric}'.")

    eligible = metrics_df[
        (metrics_df["scheme"] == criterion) & metrics_df["model"].isin(models)
    ]
    if eligible.empty:
        raise ValueError("Brak modelu kandydującego w wynikach walidacji.")
    best_row = eligible.sort_values(
        [selection_metric, "accuracy", "model"],
        ascending=[False, False, True],
    ).iloc[0]
    best_name = str(best_row["model"])
    best_model = clone(models[best_name]).fit(X, y)

    subjects = np.zeros(len(y), dtype=np.int16) if subjects is None else np.asarray(subjects)
    trials = np.arange(len(y), dtype=np.int32) if trials is None else np.asarray(trials)
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": best_model,
        "model_name": best_name,
        "model_parameters": best_model.get_params(deep=True),
        "selection_scheme": criterion,
        "selection_metric": selection_metric,
        "selection_score": float(best_row[selection_metric]),
        "selection_accuracy": float(best_row["accuracy"]),
        "selection_f1_macro": float(best_row["f1_macro"]),
        "emotion_map": EMOTION_MAP,
        "metrics": metrics_df.to_dict(orient="records"),
        "feature_type": feature_type,
        "dataset_sha256": dataset_fingerprint(X, y, subjects, trials),
        "data_summary": {
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
            "n_subjects": int(len(np.unique(subjects))),
            "n_trials": int(len(np.unique(trials))),
            "class_counts": np.bincount(y).tolist(),
        },
        "library_versions": library_versions(),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    print(
        f"Wybrano {best_name}: {criterion}, {selection_metric}="
        f"{float(best_row[selection_metric]):.4f} -> {model_path}",
        flush=True,
    )
    return artifact


def plot_comparison(metrics_df: pd.DataFrame, tag: str = ""):
    """Wykres słupkowy porównania modeli i schematów walidacji."""
    pivot = metrics_df.pivot(index="model", columns="scheme", values="accuracy")
    ax = pivot.plot.bar(rot=20, figsize=(10, 6))
    ax.set_ylabel("Accuracy")
    ax.set_title("Porównanie modeli i schematów walidacji")
    ax.axhline(1 / 7, color="gray", linestyle="--", label="losowe zgadywanie (1/7)")
    ax.legend()
    ax.figure.tight_layout()
    ax.figure.savefig(REPORTS_DIR / f"model_comparison{tag}.png", dpi=150)
    plt.close(ax.figure)


def plot_loso_per_subject(model_names, tag: str = ""):
    """Zapisuje porównanie accuracy poszczególnych osób w LOSO."""
    frames = {}
    for name in model_names:
        safe_name = name.lower().replace(" ", "_")
        csv_path = REPORTS_DIR / f"accuracy_by_subject{tag}_{safe_name}.csv"
        if csv_path.exists():
            frames[name] = pd.read_csv(csv_path, index_col="subject")["accuracy"]
    if not frames:
        return
    df = pd.DataFrame(frames)
    ax = df.plot.bar(figsize=(12, 5), width=0.8)
    ax.set_xlabel("Osoba")
    ax.set_ylabel("Accuracy")
    ax.set_title("LOSO — accuracy per osoba")
    ax.axhline(1 / 7, color="gray", linestyle="--", label="losowe zgadywanie (1/7)")
    ax.legend()
    ax.figure.tight_layout()
    ax.figure.savefig(REPORTS_DIR / f"loso_accuracy_per_subject{tag}.png", dpi=150)
    plt.close(ax.figure)


def resolve_schemes(value: str) -> list[str]:
    if value == "both":
        return ["subject_dependent", "loso"]
    if value == "all":
        return ["subject_dependent", "loso", "stimulus_independent"]
    return [value]


def main():
    parser = argparse.ArgumentParser(description="Trening modeli rozpoznawania emocji SEED-VII")
    parser.add_argument(
        "--scheme",
        choices=["subject_dependent", "loso", "stimulus_independent", "both", "all"],
        default="all",
        help="Schemat walidacji; 'all' uruchamia wszystkie trzy schematy.",
    )
    parser.add_argument("--features", choices=list(FEATURE_VARIANTS),
                        default="reference",
                        help="reference: cechy de_LDS z EEG_features; "
                             "own: cechy z własnego ekstraktora (dataset_own.npz)")
    parser.add_argument(
        "--selection-metric",
        choices=["accuracy", "f1_macro"],
        default=PRIMARY_METRIC,
        help="Metryka wyboru modelu; domyślnie F1-macro.",
    )
    args = parser.parse_args()
    schemes = resolve_schemes(args.scheme)
    variant = FEATURE_VARIANTS[args.features]
    tag = variant["tag"]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    X, y, subjects, trials = load_dataset(variant["dataset"])
    print(
        f"Dane ({args.features}): X={X.shape}, klasy={np.unique(y)}, "
        f"osoby={len(np.unique(subjects))}, bodźce={len(np.unique(trials))}",
        flush=True,
    )

    models = build_models()
    baselines = build_baselines()
    evaluation_models = {**models, **baselines}
    frames = [
        evaluate_scheme(
            evaluation_models,
            X,
            y,
            subjects,
            trials,
            scheme,
            tag=tag,
            baseline_names=set(baselines),
        )
        for scheme in schemes
    ]
    current_metrics_df = pd.concat(frames, ignore_index=True)
    metrics_df = current_metrics_df

    # Dopisanie do istniejących metryk (jeśli wcześniej liczono inny schemat)
    metrics_path = REPORTS_DIR / f"metrics{tag}.csv"
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        old = old[~old["scheme"].isin(schemes)]
        metrics_df = pd.concat([old, metrics_df], ignore_index=True)
    metrics_df.to_csv(metrics_path, index=False)
    (REPORTS_DIR / f"metrics{tag}.json").write_text(
        json.dumps(json.loads(metrics_df.to_json(orient="records")), indent=2),
        encoding="utf-8",
    )

    plot_comparison(metrics_df, tag)

    if "loso" in schemes:
        plot_loso_per_subject(models, tag)

    if "loso" in schemes:
        save_best_model(
            models,
            current_metrics_df,
            X,
            y,
            subjects,
            trials,
            model_path=variant["model"],
            feature_type=variant["feature_type"],
            criterion="loso",
            selection_metric=args.selection_metric,
        )
    else:
        print("Nie zapisano modelu: bieżący przebieg nie obejmował walidacji LOSO.")

    print(f"\nMetryki zapisane w {REPORTS_DIR}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
