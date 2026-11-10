import importlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

from .config import FEET, HANDS, LABELS, TEST_RUN, TRAIN_RUNS, Settings, feature_names
from .data import load_run
from .signal import training_windows


def get_implementation(solution: bool):
    return importlib.import_module("eeg_drone.solution" if solution else "eeg_drone.student")


def train(subject: int, data_dir: Path, output_dir: Path, solution: bool):
    settings = Settings()
    implementation = get_implementation(solution)
    X_parts, y_parts = [], []
    for run_id in TRAIN_RUNS:
        run = load_run(subject, run_id, data_dir, settings)
        X, y, _ = training_windows(run, settings, implementation.extract_features)
        X_parts.append(X)
        y_parts.append(y)
    X_train, y_train = np.concatenate(X_parts), np.concatenate(y_parts)
    model = implementation.fit_classifier(X_train, y_train)
    test_run = load_run(subject, TEST_RUN, data_dir, settings)
    X_test, y_test, trial_ids = training_windows(test_run, settings, implementation.extract_features)
    predicted = model.predict(X_test)
    p_hands = model.predict_proba(X_test)[:, list(model.classes_).index(HANDS)]
    # Trial aggregation is an extra view, not an independent replication.
    trial_truth, trial_predictions = [], []
    for trial_id in np.unique(trial_ids):
        mask = trial_ids == trial_id
        trial_truth.append(int(y_test[mask][0]))
        trial_predictions.append(int(p_hands[mask].mean() >= 0.5))
    metrics = {
        "subject": subject, "train_runs": list(TRAIN_RUNS), "test_run": TEST_RUN,
        "implementation": "solution" if solution else "student",
        "n_train_windows": len(X_train), "n_test_windows": len(X_test),
        "n_test_trials": len(trial_truth), "feature_names": feature_names(settings),
        "window_accuracy": float(accuracy_score(y_test, predicted)),
        "window_balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "trial_accuracy": float(accuracy_score(trial_truth, trial_predictions)),
        "confusion_matrix_feet_hands": confusion_matrix(y_test, predicted, labels=[FEET, HANDS]).tolist(),
        "classification_report": classification_report(
            y_test, predicted, labels=[FEET, HANDS], target_names=[LABELS[FEET], LABELS[HANDS]],
            output_dict=True, zero_division=0,
        ),
        "majority_class_baseline": float(max(np.mean(y_test == FEET), np.mean(y_test == HANDS))),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema": 1, "model": model, "settings": settings.to_dict(),
        "subject": subject, "sfreq": test_run.sfreq,
        "implementation": metrics["implementation"], "train_runs": TRAIN_RUNS, "test_run": TEST_RUN,
    }
    joblib.dump(bundle, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _plot_report(metrics, model, X_test, y_test, output_dir)
    return metrics


def _plot_report(metrics, model, X_test, y_test, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    matrix = np.asarray(metrics["confusion_matrix_feet_hands"])
    axes[0].imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        axes[0].text(col, row, str(value), ha="center", va="center")
    axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Feet", "Hands"],
                yticklabels=["Feet", "Hands"], xlabel="Predicted", ylabel="Actual",
                title=f"Held-out run 14: {metrics['window_accuracy']:.1%} accuracy")
    names = metrics["feature_names"]
    # Inspect coefficients when students used the specified sklearn pipeline.
    if hasattr(model, "steps") and hasattr(model.steps[-1][1], "coef_"):
        weights = model.steps[-1][1].coef_[0]
        axes[1].barh(names, weights, color=["#800028" if w > 0 else "#487a9b" for w in weights])
        axes[1].axvline(0, color="black", linewidth=0.6)
        axes[1].set(title="Logistic regression weights", xlabel="Standardized coefficient (+ favors hands)")
    else:
        axes[1].text(0.5, 0.5, "Inspect your model's feature weights.", ha="center")
    fig.savefig(output_dir / "evaluation.png", dpi=160)
    plt.close(fig)


def load_bundle(path: Path):
    # Joblib/pickle can execute code. Load only artifacts you created or trust.
    bundle = joblib.load(path)
    if bundle.get("schema") != 1:
        raise ValueError("Unsupported model artifact; retrain with this repository.")
    return bundle, Settings(**bundle["settings"])
