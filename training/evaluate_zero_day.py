"""
Threat Sentinel - leakage-safe held-out-family evaluation.

The exact live ensemble remains:
    CNN-LSTM      50%
    Random Forest 25%
    Autoencoder   25%

This script evaluates:
1. Binary BENIGN vs ATTACK detection on a family never used for fitting.
2. Potential-zero-day behavior using an AE reconstruction threshold calibrated
   from the KNOWN ATTACK family during training/calibration.

This does not prove detection of every future zero-day attack.
"""

import json
import pickle
import re
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_prep import prepare_data, prepare_sequences

CNN_WEIGHT = 0.50
RF_WEIGHT = 0.25
AE_WEIGHT = 0.25
WINDOW_SIZE = 10
DATA_DIR = r"C:\Users\Tejar\Downloads\archive"
HOLDOUT_FAMILY = "MIRAI"
MODEL_DIR = Path("models")

# The .keras/.h5 artifacts in models/ were serialized by Keras >= 3.15, whose
# VarianceScaling/GlorotUniform config includes `input_axes` and `output_axes`.
# Keras 3.14 and older reject that config with
# "GlorotUniform.__init__() got an unexpected keyword argument 'input_axes'".
# The artifacts must not be retrained or re-exported to work around this; the
# evaluation just needs an interpreter whose Keras can read them.
MINIMUM_ARTIFACT_KERAS = (3, 15)


def keras_version_tuple(version):
    """Return (major, minor) for a Keras version string, or None."""
    parts = []
    for chunk in str(version).split(".")[:2]:
        match = re.match(r"\d+", chunk)
        if match is None:
            return None
        parts.append(int(match.group()))
    return tuple(parts) if len(parts) == 2 else None


def require_artifact_compatible_keras():
    """Fail early and actionably when Keras cannot read the artifacts."""
    version = getattr(keras, "__version__", "unknown")
    parsed = keras_version_tuple(version)
    if parsed is not None and parsed >= MINIMUM_ARTIFACT_KERAS:
        return version
    raise RuntimeError(
        "Model artifacts were written by Keras >= "
        f"{MINIMUM_ARTIFACT_KERAS[0]}.{MINIMUM_ARTIFACT_KERAS[1]} (their "
        "GlorotUniform config carries input_axes/output_axes), but this "
        f"interpreter has TensorFlow {tf.__version__} with Keras {version}. "
        "Keras 3.14 and older fail with \"GlorotUniform.__init__() got an "
        "unexpected keyword argument 'input_axes'\". Do not retrain or "
        "overwrite the artifacts; rerun this evaluation with the project's "
        "compatible environment:\n"
        "  Windows: .venv\\Scripts\\python.exe training\\evaluate_zero_day.py\n"
        "  POSIX  : ./.venv/bin/python training/evaluate_zero_day.py"
    )


class ZeroDayEvaluator:
    def __init__(self):
        self.models = {}
        self.thresholds = {}

    def load_artifacts(self):
        print("=" * 80)
        print("LOADING THREAT SENTINEL ARTIFACTS")
        print("=" * 80)

        with open("thresholds.json", "r", encoding="utf-8") as f:
            self.thresholds = json.load(f)

        saved_weights = (
            float(self.thresholds["cnn_weight"]),
            float(self.thresholds["random_forest_weight"]),
            float(self.thresholds["autoencoder_weight"]),
        )
        if saved_weights != (CNN_WEIGHT, RF_WEIGHT, AE_WEIGHT):
            raise RuntimeError(
                "Ensemble weights changed. Expected exact 50/25/25, "
                f"got {saved_weights}."
            )

        saved_holdout = self.thresholds.get("holdout_family")
        if saved_holdout != HOLDOUT_FAMILY:
            raise RuntimeError(
                f"Artifact holdout family is {saved_holdout}, expected {HOLDOUT_FAMILY}."
            )

        keras_version = require_artifact_compatible_keras()
        print(f"Keras runtime: {keras_version}")

        rf_path = MODEL_DIR / "random_forest.pkl"
        with open(rf_path, "rb") as f:
            self.models["random_forest"] = pickle.load(f)

        cnn_path = MODEL_DIR / "cnn_lstm.keras"
        ae_path = MODEL_DIR / "autoencoder.keras"

        # Fallback for older artifacts only.
        if not cnn_path.exists():
            cnn_path = MODEL_DIR / "cnn_lstm.h5"
        if not ae_path.exists():
            ae_path = MODEL_DIR / "autoencoder.h5"

        try:
            self.models["cnn_lstm"] = tf.keras.models.load_model(
                str(cnn_path), compile=False
            )
            self.models["autoencoder"] = tf.keras.models.load_model(
                str(ae_path), compile=False
            )
        except (OSError, TypeError, ValueError) as error:
            detail = " ".join(str(error).split())
            raise RuntimeError(
                "Model artifacts could not be loaded by TensorFlow "
                f"{tf.__version__} / Keras {keras_version}. Do not retrain or "
                "overwrite the artifacts; use a compatible TensorFlow/Keras "
                "environment and rerun this evaluation. "
                f"Original error: {type(error).__name__}: {detail[:400]}"
            ) from error

        print("\nLoaded:")
        print(f"  CNN-LSTM: {cnn_path}")
        print(f"  Autoencoder: {ae_path}")
        print(f"  Random Forest: {rf_path}")
        print(f"  Ensemble threshold: {self.thresholds['ensemble_threshold']:.6f}")
        print(
            "  Known-attack novelty threshold: "
            f"{self.thresholds['known_attack_novelty_threshold']:.8f}"
        )

    @staticmethod
    def describe(name, values):
        values = np.asarray(values, dtype=float)
        print(f"\n{name}")
        if len(values) == 0:
            print("  NO DATA")
            return
        for p in [0, 25, 50, 75, 90, 95, 99, 100]:
            print(f"  P{p:03d}: {np.percentile(values, p):.8f}")
        print(f"  Mean: {np.mean(values):.8f}")
        print(f"  Std : {np.std(values):.8f}")

    @staticmethod
    def print_metrics(title, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"F1       : {f1_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"FPR      : {fpr:.4f}")
        print("\nConfusion matrix:")
        print(cm)

    def calculate_scores(self, X_flat, X_seq):
        if len(X_flat) != len(X_seq):
            raise RuntimeError(
                f"Score alignment error: flat={len(X_flat)}, sequence={len(X_seq)}"
            )

        ae = self.models["autoencoder"]
        cnn = self.models["cnn_lstm"]
        rf = self.models["random_forest"]

        reconstruction = ae.predict(X_flat, batch_size=512, verbose=0)
        ae_mse = np.mean(np.square(X_flat - reconstruction), axis=1)

        ae_threshold = float(self.thresholds["autoencoder_threshold"])
        ae_score = np.clip(
            (ae_mse / max(ae_threshold, 1e-12)) * 100.0,
            0.0,
            100.0,
        )
        ae_anomaly = ae_mse >= ae_threshold

        cnn_probability = cnn.predict(X_seq, batch_size=512, verbose=0).flatten()
        cnn_score = np.clip(cnn_probability * 100.0, 0.0, 100.0)
        cnn_threshold = float(self.thresholds["cnn_lstm_threshold"])
        cnn_anomaly = cnn_probability >= cnn_threshold

        rf_probability = rf.predict_proba(X_flat)[:, 1]
        rf_score = np.clip(rf_probability * 100.0, 0.0, 100.0)
        rf_threshold = float(self.thresholds["random_forest_confidence_threshold"])
        rf_attack = rf_probability >= rf_threshold
        rf_unknown = rf_probability < rf_threshold

        # EXACT 50 / 25 / 25 FUSION.
        final_score = np.clip(
            CNN_WEIGHT * cnn_score
            + RF_WEIGHT * rf_score
            + AE_WEIGHT * ae_score,
            0.0,
            100.0,
        )

        return {
            "ae_mse": ae_mse,
            "ae_score": ae_score,
            "ae_anomaly": ae_anomaly,
            "cnn_probability": cnn_probability,
            "cnn_score": cnn_score,
            "cnn_anomaly": cnn_anomaly,
            "rf_probability": rf_probability,
            "rf_score": rf_score,
            "rf_attack": rf_attack,
            "rf_unknown": rf_unknown,
            "final_score": final_score,
        }

    def classify(self, scores):
        ensemble_threshold = float(self.thresholds["ensemble_threshold"])
        novelty_threshold = float(
            self.thresholds["known_attack_novelty_threshold"]
        )

        attack = scores["final_score"] >= ensemble_threshold

        classification = np.full(
            len(attack), "BENIGN", dtype=object
        )

        # Novelty is evaluated only AFTER the sample has been classified as
        # malicious. A potential zero-day must have reconstruction error beyond
        # the 99th percentile of the known-attack calibration distribution.
        potential_zero_day = attack & (
            scores["ae_mse"] >= novelty_threshold
        )

        classification[attack & ~potential_zero_day] = "KNOWN_ATTACK"
        classification[potential_zero_day] = "POTENTIAL_ZERO_DAY"

        return classification

    def run(self):
        self.load_artifacts()

        print("\n" + "=" * 80)
        print("PREPARING HELD-OUT-FAMILY TEST DATA")
        print("=" * 80)

        data = prepare_data(
            DATA_DIR,
            subsample_frac=float(self.thresholds["subsample_frac"]),
            holdout_family=HOLDOUT_FAMILY,
            random_state=int(self.thresholds["seed"]),
        )
        seq = prepare_sequences(data, WINDOW_SIZE)

        # Sequence target_indices_test refer directly to data["X_test"].
        target_idx = seq["target_indices_test"]
        X_flat = data["X_test"][target_idx]
        y_true = data["y_test"][target_idx]
        family_true = data["family_test"][target_idx]
        X_seq = seq["X_test"]

        if not np.array_equal(y_true, seq["y_test"]):
            raise RuntimeError("TEST sequence/flat labels are not aligned")
        if not np.array_equal(family_true, seq["family_test"]):
            raise RuntimeError("TEST sequence/flat families are not aligned")

        allowed = {"BENIGN", HOLDOUT_FAMILY}
        if not set(family_true).issubset(allowed):
            raise RuntimeError(
                f"Unexpected test family values: {set(family_true)}"
            )

        print(f"\nAligned evaluation samples: {len(X_flat)}")
        print("Evaluation families:", dict(zip(*np.unique(family_true, return_counts=True))))

        scores = self.calculate_scores(X_flat, X_seq)
        classification = self.classify(scores)

        y_pred = (classification != "BENIGN").astype(int)

        self.print_metrics("PRIMARY ATTACK DETECTION", y_true, y_pred)

        # --------------------------------------------------------------
        # HELD-OUT FAMILY / NOVELTY
        # --------------------------------------------------------------
        attack_mask = y_true == 1
        benign_mask = y_true == 0

        zero_day_mask = classification == "POTENTIAL_ZERO_DAY"
        known_mask = classification == "KNOWN_ATTACK"

        attack_count = int(np.sum(attack_mask))
        detected_attack = int(np.sum(y_pred[attack_mask] == 1))
        zero_day_count = int(np.sum(zero_day_mask & attack_mask))
        known_count = int(np.sum(known_mask & attack_mask))
        false_alarms = int(np.sum(y_pred[benign_mask] == 1))
        benign_count = int(np.sum(benign_mask))

        print("\n" + "=" * 80)
        print("HELD-OUT FAMILY / NOVELTY ANALYSIS")
        print("=" * 80)
        print(f"Held-out family          : {HOLDOUT_FAMILY}")
        print(f"Held-out attack samples  : {attack_count}")
        print(f"Detected as attack       : {detected_attack}")
        print(f"Potential zero-day       : {zero_day_count}")
        print(f"Known attack             : {known_count}")
        print(
            f"Potential zero-day rate  : "
            f"{zero_day_count / attack_count:.4f}"
            if attack_count else "Potential zero-day rate  : N/A"
        )
        print(f"\nBenign samples           : {benign_count}")
        print(f"False alarms             : {false_alarms}")
        print(
            f"Benign detection rate    : "
            f"{1 - false_alarms / benign_count:.4f}"
            if benign_count else "Benign detection rate    : N/A"
        )

        # Report novelty gate behavior on the actual held-out attack family.
        novelty_threshold = self.thresholds["known_attack_novelty_threshold"]
        attack_ae = scores["ae_mse"][attack_mask]
        print(
            "\nKnown-attack novelty threshold (calibration P99): "
            f"{novelty_threshold:.8f}"
        )
        print(
            "Held-out attacks above novelty threshold: "
            f"{np.sum(attack_ae >= novelty_threshold)} / {len(attack_ae)}"
        )

        try:
            auc = roc_auc_score(y_true, scores["final_score"])
            print(f"\nROC-AUC                  : {auc:.4f}")
        except ValueError:
            print("\nROC-AUC unavailable.")

        self.describe(
            "FINAL SCORE - BENIGN",
            scores["final_score"][benign_mask],
        )
        self.describe(
            "FINAL SCORE - HELD-OUT ATTACK",
            scores["final_score"][attack_mask],
        )
        self.describe(
            "CNN PROBABILITY - BENIGN",
            scores["cnn_probability"][benign_mask],
        )
        self.describe(
            "CNN PROBABILITY - HELD-OUT ATTACK",
            scores["cnn_probability"][attack_mask],
        )
        self.describe(
            "RF PROBABILITY - BENIGN",
            scores["rf_probability"][benign_mask],
        )
        self.describe(
            "RF PROBABILITY - HELD-OUT ATTACK",
            scores["rf_probability"][attack_mask],
        )
        self.describe(
            "AE RECONSTRUCTION ERROR - BENIGN",
            scores["ae_mse"][benign_mask],
        )
        self.describe(
            "AE RECONSTRUCTION ERROR - HELD-OUT ATTACK",
            scores["ae_mse"][attack_mask],
        )

        print("\n" + "=" * 80)
        print("FINAL ENSEMBLE CONFIGURATION")
        print("=" * 80)
        print("CNN-LSTM      = 50%")
        print("Random Forest = 25%")
        print("Autoencoder   = 25%")
        print(
            f"Ensemble threshold = {self.thresholds['ensemble_threshold']:.6f}"
        )
        print(
            "Novelty threshold = "
            f"{self.thresholds['known_attack_novelty_threshold']:.8f}"
        )
        print("\nEvaluation complete.")


if __name__ == "__main__":
    ZeroDayEvaluator().run()
