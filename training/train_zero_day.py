"""
Threat Sentinel - leakage-safe zero-day training.

Fixed live ensemble:
    CNN-LSTM      50%
    Random Forest 25%
    Autoencoder   25%

This version uses a true family-held-out split and keeps all metadata aligned.
The held-out family is never used for fitting or calibration.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras
from tensorflow.keras import layers

from data_prep import prepare_data, prepare_sequences

SEED = 42
DATA_DIR = r"C:\Users\Tejar\Downloads\archive"
HOLDOUT_FAMILY = "MIRAI"
SUBSAMPLE_FRAC = 0.20
WINDOW_SIZE = 10
EPOCHS = 50
BATCH_SIZE = 128

CNN_WEIGHT = 0.50
RF_WEIGHT = 0.25
AE_WEIGHT = 0.25

MODEL_DIR = Path("models")

np.random.seed(SEED)
tf.random.set_seed(SEED)


def metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "confusion_matrix": cm.tolist(),
    }


def print_metrics(title, y_true, y_pred):
    m = metrics(y_true, y_pred)
    print(f"\n{title}")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall   : {m['recall']:.4f}")
    print(f"  F1       : {m['f1']:.4f}")
    print(f"  FPR      : {m['fpr']:.4f}")
    print("  Confusion matrix:")
    print(np.asarray(m["confusion_matrix"]))


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


class ZeroDayTrainer:
    def __init__(self):
        print("=" * 80)
        print("THREAT SENTINEL - LEAKAGE-SAFE ZERO-DAY TRAINING")
        print("=" * 80)

        self.data = prepare_data(
            DATA_DIR,
            subsample_frac=SUBSAMPLE_FRAC,
            holdout_family=HOLDOUT_FAMILY,
            random_state=SEED,
        )
        self.seq = prepare_sequences(self.data, WINDOW_SIZE)
        self.models = {}
        self.thresholds = {}

        self._verify_split()

    def _verify_split(self):
        train_families = set(self.data["family_train"].tolist())
        cal_families = set(self.data["family_cal"].tolist())
        test_families = set(self.data["family_test"].tolist())

        if HOLDOUT_FAMILY in train_families or HOLDOUT_FAMILY in cal_families:
            raise RuntimeError("Held-out family leaked into training/calibration")
        if not test_families.issubset({"BENIGN", HOLDOUT_FAMILY}):
            raise RuntimeError(f"Unexpected test families: {test_families}")

        print("\nSplit verification: PASSED")
        print("  Train families:", sorted(train_families))
        print("  Calibration families:", sorted(cal_families))
        print("  Test families:", sorted(test_families))

    # ------------------------------------------------------------------
    # AUTOENCODER
    # ------------------------------------------------------------------
    def train_autoencoder(self):
        print("\n" + "=" * 80)
        print("TRAINING AUTOENCODER")
        print("=" * 80)

        X_train = self.data["X_train"]
        y_train = self.data["y_train"]
        X_cal = self.data["X_cal"]
        y_cal = self.data["y_cal"]

        X_benign = X_train[y_train == 0]
        if len(X_benign) < 100:
            raise RuntimeError("Too few benign training samples for autoencoder")

        input_dim = X_train.shape[1]

        model = keras.Sequential(
            [
                layers.Input(shape=(input_dim,)),
                layers.Dense(128, activation="relu"),
                layers.BatchNormalization(),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(input_dim, activation="linear"),
            ],
            name="benign_autoencoder",
        )
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
        ]

        # Chronological calibration benign data acts as validation.
        X_cal_benign = X_cal[y_cal == 0]
        if len(X_cal_benign) == 0:
            raise RuntimeError("No benign calibration samples")

        history = model.fit(
            X_benign,
            X_benign,
            validation_data=(X_cal_benign, X_cal_benign),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            callbacks=callbacks,
            verbose=1,
        )

        self.models["autoencoder"] = model

        cal_reconstruction = model.predict(X_cal, batch_size=512, verbose=0)
        cal_mse = np.mean(np.square(X_cal - cal_reconstruction), axis=1)
        benign_mse = cal_mse[y_cal == 0]
        attack_mse = cal_mse[y_cal == 1]

        ae_threshold = float(np.percentile(benign_mse, 99.0))
        if ae_threshold <= 0:
            raise RuntimeError("Invalid autoencoder threshold")

        self.thresholds["autoencoder_threshold"] = ae_threshold
        # Novelty threshold is calibrated from KNOWN ATTACK traffic only.
        # A held-out family exceeding this level is behaviorally more
        # anomalous than the known attack family used for training.
        if len(attack_mse) == 0:
            raise RuntimeError("No known-attack calibration samples for novelty threshold")
        self.thresholds["known_attack_novelty_threshold"] = float(
            np.percentile(attack_mse, 99.0)
        )
        print(
            "Known-attack novelty threshold (ATTACK P99): "
            f"{self.thresholds["known_attack_novelty_threshold"]:.8f}"
        )

        print(f"\nAutoencoder epochs completed: {len(history.history['loss'])}")
        describe("AE RECONSTRUCTION ERROR - BENIGN CALIBRATION", benign_mse)
        describe("AE RECONSTRUCTION ERROR - KNOWN ATTACK CALIBRATION", attack_mse)
        print(f"\nAutoencoder threshold (BENIGN P99): {ae_threshold:.8f}")

        # This threshold is used only as anomaly evidence, not as the attack
        # classifier by itself.

    # ------------------------------------------------------------------
    # CNN-LSTM
    # ------------------------------------------------------------------
    def train_cnn_lstm(self):
        print("\n" + "=" * 80)
        print("TRAINING CNN-LSTM")
        print("=" * 80)

        X_train = self.seq["X_train"]
        y_train = self.seq["y_train"]
        X_cal = self.seq["X_cal"]
        y_cal = self.seq["y_cal"]

        model = keras.Sequential(
            [
                layers.Input(shape=X_train.shape[1:]),
                layers.Conv1D(64, 3, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv1D(128, 3, padding="same", activation="relu"),
                layers.MaxPooling1D(2),
                layers.LSTM(64, dropout=0.25),
                layers.Dense(32, activation="relu"),
                layers.Dropout(0.30),
                layers.Dense(1, activation="sigmoid"),
            ],
            name="cnn_lstm",
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-4),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
        )

        n0 = int(np.sum(y_train == 0))
        n1 = int(np.sum(y_train == 1))
        total = n0 + n1
        class_weight = {
            0: total / (2.0 * max(n0, 1)),
            1: total / (2.0 * max(n1, 1)),
        }

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_cal, y_cal),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight,
            shuffle=True,
            callbacks=callbacks,
            verbose=1,
        )

        self.models["cnn_lstm"] = model

        proba = model.predict(X_cal, batch_size=512, verbose=0).flatten()
        threshold = self._select_probability_threshold(y_cal, proba, "CNN-LSTM")
        self.thresholds["cnn_lstm_threshold"] = threshold

        print(f"\nCNN-LSTM epochs completed: {len(history.history['loss'])}")
        print_metrics("CNN-LSTM CALIBRATION", y_cal, (proba >= threshold).astype(int))
        describe("CNN PROBABILITY - BENIGN CALIBRATION", proba[y_cal == 0])
        describe("CNN PROBABILITY - KNOWN ATTACK CALIBRATION", proba[y_cal == 1])
        print(f"\nCNN-LSTM threshold: {threshold:.6f}")

    # ------------------------------------------------------------------
    # RANDOM FOREST
    # ------------------------------------------------------------------
    def train_random_forest(self):
        print("\n" + "=" * 80)
        print("TRAINING RANDOM FOREST")
        print("=" * 80)

        rf = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        )
        rf.fit(self.data["X_train"], self.data["y_train"])

        if list(rf.classes_) != [0, 1]:
            raise RuntimeError(f"Unexpected RF classes: {rf.classes_}")

        self.models["random_forest"] = rf

        proba = rf.predict_proba(self.data["X_cal"])[:, 1]
        threshold = self._select_probability_threshold(
            self.data["y_cal"], proba, "Random Forest"
        )
        self.thresholds["random_forest_confidence_threshold"] = threshold

        print_metrics(
            "RANDOM FOREST CALIBRATION",
            self.data["y_cal"],
            (proba >= threshold).astype(int),
        )
        describe("RF PROBABILITY - BENIGN CALIBRATION", proba[self.data["y_cal"] == 0])
        describe("RF PROBABILITY - KNOWN ATTACK CALIBRATION", proba[self.data["y_cal"] == 1])
        print(f"\nRF selected threshold: {threshold:.6f}")

    @staticmethod
    def _select_probability_threshold(y_true, probability, name):
        best = None
        for threshold in np.linspace(0.05, 0.95, 181):
            pred = (probability >= threshold).astype(int)
            cm = confusion_matrix(y_true, pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            f1 = f1_score(y_true, pred, zero_division=0)
            recall = recall_score(y_true, pred, zero_division=0)
            candidate = (fpr <= 0.05, f1, recall, -threshold, threshold)
            if best is None or candidate > best:
                best = candidate

        if best is None:
            raise RuntimeError(f"Could not calibrate {name} threshold")
        return float(best[-1])

    # ------------------------------------------------------------------
    # ENSEMBLE CALIBRATION
    # ------------------------------------------------------------------
    def _scores(self, X_flat, X_seq):
        if len(X_flat) != len(X_seq):
            raise ValueError(
                f"Flat/sequence score length mismatch: {len(X_flat)} vs {len(X_seq)}"
            )

        ae = self.models["autoencoder"]
        cnn = self.models["cnn_lstm"]
        rf = self.models["random_forest"]

        reconstruction = ae.predict(X_flat, batch_size=512, verbose=0)
        ae_mse = np.mean(np.square(X_flat - reconstruction), axis=1)
        ae_threshold = self.thresholds["autoencoder_threshold"]
        ae_score = np.clip((ae_mse / max(ae_threshold, 1e-12)) * 100.0, 0, 100)

        cnn_probability = cnn.predict(X_seq, batch_size=512, verbose=0).flatten()
        cnn_score = np.clip(cnn_probability * 100.0, 0, 100)

        rf_probability = rf.predict_proba(X_flat)[:, 1]
        rf_score = np.clip(rf_probability * 100.0, 0, 100)

        final_score = np.clip(
            CNN_WEIGHT * cnn_score
            + RF_WEIGHT * rf_score
            + AE_WEIGHT * ae_score,
            0,
            100,
        )

        return {
            "ae_mse": ae_mse,
            "ae_score": ae_score,
            "cnn_probability": cnn_probability,
            "cnn_score": cnn_score,
            "rf_probability": rf_probability,
            "rf_score": rf_score,
            "final_score": final_score,
        }

    def calibrate_ensemble(self):
        print("\n" + "=" * 80)
        print("CALIBRATING EXACT 50/25/25 ENSEMBLE")
        print("=" * 80)

        target_idx = self.seq["target_indices_cal"]
        X_flat_cal = self.data["X_cal"][target_idx]
        y_cal = self.data["y_cal"][target_idx]

        if not np.array_equal(y_cal, self.seq["y_cal"]):
            raise RuntimeError("Calibration sequence/flat labels are misaligned")

        scores = self._scores(X_flat_cal, self.seq["X_cal"])

        best = None
        for threshold in np.linspace(1.0, 99.0, 393):
            pred = (scores["final_score"] >= threshold).astype(int)
            cm = confusion_matrix(y_cal, pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            f1 = f1_score(y_cal, pred, zero_division=0)
            recall = recall_score(y_cal, pred, zero_division=0)
            candidate = (fpr <= 0.05, f1, recall, -threshold, threshold)
            if best is None or candidate > best:
                best = candidate

        threshold = float(best[-1])
        self.thresholds["ensemble_threshold"] = threshold

        describe("FINAL SCORE - BENIGN CALIBRATION", scores["final_score"][y_cal == 0])
        describe("FINAL SCORE - KNOWN ATTACK CALIBRATION", scores["final_score"][y_cal == 1])
        print(f"\nSelected ensemble threshold: {threshold:.4f}")
        print_metrics(
            "FINAL ENSEMBLE CALIBRATION",
            y_cal,
            (scores["final_score"] >= threshold).astype(int),
        )

    def save(self):
        print("\n" + "=" * 80)
        print("SAVING ARTIFACTS")
        print("=" * 80)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Native Keras format avoids the legacy HDF5 warning.
        self.models["cnn_lstm"].save(MODEL_DIR / "cnn_lstm.keras")
        self.models["autoencoder"].save(MODEL_DIR / "autoencoder.keras")

        with open(MODEL_DIR / "random_forest.pkl", "wb") as f:
            pickle.dump(self.models["random_forest"], f)

        with open(MODEL_DIR / "scaler.pkl", "wb") as f:
            pickle.dump(self.data["scaler"], f)

        artifact = {
            **self.thresholds,
            "cnn_weight": CNN_WEIGHT,
            "random_forest_weight": RF_WEIGHT,
            "autoencoder_weight": AE_WEIGHT,
            "holdout_family": HOLDOUT_FAMILY,
            "window_size": WINDOW_SIZE,
            "seed": SEED,
            "subsample_frac": SUBSAMPLE_FRAC,
            "method": "family-held-out, stream-safe, leakage-safe",
        }

        with open("thresholds.json", "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)

        print("Saved:")
        print("  models/cnn_lstm.keras")
        print("  models/autoencoder.keras")
        print("  models/random_forest.pkl")
        print("  models/scaler.pkl")
        print("  thresholds.json")

    def run(self):
        self.train_autoencoder()
        self.train_cnn_lstm()
        self.train_random_forest()
        self.calibrate_ensemble()
        self.save()
        print("\n" + "=" * 80)
        print("TRAINING PIPELINE COMPLETED")
        print("=" * 80)


if __name__ == "__main__":
    ZeroDayTrainer().run()
