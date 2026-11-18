import sys
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Allow importing training/data_prep.py
BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR = BASE_DIR / "training"

sys.path.insert(0, str(TRAINING_DIR))

from data_prep import prepare_data, prepare_sequences


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

DATA_DIR = r"C:\Users\Tejar\Downloads\archive"

MODEL_PATH = (
    TRAINING_DIR / "models" / "cnn_lstm.h5"
)

SCALER_PATH = (
    TRAINING_DIR / "models" / "scaler.pkl"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("=" * 70)
print("CNN-LSTM MODEL EVALUATION")
print("=" * 70)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("CNN-LSTM loaded.")

print("\nLoading scaler...")

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print("Scaler loaded.")

print(f"Expected features: {scaler.n_features_in_}")
print(f"Model input shape: {model.input_shape}")


# ---------------------------------------------------------
# RECREATE TRAINING PREPROCESSING
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RECREATING TRAINING DATA PIPELINE")
print("=" * 70)

data = prepare_data(
    data_dir=DATA_DIR,
    subsample_frac=0.1,
    test_size=0.2,
    random_state=42,
)


# ---------------------------------------------------------
# VERIFY SCALER
# ---------------------------------------------------------

print("\nChecking scaler compatibility...")

print(
    "Saved scaler features:",
    scaler.n_features_in_,
)

print(
    "Generated data features:",
    data["X_test"].shape[1],
)

if (
    scaler.n_features_in_
    != data["X_test"].shape[1]
):
    raise ValueError(
        "Scaler and dataset feature counts do not match."
    )

# IMPORTANT:
# prepare_data() creates a new scaler internally.
# We use the saved scaler that belongs to the trained model.
#
# Transform the raw test data again using the saved scaler.

X_test_raw = data["X_test"]

# data["X_test"] is already standardized by prepare_data(),
# so recover the original scale first using the scaler
# created inside prepare_data().

training_scaler = data["scaler"]

X_test_raw = training_scaler.inverse_transform(
    X_test_raw
)

X_test_scaled = scaler.transform(
    X_test_raw
)


# ---------------------------------------------------------
# CREATE SEQUENCES
# ---------------------------------------------------------

print("\nCreating 10-record sequences...")

X_test_seq = []

window_size = 10

for i in range(
    len(X_test_scaled) - window_size + 1
):
    X_test_seq.append(
        X_test_scaled[
            i : i + window_size
        ]
    )

X_test_seq = np.asarray(
    X_test_seq,
    dtype=np.float32,
)

# Match labels exactly with the sequence creation
y_test_seq = data["y_test"][
    window_size - 1 :
]


print(
    "Sequence shape:",
    X_test_seq.shape,
)

print(
    "Label shape:",
    y_test_seq.shape,
)


# ---------------------------------------------------------
# MODEL PREDICTION
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RUNNING CNN-LSTM PREDICTIONS")
print("=" * 70)

probabilities = model.predict(
    X_test_seq,
    batch_size=256,
    verbose=1,
)

probabilities = np.asarray(
    probabilities
).reshape(-1)


# ---------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------

y_pred = (
    probabilities >= 0.5
).astype(int)


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test_seq,
    y_pred,
)

precision = precision_score(
    y_test_seq,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_test_seq,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_test_seq,
    y_pred,
    zero_division=0,
)

cm = confusion_matrix(
    y_test_seq,
    y_pred,
)


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CNN-LSTM EVALUATION RESULTS")
print("=" * 70)

print(
    f"\nAccuracy  : {accuracy:.4f}"
    f" ({accuracy * 100:.2f}%)"
)

print(
    f"Precision : {precision:.4f}"
    f" ({precision * 100:.2f}%)"
)

print(
    f"Recall    : {recall:.4f}"
    f" ({recall * 100:.2f}%)"
)

print(
    f"F1 Score  : {f1:.4f}"
    f" ({f1 * 100:.2f}%)"
)

print("\nConfusion Matrix:")

print(
    "                 Predicted"
)

print(
    "              Benign  Attack"
)

print(
    f"Actual Benign   "
    f"{cm[0, 0]:7d} "
    f"{cm[0, 1]:7d}"
)

print(
    f"Actual Attack   "
    f"{cm[1, 0]:7d} "
    f"{cm[1, 1]:7d}"
)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)