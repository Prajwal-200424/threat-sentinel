import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "training" / "models"

CNN_LSTM_PATH = MODEL_DIR / "cnn_lstm.h5"
SCALER_PATH = MODEL_DIR / "scaler.pkl"


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

print("=" * 60)
print("LOADING IOT MALWARE DETECTION MODEL")
print("=" * 60)

print(f"Model:  {CNN_LSTM_PATH}")
print(f"Scaler: {SCALER_PATH}")


if not CNN_LSTM_PATH.exists():
    raise FileNotFoundError(
        f"CNN-LSTM model not found: {CNN_LSTM_PATH}"
    )


if not SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Scaler not found: {SCALER_PATH}"
    )


# Load trained CNN-LSTM
model = tf.keras.models.load_model(
    CNN_LSTM_PATH,
    compile=False,
)

# Load StandardScaler
with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)


print("CNN-LSTM loaded successfully.")
print("Scaler loaded successfully.")

print(
    f"Expected feature count: "
    f"{scaler.n_features_in_}"
)

print(
    f"Model input shape: "
    f"{model.input_shape}"
)

print("=" * 60)


# ---------------------------------------------------------
# SINGLE RECORD PREDICTION
# ---------------------------------------------------------

def predict(features: np.ndarray) -> dict:
    """
    Predict whether a network behaviour sample is benign
    or malicious.

    Expected input:
        features shape = (115,)

    CNN-LSTM requires:
        (1, 10, 115)

    Therefore this function expects a single sequence
    containing 10 records.
    """

    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 3:
        raise ValueError(
            "Expected input shape (batch, 10, 115). "
            f"Received: {features.shape}"
        )

    if features.shape[1] != 10:
        raise ValueError(
            "CNN-LSTM requires a sequence length of 10. "
            f"Received: {features.shape[1]}"
        )

    if features.shape[2] != scaler.n_features_in_:
        raise ValueError(
            f"Expected {scaler.n_features_in_} features. "
            f"Received: {features.shape[2]}"
        )

    # -----------------------------------------------------
    # STANDARDIZE
    # -----------------------------------------------------

    batch_size, sequence_length, feature_count = features.shape

    flat_features = features.reshape(
        -1,
        feature_count,
    )

    scaled_features = scaler.transform(
        flat_features
    )

    scaled_features = scaled_features.reshape(
        batch_size,
        sequence_length,
        feature_count,
    )

    # -----------------------------------------------------
    # CNN-LSTM INFERENCE
    # -----------------------------------------------------

    probability = model.predict(
        scaled_features,
        verbose=0,
    )

    attack_probability = float(
        np.asarray(probability).reshape(-1)[0]
    )

    attack_probability = max(
        0.0,
        min(1.0, attack_probability),
    )

    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    if attack_probability >= 0.5:

        prediction = "ATTACK"

        confidence = attack_probability

    else:

        prediction = "BENIGN"

        confidence = 1.0 - attack_probability

    # -----------------------------------------------------
    # RISK LEVEL
    # -----------------------------------------------------

    if attack_probability >= 0.80:

        risk = "CRITICAL"

    elif attack_probability >= 0.60:

        risk = "HIGH"

    elif attack_probability >= 0.30:

        risk = "MEDIUM"

    else:

        risk = "LOW"

    return {
        "prediction": prediction,
        "attack_probability": round(
            attack_probability,
            6,
        ),
        "confidence_score": round(
            confidence,
            6,
        ),
        "risk_level": risk,
        "model_name": "CNN-LSTM",
    }


# ---------------------------------------------------------
# TEST FUNCTION
# ---------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    DATA_DIR = Path(
        r"C:\Users\Tejar\Downloads\archive"
    )

    print("\n" + "=" * 60)
    print("REAL N-BAIoT MODEL VALIDATION")
    print("=" * 60)

    # -----------------------------------------------------
    # Find one benign and one attack CSV
    # -----------------------------------------------------

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    benign_file = None
    attack_file = None

    for csv_file in csv_files:
        filename = csv_file.name.lower()

        if "benign" in filename and benign_file is None:
            benign_file = csv_file

        if (
            ("mirai" in filename or "gafgyt" in filename)
            and attack_file is None
        ):
            attack_file = csv_file

    if benign_file is None:
        raise FileNotFoundError(
            "Could not find a benign N-BaIoT CSV."
        )

    if attack_file is None:
        raise FileNotFoundError(
            "Could not find a Mirai/Gafgyt attack CSV."
        )

    print(f"\nBenign file:")
    print(benign_file.name)

    print(f"\nAttack file:")
    print(attack_file.name)

    # -----------------------------------------------------
    # Helper for loading a CSV
    # -----------------------------------------------------

    def load_sequence(csv_path: Path):
        df = pd.read_csv(csv_path)

        # Remove any label column if present
        if "label" in df.columns:
            df = df.drop("label", axis=1)

        # Take first 10 records
        sample = df.iloc[:10].values

        if sample.shape[0] < 10:
            raise ValueError(
                f"{csv_path.name} has fewer than 10 records."
            )

        if sample.shape[1] != scaler.n_features_in_:
            raise ValueError(
                f"{csv_path.name} has {sample.shape[1]} "
                f"features, expected "
                f"{scaler.n_features_in_}."
            )

        return sample.astype(np.float32)

    # -----------------------------------------------------
    # BENIGN TEST
    # -----------------------------------------------------

    print("\n" + "-" * 60)
    print("TEST 1: BENIGN TRAFFIC")
    print("-" * 60)

    benign_features = load_sequence(
        benign_file
    )

    benign_sequence = benign_features.reshape(
        1,
        10,
        scaler.n_features_in_,
    )

    benign_result = predict(
        benign_sequence
    )

    for key, value in benign_result.items():
        print(f"{key}: {value}")

    # -----------------------------------------------------
    # ATTACK TEST
    # -----------------------------------------------------

    print("\n" + "-" * 60)
    print("TEST 2: ATTACK TRAFFIC")
    print("-" * 60)

    attack_features = load_sequence(
        attack_file
    )

    attack_sequence = attack_features.reshape(
        1,
        10,
        scaler.n_features_in_,
    )

    attack_result = predict(
        attack_sequence
    )

    for key, value in attack_result.items():
        print(f"{key}: {value}")

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(
        f"Benign sample → "
        f"{benign_result['prediction']} "
        f"({benign_result['confidence_score']:.2%})"
    )

    print(
        f"Attack sample → "
        f"{attack_result['prediction']} "
        f"({attack_result['confidence_score']:.2%})"
    )

    print("=" * 60)