"""
Threat Sentinel - 3 Model Ensemble Inference

Models:
    1. CNN-LSTM
    2. Random Forest
    3. Autoencoder

CNN-LSTM is the primary temporal malware classifier.
Random Forest provides supervised feature-based evidence.
Autoencoder provides unsupervised anomaly evidence.

Final decision is based ONLY on the weighted ensemble score.

Weights:
    CNN-LSTM      = 50%
    Random Forest = 25%
    Autoencoder   = 25%

Formula:

    Ensemble Score =
        CNN-LSTM Score * 0.50
        + Random Forest Score * 0.25
        + Autoencoder Score * 0.25
"""

import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "training" / "models"

VALIDATED_DIR = (
    BASE_DIR
    / "ml_server"
    / "validated_sequences"
)

CNN_LSTM_PATH = MODEL_DIR / "cnn_lstm.h5"
RANDOM_FOREST_PATH = MODEL_DIR / "random_forest.pkl"
AUTOENCODER_PATH = MODEL_DIR / "autoencoder.h5"
SCALER_PATH = MODEL_DIR / "scaler.pkl"


# =========================================================
# CONFIGURATION
# =========================================================

FEATURE_COUNT = 115
SEQUENCE_LENGTH = 10


# =========================================================
# ENSEMBLE WEIGHTS
# =========================================================

CNN_WEIGHT = 0.50
RF_WEIGHT = 0.25
AUTOENCODER_WEIGHT = 0.25


# =========================================================
# CLASSIFICATION THRESHOLDS
# =========================================================

# 0 - 29.99  -> BENIGN
# 30 - 49.99 -> SUSPICIOUS
# 50 - 100   -> ATTACK

ATTACK_THRESHOLD = 50.0
SUSPICIOUS_THRESHOLD = 30.0


# =========================================================
# RISK THRESHOLDS
# =========================================================

# 0 - 29.99  -> LOW
# 30 - 49.99 -> MEDIUM
# 50 - 84.99 -> HIGH
# 85 - 100   -> CRITICAL

CRITICAL_THRESHOLD = 85.0
HIGH_THRESHOLD = 50.0
MEDIUM_THRESHOLD = 30.0


# =========================================================
# AUTOENCODER THRESHOLD
# =========================================================

# Reconstruction anomaly score >= 30
# is considered anomalous.

AUTOENCODER_ANOMALY_THRESHOLD = 30.0


# =========================================================
# MODEL LOADING
# =========================================================

print("=" * 70)
print("THREAT SENTINEL - 3 MODEL ENSEMBLE")
print("=" * 70)

print()
print("Loading models...")


required_files = [
    CNN_LSTM_PATH,
    RANDOM_FOREST_PATH,
    AUTOENCODER_PATH,
    SCALER_PATH,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ---------------------------------------------------------
# CNN-LSTM
# ---------------------------------------------------------

cnn_lstm = tf.keras.models.load_model(
    CNN_LSTM_PATH,
    compile=False,
)

print("[OK] CNN-LSTM loaded")


# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

with open(
    RANDOM_FOREST_PATH,
    "rb",
) as file:

    random_forest = pickle.load(file)

print("[OK] Random Forest loaded")


# ---------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------

autoencoder = tf.keras.models.load_model(
    AUTOENCODER_PATH,
    compile=False,
)

print("[OK] Autoencoder loaded")


# ---------------------------------------------------------
# Scaler
# ---------------------------------------------------------

with open(
    SCALER_PATH,
    "rb",
) as file:

    scaler = pickle.load(file)

print("[OK] Scaler loaded")


# =========================================================
# MODEL INFORMATION
# =========================================================

print()
print(
    "Expected features:",
    scaler.n_features_in_,
)

print(
    "CNN-LSTM input:",
    cnn_lstm.input_shape,
)

print(
    "Random Forest features:",
    getattr(
        random_forest,
        "n_features_in_",
        "unknown",
    ),
)

print(
    "Autoencoder input:",
    autoencoder.input_shape,
)

print()
print("Ensemble weights:")
print(
    f"  CNN-LSTM      : {CNN_WEIGHT:.0%}"
)

print(
    f"  Random Forest : {RF_WEIGHT:.0%}"
)

print(
    f"  Autoencoder   : {AUTOENCODER_WEIGHT:.0%}"
)

print()
print("Classification thresholds:")
print(
    f"  BENIGN       : < {SUSPICIOUS_THRESHOLD:.0f}%"
)

print(
    f"  SUSPICIOUS   : "
    f"{SUSPICIOUS_THRESHOLD:.0f}% - "
    f"{ATTACK_THRESHOLD - 0.01:.2f}%"
)

print(
    f"  ATTACK       : >= {ATTACK_THRESHOLD:.0f}%"
)

print("=" * 70)


# =========================================================
# AUTOENCODER CALIBRATION
# =========================================================

AUTOENCODER_NORMAL_ERROR = None
AUTOENCODER_ATTACK_ERROR = None


def calibrate_autoencoder(
    benign_sequence: np.ndarray,
    attack_sequence: np.ndarray,
):
    """
    Calibrate the Autoencoder using one validated
    benign sequence and one validated attack sequence.

    The 95th percentile reconstruction error is used
    as the calibration boundary.
    """

    global AUTOENCODER_NORMAL_ERROR
    global AUTOENCODER_ATTACK_ERROR

    benign_sequence = np.asarray(
        benign_sequence,
        dtype=np.float32,
    )

    attack_sequence = np.asarray(
        attack_sequence,
        dtype=np.float32,
    )

    expected_shape = (
        SEQUENCE_LENGTH,
        FEATURE_COUNT,
    )

    if benign_sequence.shape != expected_shape:

        raise ValueError(
            "Invalid benign sequence shape: "
            f"{benign_sequence.shape}. "
            f"Expected {expected_shape}."
        )

    if attack_sequence.shape != expected_shape:

        raise ValueError(
            "Invalid attack sequence shape: "
            f"{attack_sequence.shape}. "
            f"Expected {expected_shape}."
        )

    # Scale the records exactly as the other models use them.
    benign_scaled = scaler.transform(
        benign_sequence
    ).astype(np.float32)

    attack_scaled = scaler.transform(
        attack_sequence
    ).astype(np.float32)

    # Autoencoder works on individual records.
    benign_reconstructed = autoencoder.predict(
        benign_scaled,
        verbose=0,
    )

    attack_reconstructed = autoencoder.predict(
        attack_scaled,
        verbose=0,
    )

    benign_errors = np.mean(
        np.square(
            benign_scaled
            - benign_reconstructed
        ),
        axis=1,
    )

    attack_errors = np.mean(
        np.square(
            attack_scaled
            - attack_reconstructed
        ),
        axis=1,
    )

    AUTOENCODER_NORMAL_ERROR = float(
        np.percentile(
            benign_errors,
            95,
        )
    )

    AUTOENCODER_ATTACK_ERROR = float(
        np.percentile(
            attack_errors,
            95,
        )
    )

    print()
    print("AUTOENCODER CALIBRATION")
    print("-" * 70)

    print(
        "Benign 95th percentile:",
        AUTOENCODER_NORMAL_ERROR,
    )

    print(
        "Attack 95th percentile:",
        AUTOENCODER_ATTACK_ERROR,
    )


# =========================================================
# AUTOENCODER SCORE
# =========================================================

def calculate_autoencoder_score(
    reconstruction_error: float,
) -> float:
    """
    Convert reconstruction error into a 0-100
    anomaly score.

    0   = normal behaviour
    100 = strong anomaly evidence
    """

    normal = AUTOENCODER_NORMAL_ERROR
    attack = AUTOENCODER_ATTACK_ERROR

    if normal is None or attack is None:

        raise RuntimeError(
            "Autoencoder has not been calibrated."
        )

    # Safety condition.
    if attack <= normal:

        if reconstruction_error <= normal:
            return 0.0

        return 100.0

    score = (
        reconstruction_error - normal
    ) / (
        attack - normal
    )

    score = max(
        0.0,
        min(
            1.0,
            score,
        ),
    )

    return score * 100.0


# =========================================================
# PREPARE SEQUENCE
# =========================================================

def prepare_sequence(
    sequence: np.ndarray,
):
    """
    Input:
        (10, 115)

    Returns:
        CNN input -> (1, 10, 115)
        Flat input -> (10, 115)
    """

    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    expected_shape = (
        SEQUENCE_LENGTH,
        FEATURE_COUNT,
    )

    if sequence.shape != expected_shape:

        raise ValueError(
            "Expected sequence shape "
            f"{expected_shape}, "
            f"received {sequence.shape}"
        )

    scaled = scaler.transform(
        sequence
    ).astype(np.float32)

    cnn_input = scaled.reshape(
        1,
        SEQUENCE_LENGTH,
        FEATURE_COUNT,
    )

    records = scaled.reshape(
        SEQUENCE_LENGTH,
        FEATURE_COUNT,
    )

    return cnn_input, records


# =========================================================
# THREE MODEL PREDICTION
# =========================================================

def predict_three_models(
    sequence: np.ndarray,
) -> dict:
    """
    Run CNN-LSTM, Random Forest and Autoencoder.

    The final classification is determined ONLY by
    the weighted ensemble score.

    No model can override the ensemble score.
    """

    cnn_input, records = prepare_sequence(
        sequence
    )


    # =====================================================
    # MODEL 1 — CNN-LSTM
    # =====================================================

    cnn_output = cnn_lstm.predict(
        cnn_input,
        verbose=0,
    )

    cnn_probability = float(
        np.asarray(
            cnn_output
        ).reshape(-1)[0]
    )

    cnn_probability = max(
        0.0,
        min(
            1.0,
            cnn_probability,
        ),
    )

    cnn_score = (
        cnn_probability * 100.0
    )

    cnn_prediction = (
        "ATTACK"
        if cnn_probability >= 0.5
        else "BENIGN"
    )


    # =====================================================
    # MODEL 2 — RANDOM FOREST
    # =====================================================

    rf_probabilities = (
        random_forest.predict_proba(
            records
        )[:, 1]
    )

    rf_probability = float(
        np.mean(
            rf_probabilities
        )
    )

    rf_probability = max(
        0.0,
        min(
            1.0,
            rf_probability,
        ),
    )

    rf_score = (
        rf_probability * 100.0
    )

    rf_prediction = (
        "ATTACK"
        if rf_probability >= 0.5
        else "BENIGN"
    )


    # =====================================================
    # MODEL 3 — AUTOENCODER
    # =====================================================

    reconstructed = autoencoder.predict(
        records,
        verbose=0,
    )

    reconstruction_errors = np.mean(
        np.square(
            records
            - reconstructed
        ),
        axis=1,
    )

    reconstruction_error = float(
        np.mean(
            reconstruction_errors
        )
    )

    autoencoder_score = (
        calculate_autoencoder_score(
            reconstruction_error
        )
    )

    autoencoder_prediction = (
        "ANOMALY"
        if (
            autoencoder_score
            >= AUTOENCODER_ANOMALY_THRESHOLD
        )
        else "NORMAL"
    )


    # =====================================================
    # WEIGHTED ENSEMBLE
    # =====================================================

    ensemble_score = (
        cnn_score * CNN_WEIGHT
        + rf_score * RF_WEIGHT
        + autoencoder_score
        * AUTOENCODER_WEIGHT
    )

    ensemble_score = max(
        0.0,
        min(
            100.0,
            ensemble_score,
        ),
    )


    # =====================================================
    # ATTACK VOTES
    # =====================================================

    attack_votes = 0

    if cnn_prediction == "ATTACK":
        attack_votes += 1

    if rf_prediction == "ATTACK":
        attack_votes += 1

    if autoencoder_prediction == "ANOMALY":
        attack_votes += 1


    # =====================================================
    # FINAL CLASSIFICATION
    # =====================================================

    if ensemble_score >= ATTACK_THRESHOLD:

        classification = "ATTACK"

    elif ensemble_score >= SUSPICIOUS_THRESHOLD:

        classification = "SUSPICIOUS"

    else:

        classification = "BENIGN"


    # =====================================================
    # RISK LEVEL
    # =====================================================

    if ensemble_score >= CRITICAL_THRESHOLD:

        risk_level = "CRITICAL"

    elif ensemble_score >= HIGH_THRESHOLD:

        risk_level = "HIGH"

    elif ensemble_score >= MEDIUM_THRESHOLD:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # =====================================================
    # RESULT
    # =====================================================

    return {

        "cnn_lstm": {

            "score": round(
                cnn_score,
                2,
            ),

            "prediction":
                cnn_prediction,

        },


        "random_forest": {

            "score": round(
                rf_score,
                2,
            ),

            "prediction":
                rf_prediction,

        },


        "autoencoder": {

            "score": round(
                autoencoder_score,
                2,
            ),

            "reconstruction_error":
                round(
                    reconstruction_error,
                    6,
                ),

            "prediction":
                autoencoder_prediction,

        },


        "ensemble": {

            "ensemble_score":
                round(
                    ensemble_score,
                    2,
                ),

            "classification":
                classification,

            "risk_level":
                risk_level,

            "attack_votes":
                attack_votes,

            "total_models":
                3,

            "primary_model":
                "CNN-LSTM",

            "weights": {

                "cnn_lstm":
                    CNN_WEIGHT,

                "random_forest":
                    RF_WEIGHT,

                "autoencoder":
                    AUTOENCODER_WEIGHT,

            },

        },

    }


# =========================================================
# PRINT RESULT
# =========================================================

def print_result(
    title: str,
    result: dict,
):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print()

    print(
        f"CNN-LSTM       : "
        f"{result['cnn_lstm']['score']:.2f} "
        f"-> "
        f"{result['cnn_lstm']['prediction']}"
    )

    print(
        f"Random Forest  : "
        f"{result['random_forest']['score']:.2f} "
        f"-> "
        f"{result['random_forest']['prediction']}"
    )

    print(
        f"Autoencoder    : "
        f"{result['autoencoder']['score']:.2f} "
        f"-> "
        f"{result['autoencoder']['prediction']}"
    )

    print("-" * 70)

    print(
        "Weighted calculation:"
    )

    print(
        f"  CNN-LSTM      "
        f"{result['cnn_lstm']['score']:.2f} × "
        f"{CNN_WEIGHT:.2f}"
    )

    print(
        f"  Random Forest "
        f"{result['random_forest']['score']:.2f} × "
        f"{RF_WEIGHT:.2f}"
    )

    print(
        f"  Autoencoder   "
        f"{result['autoencoder']['score']:.2f} × "
        f"{AUTOENCODER_WEIGHT:.2f}"
    )

    print("-" * 70)

    print(
        f"Ensemble Score : "
        f"{result['ensemble']['ensemble_score']:.2f}"
    )

    print(
        f"Classification : "
        f"{result['ensemble']['classification']}"
    )

    print(
        f"Risk Level     : "
        f"{result['ensemble']['risk_level']}"
    )

    print(
        f"Attack Votes   : "
        f"{result['ensemble']['attack_votes']}/"
        f"{result['ensemble']['total_models']}"
    )


# =========================================================
# LOAD VALIDATED SEQUENCE
# =========================================================

def load_validated_sequence(
    scenario_name: str,
):

    path = (
        VALIDATED_DIR
        / f"{scenario_name}.npy"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Validated sequence not found:\n"
            f"{path}"
        )

    sequence = np.load(
        path
    ).astype(
        np.float32
    )

    expected_shape = (
        SEQUENCE_LENGTH,
        FEATURE_COUNT,
    )

    if sequence.shape != expected_shape:

        raise ValueError(
            f"{scenario_name}: invalid "
            f"sequence shape "
            f"{sequence.shape}. "
            f"Expected {expected_shape}."
        )

    return sequence


# =========================================================
# MAIN VALIDATION
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("3 MODEL ENSEMBLE VALIDATION")
    print("=" * 70)


    # =====================================================
    # LOAD NORMAL
    # =====================================================

    normal_sequence = (
        load_validated_sequence(
            "normal"
        )
    )


    # =====================================================
    # LOAD GAFGYT
    # =====================================================

    gafgyt_sequence = (
        load_validated_sequence(
            "gafgyt_combo"
        )
    )


    # =====================================================
    # AUTOENCODER CALIBRATION
    # =====================================================

    calibrate_autoencoder(
        normal_sequence,
        gafgyt_sequence,
    )


    # =====================================================
    # SCENARIOS
    # =====================================================

    scenarios = [

        (
            "NORMAL",
            "normal",
            "BENIGN",
        ),

        (
            "GAFGYT_COMBO",
            "gafgyt_combo",
            "ATTACK",
        ),

        (
            "MIRAI_ACK",
            "mirai_ack",
            "ATTACK",
        ),

        (
            "MIRAI_SCAN",
            "mirai_scan",
            "ATTACK",
        ),

        (
            "MIRAI_SYN",
            "mirai_syn",
            "ATTACK",
        ),

        (
            "MIRAI_UDP",
            "mirai_udp",
            "ATTACK",
        ),

        (
            "MIRAI_UDPPLAIN",
            "mirai_udpplain",
            "ATTACK",
        ),

    ]


    # =====================================================
    # TEST ALL SCENARIOS
    # =====================================================

    results = []

    print()
    print("=" * 70)
    print("SCENARIO TESTING")
    print("=" * 70)


    for (
        display_name,
        filename,
        expected,
    ) in scenarios:

        print()
        print("-" * 70)

        print(
            f"TESTING: {display_name}"
        )

        print(
            f"Expected: {expected}"
        )

        try:

            sequence = (
                load_validated_sequence(
                    filename
                )
            )

            result = (
                predict_three_models(
                    sequence
                )
            )

            results.append(
                (
                    display_name,
                    expected,
                    result,
                )
            )

            print_result(
                display_name,
                result,
            )

        except Exception as error:

            print(
                f"ERROR: {error}"
            )


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print()
    print("=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)

    valid_count = 0


    for (
        name,
        expected,
        result,
    ) in results:

        predicted = (
            result["ensemble"]
            ["classification"]
        )

        score = (
            result["ensemble"]
            ["ensemble_score"]
        )


        if expected == "BENIGN":

            valid = (
                predicted == "BENIGN"
            )

        else:

            valid = (
                predicted == "ATTACK"
            )


        if valid:

            valid_count += 1
            mark = "[OK]"

        else:

            mark = "[FAILED]"


        print(
            f"{name:<16} -> "
            f"{predicted:<11} "
            f"({score:6.2f}) "
            f"{mark}"
        )


    print()
    print("-" * 70)

    print(
        f"Validated scenarios: "
        f"{valid_count}/{len(results)}"
    )


    if valid_count == len(results):

        print()
        print(
            "[OK] ALL VALIDATED SCENARIOS "
            "CLASSIFIED CORRECTLY"
        )

    else:

        print()
        print(
            "[WARNING] SOME SCENARIOS NEED "
            "FURTHER CALIBRATION"
        )


    print()
    print("=" * 70)
