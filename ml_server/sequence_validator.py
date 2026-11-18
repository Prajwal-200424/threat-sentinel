import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(
    r"C:\Users\Tejar\Downloads\archive"
)

MODEL_PATH = (
    BASE_DIR
    / "training"
    / "models"
    / "cnn_lstm.h5"
)

SCALER_PATH = (
    BASE_DIR
    / "training"
    / "models"
    / "scaler.pkl"
)

OUTPUT_DIR = (
    BASE_DIR
    / "ml_server"
    / "validated_sequences"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SIZE = 10

SEARCH_WINDOWS = 100


SCENARIOS = {

    "NORMAL": {
        "file_pattern": "*.benign.csv",
        "expected": "BENIGN",
        "output": "normal.npy",
    },

    "GAFGYT_COMBO": {
        "file_pattern": "*.gafgyt.combo.csv",
        "expected": "ATTACK",
        "output": "gafgyt_combo.npy",
    },

    "MIRAI_ACK": {
        "file_pattern": "*.mirai.ack.csv",
        "expected": "ATTACK",
        "output": "mirai_ack.npy",
    },

    "MIRAI_SCAN": {
        "file_pattern": "*.mirai.scan.csv",
        "expected": "ATTACK",
        "output": "mirai_scan.npy",
    },

    "MIRAI_SYN": {
        "file_pattern": "*.mirai.syn.csv",
        "expected": "ATTACK",
        "output": "mirai_syn.npy",
    },

    "MIRAI_UDP": {
        "file_pattern": "*.mirai.udp.csv",
        "expected": "ATTACK",
        "output": "mirai_udp.npy",
    },

    "MIRAI_UDPPLAIN": {
        "file_pattern": "*.mirai.udpplain.csv",
        "expected": "ATTACK",
        "output": "mirai_udpplain.npy",
    },

}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("DIGITAL TWIN SEQUENCE VALIDATOR")
print("=" * 70)

print("\nLoading CNN-LSTM...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("CNN-LSTM loaded.")


print("\nLoading scaler...")

with open(
    SCALER_PATH,
    "rb",
) as file:

    scaler = pickle.load(file)


print("Scaler loaded.")

print(
    f"Features : {scaler.n_features_in_}"
)

print(
    f"Input    : {model.input_shape}"
)


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(csv_path: Path):

    df = pd.read_csv(
        csv_path
    )

    if "label" in df.columns:

        df = df.drop(
            "label",
            axis=1,
        )

    if len(df) < WINDOW_SIZE:

        raise ValueError(
            f"{csv_path.name} has fewer than "
            f"{WINDOW_SIZE} rows."
        )

    if (
        len(df.columns)
        != scaler.n_features_in_
    ):

        raise ValueError(
            f"{csv_path.name} contains "
            f"{len(df.columns)} features. "
            f"Expected "
            f"{scaler.n_features_in_}."
        )

    return df.values.astype(
        np.float32
    )


# ============================================================
# CREATE WINDOWS
# ============================================================

def create_windows(
    data: np.ndarray,
    count: int = SEARCH_WINDOWS,
):

    max_start = (
        len(data)
        - WINDOW_SIZE
        + 1
    )

    count = min(
        count,
        max_start,
    )

    windows = []

    for i in range(count):

        window = data[
            i:i + WINDOW_SIZE
        ]

        windows.append(
            window
        )

    return np.asarray(
        windows,
        dtype=np.float32,
    )


# ============================================================
# PREDICT WINDOWS
# ============================================================

def predict_windows(
    windows: np.ndarray,
):

    batch_size = (
        windows.shape[0]
    )

    flat = windows.reshape(
        -1,
        windows.shape[-1],
    )

    scaled = scaler.transform(
        flat
    )

    scaled = scaled.reshape(
        batch_size,
        WINDOW_SIZE,
        scaler.n_features_in_,
    )

    probabilities = model.predict(
        scaled,
        verbose=0,
    )

    probabilities = np.asarray(
        probabilities
    ).reshape(-1)

    return probabilities


# ============================================================
# FIND VALIDATED SEQUENCE
# ============================================================

def validate_scenario(
    scenario_name: str,
    config: dict,
):

    print("\n" + "=" * 70)

    print(
        f"SEARCHING: {scenario_name}"
    )

    print("=" * 70)


    files = sorted(
        DATA_DIR.glob(
            config["file_pattern"]
        )
    )


    if not files:

        print(
            "ERROR: No matching CSV files found."
        )

        return None


    # Use first matching dataset file
    csv_path = files[0]


    print(
        f"Dataset file : {csv_path.name}"
    )


    data = load_csv(
        csv_path
    )


    print(
        f"Dataset target : "
        f"{config['expected']}"
    )

    print(
        f"Dataset rows   : "
        f"{len(data)}"
    )


    windows = create_windows(
        data,
        SEARCH_WINDOWS,
    )


    print(
        f"Windows tested : "
        f"{len(windows)}"
    )


    probabilities = predict_windows(
        windows
    )


    expected = config[
        "expected"
    ]


    # --------------------------------------------------------
    # Find the best suitable window
    # --------------------------------------------------------

    if expected == "BENIGN":

        # Lowest attack probability
        best_index = int(
            np.argmin(
                probabilities
            )
        )

        attack_probability = float(
            probabilities[
                best_index
            ]
        )

        prediction = (
            "ATTACK"
            if attack_probability >= 0.5
            else "BENIGN"
        )

        confidence = (
            attack_probability
            if prediction == "ATTACK"
            else 1.0 - attack_probability
        )

    else:

        # Highest attack probability
        best_index = int(
            np.argmax(
                probabilities
            )
        )

        attack_probability = float(
            probabilities[
                best_index
            ]
        )

        prediction = (
            "ATTACK"
            if attack_probability >= 0.5
            else "BENIGN"
        )

        confidence = (
            attack_probability
            if prediction == "ATTACK"
            else 1.0 - attack_probability
        )


    # --------------------------------------------------------
    # Save only if classification matches expectation
    # --------------------------------------------------------

    if prediction == expected:

        validated_sequence = (
            windows[
                best_index
            ]
        )


        output_path = (
            OUTPUT_DIR
            / config["output"]
        )


        np.save(
            output_path,
            validated_sequence,
        )


        print(
            "\nSaved validated sequence:"
        )

        print(
            output_path
        )


        print(
            f"\n{scenario_name} RESULT"
        )

        print(
            "-" * 50
        )

        print(
            f"Expected    : "
            f"{expected}"
        )

        print(
            f"Prediction  : "
            f"{prediction}"
        )

        print(
            f"Confidence  : "
            f"{confidence:.2%}"
        )

        print(
            f"Attack Prob : "
            f"{attack_probability:.2%}"
        )

        print(
            f"Window      : "
            f"{best_index}"
        )


        return {
            "scenario": scenario_name,
            "expected": expected,
            "prediction": prediction,
            "confidence": confidence,
            "attack_probability":
                attack_probability,
            "window": best_index,
            "file": csv_path.name,
            "output": str(output_path),
            "valid": True,
        }


    print(
        f"\n{scenario_name} RESULT"
    )

    print(
        "-" * 50
    )

    print(
        f"Expected    : "
        f"{expected}"
    )

    print(
        f"Prediction  : "
        f"{prediction}"
    )

    print(
        f"Confidence  : "
        f"{confidence:.2%}"
    )

    print(
        f"Attack Prob : "
        f"{attack_probability:.2%}"
    )

    print(
        f"Window      : "
        f"{best_index}"
    )

    print(
        "\nWARNING:"
    )

    print(
        "No suitable sequence matched "
        "the expected classification."
    )


    return {
        "scenario": scenario_name,
        "expected": expected,
        "prediction": prediction,
        "confidence": confidence,
        "attack_probability":
            attack_probability,
        "window": best_index,
        "file": csv_path.name,
        "output": None,
        "valid": False,
    }


# ============================================================
# RUN VALIDATION
# ============================================================

results = []


for scenario_name, config in SCENARIOS.items():

    result = validate_scenario(
        scenario_name,
        config,
    )

    if result is not None:

        results.append(
            result
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)


for result in results:

    status = (
        "✓ VALID"
        if result["valid"]
        else "✗ INVALID"
    )


    print(
        f"{result['scenario']:<15}"
        f" → "
        f"{result['prediction']:<7}"
        f" "
        f"({result['confidence']:.2%})"
        f"  {status}"
    )


# ============================================================
# FINAL STATUS
# ============================================================

valid_count = sum(
    1
    for result in results
    if result["valid"]
)


print("\n" + "-" * 70)

print(
    f"Validated scenarios: "
    f"{valid_count}/{len(results)}"
)


if valid_count == len(results):

    print(
        "\n✓ ALL SCENARIOS HAVE "
        "SUITABLE VALIDATED SEQUENCES."
    )

else:

    print(
        "\n⚠ Some scenarios need "
        "additional investigation."
    )


print("=" * 70)