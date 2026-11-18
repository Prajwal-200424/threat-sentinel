"""
4-Model Ensemble Scoring Engine
Threat Sentinel

Models:
1. CNN-LSTM
2. XGBoost
3. Random Forest
4. Autoencoder

All model outputs are normalized to a 0-100 score.
The engine produces:
- individual model scores
- individual classifications
- final ensemble score
- final classification
- risk level
"""

from typing import Dict


# =========================================================
# MODEL WEIGHTS
# =========================================================

MODEL_WEIGHTS = {
    "cnn_lstm": 0.40,
    "xgboost": 0.25,
    "random_forest": 0.20,
    "autoencoder": 0.15,
}


# =========================================================
# CLASSIFICATION THRESHOLDS
# =========================================================

BENIGN_THRESHOLD = 30.0
SUSPICIOUS_THRESHOLD = 60.0
HIGH_RISK_THRESHOLD = 80.0


# =========================================================
# HELPERS
# =========================================================

def clamp_score(value: float) -> float:
    """Keep score between 0 and 100."""

    return max(0.0, min(100.0, float(value)))


def probability_to_score(probability: float) -> float:
    """
    Convert probability 0-1 to score 0-100.
    """

    return clamp_score(
        float(probability) * 100.0
    )


def classify_score(score: float) -> str:
    """
    Classify an individual model score.
    """

    score = clamp_score(score)

    if score < BENIGN_THRESHOLD:
        return "BENIGN"

    if score < SUSPICIOUS_THRESHOLD:
        return "SUSPICIOUS"

    return "ATTACK"


def calculate_risk(score: float) -> str:
    """
    Convert final anomaly score into risk level.
    """

    score = clamp_score(score)

    if score < BENIGN_THRESHOLD:
        return "LOW"

    if score < SUSPICIOUS_THRESHOLD:
        return "MEDIUM"

    if score < HIGH_RISK_THRESHOLD:
        return "HIGH"

    return "CRITICAL"


# =========================================================
# ENSEMBLE
# =========================================================

def calculate_ensemble(
    cnn_lstm_probability: float,
    xgboost_probability: float,
    random_forest_probability: float,
    autoencoder_score: float,
) -> Dict:
    """
    Calculate final 4-model threat score.

    Classifier probabilities:
        0.0 -> benign
        1.0 -> attack

    Autoencoder:
        already supplied as 0-100 anomaly score.
    """

    # -----------------------------------------------------
    # NORMALIZE INDIVIDUAL SCORES
    # -----------------------------------------------------

    scores = {
        "cnn_lstm": probability_to_score(
            cnn_lstm_probability
        ),

        "xgboost": probability_to_score(
            xgboost_probability
        ),

        "random_forest": probability_to_score(
            random_forest_probability
        ),

        "autoencoder": clamp_score(
            autoencoder_score
        ),
    }

    # -----------------------------------------------------
    # INDIVIDUAL CLASSIFICATIONS
    # -----------------------------------------------------

    classifications = {
        model: classify_score(score)
        for model, score in scores.items()
    }

    # -----------------------------------------------------
    # WEIGHTED ENSEMBLE
    # -----------------------------------------------------

    final_score = (
        scores["cnn_lstm"]
        * MODEL_WEIGHTS["cnn_lstm"]

        + scores["xgboost"]
        * MODEL_WEIGHTS["xgboost"]

        + scores["random_forest"]
        * MODEL_WEIGHTS["random_forest"]

        + scores["autoencoder"]
        * MODEL_WEIGHTS["autoencoder"]
    )

    final_score = clamp_score(
        final_score
    )

    # -----------------------------------------------------
    # FINAL CLASSIFICATION
    # -----------------------------------------------------

    if final_score < BENIGN_THRESHOLD:

        classification = "BENIGN"

    elif final_score < SUSPICIOUS_THRESHOLD:

        classification = "SUSPICIOUS"

    else:

        classification = "ATTACK"

    # -----------------------------------------------------
    # RISK
    # -----------------------------------------------------

    risk_level = calculate_risk(
        final_score
    )

    # -----------------------------------------------------
    # ATTACK MODEL COUNT
    # -----------------------------------------------------

    attack_votes = sum(
        1
        for value in classifications.values()
        if value == "ATTACK"
    )

    # -----------------------------------------------------
    # RETURN
    # -----------------------------------------------------

    return {
        "model_scores": {
            "CNN-LSTM": round(
                scores["cnn_lstm"],
                2,
            ),

            "XGBoost": round(
                scores["xgboost"],
                2,
            ),

            "Random Forest": round(
                scores["random_forest"],
                2,
            ),

            "Autoencoder": round(
                scores["autoencoder"],
                2,
            ),
        },

        "model_classifications": {
            "CNN-LSTM":
                classifications["cnn_lstm"],

            "XGBoost":
                classifications["xgboost"],

            "Random Forest":
                classifications["random_forest"],

            "Autoencoder":
                classifications["autoencoder"],
        },

        "model_weights": {
            "CNN-LSTM": MODEL_WEIGHTS["cnn_lstm"],
            "XGBoost": MODEL_WEIGHTS["xgboost"],
            "Random Forest": MODEL_WEIGHTS["random_forest"],
            "Autoencoder": MODEL_WEIGHTS["autoencoder"],
        },

        "attack_votes": attack_votes,

        "total_models": 4,

        "ensemble_score": round(
            final_score,
            2,
        ),

        "classification":
            classification,

        "risk_level":
            risk_level,
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("THREAT SENTINEL - 4 MODEL ENSEMBLE TEST")
    print("=" * 70)

    # -----------------------------------------------------
    # ATTACK TEST
    # -----------------------------------------------------

    print("\nATTACK TEST")
    print("-" * 70)

    attack_result = calculate_ensemble(
        cnn_lstm_probability=0.9966,
        xgboost_probability=0.98,
        random_forest_probability=0.97,
        autoencoder_score=94.0,
    )

    print(
        f"CNN-LSTM       : "
        f"{attack_result['model_scores']['CNN-LSTM']:.2f}"
    )

    print(
        f"XGBoost        : "
        f"{attack_result['model_scores']['XGBoost']:.2f}"
    )

    print(
        f"Random Forest  : "
        f"{attack_result['model_scores']['Random Forest']:.2f}"
    )

    print(
        f"Autoencoder    : "
        f"{attack_result['model_scores']['Autoencoder']:.2f}"
    )

    print(
        f"\nEnsemble Score : "
        f"{attack_result['ensemble_score']:.2f}"
    )

    print(
        f"Classification : "
        f"{attack_result['classification']}"
    )

    print(
        f"Risk Level     : "
        f"{attack_result['risk_level']}"
    )

    print(
        f"Attack Votes   : "
        f"{attack_result['attack_votes']}/4"
    )

    # -----------------------------------------------------
    # BENIGN TEST
    # -----------------------------------------------------

    print("\nBENIGN TEST")
    print("-" * 70)

    benign_result = calculate_ensemble(
        cnn_lstm_probability=0.000005,
        xgboost_probability=0.001,
        random_forest_probability=0.002,
        autoencoder_score=2.0,
    )

    print(
        f"CNN-LSTM       : "
        f"{benign_result['model_scores']['CNN-LSTM']:.2f}"
    )

    print(
        f"XGBoost        : "
        f"{benign_result['model_scores']['XGBoost']:.2f}"
    )

    print(
        f"Random Forest  : "
        f"{benign_result['model_scores']['Random Forest']:.2f}"
    )

    print(
        f"Autoencoder    : "
        f"{benign_result['model_scores']['Autoencoder']:.2f}"
    )

    print(
        f"\nEnsemble Score : "
        f"{benign_result['ensemble_score']:.2f}"
    )

    print(
        f"Classification : "
        f"{benign_result['classification']}"
    )

    print(
        f"Risk Level     : "
        f"{benign_result['risk_level']}"
    )

    print(
        f"Attack Votes   : "
        f"{benign_result['attack_votes']}/4"
    )

    print("\n" + "=" * 70)
    print("ENSEMBLE ENGINE TEST COMPLETE")
    print("=" * 70)