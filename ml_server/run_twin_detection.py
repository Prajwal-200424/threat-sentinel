import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
ML_SERVER_DIR = BASE_DIR / "ml_server"

sys.path.insert(0, str(ML_SERVER_DIR))

from digital_twin import DigitalTwin
from inference import predict


def run_detection(twin: DigitalTwin):
    """
    Get a Digital Twin network sequence and
    send it through the real CNN-LSTM model.
    """

    sequence = twin.get_next_sequence()

    # CNN-LSTM expects:
    # (batch, 10, 115)
    model_input = np.expand_dims(
        sequence,
        axis=0,
    )

    result = predict(model_input)

    event = twin.record_event(result)

    return event


def print_result(event):
    print("\n" + "=" * 65)
    print("DIGITAL TWIN → ML DETECTION")
    print("=" * 65)

    print(f"Device       : {event['device']}")
    print(f"Scenario     : {event['scenario']}")
    print(f"Scenario Name: {event['scenario_label']}")

    print("-" * 65)

    print(f"ML Model     : {event['model_name']}")
    print(f"Prediction   : {event['prediction']}")
    print(
        f"Confidence   : "
        f"{event['confidence_score'] * 100:.2f}%"
    )
    print(f"Risk Level   : {event['risk_level']}")

    print("=" * 65)


if __name__ == "__main__":

    twin = DigitalTwin()

    # =====================================================
    # TEST 1 — NORMAL
    # =====================================================

    print("\n\nTEST 1 — NORMAL NETWORK")

    twin.set_target("ESP32-01")
    twin.set_scenario("NORMAL")

    normal_event = run_detection(twin)

    print_result(normal_event)

    # =====================================================
    # TEST 2 — GAFGYT
    # =====================================================

    print("\n\nTEST 2 — GAFGYT ATTACK")

    twin.set_target("ESP32-01")
    twin.set_scenario("GAFGYT_COMBO")

    attack_event = run_detection(twin)

    print_result(attack_event)

    # =====================================================
    # HISTORY
    # =====================================================

    print("\nDetection history:")

    for event in twin.get_history():
        print(
            f"{event['device']} | "
            f"{event['scenario']} | "
            f"{event['prediction']} | "
            f"{event['confidence_score']:.4f}"
        )