"""HTTP contract tests for the Threat Sentinel ML server.

These tests exercise the Flask routes with simulated JSON payloads. They do not
contact an ESP32, and they do not prove physical hardware integration. Inference
itself is replaced by a stub so the suite does not depend on local TensorFlow or
model artifacts; the real models are covered by the training evaluation scripts.
"""

import copy
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


# The route tests exercise validation and status semantics.  Inference itself is
# covered by its model-specific code and would make this suite depend on local
# TensorFlow/model-artifact availability.
ensemble_stub = types.ModuleType("ensemble_inference")
ensemble_stub.predict_three_models = lambda sequence: {}
ensemble_stub.calibrate_autoencoder = lambda benign, attack: None
ensemble_stub.load_validated_sequence = lambda name: [[0.0] * 115] * 10
sys.modules.setdefault("ensemble_inference", ensemble_stub)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app as server  # noqa: E402


def valid_telemetry_payload(**overrides):
    payload = {
        "device": "ESP32-01",
        "wifi_rssi": -62,
        "voltage": 3.3,
        "messages_per_second": 4,
        "bytes_per_second": 512,
        "active_connections": 2,
    }
    payload.update(overrides)
    return payload


VALID_MODEL_RESULT = {
    "cnn_lstm": {"score": 90.0, "prediction": "ATTACK"},
    "random_forest": {"score": 60.0, "prediction": "ATTACK"},
    "autoencoder": {"score": 40.0, "prediction": "NORMAL"},
    "ensemble": {
        "ensemble_score": 70.0,
        "classification": "ATTACK",
        "risk_level": "HIGH",
        "attack_votes": 2,
        "total_models": 3,
        "primary_model": "CNN-LSTM",
    },
}


def seconds_ago(seconds):
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


class ServerTestCase(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        self.saved_telemetry = server.latest_telemetry
        self.saved_current_state = copy.deepcopy(server.current_state)
        self.saved_history = list(server.twin.history)
        self.saved_predictor = server.predict_three_models
        self.saved_model_ready = server.MODEL_READY
        server.latest_telemetry = {
            "device": "ESP32-01",
            "timestamp": None,
            "status": "WAITING",
        }
        server.current_state["telemetry"] = dict(server.latest_telemetry)
        server.twin.history.clear()
        server.predict_three_models = lambda sequence: dict(VALID_MODEL_RESULT)

    def tearDown(self):
        server.latest_telemetry = self.saved_telemetry
        server.current_state.clear()
        server.current_state.update(self.saved_current_state)
        server.twin.history.clear()
        server.twin.history.extend(self.saved_history)
        server.predict_three_models = self.saved_predictor
        server.MODEL_READY = self.saved_model_ready


class TelemetryApiTests(ServerTestCase):
    def test_accepts_valid_telemetry_without_weather_dependency(self):
        response = self.client.post(
            "/api/telemetry",
            json=valid_telemetry_payload(wifi_rssi="-62", voltage="3.30"),
        )
        self.assertEqual(response.status_code, 200)

        telemetry = response.get_json()["telemetry"]
        self.assertEqual(telemetry["status"], "ONLINE")
        self.assertEqual(telemetry["wifi_rssi"], -62.0)
        self.assertEqual(telemetry["voltage"], 3.3)
        self.assertEqual(telemetry["device"], "ESP32-01")
        # Environmental values are attributed to Open-Meteo, never to the ESP32.
        self.assertIn("weather_source", telemetry)

    def test_receipt_timestamp_is_server_generated(self):
        device_timestamp = "2001-02-03T04:05:06+00:00"
        response = self.client.post(
            "/api/telemetry",
            json=valid_telemetry_payload(timestamp=device_timestamp),
        )
        self.assertEqual(response.status_code, 200)

        telemetry = response.get_json()["telemetry"]
        self.assertNotEqual(telemetry["timestamp"], device_timestamp)
        received_at = datetime.fromisoformat(telemetry["timestamp"])
        self.assertIsNotNone(received_at.tzinfo)
        self.assertLess(abs((datetime.now(timezone.utc) - received_at).total_seconds()), 30)

    def test_rejects_invalid_or_unknown_telemetry(self):
        self.assertEqual(
            self.client.post("/api/telemetry", data="not json").status_code,
            400,
        )
        response = self.client.post("/api/telemetry", json={"device": "unknown"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown device", response.get_json()["error"])
        self.assertIn("ESP32-01", response.get_json()["registered_devices"])

    def test_rejects_missing_or_non_object_payload(self):
        self.assertEqual(self.client.post("/api/telemetry", json={}).status_code, 400)
        response = self.client.post("/api/telemetry", json=[1, 2, 3])
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])

    def test_rejects_non_numeric_or_non_finite_values(self):
        for field in ("wifi_rssi", "voltage", "messages_per_second", "active_connections"):
            for bad_value in ("abc", "NaN", "inf", "-inf"):
                with self.subTest(field=field, value=bad_value):
                    response = self.client.post(
                        "/api/telemetry",
                        json=valid_telemetry_payload(**{field: bad_value}),
                    )
                    self.assertEqual(response.status_code, 400)
                    self.assertIn(field, response.get_json()["error"])

    def test_rejects_negative_traffic_and_connection_counts(self):
        for field in ("messages_per_second", "bytes_per_second", "active_connections"):
            with self.subTest(field=field):
                response = self.client.post(
                    "/api/telemetry",
                    json=valid_telemetry_payload(**{field: -1}),
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("cannot be negative", response.get_json()["error"])

    def test_rejects_oversized_telemetry(self):
        response = self.client.post(
            "/api/telemetry",
            data=b"x" * (65 * 1024),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 413)
        self.assertFalse(response.get_json()["success"])


class TelemetryFreshnessTests(ServerTestCase):
    def test_never_received_telemetry_reports_waiting(self):
        telemetry = self.client.get("/api/telemetry").get_json()["telemetry"]
        self.assertEqual(telemetry["status"], "WAITING")
        self.assertIsNone(telemetry["age_seconds"])

    def test_fresh_receipt_reports_online_with_age(self):
        server.latest_telemetry = {
            "device": "ESP32-01",
            "timestamp": seconds_ago(1),
            "status": "ONLINE",
        }
        telemetry = self.client.get("/api/telemetry").get_json()["telemetry"]
        self.assertEqual(telemetry["status"], "ONLINE")
        self.assertGreaterEqual(telemetry["age_seconds"], 0)
        self.assertLess(telemetry["age_seconds"], server.TELEMETRY_STALE_SECONDS)

    def test_old_receipt_is_downgraded_to_stale(self):
        server.latest_telemetry = {
            "device": "ESP32-01",
            "timestamp": seconds_ago(server.TELEMETRY_STALE_SECONDS + 5),
            # A stored ONLINE status must not survive the freshness check.
            "status": "ONLINE",
        }
        telemetry = self.client.get("/api/telemetry").get_json()["telemetry"]
        self.assertEqual(telemetry["status"], "STALE")
        self.assertGreater(telemetry["age_seconds"], server.TELEMETRY_STALE_SECONDS)

    def test_unparsable_timestamp_is_reported_not_online(self):
        for timestamp in ("not-a-timestamp", "2026-01-01T00:00:00", 12345):
            with self.subTest(timestamp=timestamp):
                server.latest_telemetry = {
                    "device": "ESP32-01",
                    "timestamp": timestamp,
                    "status": "ONLINE",
                }
                telemetry = self.client.get("/api/telemetry").get_json()["telemetry"]
                self.assertEqual(telemetry["status"], "INVALID_TIMESTAMP")
                self.assertIsNone(telemetry["age_seconds"])

    def test_future_timestamp_is_clamped_to_zero_age(self):
        server.latest_telemetry = {
            "device": "ESP32-01",
            "timestamp": seconds_ago(-120),
            "status": "ONLINE",
        }
        telemetry = self.client.get("/api/telemetry").get_json()["telemetry"]
        self.assertEqual(telemetry["status"], "ONLINE")
        self.assertEqual(telemetry["age_seconds"], 0.0)

    def test_staleness_is_reflected_in_status_and_latest_prediction(self):
        server.latest_telemetry = {
            "device": "ESP32-01",
            "timestamp": seconds_ago(server.TELEMETRY_STALE_SECONDS + 60),
            "status": "ONLINE",
        }

        status_payload = self.client.get("/api/digital-twin/status").get_json()
        self.assertEqual(
            status_payload["status"]["telemetry"]["status"],
            "STALE",
        )

        latest_payload = self.client.get("/api/predictions/latest").get_json()
        self.assertEqual(latest_payload["telemetry"]["status"], "STALE")

    def test_received_packet_resets_staleness(self):
        server.latest_telemetry = {
            "device": "ESP32-01",
            "timestamp": seconds_ago(server.TELEMETRY_STALE_SECONDS + 60),
            "status": "ONLINE",
        }
        self.assertEqual(
            self.client.get("/api/telemetry").get_json()["telemetry"]["status"],
            "STALE",
        )

        self.client.post("/api/telemetry", json=valid_telemetry_payload())
        telemetry = self.client.get("/api/telemetry").get_json()["telemetry"]
        self.assertEqual(telemetry["status"], "ONLINE")


class ModelReadinessTests(ServerTestCase):
    def test_health_reports_ready_and_exact_ensemble_weights(self):
        response = self.client.get("/api/health")
        payload = response.get_json()

        self.assertEqual(payload["status"], "READY")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["model_ready"])
        self.assertIsNone(payload["model_error"])
        # The deployed ensemble is fixed at 50/25/25.
        self.assertEqual(
            payload["weights"],
            {
                "cnn_lstm": "50%",
                "random_forest": "25%",
                "autoencoder": "25%",
            },
        )
        self.assertEqual(
            payload["models"],
            ["CNN-LSTM", "Random Forest", "Autoencoder"],
        )

    def test_health_reports_degraded_when_models_are_unavailable(self):
        server.MODEL_READY = False

        response = self.client.get("/api/health")
        payload = response.get_json()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(payload["status"], "DEGRADED")
        self.assertFalse(payload["model_ready"])

    def test_detection_endpoints_fail_loudly_when_models_are_unavailable(self):
        server.MODEL_READY = False

        for path in (
            "/api/digital-twin/detect",
            "/api/digital-twin/attack",
            "/api/digital-twin/normal",
        ):
            with self.subTest(path=path):
                response = self.client.post(path, json={})
                self.assertEqual(response.status_code, 500)
                payload = response.get_json()
                self.assertFalse(payload["success"])
                self.assertIn("error", payload)

        # Nothing may be recorded as a detection when inference did not run.
        self.assertEqual(server.twin.get_history(), [])


class PredictionContractTests(ServerTestCase):
    def test_predictions_endpoint_returns_list_envelope(self):
        response = self.client.get("/api/predictions")
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("predictions", payload)
        self.assertIsInstance(payload["predictions"], list)

    def test_detect_returns_dashboard_contract_fields(self):
        response = self.client.post("/api/digital-twin/detect", json={})
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])

        result = payload["result"]
        for field in (
            "id",
            "created_at",
            "timestamp",
            "device",
            "source_ip",
            "destination_ip",
            "protocol",
            "device_status",
            "prediction",
            "confidence_score",
            "attack_probability",
            "risk_level",
            "ensemble_score",
            "attack_votes",
            "total_models",
            "models",
            "telemetry",
        ):
            self.assertIn(field, result)

        self.assertEqual(result["prediction"], "ATTACK")
        self.assertEqual(result["ensemble_score"], 70.0)
        # confidence_score is the probability of the reported verdict.
        self.assertAlmostEqual(result["confidence_score"], 0.70)
        self.assertAlmostEqual(result["attack_probability"], 0.70)
        self.assertEqual(result["models"]["ensemble"]["attack_votes"], 2)

    def test_detect_returns_error_for_malformed_inference_result(self):
        malformed_results = [
            {},
            {"cnn_lstm": {"score": 1.0, "prediction": "BENIGN"}},
            {"ensemble": {"classification": "ATTACK"}},
        ]

        for malformed in malformed_results:
            with self.subTest(result=malformed):
                server.predict_three_models = lambda sequence, value=malformed: value
                response = self.client.post("/api/digital-twin/detect", json={})

                self.assertEqual(response.status_code, 500)
                payload = response.get_json()
                self.assertFalse(payload["success"])
                self.assertTrue(payload["error"])

        self.assertEqual(server.twin.get_history(), [])

    def test_latest_prediction_shape_does_not_claim_benign_before_inference(self):
        payload = self.client.get("/api/predictions/latest").get_json()

        self.assertEqual(payload["model_name"], "3-Model Ensemble")
        # No detection has run in this test, so the dashboard must receive an
        # explicit unavailable verdict rather than a benign one.
        self.assertEqual(payload["prediction"], "UNAVAILABLE")
        self.assertIsNone(payload["ensemble_score"])
        self.assertIn("telemetry", payload)
        self.assertEqual(payload["telemetry"]["status"], "WAITING")


if __name__ == "__main__":
    unittest.main()
