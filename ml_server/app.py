from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timezone
import os
import threading
import requests

from digital_twin import DigitalTwin

# Keep the HTTP service available even when model artifacts or their runtime are
# unavailable.  Detection endpoints report that condition explicitly and the
# health endpoint exposes it, instead of a successful-looking default result.
MODEL_READY = False
MODEL_ERROR = None
try:
    from ensemble_inference import (
        predict_three_models,
        calibrate_autoencoder,
        load_validated_sequence,
    )
    MODEL_READY = True
except Exception as error:  # pragma: no cover - depends on local model runtime
    MODEL_ERROR = str(error)
    predict_three_models = None
    calibrate_autoencoder = None
    load_validated_sequence = None


# =========================================================
# APP
# =========================================================

app = Flask(__name__)
# Telemetry is a small JSON message; reject unexpectedly large request bodies
# before parsing them.
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024


@app.errorhandler(413)
def request_too_large(error):
    return jsonify({"success": False, "error": "Request body is too large."}), 413

allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080",
    ).split(",")
    if origin.strip()
]
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})


# =========================================================
# DIGITAL TWIN
# =========================================================

twin = DigitalTwin()
lock = threading.Lock()


# =========================================================
# SYNTHETIC TEST SCENARIOS
# =========================================================
#
# IMPORTANT:
#
# These scenario names are application-level test cases.
#
# They DO NOT train the ML models.
#
# They DO NOT hardcode the ML prediction.
#
# Each scenario is mapped to an existing validated Digital
# Twin traffic pattern and then passed through the actual:
#
#     CNN-LSTM
#     Random Forest
#     Autoencoder
#
# The models still perform the prediction.
#
# Later, these can be replaced with dedicated 115-feature
# generators for true unseen-behaviour testing.
# =========================================================

SYNTHETIC_SCENARIOS = {
    "PORT_SCAN": {
        "name": "Port Scan",
        "type": "ATTACK",
        "attack_type": "NETWORK_RECONNAISSANCE",
        "expected": "ATTACK",
        "base_scenario": "MIRAI_SCAN",
    },
    "UDP_FLOOD": {
        "name": "UDP Flood",
        "type": "ATTACK",
        "attack_type": "UDP_FLOOD",
        "expected": "ATTACK",
        "base_scenario": "MIRAI_UDP",
    },
    "TCP_SYN_FLOOD": {
        "name": "TCP SYN Flood",
        "type": "ATTACK",
        "attack_type": "TCP_SYN_FLOOD",
        "expected": "ATTACK",
        "base_scenario": "MIRAI_SYN",
    },
    "ICMP_FLOOD": {
        "name": "ICMP Flood",
        "type": "ATTACK",
        "attack_type": "ICMP_FLOOD",
        "expected": "ATTACK",
        "base_scenario": "MIRAI_UDP",
    },
    "DNS_TUNNEL": {
        "name": "DNS Tunneling",
        "type": "ATTACK",
        "attack_type": "DNS_TUNNEL",
        "expected": "ATTACK",
        "base_scenario": "GAFGYT_COMBO",
    },
    "BRUTE_FORCE": {
        "name": "Brute Force",
        "type": "ATTACK",
        "attack_type": "BRUTE_FORCE",
        "expected": "ATTACK",
        "base_scenario": "GAFGYT_COMBO",
    },
    "MQTT_FLOOD": {
        "name": "MQTT Flood",
        "type": "ATTACK",
        "attack_type": "MQTT_FLOOD",
        "expected": "ATTACK",
        "base_scenario": "MIRAI_UDP",
    },
    "MQTT_ANOMALY": {
        "name": "MQTT Anomalous Traffic",
        "type": "ANOMALY",
        "attack_type": "MQTT_ANOMALY",
        "expected": "ANOMALY",
        "base_scenario": "GAFGYT_COMBO",
    },
    "HTTP_FLOOD": {
        "name": "HTTP Flood",
        "type": "ATTACK",
        "attack_type": "HTTP_FLOOD",
        "expected": "ATTACK",
        "base_scenario": "MIRAI_ACK",
    },
    "DEVICE_SCAN": {
        "name": "IoT Device Scan",
        "type": "ATTACK",
        "attack_type": "DEVICE_SCAN",
        "expected": "ATTACK",
        "base_scenario": "MIRAI_SCAN",
    },
    "CREDENTIAL_ATTACK": {
        "name": "Credential Attack",
        "type": "ATTACK",
        "attack_type": "CREDENTIAL_ATTACK",
        "expected": "ATTACK",
        "base_scenario": "GAFGYT_COMBO",
    },
    "DATA_EXFILTRATION": {
        "name": "Data Exfiltration",
        "type": "ATTACK",
        "attack_type": "DATA_EXFILTRATION",
        "expected": "ATTACK",
        "base_scenario": "MIRAI_UDP",
    },
    "BEACONING": {
        "name": "Command and Control Beaconing",
        "type": "ANOMALY",
        "attack_type": "BEACONING",
        "expected": "ANOMALY",
        "base_scenario": "GAFGYT_COMBO",
    },
    "RESOURCE_ABUSE": {
        "name": "IoT Resource Abuse",
        "type": "ANOMALY",
        "attack_type": "RESOURCE_ABUSE",
        "expected": "ANOMALY",
        "base_scenario": "GAFGYT_COMBO",
    },
    "ZERO_DAY_BEHAVIOUR": {
        "name": "Synthetic Zero-Day-Like Behaviour",
        "type": "UNKNOWN",
        "attack_type": "UNKNOWN_ANOMALOUS_BEHAVIOUR",
        "expected": "ANOMALY",
        "base_scenario": "GAFGYT_COMBO",
    },
}


# =========================================================
# SCENARIO RESOLVER
# =========================================================

def resolve_scenario(scenario):
    scenario = str(scenario or "NORMAL").upper().strip()

    # -----------------------------------------------------
    # Existing Digital Twin scenarios
    # -----------------------------------------------------
    existing_scenarios = {
        "NORMAL",
        "GAFGYT_COMBO",
        "MIRAI_ACK",
        "MIRAI_SCAN",
        "MIRAI_SYN",
        "MIRAI_UDP",
        "MIRAI_UDPPLAIN",
    }

    if scenario in existing_scenarios:
        return scenario, {
            "scenario": scenario,
            "label": scenario.replace("_", " ").title(),
            "attack_type": "BENIGN" if scenario == "NORMAL" else "KNOWN_ATTACK",
            "expected": "BENIGN" if scenario == "NORMAL" else "ATTACK",
            "synthetic": False,
        }

    # -----------------------------------------------------
    # Synthetic scenarios
    # -----------------------------------------------------
    if scenario in SYNTHETIC_SCENARIOS:
        info = SYNTHETIC_SCENARIOS[scenario]
        return info["base_scenario"], {
            "scenario": scenario,
            "label": info["name"],
            "attack_type": info["attack_type"],
            "expected": info["expected"],
            "synthetic": True,
            "base_scenario": info["base_scenario"],
        }

    raise ValueError(f"Unknown scenario: {scenario}")


# =========================================================
# ENSEMBLE CALIBRATION
# =========================================================

print()
print("=" * 65)
print("INITIALIZING 3-MODEL ENSEMBLE API")
print("=" * 65)

try:
    if not MODEL_READY:
        raise RuntimeError(MODEL_ERROR or "Model runtime is unavailable")
    normal_sequence = load_validated_sequence("normal")
    attack_sequence = load_validated_sequence("gafgyt_combo")
    calibrate_autoencoder(normal_sequence, attack_sequence)
    print("[OK] Ensemble calibration complete")
except Exception as error:
    print(f"[WARNING] Ensemble calibration failed: {error}")


# =========================================================
# EXTERNAL TEMPERATURE / HUMIDITY API
# =========================================================
#
# NO physical temperature/humidity sensors are used.
#
# Flask obtains environmental data from Open-Meteo.
#
# ESP32 -> Flask -> Open-Meteo
#       -> combined telemetry -> Dashboard
# =========================================================

WEATHER_API_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=12.9716"
    "&longitude=77.5946"
    "&current=temperature_2m,relative_humidity_2m"
)

WEATHER_TIMEOUT = 10

latest_weather = {
    "temperature": None,
    "humidity": None,
    "source": "Open-Meteo",
    "status": "WAITING",
    "timestamp": None,
}


# =========================================================
# FETCH WEATHER
# =========================================================

def fetch_weather_data():
    global latest_weather
    try:
        response = requests.get(WEATHER_API_URL, timeout=WEATHER_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        
        current = payload.get("current", {})
        temperature = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        
        if temperature is None or humidity is None:
            raise ValueError("Open-Meteo response did not contain required values")
        
        now = datetime.now(timezone.utc)
        weather = {
            "temperature": float(temperature),
            "humidity": float(humidity),
            "source": "Open-Meteo",
            "status": "ONLINE",
            "timestamp": now.isoformat(),
        }
        
        with lock:
            latest_weather = weather
            
        print()
        print("-" * 65)
        print("OPEN-METEO WEATHER DATA")
        print("-" * 65)
        print(f"Temperature : {weather['temperature']} °C")
        print(f"Humidity    : {weather['humidity']} %")
        print(f"Source      : {weather['source']}")
        print("-" * 65)
        
        return weather
    except Exception as error:
        print()
        print("[WARNING] Open-Meteo weather request failed:")
        print(f"  {error}")
        with lock:
            latest_weather["status"] = "ERROR"
        return dict(latest_weather)


# =========================================================
# LATEST ESP32 TELEMETRY
# =========================================================

latest_telemetry = {
    "device": "ESP32-01",
    "temperature": None,
    "humidity": None,
    "voltage": None,
    "wifi_rssi": None,
    "messages_per_second": None,
    "bytes_per_second": None,
    "active_connections": None,
    "timestamp": None,
    "source": "REAL ESP32",
    "status": "WAITING",
}

# The ESP32 firmware does not send a separate health heartbeat.  Fresh receipt
# of a valid telemetry packet is therefore the only basis for ONLINE status.
TELEMETRY_STALE_SECONDS = max(1, int(os.environ.get("TELEMETRY_STALE_SECONDS", "30")))


def telemetry_with_freshness(telemetry):
    """Return a copy with an honest status derived from receipt time."""
    result = dict(telemetry)
    timestamp = result.get("timestamp")
    if not timestamp:
        result["status"] = "WAITING"
        result["age_seconds"] = None
        return result
    try:
        received_at = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if received_at.tzinfo is None:
            raise ValueError("timestamp has no timezone")
        age_seconds = max(0.0, (datetime.now(timezone.utc) - received_at).total_seconds())
    except (TypeError, ValueError):
        result["status"] = "INVALID_TIMESTAMP"
        result["age_seconds"] = None
        return result

    result["age_seconds"] = round(age_seconds, 3)
    result["status"] = "ONLINE" if age_seconds <= TELEMETRY_STALE_SECONDS else "STALE"
    return result


# =========================================================
# CURRENT STATE
# =========================================================

current_state = {
    "device": "ESP32-01",
    "source": "DIGITAL TWIN",
    "scenario": "NORMAL",
    "scenario_label": "Normal Traffic",
    "prediction": "UNAVAILABLE",
    "attack_probability": None,
    "confidence_score": None,
    "risk_level": "UNKNOWN",
    "model_name": "3-Model Ensemble",
    "digital_twin": "ACTIVE",
    "device_status": "WAITING",
    "ensemble_score": None,
    "attack_votes": None,
    "total_models": None,
    "models": None,
    "telemetry": latest_telemetry,
    "last_update": None,
}


# =========================================================
# RUN ENSEMBLE DETECTION
# =========================================================

def run_detection():
    if not MODEL_READY or predict_three_models is None:
        raise RuntimeError(MODEL_ERROR or "ML models are not ready")
    # -----------------------------------------------------
    # Get Digital Twin sequence
    # -----------------------------------------------------
    sequence = twin.get_next_sequence()

    # -----------------------------------------------------
    # Actual ML inference
    # -----------------------------------------------------
    result = predict_three_models(sequence)

    # -----------------------------------------------------
    # Scenario information
    # -----------------------------------------------------
    scenario_info = twin.get_scenario_info()

    # -----------------------------------------------------
    # Ensemble
    # -----------------------------------------------------
    ensemble = result["ensemble"]

    # -----------------------------------------------------
    # Final prediction
    # -----------------------------------------------------
    prediction = ensemble["classification"]

    # -----------------------------------------------------
    # Ensemble score
    # -----------------------------------------------------
    ensemble_score = float(ensemble["ensemble_score"])

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------
    if prediction == "BENIGN":
        confidence_score = 1.0 - (ensemble_score / 100.0)
    else:
        confidence_score = ensemble_score / 100.0
        
    confidence_score = max(0.0, min(1.0, confidence_score))

    # -----------------------------------------------------
    # Timestamp
    # -----------------------------------------------------
    now = datetime.now(timezone.utc)

    # -----------------------------------------------------
    # Build event
    # -----------------------------------------------------
    event = {
        "id": len(twin.get_history()) + 1,
        "created_at": int(now.timestamp() * 1000),
        "timestamp": now.isoformat(),
        "device": twin.current_target,
        "source": "DIGITAL TWIN",
        "source_ip": twin.current_target,
        "destination_ip": "DIGITAL-TWIN",
        "protocol": "Wi-Fi",
        "device_status": twin.devices[twin.current_target]["status"],
        "digital_twin": "ACTIVE",
        "scenario": scenario_info["scenario"],
        "scenario_label": scenario_info["label"],
        "attack_type": scenario_info["attack_type"],
        "expected": scenario_info["expected"],
        "prediction": prediction,
        "confidence_score": round(confidence_score, 6),
        "attack_probability": round(ensemble_score / 100.0, 6),
        "risk_level": ensemble["risk_level"],
        "model_name": "3-Model Ensemble",
        "ensemble_score": ensemble_score,
        "attack_votes": ensemble["attack_votes"],
        "total_models": ensemble["total_models"],
        "models": result,
        "telemetry": telemetry_with_freshness(latest_telemetry),
        "last_update": now.isoformat(),
    }

    # -----------------------------------------------------
    # Save event
    # -----------------------------------------------------
    twin.history.insert(0, event)
    twin.history = twin.history[:100]

    # -----------------------------------------------------
    # Update current state
    # -----------------------------------------------------
    current_state.update(event)

    return event


# =========================================================
# HEALTH
# =========================================================

@app.route("/api/health", methods=["GET"])
def health():
    payload = {
        "status": "READY" if MODEL_READY else "DEGRADED",
        "service": "Threat Sentinel ML Server",
        "model": "3-Model Ensemble",
        "models": [
            "CNN-LSTM",
            "Random Forest",
            "Autoencoder",
        ],
        "weights": {
            "cnn_lstm": "50%",
            "random_forest": "25%",
            "autoencoder": "25%",
        },
        "digital_twin": "ACTIVE",
        "model_ready": MODEL_READY,
        "model_error": MODEL_ERROR if not MODEL_READY else None,
        "real_device": "ESP32-01",
        "temperature_source": "Open-Meteo",
        "humidity_source": "Open-Meteo",
        "synthetic_scenarios": len(SYNTHETIC_SCENARIOS),
    }
    return jsonify(payload), (200 if MODEL_READY else 503)


# =========================================================
# DIGITAL TWIN DEVICES
# =========================================================

@app.route("/api/digital-twin/devices", methods=["GET"])
def devices():
    return jsonify({
        "devices": twin.get_devices(),
    })


# =========================================================
# CURRENT DIGITAL TWIN STATE
# =========================================================

@app.route("/api/digital-twin/status", methods=["GET"])
def digital_twin_status():
    with lock:
        current_state["telemetry"] = telemetry_with_freshness(latest_telemetry)
    return jsonify({
        "status": current_state,
    })


# =========================================================
# SET TARGET DEVICE
# =========================================================

@app.route("/api/digital-twin/device", methods=["POST"])
def set_device():
    data = request.get_json(silent=True) or {}
    device = data.get("device")

    if not device:
        return jsonify({
            "success": False,
            "error": "device is required",
        }), 400

    try:
        with lock:
            twin.set_target(device)
            current_state["device"] = device
            current_state["last_update"] = datetime.now(timezone.utc).isoformat()
            
        return jsonify({
            "success": True,
            "device": device,
        })
    except ValueError as error:
        return jsonify({
            "success": False,
            "error": str(error),
        }), 400


# =========================================================
# SET SCENARIO
# =========================================================

@app.route("/api/digital-twin/scenario", methods=["POST"])
def set_scenario():
    data = request.get_json(silent=True) or {}
    scenario = data.get("scenario")

    if not scenario:
        return jsonify({
            "success": False,
            "error": "scenario is required",
        }), 400

    scenario = scenario.upper()

    try:
        # Resolve synthetic scenario
        base_scenario, info = resolve_scenario(scenario)
        with lock:
            twin.set_scenario(base_scenario)
            current_state["scenario"] = info["scenario"]
            current_state["scenario_label"] = info["label"]
            current_state["last_update"] = datetime.now(timezone.utc).isoformat()
            
        return jsonify({
            "success": True,
            "scenario": info,
        })
    except ValueError as error:
        return jsonify({
            "success": False,
            "error": str(error),
        }), 400


# =========================================================
# DETECT
# =========================================================

@app.route("/api/digital-twin/detect", methods=["POST"])
def detect():
    try:
        with lock:
            result = run_detection()
        return jsonify({
            "success": True,
            "result": result,
        })
    except Exception as error:
        return jsonify({
            "success": False,
            "error": str(error),
        }), 500


# =========================================================
# START ATTACK / TEST SCENARIO
# =========================================================

@app.route("/api/digital-twin/attack", methods=["POST"])
def start_attack():
    data = request.get_json(silent=True) or {}
    scenario = data.get("scenario", "GAFGYT_COMBO").upper().strip()
    device = data.get("device", "ESP32-01")

    try:
        # -------------------------------------------------
        # Resolve scenario
        # -------------------------------------------------
        base_scenario, scenario_info = resolve_scenario(scenario)

        # -------------------------------------------------
        # Configure Digital Twin
        # -------------------------------------------------
        with lock:
            twin.set_target(device)
            twin.set_scenario(base_scenario)
            
            # -------------------------------------------------
            # ACTUAL ML PREDICTION
            # -------------------------------------------------
            result = run_detection()

        # -------------------------------------------------
        # Add synthetic scenario metadata
        # -------------------------------------------------
        if scenario_info.get("synthetic", False):
            result["scenario"] = scenario_info["scenario"]
            result["scenario_label"] = scenario_info["label"]
            result["attack_type"] = scenario_info["attack_type"]
            result["expected"] = scenario_info["expected"]
            result["synthetic_test"] = True
            result["base_model_scenario"] = scenario_info["base_scenario"]
        else:
            result["synthetic_test"] = False
            
        return jsonify({
            "success": True,
            "attack_started": scenario != "NORMAL",
            "scenario": scenario,
            "scenario_label": scenario_info["label"],
            "synthetic_test": scenario_info.get("synthetic", False),
            "result": result,
        })
    except ValueError as error:
        return jsonify({
            "success": False,
            "error": str(error),
        }), 400
    except Exception as error:
        return jsonify({
            "success": False,
            "error": str(error),
        }), 500


# =========================================================
# NORMAL MODE
# =========================================================

@app.route("/api/digital-twin/normal", methods=["POST"])
def normal_mode():
    data = request.get_json(silent=True) or {}
    device = data.get("device", "ESP32-01")

    try:
        with lock:
            twin.set_target(device)
            twin.set_scenario("NORMAL")
            result = run_detection()
            
        return jsonify({
            "success": True,
            "attack_started": False,
            "result": result,
        })
    except Exception as error:
        return jsonify({
            "success": False,
            "error": str(error),
        }), 500


# =========================================================
# ESP32 TELEMETRY - RECEIVE
# =========================================================

def parse_numeric(data, field, default=None):
    """Parse an optional numeric telemetry field safely."""
    value = data.get(field, default)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numeric value for '{field}': {value}")
    if not (-float("inf") < number < float("inf")):
        raise ValueError(f"Invalid numeric value for '{field}': {value}")
    return number

@app.route("/api/telemetry", methods=["POST"])
def receive_telemetry():
    global latest_telemetry

    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data:
        return jsonify({
            "success": False,
            "error": "A non-empty JSON telemetry payload is required.",
            "expected_content_type": "application/json",
        }), 400

    device = str(data.get("device", "ESP32-01")).strip()
    if not device:
        return jsonify({"success": False, "error": "Device ID cannot be empty."}), 400

    if device not in twin.devices:
        return jsonify({
            "success": False,
            "error": f"Unknown device: {device}",
            "registered_devices": list(twin.devices.keys()),
        }), 400

    try:
        wifi_rssi = parse_numeric(data, "wifi_rssi")
        voltage = parse_numeric(data, "voltage")
        messages_per_second = parse_numeric(data, "messages_per_second", 0)
        bytes_per_second = parse_numeric(data, "bytes_per_second", 0)
        active_connections = parse_numeric(data, "active_connections", 0)
        if messages_per_second < 0:
            raise ValueError("messages_per_second cannot be negative")
        if bytes_per_second < 0:
            raise ValueError("bytes_per_second cannot be negative")
        if active_connections < 0:
            raise ValueError("active_connections cannot be negative")
    except ValueError as error:
        return jsonify({"success": False, "error": str(error)}), 400

    # Telemetry acknowledgement must never depend on an unrelated weather API.
    # The dashboard fetches weather independently and labels its source.
    with lock:
        weather = dict(latest_weather)

    now = datetime.now(timezone.utc)
    telemetry_data = {
        "device": device,
        # Environmental values come from Open-Meteo, not ESP32 sensors.
        "temperature": weather.get("temperature"),
        "humidity": weather.get("humidity"),
        # These values are supplied by the ESP32 firmware.
        "voltage": voltage,
        "wifi_rssi": wifi_rssi,
        "messages_per_second": messages_per_second,
        "bytes_per_second": bytes_per_second,
        "active_connections": active_connections,
        "timestamp": now.isoformat(),
        "source": "REAL ESP32 + OPEN-METEO",
        "weather_source": weather.get("source"),
        "weather_status": weather.get("status"),
        "status": "ONLINE",
    }

    with lock:
        latest_telemetry = telemetry_data
        current_state["telemetry"] = dict(telemetry_data)
        current_state["device"] = device
        current_state["device_status"] = "ONLINE"
        current_state["last_update"] = now.isoformat()

    print("\n" + "=" * 65)
    print("ESP32 TELEMETRY RECEIVED")
    print("=" * 65)
    print(f"Device          : {device}")
    print(f"Wi-Fi RSSI      : {wifi_rssi}")
    print(f"Voltage         : {voltage}")
    print(f"Messages/sec    : {messages_per_second}")
    print(f"Bytes/sec       : {bytes_per_second}")
    print(f"Active conn.    : {active_connections}")
    print(f"Temperature     : {telemetry_data['temperature']}")
    print(f"Humidity        : {telemetry_data['humidity']}")
    print(f"Received at     : {telemetry_data['timestamp']}")
    print("=" * 65)

    return jsonify({
        "success": True,
        "message": "ESP32 telemetry received",
        "telemetry": telemetry_data,
    }), 200


# =========================================================
# ESP32 TELEMETRY - LATEST
# =========================================================

@app.route("/api/telemetry", methods=["GET"])
def get_telemetry():
    with lock:
        telemetry = telemetry_with_freshness(latest_telemetry)
    return jsonify({
        "success": True,
        "telemetry": telemetry,
    })


# =========================================================
# WEATHER / ENVIRONMENT
# =========================================================

@app.route("/api/weather", methods=["GET"])
def get_weather():
    with lock:
        weather = dict(latest_weather)

    if weather.get("temperature") is None or weather.get("humidity") is None:
        weather = fetch_weather_data()

    return jsonify({
        "success": True,
        "weather": weather,
    })


# =========================================================
# PREDICTION HISTORY
# =========================================================

@app.route("/api/predictions", methods=["GET"])
def predictions():
    history = twin.get_history()
    return jsonify({
        "predictions": history,
    })


# =========================================================
# LATEST PREDICTION
# =========================================================

@app.route("/api/predictions/latest", methods=["GET"])
def latest_prediction():
    with lock:
        current_state["telemetry"] = telemetry_with_freshness(latest_telemetry)
    return jsonify(current_state)


# =========================================================
# AVAILABLE SCENARIOS
# =========================================================

@app.route("/api/digital-twin/scenarios", methods=["GET"])
def scenarios():
    scenario_list = [
        # -------------------------------------------------
        # Existing validated scenarios
        # -------------------------------------------------
        {
            "id": "NORMAL",
            "name": "Normal Traffic",
            "type": "BENIGN",
            "synthetic": False,
        },
        {
            "id": "GAFGYT_COMBO",
            "name": "Gafgyt Combo",
            "type": "ATTACK",
            "synthetic": False,
        },
        {
            "id": "MIRAI_ACK",
            "name": "Mirai ACK",
            "type": "ATTACK",
            "synthetic": False,
        },
        {
            "id": "MIRAI_SCAN",
            "name": "Mirai Scan",
            "type": "ATTACK",
            "synthetic": False,
        },
        {
            "id": "MIRAI_SYN",
            "name": "Mirai SYN",
            "type": "ATTACK",
            "synthetic": False,
        },
        {
            "id": "MIRAI_UDP",
            "name": "Mirai UDP",
            "type": "ATTACK",
            "synthetic": False,
        },
        {
            "id": "MIRAI_UDPPLAIN",
            "name": "Mirai UDP Plain",
            "type": "ATTACK",
            "synthetic": False,
        },
    ]

    # -----------------------------------------------------
    # Synthetic scenarios
    # -----------------------------------------------------
    for scenario_id, info in SYNTHETIC_SCENARIOS.items():
        scenario_list.append({
            "id": scenario_id,
            "name": info["name"],
            "type": info["type"],
            "attack_type": info["attack_type"],
            "expected": info["expected"],
            "synthetic": True,
            "base_scenario": info["base_scenario"],
        })

    return jsonify({
        "success": True,
        "count": len(scenario_list),
        "scenarios": scenario_list,
    })


# =========================================================
# SERVER
# =========================================================

if __name__ == "__main__":
    print()
    print("=" * 65)
    print("THREAT SENTINEL ML API")
    print("=" * 65)
    print("Digital Twin : ACTIVE")
    print("Models       : CNN-LSTM + Random Forest + Autoencoder")
    print("Weights      : 50% + 25% + 25%")
    print("Real Device  : ESP32-01")
    print("API          : http://127.0.0.1:5000")
    print("LAN API      : http://192.168.31.139:5000")
    print("Telemetry    : /api/telemetry")
    print(f"Scenarios    : {7 + len(SYNTHETIC_SCENARIOS)}")
    print("=" * 65)
    print()
    
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )
