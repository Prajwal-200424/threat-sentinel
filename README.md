# Threat Sentinel

Threat Sentinel is an IoT monitoring dashboard backed by a local Flask ML API.
The deployed ensemble is fixed at CNN-LSTM 50%, Random Forest 25%, and
Autoencoder 25%. The dashboard never treats missing inference as benign.

## Architecture

`ESP32 -> POST /api/telemetry -> Flask receipt store -> React dashboard`

The digital-twin routes use prevalidated feature sequences for controlled ML
demonstrations. They are not ESP32 traffic capture. Weather is fetched and
displayed independently from ESP32 network telemetry, and is labelled
Open-Meteo in the UI.

## Run locally

In one PowerShell terminal:

```powershell
cd ml_server
python app.py
```

In another:

```powershell
npm install
npm run dev
```

The dashboard defaults to `http://127.0.0.1:5000`. To use a LAN-hosted Flask
server, create a local `.env` file (do not commit it):

```text
VITE_API_BASE_URL=http://192.168.x.x:5000
```

Flask accepts browser origins in `CORS_ORIGINS` (comma-separated). Its safe
local default includes Vite's common 5173 and 8080 origins. Device freshness is
controlled by `TELEMETRY_STALE_SECONDS` (default 30); without a fresh packet the
API reports `WAITING` or `STALE`, never `ONLINE`.

## ESP32 telemetry contract

POST JSON to `/api/telemetry`:

```json
{
  "device": "ESP32-01",
  "wifi_rssi": -62,
  "voltage": 3.3,
  "messages_per_second": 4,
  "bytes_per_second": 512,
  "active_connections": 2
}
```

`device` must be a registered device ID. Numeric values are validated; traffic
rates and active connections cannot be negative. Receipt timestamps are created
by Flask, not taken from device input. `GET /api/telemetry` returns the most
recent payload plus a derived `ONLINE`, `STALE`, or `WAITING` status.

## Model artifacts and readiness

The live ML server requires these artifacts in `training/models/`:

- `cnn_lstm.h5`
- `random_forest.pkl`
- `autoencoder.h5`
- `scaler.pkl`

`GET /api/health` returns HTTP 200 and `READY` only if those models load. If an
artifact or runtime is missing it returns HTTP 503 and `DEGRADED`; detection
routes return an explicit error rather than a benign fallback.

## Validation

```powershell
npm run build
npm run lint
python -m unittest ml_server/test_app.py
# The saved Keras artifacts were serialized by Keras 3.15, so the evaluation
# runs in the project virtual environment that provides it:
.venv\Scripts\python.exe training\evaluate_zero_day.py
```

`training/evaluate_zero_day.py` refuses to run on an interpreter whose Keras is
older than 3.15 (`GlorotUniform.__init__() got an unexpected keyword argument
'input_axes'`) and prints the compatible command instead. Do not retrain or
re-export the artifacts to work around that error.

The zero-day evaluation needs the N-BaIoT archive at the configured path in
`training/evaluate_zero_day.py`; it reports only results computed from the
available data. Its family-held-out experiment is not evidence that every
future unseen attack will be detected.

## Current limitations

Physical ESP32 ingestion has not been verified in this repository environment.
The telemetry tests send simulated JSON to `POST /api/telemetry`; they prove the
HTTP contract, validation and freshness rules, not a real device link.

Two artifact sets exist and are not interchangeable. The live server in
`ml_server/ensemble_inference.py` loads `training/models/*.h5`, while
`training/evaluate_zero_day.py` and `thresholds.json` describe the Keras 3.15
artifacts in `models/`. Results computed from one set are not evidence for the
other.

Prediction history is an in-memory digital-twin demonstration history and is
cleared when Flask restarts. It must not be used as an incident record until a
project-approved persistence layer is added. No Raspberry Pi capability,
containment action, firewall change, or automated network isolation is part of
this project.

The system detects malware and anomalies in IoT networks using a Hybrid CNN + LSTM deep learning model. Prediction results are stored in Xano backend using the API below.

Use the local Flask ML API as the live backend data source:

POST / GET API Endpoint:
http://127.0.0.1:5000/api/predictions

The dashboard must fetch and display real-time prediction records from the local Flask ML API.

Build a dark theme Security Operations Center (SOC) dashboard with a professional cybersecurity look.

Required sections:

Top Summary Cards

Total Network Traffic

Safe Traffic

Malware Detected

Threat Level

Live Prediction Section
Display latest prediction from Flask:

Prediction (ATTACK / BENIGN)

Confidence Score

Source IP

Destination IP

Protocol

Timestamp

Status

Graphs

Attack Trend Over Time (Line Chart)

Attack Type Distribution (Pie Chart)

Daily Threat Detection (Bar Chart)

Prediction History Table
Fetch records from Flask and display:

Time

Source IP

Destination IP

Protocol

Prediction

Confidence Score

Status

Real-Time Alert Section
Show red warning alerts when:
Prediction = ATTACK

Example:
⚠ High Risk Malware Detected

Professional Design Requirements

Dark cybersecurity theme

Modern SOC dashboard style

Security monitoring interface

Red alert highlights

Glowing warning cards

Strong professional presentation quality

Final Year Project review ready

Functional Requirements

Auto-refresh live prediction data from Xano

Display newest attacks first

Highlight ATTACK rows in red

Highlight BENIGN rows in green

Real-time dashboard behavior

Clean responsive design

Make the UI visually impressive, professional, and suitable for final-year project presentation with real backend integration.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/a28a2fa3-fe45-4852-92c5-ca2c230672d0).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
