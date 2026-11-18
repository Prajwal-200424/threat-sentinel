import {
  useEffect,
  useState,
} from "react";

import {
  Activity,
  Cpu,
  Shield,
  ShieldAlert,
  Wifi,
  Zap,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Play,
  Square,
} from "lucide-react";

import {
  type AttackType,
  type RiskLevel,
} from "./types";


const ML_SERVER =
  "http://127.0.0.1:5000";


interface DetectionResult {
  id: number;

  device: string;

  prediction:
    | "BENIGN"
    | "ATTACK";

  attack_probability: number;

  confidence_score: number;

  risk_level: RiskLevel;

  attack_type:
    | AttackType
    | null;

  scenario: string;

  scenario_label: string;

  model_name: string;

  digital_twin: string;

  device_status: string;

  protocol: string;

  timestamp: string;

  created_at: number;
}


interface ApiResponse {
  success: boolean;

  attack_started: boolean;

  result: DetectionResult;
}


interface SimulationStatus {
  running: boolean;

  cycles_completed: number;
}


interface DeviceState {
  deviceId: string;

  source: "REAL" | "VIRTUAL";

  status: "ONLINE" | "OFFLINE";

  ipAddress: string;

  temperature: number;

  humidity: number;

  voltage: number;

  wifiRssi: number;

  messagesPerSecond: number;

  bytesPerSecond: number;

  activeConnections: number;
}


const DEVICES: DeviceState[] = [

  {
    deviceId: "ESP32-01",
    source: "REAL",
    status: "ONLINE",
    ipAddress: "ESP32-01",
    temperature: 28.4,
    humidity: 54,
    voltage: 3.29,
    wifiRssi: -48,
    messagesPerSecond: 12,
    bytesPerSecond: 4200,
    activeConnections: 2,
  },

  {
    deviceId: "ESP32-02",
    source: "VIRTUAL",
    status: "ONLINE",
    ipAddress: "ESP32-02",
    temperature: 27.9,
    humidity: 51,
    voltage: 3.30,
    wifiRssi: -52,
    messagesPerSecond: 10,
    bytesPerSecond: 3500,
    activeConnections: 2,
  },

  {
    deviceId: "ESP32-03",
    source: "VIRTUAL",
    status: "ONLINE",
    ipAddress: "ESP32-03",
    temperature: 29.1,
    humidity: 57,
    voltage: 3.28,
    wifiRssi: -55,
    messagesPerSecond: 9,
    bytesPerSecond: 3100,
    activeConnections: 1,
  },

];


const ATTACK_SCENARIOS = [

  {
    value: "GAFGYT_COMBO",
    label: "Gafgyt Combo",
  },

  {
    value: "MIRAI_ACK",
    label: "Mirai ACK",
  },

  {
    value: "MIRAI_SCAN",
    label: "Mirai Scan",
  },

  {
    value: "MIRAI_SYN",
    label: "Mirai SYN",
  },

  {
    value: "MIRAI_UDP",
    label: "Mirai UDP",
  },

  {
    value: "MIRAI_UDPPLAIN",
    label: "Mirai UDP Plain",
  },

];


export function DigitalTwin() {

  // =====================================================
  // DEVICE
  // =====================================================

  const [
    selectedDevice,
    setSelectedDevice,
  ] = useState(
    "ESP32-01"
  );


  // =====================================================
  // DETECTION
  // =====================================================

  const [
    detection,
    setDetection,
  ] = useState<DetectionResult | null>(
    null
  );


  // =====================================================
  // SCENARIO
  // =====================================================

  const [
    selectedScenario,
    setSelectedScenario,
  ] = useState(
    "GAFGYT_COMBO"
  );


  // =====================================================
  // LOADING
  // =====================================================

  const [
    isLoading,
    setIsLoading,
  ] = useState(false);


  // =====================================================
  // ERROR
  // =====================================================

  const [
    error,
    setError,
  ] = useState<string | null>(
    null
  );


  // =====================================================
  // SERVER STATUS
  // =====================================================

  const [
    serverOnline,
    setServerOnline,
  ] = useState(false);


  const [
    simulation,
    setSimulation,
  ] = useState<SimulationStatus>({
    running: false,
    cycles_completed: 0,
  });


  // =====================================================
  // CHECK ML SERVER
  // =====================================================

  useEffect(() => {

    let cancelled = false;


    async function checkServer() {

      try {

        const response =
          await fetch(
            `${ML_SERVER}/api/health`,
            {
              cache: "no-store",
            }
          );


        if (!response.ok) {

          throw new Error(
            "ML server unavailable"
          );

        }


        if (!cancelled) {

          setServerOnline(true);

          const simulationResponse = await fetch(
            `${ML_SERVER}/api/digital-twin/simulation`,
            { cache: "no-store" },
          );

          if (simulationResponse.ok) {
            const simulationData = await simulationResponse.json();
            setSimulation(simulationData.simulation);
          }

        }

      }

      catch {

        if (!cancelled) {

          setServerOnline(false);

        }

      }

    }


    checkServer();


    const interval =
      setInterval(
        checkServer,
        5000
      );


    return () => {

      cancelled = true;

      clearInterval(
        interval
      );

    };

  }, []);


  async function toggleSimulation() {

    setError(null);

    try {
      const endpoint = simulation.running
        ? "/api/digital-twin/simulation/stop"
        : "/api/digital-twin/simulation/start";
      const body = simulation.running
        ? undefined
        : JSON.stringify({
            device: selectedDevice,
            scenarios: ["NORMAL", selectedScenario],
            interval_seconds: 5,
          });

      const response = await fetch(`${ML_SERVER}${endpoint}`, {
        method: "POST",
        headers: body
          ? { "Content-Type": "application/json" }
          : undefined,
        body,
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error ?? "Simulation request failed");
      }

      setSimulation(data.simulation);
      setServerOnline(true);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to control Digital Twin simulation",
      );
    }
  }


  // =====================================================
  // RUN NORMAL TRAFFIC
  // =====================================================

  async function runNormal() {

    if (isLoading) {
      return;
    }


    setIsLoading(true);

    setError(null);


    try {

      const response =
        await fetch(
          `${ML_SERVER}/api/digital-twin/normal`,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body: JSON.stringify({
              device:
                selectedDevice,
            }),

          }
        );


      if (!response.ok) {

        throw new Error(
          "Normal traffic request failed"
        );

      }


      const data:
        ApiResponse =
        await response.json();


      if (!data.success) {

        throw new Error(
          "ML server returned an unsuccessful response"
        );

      }


      setDetection(
        data.result
      );

      setServerOnline(true);

    }

    catch (err) {

      setError(
        err instanceof Error
          ? err.message
          : "Unable to contact ML server"
      );

    }

    finally {

      setIsLoading(false);

    }

  }


  // =====================================================
  // RUN ATTACK
  // =====================================================

  async function runAttack() {

    if (isLoading) {
      return;
    }


    setIsLoading(true);

    setError(null);


    try {

      const response =
        await fetch(
          `${ML_SERVER}/api/digital-twin/attack`,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body: JSON.stringify({

              device:
                selectedDevice,

              scenario:
                selectedScenario,

            }),

          }
        );


      if (!response.ok) {

        throw new Error(
          "Attack request failed"
        );

      }


      const data:
        ApiResponse =
        await response.json();


      if (!data.success) {

        throw new Error(
          "ML server returned an unsuccessful response"
        );

      }


      setDetection(
        data.result
      );

      setServerOnline(true);

    }

    catch (err) {

      setError(
        err instanceof Error
          ? err.message
          : "Unable to contact ML server"
      );

    }

    finally {

      setIsLoading(false);

    }

  }


  // =====================================================
  // CURRENT DEVICE
  // =====================================================

  const device =
    DEVICES.find(
      (item) =>
        item.deviceId ===
        selectedDevice
    ) ??
    DEVICES[0];


  // =====================================================
  // DISPLAY VALUES
  // =====================================================

  const isAttack =
    detection?.prediction ===
    "ATTACK";


  const confidence =
    detection
      ? detection.confidence_score * 100
      : 0;


  const probability =
    detection
      ? detection.attack_probability * 100
      : 0;


  const risk =
    detection?.risk_level ??
    "LOW";


  // =====================================================
  // RISK DISPLAY
  // =====================================================

  const riskClass =

    risk === "CRITICAL"
      ? "text-danger border-danger/40 bg-danger/10"

      : risk === "HIGH"
        ? "text-orange-400 border-orange-400/40 bg-orange-400/10"

        : risk === "MEDIUM"
          ? "text-warning border-warning/40 bg-warning/10"

          : "text-safe border-safe/40 bg-safe/10";


  return (

    <section
      className="
        rounded-xl
        border
        border-border/60
        bg-card/60
        backdrop-blur-md
        p-5
        space-y-5
      "
    >

      {/* ================================================= */}
      {/* HEADER                                            */}
      {/* ================================================= */}

      <div
        className="
          flex
          flex-col
          md:flex-row
          md:items-center
          md:justify-between
          gap-4
        "
      >

        <div>

          <div className="
            flex
            items-center
            gap-2
          ">

            <Cpu
              className="
                h-5
                w-5
                text-primary
              "
            />

            <h2 className="
              text-lg
              font-semibold
            ">
              IoT Digital Twin
            </h2>

          </div>


          <p className="
            text-xs
            text-muted-foreground
            mt-1
          ">
            Background device simulation
            connected to the CNN-LSTM
            detection engine.
          </p>

        </div>


        {/* ML SERVER STATUS */}

        <div className="
          flex
          items-center
          gap-2
          text-xs
          font-mono
        ">

          <span
            className={`
              h-2
              w-2
              rounded-full
              ${
                serverOnline
                  ? "bg-safe"
                  : "bg-danger"
              }
            `}
          />

          {serverOnline
            ? "ML SERVER ONLINE"
            : "ML SERVER OFFLINE"}

        </div>

      </div>


      {/* ================================================= */}
      {/* DEVICE SELECTOR                                   */}
      {/* ================================================= */}

      <div className="
        grid
        grid-cols-1
        md:grid-cols-3
        gap-3
      ">

        {DEVICES.map(
          (item) => (

            <button
              key={
                item.deviceId
              }

              type="button"

              onClick={() =>
                setSelectedDevice(
                  item.deviceId
                )
              }

              className={`
                rounded-lg
                border
                p-3
                text-left
                transition
                ${
                  selectedDevice ===
                  item.deviceId

                    ? "border-primary bg-primary/10"

                    : "border-border/60 hover:border-primary/50"
                }
              `}
            >

              <div className="
                flex
                items-center
                justify-between
              ">

                <span className="
                  font-mono
                  text-sm
                  font-semibold
                ">
                  {item.deviceId}
                </span>


                <span className="
                  text-[10px]
                  uppercase
                  text-muted-foreground
                ">
                  {item.source}
                </span>

              </div>


              <div className="
                mt-2
                flex
                items-center
                gap-2
                text-xs
                text-safe
              ">

                <Wifi
                  className="
                    h-3
                    w-3
                  "
                />

                {item.status}

              </div>

            </button>

          )
        )}

      </div>


      {/* ================================================= */}
      {/* DEVICE TELEMETRY                                  */}
      {/* ================================================= */}

      <div className="
        grid
        grid-cols-2
        md:grid-cols-4
        gap-3
      ">

        <Telemetry
          icon={
            <Activity
              className="h-4 w-4"
            />
          }
          label="Messages/sec"
          value={
            device.messagesPerSecond
          }
        />


        <Telemetry
          icon={
            <Zap
              className="h-4 w-4"
            />
          }
          label="Bytes/sec"
          value={
            device.bytesPerSecond
          }
        />


        <Telemetry
          icon={
            <Cpu
              className="h-4 w-4"
            />
          }
          label="Temperature"
          value={
            `${device.temperature}°C`
          }
        />


        <Telemetry
          icon={
            <Wifi
              className="h-4 w-4"
            />
          }
          label="Wi-Fi RSSI"
          value={
            `${device.wifiRssi} dBm`
          }
        />

      </div>


      {/* ================================================= */}
      {/* CONTINUOUS SIMULATION                              */}
      {/* ================================================= */}

      <div className="
        flex
        flex-col
        gap-3
        rounded-lg
        border
        border-primary/30
        bg-primary/5
        p-4
        md:flex-row
        md:items-center
        md:justify-between
      ">

        <div>
          <div className="flex items-center gap-2 text-sm font-semibold">
            {simulation.running ? (
              <Activity className="h-4 w-4 animate-pulse text-safe" />
            ) : (
              <Play className="h-4 w-4 text-primary" />
            )}
            Continuous Digital Twin
          </div>

          <p className="mt-1 text-xs text-muted-foreground">
            {simulation.running
              ? `${simulation.cycles_completed} predictions generated`
              : "Generate recurring validated traffic windows"}
          </p>
        </div>

        <button
          type="button"
          disabled={!serverOnline}
          onClick={toggleSimulation}
          className="
            flex
            items-center
            justify-center
            gap-2
            rounded-lg
            border
            border-primary/40
            bg-primary/10
            px-4
            py-2
            text-sm
            font-semibold
            text-primary
            transition
            hover:bg-primary/20
            disabled:cursor-not-allowed
            disabled:opacity-50
          "
        >
          {simulation.running ? (
            <Square className="h-4 w-4" />
          ) : (
            <Play className="h-4 w-4" />
          )}
          {simulation.running ? "STOP SIMULATION" : "START SIMULATION"}
        </button>

      </div>


      {/* ================================================= */}
      {/* ATTACK CONTROL                                    */}
      {/* ================================================= */}

      <div className="
        rounded-lg
        border
        border-border/50
        bg-background/40
        p-4
      ">

        <div className="
          flex
          items-center
          gap-2
          mb-3
        ">

          <ShieldAlert
            className="
              h-4
              w-4
              text-danger
            "
          />

          <span className="
            text-sm
            font-semibold
          ">
            Digital Twin Scenario Control
          </span>

        </div>


        <div className="
          flex
          flex-col
          md:flex-row
          gap-3
        ">

          {/* NORMAL */}

          <button
            type="button"

            disabled={
              isLoading ||
              !serverOnline
            }

            onClick={
              runNormal
            }

            className="
              flex-1
              rounded-lg
              border
              border-safe/40
              bg-safe/10
              px-4
              py-3
              text-sm
              font-semibold
              text-safe
              transition
              hover:bg-safe/20
              disabled:opacity-50
              disabled:cursor-not-allowed
            "
          >

            {isLoading ? (

              <Loader2
                className="
                  mx-auto
                  h-4
                  w-4
                  animate-spin
                "
              />

            ) : (

              <span className="
                flex
                items-center
                justify-center
                gap-2
              ">

                <CheckCircle2
                  className="
                    h-4
                    w-4
                  "
                />

                RETURN TO NORMAL

              </span>

            )}

          </button>


          {/* ATTACK */}

          <div className="
            flex
            flex-1
            gap-2
          ">

            <select
              value={
                selectedScenario
              }

              onChange={(event) =>
                setSelectedScenario(
                  event.target.value
                )
              }

              disabled={
                isLoading ||
                !serverOnline
              }

              className="
                w-full
                rounded-lg
                border
                border-border/60
                bg-background
                px-3
                text-sm
                outline-none
              "
            >

              {ATTACK_SCENARIOS.map(
                (scenario) => (

                  <option
                    key={
                      scenario.value
                    }

                    value={
                      scenario.value
                    }
                  >
                    {scenario.label}
                  </option>

                )
              )}

            </select>


            <button
              type="button"

              disabled={
                isLoading ||
                !serverOnline
              }

              onClick={
                runAttack
              }

              className="
                whitespace-nowrap
                rounded-lg
                border
                border-danger/40
                bg-danger/10
                px-4
                py-3
                text-sm
                font-semibold
                text-danger
                transition
                hover:bg-danger/20
                disabled:opacity-50
                disabled:cursor-not-allowed
              "
            >

              <span className="
                flex
                items-center
                gap-2
              ">

                <AlertTriangle
                  className="
                    h-4
                    w-4
                  "
                />

                START ATTACK

              </span>

            </button>

          </div>

        </div>


        {error && (

          <div className="
            mt-3
            rounded-md
            border
            border-danger/40
            bg-danger/10
            px-3
            py-2
            text-xs
            text-danger
          ">

            {error}

          </div>

        )}

      </div>


      {/* ================================================= */}
      {/* REAL ML RESULT                                    */}
      {/* ================================================= */}

      {detection && (

        <div className="
          rounded-lg
          border
          border-border/60
          bg-background/40
          p-4
        ">

          <div className="
            flex
            items-center
            justify-between
            gap-3
            mb-4
          ">

            <div>

              <p className="
                text-xs
                uppercase
                tracking-widest
                text-muted-foreground
              ">
                CNN-LSTM Detection
              </p>

              <p className="
                mt-1
                text-xs
                font-mono
                text-muted-foreground
              ">
                {detection.scenario_label}
              </p>

            </div>


            <div
              className={`
                rounded-full
                border
                px-3
                py-1
                text-xs
                font-semibold
                ${riskClass}
              `}
            >
              {risk}
            </div>

          </div>


          <div className="
            grid
            grid-cols-1
            sm:grid-cols-2
            lg:grid-cols-5
            gap-3
          ">

            <ResultMetric
              label="Prediction"
              value={
                detection.prediction
              }
              danger={
                isAttack
              }
            />


            <ResultMetric
              label="Confidence"
              value={
                `${confidence.toFixed(2)}%`
              }
              danger={
                isAttack
              }
            />


            <ResultMetric
              label="Attack Probability"
              value={
                `${probability.toFixed(2)}%`
              }
              danger={
                probability >= 50
              }
            />


            <ResultMetric
              label="Device"
              value={
                detection.device
              }
            />


            <ResultMetric
              label="Model"
              value={
                detection.model_name
              }
            />

          </div>


          <div className="
            mt-4
            flex
            flex-wrap
            gap-x-6
            gap-y-2
            text-xs
            text-muted-foreground
            font-mono
          ">

            <span>
              Scenario:
              {" "}
              {detection.scenario}
            </span>


            <span>
              Protocol:
              {" "}
              {detection.protocol}
            </span>


            <span>
              Twin:
              {" "}
              {detection.digital_twin}
            </span>


            <span>
              Event ID:
              {" "}
              #{detection.id}
            </span>

          </div>

        </div>

      )}

    </section>

  );

}


// =========================================================
// TELEMETRY COMPONENT
// =========================================================

function Telemetry({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
}) {

  return (

    <div className="
      rounded-lg
      border
      border-border/50
      bg-background/30
      p-3
    ">

      <div className="
        flex
        items-center
        gap-2
        text-muted-foreground
      ">

        {icon}

        <span className="
          text-[11px]
          uppercase
          tracking-wide
        ">
          {label}
        </span>

      </div>


      <p className="
        mt-2
        font-mono
        text-sm
        font-semibold
      ">
        {value}
      </p>

    </div>

  );

}


// =========================================================
// RESULT METRIC
// =========================================================

function ResultMetric({
  label,
  value,
  danger = false,
}: {
  label: string;
  value: string;
  danger?: boolean;
}) {

  return (

    <div className="
      rounded-lg
      border
      border-border/50
      bg-card/50
      p-3
    ">

      <p className="
        text-[10px]
        uppercase
        tracking-widest
        text-muted-foreground
      ">
        {label}
      </p>


      <p
        className={`
          mt-2
          text-sm
          font-bold
          font-mono
          ${
            danger
              ? "text-danger"
              : "text-foreground"
          }
        `}
      >
        {value}
      </p>

    </div>

  );

}