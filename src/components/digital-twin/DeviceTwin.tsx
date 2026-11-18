import type { DeviceTwinState } from "./types";

interface DeviceTwinProps {
  device: DeviceTwinState;
}

function getRiskLabel(score: number) {
  if (score >= 0.8) return "CRITICAL";
  if (score >= 0.6) return "HIGH";
  if (score >= 0.3) return "MEDIUM";
  return "LOW";
}

function getRiskClass(score: number) {
  if (score >= 0.8) return "text-danger";
  if (score >= 0.6) return "text-warning";
  if (score >= 0.3) return "text-yellow-500";
  return "text-success";
}

export function DeviceTwin({ device }: DeviceTwinProps) {
  const riskLabel = getRiskLabel(device.anomalyScore);

  return (
    <section className="rounded-xl border border-border bg-card p-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <div
              className={`h-3 w-3 rounded-full ${
                device.status === "ONLINE"
                  ? "bg-success"
                  : device.status === "OFFLINE"
                    ? "bg-muted-foreground"
                    : "bg-danger"
              }`}
            />

            <h2 className="text-lg font-semibold">
              {device.deviceId}
            </h2>
          </div>

          <p className="text-sm text-muted-foreground mt-1">
            ESP32 Digital Twin
          </p>
        </div>

        <div className="text-left md:text-right">
          <p className="text-xs text-muted-foreground">
            Device Status
          </p>

          <p className="font-semibold mt-1">
            {device.status}
          </p>
        </div>
      </div>

      {/* Device information */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-3">
        <Metric
          label="IP Address"
          value={device.ipAddress}
        />

        <Metric
          label="Firmware"
          value={device.firmwareVersion}
        />

        <Metric
          label="Wi-Fi RSSI"
          value={`${device.wifiRssi} dBm`}
        />

        <Metric
          label="Uptime"
          value={formatUptime(device.uptimeSeconds)}
        />
      </div>

      {/* Sensor State */}
      <div className="mt-6">
        <SectionTitle title="Physical State" />

        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mt-3">
          <Metric
            label="Temperature"
            value={`${device.temperature.toFixed(1)} °C`}
          />

          <Metric
            label="Humidity"
            value={`${device.humidity.toFixed(1)} %`}
          />

          <Metric
            label="Voltage"
            value={`${device.voltage.toFixed(2)} V`}
          />
        </div>
      </div>

      {/* Network State */}
      <div className="mt-6">
        <SectionTitle title="Network State" />

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
          <Metric
            label="MQTT"
            value={device.mqttConnected ? "CONNECTED" : "DISCONNECTED"}
          />

          <Metric
            label="Messages"
            value={`${device.messagesPerSecond.toFixed(1)} /s`}
          />

          <Metric
            label="Data Rate"
            value={`${device.bytesPerSecond} B/s`}
          />

          <Metric
            label="Last Seen"
            value={formatLastSeen(device.lastSeen)}
          />
        </div>
      </div>

      {/* Security State */}
      <div className="mt-6">
        <SectionTitle title="Security State" />

        <div className="mt-3 rounded-lg border border-border bg-background p-4">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <p className="text-xs text-muted-foreground">
                Anomaly Score
              </p>

              <p
                className={`text-2xl font-bold mt-1 ${getRiskClass(
                  device.anomalyScore
                )}`}
              >
                {(device.anomalyScore * 100).toFixed(0)}%
              </p>
            </div>

            <div>
              <p className="text-xs text-muted-foreground">
                Risk Level
              </p>

              <p
                className={`font-semibold mt-1 ${getRiskClass(
                  device.anomalyScore
                )}`}
              >
                {riskLabel}
              </p>
            </div>

            <div>
              <p className="text-xs text-muted-foreground">
                Attack Detection
              </p>

              <p
                className={`font-semibold mt-1 ${
                  device.attackDetected
                    ? "text-danger"
                    : "text-success"
                }`}
              >
                {device.attackDetected
                  ? "POTENTIAL ATTACK"
                  : "NO ATTACK DETECTED"}
              </p>
            </div>
          </div>

          {/* Anomaly progress bar */}
          <div className="mt-5">
            <div className="h-2 rounded-full bg-muted overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${
                  device.anomalyScore >= 0.8
                    ? "bg-danger"
                    : device.anomalyScore >= 0.6
                      ? "bg-warning"
                      : device.anomalyScore >= 0.3
                        ? "bg-yellow-500"
                        : "bg-success"
                }`}
                style={{
                  width: `${Math.min(
                    Math.max(device.anomalyScore * 100, 0),
                    100
                  )}%`,
                }}
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function Metric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg bg-background border border-border p-3">
      <p className="text-xs text-muted-foreground">
        {label}
      </p>

      <p className="text-sm font-medium mt-1 break-words">
        {value}
      </p>
    </div>
  );
}

function SectionTitle({ title }: { title: string }) {
  return (
    <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
      {title}
    </h3>
  );
}

function formatUptime(seconds: number) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }

  return `${minutes}m`;
}

function formatLastSeen(timestamp: string) {
  const date = new Date(timestamp);

  if (Number.isNaN(date.getTime())) {
    return "Unknown";
  }

  return date.toLocaleTimeString();
}