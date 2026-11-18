import type { GatewayTwinState } from "./types";

interface GatewayTwinProps {
  gateway: GatewayTwinState;
}

function getStatusClass(status: string) {
  switch (status) {
    case "ONLINE":
      return "text-success";
    case "SUSPICIOUS":
      return "text-warning";
    case "HIGH_RISK":
    case "POTENTIAL_ATTACK":
      return "text-danger";
    default:
      return "text-muted-foreground";
  }
}

export function GatewayTwin({ gateway }: GatewayTwinProps) {
  return (
    <section className="rounded-xl border border-border bg-card p-6">
      {/* Header */}
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <span
              className={`h-3 w-3 rounded-full ${
                gateway.status === "ONLINE"
                  ? "bg-success"
                  : gateway.status === "OFFLINE"
                    ? "bg-muted-foreground"
                    : "bg-danger"
              }`}
            />

            <h2 className="text-lg font-semibold">
              {gateway.deviceId}
            </h2>
          </div>

          <p className="text-sm text-muted-foreground mt-1">
            Raspberry Pi Edge Gateway Digital Twin
          </p>
        </div>

        <div className="text-right">
          <p className="text-xs text-muted-foreground">
            Gateway Status
          </p>

          <p
            className={`text-sm font-semibold mt-1 ${getStatusClass(
              gateway.status
            )}`}
          >
            {gateway.status}
          </p>
        </div>
      </div>

      {/* Gateway Information */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-3">
        <Metric
          label="IP Address"
          value={gateway.ipAddress}
        />

        <Metric
          label="MQTT"
          value={
            gateway.mqttConnected
              ? "CONNECTED"
              : "DISCONNECTED"
          }
        />

        <Metric
          label="Connected Devices"
          value={String(gateway.connectedDevices)}
        />

        <Metric
          label="Messages/sec"
          value={`${gateway.messagesPerSecond.toFixed(1)} /s`}
        />
      </div>

      {/* Network Traffic */}
      <div className="mt-6">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Gateway Network State
        </h3>

        <div className="mt-3 rounded-lg border border-border bg-background p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-muted-foreground">
                Current traffic
              </p>

              <p className="text-xl font-semibold mt-1">
                {gateway.bytesPerSecond} B/s
              </p>
            </div>

            <div className="text-right">
              <p className="text-xs text-muted-foreground">
                MQTT Messages
              </p>

              <p className="text-xl font-semibold mt-1">
                {gateway.messagesPerSecond.toFixed(1)}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Security */}
      <div className="mt-6">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Gateway Security
        </h3>

        <div className="mt-3 rounded-lg border border-border bg-background p-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-muted-foreground">
                Anomaly Score
              </p>

              <p
                className={`text-2xl font-bold mt-1 ${getRiskClass(
                  gateway.anomalyScore
                )}`}
              >
                {(gateway.anomalyScore * 100).toFixed(0)}%
              </p>
            </div>

            <div>
              <p className="text-xs text-muted-foreground">
                Risk Level
              </p>

              <p
                className={`text-2xl font-bold mt-1 ${getRiskClass(
                  gateway.anomalyScore
                )}`}
              >
                {gateway.riskLevel}
              </p>
            </div>
          </div>

          <div className="mt-5">
            <div className="h-2 rounded-full bg-muted overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${
                  gateway.anomalyScore >= 0.8
                    ? "bg-danger"
                    : gateway.anomalyScore >= 0.6
                      ? "bg-warning"
                      : gateway.anomalyScore >= 0.3
                        ? "bg-yellow-500"
                        : "bg-success"
                }`}
                style={{
                  width: `${Math.min(
                    Math.max(gateway.anomalyScore * 100, 0),
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
    <div className="rounded-lg border border-border bg-background p-3">
      <p className="text-xs text-muted-foreground">
        {label}
      </p>

      <p className="text-sm font-medium mt-1">
        {value}
      </p>
    </div>
  );
}

function getRiskClass(score: number) {
  if (score >= 0.8) return "text-danger";
  if (score >= 0.6) return "text-warning";
  if (score >= 0.3) return "text-yellow-500";

  return "text-success";
}