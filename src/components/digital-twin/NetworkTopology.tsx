import type { DeviceTwinState, GatewayTwinState, NetworkConnection } from "./types";

interface NetworkTopologyProps {
  esp32: DeviceTwinState;
  gateway: GatewayTwinState;
  connection: NetworkConnection;
}

function statusColor(status: string) {
  switch (status) {
    case "ONLINE":
      return "text-success";
    case "SUSPICIOUS":
      return "text-warning";
    case "HIGH_RISK":
    case "POTENTIAL_ATTACK":
      return "text-danger";
    case "OFFLINE":
    default:
      return "text-muted-foreground";
  }
}

function statusDot(status: string) {
  switch (status) {
    case "ONLINE":
      return "bg-success";
    case "SUSPICIOUS":
      return "bg-warning";
    case "HIGH_RISK":
    case "POTENTIAL_ATTACK":
      return "bg-danger";
    default:
      return "bg-muted-foreground";
  }
}

export function NetworkTopology({
  esp32,
  gateway,
  connection,
}: NetworkTopologyProps) {
  const active = connection.status === "ACTIVE";

  return (
    <section className="rounded-xl border border-border bg-card p-6">
      <div className="mb-6">
        <h2 className="text-lg font-semibold">
          IoT Network Digital Twin
        </h2>

        <p className="text-sm text-muted-foreground mt-1">
          Live virtual representation of the IoT device and edge gateway
        </p>
      </div>

      <div className="relative min-h-[300px] flex items-center justify-center">
        <div className="flex items-center gap-8 md:gap-16 w-full max-w-4xl">
          
          {/* ESP32 */}
          <div className="flex-1 flex justify-center">
            <div className="w-52 rounded-xl border border-border bg-background p-5 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-mono text-muted-foreground">
                  IoT DEVICE
                </span>

                <span
                  className={`h-2.5 w-2.5 rounded-full ${statusDot(
                    esp32.status
                  )}`}
                />
              </div>

              <h3 className="font-semibold">
                {esp32.deviceId}
              </h3>

              <p className={`text-sm mt-1 ${statusColor(esp32.status)}`}>
                {esp32.status}
              </p>

              <div className="mt-4 space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">
                    Temperature
                  </span>
                  <span>{esp32.temperature.toFixed(1)}°C</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-muted-foreground">
                    Wi-Fi
                  </span>
                  <span>{esp32.wifiRssi} dBm</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-muted-foreground">
                    MQTT
                  </span>
                  <span>
                    {esp32.mqttConnected ? "Connected" : "Disconnected"}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Connection */}
          <div className="flex-1 flex flex-col items-center">
            <div className="text-xs text-muted-foreground mb-2">
              {connection.protocol}
            </div>

            <div className="relative w-full h-1">
              <div className="absolute inset-0 bg-border rounded-full" />

              <div
                className={`absolute left-0 top-0 h-1 rounded-full transition-all duration-500 ${
                  active ? "w-full bg-success" : "w-0"
                }`}
              />
            </div>

            <div className="mt-3 text-center">
              <p
                className={`text-xs font-medium ${
                  active ? "text-success" : "text-muted-foreground"
                }`}
              >
                {active ? "ACTIVE" : "INACTIVE"}
              </p>

              <p className="text-[11px] text-muted-foreground mt-1">
                {connection.messagesPerSecond.toFixed(1)} msg/s
              </p>
            </div>
          </div>

          {/* Raspberry Pi */}
          <div className="flex-1 flex justify-center">
            <div className="w-52 rounded-xl border border-border bg-background p-5 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-mono text-muted-foreground">
                  EDGE GATEWAY
                </span>

                <span
                  className={`h-2.5 w-2.5 rounded-full ${statusDot(
                    gateway.status
                  )}`}
                />
              </div>

              <h3 className="font-semibold">
                {gateway.deviceId}
              </h3>

              <p className={`text-sm mt-1 ${statusColor(gateway.status)}`}>
                {gateway.status}
              </p>

              <div className="mt-4 space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">
                    MQTT
                  </span>
                  <span>
                    {gateway.mqttConnected
                      ? "Connected"
                      : "Disconnected"}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-muted-foreground">
                    Devices
                  </span>
                  <span>{gateway.connectedDevices}</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-muted-foreground">
                    Traffic
                  </span>
                  <span>
                    {gateway.messagesPerSecond.toFixed(1)} msg/s
                  </span>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-lg bg-background p-3">
          <p className="text-xs text-muted-foreground">
            Connection
          </p>
          <p className="text-sm font-medium mt-1">
            {connection.status}
          </p>
        </div>

        <div className="rounded-lg bg-background p-3">
          <p className="text-xs text-muted-foreground">
            Protocol
          </p>
          <p className="text-sm font-medium mt-1">
            {connection.protocol}
          </p>
        </div>

        <div className="rounded-lg bg-background p-3">
          <p className="text-xs text-muted-foreground">
            Messages
          </p>
          <p className="text-sm font-medium mt-1">
            {connection.messagesPerSecond.toFixed(1)}/s
          </p>
        </div>

        <div className="rounded-lg bg-background p-3">
          <p className="text-xs text-muted-foreground">
            Data Rate
          </p>
          <p className="text-sm font-medium mt-1">
            {connection.bytesPerSecond} B/s
          </p>
        </div>
      </div>
    </section>
  );
}