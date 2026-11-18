import { useEffect, useState } from "react";
import {
  Thermometer,
  Droplets,
  Wifi,
  Zap,
  Activity,
  Clock,
  Gauge,
} from "lucide-react";

import { Card } from "@/components/ui/card";

// Configure VITE_API_BASE_URL in .env if the frontend and Flask
// backend are hosted on different machines. Example:
// VITE_API_BASE_URL=http://192.168.1.20:5000
const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000"
).replace(/\/+$/, "");

const TELEMETRY_API = `${API_BASE_URL}/api/telemetry`;
const WEATHER_API = `${API_BASE_URL}/api/weather`;

interface TelemetryData {
  device?: string;
  wifi_rssi?: number | string | null;
  active_connections?: number | string | null;
  messages_per_second?: number | string | null;
  bytes_per_second?: number | string | null;
  status?: string;
  timestamp?: string | null;
  source?: string;
  voltage?: number | string | null;
  age_seconds?: number | null;
}

interface WeatherData {
  temperature?: number | string | null;
  humidity?: number | string | null;
  timestamp?: string | null;
  source?: string;
  status?: string;
}

function numericValue(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatNumber(
  value: unknown,
  digits = 1,
  suffix = "",
): string {
  const parsed = numericValue(value);
  return parsed === null ? "—" : `${parsed.toFixed(digits)}${suffix}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function telemetryFromPayload(payload: unknown): TelemetryData {
  if (!isRecord(payload)) return {};
  return isRecord(payload.telemetry)
    ? (payload.telemetry as TelemetryData)
    : (payload as TelemetryData);
}

function weatherFromPayload(payload: unknown): WeatherData {
  if (!isRecord(payload)) return {};
  return isRecord(payload.weather)
    ? (payload.weather as WeatherData)
    : (payload as WeatherData);
}

async function getJson(url: string, timeoutMs = 8000): Promise<unknown> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  let response: Response;
  try {
    response = await fetch(url, { cache: "no-store", signal: controller.signal });
  } finally {
    window.clearTimeout(timeout);
  }
  if (!response.ok) {
    throw new Error(`${url} returned HTTP ${response.status}`);
  }
  try {
    return await response.json();
  } catch {
    throw new Error(`${url} returned invalid JSON`);
  }
}

export function DeviceTelemetry() {
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [loading, setLoading] = useState(true);
  const [telemetryError, setTelemetryError] = useState<string | null>(null);
  const [weatherError, setWeatherError] = useState<string | null>(null);
  const [lastSuccessfulPoll, setLastSuccessfulPoll] = useState<Date | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function pollTelemetry() {
      try {
        const json = await getJson(TELEMETRY_API);
        if (cancelled) return;
        const data = telemetryFromPayload(json);
        setTelemetry(data);
        setTelemetryError(null);
        setLastSuccessfulPoll(new Date());
      } catch (reason) {
        if (cancelled) return;
        setTelemetryError(
          reason instanceof Error ? reason.message : "Failed to fetch telemetry",
        );
      } finally {
        if (!cancelled) {
          setLoading(false);
          timer = setTimeout(pollTelemetry, 5000);
        }
      }
    }

    async function pollWeather() {
      try {
        const json = await getJson(WEATHER_API, 12000);
        if (cancelled) return;
        const data = weatherFromPayload(json);
        setWeather(data);
        setWeatherError(null);
      } catch (reason) {
        if (cancelled) return;
        setWeatherError(
          reason instanceof Error ? reason.message : "Failed to fetch weather",
        );
      } finally {
        if (!cancelled) weatherTimer = setTimeout(pollWeather, 30_000);
      }
    }

    let weatherTimer: ReturnType<typeof setTimeout> | undefined;
    void pollTelemetry();
    void pollWeather();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
      if (weatherTimer) clearTimeout(weatherTimer);
    };
  }, []);

  const deviceName = telemetry?.device ?? "ESP32-01";

  // The backend derives ONLINE / STALE / WAITING from the receipt time of the
  // last valid packet. If telemetry cannot be fetched at all, the last known
  // status must not keep claiming the device is online.
  const deviceStatus = telemetryError
    ? "UNREACHABLE"
    : (telemetry?.status ?? "WAITING");

  const statusUpper = deviceStatus.toUpperCase();
  const isOnline = statusUpper === "ONLINE";
  const isStale = statusUpper === "STALE";
  const statusDotClass = isOnline
    ? "bg-green-500"
    : isStale
      ? "bg-orange-500"
      : "bg-yellow-500";

  const lastTelemetryUpdate = telemetry?.timestamp
    ? new Date(telemetry.timestamp).toLocaleTimeString()
    : "—";

  return (
    <section className="space-y-4">
      <div>
        <h2 className="text-xl font-bold flex items-center gap-2">
          <Zap className="h-5 w-5 text-cyber-cyan" />
          Real ESP32 Telemetry
        </h2>
        <p className="text-xs text-muted-foreground mt-1">
          Temperature and humidity from <span className="font-semibold">Open-Meteo</span>
          {" • "}
          Network metrics from <span className="font-semibold">ESP32 telemetry API</span>
        </p>
      </div>

      {telemetryError && (
        <Card className="p-4 border-warning/40 bg-warning/5">
          <p className="text-sm text-warning">
            Telemetry request failed: {telemetryError}
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Confirm Flask is running and GET {TELEMETRY_API} responds successfully.
          </p>
        </Card>
      )}

      {weatherError && (
        <Card className="p-3 border-border/40 bg-background/30">
          <p className="text-xs text-muted-foreground">
            Weather is currently unavailable: {weatherError}. Device telemetry will continue to update.
          </p>
        </Card>
      )}

      <div className="flex items-center justify-between rounded-lg border border-border/40 bg-background/40 px-4 py-3">
        <div>
          <p className="text-[10px] uppercase tracking-widest text-muted-foreground">
            Connected Device
          </p>
          <p className="font-mono font-bold mt-1">{deviceName}</p>
          <p className="text-[10px] text-muted-foreground mt-1">
            Last successful API poll: {lastSuccessfulPoll?.toLocaleTimeString() ?? "—"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`h-2.5 w-2.5 rounded-full ${statusDotClass}`}
          />
          <span className="text-xs font-mono font-semibold">{deviceStatus}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <TelemetryMetric
          icon={<Thermometer className="h-4 w-4" />}
          label="Temperature"
          value={formatNumber(weather?.temperature, 1, "°C")}
          source={weather?.source ?? "Open-Meteo"}
          loading={loading}
        />
        <TelemetryMetric
          icon={<Droplets className="h-4 w-4" />}
          label="Humidity"
          value={formatNumber(weather?.humidity, 1, "%")}
          source={weather?.source ?? "Open-Meteo"}
          loading={loading}
        />
        <TelemetryMetric
          icon={<Wifi className="h-4 w-4" />}
          label="Wi-Fi RSSI"
          value={formatNumber(telemetry?.wifi_rssi, 0, " dBm")}
          source={deviceName}
          loading={loading}
        />
        <TelemetryMetric
          icon={<Gauge className="h-4 w-4" />}
          label="Voltage"
          value={formatNumber(telemetry?.voltage, 2, " V")}
          source={deviceName}
          loading={loading}
        />
        <TelemetryMetric
          icon={<Activity className="h-4 w-4" />}
          label="Active Connections"
          value={formatNumber(telemetry?.active_connections, 0)}
          source={deviceName}
          loading={loading}
        />
        <TelemetryMetric
          icon={<Zap className="h-4 w-4" />}
          label="Messages/sec"
          value={formatNumber(telemetry?.messages_per_second, 1)}
          source={deviceName}
          loading={loading}
        />
        <TelemetryMetric
          icon={<Activity className="h-4 w-4" />}
          label="Bytes/sec"
          value={formatNumber(telemetry?.bytes_per_second, 0)}
          source={deviceName}
          loading={loading}
        />
        <TelemetryMetric
          icon={<Clock className="h-4 w-4" />}
          label="Last Telemetry Update"
          value={lastTelemetryUpdate}
          source="Backend timestamp"
          loading={loading}
        />
      </div>

      {!loading && telemetry && (
        <p className="text-[10px] text-muted-foreground">
          Telemetry source: {telemetry.source ?? "Not specified"}.
          Values shown as “—” were absent or not numeric in the API response.
        </p>
      )}
    </section>
  );
}

function TelemetryMetric({
  icon,
  label,
  value,
  source,
  loading,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  source: string;
  loading: boolean;
}) {
  return (
    <Card className="p-4 border-border/40 bg-gradient-to-br from-background/60 to-background/40 backdrop-blur-sm">
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold flex items-center gap-1.5 mb-2">
            <span className="text-cyber-cyan">{icon}</span>
            {label}
          </p>
          <p className="text-lg font-mono font-bold text-foreground">
            {loading ? "…" : value}
          </p>
          <p className="text-[9px] text-muted-foreground/60 mt-1.5 uppercase tracking-wider truncate">
            {source}
          </p>
        </div>
      </div>
    </Card>
  );
}
