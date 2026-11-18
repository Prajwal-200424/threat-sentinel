import { Card } from "@/components/ui/card";
import type { Prediction } from "@/lib/predictions";
import {
  getEnsembleScoreOrNull,
  getVerdict,
} from "@/lib/predictions";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

interface ChartsProps {
  predictions: Prediction[];
}

// =========================================================
// MAIN CHARTS
// =========================================================

export function Charts({
  predictions,
}: ChartsProps) {
  // -------------------------------------------------------
  // Prepare chart data
  // -------------------------------------------------------

  const orderedPredictions = [
    ...predictions,
  ]
    .sort(
      (a, b) =>
        Number(a.created_at ?? 0) -
        Number(b.created_at ?? 0),
    )
    .slice(-20);

  const threatData = orderedPredictions.map(
    (prediction, index) => ({
      index: index + 1,
      score: getEnsembleScoreOrNull(prediction),
      prediction:
        prediction.prediction ?? "UNKNOWN",
      time: formatChartTime(prediction),
    }),
  );

  // -------------------------------------------------------
  // Detection activity
  // -------------------------------------------------------
  // Only records the ensemble actually classified as BENIGN count as benign.
  // Suspicious and unrecognised verdicts are reported separately instead of
  // being folded into the benign total.

  const verdictCounts = {
    BENIGN: 0,
    SUSPICIOUS: 0,
    ATTACK: 0,
    UNKNOWN: 0,
  };

  for (const prediction of predictions) {
    verdictCounts[getVerdict(prediction)] += 1;
  }

  const detectionData: Array<{ name: string; count: number }> = [
    {
      name: "Benign",
      count: verdictCounts.BENIGN,
    },
    {
      name: "Suspicious",
      count: verdictCounts.SUSPICIOUS,
    },
    {
      name: "Attack",
      count: verdictCounts.ATTACK,
    },
  ];

  if (verdictCounts.UNKNOWN > 0) {
    detectionData.push({
      name: "Unknown",
      count: verdictCounts.UNKNOWN,
    });
  }

  // -------------------------------------------------------
  // No data state
  // -------------------------------------------------------

  if (predictions.length === 0) {
    return (
      <section className="space-y-4">
        <div>
          <h2 className="text-lg font-bold tracking-wide">
            THREAT ANALYTICS
          </h2>

          <p className="text-xs text-muted-foreground mt-1">
            Waiting for live ML detection data...
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ChartCard title="LIVE ENSEMBLE THREAT SCORE">
            <EmptyChart />
          </ChartCard>

          <ChartCard title="DETECTION ACTIVITY">
            <EmptyChart />
          </ChartCard>
        </div>
      </section>
    );
  }

  // -------------------------------------------------------
  // Render
  // -------------------------------------------------------

  return (
    <section className="space-y-4">
      {/* ================================================= */}
      {/* SECTION HEADER */}
      {/* ================================================= */}

      <div>
        <h2 className="text-lg font-bold tracking-wide">
          THREAT ANALYTICS
        </h2>

        <p className="text-xs text-muted-foreground mt-1">
          Real-time analysis from the 3-model ensemble
        </p>
      </div>

      {/* ================================================= */}
      {/* CHART GRID */}
      {/* ================================================= */}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* =============================================== */}
        {/* ENSEMBLE THREAT SCORE */}
        {/* =============================================== */}

        <ChartCard title="LIVE ENSEMBLE THREAT SCORE">
          <ResponsiveContainer
            width="100%"
            height={280}
          >
            <LineChart
              data={threatData}
              margin={{
                top: 10,
                right: 10,
                left: 0,
                bottom: 5,
              }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                className="opacity-20"
              />

              <XAxis
                dataKey="index"
                tick={{
                  fontSize: 10,
                }}
                tickLine={false}
                axisLine={false}
              />

              <YAxis
                domain={[0, 100]}
                tick={{
                  fontSize: 10,
                }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(value) =>
                  `${value}%`
                }
              />

              <Tooltip
                content={<ThreatTooltip />}
              />

              <Line
                type="monotone"
                dataKey="score"
                stroke="currentColor"
                className="text-cyber-cyan"
                strokeWidth={3}
                dot={{
                  r: 3,
                }}
                activeDot={{
                  r: 5,
                }}
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* =============================================== */}
        {/* DETECTION ACTIVITY */}
        {/* =============================================== */}

        <ChartCard title="DETECTION ACTIVITY">
          <ResponsiveContainer
            width="100%"
            height={280}
          >
            <BarChart
              data={detectionData}
              margin={{
                top: 10,
                right: 10,
                left: 0,
                bottom: 5,
              }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                className="opacity-20"
              />

              <XAxis
                dataKey="name"
                tick={{
                  fontSize: 11,
                }}
                tickLine={false}
                axisLine={false}
              />

              <YAxis
                allowDecimals={false}
                tick={{
                  fontSize: 10,
                }}
                tickLine={false}
                axisLine={false}
              />

              <Tooltip />

              <Bar
                dataKey="count"
                fill="currentColor"
                className="text-cyber-cyan"
                radius={[
                  6,
                  6,
                  0,
                  0,
                ]}
              />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* ================================================= */}
      {/* CURRENT ENSEMBLE INFORMATION */}
      {/* ================================================= */}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          label="CNN-LSTM"
          value="50%"
          description="Primary model weight"
        />

        <MetricCard
          label="Random Forest"
          value="25%"
          description="Classical ML weight"
        />

        <MetricCard
          label="Autoencoder"
          value="25%"
          description="Anomaly detection weight"
        />
      </div>
    </section>
  );
}

// =========================================================
// CHART CARD
// =========================================================

function ChartCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Card className="p-4 border-border/60 gradient-cyber">
      <h3 className="text-xs uppercase tracking-widest text-muted-foreground mb-3 font-semibold">
        {title}
      </h3>

      {children}
    </Card>
  );
}

// =========================================================
// EMPTY CHART
// =========================================================

function EmptyChart() {
  return (
    <div className="h-[280px] flex items-center justify-center">
      <div className="text-center">
        <p className="text-sm text-muted-foreground">
          No detection data available
        </p>

        <p className="text-xs text-muted-foreground mt-1">
          Waiting for the ML server...
        </p>
      </div>
    </div>
  );
}

// =========================================================
// METRIC CARD
// =========================================================

function MetricCard({
  label,
  value,
  description,
}: {
  label: string;
  value: string;
  description: string;
}) {
  return (
    <Card className="p-4 border-border/60 gradient-cyber">
      <p className="text-[10px] uppercase tracking-widest text-muted-foreground">
        {label}
      </p>

      <p className="mt-2 text-2xl font-bold text-cyber-cyan">
        {value}
      </p>

      <p className="text-xs text-muted-foreground mt-1">
        {description}
      </p>
    </Card>
  );
}

// =========================================================
// THREAT TOOLTIP
// =========================================================

function ThreatTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{
    payload?: {
      score?: number | null;
      prediction?: string;
      time?: string;
    };
  }>;
}) {
  if (
    !active ||
    !payload ||
    payload.length === 0
  ) {
    return null;
  }

  const data =
    payload[0]?.payload;

  if (!data) {
    return null;
  }

  return (
    <div className="rounded-md border border-border/60 bg-background/95 px-3 py-2 shadow-xl">
      <p className="text-xs text-muted-foreground">
        {data.time}
      </p>

      <p className="text-sm font-semibold text-cyber-cyan mt-1">
        Ensemble:{" "}
        {data.score === null ||
        data.score === undefined
          ? "—"
          : `${Number(data.score).toFixed(2)}%`}
      </p>

      <p className="text-xs mt-1">
        Classification:{" "}
        <span className="font-semibold">
          {data.prediction}
        </span>
      </p>
    </div>
  );
}

// =========================================================
// TIME FORMAT
// =========================================================

function formatChartTime(
  prediction: Prediction,
): string {
  const timestamp =
    prediction.timestamp ||
    new Date(
      prediction.created_at,
    ).toISOString();

  const date =
    new Date(timestamp);

  if (
    Number.isNaN(
      date.getTime(),
    )
  ) {
    return "—";
  }

  return date.toLocaleTimeString(
    [],
    {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    },
  );
}