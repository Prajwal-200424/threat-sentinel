import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  AlertTriangle,
  Cpu,
  ShieldAlert,
  ShieldCheck,
} from "lucide-react";

import {
  MODELS,
  fetchModelMetrics,
  hasRealMetrics,
  pct,
  type ModelId,
  type ModelMetrics,
} from "@/lib/models";

import type { Prediction } from "@/lib/predictions";

import {
  getEnsembleScore,
  getAttackVotes,
  getTotalModels,
  getRiskLevel,
} from "@/lib/predictions";

/* =========================================================
   SCORE NORMALIZATION
   ========================================================= */

function normalizeLiveScore(score: number): number {
  const value = Number(score);

  if (!Number.isFinite(value)) {
    return 0;
  }

  /*
   Backend may return:
   0–1       -> probability
   0–100     -> percentage
   0–10000   -> accidentally scaled percentage
  */

  if (value <= 1) {
    return value * 100;
  }

  if (value > 100) {
    return value / 100;
  }

  return value;
}

/* =========================================================
   TOOLTIP
   ========================================================= */

const tooltipStyle = {
  backgroundColor: "oklch(0.18 0.02 250)",
  border: "1px solid oklch(0.3 0.03 250)",
  borderRadius: 8,
  fontSize: 12,
};

/* =========================================================
   ONLY THE 3 ACTUAL ENSEMBLE MODELS
   ========================================================= */

const ENSEMBLE_MODEL_IDS = [
  "cnn_lstm",
  "random_forest",
  "autoencoder",
] as ModelId[];

/* =========================================================
   PROPS
   ========================================================= */

interface Props {
  selectedModel: ModelId;

  onSelectModel: (id: ModelId) => void;

  prediction?: Prediction | null;
}

/* =========================================================
   MODEL COMPARISON
   ========================================================= */

export function ModelComparison({
  selectedModel,
  onSelectModel,
  prediction,
}: Props) {
  const [metrics, setMetrics] = useState<ModelMetrics[]>([]);

  /* =======================================================
     LOAD TRAINED MODEL METRICS
     ======================================================= */

  useEffect(() => {
    let cancelled = false;

    async function loadMetrics() {
      try {
        const result = await fetchModelMetrics();

        if (!cancelled) {
          setMetrics(result);
        }
      } catch (error) {
        console.error("Failed to load model metrics:", error);

        if (!cancelled) {
          setMetrics([]);
        }
      }
    }

    loadMetrics();

    return () => {
      cancelled = true;
    };
  }, []);

  /* =======================================================
     FILTER TO ACTUAL 3-MODEL ENSEMBLE
     ======================================================= */

  const ensembleModels = MODELS.filter((model) =>
    ENSEMBLE_MODEL_IDS.includes(model.id),
  );

  /*
   * Safety fallback:
   * If the current parent state is pointing to XGBoost/SVM/etc.,
   * automatically display CNN-LSTM as the active benchmark model.
   */

  const activeModel =
    ensembleModels.find(
      (model) => model.id === selectedModel,
    ) ?? ensembleModels[0];

  const activeModelId =
    activeModel?.id ?? ("cnn_lstm" as ModelId);

  /* =======================================================
     METRICS MAP
     ======================================================= */

  const byId = new Map(
    metrics.map((metric) => [
      metric.model_id,
      metric,
    ]),
  );

  const isReal = hasRealMetrics(metrics);

  const activeMetrics = byId.get(activeModelId);

  /* =======================================================
     LIVE ENSEMBLE DATA
     ======================================================= */

  const liveModels = prediction?.models;

  const ensembleScore = prediction
    ? getEnsembleScore(prediction)
    : 0;

  const attackVotes = prediction
    ? getAttackVotes(prediction)
    : 0;

  const totalModels = prediction
    ? getTotalModels(prediction)
    : 3;

  const riskLevel = prediction
    ? getRiskLevel(prediction)
    : "LOW";

  const finalClassification =
    prediction?.prediction ?? "WAITING";

  /* =======================================================
     LIVE 3-MODEL DATA
     ======================================================= */

  const liveModelData = [
    {
      name: "CNN-LSTM",
      key: "cnn_lstm" as const,

      score:
        liveModels?.cnn_lstm?.score ?? 0,

      prediction:
        liveModels?.cnn_lstm?.prediction ?? "—",

      weight: 50,
    },

    {
      name: "Random Forest",
      key: "random_forest" as const,

      score:
        liveModels?.random_forest?.score ?? 0,

      prediction:
        liveModels?.random_forest?.prediction ?? "—",

      weight: 25,
    },

    {
      name: "Autoencoder",
      key: "autoencoder" as const,

      score:
        liveModels?.autoencoder?.score ?? 0,

      prediction:
        liveModels?.autoencoder?.prediction ?? "—",

      weight: 25,
    },
  ];

  /* =======================================================
     TRAINED METRICS CHART
     ONLY 3 MODELS
     ======================================================= */

  const chartData = ensembleModels.map((model) => {
    const metric = byId.get(model.id);

    return {
      name: model.name,

      accuracy:
        metric?.accuracy != null
          ? metric.accuracy * 100
          : 0,

      f1:
        metric?.f1_score != null
          ? metric.f1_score * 100
          : 0,

      detection:
        metric?.detection_rate != null
          ? metric.detection_rate * 100
          : 0,

      fpr:
        metric?.false_positive_rate != null
          ? metric.false_positive_rate * 100
          : 0,
    };
  });

  /* =======================================================
     RADAR DATA
     ======================================================= */

  const radarData = activeMetrics
    ? [
        {
          metric: "Accuracy",
          value: activeMetrics.accuracy * 100,
        },

        {
          metric: "Precision",
          value: activeMetrics.precision * 100,
        },

        {
          metric: "Recall",
          value: activeMetrics.recall * 100,
        },

        {
          metric: "F1",
          value: activeMetrics.f1_score * 100,
        },

        {
          metric: "Detection",
          value: activeMetrics.detection_rate * 100,
        },
      ]
    : [];

  /* =======================================================
     RETURN
     ======================================================= */

  return (
    <section className="space-y-4">

      {/* ===================================================
          LIVE ENSEMBLE
          =================================================== */}

      <Card className="border-border/60 gradient-cyber overflow-hidden">

        {/* HEADER */}

        <div className="px-5 py-4 border-b border-border/60">

          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">

            <div>

              <h2 className="text-xs uppercase tracking-widest text-muted-foreground font-semibold">
                Live ML Ensemble Analysis
              </h2>

              <p className="text-xs text-muted-foreground mt-1">
                Three independent models contribute to the
                final security decision.
              </p>

            </div>

            {/* FINAL CLASSIFICATION */}

            <Badge
              variant="outline"
              className={
                finalClassification === "ATTACK"
                  ? "border-danger text-danger text-glow-danger"
                  : finalClassification === "SUSPICIOUS"
                    ? "border-warning text-warning"
                    : "border-safe text-safe text-glow-safe"
              }
            >
              {finalClassification}
            </Badge>

          </div>

        </div>

        {/* MODEL CARDS */}

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-5">

          {liveModelData.map((model) => {

            const predictionText =
              model.prediction?.toUpperCase() ?? "";

            const isAttack =
              predictionText.includes("ATTACK") ||
              predictionText.includes("MALICIOUS");

            const isAvailable =
              model.prediction !== "—";

            const displayScore =
              normalizeLiveScore(model.score);

            return (

              <div
                key={model.key}
                className={`rounded-lg border p-4 transition-all ${
                  isAttack
                    ? "border-danger/50 bg-danger/5"
                    : isAvailable
                      ? "border-safe/40 bg-safe/5"
                      : "border-border/50 bg-background/30"
                }`}
              >

                {/* MODEL HEADER */}

                <div className="flex items-center justify-between">

                  <div className="flex items-center gap-2">

                    {isAttack ? (
                      <ShieldAlert className="h-4 w-4 text-danger" />
                    ) : (
                      <ShieldCheck className="h-4 w-4 text-safe" />
                    )}

                    <p className="text-xs uppercase tracking-widest font-semibold">
                      {model.name}
                    </p>

                  </div>

                  <Badge
                    variant="outline"
                    className={
                      isAttack
                        ? "border-danger text-danger text-[10px]"
                        : "border-safe text-safe text-[10px]"
                    }
                  >
                    {model.weight}%
                  </Badge>

                </div>

                {/* SCORE */}

                <p
                  className={`mt-4 text-3xl font-bold ${
                    isAttack
                      ? "text-danger text-glow-danger"
                      : "text-safe text-glow-safe"
                  }`}
                >
                  {isAvailable
                    ? `${displayScore.toFixed(2)}%`
                    : "—"}
                </p>

                {/* CLASSIFICATION */}

                <p
                  className={`mt-1 text-xs font-semibold ${
                    isAttack
                      ? "text-danger"
                      : "text-safe"
                  }`}
                >
                  {model.prediction}
                </p>

                <p className="text-[10px] text-muted-foreground mt-2">
                  Ensemble weight: {model.weight}%
                </p>

              </div>
            );
          })}

        </div>

        {/* ENSEMBLE RESULT */}

        <div className="mx-5 mb-5 rounded-lg border border-border/50 bg-background/30 p-5">

          <div className="flex items-center gap-2 mb-4">

            <Cpu className="h-5 w-5 text-cyber-cyan" />

            <div>

              <p className="text-xs uppercase tracking-widest font-semibold">
                Final Ensemble Decision
              </p>

              <p className="text-[10px] text-muted-foreground">
                CNN-LSTM 50% · Random Forest 25% · Autoencoder 25%
              </p>

            </div>

          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-5">

            <LiveMetric
              label="Ensemble Score"
              value={
                prediction
                  ? `${ensembleScore.toFixed(2)}%`
                  : "—"
              }
              highlight
            />

            <LiveMetric
              label="Attack Votes"
              value={
                prediction
                  ? `${attackVotes}/${totalModels}`
                  : "—"
              }
            />

            <LiveMetric
              label="Final Classification"
              value={
                prediction
                  ? finalClassification
                  : "—"
              }
            />

            <LiveMetric
              label="Risk Level"
              value={
                prediction
                  ? riskLevel
                  : "—"
              }
            />

          </div>

        </div>

      </Card>

      {/* ===================================================
          TRAINED MODEL PERFORMANCE
          =================================================== */}

      <Card className="border-border/60 gradient-cyber overflow-hidden">

        <div className="px-5 py-4 border-b border-border/60">

          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">

            <div>

              <h2 className="text-xs uppercase tracking-widest text-muted-foreground font-semibold">
                Trained Model Performance
              </h2>

              <p className="text-xs text-muted-foreground mt-1">
                Evaluation metrics from the trained models.
              </p>

            </div>

            <Badge
              variant="outline"
              className="border-cyber-cyan/40 text-cyber-cyan"
            >
              3-MODEL ENSEMBLE
            </Badge>

          </div>

        </div>

        {/* WARNING */}

        {!isReal && (

          <div className="m-5 flex items-start gap-2 rounded-md border border-warning/50 bg-warning/10 px-3 py-2 text-xs text-warning">

            <AlertTriangle className="h-4 w-4 shrink-0" />

            <span>
              Trained model metrics are not currently available.
              Live ensemble detection remains active independently.
            </span>

          </div>

        )}

        {/* METRIC TABLE */}

        <div className="overflow-x-auto">

          <Table>

            <TableHeader>

              <TableRow className="border-border/60 hover:bg-transparent">

                <TableHead className="text-[10px] uppercase tracking-widest">
                  Model
                </TableHead>

                <TableHead className="text-[10px] uppercase tracking-widest">
                  Weight
                </TableHead>

                <TableHead className="text-[10px] uppercase tracking-widest">
                  Accuracy
                </TableHead>

                <TableHead className="text-[10px] uppercase tracking-widest">
                  Precision
                </TableHead>

                <TableHead className="text-[10px] uppercase tracking-widest">
                  Recall
                </TableHead>

                <TableHead className="text-[10px] uppercase tracking-widest">
                  F1 Score
                </TableHead>

                <TableHead className="text-[10px] uppercase tracking-widest">
                  Detection Rate
                </TableHead>

                <TableHead className="text-[10px] uppercase tracking-widest">
                  FPR
                </TableHead>

                <TableHead className="text-[10px] uppercase tracking-widest">
                  Source
                </TableHead>

              </TableRow>

            </TableHeader>

            <TableBody>

              {ensembleModels.map((model) => {

                const metric =
                  byId.get(model.id);

                const selected =
                  model.id === activeModelId;

                const weight =
                  model.id === "cnn_lstm"
                    ? 50
                    : 25;

                return (

                  <TableRow
                    key={model.id}
                    className={`border-border/40 cursor-pointer ${
                      selected
                        ? "bg-primary/10 hover:bg-primary/15"
                        : "hover:bg-muted/30"
                    }`}
                    onClick={() =>
                      onSelectModel(model.id)
                    }
                  >

                    {/* MODEL */}

                    <TableCell className="text-xs font-semibold">

                      <span className="flex items-center gap-2">

                        {model.name}

                        {model.id === "cnn_lstm" && (

                          <Badge className="bg-safe/20 text-safe border border-safe/40 text-[10px]">
                            Primary
                          </Badge>

                        )}

                      </span>

                    </TableCell>

                    {/* WEIGHT */}

                    <TableCell className="font-mono text-xs text-cyber-cyan">
                      {weight}%
                    </TableCell>

                    {/* ACCURACY */}

                    <TableCell className="font-mono text-xs">
                      {metric
                        ? pct(metric.accuracy)
                        : "—"}
                    </TableCell>

                    {/* PRECISION */}

                    <TableCell className="font-mono text-xs">
                      {metric
                        ? pct(metric.precision)
                        : "—"}
                    </TableCell>

                    {/* RECALL */}

                    <TableCell className="font-mono text-xs">
                      {metric
                        ? pct(metric.recall)
                        : "—"}
                    </TableCell>

                    {/* F1 */}

                    <TableCell className="font-mono text-xs">
                      {metric
                        ? pct(metric.f1_score)
                        : "—"}
                    </TableCell>

                    {/* DETECTION */}

                    <TableCell className="font-mono text-xs">
                      {metric
                        ? pct(metric.detection_rate)
                        : "—"}
                    </TableCell>

                    {/* FPR */}

                    <TableCell className="font-mono text-xs">
                      {metric
                        ? pct(metric.false_positive_rate)
                        : "—"}
                    </TableCell>

                    {/* SOURCE */}

                    <TableCell className="text-[10px] uppercase tracking-wide text-muted-foreground">
                      {metric?.source ?? "—"}
                    </TableCell>

                  </TableRow>

                );
              })}

            </TableBody>

          </Table>

        </div>

      </Card>

      {/* ===================================================
          MODEL SELECTOR
          =================================================== */}

      <Card className="border-border/60 gradient-cyber p-5">

        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">

          <div>

            <h2 className="text-xs uppercase tracking-widest text-muted-foreground font-semibold">
              Ensemble Model
            </h2>

            <div className="mt-3 flex items-center gap-3">

              <Cpu
                className="h-8 w-8 text-cyber-cyan text-glow-cyber"
                strokeWidth={1.5}
              />

              <div>

                <p className="text-2xl font-bold text-cyber-cyan text-glow-cyber">
                  {activeModel?.name ?? "CNN-LSTM"}
                </p>

                <p className="text-xs text-muted-foreground">
                  {activeModel?.description ??
                    "Primary deep learning temporal model"}
                </p>

              </div>

              {activeModel?.id === "cnn_lstm" && (

                <Badge className="bg-safe/20 text-safe border border-safe/40">
                  Primary
                </Badge>

              )}

            </div>

          </div>

          {/* MODEL SELECTOR */}

          <div className="w-full lg:w-72">

            <label className="text-[10px] uppercase tracking-widest text-muted-foreground">
              Select model
            </label>

            <Select
              value={activeModelId}
              onValueChange={(value) =>
                onSelectModel(value as ModelId)
              }
            >

              <SelectTrigger className="mt-2 font-mono text-sm">

                <SelectValue />

              </SelectTrigger>

              <SelectContent>

                {ensembleModels.map((model) => (

                  <SelectItem
                    key={model.id}
                    value={model.id}
                  >
                    {model.name}
                    {model.id === "cnn_lstm"
                      ? " · Primary"
                      : ""}
                  </SelectItem>

                ))}

              </SelectContent>

            </Select>

          </div>

        </div>

      </Card>

      {/* ===================================================
          BENCHMARK CHARTS
          =================================================== */}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* BAR CHART */}

        <Card className="p-4 border-border/60 gradient-cyber lg:col-span-2">

          <h3 className="text-xs uppercase tracking-widest text-muted-foreground mb-3 font-semibold">
            Model Performance Comparison
          </h3>

          <ResponsiveContainer
            width="100%"
            height={280}
          >

            <BarChart data={chartData}>

              <CartesianGrid
                strokeDasharray="3 3"
                stroke="var(--border)"
              />

              <XAxis
                dataKey="name"
                stroke="var(--muted-foreground)"
                fontSize={10}
                interval={0}
                angle={-10}
                textAnchor="end"
                height={60}
              />

              <YAxis
                stroke="var(--muted-foreground)"
                fontSize={11}
                domain={[0, 100]}
                unit="%"
              />

              <Tooltip
                contentStyle={tooltipStyle}
              />

              <Legend
                wrapperStyle={{
                  fontSize: 12,
                }}
              />

              <Bar
                dataKey="accuracy"
                fill="var(--cyber-cyan)"
                name="Accuracy"
                radius={[4, 4, 0, 0]}
              />

              <Bar
                dataKey="f1"
                fill="var(--accent)"
                name="F1 Score"
                radius={[4, 4, 0, 0]}
              />

              <Bar
                dataKey="detection"
                fill="var(--safe)"
                name="Detection Rate"
                radius={[4, 4, 0, 0]}
              />

              <Bar
                dataKey="fpr"
                fill="var(--danger)"
                name="False Positive Rate"
                radius={[4, 4, 0, 0]}
              />

            </BarChart>

          </ResponsiveContainer>

        </Card>

        {/* RADAR */}

        <Card className="p-4 border-border/60 gradient-cyber">

          <h3 className="text-xs uppercase tracking-widest text-muted-foreground mb-3 font-semibold">
            {activeModel?.name ?? "CNN-LSTM"} — Metric Profile
          </h3>

          <ResponsiveContainer
            width="100%"
            height={280}
          >

            <RadarChart data={radarData}>

              <PolarGrid
                stroke="var(--border)"
              />

              <PolarAngleAxis
                dataKey="metric"
                stroke="var(--muted-foreground)"
                fontSize={11}
              />

              <PolarRadiusAxis
                domain={[0, 100]}
                stroke="var(--muted-foreground)"
                fontSize={10}
              />

              <Radar
                dataKey="value"
                stroke="var(--cyber-cyan)"
                fill="var(--cyber-cyan)"
                fillOpacity={0.25}
                name={activeModel?.name ?? "Model"}
              />

              <Tooltip
                contentStyle={tooltipStyle}
              />

            </RadarChart>

          </ResponsiveContainer>

        </Card>

      </div>

    </section>
  );
}

/* =========================================================
   LIVE METRIC
   ========================================================= */

function LiveMetric({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div>

      <p className="text-[10px] uppercase tracking-widest text-muted-foreground">
        {label}
      </p>

      <p
        className={`mt-1 text-lg font-bold ${
          highlight
            ? "text-cyber-cyan text-glow-cyber"
            : "text-foreground"
        }`}
      >
        {value}
      </p>

    </div>
  );
}