import { Brain, Zap } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

import type { Prediction } from "@/lib/predictions";

import {
  getEnsembleScoreOrNull,
  getAttackVotes,
  getTotalModels,
  getRiskLevel,
  formatScore,
} from "@/lib/predictions";

interface ModelAnalysisProps {
  latest: Prediction | null;
}

// =========================================================
// SCORE NORMALIZATION
// =========================================================
// Backend scores are reported on a 0–100 scale and every model in the
// ensemble is scored the same way.
//
// A missing or out-of-range value is treated as unavailable (null) instead of
// being silently rescaled, so a malformed API response cannot be presented as a
// plausible-looking metric.
// =========================================================

function normalizeScore(
  score: unknown,
): number | null {
  if (
    score === null ||
    score === undefined ||
    score === ""
  ) {
    return null;
  }

  const value = Number(score);

  if (
    !Number.isFinite(value) ||
    value < 0 ||
    value > 100
  ) {
    return null;
  }

  return value;
}

// =========================================================
// MODEL ANALYSIS
// =========================================================

export function ModelAnalysis({
  latest,
}: ModelAnalysisProps) {
  if (
    !latest ||
    !latest.models ||
    !latest.models.ensemble
  ) {
    return (
      <Card className="p-8 border-border/60 bg-gradient-to-br from-background/80 to-background/40">
        <div className="flex items-center gap-3 text-center justify-center">
          <Brain className="h-5 w-5 text-cyber-cyan" />

          <p className="text-sm text-muted-foreground">
            Awaiting model predictions…
          </p>
        </div>
      </Card>
    );
  }

  const models =
    latest.models;

  // =======================================================
  // ENSEMBLE VALUES
  // =======================================================

  const ensembleScore =
    normalizeScore(
      getEnsembleScoreOrNull(latest),
    );

  const attackVotes =
    getAttackVotes(latest);

  const totalModels =
    getTotalModels(latest);

  const riskLevel =
    getRiskLevel(latest);

  // =======================================================
  // MODEL CONFIGURATION
  // =======================================================

  const modelConfigs = [
    {
      name: "CNN-LSTM",

      weight: 50,

      icon: "🧠",

      description:
        "Deep Learning Temporal Sequence",

      data: models.cnn_lstm,

      isPrimary: true,
    },
    {
      name: "Random Forest",

      weight: 25,

      icon: "🌲",

      description:
        "Classical ML Ensemble",

      data: models.random_forest,
    },

    {
      name: "Autoencoder",

      weight: 25,

      icon: "🔍",

      description:
        "Anomaly Detection",

      data: models.autoencoder,
    },
  ];

  // =======================================================
  // RENDER
  // =======================================================

  return (
    <section className="space-y-4">

      {/* ================================================= */}
      {/* HEADER */}
      {/* ================================================= */}

      <div>
        <h2 className="text-xl font-bold flex items-center gap-2">
          <Brain className="h-5 w-5 text-cyber-cyan" />

          Ensemble Model Analysis
        </h2>

        <p className="text-xs text-muted-foreground mt-1">
          3-model ensemble: CNN-LSTM (50%) · Random Forest
          (25%) · Autoencoder (25%)
        </p>
      </div>

      {/* ================================================= */}
      {/* MODEL CARDS */}
      {/* ================================================= */}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {modelConfigs.map(
          (config) => (
            <ModelCard
              key={config.name}
              config={config}
            />
          ),
        )}

      </div>

      {/* ================================================= */}
      {/* ENSEMBLE SUMMARY */}
      {/* ================================================= */}

      <Card className="p-6 border-border/40 bg-gradient-to-br from-primary/10 to-background/40 backdrop-blur-sm">

        <div className="grid grid-cols-1 md:grid-cols-5 gap-6">

          {/* ============================================= */}
          {/* ENSEMBLE SCORE */}
          {/* ============================================= */}

          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold flex items-center gap-1.5 mb-2">

              <Zap className="h-3.5 w-3.5 text-cyber-cyan" />

              Ensemble Score

            </p>

            <p className="text-2xl font-mono font-bold text-cyber-cyan">

              {formatScore(ensembleScore)}

            </p>
          </div>

          {/* ============================================= */}
          {/* CLASSIFICATION */}
          {/* ============================================= */}

          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold mb-2">

              Final Classification

            </p>

            <p className="text-2xl font-mono font-bold text-foreground">

              {models.ensemble.classification || "UNKNOWN"}

            </p>
          </div>

          {/* ============================================= */}
          {/* RISK */}
          {/* ============================================= */}

          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold mb-2">

              Risk Level

            </p>

            <Badge
              className={`font-mono text-sm mt-1 ${
                riskLevel ===
                  "CRITICAL" ||
                riskLevel ===
                  "HIGH"
                  ? "bg-danger/20 text-danger border-danger/40"
                  : riskLevel ===
                      "MEDIUM"
                    ? "bg-warning/20 text-warning border-warning/40"
                    : riskLevel === "LOW"
                      ? "bg-safe/20 text-safe border-safe/40"
                      : "bg-muted/20 text-muted-foreground border-border/40"
              }`}
              variant="outline"
            >
              {riskLevel}
            </Badge>
          </div>

          {/* ============================================= */}
          {/* ATTACK VOTES */}
          {/* ============================================= */}

          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold mb-2">

              Attack Votes

            </p>

            <p className="text-2xl font-mono font-bold text-foreground">

              {attackVotes}/{totalModels}

            </p>
          </div>

          {/* ============================================= */}
          {/* PRIMARY MODEL */}
          {/* ============================================= */}

          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold mb-2">

              Primary Model

            </p>

            <p className="text-sm font-mono font-bold text-cyber-cyan">

              {models.ensemble.primary_model ||
                "CNN-LSTM"}

            </p>
          </div>

        </div>

      </Card>

    </section>
  );
}

// =========================================================
// MODEL CARD PROPS
// =========================================================

interface ModelCardProps {
  config: {
    name: string;

    weight: number;

    icon: string;

    description: string;

    data?: {
      score?: number;

      prediction?: string;
    };

    isPrimary?: boolean;
  };
}

// =========================================================
// MODEL CARD
// =========================================================

function ModelCard({
  config,
}: ModelCardProps) {

  const score =
    normalizeScore(
      config.data?.score,
    );

  const rawPrediction =
    config.data?.prediction;

  const predictionLabel =
    typeof rawPrediction === "string" &&
    rawPrediction.length > 0
      ? rawPrediction
      : "—";

  const normalizedPrediction =
    predictionLabel.toUpperCase();

  const isPredictionAttack =
    normalizedPrediction.includes("ATTACK") ||
    normalizedPrediction.includes("MAL");

  const isPredictionBenign =
    normalizedPrediction === "BENIGN";

  return (
    <Card
      className={`p-5 border-border/40 backdrop-blur-sm hover:border-border/60 transition-colors ${
        config.isPrimary
          ? "bg-gradient-to-br from-primary/15 to-background/40 border-primary/50"
          : "bg-gradient-to-br from-background/60 to-background/40"
      }`}
    >

      <div className="space-y-4">

        {/* ================================================= */}
        {/* HEADER */}
        {/* ================================================= */}

        <div className="flex items-start justify-between">

          <div>

            <div className="flex items-center gap-2 mb-1">

              <span className="text-2xl">
                {config.icon}
              </span>

              <div>

                <h3 className="font-bold text-sm flex items-center gap-2">

                  {config.name}

                  {config.isPrimary && (
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyber-cyan/20 text-cyber-cyan font-semibold">
                      PRIMARY
                    </span>
                  )}

                </h3>

              </div>

            </div>

            <p className="text-[10px] text-muted-foreground">
              {config.description}
            </p>

          </div>

          {/* MODEL WEIGHT */}

          <Badge className="bg-cyber-cyan/10 text-cyber-cyan border-cyber-cyan/30 font-mono text-xs">

            {config.weight}%

          </Badge>

        </div>

        {/* ================================================= */}
        {/* SCORE */}
        {/* ================================================= */}

        <div className="space-y-2">

          <div>

            <p className="text-[9px] uppercase tracking-wider text-muted-foreground font-semibold mb-1">
              Score
            </p>

            <div className="flex items-center gap-2">

              {/* ----------------------------------------- */}
              {/* CORRECT SCORE DISPLAY */}
              {/* ----------------------------------------- */}

              <p className="text-lg font-mono font-bold text-cyber-cyan">

                {formatScore(score)}

              </p>

              {/* ----------------------------------------- */}
              {/* SCORE BAR */}
              {/* ----------------------------------------- */}

              <div className="flex-1 h-1.5 bg-background rounded-full overflow-hidden">

                <div
                  className="h-full bg-gradient-to-r from-cyber-cyan to-primary"
                  style={{
                    width: `${score ?? 0}%`,
                  }}
                />

              </div>

            </div>

          </div>

          {/* ================================================= */}
          {/* PREDICTION */}
          {/* ================================================= */}

          <div>

            <p className="text-[9px] uppercase tracking-wider text-muted-foreground font-semibold mb-1">
              Prediction
            </p>

            <Badge
              className={`font-mono text-xs ${
                isPredictionAttack
                  ? "bg-danger/20 text-danger border-danger/40"
                  : isPredictionBenign
                    ? "bg-safe/20 text-safe border-safe/40"
                    : "bg-muted/20 text-muted-foreground border-border/40"
              }`}
              variant="outline"
            >
              {predictionLabel}
            </Badge>

          </div>

        </div>

      </div>

    </Card>
  );
}