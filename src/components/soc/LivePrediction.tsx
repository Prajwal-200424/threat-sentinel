import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Radio,
  ShieldAlert,
  ShieldCheck,
  Brain,
  TreePine,
  Activity,
} from "lucide-react";

import type { Prediction } from "@/lib/predictions";

import {
  isAttack,
  isSuspicious,
  formatTime,
  getEnsembleScore,
  getModelScore,
  getModelPrediction,
  getAttackVotes,
  getTotalModels,
  getRiskLevel,
} from "@/lib/predictions";

interface LivePredictionProps {
  latest: Prediction | null;
}


// =========================================================
// LIVE PREDICTION
// =========================================================

export function LivePrediction({
  latest,
}: LivePredictionProps) {

  // -------------------------------------------------------
  // NO DATA
  // -------------------------------------------------------

  if (!latest) {
    return (
      <Card className="p-6 border-border/60 gradient-cyber">

        <div className="flex items-center gap-3">

          <Radio className="h-5 w-5 text-cyber-cyan" />

          <div>

            <p className="text-sm font-semibold">
              Live Prediction
            </p>

            <p className="text-xs text-muted-foreground mt-1">
              Awaiting live prediction stream…
            </p>

          </div>

        </div>

      </Card>
    );
  }


  // =======================================================
  // FINAL ENSEMBLE RESULT
  // =======================================================

  const attack = isAttack(latest);

  const suspicious = isSuspicious(latest);

  const ensembleScore =
    getEnsembleScore(latest);

  const attackVotes =
    getAttackVotes(latest);

  const totalModels =
    getTotalModels(latest);

  const riskLevel =
    getRiskLevel(latest);


  // =======================================================
  // MODEL SCORES
  // =======================================================

  const cnnScore =
    getModelScore(
      latest,
      "cnn_lstm",
    );

  const rfScore =
    getModelScore(
      latest,
      "random_forest",
    );

  const autoencoderScore =
    getModelScore(
      latest,
      "autoencoder",
    );


  // =======================================================
  // MODEL PREDICTIONS
  // =======================================================

  const cnnPrediction =
    getModelPrediction(
      latest,
      "cnn_lstm",
    );

  const rfPrediction =
    getModelPrediction(
      latest,
      "random_forest",
    );

  const autoencoderPrediction =
    getModelPrediction(
      latest,
      "autoencoder",
    );


  // =======================================================
  // ATTACK TYPE
  // =======================================================

  const attackType =
    latest.attack_type ||
    (
      attack
        ? latest.scenario_label ||
          latest.scenario ||
          "Malicious Activity"
        : "None"
    );


  // =======================================================
  // CARD STATE
  // =======================================================

  const cardClass = attack
    ? "border-danger/60 gradient-danger animate-scan"
    : suspicious
      ? "border-yellow-500/60 bg-yellow-500/5"
      : "border-safe/50 gradient-safe";


  // =======================================================
  // BADGE
  // =======================================================

  const badgeClass = attack
    ? "border-danger text-danger text-glow-danger animate-pulse-danger"
    : suspicious
      ? "border-yellow-500 text-yellow-500"
      : "border-safe text-safe text-glow-safe";


  const badgeText = attack
    ? "● THREAT"
    : suspicious
      ? "● SUSPICIOUS"
      : "● BENIGN";


  // =======================================================
  // MAIN
  // =======================================================

  return (

    <Card
      className={`relative overflow-hidden p-6 border ${cardClass}`}
    >

      {/* ================================================= */}
      {/* HEADER */}
      {/* ================================================= */}

      <div className="flex items-center justify-between mb-6">

        <div className="flex items-center gap-2">

          <Radio
            className={`h-4 w-4 ${
              attack
                ? "text-danger animate-blink"
                : suspicious
                  ? "text-yellow-500"
                  : "text-safe"
            }`}
          />

          <h3 className="text-sm uppercase tracking-widest font-semibold text-muted-foreground">
            Live Prediction
          </h3>

        </div>


        <Badge
          variant="outline"
          className={badgeClass}
        >
          {badgeText}
        </Badge>

      </div>


      {/* ================================================= */}
      {/* FINAL CLASSIFICATION */}
      {/* ================================================= */}

      <div className="flex items-center gap-4 mb-7">

        <div
          className={`flex h-14 w-14 items-center justify-center rounded-full border ${
            attack
              ? "border-danger/50 bg-danger/10"
              : suspicious
                ? "border-yellow-500/50 bg-yellow-500/10"
                : "border-safe/50 bg-safe/10"
          }`}
        >

          {attack ? (

            <ShieldAlert className="h-7 w-7 text-danger" />

          ) : suspicious ? (

            <Activity className="h-7 w-7 text-yellow-500" />

          ) : (

            <ShieldCheck className="h-7 w-7 text-safe" />

          )}

        </div>


        <div>

          <p className="text-[10px] uppercase tracking-widest text-muted-foreground">
            Final Classification
          </p>

          <p
            className={`text-2xl font-bold ${
              attack
                ? "text-danger text-glow-danger"
                : suspicious
                  ? "text-yellow-500"
                  : "text-safe text-glow-safe"
            }`}
          >
            {latest.prediction || "UNKNOWN"}
          </p>

          <p className="text-xs text-muted-foreground mt-1">

            {attack
              ? "Malicious network behaviour detected"
              : suspicious
                ? "Potentially anomalous behaviour detected"
                : "Network behaviour appears normal"}

          </p>

        </div>

      </div>


      {/* ================================================= */}
      {/* ENSEMBLE SUMMARY */}
      {/* ================================================= */}

      <div className="rounded-lg border border-border/50 bg-background/30 p-5 mb-6">

        <div className="flex items-center justify-between mb-4">

          <div>

            <p className="text-[10px] uppercase tracking-widest text-muted-foreground">
              Ensemble Threat Score
            </p>

            <p
              className={`text-3xl font-bold mt-1 ${
                attack
                  ? "text-danger text-glow-danger"
                  : suspicious
                    ? "text-yellow-500"
                    : "text-safe text-glow-safe"
              }`}
            >
              {ensembleScore.toFixed(2)}%
            </p>

          </div>


          <div className="text-right">

            <p className="text-[10px] uppercase tracking-widest text-muted-foreground">
              Risk Level
            </p>

            <p
              className={`text-lg font-bold mt-1 ${
                riskLevel === "CRITICAL"
                  ? "text-danger"
                  : riskLevel === "HIGH"
                    ? "text-orange-500"
                    : riskLevel === "MEDIUM"
                      ? "text-yellow-500"
                      : "text-safe"
              }`}
            >
              {riskLevel}
            </p>

          </div>

        </div>


        <div className="grid grid-cols-2 gap-4">

          <Field
            label="Attack Votes"
            value={`${attackVotes}/${totalModels}`}
            accent={
              attack
                ? "text-danger"
                : "text-safe"
            }
          />

          <Field
            label="Primary Model"
            value={
              latest.models?.ensemble
                ?.primary_model ||
              "CNN-LSTM"
            }
          />

        </div>

      </div>


      {/* ================================================= */}
      {/* MODEL EVIDENCE */}
      {/* ================================================= */}

      <div className="mb-6">

        <div className="flex items-center gap-2 mb-4">

          <Brain className="h-4 w-4 text-cyber-cyan" />

          <h4 className="text-xs uppercase tracking-widest font-semibold text-muted-foreground">
            Model Evidence
          </h4>

        </div>


        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">


          {/* ============================================= */}
          {/* CNN-LSTM */}
          {/* ============================================= */}

          <ModelCard
            title="CNN-LSTM"
            icon={
              <Brain className="h-4 w-4" />
            }
            score={cnnScore}
            prediction={cnnPrediction}
            primary
          />


          {/* ============================================= */}
          {/* RANDOM FOREST */}
          {/* ============================================= */}

          <ModelCard
            title="Random Forest"
            icon={
              <TreePine className="h-4 w-4" />
            }
            score={rfScore}
            prediction={rfPrediction}
          />


          {/* ============================================= */}
          {/* AUTOENCODER */}
          {/* ============================================= */}

          <ModelCard
            title="Autoencoder"
            icon={
              <Activity className="h-4 w-4" />
            }
            score={autoencoderScore}
            prediction={autoencoderPrediction}
            reconstructionError={
              latest.models?.autoencoder
                ?.reconstruction_error
            }
          />

        </div>

      </div>


      {/* ================================================= */}
      {/* CORE INFORMATION */}
      {/* ================================================= */}

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">

        <Field
          label="Prediction"
          value={latest.prediction}
          accent={
            attack
              ? "text-danger"
              : suspicious
                ? "text-yellow-500"
                : "text-safe"
          }
        />

        <Field
          label="Confidence"
          value={`${(
            Number(
              latest.confidence_score ??
                0,
            ) * 100
          ).toFixed(2)}%`}
          accent="text-cyber-cyan"
        />

        <Field
          label="Ensemble Score"
          value={`${ensembleScore.toFixed(2)}%`}
          accent={
            attack
              ? "text-danger"
              : "text-safe"
          }
        />

        <Field
          label="Model Count"
          value={`${totalModels}`}
        />

        <Field
          label="Risk Level"
          value={riskLevel}
          accent={
            attack
              ? "text-danger"
              : suspicious
                ? "text-yellow-500"
                : "text-safe"
          }
        />

        <Field
          label="Attack Type"
          value={
            attack
              ? attackType
              : "None"
          }
          accent={
            attack
              ? "text-danger"
              : "text-safe"
          }
        />

      </div>


      {/* ================================================= */}
      {/* DEVICE / NETWORK */}
      {/* ================================================= */}

      <div className="mt-6 pt-5 border-t border-border/40">

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">

          <Field
            label="Device / Source"
            value={
              latest.source_ip
            }
            mono
          />

          <Field
            label="Destination"
            value={
              latest.destination_ip
            }
            mono
          />

          <Field
            label="Protocol"
            value={
              latest.protocol
            }
          />

          <Field
            label="Status"
            value={
              latest.status ||
              latest.device_status
            }
            accent={
              attack
                ? "text-danger"
                : suspicious
                  ? "text-yellow-500"
                  : "text-safe"
            }
          />

        </div>

      </div>


      {/* ================================================= */}
      {/* ATTACK INFORMATION / ZERO-DAY ALERT */}
      {/* ================================================= */}

      {latest.classification === "POTENTIAL_ZERO_DAY" || latest.is_zero_day_candidate ? (
        <div className="mt-5 rounded-md border border-purple-500/40 bg-purple-500/10 px-4 py-3 animate-pulse">
          <div className="flex items-start gap-3">
            <ShieldAlert className="h-5 w-5 text-purple-400 mt-0.5 shrink-0" />
            <div>
              <p className="text-xs uppercase tracking-widest font-semibold text-purple-400">
                ⚠️ POTENTIAL ZERO-DAY ATTACK
              </p>
              <p className="text-sm font-semibold mt-1">
                Unknown attack behavior detected
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                The 50/25/25 ensemble detected anomalous patterns deviating from learned benign baselines that do not match known attack signatures.
              </p>
            </div>
          </div>
        </div>
      ) : attack ? (
        <div className="mt-5 rounded-md border border-danger/40 bg-danger/10 px-4 py-3">
          <div className="flex items-start gap-3">
            <ShieldAlert className="h-5 w-5 text-danger mt-0.5 shrink-0" />
            <div>
              <p className="text-xs uppercase tracking-widest font-semibold text-danger">
                Active Threat Detected
              </p>
              <p className="text-sm font-semibold mt-1">
                {attackType}
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                The 3-model ensemble has classified the current network behaviour as malicious.
              </p>
            </div>
          </div>
        </div>
      ) : null}


      {/* ================================================= */}
      {/* TIMESTAMP */}
      {/* ================================================= */}

      <div className="mt-5 text-xs text-muted-foreground font-mono">

        ⏱ {formatTime(latest)}

      </div>

    </Card>
  );
}


// =========================================================
// MODEL CARD
// =========================================================

function ModelCard({
  title,
  icon,
  score,
  prediction,
  primary = false,
  reconstructionError,
}: {
  title: string;

  icon: React.ReactNode;

  score: number;

  prediction: string;

  primary?: boolean;

  reconstructionError?: number;
}) {

  const attack =
    prediction?.toUpperCase() ===
    "ATTACK";

  const anomaly =
    prediction?.toUpperCase() ===
    "ANOMALY";


  return (

    <div
      className={`rounded-lg border p-4 ${
        attack
          ? "border-danger/40 bg-danger/5"
          : anomaly
            ? "border-yellow-500/40 bg-yellow-500/5"
            : "border-border/50 bg-background/20"
      }`}
    >

      {/* Header */}

      <div className="flex items-center justify-between">

        <div className="flex items-center gap-2">

          <span
            className={
              attack
                ? "text-danger"
                : anomaly
                  ? "text-yellow-500"
                  : "text-cyber-cyan"
            }
          >
            {icon}
          </span>

          <p className="text-xs font-semibold">
            {title}
          </p>

        </div>


        {primary && (

          <Badge
            variant="outline"
            className="text-[9px] border-cyber-cyan text-cyber-cyan"
          >
            PRIMARY
          </Badge>

        )}

      </div>


      {/* Score */}

      <p
        className={`text-2xl font-bold mt-4 ${
          attack
            ? "text-danger"
            : anomaly
              ? "text-yellow-500"
              : "text-safe"
        }`}
      >
        {Number(score ?? 0).toFixed(2)}%
      </p>


      {/* Prediction */}

      <div className="flex items-center justify-between mt-2">

        <p className="text-[10px] uppercase tracking-widest text-muted-foreground">
          Classification
        </p>

        <Badge
          variant="outline"
          className={
            attack
              ? "border-danger text-danger"
              : anomaly
                ? "border-yellow-500 text-yellow-500"
                : "border-safe text-safe"
          }
        >
          {prediction}
        </Badge>

      </div>


      {/* Reconstruction error */}

      {reconstructionError !==
        undefined && (

        <div className="mt-3 pt-3 border-t border-border/30">

          <p className="text-[10px] uppercase tracking-widest text-muted-foreground">
            Reconstruction Error
          </p>

          <p className="text-xs font-mono mt-1 text-muted-foreground">
            {Number(
              reconstructionError,
            ).toFixed(6)}
          </p>

        </div>

      )}

    </div>
  );
}


// =========================================================
// FIELD
// =========================================================

function Field({
  label,
  value,
  accent,
  mono,
}: {
  label: string;

  value?: string | number | null;

  accent?: string;

  mono?: boolean;
}) {

  return (

    <div>

      <p className="text-[10px] uppercase tracking-widest text-muted-foreground">
        {label}
      </p>

      <p
        className={`mt-1 text-sm font-semibold ${
          mono
            ? "font-mono"
            : ""
        } ${
          accent ??
          "text-foreground"
        }`}
      >

        {value !== null &&
        value !== undefined &&
        String(value).length > 0
          ? String(value)
          : "—"}

      </p>

    </div>
  );
}