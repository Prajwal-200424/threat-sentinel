import { AlertTriangle, Shield, AlertCircle } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Prediction } from "@/lib/predictions";
import {
  getVerdict,
  getEnsembleScoreOrNull,
  formatScore,
  getAttackVotes,
  getTotalModels,
  getRiskLevel,
} from "@/lib/predictions";

interface ThreatStatusProps {
  latest: Prediction | null;
}

export function ThreatStatus({ latest }: ThreatStatusProps) {
  if (!latest) {
    return (
      <Card className="p-8 border-border/60 bg-gradient-to-br from-background/80 to-background/40 backdrop-blur-sm">
        <div className="flex items-center gap-4 justify-center text-center">
          <AlertCircle className="h-5 w-5 text-cyber-cyan" />
          <div>
            <p className="text-sm font-semibold">Awaiting Real-Time Prediction</p>
            <p className="text-xs text-muted-foreground mt-1">
              Connecting to ML server…
            </p>
          </div>
        </div>
      </Card>
    );
  }

  const verdict = getVerdict(latest);
  const ensembleScore = getEnsembleScoreOrNull(latest);
  const attackVotes = getAttackVotes(latest);
  const totalModels = getTotalModels(latest);
  const riskLevel = getRiskLevel(latest);

  // Determine threat state. A verdict outside ATTACK / SUSPICIOUS / BENIGN
  // (for example "UNAVAILABLE" while the ensemble has not produced a result) is
  // shown as UNKNOWN, never styled as a safe/benign reading.
  let threatState: "BENIGN" | "SUSPICIOUS" | "ATTACK" | "UNKNOWN" = verdict;
  let stateColor = "text-safe";
  let bgColor = "bg-safe/5";
  let borderColor = "border-safe/30";
  let glowColor = "shadow-glow-safe";
  let Icon = Shield;

  if (verdict === "ATTACK") {
    stateColor = "text-danger";
    bgColor = "bg-danger/5";
    borderColor = "border-danger/40";
    glowColor = "shadow-glow-danger";
    Icon = AlertTriangle;
  } else if (verdict === "SUSPICIOUS") {
    stateColor = "text-warning";
    bgColor = "bg-warning/5";
    borderColor = "border-warning/40";
    glowColor = "";
    Icon = AlertCircle;
  } else if (verdict === "UNKNOWN") {
    threatState = "UNKNOWN";
    stateColor = "text-muted-foreground";
    bgColor = "bg-muted/10";
    borderColor = "border-border/60";
    glowColor = "";
    Icon = AlertCircle;
  }

  return (
    <Card
      className={`p-8 border ${borderColor} ${bgColor} backdrop-blur-sm ${glowColor} transition-all duration-300`}
    >
      <div className="space-y-6">
        {/* Main Threat State */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div className="flex items-start gap-4">
            <div className={`rounded-full p-3 ${bgColor} border ${borderColor}`}>
              <Icon className={`h-8 w-8 ${stateColor}`} />
            </div>
            <div>
              <div className="flex items-center gap-3">
                <h2 className={`text-4xl md:text-5xl font-bold ${stateColor}`}>
                  {threatState}
                </h2>
                <Badge
                  className={`px-3 py-1 text-xs font-mono font-bold uppercase ${stateColor} border ${borderColor} ${bgColor}`}
                  variant="outline"
                >
                  {threatState}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground mt-2">
                {verdict === "UNKNOWN"
                  ? `Current Threat Status · No recognised verdict (API reported "${latest.prediction ?? "missing"}")`
                  : "Current Threat Status · Real-Time Classification"}
              </p>
            </div>
          </div>

          {/* Risk Level Badge */}
          <div className="flex flex-col gap-3">
            <div className="text-right">
              <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">
                Risk Level
              </p>
              <p
                className={`text-2xl font-bold font-mono mt-1 ${
                  riskLevel === "CRITICAL" || riskLevel === "HIGH"
                    ? "text-danger"
                    : riskLevel === "MEDIUM"
                      ? "text-warning"
                      : riskLevel === "LOW"
                        ? "text-safe"
                        : "text-muted-foreground"
                }`}
              >
                {riskLevel}
              </p>
            </div>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricBox
            label="Ensemble Score"
            value={formatScore(ensembleScore)}
            accent="text-cyber-cyan"
          />
          <MetricBox
            label="Attack Votes"
            value={`${attackVotes}/${totalModels}`}
            accent="text-cyber-cyan"
          />
          <MetricBox
            label="Device"
            value={latest.device || "ESP32-01"}
            accent="text-cyber-cyan"
          />
          <MetricBox
            label="Protocol"
            value={latest.protocol || "Wi-Fi"}
            accent="text-cyber-cyan"
          />
        </div>
      </div>
    </Card>
  );
}

function MetricBox({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent: string;
}) {
  return (
    <div className="rounded-lg border border-border/40 bg-background/40 p-3">
      <p className="text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">
        {label}
      </p>
      <p className={`text-sm font-mono font-bold ${accent} mt-1.5 truncate`}>
        {value}
      </p>
    </div>
  );
}
