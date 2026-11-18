import { AlertTriangle, Cpu, ShieldAlert, Zap } from "lucide-react";
import type { Prediction } from "@/lib/predictions";
import {
  isAttack,
  getEnsembleScoreOrNull,
  formatScore,
  getAttackVotes,
  getTotalModels,
  getRiskLevel,
} from "@/lib/predictions";

export function AlertBanner({ latest }: { latest: Prediction | null }) {
  if (!latest || !isAttack(latest)) return null;

  const name =
    latest.scenario_label ||
    latest.attack_type ||
    latest.scenario ||
    "Malware Attack";

  const ensembleScore = getEnsembleScoreOrNull(latest);
  const riskLevel = getRiskLevel(latest);
  const attackVotes = getAttackVotes(latest);
  const totalModels = getTotalModels(latest);

  return (
    <section className="relative overflow-hidden rounded-lg border border-danger/60 bg-gradient-to-r from-danger/10 via-danger/5 to-background/40 p-6 shadow-glow-danger backdrop-blur-sm">
      {/* Animated left border */}
      <div className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-danger via-danger/50 to-transparent" />

      <div className="flex flex-col gap-6 pl-4">
        {/* Main Alert Header */}
        <div className="flex items-start gap-4">
          <div className="rounded-full bg-danger/15 border border-danger/40 p-3 flex-shrink-0">
            <AlertTriangle className="h-6 w-6 text-danger animate-pulse" />
          </div>

          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <p className="text-[9px] font-semibold uppercase tracking-[0.2em] text-danger">
                🚨 Active Security Alert
              </p>
            </div>
            <h2 className="text-2xl font-bold text-danger mb-1">
              {name} Detected
            </h2>
            <p className="text-sm text-muted-foreground">
              Ensemble classified network behavior as <span className="font-semibold text-danger">malicious</span>. Immediate investigation recommended.
            </p>
            <p className="mt-3 font-mono text-xs text-muted-foreground/70">
              {latest.device || "ESP32-01"} · {latest.source_ip || "DEVICE"} →{" "}
              {latest.destination_ip || "REMOTE"}
            </p>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <AlertMetric
            label="Risk Level"
            value={riskLevel}
            icon={<ShieldAlert className="h-3.5 w-3.5" />}
            highlight={
              riskLevel === "CRITICAL" || riskLevel === "HIGH"
            }
          />
          <AlertMetric
            label="Ensemble Score"
            value={formatScore(ensembleScore)}
            icon={<Zap className="h-3.5 w-3.5" />}
          />
          <AlertMetric
            label="Attack Votes"
            value={`${attackVotes}/${totalModels}`}
            icon={<Cpu className="h-3.5 w-3.5" />}
          />
          <AlertMetric
            label="Protocol"
            value={latest.protocol || "Wi-Fi"}
          />
        </div>
      </div>
    </section>
  );
}

function AlertMetric({
  label,
  value,
  icon,
  highlight,
}: {
  label: string;
  value: string;
  icon?: React.ReactNode;
  highlight?: boolean;
}) {
  return (
    <div
      className={`rounded-lg border ${highlight ? "border-danger/50 bg-danger/10" : "border-danger/20 bg-danger/5"} px-3 py-2.5 backdrop-blur-sm`}
    >
      <p className="flex items-center gap-1.5 text-[8px] uppercase tracking-widest text-muted-foreground font-semibold mb-1">
        {icon}
        {label}
      </p>
      <p className={`font-mono text-sm font-bold ${highlight ? "text-danger" : "text-foreground"}`}>
        {value}
      </p>
    </div>
  );
}
