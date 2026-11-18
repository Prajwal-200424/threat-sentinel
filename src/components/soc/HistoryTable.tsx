import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { History } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { Prediction } from "@/lib/predictions";
import {
  getVerdict,
  formatTime,
  getEnsembleScoreOrNull,
  formatScore,
  getRiskLevel,
} from "@/lib/predictions";

export function HistoryTable({
  predictions,
  activeModelName = "CNN-LSTM",
}: {
  predictions: Prediction[];
  activeModelName?: string;
}) {
  return (
    <section className="space-y-4">
      <div className="flex items-center gap-2">
        <History className="h-5 w-5 text-cyber-cyan" />
        <div>
          <h2 className="text-xl font-bold">Detection History</h2>
          <p className="text-xs text-muted-foreground mt-1">
            {predictions.length} total predictions logged
          </p>
        </div>
      </div>

      <Card className="border-border/60 gradient-cyber overflow-hidden">
        <div className="overflow-x-auto max-h-[600px]">
          <Table>
            <TableHeader className="sticky top-0 bg-card/95 backdrop-blur z-10">
              <TableRow className="border-border/60 hover:bg-transparent">
                <TableHead className="text-[10px] uppercase tracking-widest text-cyber-cyan">
                  Time
                </TableHead>
                <TableHead className="text-[10px] uppercase tracking-widest text-cyber-cyan">
                  Device
                </TableHead>
                <TableHead className="text-[10px] uppercase tracking-widest text-cyber-cyan">
                  Scenario
                </TableHead>
                <TableHead className="text-[10px] uppercase tracking-widest text-cyber-cyan">
                  Prediction
                </TableHead>
                <TableHead className="text-[10px] uppercase tracking-widest text-cyber-cyan">
                  Ensemble Score
                </TableHead>
                <TableHead className="text-[10px] uppercase tracking-widest text-cyber-cyan">
                  Risk
                </TableHead>
                <TableHead className="text-[10px] uppercase tracking-widest text-cyber-cyan">
                  Protocol
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {predictions.map((p, index) => {
                const verdict = getVerdict(p);
                const attack = verdict === "ATTACK";
                const suspicious = verdict === "SUSPICIOUS";
                const ensembleScore = getEnsembleScoreOrNull(p);
                const riskLevel = getRiskLevel(p);
                const scenario = p.scenario_label || p.attack_type || p.scenario || "NORMAL";
                // Flask event ids restart once the retained history window is
                // full, so the timestamp keeps React keys unique.
                const rowKey = `${p.created_at ?? p.timestamp ?? "event"}-${p.id ?? index}`;

                return (
                  <TableRow
                    key={rowKey}
                    className={`border-border/40 transition-colors ${
                      attack
                        ? "bg-danger/5 hover:bg-danger/10"
                        : suspicious
                          ? "bg-warning/5 hover:bg-warning/10"
                          : verdict === "BENIGN"
                            ? "bg-safe/5 hover:bg-safe/10"
                            : "bg-muted/5 hover:bg-muted/10"
                    }`}
                  >
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {formatTime(p)}
                    </TableCell>

                    <TableCell className="font-mono text-xs font-semibold">
                      {p.device || "ESP32-01"}
                    </TableCell>

                    <TableCell className="text-xs">
                      <span className="font-mono text-foreground">{scenario}</span>
                    </TableCell>

                    <TableCell>
                      <Badge
                        className={`font-mono text-xs font-bold border ${
                          attack
                            ? "bg-danger/20 text-danger border-danger/40"
                            : suspicious
                              ? "bg-warning/20 text-warning border-warning/40"
                              : verdict === "BENIGN"
                                ? "bg-safe/20 text-safe border-safe/40"
                                : "bg-muted/20 text-muted-foreground border-border/40"
                        }`}
                        variant="outline"
                      >
                        {p.prediction || "UNKNOWN"}
                      </Badge>
                    </TableCell>

                    <TableCell className="font-mono text-xs font-semibold text-cyber-cyan">
                      {formatScore(ensembleScore)}
                    </TableCell>

                    <TableCell>
                      <Badge
                        className={`font-mono text-xs font-bold border ${
                          riskLevel === "CRITICAL" || riskLevel === "HIGH"
                            ? "bg-danger/20 text-danger border-danger/40"
                            : riskLevel === "MEDIUM"
                              ? "bg-warning/20 text-warning border-warning/40"
                              : riskLevel === "LOW"
                                ? "bg-safe/20 text-safe border-safe/40"
                                : "bg-muted/20 text-muted-foreground border-border/40"
                        }`}
                        variant="outline"
                      >
                        {riskLevel}
                      </Badge>
                    </TableCell>

                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {p.protocol || "Wi-Fi"}
                    </TableCell>
                  </TableRow>
                );
              })}
              {predictions.length === 0 && (
                <TableRow>
                  <TableCell
                    colSpan={7}
                    className="text-center text-muted-foreground py-8"
                  >
                    No predictions logged yet. Awaiting data…
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </Card>
    </section>
  );
}
