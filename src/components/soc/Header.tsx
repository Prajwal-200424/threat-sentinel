import { Cpu, Wifi, Shield, Clock } from "lucide-react";

export function SOCHeader({
  isLive,
  lastUpdate,
}: {
  isLive: boolean;
  lastUpdate: Date | null;
}) {
  return (
    <header className="flex flex-col gap-6 pb-8 border-b border-border/60">
      {/* ===== MAIN HEADER ===== */}
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
        {/* Logo and Title */}
        <div className="flex items-start gap-4">
          <div className="relative h-14 w-14 rounded-lg gradient-primary-soft border border-primary/40 flex items-center justify-center shadow-glow-cyber">
            <Shield className="h-7 w-7 text-cyber-cyan text-glow-cyber" />
          </div>
          <div>
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
              <span className="text-cyber-cyan text-glow-cyber">THREAT SENTINEL</span>
            </h1>
            <p className="text-sm uppercase tracking-[0.15em] text-cyber-cyan/70 mt-1">
              Real-Time IoT Anomaly & Malware Detection Platform
            </p>
          </div>
        </div>

        {/* Connection Status */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-4 py-2 rounded-lg border border-border/60 bg-card/60 backdrop-blur-sm">
            <Wifi
              className={`h-4 w-4 ${isLive ? "text-safe animate-blink" : "text-muted-foreground"}`}
            />
            <span className="text-xs font-mono font-semibold">
              {isLive ? "ONLINE" : "OFFLINE"}
            </span>
          </div>
        </div>
      </div>

      {/* ===== INFO ROW ===== */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* ML Engine */}
        <div className="flex items-start gap-3 p-3 rounded-lg border border-border/40 bg-background/40">
          <Cpu className="h-4 w-4 text-cyber-cyan mt-0.5 flex-shrink-0" />
          <div className="min-w-0">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">
              ML Engine
            </p>
            <p className="text-sm font-mono font-bold text-foreground mt-1">
              3-MODEL
            </p>
          </div>
        </div>

        {/* Device */}
        <div className="flex items-start gap-3 p-3 rounded-lg border border-border/40 bg-background/40">
          <Shield className="h-4 w-4 text-cyber-cyan mt-0.5 flex-shrink-0" />
          <div className="min-w-0">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">
              Device
            </p>
            <p className="text-sm font-mono font-bold text-foreground mt-1">
              ESP32-01
            </p>
          </div>
        </div>

        {/* Connection */}
        <div className="flex items-start gap-3 p-3 rounded-lg border border-border/40 bg-background/40">
          <Wifi className="h-4 w-4 text-cyber-cyan mt-0.5 flex-shrink-0" />
          <div className="min-w-0">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">
              Connection
            </p>
            <p className={`text-sm font-mono font-bold mt-1 ${isLive ? "text-safe" : "text-muted-foreground"}`}>
              {isLive ? "ONLINE" : "OFFLINE"}
            </p>
          </div>
        </div>

        {/* Last Update */}
        <div className="flex items-start gap-3 p-3 rounded-lg border border-border/40 bg-background/40">
          <Clock className="h-4 w-4 text-cyber-cyan mt-0.5 flex-shrink-0" />
          <div className="min-w-0">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">
              Last Update
            </p>
            <p className="text-sm font-mono font-bold text-foreground mt-1 truncate">
              {lastUpdate ? lastUpdate.toLocaleTimeString() : "—"}
            </p>
          </div>
        </div>
      </div>
    </header>
  );
}
