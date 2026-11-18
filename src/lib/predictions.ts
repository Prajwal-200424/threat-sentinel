// =========================================================
// PREDICTION TYPES
// =========================================================

export type ModelPrediction = {
  score: number;
  prediction: string;
};

export type AutoencoderPrediction = {
  score: number;
  prediction: string;
  reconstruction_error?: number;
};

export type EnsembleResult = {
  ensemble_score: number;
  classification: string;
  risk_level: string;
  attack_votes: number;
  total_models: number;
  primary_model?: string;
};

export type ModelResults = {
  cnn_lstm: ModelPrediction;

  random_forest: ModelPrediction;

  autoencoder: AutoencoderPrediction;

  ensemble: EnsembleResult;
};


// =========================================================
// MAIN PREDICTION
// =========================================================

export interface Prediction {
  id: number;
  created_at: number;
  timestamp: string;
  source_ip: string;
  destination_ip: string;
  protocol: string;
  prediction: string;
  confidence_score: number;
  attack_probability?: number;
  status: string;
  risk_level?: string;
  classification?: "BENIGN" | "KNOWN_ATTACK" | "POTENTIAL_ZERO_DAY" | string;
  is_zero_day_candidate?: boolean;
  model_name?: string;
  models?: ModelResults;
  ensemble_score?: number;
  attack_votes?: number;
  total_models?: number;
  attack_type?: string;
  scenario?: string;
  scenario_label?: string;
  device?: string;
  device_name?: string;
  ip_address?: string;
  device_status?: string;
  digital_twin?: string;
  last_update?: string;
}


// =========================================================
// API
// =========================================================

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000"
).replace(/\/+$/, "");

const API_URL = `${API_BASE_URL}/api/predictions`;


// =========================================================
// FETCH PREDICTIONS
// =========================================================

export async function fetchPredictions(): Promise<
  Prediction[]
> {
  const response = await fetch(
    API_URL,
    {
      cache: "no-store",
    },
  );


  if (!response.ok) {
    throw new Error(
      `ML server returned ${response.status}`,
    );
  }


  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new Error("ML server returned invalid JSON");
  }


  // Flask returns:
  //
  // {
  //   predictions: [...]
  // }
  //
  // Support both:
  //
  // [...]
  //
  // and:
  //
  // { predictions: [...] }

  const predictions: unknown[] = Array.isArray(data)
    ? data
    : data && typeof data === "object" && Array.isArray((data as { predictions?: unknown }).predictions)
      ? (data as { predictions: unknown[] }).predictions
      : (() => { throw new Error("ML server returned an invalid predictions payload"); })();


  return predictions
    .filter(
      (prediction): prediction is Prediction =>
        Boolean(
          prediction &&
          typeof prediction === "object" &&
          typeof (prediction as Prediction).prediction === "string" &&
          typeof (prediction as Prediction).timestamp === "string",
        ),
    )
    .sort(
      (a, b) =>
        Number(
          b.created_at ?? 0,
        ) -
        Number(
          a.created_at ?? 0,
        ),
    );
}


// =========================================================
// VERDICT
// =========================================================

// Only the three verdicts the ensemble can actually produce are treated as
// recognised states. Anything else ("UNAVAILABLE", a missing field, a typo or
// an unknown future label) is reported as UNKNOWN instead of being rendered as
// if it were benign.

export type PredictionVerdict =
  | "ATTACK"
  | "SUSPICIOUS"
  | "BENIGN"
  | "UNKNOWN";

export function getVerdict(
  prediction: Prediction,
): PredictionVerdict {
  const value = String(
    prediction.prediction ?? "",
  ).toUpperCase();

  if (value === "ATTACK") return "ATTACK";
  if (value === "SUSPICIOUS") return "SUSPICIOUS";
  if (value === "BENIGN") return "BENIGN";

  return "UNKNOWN";
}


// =========================================================
// ATTACK CHECK
// =========================================================

export function isAttack(
  prediction: Prediction,
): boolean {
  return (
    prediction.prediction
      ?.toUpperCase() ===
    "ATTACK"
  );
}


// =========================================================
// BENIGN CHECK
// =========================================================

export function isBenign(
  prediction: Prediction,
): boolean {
  return (
    prediction.prediction
      ?.toUpperCase() ===
    "BENIGN"
  );
}


// =========================================================
// SUSPICIOUS CHECK
// =========================================================

export function isSuspicious(
  prediction: Prediction,
): boolean {
  return (
    prediction.prediction
      ?.toUpperCase() ===
    "SUSPICIOUS"
  );
}


// =========================================================
// ENSEMBLE SCORE
// =========================================================

// Returns null when the API did not provide a usable score, so callers can show
// "—" instead of inventing a 0.00% reading.

export function getEnsembleScoreOrNull(
  prediction: Prediction,
): number | null {
  const raw =
    prediction.ensemble_score ??
    prediction.models?.ensemble
      ?.ensemble_score;

  if (raw === null || raw === undefined) {
    return null;
  }

  const value = Number(raw);

  return Number.isFinite(value)
    ? value
    : null;
}

export function getEnsembleScore(
  prediction: Prediction,
): number {
  return (
    getEnsembleScoreOrNull(prediction) ??
    0
  );
}

// Formats an ensemble score for display.

export function formatScore(
  score: number | null,
  digits = 2,
): string {
  if (score === null || !Number.isFinite(score)) {
    return "—";
  }

  return `${score.toFixed(digits)}%`;
}


// =========================================================
// MODEL SCORE
// =========================================================

export function getModelScore(
  prediction: Prediction,
  model:
    | "cnn_lstm"
    | "random_forest"
    | "autoencoder",
): number {
  return Number(
    prediction.models?.[model]
      ?.score ?? 0,
  );
}


// =========================================================
// MODEL PREDICTION
// =========================================================

export function getModelPrediction(
  prediction: Prediction,
  model:
    | "cnn_lstm"
    | "random_forest"
    | "autoencoder",
): string {
  return (
    prediction.models?.[model]
      ?.prediction ?? "—"
  );
}


// =========================================================
// ATTACK VOTES
// =========================================================

export function getAttackVotes(
  prediction: Prediction,
): number {
  return Number(
    prediction.attack_votes ??
      prediction.models?.ensemble
        ?.attack_votes ??
      0,
  );
}


// =========================================================
// TOTAL MODELS
// =========================================================

export function getTotalModels(
  prediction: Prediction,
): number {
  return Number(
    prediction.total_models ??
      prediction.models?.ensemble
        ?.total_models ??
      3,
  );
}


// =========================================================
// RISK LEVEL
// =========================================================

export function getRiskLevel(
  prediction: Prediction,
): string {
  // A missing risk level is unknown, not "LOW".
  return (
    prediction.risk_level ??
    prediction.models?.ensemble
      ?.risk_level ??
    "UNKNOWN"
  );
}


// =========================================================
// TIME FORMAT
// =========================================================

export function formatTime(
  prediction: Prediction,
): string {
  const timestamp =
    prediction.timestamp ||
    new Date(
      prediction.created_at,
    ).toISOString();


  return timestamp
    .replace("T", " ")
    .split(".")[0];
}
