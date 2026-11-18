/**
 * ML Model registry + metrics layer.
 *
 * The dashboard prefers the real benchmark results from the training pipeline and
 * falls back to a configured API endpoint only when one is explicitly set.
 */

import trainedModelMetrics from "../../training/model_metrics.json";

export type ModelId =
  | "random_forest"
  | "xgboost"
  | "svm"
  | "isolation_forest"
  | "autoencoder"
  | "cnn"
  | "cnn_lstm";

export interface ModelInfo {
  id: ModelId;
  name: string;
  family: "Classical ML" | "Anomaly Detection" | "Deep Learning";
  description: string;
  isProposed?: boolean;
}

export interface ModelMetrics {
  model_id: ModelId;
  accuracy: number; // 0..1
  precision: number;
  recall: number;
  f1_score: number;
  detection_rate: number;
  false_positive_rate: number;
  /** Real benchmark source: training export or evaluation API */
  source: "api" | "trained";
  evaluated_at?: string;
}

export const MODELS: ModelInfo[] = [
  {
    id: "random_forest",
    name: "Random Forest",
    family: "Classical ML",
    description: "Bagged decision-tree ensemble baseline.",
  },
  {
    id: "xgboost",
    name: "XGBoost",
    family: "Classical ML",
    description: "Gradient-boosted trees, strong tabular baseline.",
  },
  {
    id: "svm",
    name: "SVM",
    family: "Classical ML",
    description: "Support Vector Machine with RBF kernel.",
  },
  {
    id: "isolation_forest",
    name: "Isolation Forest",
    family: "Anomaly Detection",
    description: "Unsupervised outlier isolation for zero-day traffic.",
  },
  {
    id: "autoencoder",
    name: "Autoencoder",
    family: "Anomaly Detection",
    description: "Reconstruction-error based anomaly scoring.",
  },
  {
    id: "cnn",
    name: "CNN",
    family: "Deep Learning",
    description: "1D convolutional spatial feature extractor.",
  },
  {
    id: "cnn_lstm",
    name: "CNN-LSTM",
    family: "Deep Learning",
    description:
      "Proposed hybrid: CNN spatial features + LSTM temporal sequence modelling.",
    isProposed: true,
  },
];

export const PRIMARY_MODEL_ID: ModelId = "cnn_lstm";

export function getModel(id: ModelId): ModelInfo {
  return MODELS.find((m) => m.id === id) ?? MODELS[MODELS.length - 1];
}

/**
 * Set this once the Python model-evaluation API is deployed, e.g.
 * "https://x8ki-letl-twmt.n7.xano.io/api:Qnw6FEMs/model_metrics"
 */
export const MODEL_METRICS_API_URL: string | null = null;

/** Real N-BaIoT evaluation results imported from the training export. */
const normalizeModelMetrics = (
  rows: Array<Record<string, unknown>>,
  source: ModelMetrics["source"] = "trained",
): ModelMetrics[] =>
  rows.map((row) => {
    const modelId = row.model_id as ModelId;
    return {
      model_id: modelId,
      accuracy: Number(row.accuracy ?? 0),
      precision: Number(row.precision ?? 0),
      recall: Number(row.recall ?? 0),
      f1_score: Number((row.f1_score ?? row.f1) ?? 0),
      detection_rate: Number(row.detection_rate ?? row.recall ?? 0),
      false_positive_rate: Number(row.false_positive_rate ?? 0),
      source,
      evaluated_at: typeof row.evaluated_at === "string" ? row.evaluated_at : undefined,
    };
  });

const EVALUATED_METRICS: ModelMetrics[] = normalizeModelMetrics(
  trainedModelMetrics as Array<Record<string, unknown>>,
  "trained",
);

export async function fetchModelMetrics(): Promise<ModelMetrics[]> {
  if (!MODEL_METRICS_API_URL) return EVALUATED_METRICS;

  try {
    const res = await fetch(MODEL_METRICS_API_URL, { cache: "no-store" });
    if (!res.ok) throw new Error("metrics request failed");
    const data = (await res.json()) as Array<Record<string, unknown>>;
    return normalizeModelMetrics(data, "api");
  } catch {
    return EVALUATED_METRICS;
  }
}

export function hasRealMetrics(metrics: ModelMetrics[]) {
  return metrics.some((m) => m.source === "api" || m.source === "trained");
}

export function pct(v: number) {
  return `${(v * 100).toFixed(2)}%`;
}
