export const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export type Label = "Urgency" | "Scarcity" | "Social Proof" | "Normal";
export type Probabilities = Record<Label, number>;

export interface ModelPrediction {
  label: Label;
  probabilities: Probabilities;
  top_contributing_words: string[];
}

export interface PredictResponse {
  prediction_id: number;
  label: Label;
  confidence_percent: number;
  per_model: Record<string, ModelPrediction>;
  calibrated_ensemble: ModelPrediction;
  top_contributing_words: string[];
}

export interface HistoryItem {
  id: number;
  created_at: string;
  text_hash: string;
  text: string | null;
  final_label: Label;
  confidence: number;
  per_model_probs: Record<string, unknown>;
}

export interface HistoryResponse {
  page: number;
  page_size: number;
  total: number;
  items: HistoryItem[];
}

export interface FeedbackResponse {
  id: number;
  prediction_id: number;
  user_label: Label;
  comment: string | null;
  created_at: string;
}

export interface ModelMetadata {
  model: string;
  version: string;
  macro_f1: number | null;
  [key: string]: unknown;
}

export interface StatsResponse {
  label_distribution: Partial<Record<Label, number>>;
  average_confidence: number;
  feedback: { total: number; agreement: number; agreement_rate: number };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const predict = (text: string, privacyFlag = false) =>
  request<PredictResponse>("/predict", {
    method: "POST",
    body: JSON.stringify({ text, privacy_flag: privacyFlag }),
  });

export const predictBatch = (texts: string[]) =>
  request<{ predictions: PredictResponse[] }>("/predict/batch", {
    method: "POST",
    body: JSON.stringify({ texts }),
  });

export const getHistory = (page = 1, pageSize = 20) =>
  request<HistoryResponse>(`/history?page=${page}&page_size=${pageSize}`);

export const sendFeedback = (predictionId: number, userLabel: Label, comment?: string) =>
  request<FeedbackResponse>("/feedback", {
    method: "POST",
    body: JSON.stringify({ prediction_id: predictionId, user_label: userLabel, comment }),
  });

export const getModels = async () => (await request<{ models: ModelMetadata[] }>("/models")).models;
export const getStats = () => request<StatsResponse>("/stats");

export interface Explanation {
  text: string;
  model: { name: string; seed: number; checkpoint_sha256: string; tokenizer_sha256: string };
  label: Label;
  confidence: number;
  method: string;
  description: string;
  truncation_warning: string;
  tokens: { text: string; start: number; end: number; score: number }[];
  feature_attributions?: {
    target_class: Label; bias: number; logit: number; description: string;
    features: { feature: string; score: number }[];
  };
  comparison?: {
    text: string; label: Label; confidence: number; prediction_changed: boolean;
    original_class_probability_delta: number; note: string;
  };
}

export const explainText = (text: string, perturbedText?: string) =>
  request<Explanation>("/explain", {
    method: "POST",
    body: JSON.stringify({ text, perturbed_text: perturbedText || null }),
  });
