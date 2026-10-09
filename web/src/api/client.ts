import type {
  CohortResponse,
  ExplainResponse,
  HealthResponse,
  MetricsResponse,
  PatientFormInput,
  PredictResponse,
  SimilarResponse,
} from "../types";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errDetail = `HTTP ${res.status}: ${res.statusText}`;
    try {
      const data = await res.json();
      if (data.detail) {
        if (typeof data.detail === "string") {
          errDetail = data.detail;
        } else {
          errDetail = JSON.stringify(data.detail);
        }
      }
    } catch {
      // ignore json parse error
    }
    throw new Error(errDetail);
  }
  return res.json();
}

export const apiClient = {
  async getHealth(): Promise<HealthResponse> {
    const res = await fetch(`${API_BASE_URL}/health`);
    return handleResponse<HealthResponse>(res);
  },

  async predict(patient: PatientFormInput): Promise<PredictResponse> {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patient),
    });
    return handleResponse<PredictResponse>(res);
  },

  async retrieveSimilar(
    patient: Partial<PatientFormInput>,
    k: number = 20,
  ): Promise<SimilarResponse> {
    const res = await fetch(`${API_BASE_URL}/similar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_features: patient, k }),
    });
    return handleResponse<SimilarResponse>(res);
  },

  async explain(
    patient: Partial<PatientFormInput>,
    topK: number = 3,
  ): Promise<ExplainResponse> {
    const res = await fetch(`${API_BASE_URL}/explain?top_k=${topK}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patient),
    });
    return handleResponse<ExplainResponse>(res);
  },

  async getCohort(capacity: number = 0.1): Promise<CohortResponse> {
    const res = await fetch(
      `${API_BASE_URL}/cohort?capacity=${capacity.toFixed(2)}`,
    );
    return handleResponse<CohortResponse>(res);
  },

  async getMetrics(): Promise<MetricsResponse> {
    const res = await fetch(`${API_BASE_URL}/metrics`);
    return handleResponse<MetricsResponse>(res);
  },
};
