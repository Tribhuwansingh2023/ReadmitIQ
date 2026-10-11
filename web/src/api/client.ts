import type {
  CohortResponse,
  DischargeDetailResponse,
  ExplainResponse,
  FeedStatusResponse,
  GlobalExplanationResponse,
  HealthResponse,
  MetricsResponse,
  ModelInfoResponse,
  MonitoringResponse,
  PatientFormInput,
  PredictResponse,
  SimilarResponse,
  TaskCreateInput,
  TaskItem,
  TaskPatchInput,
  WhatIfResponse,
  WorklistResponse,
} from "../types";

const getApiBaseUrl = () => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  if (typeof window !== "undefined") {
    const hostname = window.location.hostname || "127.0.0.1";
    // In local standalone dev on port 5173, point directly to FastAPI on port 8000
    if ((hostname === "localhost" || hostname === "127.0.0.1") && window.location.port === "5173") {
      return `http://${hostname}:8000`;
    }
    // In multi-service deployments (e.g. Vercel, Docker Nginx proxy), use same-origin relative URLs
    return "";
  }
  return "http://127.0.0.1:8000";
};

export const API_BASE_URL = getApiBaseUrl();

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

  // --- v1 Worklist & Discharges ---

  async getWorklist(params?: {
    status?: string;
    tier?: string;
    search?: string;
    capacity?: number;
    page?: number;
    pageSize?: number;
  }): Promise<WorklistResponse> {
    const url = new URL(`${API_BASE_URL}/v1/worklist`);
    if (params?.status) url.searchParams.set("status", params.status);
    if (params?.tier) url.searchParams.set("tier", params.tier);
    if (params?.search) url.searchParams.set("search", params.search);
    if (params?.capacity !== undefined)
      url.searchParams.set("capacity", params.capacity.toString());
    if (params?.page) url.searchParams.set("page", params.page.toString());
    if (params?.pageSize)
      url.searchParams.set("page_size", params.pageSize.toString());

    const res = await fetch(url.toString());
    return handleResponse<WorklistResponse>(res);
  },

  async getDischarge(id: number): Promise<DischargeDetailResponse> {
    const res = await fetch(`${API_BASE_URL}/v1/discharges/${id}`);
    return handleResponse<DischargeDetailResponse>(res);
  },

  async createDischarge(
    patient: Partial<PatientFormInput>,
  ): Promise<any> {
    const res = await fetch(`${API_BASE_URL}/v1/discharges`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patient),
    });
    return handleResponse<any>(res);
  },

  async uploadCsv(file: File): Promise<any> {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API_BASE_URL}/v1/discharges/upload`, {
      method: "POST",
      body: formData,
    });
    return handleResponse<any>(res);
  },

  getTemplateCsvUrl(): string {
    return `${API_BASE_URL}/v1/templates/discharge.csv`;
  },

  async updateDischarge(
    id: number,
    data: {
      status?: string;
      assignee?: string;
      clinician_override_tier?: string;
      override_reason?: string;
      actor?: string;
    },
  ): Promise<any> {
    const res = await fetch(`${API_BASE_URL}/v1/discharges/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return handleResponse<any>(res);
  },

  async addAction(
    id: number,
    action: { type: string; note: string; actor?: string },
  ): Promise<any> {
    const res = await fetch(`${API_BASE_URL}/v1/discharges/${id}/actions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(action),
    });
    return handleResponse<any>(res);
  },

  async recordOutcome(
    id: number,
    outcome: { outcome_readmitted_30d: number; actor?: string },
  ): Promise<any> {
    const res = await fetch(`${API_BASE_URL}/v1/discharges/${id}/outcome`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(outcome),
    });
    return handleResponse<any>(res);
  },

  // --- Care Transition Tasks ---

  async getTasks(dischargeId: number): Promise<TaskItem[]> {
    const res = await fetch(`${API_BASE_URL}/v1/discharges/${dischargeId}/tasks`);
    return handleResponse<TaskItem[]>(res);
  },

  async createTask(
    dischargeId: number,
    task: TaskCreateInput,
  ): Promise<TaskItem> {
    const res = await fetch(`${API_BASE_URL}/v1/discharges/${dischargeId}/tasks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(task),
    });
    return handleResponse<TaskItem>(res);
  },

  async updateTask(
    dischargeId: number,
    taskId: number,
    data: TaskPatchInput,
  ): Promise<any> {
    const res = await fetch(
      `${API_BASE_URL}/v1/discharges/${dischargeId}/tasks/${taskId}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      },
    );
    return handleResponse<any>(res);
  },

  // --- What-If Risk Sensitivity ---

  async whatIfSensitivity(data: {
    discharge_id?: number;
    original_features?: Record<string, any>;
    modified_features: Record<string, any>;
  }): Promise<WhatIfResponse> {
    const res = await fetch(`${API_BASE_URL}/v1/what-if`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return handleResponse<WhatIfResponse>(res);
  },

  // --- v1 Simulated Feed & Time Machine ---

  async startFeed(ratePerMinute: number = 60, seed: number = 42): Promise<FeedStatusResponse> {
    const res = await fetch(`${API_BASE_URL}/v1/feed/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rate_per_minute: ratePerMinute, seed }),
    });
    return handleResponse<FeedStatusResponse>(res);
  },

  async stopFeed(): Promise<FeedStatusResponse> {
    const res = await fetch(`${API_BASE_URL}/v1/feed/stop`, {
      method: "POST",
    });
    return handleResponse<FeedStatusResponse>(res);
  },

  async getFeedStatus(): Promise<FeedStatusResponse> {
    const res = await fetch(`${API_BASE_URL}/v1/feed/status`);
    return handleResponse<FeedStatusResponse>(res);
  },

  async advanceTimeMachine(days: number = 30): Promise<any> {
    const res = await fetch(`${API_BASE_URL}/v1/feed/time-machine`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ days }),
    });
    return handleResponse<any>(res);
  },

  // --- v1 Monitoring & Model Info ---

  async getMonitoring(): Promise<MonitoringResponse> {
    const res = await fetch(`${API_BASE_URL}/v1/monitoring`);
    return handleResponse<MonitoringResponse>(res);
  },

  async getModelInfo(): Promise<ModelInfoResponse> {
    const res = await fetch(`${API_BASE_URL}/v1/model/info`);
    return handleResponse<ModelInfoResponse>(res);
  },

  // --- Core / Legacy inference endpoints ---

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

  async getGlobalExplanation(): Promise<GlobalExplanationResponse> {
    try {
      const res = await fetch(`${API_BASE_URL}/v1/explain/global`);
      return await handleResponse<GlobalExplanationResponse>(res);
    } catch {
      // Fallback to core explain/global
      const res = await fetch(`${API_BASE_URL}/explain/global`);
      return handleResponse<GlobalExplanationResponse>(res);
    }
  },

  async getXaiGlobal(): Promise<any> {
    try {
      const res = await fetch(`${API_BASE_URL}/v1/xai/global`);
      return await handleResponse<any>(res);
    } catch {
      const res = await fetch(`${API_BASE_URL}/xai/global`);
      return handleResponse<any>(res);
    }
  },

  async getDischargeCounterfactual(dischargeId: number): Promise<any> {
    const res = await fetch(`${API_BASE_URL}/v1/discharges/${dischargeId}/counterfactual`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    return handleResponse<any>(res);
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

