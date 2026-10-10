# ReadmitIQ Structured RAG Retrieval Architecture

ReadmitIQ implements a Structured Retrieval-Augmented Generation (RAG) pipeline designed for patient cohort discovery and precedent case explanation.

---

## 1. Vector Projection & Latent Space
- **Dense Representation:** Categorical and numerical variables are encoded into a high-dimensional feature space ($D = 141$ dimensions) via standard scaling and one-hot encoding.
- **Dimensionality Reduction:** Truncated Singular Value Decomposition (SVD) reduces the space to $d = 64$ latent clinical dimensions (`models/final/structured_svd.joblib`), preserving principal components while minimizing noise.
- **L2 Normalization:** Embedding vectors $\mathbf{v}$ are unit-normalized:
  $$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$
  Ensuring Euclidean inner products in FAISS directly compute cosine similarity:
  $$\cos(\theta) = \hat{\mathbf{u}} \cdot \hat{\mathbf{v}}$$

---

## 2. FAISS Index Configuration
- **Index Architecture:** `faiss.IndexFlatIP` (Exact brute-force Inner Product).
- **Rationale for Exact Search:** With $N = 69,535$ training vectors in 64 dimensions, query latency is $< 2.5\text{ ms}$ on standard laptop CPUs. Exact search guarantees 100% recall with zero approximation error.
- **Artifact Size:** The serialized index file is $17.8\text{ MB}$ (`models/final/retriever_faiss.index`), fitting comfortably within ordinary Git storage without requiring Git LFS.

---

## 3. Disjoint Precedent Reference Split
- **Leakage Prevention:** To maintain scientific integrity, the vector store index is populated exclusively from encounters in the **training split** ($n = 69,535$).
- **Held-out Split Isolation:** Zero encounters from the validation ($n = 17,328$) or test ($n = 14,903$) splits are indexed, preventing test contamination.
- **Encounter Metadata:** Metadata records (`models/final/retriever_meta.parquet`) store non-PHI encounter facts: age, gender, admission length, primary diagnosis, and verified 30-day readmission status.

---

## 4. Degraded Mode & Graceful Fallback
- If the FAISS index or reference metadata is unavailable or damaged, the API degrades gracefully:
  - Inference continues with tabular gradient boosting.
  - An explicit degraded status disclosure (`degraded_mode: true`) is attached to API responses.
  - The system rejects fabricating synthetic peer neighbors or hallucinating historical precedent cases.
