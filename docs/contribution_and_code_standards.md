# ReadmitIQ Contribution & Code Standards

*Repository Owner & Lead: **Tribhuwan Singh** ([@Tribhuwansingh2023](https://github.com/Tribhuwansingh2023), `webosingh93@gmail.com`)*

Guidelines for maintaining code hygiene, documentation integrity, and scientific rigor across the ReadmitIQ repository.

---

## 1. Frozen Artifact Protection
- **Rule 1:** Never retrain, overwrite, replace, or alter serialized artifacts in `models/final/`.
- **Rule 2:** Never modify or fabricate evaluation metrics in `reports/metrics/`.
- **Rule 3:** All empirical claims in documentation must trace directly to automated pytest tests or verified benchmark JSON files.

---

## 2. Secrets & Privacy Guardrails
- **Zero Secrets Policy:** Never commit `.env` files, API keys, private passwords, internal server IPs, or machine-specific absolute file paths.
- **Zero PHI Policy:** Never upload raw, de-anonymized, or identifiable patient records. All demo data must be derived from public benchmark splits or synthetic schemas.
- **Media Exclusions:** Do not commit screen recordings, videos, browser cache downloads, or transient debugging dumps.

---

## 3. Python Code Standards
- **Python Version:** 3.11 - 3.12 compatibility.
- **Formatting:** Clean PEP 8 formatting with standard type annotations.
- **Type Safety:** Use Pydantic models for request/response payloads in `api/schemas.py`.
- **Testing:** Every new endpoint, mutation, or feature must include automated pytest tests in `tests/`.

---

## 4. Frontend Code Standards
- **Framework:** React 19 + TypeScript + Vite.
- **Styling:** Semantic Tailwind CSS using defined design system tokens (`surface`, `canvas`, `hairline-border`, `accent`, `protective`, `caution`, `critical`, `muted`, `font-mono`, `font-serif`).
- **Type Checking:** Must pass `tsc -b && vite build` with zero TypeScript errors or warnings.
- **Accessibility:** All interactive elements must maintain clear focus rings, legible text contrast, and descriptive `aria-label` tags.
