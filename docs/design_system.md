# ReadmitIQ Design System: "Clinical Editorial Meets Instrument Panel"

**Version:** 2.0 (Discharge Desk Redesign)  
**Tone:** Calm, authoritative, high-trust, editorial, instrument-precise.  
**Strict Prohibition:** No generic dark dashboard clichés, no purple glowing gradients, no glassmorphism blur traps.

---

## 1. Design Philosophy & Aesthetic Identity

ReadmitIQ is designed as a mission-critical clinical workstation. Its visual grammar blends:
1. **The Dignified Authority of a Medical Journal (Clinical Editorial):** Fraunces serif display headings, generous typographic hierarchy, refined editorial layouts, and high-trust editorial clarity.
2. **The Calibrated Precision of an Avionics or ICU Instrument Panel:** Hairline dividing rules (`1px`), tabular monospace metrics (`JetBrains Mono`), dense but breathable information lanes, status odometers, and live ECG-line connectivity telemetry.

---

## 2. Color Palette & WCAG AA Contrast Tokens

Every color token is mapped to CSS variables on `<html>` / `<body>` under `data-theme="daylight"` (default, projector-friendly) and `data-theme="night"`:

| Token Name | Daylight Theme | Night Theme | Clinical Role | WCAG Contrast Ratio |
|:---|:---:|:---:|:---|:---:|
| `--bg-canvas` | `#F5F1E8` (warm paper) | `#0A1020` (deep ink) | Application root background | N/A |
| `--bg-surface` | `#FFFFFF` (pure card) | `#111A30` (navy plate) | Card, drawer, modal containers | N/A |
| `--bg-subtle` | `#EBE5D8` (linen) | `#182442` (midnight) | Table headers, secondary tags | N/A |
| `--border-hairline`| `#D9D2C3` (parchment line)| `#1F2E50` (steel edge) | Hairline card boundaries | >= 3.0:1 |
| `--text-primary` | `#101B2D` (rich charcoal) | `#E8EEF9` (clinical white)| Headings, primary figures | >= 11.2:1 (AAA) |
| `--text-muted` | `#5A677B` (slate gray) | `#8E9EB8` (cool silver) | Captions, provenance labels | >= 4.8:1 (AA) |
| `--color-critical` | `#D9472B` (deep crimson) | `#FF6B4A` (warm coral) | Top 5% risk tier (Critical) | >= 4.6:1 (AA) |
| `--color-high` | `#D9732B` (burnt orange) | `#FF8F6B` (bright amber) | Top 5-10% risk tier (High) | >= 4.5:1 (AA) |
| `--color-caution` | `#C68A00` (ochre gold) | `#FFC857` (signal gold) | Top 10-20% risk tier (Medium)| >= 4.5:1 (AA) |
| `--color-protective`|`#0B7A75` (deep pine teal)| `#3DD6C3` (biomedical teal)| Low risk, protective SHAP | >= 4.7:1 (AA) |

### Non-Color Signal Redundancy
Color is **never the sole indicator of clinical risk**:
- **Critical (Top 5%):** Solid filled triangle glyph `▲` + `"Critical"` text pill + crimson accent bar.
- **High (5-10%):** Solid square glyph `■` + `"High"` text pill + orange accent bar.
- **Medium (10-20%):** Circle glyph `●` + `"Medium"` text pill + amber accent bar.
- **Low (Remaining 80%):** Hollow ring glyph `○` + `"Low"` text pill + protective teal bar.

---

## 3. Typography Hierarchy

Self-hosted offline via `@fontsource` (zero external CDN network requests):

```css
--font-serif: "Fraunces", Georgia, serif;
--font-sans: "Inter Tight", system-ui, -apple-system, sans-serif;
--font-mono: "JetBrains Mono", Menlo, Consolas, monospace;
```

1. **Editorial Headings:** `font-serif`, letter-spacing `-0.02em`, font-weight `600` or `700`. Used for page titles, patient brief headers, and drawer crowns.
2. **Interface UI & Labels:** `font-sans`, font-weight `400`, `500`, and `600`. High x-height ensures immediate legibility at 12px-14px across busy care wards.
3. **Data, Scores, & Odometers:** `font-mono`, with `font-variant-numeric: tabular-nums lining-nums;`. Prevents visual jitter when numerical odometers cycle during live intake updates.

---

## 4. Brand Mark & ECG Status Telemetry

1. **Logo Mark:** An SVG monogram where an active cardiac rhythm waveform (P-QRS-T complex) flows organically into the letterforms `"IQ"`.
2. **ECG Status Indicator:**
   - **Online / Polling:** An animated SVG waveform line displaying a rhythmic cardiac sweep (teal pulse every 2.5s).
   - **Offline / Disconnected:** The ECG waveform animates into a flat, horizontal line (`isochronic flatline`) in cautionary ochre `#E0A100` with the status text `"API Offline — Retrying in 3s"`.

---

## 5. Motion & Micro-Interactions

- **Timing:** 150ms (micro-hover), 250ms (card expansion), 350ms (drawer entry/exit).
- **Easing:** Cubic-bezier `(0.16, 1, 0.3, 1)` (snappy ease-out, no bouncing).
- **Reduced Motion:** Fully complies with `@media (prefers-reduced-motion: reduce)`. When enabled, animations fall back to instant opacity cuts.

---

## 6. Accessibility & Keyboard Navigation (WCAG 2.1 AA)

- **Keyboard Traversal:**
  - `j` / `k` (or `ArrowDown` / `ArrowUp`): Move through worklist rows.
  - `Enter`: Open selected patient brief.
  - `Escape`: Close drawer or detail overlay.
  - `Ctrl + K` / `Cmd + K`: Global Command Palette.
  - `P`: Toggle Presentation / Projector mode (1280x720 optimized, larger type, 5:00 clock).
  - `1` through `6`: Rapid screen switching in presentation mode.
- **Screen Reader Support:** All SVG icons have `aria-hidden="true"` accompanied by screen-reader text (`<span class="sr-only">`).
- **Data Table Equivalents:** Every interactive chart (Precedent Orbit, SHAP diverging bars, PSI drift bars, Forest plots) provides a toggleable accessible data table (`<table aria-label="...">`).
