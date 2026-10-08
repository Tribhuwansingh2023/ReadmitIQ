# ReadmitIQ Validation Ablation Table

Cohort: Train=69,535 encounters, Val=14,902 encounters (UCI 130-US Hospitals).

| Model | RAG Feature | Val AUC (95% CI) | F1 | Precision@10% | Lift@10% | Brier | ECE | Paired Delta AUC (95% CI) | Crosses 0? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | No | 0.6619 [0.6484, 0.6757] | 0.2770 | 0.2376 | 2.09x | 0.2291 | 0.3557 | Ref - | - |
| Logistic Regression | Yes (k=20) | 0.6632 [0.6501, 0.6772] | 0.2766 | 0.2430 | 2.13x | 0.2287 | 0.3554 | +0.0013 [-0.0003, +0.0030] | Yes |
| Random Forest | No | 0.6671 [0.6547, 0.6804] | 0.2795 | 0.2403 | 2.11x | 0.2112 | 0.3348 | Ref - | - |
| Random Forest | Yes (k=20) | 0.6674 [0.6548, 0.6806] | 0.2778 | 0.2510 | 2.20x | 0.2096 | 0.3325 | +0.0003 [-0.0022, +0.0030] | Yes |
| LightGBM | No | 0.6722 [0.6593, 0.6854] | 0.2819 | 0.2557 | 2.25x | 0.2196 | 0.3425 | Ref - | - |
| LightGBM | Yes (k=20) | 0.6731 [0.6602, 0.6863] | 0.2822 | 0.2530 | 2.22x | 0.2194 | 0.3420 | +0.0009 [-0.0016, +0.0036] | Yes |
| DistilBERT (Notes) | None | *Pending Colab run* | - | - | - | - | - | - | - |

### Calibration Performance (LightGBM + RAG)
- **Brier Score**: 0.2194 (uncalibrated) -> **0.0966** (calibrated via Train OOF isotonic regression)
- **Expected Calibration Error (ECE)**: 0.3420 -> **0.0024**

### Key Methodological Findings
- All models evaluated on the FULL validation set (14,902 encounters) with no subsampling.
- Paired bootstrap confidence intervals directly test if RAG improvements are statistically significant vs base feature set.
- **RAG Finding**: Across all three model families (LR, RF, LightGBM), the 95% CI for paired Delta AUC spans zero (e.g. LightGBM Delta AUC = +0.0009, 95% CI: [-0.0016, +0.0036]), showing no statistically significant gain over structured clinical features alone.
