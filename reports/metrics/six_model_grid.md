# Six-model grid, validation split

All numbers are on the validation split. 'pending' means the artifact does not exist yet.

| Model | RAG | Role | AUC [95% CI] | F1 | Precision | Recall | P@10% | Lift@10% | Delta AUC vs without [95% CI] | Retriever |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Logistic Regression | without RAG | use case | pending | pending | pending | pending | pending | pending | pending | - |
| Logistic Regression | with note-RAG | use case | pending | pending | pending | pending | pending | pending | pending | - |
| Random Forest | without RAG | use case | pending | pending | pending | pending | pending | pending | pending | - |
| Random Forest | with note-RAG | use case | pending | pending | pending | pending | pending | pending | pending | - |
| Transformer | without RAG | use case | 0.6497 [0.6367, 0.6630] | 0.2708 | 0.1781 | 0.5651 | 0.2369 | 2.08 | - | - |
| Transformer | with note-RAG | use case | 0.6491 [0.6360, 0.6632] | 0.2662 | 0.1715 | 0.5946 | 0.2389 | 2.10 | -0.0006 [-0.0056, +0.0043] | - |
| LightGBM | without RAG | extra | pending | pending | pending | pending | pending | pending | pending | - |
| LightGBM | with note-RAG | extra | pending | pending | pending | pending | pending | pending | pending | - |

Pending entries:
- Logistic Regression, without RAG: artifact not found: reports/metrics/ablation_note_rag.json
- Logistic Regression, with note-RAG: artifact not found: reports/metrics/ablation_note_rag.json
- Random Forest, without RAG: artifact not found: reports/metrics/ablation_note_rag.json
- Random Forest, with note-RAG: artifact not found: reports/metrics/ablation_note_rag.json
- LightGBM, without RAG: artifact not found: reports/metrics/ablation_note_rag.json
- LightGBM, with note-RAG: artifact not found: reports/metrics/ablation_note_rag.json
