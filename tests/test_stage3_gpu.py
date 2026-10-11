"""Tests for Stage 3 GPU detection, active retriever badge, and notebook cleanliness."""
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_stage3_notebook_has_no_pending_callout():
    """Verify stage3.ipynb contains zero pending callouts or missing artifact excuses."""
    nb_path = REPO_ROOT / "notebooks" / "stages" / "stage3.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for idx, cell in enumerate(nb["cells"]):
        text = "".join(cell.get("source", []))
        assert "render_pending_box" not in text, f"Cell {idx} contains render_pending_box"
        assert "[PENDING]" not in text, f"Cell {idx} contains [PENDING]"
        assert "colab_embed_notes.ipynb in Google Colab" not in text, f"Cell {idx} references Colab execution"


def test_stage3_hardware_and_figures_exist():
    """Verify Stage 3 figures exist and HTML contains active GPU accelerated status."""
    fig1 = REPO_ROOT / "reports" / "stages" / "figures" / "stage3_retrieval_architecture.png"
    fig2 = REPO_ROOT / "reports" / "stages" / "figures" / "stage3_rag_feature_distributions.png"
    fig3 = REPO_ROOT / "reports" / "stages" / "figures" / "stage3_patient_precedent_case.png"
    fig4 = REPO_ROOT / "reports" / "stages" / "figures" / "stage3_univariate_roc_curve.png"
    html_path = REPO_ROOT / "reports" / "stages" / "stage3.html"

    for fig in [fig1, fig2, fig3, fig4]:
        assert fig.exists(), f"Missing figure: {fig.name}"
        assert fig.stat().st_size > 0, f"Empty figure: {fig.name}"

    assert html_path.exists(), "stage3.html does not exist"
    html = html_path.read_text(encoding="utf-8")
    assert "ACTIVE / GPU ACCELERATED" in html
    assert "NVIDIA GeForce RTX 3050" in html
    assert "[PENDING]" not in html
