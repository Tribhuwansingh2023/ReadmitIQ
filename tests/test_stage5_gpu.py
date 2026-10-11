"""Tests for Stage 5 GPU detection, banner rendering, and notebook integrity."""
import json
from pathlib import Path
import pytest
from notebooks.stages._stage_helpers import (
    get_local_gpu_info,
    render_gpu_active_banner,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_get_local_gpu_info():
    """Verify local GPU interrogation returns structured hardware metadata."""
    gpu_info = get_local_gpu_info()
    assert isinstance(gpu_info, dict)
    assert "available" in gpu_info
    assert "name" in gpu_info
    assert "memory_mb" in gpu_info
    assert "driver" in gpu_info
    assert len(gpu_info["name"]) > 0


def test_render_gpu_active_banner():
    """Verify active GPU badge renders clean HTML with hardware specs."""
    banner = render_gpu_active_banner(
        device_name="NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        memory_mb=6144,
        driver_version="592.82",
        details="Direct on-device multi-head attention extraction active."
    )
    assert hasattr(banner, "data")
    html = banner.data
    assert "ACTIVE / GPU ACCELERATED" in html
    assert "NVIDIA GeForce RTX 3050 6GB Laptop GPU" in html
    assert "6,144 MiB" in html
    assert "592.82" in html


def test_stage5_notebook_has_no_pending_callout():
    """Verify stage5.ipynb contains no pending callout or missing GPU excuses."""
    nb_path = REPO_ROOT / "notebooks" / "stages" / "stage5.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for idx, cell in enumerate(nb["cells"]):
        text = "".join(cell.get("source", []))
        assert "render_pending_box" not in text, f"Cell {idx} still contains render_pending_box"
        assert "[PENDING]" not in text, f"Cell {idx} still contains [PENDING]"
        assert "Full PyTorch model checkpoint (.pt) not stored" not in text, f"Cell {idx} contains missing checkpoint excuse"


def test_stage5_figure_and_html_exist():
    """Verify Stage 5 attention diagram and exported HTML report exist and are populated."""
    fig_path = REPO_ROOT / "reports" / "stages" / "figures" / "stage5_attention_mechanism_diagram.png"
    html_path = REPO_ROOT / "reports" / "stages" / "stage5.html"

    assert fig_path.exists(), "stage5_attention_mechanism_diagram.png does not exist"
    assert fig_path.stat().st_size > 50000, "stage5_attention_mechanism_diagram.png is too small"

    assert html_path.exists(), "stage5.html does not exist"
    assert html_path.stat().st_size > 100000, "stage5.html is too small"
