"""Test verification of headline provenance paths in ReadmitIQ."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_provenance_paths_exist():
    """Verify that every verifiedArtifactPath referenced in Provenances.ts exists on disk."""
    provenances_file = REPO_ROOT / "web" / "src" / "components" / "Provenances.ts"
    assert provenances_file.exists(), "Provenances.ts must exist"

    content = provenances_file.read_text(encoding="utf-8")

    # Match verifiedArtifactPath: "..."
    matches = re.findall(r'verifiedArtifactPath:\s*"([^"]+)"', content)
    assert len(matches) > 0, "Provenances.ts must contain verifiedArtifactPath definitions"

    for rel_path in matches:
        full_path = REPO_ROOT / rel_path
        assert full_path.exists(), f"Provenance path does not exist on disk: {rel_path}"
