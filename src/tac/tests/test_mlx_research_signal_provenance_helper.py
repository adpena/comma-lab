# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.local_acceleration import (
    EVIDENCE_GRADE_MLX,
    EVIDENCE_TAG_MLX,
    PROVENANCE_EVIDENCE_GRADE_MLX,
    build_mlx_research_signal_provenance,
)


def test_mlx_research_signal_helper_bridges_legacy_grade_to_canonical_provenance() -> None:
    prov = build_mlx_research_signal_provenance(
        artifact_sha256="a" * 64,
        source_path="experiments/results/mlx_signal/manifest.json",
    )

    assert EVIDENCE_GRADE_MLX == "macOS-MLX-research-signal"
    assert PROVENANCE_EVIDENCE_GRADE_MLX == "macos_mlx_research_signal"
    assert prov.evidence_grade.value == PROVENANCE_EVIDENCE_GRADE_MLX
    assert prov.measurement_axis == EVIDENCE_TAG_MLX
    assert prov.hardware_substrate == "macos_arm64_mlx"
    assert prov.score_claim_valid is False
    assert prov.promotion_eligible is False
