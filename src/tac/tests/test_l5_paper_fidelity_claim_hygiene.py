# SPDX-License-Identifier: MIT
"""Paper-fidelity claim hygiene for L5-family planning surfaces."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_l5_v2_source_ledgers_pin_primary_sources_and_claim_blockers() -> None:
    source_basis = _read(
        ".omx/research/l5_v2_latest_neural_video_codec_source_basis_20260516_codex.md"
    )
    architecture = _read(
        ".omx/research/time_traveler_architecture_reverse_engineered_20260513.md"
    )
    campaign = _read(
        ".omx/research/campaign_lane_c2_z7_mature_predictive_receiver_l5_20260514.md"
    )

    combined = "\n".join([source_basis, architecture, campaign])
    for required_source in (
        "https://github.com/commaai/comma_video_compression_challenge",
        "https://github.com/commaai/comma_video_compression_challenge/pull/95",
        "https://github.com/commaai/comma_video_compression_challenge/pull/101",
        "https://github.com/commaai/comma_video_compression_challenge/pull/106",
        "https://openaccess.thecvf.com/content/CVPR2023/html/"
        "Chen_HNeRV_A_Hybrid_Neural_Representation_for_Videos_CVPR_2023_paper.html",
        "https://arxiv.org/abs/2502.20762",
        "https://arxiv.org/abs/2602.16711",
        "https://doi.org/10.1109/TIT.1973.1055037",
        "https://doi.org/10.1109/TIT.1976.1055508",
    ):
        assert required_source in combined

    for text in (source_basis, architecture, campaign):
        assert "Retrieved 2026-05-16" in text
        assert (
            "do not authorize" in text
            or "Claim Scope" in text
            or "Claim blockers" in text
            or "Claim-Blocking Notes" in text
        )
        assert "paired CPU/CUDA exact" in text or "CPU/CUDA axis labels" in text

    assert "planning prior" in architecture
    assert "planning prior" in campaign
    assert "not contest score evidence" in source_basis


def test_l5_family_docstrings_keep_literature_and_score_claims_separate() -> None:
    tt5l = _read("src/tac/substrates/time_traveler_l5_autonomy/__init__.py")
    z5 = _read("src/tac/substrates/z5_predictive_coding_world_model/__init__.py")
    c6 = _read("src/tac/substrates/c6_e4_mdl_ibps/__init__.py")

    assert "5-10x effective resolution gain" not in tt5l
    assert "Predicted contest-CPU score" not in tt5l
    assert "Planning band: 0.150-0.170" in tt5l
    assert "not paper-derived evidence of contest score movement" in tt5l

    assert "predicted 20-40% reduction" not in z5
    assert "mathematical-derivation; Time-Traveler-asymptote" not in z5
    assert "class-shift reward is valid until a paired exact anchor exists" in z5

    assert "empirically proved the HNeRV-family substrate class is saturated" not in c6
    assert "not a theorem from IB/MDL literature" in c6


def test_wyner_ziv_docstring_blocks_scorer_at_inflate_sideinfo_claim() -> None:
    text = _read("src/tac/substrates/wyner_ziv_cooperative_receiver/__init__.py")

    forbidden_phrases = [
        "contest IS Wyner-Ziv compression",
        "scorer is the\nKNOWN side-information at the receiver",
        "decoder has access to ``Y`` (here: scorer",
        "decoder\n  computes ``Y`` from the same scorer",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in text
    assert "inflate runtime must remain scorer-free" in text
    assert "scorer-free at inflate" in text
    assert "Planning band: 0.140-0.150" in text
    assert "not a literature-derived contest" in text


def test_composition_matrix_advertises_planning_not_empirical_validation() -> None:
    text = _read("src/tac/optimization/substrate_composition_matrix.py")

    assert "council has empirically validated" not in text
    assert "planning composability classes" in text
    assert "pending pairwise exact anchors" in text
