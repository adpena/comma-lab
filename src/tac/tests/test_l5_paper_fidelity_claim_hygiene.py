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
    assert "Predicted contest-CPU score" not in architecture
    assert "Why this beats PR101" not in architecture
    assert "Provides 5-10x effective resolution gain" not in architecture
    assert "Paired exact CPU/CUDA on the same archive/runtime" in architecture


def test_l5_xray_planning_proxies_do_not_claim_score_or_archive_bytes() -> None:
    foveation = _read("src/tac/xray/foveation_ego_motion.py")
    predictive = _read("src/tac/xray/predictive_coding_hierarchy.py")

    assert "~70%" not in foveation
    assert "in ~25% of pixels" not in foveation
    assert "planning hypothesis" in foveation
    assert "requires_pact_video_scorer_measurement" in foveation
    assert "requires_paired_cpu_cuda_exact_eval" in foveation

    assert "should yield additional rate savings" not in predictive
    assert "quantifies that claim" not in predictive
    assert "does not report archive bytes, rate savings, or\nscore movement" in predictive
    assert "budget_is_archive_bytes" in predictive
    assert "requires_entropy_coded_archive_bytes" in predictive


def test_c1_z4_z5_campaign_ledgers_keep_sources_and_promotion_blockers_near_claims() -> None:
    c1 = _read(".omx/research/campaign_lane_c1_z6_world_model_foveation_20260514.md")
    z4 = _read(".omx/research/campaign_z4_cooperative_receiver_loss_20260514.md")
    z5 = _read(".omx/research/campaign_z5_predictive_coding_world_model_20260514.md")
    z4_init = _read("src/tac/substrates/z4_cooperative_receiver_loss/__init__.py")

    for text in (c1, z4, z5):
        assert "retrieved 2026-05-16" in text.lower()
        assert "paired CPU/CUDA exact" in text
        assert "archive SHA" in text
        assert "runtime tree/content SHA" in text
        assert "component recomputation" in text or "recompute components" in text

    for text in (c1, z4):
        assert "https://doi.org/10.1162/neco.1990.2.3.308" in text

    assert "provides 5-10x effective resolution gain" not in c1
    assert "PROMOTE to frontier" not in c1
    assert "campaign falsified" not in c1
    assert "operator frontier review" in c1
    assert "https://doi.org/10.1068/p050437" in c1
    assert "https://doi.org/10.1038/4580" in c1
    assert "https://doi.org/10.1038/nrn2787" in c1

    assert "bit budget shrinks from ``H(X)`` to ``H(X | f_R(X))``" not in z4_init
    assert "objective analogy only" in z4
    assert "objective analogy only" in z4_init
    assert "scorer-free inflate" in z4
    assert "source-matched Z3/A1 baseline" in z4

    assert "https://doi.org/10.1038/4580" in z5
    assert "https://doi.org/10.1038/nrn2787" in z5
    assert "reduces residual entropy by 20-40%" not in z5
    assert "applies the canonical -0.02 to -0.03 class-shift reward" not in z5
    assert "no class-shift reward is valid until a byte-closed paired exact anchor exists" in z5
    assert "Production / OSS reproducibility manifest" in z5


def test_l5_campaign_advisory_axes_are_not_falsification_authority() -> None:
    c1 = _read(".omx/research/campaign_lane_c1_z6_world_model_foveation_20260514.md")
    c2 = _read(".omx/research/campaign_lane_c2_z7_mature_predictive_receiver_l5_20260514.md")

    assert "macOS-CPU > 0.150 (campaign falsified)" not in c1
    assert "macOS-CPU advisory is not falsification authority" in c2
    assert "If iteration 3 final [contest-CPU] > 0.10" not in c2
    assert "paired exact CPU/CUDA on the same archive/runtime" in c2


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
