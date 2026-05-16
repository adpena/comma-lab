# SPDX-License-Identifier: MIT
"""Paper-fidelity claim hygiene for L5-family planning surfaces."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


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
