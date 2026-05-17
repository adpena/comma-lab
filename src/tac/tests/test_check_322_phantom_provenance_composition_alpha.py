# SPDX-License-Identifier: MIT
"""Tests for Catalog #322 + sister revert helper.

Per REDO+PIVOT comprehensive fix 2026-05-17 (operator NON-NEGOTIABLE *"We
need to fix all and redo"*).

Coverage:
  - revert_phantom_source_rows helper (in-memory + matrix file scan)
  - Catalog #322 STRICT preflight gate (matrix + pairwise_alpha surfaces)
  - waiver semantics (placeholder rejection + valid rationale acceptance)
  - strict-mode raises PreflightError
  - orchestrator wire-in regression guard
  - live-repo regression (count = 0 post-quarantine)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.optimization.substrate_composition_matrix import (
    PHANTOM_PROVENANCE_CANDIDATE_TOKENS,
    SubstrateRow,
    SubstrateClass,
    ScoreAxis,
    canonical_substrate_inventory,
    revert_phantom_source_rows,
)
from tac.preflight import (
    PreflightError,
    _CATALOG_322_PHANTOM_PROVENANCE_TOKENS,
    _CATALOG_322_PHANTOM_PROVENANCE_WAIVER_TOKEN,
    _check_322_waiver_has_valid_rationale,
    check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha as gate,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ── Catalog #321 phantom token set is canonical ────────────────────────────


def test_phantom_provenance_canonical_token_set() -> None:
    """The 3 canonical phantom tokens match between helper + gate."""
    helper_set = set(PHANTOM_PROVENANCE_CANDIDATE_TOKENS)
    gate_set = set(_CATALOG_322_PHANTOM_PROVENANCE_TOKENS)
    assert helper_set == gate_set
    assert helper_set == {"pr101_state_dict", "pr106_state_dict", "posenet_class_sensitivity"}


# ── revert_phantom_source_rows helper ──────────────────────────────────────


def _make_phantom_row() -> SubstrateRow:
    """Build a SubstrateRow whose substrate_id references pr101_state_dict."""
    return SubstrateRow(
        substrate_id="pr101_state_dict_phantom_row",
        name="Synthetic phantom row",
        substrate_class=SubstrateClass.RESIDUAL,
        target_axis=ScoreAxis.RATE,
        format_id=0xFF,
        magic_bytes="PHNT",
        runtime_dep_closure=("torch",),
        byte_budget_band=(0, 1),
        predicted_delta_alone_band=(0.0, 0.0),
        requires_score_aware_training=False,
        landed_at="2026-05-17",
        landing_memo="synthetic_test_row",
    )


def test_revert_helper_canonical_inventory_clean() -> None:
    """Live canonical_inventory carries 0 phantom rows."""
    result = revert_phantom_source_rows()
    assert result["catalog_321_compliant"] is True
    assert result["inventory_phantom_count"] == 0
    assert "CLEAN" in result["advisory"] or result["matrix_phantom_count"] == 0


def test_revert_helper_detects_synthetic_phantom_row() -> None:
    """A synthetic phantom row is detected + removed from in-place list."""
    inv = list(canonical_substrate_inventory())
    inv.append(_make_phantom_row())
    n_before = len(inv)
    result = revert_phantom_source_rows(inv)
    assert result["inventory_phantom_count"] == 1
    assert "pr101_state_dict_phantom_row" in result["inventory_phantom_rows"]
    assert len(inv) == n_before - 1
    assert result["catalog_321_compliant"] is False


def test_revert_helper_detects_phantom_in_matrix_file(tmp_path: Path) -> None:
    """A synthetic matrix file with phantom pair_key is detected."""
    matrix = tmp_path / "substrate_composition_matrix.json"
    payload = {
        "entries": {
            "pr101_state_dict__x__some_other_substrate": [{"alpha": 1.5}],
            "wavelet_residual__x__siren_residual": [{"alpha": 0.8}],
        }
    }
    matrix.write_text(json.dumps(payload), encoding="utf-8")
    result = revert_phantom_source_rows(matrix_path=matrix)
    assert result["matrix_phantom_count"] == 1
    assert result["matrix_phantom_pair_keys"][0].startswith("pr101_state_dict")


def test_revert_helper_handles_missing_matrix_file(tmp_path: Path) -> None:
    """Missing matrix file is silent (returns clean)."""
    nonexistent = tmp_path / "does_not_exist.json"
    inv = list(canonical_substrate_inventory())
    result = revert_phantom_source_rows(inv, matrix_path=nonexistent)
    assert result["catalog_321_compliant"] is True
    assert result["matrix_phantom_count"] == 0


def test_revert_helper_handles_malformed_matrix_file(tmp_path: Path) -> None:
    """Malformed matrix file is silently tolerated (defers to canonical loader)."""
    matrix = tmp_path / "broken.json"
    matrix.write_text("{ not valid json", encoding="utf-8")
    result = revert_phantom_source_rows(matrix_path=matrix)
    # Should NOT raise; matrix_phantom_count = 0 because nothing parseable.
    assert result["matrix_phantom_count"] == 0


# ── Catalog #322 waiver semantics ──────────────────────────────────────────


def test_waiver_valid_rationale_accepted() -> None:
    text = f"{_CATALOG_322_PHANTOM_PROVENANCE_WAIVER_TOKEN}operator-reviewed diagnostic"
    assert _check_322_waiver_has_valid_rationale(text) is True


def test_waiver_placeholder_rationale_rejected() -> None:
    text = f"{_CATALOG_322_PHANTOM_PROVENANCE_WAIVER_TOKEN}<rationale>"
    assert _check_322_waiver_has_valid_rationale(text) is False


def test_waiver_placeholder_reason_rejected() -> None:
    text = f"{_CATALOG_322_PHANTOM_PROVENANCE_WAIVER_TOKEN}<reason>"
    assert _check_322_waiver_has_valid_rationale(text) is False


def test_waiver_empty_rationale_rejected() -> None:
    text = f"{_CATALOG_322_PHANTOM_PROVENANCE_WAIVER_TOKEN}"
    assert _check_322_waiver_has_valid_rationale(text) is False


def test_waiver_short_rationale_rejected() -> None:
    text = f"{_CATALOG_322_PHANTOM_PROVENANCE_WAIVER_TOKEN}ab"
    assert _check_322_waiver_has_valid_rationale(text) is False


def test_waiver_missing_marker_returns_false() -> None:
    assert _check_322_waiver_has_valid_rationale("unrelated text") is False


# ── Catalog #322 gate end-to-end ───────────────────────────────────────────


def test_gate_clean_repo_live_count_zero() -> None:
    """Live repo MUST have 0 phantom-provenance violations post-quarantine."""
    violations = gate(strict=False, verbose=False)
    assert violations == [], f"Expected live count 0; got {len(violations)}: {violations}"


def test_gate_flags_phantom_pair_key_in_synthetic_matrix(tmp_path: Path) -> None:
    """Synthetic matrix with phantom pair_key is flagged."""
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True)
    matrix = state_dir / "substrate_composition_matrix.json"
    payload = {
        "entries": {
            "pr106_state_dict__x__wavelet_residual": [{"alpha": 2.5, "written_at_utc": "2026-05-17T22:00:00Z"}],
        }
    }
    matrix.write_text(json.dumps(payload), encoding="utf-8")
    violations = gate(repo_root=tmp_path, strict=False)
    assert len(violations) == 1
    assert "pr106_state_dict" in violations[0]
    assert "Catalog #322 violation" in violations[0]


def test_gate_accepts_phantom_with_valid_waiver(tmp_path: Path) -> None:
    """Phantom pair_key with valid same-row waiver is accepted."""
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True)
    matrix = state_dir / "substrate_composition_matrix.json"
    payload = {
        "entries": {
            "pr101_state_dict__x__wavelet_residual": [{
                "alpha": 1.2,
                "written_at_utc": "2026-05-17T22:00:00Z",
                "_phantom_provenance_waiver": f"{_CATALOG_322_PHANTOM_PROVENANCE_WAIVER_TOKEN}operator-reviewed diagnostic post-Option-B",
            }],
        }
    }
    matrix.write_text(json.dumps(payload), encoding="utf-8")
    violations = gate(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_rejects_placeholder_waiver(tmp_path: Path) -> None:
    """Phantom pair_key with placeholder waiver is still flagged."""
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True)
    matrix = state_dir / "substrate_composition_matrix.json"
    payload = {
        "entries": {
            "posenet_class_sensitivity__x__siren_residual": [{
                "alpha": 0.9,
                "written_at_utc": "2026-05-17T22:00:00Z",
                "_phantom_provenance_waiver": f"{_CATALOG_322_PHANTOM_PROVENANCE_WAIVER_TOKEN}<rationale>",
            }],
        }
    }
    matrix.write_text(json.dumps(payload), encoding="utf-8")
    violations = gate(repo_root=tmp_path, strict=False)
    assert len(violations) == 1


def test_gate_flags_phantom_pairwise_alpha_artifact(tmp_path: Path) -> None:
    """Synthetic pairwise_alpha_*.json with phantom candidate is flagged."""
    wz_dir = tmp_path / ".omx" / "state" / "wyner_ziv_deliverability"
    wz_dir.mkdir(parents=True)
    artifact = wz_dir / "pairwise_alpha_20260517T999999Z.json"
    payload = {
        "candidates_probed": ["pr101_state_dict", "posenet_class_sensitivity"],
        "pair_results": {
            "pr101_state_dict+posenet_class_sensitivity": {"alpha_band": "ADDITIVE"},
        },
    }
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    violations = gate(repo_root=tmp_path, strict=False)
    assert len(violations) == 1
    assert "pairwise_alpha artifact" in violations[0]


def test_gate_accepts_clean_pairwise_alpha_artifact(tmp_path: Path) -> None:
    """A pairwise_alpha artifact citing only VALIDATED_CONTEST_MEMBER substrates is clean."""
    wz_dir = tmp_path / ".omx" / "state" / "wyner_ziv_deliverability"
    wz_dir.mkdir(parents=True)
    artifact = wz_dir / "pairwise_alpha_20260517T999999Z.json"
    payload = {
        "candidates_probed": ["apogee_int6_archive", "pr101_fec6_archive"],
        "pair_results": {
            "apogee_int6_archive+pr101_fec6_archive": {"alpha_band": "ADDITIVE"},
        },
    }
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    violations = gate(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_strict_mode_raises(tmp_path: Path) -> None:
    """strict=True raises PreflightError on any violation."""
    wz_dir = tmp_path / ".omx" / "state" / "wyner_ziv_deliverability"
    wz_dir.mkdir(parents=True)
    artifact = wz_dir / "pairwise_alpha_synthetic.json"
    artifact.write_text(json.dumps({"candidates_probed": ["pr106_state_dict"]}), encoding="utf-8")
    with pytest.raises(PreflightError) as exc:
        gate(repo_root=tmp_path, strict=True)
    assert "Catalog #322" in str(exc.value)
    assert "phantom-provenance" in str(exc.value).lower() or "phantom" in str(exc.value).lower()


def test_gate_strict_silent_on_clean_repo(tmp_path: Path) -> None:
    """strict=True on clean repo returns [] without raising."""
    # No matrix file, no pairwise_alpha files: clean.
    violations = gate(repo_root=tmp_path, strict=True)
    assert violations == []


def test_gate_no_state_dir_silent(tmp_path: Path) -> None:
    """Missing .omx/state dir is silent."""
    violations = gate(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_verbose_output_clean(tmp_path: Path, capsys) -> None:
    """Verbose mode prints OK on clean."""
    gate(repo_root=tmp_path, strict=False, verbose=True)
    out = capsys.readouterr().out
    assert "OK" in out


def test_gate_verbose_output_dirty(tmp_path: Path, capsys) -> None:
    """Verbose mode prints violations on dirty."""
    wz_dir = tmp_path / ".omx" / "state" / "wyner_ziv_deliverability"
    wz_dir.mkdir(parents=True)
    (wz_dir / "pairwise_alpha_x.json").write_text(
        json.dumps({"candidates_probed": ["pr101_state_dict"]}), encoding="utf-8",
    )
    gate(repo_root=tmp_path, strict=False, verbose=True)
    out = capsys.readouterr().out
    assert "pairwise_alpha_x.json" in out


# ── Orchestrator wire-in regression guard ──────────────────────────────────


def test_orchestrator_callsite_wires_strict_true() -> None:
    """preflight_all() must invoke Catalog #322 gate with strict=True."""
    preflight_src = (REPO_ROOT / "src" / "tac" / "preflight.py").read_text(encoding="utf-8")
    assert "check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha(" in preflight_src
    # The callsite must pass strict=True per CLAUDE.md "Strict-flip atomicity rule".
    idx = preflight_src.find("check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha(\n")
    assert idx > 0
    window = preflight_src[idx:idx + 400]
    assert "strict=True" in window


def test_catalog_322_row_present_in_claude_md() -> None:
    """CLAUDE.md catalog table must carry the #322 row (Catalog #176 sister)."""
    claude_md = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "322. `check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha`" in claude_md
