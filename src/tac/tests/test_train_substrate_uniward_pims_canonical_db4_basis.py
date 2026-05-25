# SPDX-License-Identifier: MIT
"""Canonical-fix tests for ``experiments/train_substrate_uniward_per_instance_multi_scale_wavelet_segnet.py``.

Pins the Probe 9c canonical-optimal db4 wavelet basis per the
SISTER_BASIS_DOMINATES_db4 verdict (commit ``efeaff5c9``;
``.omx/research/probe_9c_per_level_wavelet_basis_disambiguator_landed_20260525.md``).

Per Catalog #307 IMPLEMENTATION-LEVEL fix discipline + CLAUDE.md "Forbidden
premature KILL": the per-instance + multi-scale wavelet UNIWARD-weighted
SegNet loss paradigm is INTACT; the db8 -> db4 update is implementation-level
only. These tests prevent regression of the canonical default back to db8.

Cite chain:
- Probe 9 BREAKTHROUGH (commit ``685fe6726``)
- Probe 9-PREP (commit ``92a48616e``)
- Probe 9b 100-pair disambiguator (commit ``2fca9974b``)
- Probe 9c per-level basis disambiguator (commit ``efeaff5c9``)
- THIS canonical fix (lane ``lane_probe_9_recipe_canonical_update_db8_to_db4_20260525``)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER_PATH = (
    REPO_ROOT
    / "experiments"
    / "train_substrate_uniward_per_instance_multi_scale_wavelet_segnet.py"
)


def _load_trainer():
    """Import the substrate trainer module by path."""
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    spec = importlib.util.spec_from_file_location(
        "train_substrate_uniward_per_instance_multi_scale_wavelet_segnet",
        TRAINER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[
        "train_substrate_uniward_per_instance_multi_scale_wavelet_segnet"
    ] = module
    spec.loader.exec_module(module)
    return module


def test_wavelet_name_default_is_canonical_db4() -> None:
    """Probe 9c verdict SISTER_BASIS_DOMINATES_db4 -> default basis MUST be db4."""
    trainer = _load_trainer()
    assert trainer.WAVELET_NAME_DEFAULT == "db4", (
        "Probe 9c canonical-optimal basis is db4 (z=-3.442sigma vs db8 baseline; "
        "mean 0.3599 vs db8 0.3915; CIs disjoint at 95%); per Catalog #307 "
        "IMPLEMENTATION-LEVEL fix the trainer default MUST be db4. Regression "
        "to db8 would violate the canonical fix landing."
    )


def test_wavelet_levels_default_remains_3() -> None:
    """Probe 9c held decomposition depth at 3 levels per Mallat seat binding revision."""
    trainer = _load_trainer()
    assert trainer.WAVELET_LEVELS_DEFAULT == 3, (
        "Probe 9c canonical anchor was N=100 / 3-level / db4; the per-"
        "decomposition-level disambiguator (Probe 9e at 2-vs-3-vs-4 levels) "
        "is queued as P3 operator-routable per Catalog #308 alternative-"
        "reducer cascade but has NOT yet superseded the 3-level default."
    )


def test_tier_1_required_flag_wavelet_name_default_is_db4() -> None:
    """The TIER_1_OPERATOR_REQUIRED_FLAGS manifest must declare db4 as canonical default."""
    trainer = _load_trainer()
    flag = trainer.TIER_1_OPERATOR_REQUIRED_FLAGS["--wavelet-name"]
    assert flag["default"] == "db4", (
        "Per Catalog #151 sister discipline the TIER_1_OPERATOR_REQUIRED_FLAGS "
        "manifest default for --wavelet-name MUST be the canonical-optimal db4 "
        "basis. This default is consumed by the operator-authorize wrapper "
        "(Catalog #151) + the dispatch-protocol verifier (Catalog #270)."
    )


def test_tier_1_required_flag_wavelet_name_rationale_cites_probe_9c() -> None:
    """The flag rationale must cite Probe 9c verdict + Catalog #307 classification."""
    trainer = _load_trainer()
    rationale = trainer.TIER_1_OPERATOR_REQUIRED_FLAGS["--wavelet-name"]["rationale"]
    rationale_lower = rationale.lower()
    assert "probe 9c" in rationale_lower or "probe9c" in rationale_lower, (
        "Rationale must cite Probe 9c (commit efeaff5c9) per Catalog #344 "
        "canonical-equation traceability discipline."
    )
    assert "db4" in rationale_lower, (
        "Rationale must explicitly name db4 as the canonical-optimal basis."
    )


def test_constant_module_carries_canonical_fix_marker_in_source() -> None:
    """Source must carry the canonical-fix marker comment for grep-discoverability."""
    src = TRAINER_PATH.read_text(encoding="utf-8")
    # Per Catalog #110/#113 HISTORICAL_PROVENANCE: the canonical-fix marker
    # records WHEN the db8 -> db4 update landed + WHY (Probe 9c receipts).
    assert "Canonical fix 2026-05-25: db8 -> db4 per Probe 9c" in src, (
        "Source must carry the canonical-fix marker '# Canonical fix 2026-05-25: "
        "db8 -> db4 per Probe 9c ...' for grep-discoverability + per Catalog "
        "#229 premise-verification + #287 evidence-tag discipline."
    )
    assert "efeaff5c9" in src, (
        "Source must cite the Probe 9c commit sha (efeaff5c9) for cite-chain "
        "traceability per Catalog #344 canonical-equation discipline."
    )


def test_smoke_summary_carries_probe_9c_canonical_db4_anchor_fields() -> None:
    """Smoke summary must include Probe 9c db4 anchor empirical receipts."""
    src = TRAINER_PATH.read_text(encoding="utf-8")
    # The smoke summary dict must surface the canonical-optimal basis + receipts
    # so downstream consumers (cathedral autopilot Hook 4 / continual-learning
    # posterior Hook 5) can ingest empirical receipts via canonical contract.
    required_fields = [
        '"probe_9c_db4_anchor_min": 0.0532',
        '"probe_9c_db4_anchor_mean": 0.3599',
        '"probe_9c_db4_anchor_z_vs_db8_baseline": -3.442',
        '"probe_9c_canonical_optimal_basis": "db4"',
    ]
    for required in required_fields:
        assert required in src, (
            f"Smoke summary must contain canonical-fix receipt field "
            f"{required!r}. Per Catalog #287/#323 canonical Provenance "
            f"discipline the empirical receipts must be machine-readable."
        )


def test_smoke_summary_preserves_probe_9_historical_anchor_per_catalog_110() -> None:
    """Per Catalog #110/#113 APPEND-ONLY: original Probe 9 db8 anchor preserved verbatim."""
    src = TRAINER_PATH.read_text(encoding="utf-8")
    # The original Probe 9 N=25 db8 anchor (0.2597) is HISTORICAL_PROVENANCE
    # per Catalog #110/#113 + must NOT be mutated; only NEW fields are added.
    assert '"probe_9_anchor_min": 0.2597' in src, (
        "Per Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY discipline, "
        "the original Probe 9 N=25 db8 historical anchor (0.2597) MUST be "
        "preserved verbatim; this canonical fix ADDS new db4 fields, never "
        "mutates the historical anchor."
    )
