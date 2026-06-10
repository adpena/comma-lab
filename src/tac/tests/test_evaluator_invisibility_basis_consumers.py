# SPDX-License-Identifier: MIT
"""Tests for the invisibility-basis consumer hooks (task #47).

Verifies the two named consumers behave correctly:
  (a) the #46 waterfiller null_basis action builder (CERTIFIED zero distortion)
  (b) the PR110++ certified-free atom generator (free repair room)

Plus the null_space_exploiter surface re-export (the canonical wiring point).
"""

from __future__ import annotations

import pytest

from tac.optimization.evaluator_invisibility_basis import (
    CAMERA_H,
    CAMERA_W,
    build_evaluator_invisibility_basis,
)
from tac.optimization.evaluator_invisibility_basis_consumers import (
    CertifiedFreePerturbationAtom,
    build_null_basis_drop_action,
    build_null_basis_recode_action,
    certified_free_pixel_capacity,
    count_section_tier1_free_bytes,
    generate_pr110_certified_free_atoms,
)
from tac.optimization.lf_payload_rate_distortion import (
    ACTION_DROP,
    ACTION_RECODE,
    BaselineScoreTerms,
    CoefficientGroup,
    PayloadSection,
)


def _basis():
    return build_evaluator_invisibility_basis(
        provenance={"subagent": "test"}
    )


def _section(name="lf_blob", n_bytes=10):
    return PayloadSection(
        name=name,
        bytes=n_bytes,
        coefficient_group=CoefficientGroup(band_indices=(0,)),
    )


def _baseline():
    return BaselineScoreTerms(d_seg=0.067, d_pose=3.4e-5, archive_bytes=178_493)


# ---------------------------------------------------------------------------
# Consumer (a): null_basis waterfiller action builder.
# ---------------------------------------------------------------------------
def test_count_section_tier1_free_bytes_counts_invisible_pixels():
    basis = _basis()
    t1 = basis.tier1_resize
    zr = t1.zero_weight_rows[0]
    zc = t1.zero_weight_cols[0]
    # 6 bytes: 4 map to invisible pixels (zero-weight row/col), 2 to visible.
    locs = [
        ("frame1", 0, zr, 100),   # zero-weight ROW -> invisible
        ("frame1", 1, 50, zc),    # zero-weight COL -> invisible
        ("frame1", 2, zr, zc),    # both -> invisible
        ("frame0", 0, zr, 7),     # zero-weight row -> invisible (both heads)
        ("frame1", 0, 50, 100),   # visible
        ("frame1", 0, 51, 101),   # visible
    ]
    acc = count_section_tier1_free_bytes(
        _section(n_bytes=6), basis, pixel_locations=locs,
        basis_header_sha256="abc123",
    )
    assert acc.n_section_bytes == 6
    assert acc.n_tier1_free_bytes == 4
    assert abs(acc.tier1_free_fraction - 4 / 6) < 1e-9
    assert acc.to_row()["promotable"] is False
    assert acc.to_row()["evidence_grade"] == "mathematical-derivation"


def test_count_fails_closed_on_out_of_bounds():
    basis = _basis()
    locs = [("frame1", 0, CAMERA_H + 5, 0), ("frameX", 0, 0, 0)]
    acc = count_section_tier1_free_bytes(
        _section(n_bytes=2), basis, pixel_locations=locs,
        basis_header_sha256="x",
    )
    # both out-of-bounds / bad role -> NOT counted free (fail closed)
    assert acc.n_tier1_free_bytes == 0


def test_null_basis_recode_action_declares_certified_zero():
    basis = _basis()
    t1 = basis.tier1_resize
    zr = t1.zero_weight_rows[0]
    locs = [("frame1", 0, zr, j) for j in range(8)]  # all invisible
    acc = count_section_tier1_free_bytes(
        _section(n_bytes=8), basis, pixel_locations=locs,
        basis_header_sha256="deadbeefcafe",
    )
    action = build_null_basis_recode_action(
        _section(n_bytes=8), acc, _baseline(), free_byte_floor=2
    )
    assert action is not None
    assert action.action_kind == ACTION_RECODE
    assert action.est_delta_d_seg == 0.0  # CERTIFIED zero (not estimated)
    assert action.est_delta_d_pose == 0.0
    assert action.delta_bytes == -(8 - 2)  # frees free_bytes - floor
    # the LAW: certified-zero distortion + freed bytes => negative ΔS_total
    assert action.est_delta_score_total is not None
    assert action.est_delta_score_total < 0.0  # lowers predicted score
    assert action.value_per_byte is not None and action.value_per_byte > 0.0


def test_null_basis_recode_returns_none_below_floor():
    basis = _basis()
    t1 = basis.tier1_resize
    zr = t1.zero_weight_rows[0]
    locs = [("frame1", 0, zr, j) for j in range(3)]
    acc = count_section_tier1_free_bytes(
        _section(n_bytes=3), basis, pixel_locations=locs,
        basis_header_sha256="x",
    )
    assert build_null_basis_recode_action(
        _section(n_bytes=3), acc, _baseline(), free_byte_floor=5
    ) is None


def test_null_basis_drop_action_only_when_fully_free():
    basis = _basis()
    t1 = basis.tier1_resize
    zr = t1.zero_weight_rows[0]
    # 100% free section -> drop action
    locs_full = [("frame1", 0, zr, j) for j in range(5)]
    acc_full = count_section_tier1_free_bytes(
        _section(n_bytes=5), basis, pixel_locations=locs_full,
        basis_header_sha256="x",
    )
    drop = build_null_basis_drop_action(_section(n_bytes=5), acc_full, _baseline())
    assert drop is not None
    assert drop.action_kind == ACTION_DROP
    assert drop.delta_bytes == -5
    assert drop.est_delta_d_seg == 0.0 and drop.est_delta_d_pose == 0.0
    # partial section -> no drop
    locs_part = [("frame1", 0, zr, 0), ("frame1", 0, 50, 50)]
    acc_part = count_section_tier1_free_bytes(
        _section(n_bytes=2), basis, pixel_locations=locs_part,
        basis_header_sha256="x",
    )
    assert build_null_basis_drop_action(_section(n_bytes=2), acc_part, _baseline()) is None


# ---------------------------------------------------------------------------
# Consumer (b): PR110++ certified-free atom generator.
# ---------------------------------------------------------------------------
def test_generate_pr110_atoms_are_certified_invisible():
    basis = _basis()
    atoms = generate_pr110_certified_free_atoms(
        basis, frame_role="frame1", channels=(0,),
        basis_header_sha256="sha", max_atoms=50,
    )
    assert len(atoms) == 50
    t1 = basis.tier1_resize
    zr, zc = set(t1.zero_weight_rows), set(t1.zero_weight_cols)
    for a in atoms:
        assert isinstance(a, CertifiedFreePerturbationAtom)
        # every atom's pixel is certified invisible (zero-weight row OR col)
        assert a.row in zr or a.col in zc
        assert a.promotable is False
        assert a.evidence_grade == "mathematical-derivation"


def test_generate_pr110_atoms_max_amplitude_room():
    basis = _basis()
    atoms = generate_pr110_certified_free_atoms(
        basis, frame_role="frame1", basis_header_sha256="x", max_atoms=1
    )
    a = atoms[0]
    # at a low base value, free amplitude room is "up to 255"; at high, "down".
    assert a.as_camera_delta(0.0) == 255.0
    assert a.as_camera_delta(255.0) == 255.0
    assert a.as_camera_delta(100.0) == 155.0


def test_generate_pr110_atoms_rejects_bad_role_channel():
    from tac.optimization.evaluator_invisibility_basis import (
        EvaluatorInvisibilityBasisError,
    )

    basis = _basis()
    with pytest.raises(EvaluatorInvisibilityBasisError):
        generate_pr110_certified_free_atoms(
            basis, frame_role="bad", basis_header_sha256="x"
        )
    with pytest.raises(EvaluatorInvisibilityBasisError):
        generate_pr110_certified_free_atoms(
            basis, frame_role="frame1", channels=(9,), basis_header_sha256="x"
        )


def test_certified_free_pixel_capacity():
    basis = _basis()
    cap = certified_free_pixel_capacity(basis, n_channels=3)
    # both-head free directions = zero-weight pixels x 3 channels
    assert cap["both_head_free_directions_per_frame"] == 230_904 * 3
    assert cap["frame0_segnet_only_free_directions"] == CAMERA_H * CAMERA_W * 3
    assert cap["full_resize_null_dim_per_channel"] == basis.tier1_resize.full_null_dim
    assert cap["promotable"] is False


# ---------------------------------------------------------------------------
# The null_space_exploiter surface re-export (the canonical wiring point).
# ---------------------------------------------------------------------------
def test_null_space_exploiter_reexports_certified_atoms():
    from tac.null_space_exploiter import (
        certified_free_pixel_capacity as cap_surface,
    )
    from tac.null_space_exploiter import (
        generate_pr110_certified_free_atoms as gen_surface,
    )

    basis = _basis()
    atoms = gen_surface(basis, frame_role="frame1", basis_header_sha256="x", max_atoms=3)
    assert len(atoms) == 3
    cap = cap_surface(basis)
    assert cap["both_head_free_directions_per_frame"] > 0


# ---------------------------------------------------------------------------
# Artifact CLI smoke (tier-1-only, durable local output) — end-to-end build.
# ---------------------------------------------------------------------------
def test_artifact_cli_builds_tier1_only(tmp_path):
    """The build CLI materializes a JSONL + npz + manifest with the certified
    tier-1 surface (no atlas => tier-1 only).  Round-trips to the same dims."""
    import importlib.util
    from pathlib import Path

    import numpy as np

    from tac.optimization.evaluator_invisibility_basis import (
        EvaluatorInvisibilityBasis,
    )

    cli_path = (
        Path(__file__).resolve().parents[3] / "tools" / "build_evaluator_invisibility_basis.py"
    )
    spec = importlib.util.spec_from_file_location("_build_eib", cli_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main(["--output-root", str(tmp_path)])
    assert rc == 0
    run_dirs = list(tmp_path.glob("evaluator_invisibility_basis_*"))
    assert len(run_dirs) == 1
    out = run_dirs[0]
    assert (out / "manifest.json").exists()
    assert (out / "tier1_resize_null_space.npz").exists()
    lines = (out / "evaluator_invisibility_basis.jsonl").read_text().splitlines()
    basis = EvaluatorInvisibilityBasis.from_jsonl_lines(lines)
    assert basis.tier1_resize.n_zero_weight_pixels_per_channel == 230_904
    assert len(basis.tier2_rows) == 0  # no atlas supplied
    npz = np.load(out / "tier1_resize_null_space.npz")
    assert int(npz["zero_weight_pixel_mask"].sum()) == 230_904
