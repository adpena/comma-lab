from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_ap1_residue_purchase_scorer as ap1
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_signed_lattice_uses_nearest_quantum_and_stays_in_domain() -> None:
    source = np.asarray([-32, -31, -30, -5, -4, -3, 0, 3, 4, 5, 30, 31], dtype=np.int8)
    actual = ap1.signed_lattice(source, 4, 6)
    # Signed six-bit has one extra negative endpoint: -30 can round to -32,
    # whereas +30 must saturate to the largest in-domain lattice value (+28).
    assert actual.tolist() == [-32, -32, -32, -4, -4, -4, 0, 4, 4, 4, 28, 28]
    assert np.all(actual >= -32)
    assert np.all(actual <= 31)
    assert np.all(actual.astype(np.int16) % 4 == 0)


def test_signed_lattice_rejects_invalid_step() -> None:
    with pytest.raises(ap1.AP1Error, match="positive power of two"):
        ap1.signed_lattice(np.asarray([0], dtype=np.int8), 3, 6)


def test_depth_nibble_round_trip_including_zero_depth() -> None:
    depths = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
    packed = ap1.pack_depth_nibbles(depths)
    assert ap1.unpack_depth_nibbles(packed, len(depths)).tolist() == depths.tolist()


def test_pack_signed_rows_is_little_endian_and_zero_depth_safe() -> None:
    rows = [
        np.asarray([0, 0], dtype=np.int16),
        np.asarray([-1, 0], dtype=np.int16),
        np.asarray([-2, -1, 0, 1], dtype=np.int16),
    ]
    depths = np.asarray([0, 1, 2], dtype=np.uint8)
    payload = ap1.pack_signed_rows(rows, depths)
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    # depth-1: -1 -> 1, 0 -> 0. depth-2: -2 -> 10, -1 -> 11, 0 -> 00, 1 -> 01.
    assert bits[:10].tolist() == [1, 0, 0, 1, 1, 1, 0, 0, 1, 0]


def test_pack_signed_fixed_round_trips_receiver_bits() -> None:
    values = np.arange(-32, 32, dtype=np.int8)
    payload = ap1.pack_signed_fixed(values, 6)
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    rows = bits[: values.size * 6].reshape(values.size, 6).astype(np.int16)
    unsigned = (rows * (1 << np.arange(6, dtype=np.int16))).sum(axis=1)
    decoded = np.where(unsigned >= 32, unsigned - 64, unsigned)
    assert decoded.tolist() == values.astype(np.int16).tolist()


def test_candidate_census_is_exactly_control_plus_four_groups_times_three() -> None:
    assert len(ap1.SPECS) == 13
    assert ap1.SPECS[0].candidate_id == "control"
    assert {
        group: sum(spec.group == group for spec in ap1.SPECS)
        for group in ("semantic", "carrier", "hpac", "residual")
    } == {"semantic": 3, "carrier": 3, "hpac": 3, "residual": 3}


def test_carrier_rows_are_versioned_after_fixed_coder_correction() -> None:
    assert [spec.candidate_id for spec in ap1.SPECS if spec.group == "carrier"] == [
        "carrier_l1_fixed_coder",
        "carrier_l2_fixed_coder",
        "carrier_l3_fixed_coder",
    ]


def test_attempt_selection_preserves_fold_and_reuses_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ap1, "RECEIPT_ROOT", tmp_path)
    base = tmp_path / "advisory/semantic_l1"
    base.mkdir(parents=True)
    (base / "safe_run_status.json").write_text(json.dumps({"exit": 1}))
    retry = tmp_path / "advisory/semantic_l1_r2"
    retry.mkdir(parents=True)
    (retry / "safe_run_status.json").write_text(json.dumps({"exit": 0}))
    (retry / "contest_auth_eval.json").write_text("{}")

    attempt, folded = ap1._select_advisory_attempt("semantic_l1")

    assert attempt == "semantic_l1_r2"
    assert [(row["attempt"], row["disposition"]) for row in folded] == [
        ("semantic_l1", "FOLDED")
    ]


def test_ap1_files_pass_measure_and_discard_payload_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=Path.cwd(),
        strict=False,
        roots=(
            "experiments/ddm_ap1_residue_purchase_scorer.py",
            "tests/test_ddm_ap1_residue_purchase_scorer.py",
        ),
    )
    assert findings == []
