from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from experiments import ddm_jbp1_joint_batch_price as jbp1


def test_price_row_reports_demand_fraction_and_refusal() -> None:
    row = jbp1.price_row(-3_756)
    assert row["bytes_cut"] == 3_756
    assert row["fraction_of_demand"] == 3_756 / jbp1.DEMAND_BYTES
    assert row["candidate_joint_pool_bytes"] == jbp1.BASE_JOINT_POOL_BYTES - 3_756
    assert row["verdict"].startswith("REFUSED +")
    assert row["prior_law_falsified"] is False


def test_price_row_admits_only_the_full_demand() -> None:
    row = jbp1.price_row(-40_000)
    assert row["verdict"] == "BYTE-WIN by 477.86 B"
    assert row["prior_law_falsified"] is True


def test_parse_bhw_group0_consumes_only_the_benefit_prefix(monkeypatch) -> None:
    gf1 = bytes.fromhex("11" * 32)
    afr1 = bytes.fromhex("22" * 32)
    gt = bytes.fromhex("33" * 32)
    monkeypatch.setattr(jbp1, "GF1_FIELD_SHA256", gf1.hex())
    monkeypatch.setattr(jbp1.rxc1, "TOKENS_SHA256", afr1.hex())
    monkeypatch.setattr(jbp1, "GT_SEMANTIC_SHA256", gt.hex())
    header = jbp1.BHW_HEADER.pack(b"XBH1", 1, 0, 0, 2, 1, 1, gf1, afr1, gt)
    records = b"".join(
        jbp1.BHW_RECORD.pack(*row)
        for row in ((5, 0, 1), (9, 2, 3), (12, 1, 0), (20, 4, 0))
    )
    denominator, group0 = jbp1.parse_bhw_group0(header + records)
    assert denominator == {
        "benefit": 2,
        "harm": 1,
        "wash": 1,
        "disagreement_denominator": 4,
    }
    assert group0 == [(5, 0, 1), (9, 2, 3)]


def test_atomic_overlay_round_trips_changed_pair_planes(tmp_path: Path, monkeypatch) -> None:
    shape = (3, 2, 2)
    monkeypatch.setattr(jbp1, "FIELD_SHAPE", shape)
    base = np.zeros(shape, dtype=np.uint8)
    field = base.copy()
    field[1, 0, 1] = 3
    field[2, 1, 0] = 4
    base_path = tmp_path / "base.u8"
    field_path = tmp_path / "field.u8"
    base.tofile(base_path)
    field.tofile(field_path)
    receipt = jbp1.atomic_overlay(base_path, field_path, tmp_path / "overlay.npz")
    assert receipt["edited_pairs"] == [1, 2]
    assert receipt["changed_sites"] == 2
    with np.load(receipt["path"], allow_pickle=False) as blob:
        assert sorted(blob.files) == ["1", "2"]
        assert np.array_equal(blob["1"], field[1])
        assert np.array_equal(blob["2"], field[2])


def test_sfp1_refit_audit_refuses_fixed_gm_standin(tmp_path: Path) -> None:
    source = tmp_path / "fixed_instrument.py"
    source.write_text("HPAC = 'shipped'; group_plan = 'shipped'\n")
    candidate_set = {
        "candidates": [
            {
                "proposal_id": f"p{index}",
                "refit_required": True,
                "g_edit": {
                    "operation": "refit_cross_group_causal_schedule",
                    "transition_order": ["0->1"],
                    "stored_side_stream": False,
                },
            }
            for index in range(3)
        ]
    }
    audit = jbp1.sfp1_refit_audit(candidate_set, [source])
    assert audit["candidate_denominator"] == 3
    assert audit["all_require_cross_group_refit"] is True
    assert audit["operation_consumed_by_rxc1_or_jg2"] is False
    assert audit["status"] == "MISSING_EXECUTABLE_GM_REFIT"
    assert audit["blocks_fixed_gm_standin"] is True


def test_sha256_file_reads_large_files_incrementally(tmp_path: Path) -> None:
    payload = (b"jbp1" * 4096) + b"tail"
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)
    assert jbp1.sha256_file(path) == hashlib.sha256(payload).hexdigest()
