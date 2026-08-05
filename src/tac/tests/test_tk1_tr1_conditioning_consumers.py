# SPDX-License-Identifier: MIT
"""TK1 tests for TR1 PE3 conditioning and cheapdct4 accounting consumers."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve()
WORKTREE = _HERE.parents[3]
for _p in (str(WORKTREE), str(WORKTREE / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments import train_tr1_partition_renderer_mlx as T  # noqa: E402
from tac.optimization import ddm_od4_weak_stage1_packet as od4  # noqa: E402


def _cfg(**kw) -> T.TR1Config:
    base = {
        "variant": "plain",
        "num_pairs": 3,
        "grid_downsample": 16,
        "code_width": 4,
        "renderer_width": 8,
        "token_quant_levels": 16,
        "seed": 3,
        "lotto_seed": 118,
        "lotto_mask_density_init": 0.5,
        "seg_form_start": "ce",
        "w_seg": 100.0,
        "lr": 1e-3,
        "batch_pairs": 2,
        "epochs": 2,
        "gate_every": 1,
        "ema_decay": 0.95,
        "ema_decay_provenance": "test",
        "token_temporal_mode": "shared_base",
        "token_ste": "round",
        "class_weight_lane": 1.0,
        "margin_target": 1.0,
    }
    base.update(kw)
    return T.TR1Config(**base)


def _ns(*extra: str):
    return T.build_argparser().parse_args(["--variant", "plain", "--out-dir", "unused", *extra])


def _varint(value: int) -> bytes:
    value = int(value)
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _stage2_payload(*, k: int = 4, pairs: tuple[int, ...] = (8, 32)) -> bytes:
    body = bytearray(b"OD8S2C1\0")
    body += _varint(len(pairs))
    body += _varint(k)
    source = b"unit_test"
    body += _varint(len(source)) + source
    for pair in pairs:
        q = (np.arange(3 * k * k, dtype=np.int16).reshape(3, k * k) + int(pair)).astype("<i2")
        body += _varint(pair)
        body += _varint(q.size)
        body += q.tobytes()
    return bytes(body)


def _cheapdct_receipt(tmp_path: Path, *, d_pose: float = 0.0007918090370822028) -> Path:
    packet = od4.serialize_od5_packet([
        od4.OD5Section("od8_stage2_cheapdct4_qcoeffs", _stage2_payload()),
    ])
    packet_path = tmp_path / "stage2.od5.raw_packet"
    packet_path.write_bytes(packet)
    receipt = {
        "artifacts": {
            "best_combined_packet": {
                "path": str(packet_path),
                "sha256": T._sha256_path(packet_path),
            },
        },
        "pose_subset_scope": {
            "d_pose_after_stage2_cheapdct_mean_n32": d_pose,
            "not_projected_to_n600": True,
        },
        "combined_table": [
            ["stage2_only_cheapdct4_qcoeffs", 2157, 40444, -35860, -33964, -49556],
        ],
        "axis": {"pricing": "[macOS-CPU byte-only persisted-native pricing]"},
        "selection": "unit-test-n32",
    }
    receipt_path = tmp_path / "OD9_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    return receipt_path


# ---------------------------------------------------------------- PE3 guards/features ----
def test_tk1_parser_defaults_are_off():
    ns = _ns()
    assert ns.pe3_conditioning_mode == "off"
    assert ns.pe3_conditioning_cache is None
    assert ns.cheapdct4_pose_mode == "off"
    assert ns.cheapdct4_pose_cache is None


def test_tk1_flags_are_args_only_not_tr1_config_fields():
    cfg_fields = set(T.TR1Config.__dataclass_fields__)
    assert not any("pe3" in name or "cheapdct4" in name for name in cfg_fields)
    assert _cfg().config_hash() == _cfg().config_hash()


def test_pe3_conditioning_refuses_mode_without_cache():
    with pytest.raises(SystemExit, match="requires --pe3-conditioning-cache"):
        T.validate_tk1_consumer_args(_ns("--pe3-conditioning-mode", "conditioning_only"))


def test_pe3_conditioning_refuses_cache_while_off(tmp_path: Path):
    cache = tmp_path / "pe3.bin"
    cache.write_bytes(b"x")
    with pytest.raises(SystemExit, match="mode off"):
        T.validate_tk1_consumer_args(_ns("--pe3-conditioning-cache", str(cache)))


def test_pe3_conditioning_refuses_missing_cache(tmp_path: Path):
    missing = tmp_path / "missing.bin"
    with pytest.raises(SystemExit, match="missing"):
        T.validate_tk1_consumer_args(
            _ns("--pe3-conditioning-mode", "conditioning_only", "--pe3-conditioning-cache", str(missing))
        )


def test_pe3_conditioning_validation_accepts_existing_cache_path(tmp_path: Path):
    cache = tmp_path / "pe3.bin"
    cache.write_bytes(b"placeholder")
    ns = _ns("--pe3-conditioning-mode", "conditioning_only", "--pe3-conditioning-cache", str(cache))
    T.validate_tk1_consumer_args(ns)


def test_pe3_features_shape_summary_and_score_boundary():
    comps = [
        [
            {"mode_name": "generator_pair_bisector", "indices": np.array([0, 1, 512]), "classes": np.array([1, 1, 0])},
        ],
        [
            {"mode_name": "depth_conditioned_curve", "indices": np.array([1024, 1025]), "classes": np.array([2, 3])},
        ],
        [],
    ]
    features, summary = T.pe3_conditioning_features_from_components(_cfg(), comps)
    assert features.shape == (2, 3, 24, 32, 4)
    assert summary["label_replacement"] is False
    assert summary["score_claim"] is False
    assert summary["gate_init"] == "zeros"


def test_pe3_features_keep_modes_independent():
    comps = [
        [
            {"mode_name": "generator_pair_bisector", "indices": np.array([0]), "classes": np.array([1])},
            {"mode_name": "depth_conditioned_curve", "indices": np.array([511]), "classes": np.array([3])},
        ],
        [],
        [],
    ]
    features, summary = T.pe3_conditioning_features_from_components(_cfg(), comps)
    assert np.abs(features[0]).sum() > 0
    assert np.abs(features[1]).sum() > 0
    assert summary["described_pixels_by_mode"]["generator_pair_bisector"] == 1
    assert summary["described_pixels_by_mode"]["depth_conditioned_curve"] == 1


def test_pe3_features_refuse_too_few_pairs():
    with pytest.raises(ValueError, match="pairs < --num-pairs"):
        T.pe3_conditioning_features_from_components(_cfg(num_pairs=3), [])


def test_pe3_features_refuse_unknown_mode():
    comps = [[{"mode_name": "bogus", "indices": np.array([0]), "classes": np.array([1])}], [], []]
    with pytest.raises(ValueError, match="unknown PE3 conditioning mode"):
        T.pe3_conditioning_features_from_components(_cfg(), comps)


def test_pe3_features_refuse_index_class_length_mismatch():
    comps = [
        [{"mode_name": "generator_pair_bisector", "indices": np.array([0, 1]), "classes": np.array([1])}],
        [],
        [],
    ]
    with pytest.raises(ValueError, match="length differs"):
        T.pe3_conditioning_features_from_components(_cfg(), comps)


def test_pe3_features_refuse_out_of_range_indices():
    comps = [
        [{"mode_name": "generator_pair_bisector", "indices": np.array([T.SEG_H * T.SEG_W]), "classes": np.array([1])}],
        [],
        [],
    ]
    with pytest.raises(ValueError, match="outside scorer grid"):
        T.pe3_conditioning_features_from_components(_cfg(), comps)


def test_pe3_class_pattern_is_bounded_and_channelized():
    pat = T._pe3_class_pattern(np.array([0, 1, 4]), 6)
    assert pat.shape == (3, 6)
    assert float(np.abs(pat).max()) <= 1.0
    assert not np.array_equal(pat[0], pat[1])


def test_pe3_token_proximity_empty_is_zero():
    prox = T._token_grid_proximity(np.zeros((4, 5), dtype=bool))
    assert prox.shape == (4, 5)
    assert float(prox.sum()) == 0.0


def test_pe3_token_proximity_marks_active_and_decays():
    active = np.zeros((5, 5), dtype=bool)
    active[2, 2] = True
    prox = T._token_grid_proximity(active)
    assert prox[2, 2] == pytest.approx(1.0)
    assert 0.0 < prox[0, 0] < 1.0
    assert prox[2, 1] > prox[0, 0]


def test_extract_pe3_conditioning_missing_path_refuses(tmp_path: Path):
    with pytest.raises(ValueError, match="does not exist"):
        T._extract_pe3_conditioning_section(tmp_path / "missing.zip")


# ---------------------------------------------------------------- cheapdct4 guards/accounting ----
def test_cheapdct4_refuses_mode_without_cache():
    with pytest.raises(SystemExit, match="requires --cheapdct4-pose-cache"):
        T.validate_tk1_consumer_args(_ns("--cheapdct4-pose-mode", "accounting"))


def test_cheapdct4_refuses_cache_while_off(tmp_path: Path):
    cache = tmp_path / "OD9_RECEIPT.json"
    cache.write_text("{}\n")
    with pytest.raises(SystemExit, match="mode off"):
        T.validate_tk1_consumer_args(_ns("--cheapdct4-pose-cache", str(cache)))


def test_cheapdct4_refuses_missing_cache(tmp_path: Path):
    missing = tmp_path / "OD9_RECEIPT.json"
    with pytest.raises(SystemExit, match="missing"):
        T.validate_tk1_consumer_args(
            _ns("--cheapdct4-pose-mode", "accounting", "--cheapdct4-pose-cache", str(missing))
        )


def test_cheapdct4_validation_accepts_existing_receipt_path(tmp_path: Path):
    cache = tmp_path / "OD9_RECEIPT.json"
    cache.write_text("{}\n")
    ns = _ns("--cheapdct4-pose-mode", "accounting", "--cheapdct4-pose-cache", str(cache))
    T.validate_tk1_consumer_args(ns)


def test_decode_cheapdct4_stage2_payload_records():
    records, meta = T.decode_cheapdct4_stage2_payload(_stage2_payload(k=4, pairs=(8, 32)))
    assert meta["record_count"] == 2
    assert meta["k"] == 4
    assert meta["raw_int16_coeff_bytes"] == 2 * 3 * 4 * 4 * 2
    assert [r["pair"] for r in records] == [8, 32]
    assert records[0]["qcoeffs"].shape == (3, 16)


def test_decode_cheapdct4_refuses_bad_magic():
    with pytest.raises(ValueError, match="magic"):
        T.decode_cheapdct4_stage2_payload(b"BAD")


def test_decode_cheapdct4_refuses_bad_coeff_count():
    payload = bytearray(b"OD8S2C1\0")
    payload += _varint(1) + _varint(4) + _varint(1) + b"x"
    payload += _varint(0) + _varint(1) + b"\0\0"
    with pytest.raises(ValueError, match="qcoeff count"):
        T.decode_cheapdct4_stage2_payload(bytes(payload))


def test_decode_cheapdct4_refuses_truncated_source():
    payload = b"OD8S2C1\0" + _varint(0) + _varint(4) + _varint(5) + b"x"
    with pytest.raises(ValueError, match="source is truncated"):
        T.decode_cheapdct4_stage2_payload(payload)


def test_decode_cheapdct4_refuses_trailing_bytes():
    with pytest.raises(ValueError, match="trailing"):
        T.decode_cheapdct4_stage2_payload(_stage2_payload() + b"x")


def test_load_cheapdct4_accounting_decodes_packet_and_pose(tmp_path: Path):
    receipt = _cheapdct_receipt(tmp_path)
    acc = T.load_cheapdct4_pose_accounting_cache(receipt)
    assert acc["record_count"] == 2
    assert acc["projected_n600_bytes"] == 40444
    assert acc["d_pose_after_stage2_cheapdct_mean_n32"] == pytest.approx(0.0007918090370822028)
    assert acc["pose_contribution_n32"] == pytest.approx(math.sqrt(10 * 0.0007918090370822028))
    assert acc["score_claim"] is False


def test_load_cheapdct4_refuses_packet_sha_mismatch(tmp_path: Path):
    receipt = _cheapdct_receipt(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["artifacts"]["best_combined_packet"]["sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="SHA mismatch"):
        T.load_cheapdct4_pose_accounting_cache(receipt)


def test_load_cheapdct4_refuses_missing_pose_term(tmp_path: Path):
    receipt = _cheapdct_receipt(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["pose_subset_scope"] = {}
    receipt.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="lacks d_pose"):
        T.load_cheapdct4_pose_accounting_cache(receipt)


def test_load_cheapdct4_refuses_missing_stage2_byte_row(tmp_path: Path):
    receipt = _cheapdct_receipt(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["combined_table"] = []
    receipt.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="lacks stage2_only"):
        T.load_cheapdct4_pose_accounting_cache(receipt)


def test_load_cheapdct4_refuses_missing_packet_info(tmp_path: Path):
    receipt = tmp_path / "OD9_RECEIPT.json"
    receipt.write_text(json.dumps({"artifacts": {}}) + "\n")
    with pytest.raises(ValueError, match="lacks a packet"):
        T.load_cheapdct4_pose_accounting_cache(receipt)


def test_load_cheapdct4_refuses_non_json_cache(tmp_path: Path):
    cache = tmp_path / "packet.od5.raw_packet"
    cache.write_bytes(b"x")
    with pytest.raises(ValueError, match="requires the OD9 receipt JSON"):
        T.load_cheapdct4_pose_accounting_cache(cache)


def test_extract_cheapdct4_stage2_refuses_absent_section(tmp_path: Path):
    packet = od4.serialize_od5_packet([od4.OD5Section("other", b"x")])
    path = tmp_path / "bad.od5.raw_packet"
    path.write_bytes(packet)
    with pytest.raises(ValueError, match="exactly one cheapdct4"):
        T._extract_cheapdct4_stage2_from_od5_packet(path)


def test_attach_cheapdct4_accounting_to_receipt_copies_into_composed():
    receipt = {"composed_s_verdict": {"score_claim": False}}
    accounting = {"schema": "x", "score_claim": False}
    T.attach_cheapdct4_accounting_to_receipt(receipt, accounting)
    assert receipt["cheapdct4_pose_accounting"] == accounting
    assert receipt["composed_s_verdict"]["cheapdct4_pose_accounting"] == accounting
