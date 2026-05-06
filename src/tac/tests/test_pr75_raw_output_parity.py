from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO / "experiments" / "pr75_raw_output_parity.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("pr75_raw_output_parity_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _uleb128(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _zigzag(value: int) -> int:
    return (value << 1) ^ (value >> 31)


def test_public_qp1_float32_decode_exposes_fp16_materialization_drift() -> None:
    script = _load_script()
    q0 = 1
    q1 = 2
    payload = b"QP1" + q0.to_bytes(2, "little") + _uleb128(_zigzag(q1 - q0))

    public_pose = script.decode_public_qp1_pose_float32(payload)
    robust_pose = script.fp16_materialize_pose(public_pose)
    report = script.pose_precision_report(public_pose, robust_pose)

    assert public_pose.shape == (2, 6)
    assert public_pose[0, 0] == pytest.approx(20.0 + 1.0 / 512.0)
    assert public_pose[1, 0] == pytest.approx(20.0 + 2.0 / 512.0)
    assert report["public_float32_vs_robust_fp16"]["exact_equal"] is False
    assert report["robust_matches_public_fp16_roundtrip"] is True
    assert report["public_float32_vs_public_fp16_roundtrip"]["changed_values"] == 2


def test_split_public_pr75_fixed_payload_uses_documented_lengths() -> None:
    script = _load_script()
    payload = (
        b"m" * script.PUBLIC_PR75_MASK_LEN
        + b"r" * script.PUBLIC_PR75_MODEL_LEN
        + b"a" * script.PUBLIC_PR75_ACTIONS_LEN
        + b"p"
        * (
            script.PUBLIC_PR75_PAYLOAD_LEN
            - script.PUBLIC_PR75_MASK_LEN
            - script.PUBLIC_PR75_MODEL_LEN
            - script.PUBLIC_PR75_ACTIONS_LEN
        )
    )

    slices = script.split_public_pr75_payload(payload)

    assert slices["payload_format"] == b"public_pr75_fixed"
    assert len(slices["masks.mkv.br"]) == script.PUBLIC_PR75_MASK_LEN
    assert len(slices["renderer.bin.br"]) == script.PUBLIC_PR75_MODEL_LEN
    assert len(slices["seg_tile_actions.br"]) == script.PUBLIC_PR75_ACTIONS_LEN
    assert len(slices["optimized_poses.bin.br"]) == 899


def test_split_public_pr75_minp_fixed_payload_uses_current_lengths() -> None:
    script = _load_script()
    payload = (
        b"m" * script.PUBLIC_PR75_MASK_LEN
        + b"r" * 55_756
        + b"a" * 255
        + b"p" * 898
    )

    slices = script.split_public_pr75_payload(payload)

    assert slices["payload_format"] == b"public_pr75_fixed"
    assert len(slices["masks.mkv.br"]) == script.PUBLIC_PR75_MASK_LEN
    assert len(slices["renderer.bin.br"]) == 55_756
    assert len(slices["seg_tile_actions.br"]) == 255
    assert len(slices["optimized_poses.bin.br"]) == 898


def test_resolve_pair_indices_dedupes_and_bounds() -> None:
    script = _load_script()

    assert script.resolve_pair_indices(
        total_pairs=10, pair_indices=[3, 1, 3], max_pairs=None
    ) == [3, 1]
    assert script.resolve_pair_indices(
        total_pairs=10, pair_indices=[], max_pairs=3
    ) == [0, 1, 2]
    with pytest.raises(ValueError, match="outside"):
        script.resolve_pair_indices(total_pairs=10, pair_indices=[10], max_pairs=None)
    with pytest.raises(ValueError, match="nonnegative"):
        script.resolve_pair_indices(total_pairs=10, pair_indices=[], max_pairs=-1)


def test_streaming_accumulators_preserve_hashes_and_global_offsets() -> None:
    script = _load_script()
    lhs0 = b"\x01\x02\x03"
    rhs0 = b"\x01\x02\x03"
    lhs1 = b"\x04\x09"
    rhs1 = b"\x04\x05"

    byte_acc = script.ByteAccumulator()
    assert byte_acc.update(lhs0, rhs0)["exact_equal"] is True
    assert byte_acc.update(lhs1, rhs1)["first_diff_offset"] == 1
    byte_report = byte_acc.finish()

    assert byte_report["exact_equal"] is False
    assert byte_report["first_diff_offset"] == 4
    assert byte_report["changed_prefix_bytes"] == 1
    assert byte_report["max_abs_byte_delta"] == 4
    assert byte_report["lhs_sha256"] == script.sha256_bytes(lhs0 + lhs1)
    assert byte_report["rhs_sha256"] == script.sha256_bytes(rhs0 + rhs1)

    num_acc = script.NumericAccumulator([4], [4])
    num_acc.update(
        script.np.array([1.0, 2.0], dtype=script.np.float32),
        script.np.array([1.0, 2.0], dtype=script.np.float32),
    )
    num_acc.update(
        script.np.array([3.0, 7.0], dtype=script.np.float32),
        script.np.array([3.0, 4.0], dtype=script.np.float32),
    )
    num_report = num_acc.finish()

    assert num_report["exact_equal"] is False
    assert num_report["first_diff_flat_index"] == 3
    assert num_report["changed_values"] == 1
    assert num_report["max_abs"] == pytest.approx(3.0)


def test_all_pairs_parser_flags_are_explicit_opt_in() -> None:
    script = _load_script()

    args = script.build_arg_parser().parse_args(
        ["--all-pairs", "--chunk-size", "16", "--fast-fail"]
    )

    assert args.all_pairs is True
    assert args.chunk_size == 16
    assert args.fast_fail is True
