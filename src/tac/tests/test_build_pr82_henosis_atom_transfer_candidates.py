# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import pytest

from tac.henosis_pr82_transfer import (
    Pr82ReplayContract,
    decode_control_arrays,
    decode_randmulti_groups,
    decode_randmulti_qrm1,
    encode_randmulti_nm2,
    encode_randmulti_qrm1,
    encode_qpost,
    filter_qpost_streams_to_pairs,
    parse_pr82_bundle,
    pose_velocity_atom_ranking,
    randmulti_qrm1_parity_profile,
    qpost_stream_summary,
    summarize_pair_activity,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments/build_pr82_henosis_atom_transfer_candidates.py"
PR82_ARCHIVE = REPO_ROOT / "experiments/results/public_pr82_henosis_frontier_intake_20260503_codex/archive.zip"
PR82_REPLAY = (
    REPO_ROOT
    / "experiments/results/public_pr82_henosis_frontier_intake_20260503_codex/replay_submission/inflate.py"
)
PR79_S2_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr79_action_dictionary_repack_v2_20260503_codex/"
    "pr79_s2_fixed_adaptive_actions/archive.zip"
)


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("build_pr82_henosis_atom_transfer_candidates", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _zz(value: int) -> int:
    return (value << 1) if value >= 0 else ((-value) << 1) - 1


def _vlq(value: int) -> bytes:
    out = bytearray()
    while True:
        if value < 0x80:
            out.append(value)
            return bytes(out)
        out.append((value & 0x7F) | 0x80)
        value >>= 7


def _randmulti_sparse_row(indices: list[int], values: list[int]) -> bytes:
    assert len(indices) == len(values)
    out = bytearray([len(indices)])
    previous = -1
    for index in indices:
        out.extend(_vlq(index - previous - 1))
        previous = index
    out.extend(values)
    return bytes(out)


def _p1d1_pose(values: list[int]) -> bytes:
    stream = bytearray()
    previous = 0
    for value in values:
        stream.extend(_vlq(_zz(value - previous)))
        previous = value
    return b"P1D1" + bytes([1, 0]) + len(stream).to_bytes(2, "little") + bytes(stream)


def _synthetic_pr82_payload() -> tuple[bytes, dict[str, bytes], Pr82ReplayContract]:
    post = np.zeros((4, 600), dtype=np.uint8)
    post[0, 5] = 2
    post[2, 9] = 7
    shift = np.full(600, 40, dtype=np.uint8)
    shift[5] = 41
    frac = np.full(600, 4, dtype=np.uint8)
    frac[9] = 5
    frac2 = np.full(600, 4, dtype=np.uint8)
    frac2[5] = 6
    frac3 = np.full(600, 4, dtype=np.uint8)
    bias = np.full(600, 13, dtype=np.uint8)
    bias[9] = 14
    region = np.zeros(600, dtype=np.uint8)
    region[5] = 3
    pose_values = [0] * 600
    pose_values[5] = 8
    encoded = {
        "mask": brotli.compress(b"mask"),
        "model": brotli.compress(b"QH0" + bytes(32)),
        "pose": brotli.compress(_p1d1_pose(pose_values)),
        "post": brotli.compress(post.tobytes()),
        "shift": brotli.compress(b"SH4" + shift.tobytes()),
        "frac": brotli.compress(b"FH1" + frac.tobytes()),
        "frac2": brotli.compress(b"FH2" + frac2.tobytes()),
        "frac3": brotli.compress(b"FH3" + frac3.tobytes()),
        "bias": brotli.compress(b"BH1" + bias.tobytes()),
        "region": brotli.compress(b"RH1" + region.tobytes()),
        "randmulti": brotli.compress(b"\x00"),
    }
    header_names = ("mask", "model", "pose", "post", "shift", "frac", "frac2", "frac3")
    raw = b"".join(len(encoded[name]).to_bytes(3, "little") for name in header_names)
    raw += b"".join(encoded[name] for name in header_names)
    raw += encoded["bias"] + encoded["region"] + encoded["randmulti"]
    contract = Pr82ReplayContract(
        fixed_bias_bytes=len(encoded["bias"]),
        fixed_region_bytes=len(encoded["region"]),
        randmulti_specs=((1, 1, 1, 1),),
    )
    return raw, encoded, contract


def test_pr82_qpost_pair_filter_is_charged_and_nonselected_pairs_default() -> None:
    raw, encoded, contract = _synthetic_pr82_payload()
    bundle = parse_pr82_bundle(raw, contract)
    arrays = decode_control_arrays(bundle.encoded_segments)
    pair_rows = summarize_pair_activity(arrays)

    assert pair_rows[5]["active_atom_count"] == 4
    assert pair_rows[9]["active_atom_count"] == 3

    streams = filter_qpost_streams_to_pairs(
        bundle.encoded_segments,
        [5],
        include_streams=("post", "shift", "frac", "frac2", "frac3", "bias", "region"),
    )
    qpost = encode_qpost(streams)
    summary = qpost_stream_summary(streams, encoded)

    assert qpost.startswith(b"QPS1")
    assert len(qpost) > 4 + 8 * 4
    assert summary["randmulti"]["active"] is False

    filtered_arrays = decode_control_arrays(streams)
    assert int(np.count_nonzero(filtered_arrays["post"][:, 5])) == 1
    assert int(np.count_nonzero(filtered_arrays["post"][:, 9])) == 0
    assert filtered_arrays["bias"][9] == 13


def test_pose_velocity_atom_ranking_detects_noop_and_delta_order() -> None:
    source = np.zeros((600, 6), dtype=np.float32)
    source[:, 0] = 20.0
    pr82 = source.copy()
    pr82[10, 0] += 4 / 512.0
    pr82[2, 0] -= 9 / 512.0

    atoms = pose_velocity_atom_ranking(source, pr82)

    assert atoms[:2] == [
        {"abs_delta_q": 9, "delta_q": -9, "dimension": 0, "pair_index": 2},
        {"abs_delta_q": 4, "delta_q": 4, "dimension": 0, "pair_index": 10},
    ]


def test_randmulti_groups_decode_and_nm2_filter_round_trip() -> None:
    raw, _encoded, _contract = _synthetic_pr82_payload()
    randmulti = _randmulti_sparse_row([5, 9], [3, 7]) + _randmulti_sparse_row([9], [2])
    encoded = brotli.compress(randmulti)

    groups = decode_randmulti_groups(
        encoded,
        ((4, 4, 1, 2),),
    )
    nm2 = encode_randmulti_nm2(groups, pair_indices=[5])

    decoded = brotli.decompress(nm2)

    assert raw
    assert decoded[:4] == b"NM2\x01"
    assert decoded[4:8] == bytes([4, 4, 1, 2])
    arr = np.frombuffer(decoded, dtype=np.uint8, count=2 * 600, offset=8).reshape(2, 600)
    assert arr[0, 5] == 3
    assert arr[0, 9] == 0
    assert arr[1, 9] == 0


def test_randmulti_qrm1_round_trips_large_and_special_pr82_groups() -> None:
    raw = (
        _randmulti_sparse_row([5, 599], [3, 7])
        + _randmulti_sparse_row([9], [2])
        + _randmulti_sparse_row([8], [4])
    )
    encoded = brotli.compress(raw)
    specs = (
        (1024, 1, 1, 2),
        (223, 222, 2, 1),
    )
    groups = decode_randmulti_groups(encoded, specs)

    qrm1 = encode_randmulti_qrm1(groups)
    decoded = decode_randmulti_qrm1(qrm1, specs)
    profile = randmulti_qrm1_parity_profile(groups, decoded, encoded=qrm1, source_encoded=encoded)

    assert profile["contract"] == "QRM1_sparse_group_id_stream"
    assert profile["decoded_group_count"] == 2
    assert profile["exact_group_row_parity"] is True
    assert profile["generic_group_count"] == 1
    assert profile["replay_special_group_count"] == 1
    assert decoded[0].height == 1024
    assert decoded[0].rows[0, 599] == 7
    assert decoded[1].rows[0, 8] == 4


def test_randmulti_qrm1_pair_filter_preserves_group_specs_and_zeroes_other_pairs() -> None:
    encoded = brotli.compress(_randmulti_sparse_row([5, 9], [3, 7]))
    groups = decode_randmulti_groups(encoded, ((4096, 1, 1, 1),))

    qrm1 = encode_randmulti_qrm1(groups, pair_indices=[9])
    decoded = decode_randmulti_qrm1(qrm1, ((4096, 1, 1, 1),))

    assert decoded[0].height == 4096
    assert decoded[0].rows[0, 5] == 0
    assert decoded[0].rows[0, 9] == 7


def test_argparse_exposes_sha_bypass_flags() -> None:
    script = _load_script()
    args = script.build_arg_parser().parse_args(
        [
            "--pr82-archive",
            "pr82.zip",
            "--source-archive",
            "source.zip",
            "--output-dir",
            "out",
            "--qpost-topks",
            "2",
            "--pose-topks",
            "2",
            "--randmulti-topks",
            "1,4",
            "--no-pr82-sha-check",
            "--no-source-sha-check",
        ]
    )

    assert str(args.pr82_archive) == "pr82.zip"
    assert args.no_pr82_sha_check is True
    assert args.no_source_sha_check is True
    assert args.randmulti_topks == "1,4"


def test_brotli_best_does_not_reuse_stale_source_bytes() -> None:
    script = _load_script()
    source = brotli.compress(b"old-pose")

    encoded, params = script._brotli_best(b"new-pose", source=source)  # noqa: SLF001

    assert brotli.decompress(encoded) == b"new-pose"
    assert params != "source"


@pytest.mark.skipif(
    not (PR82_ARCHIVE.exists() and PR82_REPLAY.exists() and PR79_S2_ARCHIVE.exists()),
    reason="PR82 intake or PR79/S2 source archive missing",
)
def test_actual_pr82_transfer_builds_fail_closed_candidates(tmp_path: Path) -> None:
    script = _load_script()

    summary = script.build_candidates(
        pr82_archive=PR82_ARCHIVE,
        replay_inflate=PR82_REPLAY,
        source_archive=PR79_S2_ARCHIVE,
        source_exact_json=None,
        output_dir=tmp_path,
        qpost_topks=(2,),
        pose_topks=(2,),
        include_streams=("post", "shift", "frac", "frac2", "frac3", "bias", "region"),
        expected_pr82_sha256=None,
        expected_source_sha256=None,
    )

    assert summary["score_claim"] is False
    assert summary["no_remote_dispatch"] is True
    assert summary["candidate_count"] == 6
    assert summary["qpost_candidates"][0]["dispatch_gate"]["dispatch_ready_now"] is False
    assert summary["randmulti_candidates"][0]["dispatch_gate"]["dispatch_ready_now"] is False
    assert summary["randmulti_qrm1_candidates"][0]["qrm1_local_decode_profile"]["exact_group_row_parity"] is True
    assert summary["randmulti_qrm1_candidates"][0]["dispatch_gate"]["dispatch_ready_now"] is False
    assert summary["pose_candidates"][0]["dispatch_gate"]["remote_dispatch_performed"] is False
    assert (tmp_path / "candidate_summary.json").exists()
    assert (tmp_path / "pr82_randmulti_lowlevel_profile.json").exists()
