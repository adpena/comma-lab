from __future__ import annotations

import hashlib
import importlib.util
import struct
import sys
import zipfile
from pathlib import Path

import brotli
import numpy as np

from tac.qp1_pose_codec import QP1_MAGIC


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "plan_c091_pose_manifold_bigmove.py"
SPEC = importlib.util.spec_from_file_location("plan_c091_pose_manifold_bigmove", MODULE_PATH)
assert SPEC is not None
planner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = planner
assert SPEC.loader is not None
SPEC.loader.exec_module(planner)


def _write_zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("p", planner.FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _qp1_from_words(words: list[int]) -> bytes:
    return planner.encode_qp1_words(words)


def _make_p6_archive(path: Path, *, words: list[int]) -> None:
    mask_raw = b"\x12\x00\x0a\x0a" + b"m" * 64
    renderer_raw = b"QZS3" + b"r" * 64
    actions_raw = b"".join(struct.pack("<HBB", pair, 7 + pair, 40 + pair) for pair in range(6))
    actions_delta = planner.encode_delta_varint_actions(actions_raw)
    mask_br = brotli.compress(mask_raw, quality=0)
    renderer_br = brotli.compress(renderer_raw, quality=0)
    actions_br = brotli.compress(actions_delta, quality=0)
    pose_br = brotli.compress(_qp1_from_words(words), quality=0)
    payload = (
        b"P6"
        + struct.pack("<IHHH", len(mask_br), len(renderer_br), len(actions_br), 6)
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )
    _write_zip(path, payload)


def test_sparse_subspace_move_changes_ranked_pairs_without_copying_reference() -> None:
    source = [1000, 1001, 1002, 1003, 1004, 1005]
    basis = np.asarray([0, 40, 0, -30, 16, 0], dtype=np.float64)
    ranked = [
        {"pair_index": 1, "combined_score_contribution": 0.010},
        {"pair_index": 3, "combined_score_contribution": 0.008},
        {"pair_index": 4, "combined_score_contribution": 0.006},
    ]
    spec = planner.CandidateSpec("unit", "pr65_residual", 3, 0.25, 6, 0.1, 1.0, 1.0)

    candidate, changes = planner.apply_sparse_subspace_move(source, basis, ranked, spec=spec)

    assert [change["pair_index"] for change in changes] == [1, 3, 4]
    assert candidate != source
    assert candidate != list((np.asarray(source) + basis).astype(int))
    assert changes[0]["delta_q"] == 6
    assert changes[1]["delta_q"] == -6
    assert changes[2]["delta_q"] == 4


def test_build_candidate_preserves_decoded_non_pose_streams_and_recommends_when_break_even_plausible(
    tmp_path: Path,
) -> None:
    source_zip = tmp_path / "source.zip"
    _make_p6_archive(source_zip, words=[1000, 1001, 1002, 1003, 1004, 1005])
    source = planner.parse_source_archive("source", source_zip)
    source_words = np.asarray(planner.decode_qp1_words(source.decoded["optimized_poses.qp1"]), dtype=np.int32)
    trace = {
        i: {
            "pair_index": i,
            "frame_indices": [2 * i, 2 * i + 1],
            "pose_score_contribution": 0.004,
            "seg_score_contribution": 0.004,
            "combined_score_contribution": 0.008,
        }
        for i in range(6)
    }
    basis = np.asarray([9, -10, 12, -11, 8, -7], dtype=np.int32)
    spec = planner.CandidateSpec(
        "unit_bigmove",
        "pr65_residual",
        6,
        0.5,
        6,
        0.2,
        1.0,
        1.0,
    )

    manifest = planner.build_candidate(
        source=source,
        source_words=source_words,
        trace=trace,
        bases={"pr65_residual": basis},
        reference_words={"pr65": source_words + basis},
        spec=spec,
        output_dir=tmp_path / "out",
    )

    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["local_roundtrip_gates"]["all_passed"] is True
    assert manifest["local_roundtrip_gates"]["mask_decoded_preserved"] is True
    assert manifest["local_roundtrip_gates"]["renderer_decoded_preserved"] is True
    assert manifest["local_roundtrip_gates"]["actions_decoded_preserved"] is True
    assert manifest["local_roundtrip_gates"]["pose_stream_changed"] is True
    assert manifest["local_roundtrip_gates"]["no_full_public_or_prior_pose_copy"] is True
    assert manifest["selected_pair_count"] == 6
    assert manifest["break_even_plausible"] is True
    assert manifest["dispatch_recommendation"]["class"] == "exact_eval_candidate_after_claim_not_dispatched"
    assert Path(manifest["archive"]["path"]).exists()
    assert (tmp_path / "out" / "unit_bigmove" / "manifest.json").exists()


def test_parse_source_archive_rejects_sha_mismatch(tmp_path: Path) -> None:
    source_zip = tmp_path / "source.zip"
    _make_p6_archive(source_zip, words=[1000, 1001, 1002])
    wrong = hashlib.sha256(b"wrong").hexdigest()

    try:
        planner.parse_source_archive("source", source_zip, expected_sha256=wrong)
    except planner.C091PosePlanError as exc:
        assert "archive SHA mismatch" in str(exc)
    else:
        raise AssertionError("expected C091PosePlanError")


def test_qp1_word_roundtrip_keeps_magic() -> None:
    raw = planner.encode_qp1_words([111, 113, 112, 118])

    assert raw.startswith(QP1_MAGIC)
    assert planner.decode_qp1_words(raw) == [111, 113, 112, 118]
