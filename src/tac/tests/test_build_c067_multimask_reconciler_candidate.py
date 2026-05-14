# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import hashlib
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO_ROOT / "experiments" / "build_c067_multimask_reconciler_candidate.py"
INFLATE_PATH = REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("c067_multimask_reconciler_builder_test", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_inflate():
    spec = importlib.util.spec_from_file_location("c067_multimask_reconciler_inflate_test", INFLATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_policy_skips_noop_and_requires_score_claim_false() -> None:
    builder = _load_builder()
    plan = {
        "candidate_policies": [
            {
                "policy_id": "noop",
                "score_claim": False,
                "dispatch_relevance": {
                    "dispatchable_byte_model": False,
                    "no_op_vs_source": True,
                },
            },
            {
                "policy_id": "usable",
                "score_claim": False,
                "dispatch_relevance": {
                    "dispatchable_byte_model": True,
                    "no_op_vs_source": False,
                },
            },
        ]
    }

    assert builder.select_policy(plan)["policy_id"] == "usable"

    plan["candidate_policies"][1]["score_claim"] = True
    with pytest.raises(builder.ReconcilerBuildError, match="no dispatchable"):
        builder.select_policy(plan)


def test_materialize_majority_vote_preserves_source_ties() -> None:
    builder = _load_builder()
    source = np.array([[[0, 9, 2, 3], [4, 4, 1, 1]]], dtype=np.uint8)
    candidate_a = np.array([[[1, 8, 2, 3], [4, 7, 1, 6]]], dtype=np.uint8)
    candidate_b = np.array([[[0, 8, 5, 3], [4, 7, 5, 6]]], dtype=np.uint8)
    candidate_c = np.array([[[1, 9, 5, 3], [0, 7, 5, 6]]], dtype=np.uint8)
    plan = {"source_family_name": "source"}
    policy = {
        "fusion_reconciliation_policy": {
            "name": "majority_vote",
            "inputs": ["source", "a", "b", "c"],
        }
    }

    fused = builder.materialize_fused_mask(
        plan=plan,
        policy=policy,
        source=source,
        candidates={"a": candidate_a, "b": candidate_b, "c": candidate_c},
    )

    assert fused.dtype == np.uint8
    assert fused.tolist() == [[[0, 9, 2, 3], [4, 7, 1, 6]]]


def test_materialize_gated_veto_and_residual_fail_closed() -> None:
    builder = _load_builder()
    source = np.zeros((1, 2, 4), dtype=np.uint8)
    a = source.copy()
    b = source.copy()
    a[0, 0, 1:3] = 2
    b[0, 0, 1:3] = 2
    b[0, 1, 1:3] = 3

    veto = builder.materialize_fused_mask(
        plan={"source_family_name": "source"},
        policy={
            "candidate_family_names": ["a", "b"],
            "fusion_reconciliation_policy": {
                "name": "disagreement_gated_veto",
                "candidate_consensus_threshold": 1.0,
            },
        },
        source=source,
        candidates={"a": a, "b": b},
    )
    expected = source.copy()
    expected[0, 0, 1:3] = 2
    np.testing.assert_array_equal(veto, expected)

    with pytest.raises(builder.ReconcilerBuildError, match="unknown family"):
        builder.materialize_fused_mask(
            plan={"source_family_name": "source"},
            policy={
                "fusion_reconciliation_policy": {
                    "name": "cheap_residual_over_base",
                    "residual_family": "missing",
                },
            },
            source=source,
            candidates={"a": a},
        )


def test_target_body_parser_requires_positive_values() -> None:
    builder = _load_builder()
    assert builder._parse_target_body_bytes("200000, 212000") == (200000, 212000)
    with pytest.raises(Exception, match="positive"):
        builder._parse_target_body_bytes("200000,0")


def test_archive_cmg3_header_hash_matches_runtime_decode(tmp_path: Path) -> None:
    builder = _load_builder()
    inflate = _load_inflate()
    frontier_archive = tmp_path / "frontier.zip"
    frontier_archive.write_bytes(b"frontier-bytes")
    masks = np.zeros((1, 384, 512), dtype=np.uint8)
    masks[0, 10, 20:40] = 2
    masks[0, 10, 60:75] = 3
    source_sha = hashlib.sha256(masks.tobytes(order="C")).hexdigest()

    manifest = builder._build_one_archive(
        output_dir=tmp_path / "out",
        frontier_archive=frontier_archive,
        frontier_members={
            "renderer.bin": b"QZS3fake",
            "masks.mkv": b"old-mask",
            "optimized_poses.bin": struct.pack("<" + "e" * 6, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        },
        source=masks,
        fused=masks,
        target_body_bytes=None,
        target_extra_runs=1,
        base_runs_per_row=1,
        adaptive_max_runs_per_row=3,
        compressor="raw",
        body_search_mode="auto",
    )

    with zipfile.ZipFile(tmp_path / "out" / "extra000001" / "multimask_reconciler_source_members.zip") as zf:
        cmg3_payload = zf.read("masks.cmg3")
    header, body = inflate._decode_cmg3_payload(cmg3_payload)
    raw = inflate._decompress_cmg3_body(body, header["compressor"])
    decoded = inflate._decode_cmg3_nonzero_row_runs(raw, header)
    decoded_sha = hashlib.sha256(np.ascontiguousarray(decoded, dtype=np.uint8).tobytes(order="C")).hexdigest()

    assert manifest["cmg3"]["source_mask_u8_sha256"] == source_sha
    assert manifest["cmg3"]["reconstructed_mask_u8_sha256"] == decoded_sha
    assert header["reconstructed_mask_u8_sha256"] == decoded_sha
