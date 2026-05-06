from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.pr85_bundle import SEGMENT_ORDER, pack_pr85_bundle
from tac.pr91_hpm1_codec import (
    Pr91Hpm1Error,
    build_hpm1_mask_segment,
    extract_pr91_hpm1_payload,
    raw_tokens_to_mod5_residual_symbols,
    reconstruct_raw_tokens_from_mod5_residual_symbols,
    run_pr91_hpm1_preflight,
    run_pr91_hpm1_probability_variant_matrix,
    split_hpm1_mask_segment,
    validate_hpm1_static_contract,
)

REPO = Path(__file__).resolve().parents[3]
PREFLIGHT_SCRIPT = REPO / "experiments" / "preflight_pr91_pr92_replay_contracts.py"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_preflight_script():
    spec = importlib.util.spec_from_file_location("preflight_pr91_pr92_replay_contracts_test", PREFLIGHT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stored_x_archive(path: Path, x_body: bytes) -> None:
    info = zipfile.ZipInfo("x", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, x_body)


def _synthetic_hpm1_segment() -> bytes:
    return build_hpm1_mask_segment(
        (np.arange(32, dtype=np.uint32) % 5).tobytes(),
        b"synthetic-hpac-ppmd",
        N=2,
        H=4,
        W=4,
        P=2,
        delta=1,
        ch=4,
        use_spm=False,
        hpac_d_film=2,
    )


def _synthetic_hpm1_archive(path: Path) -> Path:
    segments = {name: f"{name}-payload".encode("ascii") for name in SEGMENT_ORDER}
    segments["mask"] = _synthetic_hpm1_segment()
    segments["bias"] = b"b" * 223
    segments["region"] = b"r" * 273
    _stored_x_archive(path, pack_pr85_bundle(segments, header_mode="v5"))
    return path


def test_hpm1_segment_builder_and_parser_are_byte_closed() -> None:
    segment = _synthetic_hpm1_segment()
    payload = split_hpm1_mask_segment(segment)

    assert payload.config() == {
        "n_frames": 2,
        "height": 4,
        "width": 4,
        "predictor_count": 2,
        "delta": 1,
        "channels": 4,
        "use_spm": 0,
        "hpac_d_film": 2,
        "tokens_len": 128,
        "hpac_len": 19,
        "ppmd_order": 4,
    }
    assert payload.extra["tokens_sha256"] == _sha(payload.tokens)
    assert payload.extra["hpac_sha256"] == _sha(payload.hpac)
    assert validate_hpm1_static_contract(payload)["status"] == "passed"


def test_hpm1_segment_builder_fails_closed_on_misaligned_tokens() -> None:
    with pytest.raises(Pr91Hpm1Error, match="tokens_blob_must_be_nonempty_uint32_aligned"):
        build_hpm1_mask_segment(
            b"abc",
            b"model",
            N=1,
            H=1,
            W=1,
            P=1,
            delta=0,
            ch=1,
            use_spm=False,
            hpac_d_film=1,
        )


def test_residual_symbol_helpers_roundtrip_raw_tokens() -> None:
    raw = np.array(
        [
            [[0, 1, 2], [3, 4, 0]],
            [[1, 1, 3], [3, 0, 4]],
            [[4, 0, 3], [2, 2, 1]],
        ],
        dtype=np.uint8,
    )

    residual, prev = raw_tokens_to_mod5_residual_symbols(raw)
    reconstructed = reconstruct_raw_tokens_from_mod5_residual_symbols(residual, prev)

    assert np.array_equal(reconstructed, raw)
    assert np.array_equal(prev[1], raw[0])


def test_pr91_probability_matrix_is_fail_closed_not_notimplemented(tmp_path: Path) -> None:
    archive = _synthetic_hpm1_archive(tmp_path / "archive.zip")

    report = run_pr91_hpm1_probability_variant_matrix(
        archive,
        variants=("source_float64_perfect_false", "source_float32_perfect_true"),
        write_json=False,
    )

    assert report["status"] == "blocked_hpm1_probability_range_contract_mismatch"
    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    assert report["failed_variants"] == ["source_float64_perfect_false", "source_float32_perfect_true"]
    assert {row["failure_stage"] for row in report["variant_results"]} == {"hpac_probability_range_decode"}


def test_pr91_preflight_is_local_only_and_blocks_dispatch(tmp_path: Path) -> None:
    archive = _synthetic_hpm1_archive(tmp_path / "archive.zip")

    report = run_pr91_hpm1_preflight(archive, max_frames=1, attempt_reencode=True, write_json=False)

    assert report["status"] == "blocked_hpm1_probability_range_contract_mismatch"
    assert report["score_claim"] is False
    assert report["dispatch_performed"] is False
    assert report["dispatch_unlocked"] is False
    assert report["hpm1_static_contract"]["status"] == "passed"


def test_preflight_pr91_pr92_replay_contracts_accepts_synthetic_pr92_and_blocks_pr91(
    tmp_path: Path,
) -> None:
    module = _load_preflight_script()
    pr91_archive = _synthetic_hpm1_archive(tmp_path / "pr91.zip")
    probability_report = tmp_path / "probability.json"
    probability_report.write_text(
        json.dumps(
            run_pr91_hpm1_probability_variant_matrix(
                pr91_archive,
                variants=("source_float64_perfect_false",),
                write_json=False,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    exact = {
        "avg_segnet_dist": 0.001,
        "avg_posenet_dist": 0.004,
        "archive_size_bytes": 12345,
        "provenance": {
            "archive_sha256": "a" * 64,
            "inflate_runtime_manifest": {
                "runtime_root": "/tmp/replay_runtime/inflate.sh",
                "runtime_tree_sha256": "b" * 64,
            },
        },
    }
    exact["score_recomputed_from_components"] = (
        100 * exact["avg_segnet_dist"]
        + math.sqrt(10 * exact["avg_posenet_dist"])
        + 25 * exact["archive_size_bytes"] / module.SCORE_DENOMINATOR
    )
    exact_json = tmp_path / "contest_auth_eval.adjudicated.json"
    exact_json.write_text(json.dumps(exact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "candidate_archive": {"sha256": "a" * 64, "bytes": 12345},
        "exact_eval_runtime_contract": {"required_inflate_sh": "inflate.sh"},
        "dispatch_readiness": {"ready_for_exact_eval_dispatch": True},
        "randmulti_decoded_row_parity": {"decoded_rows_match": True},
        "non_noop_byte_change": {"changed": True},
        "next_safe_build_command": "build synthetic",
        "next_safe_exact_eval_command_if_rebuilt": "eval synthetic",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    args = module.parse_args(
        [
            "--pr91-archive",
            str(pr91_archive),
            "--pr91-probability-report",
            str(probability_report),
            "--pr92-manifest",
            str(manifest_path),
            "--pr92-exact-json",
            str(exact_json),
            "--output-json",
            str(tmp_path / "out.json"),
            "--ledger-md",
            str(tmp_path / "ledger.md"),
        ]
    )
    report = module.build_report(args)

    assert report["status"] == "passed_pr92_a_plus_plus_pr91_fail_closed"
    assert report["pr91_hpm1"]["dispatch_allowed"] is False
    assert report["pr92_rmb1_stack"]["evidence_grade"] == "A++"


def test_preflight_pr91_pr92_replay_contracts_fails_closed_on_missing_pr92(tmp_path: Path) -> None:
    module = _load_preflight_script()
    pr91_archive = _synthetic_hpm1_archive(tmp_path / "pr91.zip")
    args = module.parse_args(
        [
            "--pr91-archive",
            str(pr91_archive),
            "--rerun-pr91-prefix",
            "--pr92-manifest",
            str(tmp_path / "missing_manifest.json"),
            "--pr92-exact-json",
            str(tmp_path / "missing_exact.json"),
            "--output-json",
            str(tmp_path / "out.json"),
            "--ledger-md",
            str(tmp_path / "ledger.md"),
        ]
    )

    report = module.build_report(args)

    assert report["status"] == "failed_closed"
    assert report["pr91_hpm1"]["status"] == "blocked_hpm1_probability_range_contract_mismatch"
    assert report["pr92_rmb1_stack"]["status"] == "failed_closed_missing_pr92_artifact"
