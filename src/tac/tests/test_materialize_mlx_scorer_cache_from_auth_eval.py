# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES, contest_formula_score
from tac.repo_io import write_json

REPO = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO / "tools" / "materialize_mlx_scorer_cache_from_auth_eval.py"
HASH_DOMAIN = "_array_sha256(dtype_string + json_shape + contiguous_bytes)"
ARRAY_HASHES = {
    "segnet_last_rgb": "s" * 64,
    "posenet_yuv6_pair": "p" * 64,
    "pair_indices": "i" * 64,
}


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "materialize_mlx_scorer_cache_from_auth_eval_under_test",
        TOOL_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_manifest_payload(cache_dir: Path, *, include_score_claim_valid: bool) -> dict:
    artifacts = {}
    for name in ("pair_indices", "posenet_yuv6_pair", "segnet_last_rgb"):
        path = cache_dir / f"{name}.npy"
        path.write_bytes(f"{name}-bytes".encode())
        artifacts[name] = {
            "path": f"/__modal/volumes/example/run/scorer_input_cache_tensors/{path.name}",
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    payload = {
        "schema_version": "mlx_scorer_input_cache.v1",
        "archive_sha256": "a" * 64,
        "inflated_outputs_aggregate_sha256": "b" * 64,
        "raw_sha256": "r" * 64,
        "hash_domain": HASH_DOMAIN,
        "pair_count": 600,
        "segnet_last_rgb_shape": [600, 3, 384, 512],
        "posenet_yuv6_pair_shape": [600, 12, 192, 256],
        "pair_indices_shape": [600, 2],
        "array_sha256": dict(ARRAY_HASHES),
        "artifacts": artifacts,
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if include_score_claim_valid:
        payload["score_claim_valid"] = False
    return payload


def _auth_eval_with_tensor_manifest(payload: dict, *, manifest_sha256: str) -> dict:
    archive_size = 178_517
    seg = 0.00056029
    pose = 0.00002943
    return {
        "canonical_score": contest_formula_score(
            seg_dist=seg,
            pose_dist=pose,
            archive_bytes=archive_size,
        ),
        "canonical_score_source": "score_recomputed_from_components",
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "archive_size_bytes": archive_size,
        "score_rate_contribution": 25.0 * archive_size / ORIGINAL_VIDEO_BYTES,
        "rate_unscaled": archive_size / ORIGINAL_VIDEO_BYTES,
        "n_samples": 600,
        "evidence_grade": "contest-CPU",
        "lane_tag": "[contest-CPU]",
        "score_axis": "contest_cpu",
        "evidence_semantics": "public_leaderboard_cpu_reproduction",
        "exact_cuda_eval_complete": False,
        "score_claim": True,
        "score_claim_valid": True,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "cpu_leaderboard_reproduction_eligible": True,
        "provenance": {
            "device": "cpu",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "archive_sha256": "a" * 64,
            "inflated_output_manifest": {
                "payload": {
                    "aggregate_sha256": "b" * 64,
                    "files": [{"sha256": "r" * 64}],
                }
            },
            "scorer_input_cache_tensor_manifest": {
                "payload": payload,
                "sha256": manifest_sha256,
            },
        },
    }


def test_runtime_custody_accepts_matching_content_tree(monkeypatch, tmp_path: Path) -> None:
    tool = _load_tool()
    inflate_sh = tmp_path / "inflate.sh"
    inflate_sh.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(tool, "_sha256", lambda path, prefix=0: "s" * 64)
    monkeypatch.setattr(
        tool,
        "_runtime_dependency_manifest",
        lambda inflate, upstream: {
            "runtime_tree_sha256": "l" * 64,
            "runtime_content_tree_sha256": "c" * 64,
            "runtime_file_count": 3,
        },
    )

    result = tool._validate_runtime_custody(
        {
            "provenance": {
                "inflate_script_sha256": "s" * 64,
                "inflate_runtime_manifest": {
                    "runtime_tree_sha256": "r" * 64,
                    "runtime_content_tree_sha256": "c" * 64,
                },
            }
        },
        {"expected_runtime_tree_sha256": "r" * 64},
        inflate_sh=inflate_sh,
        upstream_dir=tmp_path,
    )

    assert result["comparison"] == "runtime_content_tree_sha256"
    assert result["local_runtime_content_tree_sha256"] == "c" * 64


def test_runtime_custody_rejects_content_tree_mismatch(monkeypatch, tmp_path: Path) -> None:
    tool = _load_tool()
    inflate_sh = tmp_path / "inflate.sh"
    inflate_sh.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(tool, "_sha256", lambda path, prefix=0: "s" * 64)
    monkeypatch.setattr(
        tool,
        "_runtime_dependency_manifest",
        lambda inflate, upstream: {
            "runtime_tree_sha256": "l" * 64,
            "runtime_content_tree_sha256": "d" * 64,
        },
    )

    with pytest.raises(SystemExit, match="runtime content-tree SHA mismatch"):
        tool._validate_runtime_custody(
            {
                "provenance": {
                    "inflate_script_sha256": "s" * 64,
                    "inflate_runtime_manifest": {
                        "runtime_tree_sha256": "r" * 64,
                        "runtime_content_tree_sha256": "c" * 64,
                    },
                }
            },
            {},
            inflate_sh=inflate_sh,
            upstream_dir=tmp_path,
        )


def test_inflate_policy_refuses_nondefault_replay() -> None:
    tool = _load_tool()

    with pytest.raises(SystemExit, match="non-default inflate_device_policy"):
        tool._validate_default_inflate_policy(
            {"provenance": {"inflate_device_policy": "cpu"}}
        )


def test_failed_audit_deletes_materialized_cache(tmp_path: Path) -> None:
    tool = _load_tool()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text("{}", encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({"passed": False}), encoding="utf-8")

    manifest = tool._finalize_cache_after_audit(
        output_cache=cache_dir,
        audit_output=audit_path,
        audit={"passed": False, "verdict": "FAIL_CACHE_AUTH_EVAL_IDENTITY"},
        auth_dir=tmp_path / "auth",
        auth_eval_path=tmp_path / "auth" / "contest_auth_eval.json",
    )

    assert manifest == {}
    assert not cache_dir.exists()


def test_passing_audit_stamps_materialized_cache_manifest(monkeypatch, tmp_path: Path) -> None:
    tool = _load_tool()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "mlx_scorer_input_cache.v1"}),
        encoding="utf-8",
    )
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({"passed": True}), encoding="utf-8")
    monkeypatch.setattr(tool, "_sha256", lambda path, prefix=0: "a" * 64)

    manifest = tool._finalize_cache_after_audit(
        output_cache=cache_dir,
        audit_output=audit_path,
        audit={
            "schema_version": "mlx_scorer_input_cache_auth_eval_audit.v1",
            "passed": True,
            "verdict": "PASS_CACHE_AUTH_EVAL_IDENTITY",
            "identity_residual": 0,
        },
        auth_dir=tmp_path / "auth",
        auth_eval_path=tmp_path / "auth" / "contest_auth_eval.json",
    )

    assert manifest["eligible_for_local_mlx_transfer_calibration"] is True
    assert manifest["auth_eval_identity_audit"]["verdict"] == "PASS_CACHE_AUTH_EVAL_IDENTITY"
    assert manifest["auth_eval_identity_audit"]["sha256"] == "a" * 64
    persisted = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    assert persisted == manifest


def test_downloaded_tensor_cache_mode_stamps_without_reinflate(monkeypatch, tmp_path: Path) -> None:
    tool = _load_tool()
    source_cache = tmp_path / "downloaded" / "scorer_input_cache_tensors"
    source_cache.mkdir(parents=True)
    payload = _tensor_manifest_payload(source_cache, include_score_claim_valid=False)
    write_json(source_cache / "manifest.json", payload)
    manifest_sha = _file_sha256(source_cache / "manifest.json")
    volume_manifest = {
        "schema_version": "modal_auth_eval_tensor_volume_manifest.v1",
        "volume_name": "comma-auth-eval-cache-artifacts",
        "volume_run_id": "unit",
        "volume_path": "/modal_auth_cache/unit/scorer_input_cache_tensors",
        "manifest_path": "/modal_auth_cache/unit/scorer_input_cache_tensors/manifest.json",
        "manifest_sha256": manifest_sha,
        "tensor_payload_returned_via_modal_artifacts": False,
        "payload": payload,
    }
    auth_dir = tmp_path / "auth"
    auth_dir.mkdir()
    write_json(
        auth_dir / "contest_auth_eval.json",
        _auth_eval_with_tensor_manifest(payload, manifest_sha256=manifest_sha),
    )
    volume_path = auth_dir / "scorer_input_cache_tensor_volume_manifest.json"
    write_json(volume_path, volume_manifest)
    monkeypatch.setattr(
        tool,
        "_run_inflate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no re-inflate expected")),
    )

    out_dir = tmp_path / "materialized"
    audit_path = tmp_path / "audit.json"
    rc = tool.main(
        [
            "--auth-eval-dir",
            str(auth_dir),
            "--downloaded-tensor-cache-dir",
            str(source_cache),
            "--tensor-volume-manifest",
            str(volume_path),
            "--output-cache-dir",
            str(out_dir),
            "--audit-output",
            str(audit_path),
        ]
    )

    assert rc == 0
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["passed"] is True
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["score_claim_valid"] is False
    assert manifest["eligible_for_local_mlx_transfer_calibration"] is True
    assert manifest["auth_eval_identity_audit"]["passed"] is True
    identity = manifest["downloaded_tensor_cache_identity"]
    assert identity["manifest_sha256_exact_match"] is True
    assert identity["authority_contract_completed_fields"] == ["score_claim_valid"]


def test_downloaded_tensor_cache_mode_rejects_artifact_sha_mismatch(tmp_path: Path) -> None:
    tool = _load_tool()
    source_cache = tmp_path / "downloaded"
    source_cache.mkdir()
    payload = _tensor_manifest_payload(source_cache, include_score_claim_valid=True)
    write_json(source_cache / "manifest.json", payload)
    manifest_sha = _file_sha256(source_cache / "manifest.json")
    original_size = (source_cache / "segnet_last_rgb.npy").stat().st_size
    (source_cache / "segnet_last_rgb.npy").write_bytes(b"x" * original_size)

    with pytest.raises(SystemExit, match="tensor artifact SHA mismatch for segnet_last_rgb"):
        tool._validate_downloaded_tensor_cache(
            downloaded_cache_dir=source_cache,
            tensor_volume_manifest={
                "schema_version": "modal_auth_eval_tensor_volume_manifest.v1",
                "manifest_sha256": manifest_sha,
                "payload": payload,
            },
            auth_eval=_auth_eval_with_tensor_manifest(payload, manifest_sha256=manifest_sha),
        )


def test_downloaded_tensor_cache_mode_rejects_source_authority_claim(tmp_path: Path) -> None:
    tool = _load_tool()
    source_cache = tmp_path / "downloaded"
    source_cache.mkdir()
    payload = _tensor_manifest_payload(source_cache, include_score_claim_valid=True)
    payload["score_claim"] = True
    write_json(source_cache / "manifest.json", payload)
    manifest_sha = _file_sha256(source_cache / "manifest.json")

    with pytest.raises(SystemExit, match="authority field score_claim must be false"):
        tool._validate_downloaded_tensor_cache(
            downloaded_cache_dir=source_cache,
            tensor_volume_manifest={
                "schema_version": "modal_auth_eval_tensor_volume_manifest.v1",
                "manifest_sha256": manifest_sha,
                "payload": payload,
            },
            auth_eval=_auth_eval_with_tensor_manifest(payload, manifest_sha256=manifest_sha),
        )
