from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "check_pr101_proxy_promotion_blocker.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("check_pr101_proxy_promotion_blocker", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest() -> dict[str, object]:
    archive_sha = "a" * 64
    return {
        "schema": "pr101_kaggle_proxy_runtime_packet_v1",
        "candidate_id": "proxy_cmaes_0037",
        "score_claim": False,
        "score_claim_valid": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "exact_auth_eval_performed": False,
        "contest_cuda_auth_eval": False,
        "scorers_invoked": False,
        "archive_changed": False,
        "archive_unchanged_sha256": archive_sha,
        "source_archive": {"sha256": archive_sha, "bytes": 178258},
        "packet_archive": {"sha256": archive_sha, "bytes": 178258},
        "runtime_patch": {
            "patched_file": "inflate.py",
            "runtime_consumed_params": [
                {"param": "bias_r", "slot": "up[:, 0, 0]", "value": -1.01},
                {"param": "bias_b", "slot": "up[:, 0, 2]", "value": -0.79},
                {"param": "bias_g", "slot": "up[:, 1, 1]", "value": -0.88},
            ],
        },
        "blockers": [
            "proxy_substrate_not_contest_exact_eval",
            "no_contest_cuda_auth_eval",
            "full_inflate_runtime_not_executed_by_this_probe",
            "active_level2_lane_dispatch_claim_required_before_exact_eval",
        ],
    }


def _proof() -> dict[str, object]:
    archive_sha = "a" * 64
    return {
        "schema": "pr101_kaggle_proxy_runtime_consumption_proof_v1",
        "candidate_id": "proxy_cmaes_0037",
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "archive_unchanged_proof": {
            "archive_sha256": archive_sha,
            "manifest_archive_sha256": archive_sha,
        },
        "supported_bias_params_static_patch_proven": True,
        "inflate_sh_routes_to_packet_inflate_py": True,
        "runtime_consumption_proven_for_supported_bias_params": False,
        "dispatch_blockers": [
            "proxy_substrate_not_contest_exact_eval",
            "no_contest_cuda_auth_eval",
            "full_inflate_runtime_not_executed_by_this_probe",
            "active_level2_lane_dispatch_claim_required_before_exact_eval",
        ],
    }


def _a1_auth_eval() -> dict[str, object]:
    return {
        "evidence_grade": "A++",
        "lane_tag": "[contest-CUDA]",
        "score_axis": "contest_cuda",
        "score_claim_valid": True,
        "promotion_eligible": True,
        "n_samples": 600,
        "score_recomputed_from_components": 0.2263520234784395,
        "archive_size_bytes": 178262,
        "provenance": {
            "archive_sha256": "8" * 64,
            "device": "cuda",
            "gpu_t4_match": True,
        },
    }


def _xray() -> dict[str, object]:
    return {
        "schema_version": "xray_inflate_op_cost_profiler_v1",
        "files": [
            {
                "label": "pr101_hnerv_ft_microcodec",
                "per_channel_mutation_count": 3,
                "per_channel_mutations": [{"line": 49}, {"line": 50}, {"line": 51}],
            }
        ],
    }


def _write_inputs(tmp_path: Path, *, manifest: dict[str, object] | None = None, proof: dict[str, object] | None = None) -> dict[str, Path]:
    paths = {
        "manifest": tmp_path / "runtime_packet_manifest.json",
        "proof": tmp_path / "runtime_consumption_proof.json",
        "a1": tmp_path / "contest_auth_eval.json",
        "xray": tmp_path / "op_catalog.json",
    }
    _write_json(paths["manifest"], manifest or _manifest())
    _write_json(paths["proof"], proof or _proof())
    _write_json(paths["a1"], _a1_auth_eval())
    _write_json(paths["xray"], _xray())
    return paths


def test_blocks_proxy_packet_without_full_runtime_consumption_or_exact_cuda(tmp_path: Path) -> None:
    tool = _load_tool()
    paths = _write_inputs(tmp_path)

    checklist = tool.build_promotion_blocker_checklist(
        manifest_path=paths["manifest"],
        proof_path=paths["proof"],
        a1_auth_eval_path=paths["a1"],
        xray_path=paths["xray"],
    )

    assert checklist["promotable"] is False
    assert checklist["verdict"] == "BLOCKED_PROXY_ONLY_NOT_PROMOTABLE"
    assert "full_runtime_consumption_not_proven" in checklist["blockers"]
    assert "no_candidate_contest_cuda_auth_eval" in checklist["blockers"]
    assert "stale_unsupported_proxy_contract" not in checklist["blockers"]
    assert any(row["id"] == "a1_exact_cuda_anchor_available" and row["passed"] for row in checklist["checks"])
    assert any(row["id"] == "xray_op_cost_catalog_available" and row["passed"] for row in checklist["checks"])


def test_blocks_stale_unsupported_proxy_contract(tmp_path: Path) -> None:
    tool = _load_tool()
    manifest = _manifest()
    manifest["unsupported_params"] = {"smooth_weight": {"runtime_consumed": False}}
    manifest["blockers"] = list(manifest["blockers"]) + [
        "unsupported_proxy_params_not_runtime_consumed",
        "smooth_weight_not_runtime_consumed",
    ]
    proof = _proof()
    proof["unsupported_params_blocker_proof"] = {"unsupported_params": {"smooth_weight": {}}}
    paths = _write_inputs(tmp_path, manifest=manifest, proof=proof)

    checklist = tool.build_promotion_blocker_checklist(
        manifest_path=paths["manifest"],
        proof_path=paths["proof"],
        a1_auth_eval_path=paths["a1"],
        xray_path=paths["xray"],
    )

    assert checklist["promotable"] is False
    assert "stale_unsupported_proxy_contract" in checklist["blockers"]


def test_flags_proxy_authority_true_as_blocker(tmp_path: Path) -> None:
    tool = _load_tool()
    manifest = _manifest()
    manifest["ready_for_exact_eval_dispatch"] = True
    paths = _write_inputs(tmp_path, manifest=manifest)

    checklist = tool.build_promotion_blocker_checklist(
        manifest_path=paths["manifest"],
        proof_path=paths["proof"],
        a1_auth_eval_path=paths["a1"],
        xray_path=paths["xray"],
    )

    assert checklist["promotable"] is False
    assert "proxy_manifest_authority_flag_true:ready_for_exact_eval_dispatch" in checklist["blockers"]


def test_cli_exits_nonzero_by_default_and_can_write_blocked_checklist(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    output = tmp_path / "checklist.json"
    base_cmd = [
        sys.executable,
        str(TOOL_PATH),
        "--manifest",
        str(paths["manifest"]),
        "--proof",
        str(paths["proof"]),
        "--a1-auth-eval",
        str(paths["a1"]),
        "--xray-op-catalog",
        str(paths["xray"]),
        "--output",
        str(output),
    ]

    blocked = subprocess.run(base_cmd, check=False, capture_output=True, text=True)
    assert blocked.returncode == 1
    assert json.loads(blocked.stdout)["promotable"] is False
    assert output.is_file()

    allowed = subprocess.run(base_cmd + ["--allow-blocked"], check=False, capture_output=True, text=True)
    assert allowed.returncode == 0
    assert json.loads(allowed.stdout)["verdict"] == "BLOCKED_PROXY_ONLY_NOT_PROMOTABLE"
