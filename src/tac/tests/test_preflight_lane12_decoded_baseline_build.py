"""Tests for Lane 12 decoded-baseline build preflight."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT_PATH = REPO_ROOT / "experiments" / "preflight_lane12_decoded_baseline_build.py"
TRAIN_PATH = REPO_ROOT / "experiments" / "train_nerv_mask.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preflight = _load_module("preflight_lane12_decoded_baseline_build", PREFLIGHT_PATH)
train_nerv = _load_module("train_nerv_mask_for_preflight_tests", TRAIN_PATH)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _write_contract(
    path: Path,
    *,
    masks: torch.Tensor,
    source_path: Path,
    decoded_sha256: str | None = None,
) -> Path:
    shape = [int(v) for v in masks.shape]
    payload = {
        "schema_version": 1,
        "diagnostic": "alpha_geo_primitive_contract_v1",
        "score_evidence_grade": "empirical",
        "promotion_eligible": False,
        "score_claim_eligible": False,
        "exact_eval_claim": False,
        "source": {
            "baseline": {
                "path": str(source_path),
                "archive_sha256": train_nerv._sha256_file(source_path),
                "archive_size_bytes": int(source_path.stat().st_size),
                "decoded_mask_sha256": decoded_sha256
                or train_nerv._mask_tensor_sha256(masks),
                "decoded_mask_sha256_algo": "sha256(shape,dtype,contiguous-raw-bytes)",
                "decoded_mask_shape": shape,
                "decoded_mask_dtype": str(train_nerv._normalize_mask_tensor(masks, name="test").dtype),
                "promotion_eligible": False,
                "score_claim_eligible": False,
                "exact_eval_claim": False,
            },
            "failed_candidate": {
                "provenance_only": True,
                "promotion_eligible": False,
                "score_claim_eligible": False,
                "exact_eval_claim": False,
            },
        },
        "shape": {"frames": shape[0], "height": shape[1], "width": shape[2]},
        "protected_classes": [{"class_id": 1}, {"class_id": 2}],
        "ranked_critical_boxes": [],
        "decoded_baseline_boundary_recipes": [],
        "worst_transition_pairs": [],
        "threshold_gates": {
            "exploratory_retrain_gate": {
                "passed": True,
                "blockers": [],
                "observed": {},
                "thresholds": {},
            },
            "exact_eval_spend_gate": {
                "passed": False,
                "blockers": ["build-only preflight"],
                "observed": {},
                "thresholds": {},
            },
        },
    }
    return _write_json(path, payload)


def test_train_nerv_mask_hash_matches_alpha_geo_uint8_contract() -> None:
    masks = torch.zeros(2, 4, 5, dtype=torch.long)
    masks[1, :, 2:] = 2

    canonical = train_nerv._normalize_mask_tensor(masks, name="hash_test")
    digest = hashlib.sha256()
    digest.update(str(tuple(canonical.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(canonical.contiguous().numpy()))

    assert canonical.dtype == torch.uint8
    assert train_nerv._mask_tensor_sha256(masks) == digest.hexdigest()


def test_decoded_baseline_preflight_passes_contract_but_not_l2_unblock(
    tmp_path: Path,
) -> None:
    masks = torch.zeros(2, 4, 5, dtype=torch.long)
    masks[1, :, 2:] = 1
    baseline_path = tmp_path / "baseline_masks.pt"
    contract_path = tmp_path / "alpha_contract.json"
    torch.save(masks, baseline_path)
    _write_contract(contract_path, masks=masks, source_path=baseline_path)

    report = preflight.preflight_lane12_decoded_baseline_build(
        repo_root=REPO_ROOT,
        decoded_baseline_path=baseline_path,
        decoded_baseline_member="masks.mkv",
        alpha_primitive_contract=contract_path,
        clearance_json=tmp_path / ".omx" / "state" / "lane12_nerv_l2_clearance.json",
        output_json=None,
        expected_shape=[2, 4, 5],
        use_default_artifact_globs=False,
    )

    assert report["training_performed"] is False
    assert report["remote_job_launched"] is False
    assert report["clearance_state_written"] is False
    assert report["decoded_baseline_contract_preflight_passed"] is True
    assert report["ready_for_build_only_remote_training"] is False
    assert report["runtime_closure"]["passed"] is True
    assert report["runtime_closure"]["checks"][
        "unpack_renderer_payload:nrv_qzs3_qp1_parser"
    ]["passed"] is True
    assert report["runtime_closure"]["checks"][
        "inflate_renderer:archive_default_mask_source"
    ]["passed"] is True
    assert report["alpha_primitive_contract"]["consumption_gates"]["overall_passed"] is True
    assert report["decoded_baseline"]["target_mask_sha256"] == train_nerv._mask_tensor_sha256(masks)
    assert any(
        "missing or unreadable Lane 12 L2 clearance packet" in blocker
        for blocker in report["remote_training_blockers"]
    )


def test_decoded_baseline_preflight_rejects_contract_hash_mismatch(
    tmp_path: Path,
) -> None:
    masks = torch.zeros(2, 4, 5, dtype=torch.long)
    baseline_path = tmp_path / "baseline_masks.pt"
    contract_path = tmp_path / "alpha_contract.json"
    torch.save(masks, baseline_path)
    _write_contract(
        contract_path,
        masks=masks,
        source_path=baseline_path,
        decoded_sha256="0" * 64,
    )

    report = preflight.preflight_lane12_decoded_baseline_build(
        repo_root=REPO_ROOT,
        decoded_baseline_path=baseline_path,
        alpha_primitive_contract=contract_path,
        clearance_json=tmp_path / ".omx" / "state" / "lane12_nerv_l2_clearance.json",
        output_json=None,
        expected_shape=[2, 4, 5],
        use_default_artifact_globs=False,
    )

    assert report["decoded_baseline_contract_preflight_passed"] is False
    assert report["ready_for_build_only_remote_training"] is False
    assert any(
        "decoded_mask_sha256_match" in blocker
        for blocker in report["alpha_primitive_contract"]["blockers"]
    )


def test_runtime_closure_preflight_rejects_missing_nerv_qzs3_parser(tmp_path: Path) -> None:
    root = tmp_path
    unpacker = root / "submissions" / "robust_current" / "unpack_renderer_payload.py"
    inflate_renderer = root / "submissions" / "robust_current" / "inflate_renderer.py"
    inflate_sh = root / "submissions" / "robust_current" / "inflate.sh"
    unpacker.parent.mkdir(parents=True)
    unpacker.write_text("NERV_MAGIC = b\"NRV1\"\n")
    inflate_renderer.write_text(
        'mask_source = os.environ.get("INFLATE_MASK_SOURCE", "archive")\n'
        "def _load_masks_from_nrv(): pass\n"
        'x = archive / "masks.nrv"\n'
        '"INFLATE_MASK_SOURCE=archive"\n'
        'inflate_tto = os.environ.get("INFLATE_TTO", "0") == "1"\n'
        '"[strict-scorer-rule]"\n'
    )
    inflate_sh.write_text("unpack_renderer_payload.py renderer_payload_unpack_summary.json\n")

    report = preflight._runtime_closure_report(root)

    assert report["passed"] is False
    assert "unpack_renderer_payload:nrv_qzs3_qp1_parser" in report["blockers"]
    assert "unpack_renderer_payload:qzs3_renderer_content_validation" in report["blockers"]
