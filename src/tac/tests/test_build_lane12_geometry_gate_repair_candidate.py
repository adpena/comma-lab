from __future__ import annotations

import importlib.util
import json
import lzma
import sys
import zipfile
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "build_lane12_geometry_gate_repair_candidate.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_lane12_geometry_gate_repair_candidate", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _runtime_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("renderer.bin", b"R" * 10_001)
        zf.writestr("masks.nrv", b"NRV1" + b"N" * 512)
        zf.writestr("optimized_poses.bin", b"P" * 2_000)


def _plan(tmp_path: Path, *, baseline_path: Path, candidate_path: Path) -> Path:
    return _write_json(
        tmp_path / "plan.json",
        {
            "schema": "lane12_geometry_gate_repair_atom_plan_v1",
            "score_claim": False,
            "promotion_eligible": False,
            "inputs": {
                "tensor_status": {
                    "loaded": True,
                    "baseline": {"tensor_path": str(baseline_path)},
                    "candidate": {"tensor_path": str(candidate_path)},
                }
            },
            "atoms": [
                {
                    "atom_id": "box_a",
                    "atom_kind": "residual_region_patch",
                    "identity": {"frame": 0, "box_xyxy": [1, 1, 5, 3]},
                },
                {
                    "atom_id": "pair_b",
                    "atom_kind": "transition_pair_focus",
                    "identity": {"pair_index": 1, "frames": [2, 3]},
                },
            ],
            "candidate_policies": [
                {
                    "policy_id": "box_policy",
                    "selected_atom_ids": ["box_a"],
                    "score_claim": False,
                    "promotion_eligible": False,
                },
                {
                    "policy_id": "pair_policy",
                    "selected_atom_ids": ["box_a", "pair_b"],
                    "score_claim": False,
                    "promotion_eligible": False,
                },
            ],
        },
    )


def test_lane12_geometry_repair_builder_emits_cdo1_over_nrv_archive(tmp_path: Path) -> None:
    builder = _load_builder()
    baseline = torch.zeros((4, 4, 8), dtype=torch.uint8)
    candidate = baseline.clone()
    baseline[0, 1, 1:5] = 3
    candidate[0, 1, 1:5] = 0
    baseline[2, 0, 0:2] = 1
    candidate[2, 0, 0:2] = 0
    baseline_path = tmp_path / "baseline.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save(baseline, baseline_path)
    torch.save(candidate, candidate_path)
    plan_json = _plan(tmp_path, baseline_path=baseline_path, candidate_path=candidate_path)
    base_archive = tmp_path / "base.zip"
    _runtime_archive(base_archive)

    report = builder.build_lane12_geometry_gate_repair_candidate(
        plan_json=plan_json,
        policy_id="box_policy",
        base_archive=base_archive,
        output_dir=tmp_path / "out",
        overlay_compressor="lzma_xz",
        max_residual_disagreement_fraction=0.001,
        repo_root=REPO_ROOT,
    )

    archive = Path(report["output_archive"]["path"])
    assert archive.is_file()
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["exact_eval_claim"] is False
    assert report["geometry_metrics"]["selected_repair_pixels"] == 4
    assert report["geometry_metrics"]["repaired_disagreement_pixels"] == 4
    assert report["geometry_metrics"]["residual_disagreement_pixels_after"] == 2
    assert report["dispatch_allowed"] is False
    with zipfile.ZipFile(archive, "r") as zf:
        assert zf.namelist() == [
            "renderer.bin",
            "masks.nrv",
            "masks.cdo1.xz",
            "optimized_poses.bin",
        ]
        raw = lzma.decompress(zf.read("masks.cdo1.xz"), format=lzma.FORMAT_XZ)
    assert raw == (tmp_path / "out" / "box_policy" / "masks.cdo1").read_bytes()
    assert report["overlay_payload"]["header"]["base_mask_tensor_sha256"] == builder._mask_tensor_sha256(
        candidate.numpy()
    )
    assert report["overlay_payload"]["header"]["target_mask_tensor_sha256"] == builder._mask_tensor_sha256(
        baseline.numpy()
    )


def test_lane12_geometry_repair_builder_can_pass_local_residual_gate(tmp_path: Path) -> None:
    builder = _load_builder()
    baseline = torch.zeros((4, 4, 8), dtype=torch.uint8)
    candidate = baseline.clone()
    baseline[0, 1, 1:5] = 3
    candidate[0, 1, 1:5] = 0
    baseline[2, 0, 0:2] = 1
    candidate[2, 0, 0:2] = 0
    baseline_path = tmp_path / "baseline.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save(baseline, baseline_path)
    torch.save(candidate, candidate_path)
    plan_json = _plan(tmp_path, baseline_path=baseline_path, candidate_path=candidate_path)
    base_archive = tmp_path / "base.zip"
    _runtime_archive(base_archive)

    report = builder.build_lane12_geometry_gate_repair_candidate(
        plan_json=plan_json,
        policy_id="pair_policy",
        base_archive=base_archive,
        output_dir=tmp_path / "out",
        overlay_compressor="zlib",
        max_residual_disagreement_fraction=0.0,
        repo_root=REPO_ROOT,
    )

    assert report["geometry_metrics"]["residual_disagreement_pixels_after"] == 0
    assert report["dispatch_allowed"] is True
    assert report["dispatch_gate"]["exact_cuda_auth_eval_still_required"] is True


def test_lane12_geometry_repair_builder_rejects_promotable_input_plan(tmp_path: Path) -> None:
    builder = _load_builder()
    baseline = torch.zeros((4, 4, 8), dtype=torch.uint8)
    path = tmp_path / "mask.pt"
    torch.save(baseline, path)
    plan_json = _plan(tmp_path, baseline_path=path, candidate_path=path)
    payload = json.loads(plan_json.read_text())
    payload["promotion_eligible"] = True
    plan_json.write_text(json.dumps(payload))
    base_archive = tmp_path / "base.zip"
    _runtime_archive(base_archive)

    try:
        builder.build_lane12_geometry_gate_repair_candidate(
            plan_json=plan_json,
            policy_id="box_policy",
            base_archive=base_archive,
            output_dir=tmp_path / "out",
            repo_root=REPO_ROOT,
        )
    except builder.Lane12GeometryRepairBuildError as exc:
        assert "non-promotable" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected non-promotable plan guard")
