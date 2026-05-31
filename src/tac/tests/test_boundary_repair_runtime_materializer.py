from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from tac.optimization.archive_bound_candidate_contract import (
    archive_bound_candidate_contracts_from_payload,
)
from tac.optimization.boundary_repair_runtime_materializer import (
    BOUNDARY_REPAIR_RUNTIME_MATERIALIZER_SCHEMA,
    materialize_boundary_repair_runtime_candidate,
)


def _write_base_submission(root: Path, raw_bytes: bytes) -> Path:
    submission = root / "base_submission"
    submission.mkdir()
    (submission / "inflate.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes())\n",
        encoding="utf-8",
    )
    (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    with zipfile.ZipFile(submission / "archive.zip", "w") as zf:
        zf.writestr("x", raw_bytes)
    return submission


def _write_bridge_and_surface(root: Path) -> tuple[Path, Path]:
    surface = root / "surface.npz"
    source_argmax = np.zeros((2, 2, 2), dtype=np.uint8)
    candidate_argmax = source_argmax.copy()
    candidate_argmax[0, 0, 0] = 1
    candidate_argmax[1, 1, 1] = 1
    wrong = (source_argmax != candidate_argmax).astype(np.uint8)
    boundary = np.ones_like(wrong, dtype=np.uint8)
    hinge = np.zeros((2, 2, 2), dtype=np.float32)
    hinge[0, 0, 0] = 5.0
    hinge[1, 1, 1] = 3.0
    np.savez_compressed(
        surface,
        source_argmax=source_argmax,
        candidate_argmax=candidate_argmax,
        source_top2=np.ones_like(source_argmax),
        source_margin=np.ones((2, 2, 2), dtype=np.float32),
        candidate_margin=np.ones((2, 2, 2), dtype=np.float32),
        boundary_mask=boundary,
        wrong_mask=wrong,
        hinge_map=hinge,
        sample_ids=np.asarray([0, 1], dtype=np.int64),
    )
    bridge = root / "bridge.json"
    bridge.write_text(
        json.dumps(
            {
                "schema": "segnet_semantic_bridge.v1",
                "candidate_id": "tiny",
                "semantic_surface_artifacts": {
                    "argmax_margin_boundary_npz": {"path": str(surface)}
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return bridge, surface


def test_source_pixel_patch_candidate_is_byte_closed_and_receiver_proven(tmp_path: Path) -> None:
    raw_shape = (4, 4, 4, 3)
    grid_shape = (2, 2)
    base_raw = np.zeros(raw_shape, dtype=np.uint8)
    source_raw = base_raw.copy()
    source_raw[1, 0:2, 0:2, :] = 99
    source_path = tmp_path / "source.raw"
    source_path.write_bytes(source_raw.tobytes())
    base = _write_base_submission(tmp_path, base_raw.tobytes())
    bridge, surface = _write_bridge_and_surface(tmp_path)

    manifest = materialize_boundary_repair_runtime_candidate(
        bridge_path=bridge,
        surface_path=surface,
        base_submission_dir=base,
        output_dir=tmp_path / "candidate",
        repo_root=tmp_path,
        strategy="source_pixel_patch",
        candidate_id="tiny_patch",
        source_raw_path=source_path,
        video_name="0.mkv",
        raw_shape=raw_shape,
        grid_shape=grid_shape,
        max_grid_pixels=1,
        max_raw_points=16,
        expected_receiver_output_bytes=base_raw.nbytes,
        retain_receiver_output=True,
    )

    assert manifest["schema"] == BOUNDARY_REPAIR_RUNTIME_MATERIALIZER_SCHEMA
    assert manifest["receiver_contract_satisfied"] is True
    assert manifest["runtime_consumption_proof_ready"] is True
    assert manifest["byte_closed_candidate_materialized"] is True
    package = manifest["archive_bound_candidate_adapter_package"]
    contracts = archive_bound_candidate_contracts_from_payload(package)
    assert contracts[0]["archive_bound_candidate_ready"] is True
    assert contracts[0]["receiver_contract_satisfied"] is True
    with zipfile.ZipFile(tmp_path / "candidate" / "submission" / "archive.zip") as zf:
        assert zf.namelist() == ["boundary_repair_overlay.json", "x"]
        overlay = json.loads(zf.read("boundary_repair_overlay.json"))
    assert overlay["strategy"] == "source_pixel_patch"
    assert overlay["point_summary"]["selected_raw_points"] == 4
    out_raw = np.memmap(
        tmp_path / "candidate" / "receiver_proof" / "runtime_out" / "0.raw",
        dtype=np.uint8,
        mode="r",
        shape=raw_shape,
    )
    assert int(out_raw[1, 0, 0, 0]) == 99


def test_masked_local_median_candidate_uses_runtime_overlay_without_source_pixels(
    tmp_path: Path,
) -> None:
    raw_shape = (4, 4, 4, 3)
    grid_shape = (2, 2)
    base_raw = np.zeros(raw_shape, dtype=np.uint8)
    base_raw[1, 0:3, 0:3, :] = 10
    base_raw[1, 1, 1, :] = 90
    base = _write_base_submission(tmp_path, base_raw.tobytes())
    bridge, surface = _write_bridge_and_surface(tmp_path)

    manifest = materialize_boundary_repair_runtime_candidate(
        bridge_path=bridge,
        surface_path=surface,
        base_submission_dir=base,
        output_dir=tmp_path / "candidate_postfilter",
        repo_root=tmp_path,
        strategy="masked_local_median",
        candidate_id="tiny_postfilter",
        video_name="0.mkv",
        raw_shape=raw_shape,
        grid_shape=grid_shape,
        max_grid_pixels=1,
        max_raw_points=16,
        expected_receiver_output_bytes=base_raw.nbytes,
        retain_receiver_output=True,
    )

    assert manifest["receiver_contract_satisfied"] is True
    overlay = json.loads(
        (
            tmp_path
            / "candidate_postfilter"
            / "archive_dir"
            / "boundary_repair_overlay.json"
        ).read_text(encoding="utf-8")
    )
    assert overlay["strategy"] == "masked_local_median"
    assert "rgb" not in overlay
    contracts = archive_bound_candidate_contracts_from_payload(
        manifest["archive_bound_candidate_adapter_package"]
    )
    assert contracts[0]["archive_bound_candidate_ready"] is True
