# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.xray_mlx_prefilter_error_anatomy import (
    build_report_from_profile,
    write_report_outputs,
)


def _artifact(path: Path) -> dict[str, str]:
    return {"path": path.as_posix()}


def test_xray_mlx_prefilter_error_anatomy_reports_pair_and_pixel_tails(
    tmp_path: Path,
) -> None:
    components = tmp_path / "components"
    candidate = tmp_path / "candidate_cache"
    reference = tmp_path / "reference_cache"
    components.mkdir()
    candidate.mkdir()
    reference.mkdir()

    np.save(components / "posenet_distortion.npy", np.array([0.01, 0.09], dtype=np.float32))
    np.save(components / "segnet_distortion.npy", np.array([0.02, 0.01], dtype=np.float32))
    pair_indices = np.array([[0, 1], [2, 3]], dtype=np.int64)
    np.save(candidate / "pair_indices.npy", pair_indices)
    np.save(reference / "pair_indices.npy", pair_indices)
    np.save(candidate / "segnet_last_rgb.npy", np.ones((2, 2, 2, 3), dtype=np.float32))
    np.save(reference / "segnet_last_rgb.npy", np.zeros((2, 2, 2, 3), dtype=np.float32))
    np.save(candidate / "posenet_yuv6_pair.npy", np.ones((2, 2, 2, 12), dtype=np.float32))
    np.save(reference / "posenet_yuv6_pair.npy", np.zeros((2, 2, 2, 12), dtype=np.float32))
    profile = {
        "schema": "mlx_scorer_response.v1",
        "components": {
            "artifacts": {
                "posenet_distortion": _artifact(components / "posenet_distortion.npy"),
                "segnet_distortion": _artifact(components / "segnet_distortion.npy"),
            }
        },
        "cache_identity": {
            "candidate": {
                "path": candidate.as_posix(),
                "artifacts": {
                    "pair_indices": _artifact(candidate / "pair_indices.npy"),
                    "segnet_last_rgb": _artifact(candidate / "segnet_last_rgb.npy"),
                    "posenet_yuv6_pair": _artifact(candidate / "posenet_yuv6_pair.npy"),
                },
            },
            "reference": {
                "path": reference.as_posix(),
                "artifacts": {
                    "pair_indices": _artifact(reference / "pair_indices.npy"),
                    "segnet_last_rgb": _artifact(reference / "segnet_last_rgb.npy"),
                    "posenet_yuv6_pair": _artifact(reference / "posenet_yuv6_pair.npy"),
                },
            },
            "pair_indices_equal": True,
        },
        "archive_size_bytes": 123,
        "archive_sha256": "a" * 64,
    }
    profile_path = tmp_path / "local_mlx_prefilter_profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    report = build_report_from_profile(
        mlx_profile=profile_path,
        report_path=tmp_path / "out" / "mlx_prefilter_error_anatomy.json",
    )

    assert report["schema"] == "mlx_prefilter_error_anatomy.v1"
    assert report["score_claim"] is False
    assert report["component_summary"]["n_pairs"] == 2
    assert report["pixel_summary"]["computed"] is True
    assert report["top_pairs"]["combined"][0]["pair_idx"] == 0
    assert report["top_pairs"]["pose"][0]["pair_idx"] == 1
    assert report["top_pairs"]["segnet_cache_delta"][0]["segnet_cache_mean_abs_delta"] == 1.0
    work_order = report["direct_full_scorer_vjp_work_order"]
    assert work_order["ready_for_vjp_materialization"] is True
    assert "--seg-ce-weight 0" in work_order["next_commands"]["mlx_pose_p19_full_video"]
    assert "--device-type auto" in work_order["next_commands"]["mlx_pose_p19_full_video"]
    assert "--backend torch" in work_order["next_commands"]["torch_joint_p18_p19_full_video"]

    outputs = write_report_outputs(report, tmp_path / "out")
    assert outputs["json"].is_file()
    assert outputs["jsonl"].is_file()
    assert outputs["markdown"].is_file()
    written = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert str(outputs["json"]) in written["direct_full_scorer_vjp_work_order"]["next_command"]


def test_xray_mlx_prefilter_error_anatomy_merges_rich_prefilter_manifest(
    tmp_path: Path,
) -> None:
    components = tmp_path / "components"
    candidate = tmp_path / "candidate_cache"
    reference = tmp_path / "reference_cache"
    components.mkdir()
    candidate.mkdir()
    reference.mkdir()

    np.save(components / "posenet_distortion.npy", np.array([1.0, 4.0], dtype=np.float32))
    np.save(components / "segnet_distortion.npy", np.array([0.1, 0.2], dtype=np.float32))
    pair_indices = np.array([[0, 1], [2, 3]], dtype=np.int64)
    for root, value in ((candidate, 2.0), (reference, 0.0)):
        np.save(root / "pair_indices.npy", pair_indices)
        np.save(root / "segnet_last_rgb.npy", np.full((2, 3, 2, 2), value, dtype=np.float32))
        np.save(root / "posenet_yuv6_pair.npy", np.full((2, 12, 2, 2), value, dtype=np.float32))

    rich_candidate = {
        "path": candidate.as_posix(),
        "artifacts": {
            "pair_indices": _artifact(candidate / "pair_indices.npy"),
            "segnet_last_rgb": _artifact(candidate / "segnet_last_rgb.npy"),
            "posenet_yuv6_pair": _artifact(candidate / "posenet_yuv6_pair.npy"),
        },
    }
    rich_reference = {
        "path": reference.as_posix(),
        "artifacts": {
            "pair_indices": _artifact(reference / "pair_indices.npy"),
            "segnet_last_rgb": _artifact(reference / "segnet_last_rgb.npy"),
            "posenet_yuv6_pair": _artifact(reference / "posenet_yuv6_pair.npy"),
        },
    }
    profile = {
        "schema": "mlx_scorer_response.v1",
        "components": {
            "artifacts": {
                "posenet_distortion": _artifact(components / "posenet_distortion.npy"),
                "segnet_distortion": _artifact(components / "segnet_distortion.npy"),
            }
        },
        "cache_identity": {
            "candidate": {"path": candidate.as_posix(), "pair_count": 2},
            "reference": {"path": reference.as_posix(), "pair_count": 2},
            "pair_indices_equal": True,
        },
        "hinerv_receiver_raw_cache_prefilter": {
            "candidate_cache_manifest": rich_candidate,
            "reference_cache_manifest": rich_reference,
        },
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    report = build_report_from_profile(
        mlx_profile=profile_path,
        report_path=tmp_path / "out" / "mlx_prefilter_error_anatomy.json",
    )

    assert report["pixel_summary"]["computed"] is True
    assert report["rows"][0]["segnet_cache_mean_abs_delta"] == 2.0
    assert report["direct_full_scorer_vjp_work_order"]["ready_for_vjp_materialization"] is True


def test_xray_mlx_prefilter_error_anatomy_accepts_explicit_hot_cache_overrides(
    tmp_path: Path,
) -> None:
    components = tmp_path / "components"
    cleaned_candidate = tmp_path / "cleaned_candidate"
    cleaned_reference = tmp_path / "cleaned_reference"
    hot_candidate = tmp_path / "hot_candidate"
    hot_reference = tmp_path / "hot_reference"
    for root in (components, cleaned_candidate, cleaned_reference, hot_candidate, hot_reference):
        root.mkdir()

    np.save(components / "posenet_distortion.npy", np.array([1.0], dtype=np.float32))
    np.save(components / "segnet_distortion.npy", np.array([0.1], dtype=np.float32))
    pair_indices = np.array([[0, 1]], dtype=np.int64)
    for root, value in ((hot_candidate, 7.0), (hot_reference, 5.0)):
        np.save(root / "pair_indices.npy", pair_indices)
        np.save(root / "segnet_last_rgb.npy", np.full((1, 3, 2, 2), value, dtype=np.float32))
        np.save(root / "posenet_yuv6_pair.npy", np.full((1, 12, 2, 2), value, dtype=np.float32))
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "path": root.as_posix(),
                    "pair_count": 1,
                    "pair_indices_shape": [1, 2],
                    "segnet_last_rgb_shape": [1, 3, 2, 2],
                    "posenet_yuv6_pair_shape": [1, 12, 2, 2],
                    "artifacts": {
                        "pair_indices": _artifact(root / "pair_indices.npy"),
                        "segnet_last_rgb": _artifact(root / "segnet_last_rgb.npy"),
                        "posenet_yuv6_pair": _artifact(root / "posenet_yuv6_pair.npy"),
                    },
                    "array_sha256": {},
                }
            ),
            encoding="utf-8",
        )
    profile = {
        "schema": "mlx_scorer_response.v1",
        "components": {
            "artifacts": {
                "posenet_distortion": _artifact(components / "posenet_distortion.npy"),
                "segnet_distortion": _artifact(components / "segnet_distortion.npy"),
            }
        },
        "cache_identity": {
            "candidate": {"path": cleaned_candidate.as_posix(), "pair_count": 1},
            "reference": {"path": cleaned_reference.as_posix(), "pair_count": 1},
            "pair_indices_equal": True,
        },
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    report = build_report_from_profile(
        mlx_profile=profile_path,
        candidate_cache_dir=hot_candidate,
        reference_cache_dir=hot_reference,
        report_path=tmp_path / "out" / "mlx_prefilter_error_anatomy.json",
    )

    assert report["pixel_summary"]["computed"] is True
    assert report["rows"][0]["segnet_cache_mean_abs_delta"] == 2.0
    assert report["hot_cache_recovery"]["records"][0]["action"] == "explicit_hot_cache_override"
    assert report["direct_full_scorer_vjp_work_order"]["ready_for_vjp_materialization"] is True
