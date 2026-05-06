"""Tests for Alpha mask primitive component-response plan generation."""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "build_alpha_mask_primitive_response_plan.py"
SPEC = importlib.util.spec_from_file_location("build_alpha_mask_primitive_response_plan", MODULE_PATH)
assert SPEC is not None
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def _tiny_masks() -> np.ndarray:
    return np.array(
        [
            [
                [0, 0, 1, 1, 1, 4],
                [0, 2, 2, 1, 4, 4],
                [3, 3, 2, 4, 4, 4],
                [3, 0, 2, 2, 4, 4],
            ],
            [
                [0, 1, 1, 1, 1, 4],
                [0, 2, 2, 2, 4, 4],
                [3, 3, 2, 4, 4, 0],
                [3, 0, 0, 2, 4, 4],
            ],
            [
                [0, 1, 1, 1, 4, 4],
                [0, 0, 2, 2, 4, 4],
                [3, 3, 3, 4, 4, 0],
                [3, 0, 0, 2, 2, 4],
            ],
        ],
        dtype=np.uint8,
    )


def _write_archive(path: Path, *, unsafe_member: str | None = None) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("renderer.bin", b"ASYM" + bytes(range(4, 80)))
        zf.writestr("masks.mkv", b"baseline-mask-member")
        zf.writestr("optimized_poses.bin", b"pose-bytes")
        if unsafe_member is not None:
            zf.writestr(unsafe_member, b"unsafe")
    return path


def _write_masks_source(path: Path) -> Path:
    np.save(path, _tiny_masks())
    return path


def _install_fake_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_encode_masks_mkv(masks: np.ndarray, output_path: Path, *, crf: int, fps: int) -> dict[str, Any]:
        payload = (
            b"fake-mask-mkv\n"
            + f"crf={crf};fps={fps};shape={tuple(masks.shape)}\n".encode("ascii")
            + planner._validate_masks(masks).tobytes()
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)
        return {
            "path": output_path,
            "size_bytes": output_path.stat().st_size,
            "sha256": planner._sha256_file(output_path),
            "encoder": "unit_test_fake_encoder",
            "crf": int(crf),
            "fps": int(fps),
        }

    monkeypatch.setattr(planner, "_encode_masks_mkv", fake_encode_masks_mkv)


def _config() -> Any:
    return planner.PlanConfig(
        scan_frame_count=3,
        max_points=32,
        max_component_points=2,
        max_boundary_points=2,
        max_class_flip_points=2,
        max_morph_points=4,
        max_transition_points=2,
        max_components_per_class=1,
        max_pixels_per_point=4,
        boundary_width=1,
        mask_crf=17,
        mask_fps=20,
    )


def test_builds_deterministic_non_promotable_alpha_primitive_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_encoder(monkeypatch)
    baseline = _write_archive(tmp_path / "baseline.zip")
    masks_source = _write_masks_source(tmp_path / "decoded_masks.npy")
    output_dir = tmp_path / "plan"

    first = planner.build_alpha_mask_primitive_response_plan(
        baseline_archive=baseline,
        decoded_masks_source=masks_source,
        output_dir=output_dir,
        config=_config(),
        command=["build_alpha_mask_primitive_response_plan.py", "--unit-test"],
    )
    first_plan_text = Path(first["plan"]["path"]).read_text()
    first_variants_text = Path(first["archive_variants_manifest"]["path"]).read_text()

    second = planner.build_alpha_mask_primitive_response_plan(
        baseline_archive=baseline,
        decoded_masks_source=masks_source,
        output_dir=output_dir,
        config=_config(),
        command=["build_alpha_mask_primitive_response_plan.py", "--unit-test"],
        force=True,
    )

    assert Path(second["plan"]["path"]).read_text() == first_plan_text
    assert Path(second["archive_variants_manifest"]["path"]).read_text() == first_variants_text

    plan = json.loads(first_plan_text)
    assert plan["schema_version"] == 1
    assert plan["format"] == "official_component_response_plan_v1"
    assert plan["alpha_plan_format"] == "alpha_mask_primitive_component_response_plan_v1"
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["official_component_response"] is False
    assert plan["scorer_network_loaded"] is False
    assert "contest_auth_eval.py --device cuda" in plan["canonical_score_source_required"]
    assert plan["source"]["baseline_archive"]["bytes"] == baseline.stat().st_size
    assert plan["source"]["decoded_masks"]["shape"] == [3, 4, 6]

    points = plan["points"]
    assert points[0]["epsilon"] == 0.0
    assert "archive" not in points[0]
    assert len(points) == first["point_count"]
    assert first["nonzero_point_count"] > 0

    kinds = {point["primitive"]["kind"] for point in points[1:]}
    assert {
        "connected_component",
        "boundary_band",
        "class_flip",
        "transition_endpoint",
    } <= kinds
    assert {"morphology_erode", "morphology_dilate"} & kinds

    for point in points[1:]:
        archive = point["archive"]
        assert not Path(archive).is_absolute()
        assert ".." not in Path(archive).parts
        assert point["score_claim"] is False
        assert point["promotion_eligible"] is False
        assert 0.0 <= point["selection_weight"] <= 1.0
        archive_path = output_dir / archive
        assert archive_path.exists()

    first_archive = output_dir / points[1]["archive"]
    with zipfile.ZipFile(first_archive, "r") as zf:
        assert set(zf.namelist()) == {"renderer.bin", "masks.mkv", "optimized_poses.bin"}
        info = zf.getinfo("masks.mkv")
        assert list(info.date_time) == [1980, 1, 1, 0, 0, 0]
        assert zf.read("masks.mkv").startswith(b"fake-mask-mkv\n")


def test_rejects_unsafe_archive_members_even_with_decoded_mask_source(tmp_path: Path) -> None:
    baseline = _write_archive(tmp_path / "unsafe.zip", unsafe_member="../escape")
    masks_source = _write_masks_source(tmp_path / "decoded_masks.npy")

    with pytest.raises(planner.AlphaPrimitivePlanError, match="unsafe archive member path"):
        planner.build_alpha_mask_primitive_response_plan(
            baseline_archive=baseline,
            decoded_masks_source=masks_source,
            output_dir=tmp_path / "plan",
            config=_config(),
            command=["build_alpha_mask_primitive_response_plan.py", "--unit-test"],
        )


def test_rejects_hidden_archive_sidecars(tmp_path: Path) -> None:
    baseline = _write_archive(tmp_path / "hidden.zip", unsafe_member=".DS_Store")
    masks_source = _write_masks_source(tmp_path / "decoded_masks.npy")

    with pytest.raises(planner.AlphaPrimitivePlanError, match="hidden archive sidecar"):
        planner.build_alpha_mask_primitive_response_plan(
            baseline_archive=baseline,
            decoded_masks_source=masks_source,
            output_dir=tmp_path / "plan",
            config=_config(),
            command=["build_alpha_mask_primitive_response_plan.py", "--unit-test"],
        )


def test_plan_path_and_weight_validators_fail_closed() -> None:
    with pytest.raises(planner.AlphaPrimitivePlanError, match="must be relative"):
        planner._validate_plan_relative_path("/tmp/archive.zip", field="points[1].archive")
    with pytest.raises(planner.AlphaPrimitivePlanError, match="must not contain traversal"):
        planner._validate_plan_relative_path("../archive.zip", field="points[1].archive")
    with pytest.raises(planner.AlphaPrimitivePlanError, match="hidden path rejected"):
        planner._validate_plan_relative_path("archives/.hidden.zip", field="points[1].archive")
    with pytest.raises(planner.AlphaPrimitivePlanError, match="must be finite"):
        planner._finite_weight(float("nan"), field="selection_weight")
    with pytest.raises(planner.AlphaPrimitivePlanError, match="must be in \\[0, 1\\]"):
        planner._finite_weight(2.0, field="selection_weight")


def test_cli_defaults_are_bounded() -> None:
    parser = planner._build_arg_parser()
    args = parser.parse_args([])

    assert args.scan_frame_count == planner.PlanConfig.scan_frame_count
    assert args.max_points == planner.PlanConfig.max_points
    assert args.max_pixels_per_point == planner.PlanConfig.max_pixels_per_point
    assert args.mask_member == "masks.mkv"
