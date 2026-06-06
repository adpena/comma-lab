# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tac.training.long_training_canonical import (
    CANONICAL_SEGNET_ARGMAX_MIN_OCCUPIED_CLASS_FRACTION_FOR_FIT_GATE,
    CANONICAL_SEGNET_TARGET_CLASS_COVERAGE_FRACTION_FOR_FIT_GATE,
    CANONICAL_SEGNET_TARGET_CLASS_MIN_RATIO_FOR_FIT_GATE,
    CurriculumStage,
    LongTrainingConfig,
    PolyakEMAShadow,
    _archive_selection_health_sort_key,
    _export_live_ema_archive_selection,
)


def _row(
    occupied_fraction: float,
    *,
    target_coverage: float | None = None,
    target_min_ratio: float | None = None,
) -> dict[str, object]:
    score_components: dict[str, float] = {
        "selection_health_segnet_direct_live_candidate_occupied_class_fraction": (
            occupied_fraction
        )
    }
    if target_coverage is not None:
        score_components[
            "selection_health_segnet_direct_live_candidate_target_class_coverage_fraction"
        ] = target_coverage
    if target_min_ratio is not None:
        score_components[
            "selection_health_segnet_direct_live_candidate_target_class_min_ratio"
        ] = target_min_ratio
    return {
        "score_components": score_components,
    }


def test_archive_selection_treats_two_of_five_segnet_classes_as_collapsed() -> None:
    collapsed = _archive_selection_health_sort_key(
        _row(CANONICAL_SEGNET_ARGMAX_MIN_OCCUPIED_CLASS_FRACTION_FOR_FIT_GATE - 1e-6)
    )
    healthy = _archive_selection_health_sort_key(
        _row(CANONICAL_SEGNET_ARGMAX_MIN_OCCUPIED_CLASS_FRACTION_FOR_FIT_GATE)
    )

    assert collapsed[0] == 1
    assert healthy[0] == 0
    assert healthy < collapsed


def test_archive_selection_prefers_target_class_coverage_over_generic_occupancy() -> None:
    missing_target_classes = _archive_selection_health_sort_key(
        _row(
            1.0,
            target_coverage=(
                CANONICAL_SEGNET_TARGET_CLASS_COVERAGE_FRACTION_FOR_FIT_GATE - 1e-6
            ),
        )
    )
    target_classes_preserved = _archive_selection_health_sort_key(
        _row(
            CANONICAL_SEGNET_ARGMAX_MIN_OCCUPIED_CLASS_FRACTION_FOR_FIT_GATE,
            target_coverage=CANONICAL_SEGNET_TARGET_CLASS_COVERAGE_FRACTION_FOR_FIT_GATE,
        )
    )

    assert missing_target_classes[0] == 2
    assert target_classes_preserved[0] == 0
    assert target_classes_preserved < missing_target_classes


def test_archive_selection_uses_target_min_ratio_as_secondary_tiebreaker() -> None:
    weak_material_coverage = _archive_selection_health_sort_key(
        _row(
            0.8,
            target_coverage=1.0,
            target_min_ratio=CANONICAL_SEGNET_TARGET_CLASS_MIN_RATIO_FOR_FIT_GATE,
        )
    )
    stronger_material_coverage = _archive_selection_health_sort_key(
        _row(0.8, target_coverage=1.0, target_min_ratio=0.7)
    )

    assert stronger_material_coverage < weak_material_coverage


def test_archive_selection_treats_low_target_min_ratio_as_collapsed() -> None:
    target_mass_collapsed = _archive_selection_health_sort_key(
        _row(
            1.0,
            target_coverage=1.0,
            target_min_ratio=(
                CANONICAL_SEGNET_TARGET_CLASS_MIN_RATIO_FOR_FIT_GATE - 1e-6
            ),
        )
    )
    target_mass_preserved = _archive_selection_health_sort_key(
        _row(
            CANONICAL_SEGNET_ARGMAX_MIN_OCCUPIED_CLASS_FRACTION_FOR_FIT_GATE,
            target_coverage=CANONICAL_SEGNET_TARGET_CLASS_COVERAGE_FRACTION_FOR_FIT_GATE,
            target_min_ratio=CANONICAL_SEGNET_TARGET_CLASS_MIN_RATIO_FOR_FIT_GATE,
        )
    )

    assert target_mass_collapsed[0] == 2
    assert target_mass_preserved[0] == 0
    assert target_mass_preserved < target_mass_collapsed


class _ArchiveSelectionUnitModel:
    def __init__(self) -> None:
        self.value = 1.0

    def state_dict(self) -> dict[str, list[float]]:
        return {"value": [float(self.value)]}

    def load_state_dict(self, state: dict[str, list[float]]) -> None:
        self.value = float(state["value"][0])


class _ArchiveSelectionFailingAdapter:
    substrate_id = "unit_archive_selection"

    def __init__(self) -> None:
        self.model = _ArchiveSelectionUnitModel()

    def sample_batch(self, _batch_size: int, _seed: int) -> dict[str, int]:
        return {"batch": 1}

    def loss_fn(self, *_args: object, **_kwargs: object) -> dict[str, float]:
        return {"total": 0.0}

    def optimizer_step(self, *_args: object, **_kwargs: object) -> None:
        return None

    def export_state_dict(self, _model: object, path: Path) -> None:
        path.write_text("{}", encoding="utf-8")

    def score_aware_components(
        self,
        _model: object,
        _batch: object,
    ) -> dict[str, float]:
        return {"d_seg_proxy": 0.25}

    def export_archive(
        self,
        _model: object,
        output_dir: Path,
    ) -> tuple[Path, str, int] | None:
        output_dir.mkdir(parents=True, exist_ok=True)
        archive = output_dir / "archive.zip"
        archive.write_bytes(b"written-before-hard-cap-failure")
        raise ValueError("HiNeRV archive exceeds hard_byte_ceiling: 32 > 16")


class _ArchiveSelectionTargetCoverageAdapter:
    substrate_id = "unit_archive_selection"

    def __init__(self) -> None:
        self.model = _ArchiveSelectionUnitModel()

    def sample_batch(self, _batch_size: int, _seed: int) -> dict[str, int]:
        return {"batch": 1}

    def loss_fn(self, *_args: object, **_kwargs: object) -> dict[str, float]:
        return {"total": 0.0}

    def optimizer_step(self, *_args: object, **_kwargs: object) -> None:
        return None

    def export_state_dict(self, _model: object, path: Path) -> None:
        path.write_text("{}", encoding="utf-8")

    def score_aware_components(
        self,
        model: _ArchiveSelectionUnitModel,
        _batch: object,
    ) -> dict[str, float]:
        # Live has a cheaper local proxy; EMA has healthier target-class
        # coverage. Archive selection must prefer scorer-health over this
        # advisory proxy when target classes are missing.
        return {"d_seg_proxy": 0.0 if model.value == 2.0 else 1.0}

    def archive_selection_health(
        self,
        model: _ArchiveSelectionUnitModel,
        _batch: object,
    ) -> dict[str, float]:
        if model.value == 2.0:
            return {
                "segnet_direct_live_candidate_occupied_class_fraction": 1.0,
                "segnet_direct_live_candidate_target_class_coverage_fraction": 0.6,
                "segnet_direct_live_candidate_target_class_min_ratio": 0.2,
            }
        return {
            "segnet_direct_live_candidate_occupied_class_fraction": 0.4,
            "segnet_direct_live_candidate_target_class_coverage_fraction": 1.0,
            "segnet_direct_live_candidate_target_class_min_ratio": 0.7,
        }

    def export_archive(
        self,
        model: _ArchiveSelectionUnitModel,
        output_dir: Path,
    ) -> tuple[Path, str, int] | None:
        output_dir.mkdir(parents=True, exist_ok=True)
        archive = output_dir / "archive.zip"
        payload = f"value={model.value}".encode()
        archive.write_bytes(payload)
        return archive, hashlib.sha256(payload).hexdigest(), len(payload)


def test_archive_selection_failure_preserves_written_archive_evidence(
    tmp_path: Path,
) -> None:
    adapter = _ArchiveSelectionFailingAdapter()
    config = LongTrainingConfig(
        substrate_id="unit_archive_selection",
        lane_id="lane_unit_archive_selection",
        epochs=1,
        batch_pair_indices_per_step=1,
        curriculum_stages=(
            CurriculumStage(
                name="unit",
                start_epoch=0,
                end_epoch=1,
                loss_weights={"recon": 1.0},
            ),
        ),
        learning_rate=1e-3,
        seed=7,
        output_dir=tmp_path,
        device="cpu",
        notes="Unit test for preserving emitted archive evidence after selector failure.",
    )
    ema = PolyakEMAShadow(adapter.model, decay=0.5)

    archive_path, archive_sha256, archive_bytes, manifest_path = (
        _export_live_ema_archive_selection(
            adapter=adapter,
            config=config,
            ema_shadow=ema,
        )
    )

    assert archive_path is None
    assert archive_sha256 is None
    assert archive_bytes is None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["selected_archive_path"] is None
    assert manifest["exported_candidate_count"] == 0
    assert {row["candidate_kind"] for row in manifest["rows"]} == {"live", "ema"}
    for row in manifest["rows"]:
        archive = Path(row["archive_path"])
        assert row["status"] == "failed"
        assert row["emitted_archive_available"] is True
        assert row["archive_bytes"] == archive.stat().st_size
        assert row["archive_sha256"]
        assert archive.is_file()


def test_archive_selection_exports_target_coverage_health_before_proxy_score(
    tmp_path: Path,
) -> None:
    adapter = _ArchiveSelectionTargetCoverageAdapter()
    ema = PolyakEMAShadow(adapter.model, decay=0.5)
    adapter.model.value = 2.0
    config = LongTrainingConfig(
        substrate_id="unit_archive_selection",
        lane_id="lane_unit_archive_selection",
        epochs=1,
        batch_pair_indices_per_step=1,
        curriculum_stages=(
            CurriculumStage(
                name="unit",
                start_epoch=0,
                end_epoch=1,
                loss_weights={"recon": 1.0},
            ),
        ),
        learning_rate=1e-3,
        seed=7,
        output_dir=tmp_path,
        device="cpu",
        notes="Unit test for target-class health in live-vs-EMA selection.",
        ema_archive_selection_enabled=True,
    )

    archive_path, archive_sha256, archive_bytes, manifest_path = (
        _export_live_ema_archive_selection(
            adapter=adapter,
            config=config,
            ema_shadow=ema,
        )
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["selected_candidate_kind"] == "ema"
    assert archive_path == Path(manifest["selected_archive_path"])
    assert archive_sha256 == manifest["selected_archive_sha256"]
    assert archive_bytes == manifest["selected_archive_bytes"]
    rows = {row["candidate_kind"]: row for row in manifest["rows"]}
    assert rows["live"]["proxy_score"] < rows["ema"]["proxy_score"]
    assert rows["live"]["score_components"][
        "selection_health_segnet_direct_live_candidate_target_class_coverage_fraction"
    ] == pytest.approx(0.6)
    assert rows["ema"]["score_components"][
        "selection_health_segnet_direct_live_candidate_target_class_coverage_fraction"
    ] == pytest.approx(1.0)
