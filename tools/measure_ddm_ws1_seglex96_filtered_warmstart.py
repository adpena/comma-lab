#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Freshly measure the exact 96-move Seg-filtered WS1 warm start.

This closes the source-replay debt recorded by the WS1 precursor: the V19C
receiver is rebuilt from only strict negative Seg-term admissions, then the
temporal amplitude rung is measured both unmasked and decoder-hood-masked.
The runner is local-only and has no training or dispatch surface.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_hood_static_reassert import (  # noqa: E402
    derive_hood_supports,
    expand_support_to_camera,
    reassert_frame1,
)
from tac.optimization.ddm_realized_flip_menu import (  # noqa: E402
    CAMERA_HW,
    EVIDENCE_AXIS,
    SEG_HW,
    apply_temporal_affine,
    sha256_bytes,
)
from tac.optimization.direct_description_preuint8_channel import (  # noqa: E402
    receive_preuint8_q8_archive,
)
from tools.measure_ddm_mc1_hood_static_reassert import _load_all_base_cells  # noqa: E402
from tools.measure_ddm_menu1_realized_flip_menu import (  # noqa: E402
    _config_and_inputs,
    _forward,
    _geometry_statistics_camera,
    _load_models,
    _palette,
    _publish_json,
    _publish_npz,
    _read,
    _semantic_cells,
    _sha256_file,
)
from tools.measure_ddm_ws1_seg_lexicographic_warmstart import (  # noqa: E402
    DELEGATION_KEY,
    LANE_ID,
    POINTER,
    WJOINT_ID,
    WS1Error,
    _aggregate,
    _fresh_row,
    _source_rows,
)

CONFIG_SCHEMA = "DDMWS1SegLex96FilteredWarmStartConfigV1"
SCHEMA = "ddm_ws1_seglex96_filtered_warmstart_measurement.v1"
BASE_ID = "v19c_seglex96_base"
UNMASKED_ID = "temporal_affine_16knot_frame1_seglex96_unmasked"
MASKED_ID = "temporal_affine_16knot_frame1_seglex96_hood_masked"


class FilteredConfig(BaseModel):
    """SHA-bound filtered-receiver measurement contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    schema_: Literal["DDMWS1SegLex96FilteredWarmStartConfigV1"] = Field(
        default=CONFIG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    run_id: str
    seed: Literal[210] = 210
    pair_count: Literal[600] = 600
    scorer_batch_size: Literal[16] = 16
    scorer_threads: Literal[4] = 4
    ws1_precursor_receipt_path: str
    ws1_precursor_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    menu1_config_path: str
    menu1_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    filtered_archive_path: str
    filtered_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    filtered_archive_bytes: Literal[137827] = 137_827
    checkpoint_root: str
    mycar_material_defect_ceiling: Literal[40000] = 40_000
    pointer: Literal["0.1910828242 [contest-CPU]"] = POINTER
    execution_allowed: Literal[True] = True
    paid_dispatch_allowed: Literal[False] = False
    exact_eval_allowed: Literal[False] = False
    frontier_mutation_allowed: Literal[False] = False
    training_allowed: Literal[False] = False
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _sealed(self) -> FilteredConfig:
        for value in (self.filtered_archive_path, self.checkpoint_root):
            if not Path(value).is_absolute() or not value.startswith(
                "/Volumes/VertigoDataTier/pact/"
            ):
                raise ValueError("filtered artifacts must use the primary SSD tier")
        return self

    def stable_hash(self) -> str:
        return sha256_bytes(
            json.dumps(
                self.model_dump(mode="json", by_alias=True),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _paths(root: Path, candidate: str, start: int, stop: int) -> tuple[Path, Path]:
    stage = root / candidate
    return (
        stage / f"batch_{start:04d}_{stop:04d}.json",
        stage / f"batch_{start:04d}_{stop:04d}.npz",
    )


def measure(config_path: Path) -> dict[str, Any]:
    config = FilteredConfig.model_validate_json(_read(config_path))
    root = Path(config.checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 512 << 20:
        raise WS1Error("filtered WS1 SSD preflight failed")
    precursor_path = _resolve(config.ws1_precursor_receipt_path)
    if _sha256_file(precursor_path) != config.ws1_precursor_receipt_sha256:
        raise WS1Error("WS1 precursor receipt SHA-256 differs")
    precursor = json.loads(_read(precursor_path))
    lex = precursor["seg_lexicographic_rerank"]
    if (
        lex["v19c_seg_strict_accepted_count"] != 96
        or lex["v19c_seg_non_strict_rejected_count"] != 8
    ):
        raise WS1Error("Seg re-rank cardinality differs")
    menu_path = _resolve(config.menu1_config_path)
    if _sha256_file(menu_path) != config.menu1_config_sha256:
        raise WS1Error("MENU1 config SHA-256 differs")
    menu, inputs = _config_and_inputs(menu_path)
    filtered_path = Path(config.filtered_archive_path)
    if (
        _sha256_file(filtered_path) != config.filtered_archive_sha256
        or filtered_path.stat().st_size != config.filtered_archive_bytes
    ):
        raise WS1Error("filtered receiver archive custody differs")
    filtered_receiver = receive_preuint8_q8_archive(_read(filtered_path))
    original_receiver = receive_preuint8_q8_archive(inputs["archive"])
    filtered_palette = _palette(filtered_receiver)
    original_palette = _palette(original_receiver)
    source_root = Path(menu.checkpoint_root)
    temporal_payload = _read(source_root / "01c_temporal_affine_payload.bin")
    statistics_payload = _read(source_root / "01_local_statistics_payload.bin")
    if len(temporal_payload) != 204:
        raise WS1Error("temporal payload bytes differ")
    supports = derive_hood_supports(
        _load_all_base_cells(source_root, batch_size=config.scorer_batch_size)
    )
    labels = open_stored_npy_memmap(Path(menu.target_cache_path), "lstars")
    poses = open_stored_npy_memmap(Path(menu.target_cache_path), "gt_poses")
    if labels.shape != (600, *SEG_HW) or poses.shape != (600, 6):
        raise WS1Error("target geometry differs")
    source_joint_rows = _source_rows(source_root, WJOINT_ID)

    missing = any(
        not all(path.exists() for path in _paths(root, candidate, start, stop))
        for candidate in (BASE_ID, UNMASKED_ID, MASKED_ID)
        for start, stop in (
            (value, min(value + config.scorer_batch_size, config.pair_count))
            for value in range(0, config.pair_count, config.scorer_batch_size)
        )
    )
    segnet = posenet = scorer_custody = None
    if missing:
        segnet, posenet, scorer_custody = _load_models(menu)
    rows: dict[str, list[dict[str, Any]]] = {
        BASE_ID: [],
        UNMASKED_ID: [],
        MASKED_ID: [],
    }
    replay = {}
    support_pixels = 0
    for start in range(0, config.pair_count, config.scorer_batch_size):
        stop = min(start + config.scorer_batch_size, config.pair_count)
        ids = tuple(range(start, stop))
        print(f"[WS1-F96] batch {start:04d}:{stop:04d}", flush=True)
        camera = filtered_receiver.render_camera_pairs(ids)
        semantic, owned = _semantic_cells(
            filtered_receiver, ids, camera, filtered_palette
        )
        camera_support = expand_support_to_camera(
            (semantic == supports.hood_class) & owned,
            batch_size=stop - start,
            camera_hw=CAMERA_HW,
        )
        support_pixels += int(np.count_nonzero(camera_support))
        unmasked = apply_temporal_affine(
            camera,
            pair_ids=ids,
            pair_count=config.pair_count,
            payload=temporal_payload,
        )
        masked = reassert_frame1(
            winner_camera=unmasked,
            base_camera=camera,
            camera_support=camera_support,
        )
        cameras = {BASE_ID: camera, UNMASKED_ID: unmasked, MASKED_ID: masked}
        original_camera = original_receiver.render_camera_pairs(ids)
        original_semantic, original_owned = _semantic_cells(
            original_receiver, ids, original_camera, original_palette
        )
        joint_camera = _geometry_statistics_camera(
            base_camera=original_camera,
            semantic=original_semantic,
            owned=original_owned,
            palette=original_palette,
            statistics_payload=statistics_payload,
        )
        joint_source = source_joint_rows[start // config.scorer_batch_size]
        if joint_source["camera_sha256"] != sha256_bytes(joint_camera.tobytes()):
            raise WS1Error("W_joint fresh composition hash differs")

        original_arrays = (
            source_root
            / "02_measurements"
            / "v19c_base"
            / f"batch_{start:04d}_{stop:04d}.npz"
        )
        for candidate in (BASE_ID, UNMASKED_ID, MASKED_ID):
            row_path, array_path = _paths(root, candidate, start, stop)
            if row_path.exists() and array_path.exists():
                row = json.loads(_read(row_path))
                if (
                    row["typed_config_sha256"] != config.stable_hash()
                    or row["camera_sha256"]
                    != sha256_bytes(cameras[candidate].tobytes())
                ):
                    raise WS1Error("filtered checkpoint identity differs")
                rows[candidate].append(row)
                continue
            if segnet is None or posenet is None:
                raise WS1Error("filtered scorer models are absent")
            cells, pose6 = _forward(segnet, posenet, cameras[candidate])
            if start == 0:
                replay_cells, replay_pose6 = _forward(
                    segnet, posenet, cameras[candidate]
                )
                if not np.array_equal(cells, replay_cells) or not np.array_equal(
                    pose6, replay_pose6
                ):
                    raise WS1Error(
                        f"filtered {candidate} deterministic replay differs"
                    )
            parent_arrays = (
                original_arrays
                if candidate == BASE_ID
                else _paths(root, BASE_ID, start, stop)[1]
            )
            row = _fresh_row(
                config=config,  # type: ignore[arg-type]
                candidate_id=candidate,
                start=start,
                stop=stop,
                camera=cameras[candidate],
                cells=cells,
                pose6=pose6,
                labels=labels,
                poses=poses,
                base_arrays=parent_arrays,
                payload_bytes=0 if candidate == BASE_ID else len(temporal_payload),
                support_pixels=int(np.count_nonzero(camera_support)),
            )
            _publish_npz(array_path, cells=cells, pose6=pose6)
            _publish_json(row_path, row)
            rows[candidate].append(row)
        if start == 0:
            if segnet is None or posenet is None:
                segnet, posenet, scorer_custody = _load_models(menu)
            cells, pose6 = _forward(segnet, posenet, joint_camera)
            source_arrays = (
                source_root
                / "02_measurements"
                / WJOINT_ID
                / f"batch_{start:04d}_{stop:04d}.npz"
            )
            with np.load(source_arrays, allow_pickle=False) as stored:
                if not np.array_equal(cells, stored["cells"]) or not np.array_equal(
                    pose6, stored["pose6"]
                ):
                    raise WS1Error("W_joint scorer replay differs")
            replay = {
                "pair_range": [start, stop],
                "cells_sha256": sha256_bytes(cells.tobytes()),
                "pose6_sha256": sha256_bytes(pose6.tobytes()),
                "status": "PASS",
            }

    base = _aggregate(
        rows[BASE_ID],
        candidate_id=BASE_ID,
        archive_bytes=config.filtered_archive_bytes,
        payload_bytes=0,
    )
    unmasked = _aggregate(
        rows[UNMASKED_ID],
        candidate_id=UNMASKED_ID,
        archive_bytes=config.filtered_archive_bytes + len(temporal_payload),
        payload_bytes=len(temporal_payload),
    )
    masked = _aggregate(
        rows[MASKED_ID],
        candidate_id=MASKED_ID,
        archive_bytes=config.filtered_archive_bytes + len(temporal_payload),
        payload_bytes=len(temporal_payload),
    )
    wjoint = _aggregate(
        source_joint_rows,
        candidate_id=WJOINT_ID,
        archive_bytes=menu.v19c_archive_bytes + len(statistics_payload),
        payload_bytes=len(statistics_payload),
    )
    if base["per_class"]["MyCar"]["errors"] > config.mycar_material_defect_ceiling:
        raise WS1Error("filtered base materially damaged MyCar")
    if masked["per_class"]["MyCar"]["errors"] > config.mycar_material_defect_ceiling:
        raise WS1Error("filtered hood-masked W_seg materially damaged MyCar")
    if masked["d_seg"] >= base["d_seg"]:
        raise WS1Error("filtered hood-masked temporal rung does not improve d_seg")
    masked["hood_mask_delta_vs_unmasked"] = {
        "delta_errors": masked["errors"] - unmasked["errors"],
        "delta_d_seg": masked["d_seg"] - unmasked["d_seg"],
        "delta_d_pose": masked["d_pose"] - unmasked["d_pose"],
        "delta_mycar_errors": (
            masked["per_class"]["MyCar"]["errors"]
            - unmasked["per_class"]["MyCar"]["errors"]
        ),
    }
    lex["receiver_recompile_status"] = {
        "status": "COMPLETE_EXACT_SOURCE_REPLAY",
        "archive_path": config.filtered_archive_path,
        "archive_bytes": config.filtered_archive_bytes,
        "archive_sha256": config.filtered_archive_sha256,
        "strict_v19c_move_count": 96,
        "settled_v19b_prefix_move_count": 10,
    }
    receipt = {
        "schema": SCHEMA,
        "run_id": config.run_id,
        "typed_config_sha256": config.stable_hash(),
        "lane_id": LANE_ID,
        "delegation_checkpoint_key": DELEGATION_KEY,
        "evidence_axis": EVIDENCE_AXIS,
        "pointer": POINTER,
        "pointer_moved": False,
        "score_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
        "source_precursor": {
            "path": config.ws1_precursor_receipt_path,
            "sha256": config.ws1_precursor_receipt_sha256,
        },
        "seg_lexicographic_rerank": lex,
        "fresh_wjoint_first_batch_replay": replay,
        "scorer_custody": scorer_custody
        or precursor.get("scorer_custody"),
        "hood_support": {
            "hood_class": supports.hood_class,
            "kind": "decoder_derived_per_frame_semantic_owned",
            "counted_bytes": 0,
            "stored_support_reference_bytes": 139,
            "camera_support_pixels_total": support_pixels,
            "frame0_byte_identical_within_each_amplitude_composition": True,
        },
        "controls": {
            "seglex96_base": base,
            "temporal_unmasked": unmasked,
            "temporal_hood_masked": masked,
        },
        "warm_start_candidates": {
            "W_seg": {
                **masked,
                "role": "exact_strict_seg_lexicographic_hood_safe_start",
                "base_receiver_sha256": config.filtered_archive_sha256,
            },
            "W_joint": {
                **wjoint,
                "role": "current_best_joint_S_MENU1_start",
            },
        },
        "verdict": (
            "EXACT_SEGLEX96_WSEG_AND_MENU1_WJOINT_MEASURED; "
            "J5_SLOPE_FALSIFIER_SPEC_ONLY"
        ),
        "verdict_scope": (
            "[macOS-CPU frozen-scorer advisory] exact receiver-through-R "
            "warm-start comparison; no training, contest eval, or promotion"
        ),
        "research_only": True,
        "execution_allowed": True,
        "paid_dispatch_allowed": False,
        "exact_eval_allowed": False,
        "frontier_mutation_allowed": False,
        "training_allowed": False,
    }
    output = (
        REPO_ROOT
        / ".omx/research"
        / config.run_id
        / "ddm_ws1_seglex96_filtered_warmstart_receipt.json"
    )
    _publish_json(output, receipt)
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    measure(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
