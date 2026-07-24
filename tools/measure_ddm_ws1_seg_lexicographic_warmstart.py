#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure hood-safe Seg-lexicographic warm starts against the MENU1 joint arm.

The runner is deliberately local-only.  It re-renders the exact V19C receiver,
applies the two already-counted MENU1 amplitude payloads, masks each payload off
the decoder-derived ego-hood support, and performs a fresh frozen CPU-torch
forward for both hood-safe candidates.  The MENU1 joint arm and the unmasked
comparators are freshly re-composed and their immutable scorer checkpoints are
replayed on the first batch.

No training, paid dispatch, exact contest evaluation, or frontier mutation is
reachable from this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
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
    advisory_objective,
    apply_scalar_affine,
    apply_temporal_affine,
    sha256_bytes,
    transition_counts,
)
from tac.optimization.direct_description_preuint8_channel import (  # noqa: E402
    receive_preuint8_q8_archive,
)
from tools.measure_ddm_mc1_hood_static_reassert import (  # noqa: E402
    _load_all_base_cells,
)
from tools.measure_ddm_menu1_realized_flip_menu import (  # noqa: E402
    CLASS_NAMES,
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

SCHEMA = "ddm_ws1_seg_lexicographic_warmstart_measurement.v1"
CONFIG_SCHEMA = "DDMWS1SegLexicographicWarmStartConfigV1"
LANE_ID = "lane_ddm_ws1_seg_lexicographic_warmstart_20260724"
DELEGATION_KEY = "codex_delegate:ddm_ws1_seg_lexicographic_warmstart:20260724T012924Z"
POINTER = "0.1910828242 [contest-CPU]"
WSEG_IDS = (
    "scalar_gain_bias_12b_frame1_hood_masked",
    "temporal_affine_16knot_frame1_hood_masked",
)
UNMASKED_BY_WSEG = {
    WSEG_IDS[0]: "scalar_gain_bias_12b_frame1",
    WSEG_IDS[1]: "temporal_affine_16knot_frame1",
}
WJOINT_ID = "statistics_hard_analytic_composed_frame1"


class WS1Error(RuntimeError):
    """Fail-closed WS1 custody or measurement error."""


class WS1Config(BaseModel):
    """SHA-bound, non-actuating WS1 measurement contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMWS1SegLexicographicWarmStartConfigV1"] = Field(
        default=CONFIG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    run_id: str = Field(min_length=8)
    seed: Literal[210] = 210
    pair_count: Literal[600] = 600
    scorer_batch_size: Literal[16] = 16
    scorer_threads: Literal[4] = 4
    menu1_config_path: str
    menu1_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    menu1_receipt_path: str
    menu1_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mc1_config_path: str
    mc1_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mc1_receipt_path: str
    mc1_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v19c_decisions_path: str
    v19c_decision_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_root: str
    mycar_control_errors: Literal[37237] = 37_237
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
    def _sealed(self) -> WS1Config:
        root = Path(self.checkpoint_root)
        if not root.is_absolute() or not str(root).startswith(
            "/Volumes/VertigoDataTier/pact/"
        ):
            raise ValueError("checkpoint_root must use the primary SSD tier")
        if self.mycar_material_defect_ceiling <= self.mycar_control_errors:
            raise ValueError("MyCar defect ceiling must exceed the exact control")
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


def _bound_json(path: str, expected: str, label: str) -> dict[str, Any]:
    value = _resolve(path)
    if _sha256_file(value) != expected:
        raise WS1Error(f"{label} SHA-256 differs")
    payload = json.loads(_read(value))
    if not isinstance(payload, dict):
        raise WS1Error(f"{label} must be a JSON object")
    return payload


def _decision_chain(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path.glob("*.json"))
    if not files:
        raise WS1Error("V19C decision inventory is empty")
    for item in files:
        digest.update(item.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(item)))
    return digest.hexdigest()


def _lexicographic_inventory(config: WS1Config) -> dict[str, Any]:
    root = _resolve(config.v19c_decisions_path)
    observed = _decision_chain(root)
    if observed != config.v19c_decision_chain_sha256:
        raise WS1Error("V19C decision-chain SHA-256 differs")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        row = json.loads(_read(path))
        if not row.get("accepted"):
            continue
        seg_term = float(row["joint_incremental_delta"]["seg_term"])
        target = accepted if seg_term < 0.0 else rejected
        target.append(
            {
                "candidate_id": row["proposal"]["candidate_id"],
                "family": row["proposal"]["family"],
                "measured_incremental_seg_term": seg_term,
                "source_decision_path": str(path.relative_to(REPO_ROOT)),
                "source_decision_sha256": _sha256_file(path),
            }
        )
    if len(accepted) != 96 or len(rejected) != 8:
        raise WS1Error("V19C strict Seg re-rank cardinality drifted")
    return {
        "source_decision_chain_sha256": observed,
        "v19b_settled_prefix_move_count": 10,
        "v19b_prefix_rule": "all ten settled V19B moves had strict sequential d_seg decrease",
        "v19c_joint_accepted_count": len(accepted) + len(rejected),
        "v19c_seg_strict_accepted_count": len(accepted),
        "v19c_seg_non_strict_rejected_count": len(rejected),
        "acceptance_rule": "measured_incremental_seg_term_lt_zero_ignore_pose",
        "strict_accepted": accepted,
        "non_strict_rejected": rejected,
        "receiver_recompile_status": (
            "NOT_SUBTRACTED_FROM_SETTLED_ENDPOINT: same-pool nonadditivity forbids "
            "claiming an exact filtered receiver without a full source replay; "
            "the measured W_seg receiver therefore remains the SHA-bound V19C endpoint"
        ),
    }


def _storage_preflight(config: WS1Config) -> dict[str, Any]:
    root = Path(config.checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    required = 512 << 20
    if shutil.disk_usage(root).free < required:
        raise WS1Error("SSD storage preflight failed")
    return {
        "tier": "/Volumes/VertigoDataTier/pact",
        "checkpoint_root": str(root),
        "required_free_bytes": required,
        "status": "PASS",
        "cleanup": "preserve immutable per-batch measurement checkpoints",
    }


def _checkpoint_paths(
    root: Path, candidate_id: str, start: int, stop: int
) -> tuple[Path, Path]:
    stage = root / "02_measurements" / candidate_id
    return (
        stage / f"batch_{start:04d}_{stop:04d}.json",
        stage / f"batch_{start:04d}_{stop:04d}.npz",
    )


def _source_rows(root: Path, candidate_id: str) -> list[dict[str, Any]]:
    stage = root / "02_measurements" / candidate_id
    rows = [json.loads(_read(path)) for path in sorted(stage.glob("batch_*.json"))]
    if len(rows) != 38:
        raise WS1Error(f"{candidate_id} source checkpoint count differs")
    return rows


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
    archive_bytes: int,
    payload_bytes: int,
) -> dict[str, Any]:
    errors = sum(int(row["errors"]) for row in rows)
    sites = sum(int(row["sites"]) for row in rows)
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in rows)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in rows)
    classes = {
        name: {
            "errors": sum(int(row["per_class"][name]["errors"]) for row in rows),
            "sites": sum(int(row["per_class"][name]["sites"]) for row in rows),
        }
        for name in CLASS_NAMES.values()
    }
    for value in classes.values():
        value["d_seg"] = value["errors"] / value["sites"]
    d_seg = errors / sites
    d_pose = pose_sse / pose_coordinates
    return {
        "candidate_id": candidate_id,
        "errors": errors,
        "sites": sites,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "archive_bytes": archive_bytes,
        "delta_payload_bytes": payload_bytes,
        "advisory_objective": advisory_objective(
            errors=errors,
            sites=sites,
            d_pose=d_pose,
            bytes_=archive_bytes,
        ),
        "per_class": classes,
        "batch_count": len(rows),
        "batch_digest_chain_sha256": sha256_bytes(
            b"".join(
                json.dumps(
                    dict(row), sort_keys=True, separators=(",", ":")
                ).encode()
                + b"\n"
                for row in rows
            )
        ),
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
    }


def _fresh_row(
    *,
    config: WS1Config,
    candidate_id: str,
    start: int,
    stop: int,
    camera: np.ndarray,
    cells: np.ndarray,
    pose6: np.ndarray,
    labels: np.ndarray,
    poses: np.ndarray,
    base_arrays: Path,
    payload_bytes: int,
    support_pixels: int,
) -> dict[str, Any]:
    target = np.asarray(labels[start:stop], dtype=np.uint8)
    target_pose = np.asarray(poses[start:stop], dtype=np.float64)
    with np.load(base_arrays, allow_pickle=False) as stored:
        before = np.asarray(stored["cells"], dtype=np.uint8)
    transition = transition_counts(before=before, after=cells, target=target)
    per_class = {}
    for class_id, name in CLASS_NAMES.items():
        mask = target == class_id
        per_class[name] = {
            "errors": int(np.count_nonzero((cells != target) & mask)),
            "sites": int(np.count_nonzero(mask)),
        }
    return {
        "schema": "ddm_ws1_hood_masked_measurement_batch.v1",
        "typed_config_sha256": config.stable_hash(),
        "candidate_id": candidate_id,
        "pair_range": [start, stop],
        "errors": int(np.count_nonzero(cells != target)),
        "sites": int(cells.size),
        "pose_squared_error_sum": float(
            np.square(pose6 - target_pose).sum(dtype=np.float64)
        ),
        "pose_coordinates": int(pose6.size),
        "transition_from_v19c": transition,
        "per_class": per_class,
        "camera_sha256": sha256_bytes(camera.tobytes()),
        "cells_sha256": sha256_bytes(cells.tobytes()),
        "pose6_sha256": sha256_bytes(pose6.tobytes()),
        "delta_payload_bytes": payload_bytes,
        "decoder_derived_hood_camera_pixels": support_pixels,
        "frame0_byte_identical_to_v19c": True,
        "camera_batch_released_after_forward": True,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
    }


def measure(config_path: Path) -> dict[str, Any]:
    config = WS1Config.model_validate_json(_read(config_path))
    storage = _storage_preflight(config)
    menu1_payload = _resolve(config.menu1_config_path)
    if _sha256_file(menu1_payload) != config.menu1_config_sha256:
        raise WS1Error("MENU1 config SHA-256 differs")
    menu_config, inputs = _config_and_inputs(menu1_payload)
    if (
        menu_config.pair_count != config.pair_count
        or menu_config.scorer_batch_size != config.scorer_batch_size
        or menu_config.scorer_threads != config.scorer_threads
    ):
        raise WS1Error("MENU1 scorer geometry or threading differs")
    menu_receipt = _bound_json(
        config.menu1_receipt_path, config.menu1_receipt_sha256, "MENU1 receipt"
    )
    mc1_config = _bound_json(
        config.mc1_config_path, config.mc1_config_sha256, "MC1 config"
    )
    mc1_receipt = _bound_json(
        config.mc1_receipt_path, config.mc1_receipt_sha256, "MC1 receipt"
    )
    if (
        menu_receipt.get("score_claim") is not False
        or mc1_receipt.get("score_claim") is not False
        or mc1_config.get("exact_eval_allowed") is not False
    ):
        raise WS1Error("source advisory authority differs")
    lexicographic = _lexicographic_inventory(config)

    source_root = Path(menu_config.checkpoint_root)
    scalar_path = source_root / "01b_scalar_gain_bias_payload.bin"
    temporal_path = source_root / "01c_temporal_affine_payload.bin"
    statistics_path = source_root / "01_local_statistics_payload.bin"
    scalar_payload = _read(scalar_path)
    temporal_payload = _read(temporal_path)
    statistics_payload = _read(statistics_path)
    if len(scalar_payload) != 12 or len(temporal_payload) != 204:
        raise WS1Error("settled MENU1 amplitude payload sizes differ")

    receiver = receive_preuint8_q8_archive(inputs["archive"])
    palette = _palette(receiver)
    base_cells = _load_all_base_cells(
        source_root, batch_size=config.scorer_batch_size
    )
    supports = derive_hood_supports(base_cells)
    if supports.hood_class != 4:
        raise WS1Error("decoder-derived ego-hood class identity differs")
    labels = open_stored_npy_memmap(Path(menu_config.target_cache_path), "lstars")
    poses = open_stored_npy_memmap(Path(menu_config.target_cache_path), "gt_poses")
    if labels.shape != (config.pair_count, *SEG_HW) or poses.shape != (
        config.pair_count,
        6,
    ):
        raise WS1Error("target cache scorer geometry differs")

    existing_rows = {
        "v19c_base": _source_rows(source_root, "v19c_base"),
        "scalar_gain_bias_12b_frame1": _source_rows(
            source_root, "scalar_gain_bias_12b_frame1"
        ),
        "temporal_affine_16knot_frame1": _source_rows(
            source_root, "temporal_affine_16knot_frame1"
        ),
        WJOINT_ID: _source_rows(source_root, WJOINT_ID),
    }
    root = Path(config.checkpoint_root)
    missing = False
    for candidate_id in WSEG_IDS:
        for start in range(0, config.pair_count, config.scorer_batch_size):
            stop = min(start + config.scorer_batch_size, config.pair_count)
            row_path, array_path = _checkpoint_paths(
                root, candidate_id, start, stop
            )
            if not row_path.exists() or not array_path.exists():
                missing = True
                break
    segnet = posenet = scorer_custody = None
    if missing:
        segnet, posenet, scorer_custody = _load_models(menu_config)

    fresh_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in WSEG_IDS}
    fresh_replay = {}
    support_pixels_total = 0
    for start in range(0, config.pair_count, config.scorer_batch_size):
        stop = min(start + config.scorer_batch_size, config.pair_count)
        ids = tuple(range(start, stop))
        print(f"[WS1] compose batch {start:04d}:{stop:04d}", flush=True)
        base_camera = receiver.render_camera_pairs(ids)
        semantic, owned = _semantic_cells(
            receiver, ids, base_camera, palette
        )
        decoder_support = (semantic == supports.hood_class) & owned
        camera_support = expand_support_to_camera(
            decoder_support,
            batch_size=stop - start,
            camera_hw=CAMERA_HW,
        )
        support_pixels_total += int(np.count_nonzero(camera_support))
        transforms = {
            WSEG_IDS[0]: apply_scalar_affine(base_camera, scalar_payload),
            WSEG_IDS[1]: apply_temporal_affine(
                base_camera,
                pair_ids=ids,
                pair_count=config.pair_count,
                payload=temporal_payload,
            ),
        }
        unmasked_cameras = dict(transforms)
        transforms = {
            candidate_id: reassert_frame1(
                winner_camera=camera,
                base_camera=base_camera,
                camera_support=camera_support,
            )
            for candidate_id, camera in transforms.items()
        }
        joint_camera = _geometry_statistics_camera(
            base_camera=base_camera,
            semantic=semantic,
            owned=owned,
            palette=palette,
            statistics_payload=statistics_payload,
        )
        for source_id, camera in (
            ("scalar_gain_bias_12b_frame1", unmasked_cameras[WSEG_IDS[0]]),
            ("temporal_affine_16knot_frame1", unmasked_cameras[WSEG_IDS[1]]),
            (WJOINT_ID, joint_camera),
        ):
            source_row = existing_rows[source_id][start // config.scorer_batch_size]
            if source_row["camera_sha256"] != sha256_bytes(camera.tobytes()):
                raise WS1Error(f"fresh {source_id} composition hash differs")

        base_row_path, base_arrays = (
            source_root
            / "02_measurements"
            / "v19c_base"
            / f"batch_{start:04d}_{stop:04d}.json",
            source_root
            / "02_measurements"
            / "v19c_base"
            / f"batch_{start:04d}_{stop:04d}.npz",
        )
        if not base_row_path.is_file() or not base_arrays.is_file():
            raise WS1Error("V19C base checkpoint is absent")
        for candidate_id, camera in transforms.items():
            row_path, array_path = _checkpoint_paths(
                root, candidate_id, start, stop
            )
            if row_path.exists() and array_path.exists():
                row = json.loads(_read(row_path))
                if (
                    row["typed_config_sha256"] != config.stable_hash()
                    or row["candidate_id"] != candidate_id
                    or row["camera_sha256"] != sha256_bytes(camera.tobytes())
                ):
                    raise WS1Error("WS1 resume checkpoint identity differs")
                fresh_rows[candidate_id].append(row)
                continue
            if segnet is None or posenet is None:
                raise WS1Error("scorer models missing for fresh checkpoint")
            cells, pose6 = _forward(segnet, posenet, camera)
            if start == 0:
                replay_cells, replay_pose6 = _forward(segnet, posenet, camera)
                if not np.array_equal(cells, replay_cells) or not np.array_equal(
                    pose6, replay_pose6
                ):
                    raise WS1Error("first-batch deterministic replay differs")
            row = _fresh_row(
                config=config,
                candidate_id=candidate_id,
                start=start,
                stop=stop,
                camera=camera,
                cells=cells,
                pose6=pose6,
                labels=labels,
                poses=poses,
                base_arrays=base_arrays,
                payload_bytes=12 if candidate_id == WSEG_IDS[0] else 204,
                support_pixels=int(np.count_nonzero(camera_support)),
            )
            _publish_npz(array_path, cells=cells, pose6=pose6)
            _publish_json(row_path, row)
            fresh_rows[candidate_id].append(row)

        if start == 0:
            if segnet is None or posenet is None:
                segnet, posenet, scorer_custody = _load_models(menu_config)
            for source_id, camera in (
                ("temporal_affine_16knot_frame1", unmasked_cameras[WSEG_IDS[1]]),
                (WJOINT_ID, joint_camera),
            ):
                cells, pose6 = _forward(segnet, posenet, camera)
                source_arrays = (
                    source_root
                    / "02_measurements"
                    / source_id
                    / f"batch_{start:04d}_{stop:04d}.npz"
                )
                with np.load(source_arrays, allow_pickle=False) as stored:
                    if not np.array_equal(cells, stored["cells"]) or not np.array_equal(
                        pose6, stored["pose6"]
                    ):
                        raise WS1Error(f"fresh {source_id} scorer replay differs")
                fresh_replay[source_id] = {
                    "pair_range": [start, stop],
                    "cells_sha256": sha256_bytes(cells.tobytes()),
                    "pose6_sha256": sha256_bytes(pose6.tobytes()),
                    "status": "PASS",
                }

    base = _aggregate(
        existing_rows["v19c_base"],
        candidate_id="v19c_base",
        archive_bytes=menu_config.v19c_archive_bytes,
        payload_bytes=0,
    )
    unmasked = {
        WSEG_IDS[0]: _aggregate(
            existing_rows[UNMASKED_BY_WSEG[WSEG_IDS[0]]],
            candidate_id=UNMASKED_BY_WSEG[WSEG_IDS[0]],
            archive_bytes=menu_config.v19c_archive_bytes + 12,
            payload_bytes=12,
        ),
        WSEG_IDS[1]: _aggregate(
            existing_rows[UNMASKED_BY_WSEG[WSEG_IDS[1]]],
            candidate_id=UNMASKED_BY_WSEG[WSEG_IDS[1]],
            archive_bytes=menu_config.v19c_archive_bytes + 204,
            payload_bytes=204,
        ),
    }
    masked = {
        WSEG_IDS[0]: _aggregate(
            fresh_rows[WSEG_IDS[0]],
            candidate_id=WSEG_IDS[0],
            archive_bytes=menu_config.v19c_archive_bytes + 12,
            payload_bytes=12,
        ),
        WSEG_IDS[1]: _aggregate(
            fresh_rows[WSEG_IDS[1]],
            candidate_id=WSEG_IDS[1],
            archive_bytes=menu_config.v19c_archive_bytes + 204,
            payload_bytes=204,
        ),
    }
    wseg = min(masked.values(), key=lambda row: row["d_seg"])
    wjoint = _aggregate(
        existing_rows[WJOINT_ID],
        candidate_id=WJOINT_ID,
        archive_bytes=menu_config.v19c_archive_bytes + len(statistics_payload),
        payload_bytes=len(statistics_payload),
    )
    if base["per_class"]["MyCar"]["errors"] != config.mycar_control_errors:
        raise WS1Error("V19C MyCar control differs from 37,237")
    if wseg["per_class"]["MyCar"]["errors"] > config.mycar_material_defect_ceiling:
        raise WS1Error("W_seg materially re-damaged the solved MyCar bucket")
    if wseg["d_seg"] >= base["d_seg"]:
        raise WS1Error("hood-safe W_seg does not improve V19C d_seg")
    if wjoint["candidate_id"] != WJOINT_ID:
        raise WS1Error("W_joint identity differs")

    for candidate_id, row in masked.items():
        source = unmasked[candidate_id]
        row["hood_mask_delta_vs_unmasked"] = {
            "delta_errors": row["errors"] - source["errors"],
            "delta_d_seg": row["d_seg"] - source["d_seg"],
            "delta_d_pose": row["d_pose"] - source["d_pose"],
            "delta_mycar_errors": (
                row["per_class"]["MyCar"]["errors"]
                - source["per_class"]["MyCar"]["errors"]
            ),
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
        "storage_preflight": storage,
        "scorer_custody": scorer_custody
        or menu_receipt.get("scorer_custody"),
        "fresh_composition_replay": fresh_replay,
        "hood_support": {
            "hood_class": supports.hood_class,
            "support_kind": "decoder_derived_per_frame_semantic_owned",
            "counted_bytes": 0,
            "stored_static_support_reference_bytes": 139,
            "stored_static_support_reference_sha256": (
                mc1_receipt["support_derivation"]["partition"]["static_stored"][
                    "sha256"
                ]
            ),
            "camera_support_pixels_total": support_pixels_total,
            "frame0_byte_identical": True,
        },
        "seg_lexicographic_rerank": lexicographic,
        "controls": {
            "v19c": base,
            "amplitude_unmasked": unmasked,
            "amplitude_hood_masked": masked,
        },
        "warm_start_candidates": {
            "W_seg": {
                **wseg,
                "role": "best_measured_d_seg_pose_tolerant_hood_safe",
                "base_receiver": "SHA-bound V19C endpoint",
                "rate_accounting": {
                    "base_archive_bytes": menu_config.v19c_archive_bytes,
                    "amplitude_payload_counted_bytes": wseg["delta_payload_bytes"],
                    "decoder_derived_hood_support_counted_bytes": 0,
                    "stored_support_reference_bytes_not_counted": 139,
                },
            },
            "W_joint": {
                **wjoint,
                "role": "current_best_joint_S_MENU1_winner",
                "fresh_exact_first_batch_scorer_replay": fresh_replay[WJOINT_ID],
            },
        },
        "triality": {
            "dsl": str(config_path.relative_to(REPO_ROOT)),
            "dag_feed": (
                ".omx/research/ddm_ws1_seg_lexicographic_warmstart_DAG_"
                "20260724.md"
            ),
            "equation_id": "ddm_ws1_warm_start_slope_falsifier_v1",
        },
        "verdict": (
            "TWO_ADVISORY_WARM_STARTS_MEASURED; J5 SLOPE FALSIFIER "
            "SPECIFICATION ONLY; NO TRAINING LAUNCHED"
        ),
        "verdict_scope": (
            "[macOS-CPU frozen-scorer advisory] exact receiver-through-R "
            "warm-start comparison only; contest CPU/CUDA and promotion remain owed"
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
        / "ddm_ws1_seg_lexicographic_warmstart_receipt.json"
    )
    _publish_json(output, receipt)
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    measure(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
