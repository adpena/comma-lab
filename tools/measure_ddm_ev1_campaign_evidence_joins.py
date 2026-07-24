#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# ruff: noqa: E402
"""Measure the bounded n600 V19/RD1 campaign evidence joins.

The producer is deliberately local-CPU, resumable at every 16-pair batch, and
authority-preserving:

* V19's archive-rate delta has one global home, never a fabricated per-pair
  allocation.
* RD1 byte homes are an exact accounting partition derived from measured
  scorer-metric dimensions.  They are not claims of physical ZIP separability.
* Every RD1 price remains null; the ms2r arm owns the 162-dimensional solve.
* Receiver histograms describe exact uint8 absolute steps.  They never replace
  the rank-4 margin-Fisher/composite-R scorer metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import brotli
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
from tac.ddm_campaign_evidence_join import (
    G4_CLASSES,
    METRIC_ID,
    PAIR_COUNT,
    RD1_SCHEMA,
    SCHEMA,
    STRATA,
    V19_SCHEMA,
    VISIBILITIES,
    allocate_exclusive_byte_homes,
    bucket_key,
    canonical_bytes,
    derive_g4_reuse_profiles,
    pack_histogram_rows,
    unpack_histogram_rows,
    validate_campaign_evidence_join,
)
from tac.optimization.ddm_realized_flip_menu import apply_scalar_affine
from tac.optimization.direct_description_carrier_compose import (
    CLASS_ORDER,
    receive_carrier_compose_archive,
)
from tac.optimization.direct_description_coupled_margin import (
    receive_coupled_margin_archive,
)
from tools.measure_ddm_menu1_realized_flip_menu import (
    CAMERA_HW,
    SEG_HW,
    Menu1Config,
    _forward,
    _geometry_statistics_camera,
    _load_models,
    _palette,
    _semantic_cells,
)
from tools.measure_ddm_ms6_receiver_support import _fast_receiver

AXIS = "[macOS-CPU frozen-scorer advisory]"
PRICE_OWNER = "ddm_ms2r_tolerance_capped_solve_r2"
ENDPOINT_BYTES = (137_823, 137_839, 138_801, 409_526_925)
EDGE_NAMES = (
    ("v19c_admit_0005", "scalar_gain_bias_12b_frame1"),
    (
        "scalar_gain_bias_12b_frame1",
        "statistics_hard_analytic_composed_frame1",
    ),
    ("statistics_hard_analytic_composed_frame1", "c1_exact_solved_n600"),
)
SEMANTIC_CLASSES = dict(enumerate(CLASS_ORDER))
G4_STATIONARITY_LEDGER = (
    REPO
    / ".omx/research/ddm_g4_spatial_stationarity_n600_20260722T212138Z"
    / "stationarity_ledger.jsonl"
)
G4_STATIONARITY_LEDGER_SHA256 = "64713063e869d1243e1876541c3a5c60b40a5dabc5d48708312d654e82b3a51b"


class EV1Error(RuntimeError):
    """The bounded evidence producer lost custody or failed closure."""


class Config(BaseModel):
    """Strict SHA-bound local execution contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMEV1CampaignEvidenceJoinConfigV1"] = Field(
        alias="schema",
        serialization_alias="schema",
    )
    run_id: str = Field(min_length=8)
    seed: Literal[1234]
    pair_count: Literal[600]
    batch_size: Literal[16]
    scorer_threads: Literal[4]
    v19_control_archive_path: str
    v19_control_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v19_candidate_archive_path: str
    v19_candidate_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rd1_admit_archive_path: str
    rd1_admit_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rd1_base_archive_path: str
    rd1_base_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rd1_scalar_payload_path: str
    rd1_scalar_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rd1_statistics_payload_path: str
    rd1_statistics_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rd1_c1_raw_path: str
    rd1_c1_raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rd1_c1_archive_path: str
    rd1_c1_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    g4_recurrence_path: str
    g4_recurrence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    g4_xi_tracks_path: str
    g4_xi_tracks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    menu_config_path: str
    menu_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_root: str
    output_receipt_path: str
    required_free_bytes: int = Field(ge=20 << 30)
    execution_allowed: Literal[True]
    research_only: Literal[True]
    score_claim: Literal[False]

    def typed_hash(self) -> str:
        return _sha(canonical_bytes(self.model_dump(mode="json", by_alias=True)))


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO / value


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _bound(config: Config, path_name: str) -> tuple[Path, bytes]:
    path = _resolve(str(getattr(config, path_name)))
    if not path.is_file() or path.is_symlink():
        raise EV1Error(f"regular non-symlink source required: {path}")
    observed = _sha_file(path)
    expected = str(getattr(config, f"{path_name.removesuffix('_path')}_sha256"))
    if observed != expected:
        raise EV1Error(f"{path_name} SHA differs: {observed} != {expected}")
    return path, path.read_bytes()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise EV1Error(f"immutable JSON checkpoint differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _output_row(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha_file(path)}


def _load_config(path: Path) -> Config:
    if not path.is_file() or path.is_symlink():
        raise EV1Error("config must be a regular non-symlink file")
    return Config.model_validate_json(path.read_bytes())


def _batch_path(root: Path, stage: str, start: int, stop: int) -> tuple[Path, Path]:
    base = root / "stage_checkpoints" / stage / f"batch_{start:04d}_{stop:04d}"
    return base.with_suffix(".json"), base.with_suffix(".npz")


class Context:
    """Live receiver/scorer state.  Large camera tensors never persist here."""

    def __init__(self, config: Config) -> None:
        self.config = config
        sources: dict[str, dict[str, Any]] = {}
        payloads: dict[str, bytes] = {}
        for name in (
            "v19_control_archive",
            "v19_candidate_archive",
            "rd1_admit_archive",
            "rd1_base_archive",
            "rd1_scalar_payload",
            "rd1_statistics_payload",
            "rd1_c1_archive",
            "g4_xi_tracks",
            "menu_config",
        ):
            path, payload = _bound(config, f"{name}_path")
            payloads[name] = payload
            sources[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": str(getattr(config, f"{name}_sha256")),
            }
        for name in ("rd1_c1_raw", "g4_recurrence"):
            path = _resolve(str(getattr(config, f"{name}_path")))
            if not path.is_file() or path.is_symlink():
                raise EV1Error(f"regular non-symlink source required: {path}")
            if _sha_file(path) != str(getattr(config, f"{name}_sha256")):
                raise EV1Error(f"{name} SHA differs")
            sources[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": str(getattr(config, f"{name}_sha256")),
            }
        self.sources = sources
        self.control = receive_carrier_compose_archive(payloads["v19_control_archive"], verify_member_effects=False)
        self.v19 = receive_coupled_margin_archive(payloads["v19_candidate_archive"], verify_base_member_effects=False)
        self.admit = _fast_receiver(payloads["rd1_admit_archive"])
        self.base = _fast_receiver(payloads["rd1_base_archive"])
        self.scalar_payload = payloads["rd1_scalar_payload"]
        self.statistics_payload = payloads["rd1_statistics_payload"]
        self.palette = _palette(self.base)
        menu = Menu1Config.model_validate_json(payloads["menu_config"])
        self.labels = open_stored_npy_memmap(menu.target_cache_path, "lstars")
        self.poses = open_stored_npy_memmap(menu.target_cache_path, "gt_poses")
        self.segnet, self.posenet, self.scorer_custody = _load_models(menu)
        raw_path = Path(config.rd1_c1_raw_path)
        expected_values = PAIR_COUNT * 2 * CAMERA_HW[0] * CAMERA_HW[1] * 3
        if raw_path.stat().st_size != expected_values:
            raise EV1Error("C1 raw byte geometry differs")
        self.c1 = np.memmap(
            raw_path,
            mode="r",
            dtype=np.uint8,
            shape=(PAIR_COUNT, 2, *CAMERA_HW, 3),
        )
        with np.load(config.g4_recurrence_path) as arrays:
            self.transition_counts = np.asarray(arrays["transition_counts"], dtype=np.uint16)
        if self.transition_counts.shape != (25, *SEG_HW):
            raise EV1Error("G4 recurrence geometry differs")
        self.xi_membership: set[int] = set()
        xi_track_lengths: list[int] = []
        for line in payloads["g4_xi_tracks"].splitlines():
            if line:
                track = json.loads(line)
                self.xi_membership.update(track["event_ids"])
                xi_track_lengths.append(int(track["length"]))
        if (
            not G4_STATIONARITY_LEDGER.is_file()
            or G4_STATIONARITY_LEDGER.is_symlink()
            or _sha_file(G4_STATIONARITY_LEDGER) != G4_STATIONARITY_LEDGER_SHA256
        ):
            raise EV1Error("G4 stationarity ledger custody differs")
        first_row = json.loads(G4_STATIONARITY_LEDGER.read_text().splitlines()[0])
        if (
            first_row.get("schema") != "ddm_g4_spatial_stationarity_ledger.v1"
            or first_row.get("record_id") != "global"
        ):
            raise EV1Error("G4 stationarity global row differs")
        self.g4_reuse_profiles = derive_g4_reuse_profiles(
            recurrence_k_distribution=first_row["payload"]["recurrence_k_distribution"],
            xi_track_lengths=xi_track_lengths,
        )
        self.sources["g4_stationarity_ledger"] = {
            "path": str(G4_STATIONARITY_LEDGER),
            "bytes": G4_STATIONARITY_LEDGER.stat().st_size,
            "sha256": G4_STATIONARITY_LEDGER_SHA256,
            "record_id": "global",
            "field": "payload.recurrence_k_distribution.exact_k",
        }

    def camera(self, endpoint: str, ids: Sequence[int]) -> np.ndarray:
        if endpoint == "v19_control":
            result = self.control.render_camera_pairs(ids)
        elif endpoint == "v19_candidate":
            result = self.v19.render_camera_pairs(ids)
        elif endpoint == "v19c_admit_0005":
            result = self.admit.render_camera_pairs(ids)
        elif endpoint in {
            "scalar_gain_bias_12b_frame1",
            "statistics_hard_analytic_composed_frame1",
        }:
            base = self.base.render_camera_pairs(ids)
            if endpoint == "scalar_gain_bias_12b_frame1":
                result = apply_scalar_affine(base, self.scalar_payload)
            else:
                semantic, owned = _semantic_cells(self.base, ids, base, self.palette)
                result = _geometry_statistics_camera(
                    base_camera=base,
                    semantic=semantic,
                    owned=owned,
                    palette=self.palette,
                    statistics_payload=self.statistics_payload,
                )
        elif endpoint == "c1_exact_solved_n600":
            result = np.asarray(self.c1[np.asarray(ids, dtype=np.int64)])
        else:
            raise EV1Error(f"unknown endpoint {endpoint}")
        result = np.ascontiguousarray(result)
        if result.dtype != np.uint8 or result.shape != (
            len(ids),
            2,
            *CAMERA_HW,
            3,
        ):
            raise EV1Error(f"{endpoint} receiver geometry differs")
        return result


def _score_endpoint(
    context: Context,
    root: Path,
    endpoint: str,
) -> None:
    for start in range(0, PAIR_COUNT, context.config.batch_size):
        stop = min(start + context.config.batch_size, PAIR_COUNT)
        row_path, array_path = _batch_path(root, f"score_{endpoint}", start, stop)
        if row_path.exists() and array_path.exists():
            continue
        ids = tuple(range(start, stop))
        camera = context.camera(endpoint, ids)
        cells, pose6 = _forward(context.segnet, context.posenet, camera)
        authority = "FRESH_FROZEN_SCORER_FORWARD"
        row = {
            "schema": "ddm_ev1_endpoint_score_batch.v1",
            "endpoint": endpoint,
            "pair_range": [start, stop],
            "camera_sha256": _sha_array(camera),
            "cells_sha256": _sha_array(cells),
            "pose6_sha256": _sha_array(pose6),
            "score_batch_authority": authority,
            "typed_config_sha256": context.config.typed_hash(),
            "receiver_closed": True,
            "score_claim": False,
        }
        _atomic_npz(array_path, cells=cells, pose6=pose6)
        _atomic_json(row_path, row)
        print(f"[EV1] score {endpoint} {start:04d}:{stop:04d}", flush=True)


def _load_scores(
    context: Context,
    root: Path,
    endpoint: str,
) -> tuple[np.ndarray, np.ndarray]:
    cells = np.empty((PAIR_COUNT, *SEG_HW), dtype=np.uint8)
    pose = np.empty((PAIR_COUNT, 6), dtype=np.float64)
    for start in range(0, PAIR_COUNT, context.config.batch_size):
        stop = min(start + context.config.batch_size, PAIR_COUNT)
        row_path, array_path = _batch_path(root, f"score_{endpoint}", start, stop)
        row = json.loads(row_path.read_bytes())
        if row["typed_config_sha256"] != context.config.typed_hash() or row["endpoint"] != endpoint:
            raise EV1Error("endpoint score resume identity differs")
        with np.load(array_path) as arrays:
            cells[start:stop] = arrays["cells"]
            pose[start:stop] = arrays["pose6"]
    return cells, pose


def _g4_categories(
    context: Context,
    pair_ids: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
) -> np.ndarray:
    code = before.astype(np.int16) * 5 + after.astype(np.int16)
    yy = np.arange(SEG_HW[0])[:, None]
    xx = np.arange(SEG_HW[1])[None, :]
    recurrence = context.transition_counts[code, yy, xx]
    changed = before != after
    category = np.full(before.shape, 2, dtype=np.uint8)
    # Image stationarity is recurrence of the exact evaluator-cell transition
    # code. Diagonal (unchanged) codes are stationary rather than transient;
    # the Seg-visible histogram separately requires an actual argmax change.
    category[recurrence >= 2] = 0
    if context.xi_membership:
        sites = SEG_HW[0] * SEG_HW[1]
        for local, pair_id in enumerate(pair_ids):
            for event in context.xi_membership:
                if event // sites != int(pair_id):
                    continue
                row, col = divmod(event % sites, SEG_HW[1])
                if changed[local, row, col] and recurrence[local, row, col] < 2:
                    category[local, row, col] = 1
    return category


def _hist_add(hist: np.ndarray, values: np.ndarray) -> None:
    changed = values[values != 0]
    if changed.size:
        hist += np.bincount(changed, minlength=256).astype(np.uint64)


def _measure_rd1_edge(
    context: Context,
    root: Path,
    dual: int,
    before_name: str,
    after_name: str,
) -> dict[str, Any]:
    stage = root / "stage_checkpoints" / f"rd1_edge_{dual}"
    final_path = stage / "edge_measurement.npz"
    final_row = stage / "edge_measurement.json"
    if final_path.exists() and final_row.exists():
        return json.loads(final_row.read_bytes())

    before_cells, before_pose = _load_scores(context, root, before_name)
    after_cells, after_pose = _load_scores(context, root, after_name)
    histograms = {
        (stratum, visibility, g4): np.zeros(256, dtype=np.uint64)
        for stratum in STRATA
        for visibility in VISIBILITIES
        for g4 in G4_CLASSES
    }
    semantic_delta = {(stratum, g4): 0 for stratum in STRATA[:-1] for g4 in G4_CLASSES}
    pose_step_mass = dict.fromkeys(G4_CLASSES, 0)
    row_map = np.arange(CAMERA_HW[0], dtype=np.int64) * SEG_HW[0] // CAMERA_HW[0]
    col_map = np.arange(CAMERA_HW[1], dtype=np.int64) * SEG_HW[1] // CAMERA_HW[1]
    for start in range(0, PAIR_COUNT, context.config.batch_size):
        stop = min(start + context.config.batch_size, PAIR_COUNT)
        batch_json, batch_npz = _batch_path(root, f"rd1_edge_{dual}_hist", start, stop)
        if batch_json.exists() and batch_npz.exists():
            with np.load(batch_npz) as arrays:
                for key in histograms:
                    histograms[key] += arrays["hist_" + "__".join(key)]
                for key in semantic_delta:
                    semantic_delta[key] += int(arrays["delta_" + "__".join(key)][0])
                for g4 in G4_CLASSES:
                    pose_step_mass[g4] += int(arrays[f"pose_mass_{g4}"][0])
            continue
        pair_ids = np.arange(start, stop, dtype=np.int64)
        before_camera = context.camera(before_name, tuple(pair_ids))
        after_camera = context.camera(after_name, tuple(pair_ids))
        absolute = np.abs(after_camera.astype(np.int16) - before_camera.astype(np.int16)).astype(np.uint8)
        categories = _g4_categories(
            context,
            pair_ids,
            before_cells[start:stop],
            after_cells[start:stop],
        )
        target = np.asarray(context.labels[pair_ids], dtype=np.uint8)
        error_delta = (after_cells[start:stop] != target).astype(np.int8) - (before_cells[start:stop] != target).astype(
            np.int8
        )
        local_hist = {key: np.zeros(256, dtype=np.uint64) for key in histograms}
        local_delta = dict.fromkeys(semantic_delta, 0)
        local_pose_mass = dict.fromkeys(G4_CLASSES, 0)
        category_camera = categories[:, row_map[:, None], col_map[None, :]]
        target_camera = target[:, row_map[:, None], col_map[None, :]]
        changed_camera = (before_cells[start:stop] != after_cells[start:stop])[:, row_map[:, None], col_map[None, :]]
        for category_index, g4 in enumerate(G4_CLASSES):
            cell_g4 = categories == category_index
            camera_g4 = category_camera == category_index
            for class_id, stratum in SEMANTIC_CLASSES.items():
                mask = camera_g4 & changed_camera & (target_camera == class_id)
                _hist_add(
                    local_hist[(stratum, "seg-visible", g4)],
                    absolute[:, 1][mask],
                )
                local_delta[(stratum, g4)] = int(error_delta[(target == class_id) & cell_g4].sum())
            pose_values = absolute[np.broadcast_to(camera_g4[:, None, :, :, None], absolute.shape)]
            _hist_add(
                local_hist[("POSE6_GLOBAL", "pose-visible", g4)],
                pose_values,
            )
            local_pose_mass[g4] = int(pose_values.astype(np.uint64).sum())
        arrays: dict[str, np.ndarray] = {}
        for key, value in local_hist.items():
            arrays["hist_" + "__".join(key)] = value
            histograms[key] += value
        for key, value in local_delta.items():
            arrays["delta_" + "__".join(key)] = np.asarray([value], dtype=np.int64)
            semantic_delta[key] += value
        for g4, value in local_pose_mass.items():
            arrays[f"pose_mass_{g4}"] = np.asarray([value], dtype=np.uint64)
            pose_step_mass[g4] += value
        _atomic_npz(batch_npz, **arrays)
        _atomic_json(
            batch_json,
            {
                "schema": "ddm_ev1_rd1_hist_batch.v1",
                "dual_index": dual,
                "before_endpoint": before_name,
                "after_endpoint": after_name,
                "pair_range": [start, stop],
                "before_camera_sha256": _sha_array(before_camera),
                "after_camera_sha256": _sha_array(after_camera),
                "typed_config_sha256": context.config.typed_hash(),
                "receiver_closed": True,
                "score_claim": False,
            },
        )
        print(f"[EV1] RD1 edge {dual} hist {start:04d}:{stop:04d}", flush=True)

    sites = PAIR_COUNT * SEG_HW[0] * SEG_HW[1]
    semantic_D = {key: 100.0 * value / sites for key, value in semantic_delta.items()}
    poses = np.asarray(context.poses, dtype=np.float64)
    before_d_pose = float(np.mean(np.square(before_pose - poses), dtype=np.float64))
    after_d_pose = float(np.mean(np.square(after_pose - poses), dtype=np.float64))
    pose_delta = math.sqrt(10.0 * after_d_pose) - math.sqrt(10.0 * before_d_pose)
    total_mass = sum(pose_step_mass.values())
    if total_mass <= 0:
        raise EV1Error(f"RD1 edge {dual} has no receiver-closed uint8 movement")
    pose_D = {g4: pose_delta * pose_step_mass[g4] / total_mass for g4 in G4_CLASSES}
    arrays = {}
    for key, value in histograms.items():
        arrays["hist_" + "__".join(key)] = value
    for key, value in semantic_D.items():
        arrays["semantic_D_" + "__".join(key)] = np.asarray([value], dtype=np.float64)
    for key, value in pose_D.items():
        arrays[f"pose_D_{key}"] = np.asarray([value], dtype=np.float64)
    _atomic_npz(final_path, **arrays)
    row = {
        "schema": "ddm_ev1_rd1_edge_measurement.v1",
        "dual_index": dual,
        "before_endpoint": before_name,
        "after_endpoint": after_name,
        "before_counted_bytes": ENDPOINT_BYTES[dual - 1],
        "after_counted_bytes": ENDPOINT_BYTES[dual],
        "delta_counted_bytes": ENDPOINT_BYTES[dual] - ENDPOINT_BYTES[dual - 1],
        "before_d_pose": before_d_pose,
        "after_d_pose": after_d_pose,
        "pose_term_delta_D": pose_delta,
        "semantic_term_delta_D": sum(semantic_D.values()),
        "joint_delta_D": sum(semantic_D.values()) + pose_delta,
        "receiver_changed_channel_values": int(sum(value.sum(dtype=np.uint64) for value in histograms.values())),
        "receiver_uint8_abs_step_sum": int(
            sum(np.arange(256, dtype=np.uint64) @ value for value in histograms.values())
        ),
        "measurement_npz": _output_row(final_path),
        "receiver_closed": True,
        "score_claim": False,
    }
    _atomic_json(final_row, row)
    return row


def _v19_join(context: Context, root: Path) -> dict[str, Any]:
    before_cells, before_pose = _load_scores(context, root, "v19_control")
    after_cells, after_pose = _load_scores(context, root, "v19_candidate")
    labels = np.asarray(context.labels, dtype=np.uint8)
    poses = np.asarray(context.poses, dtype=np.float64)
    before_errors = int(np.count_nonzero(before_cells != labels))
    after_errors = int(np.count_nonzero(after_cells != labels))
    before_d_pose = float(np.mean(np.square(before_pose - poses), dtype=np.float64))
    after_d_pose = float(np.mean(np.square(after_pose - poses), dtype=np.float64))
    if (before_errors, after_errors) != (3_240_528, 3_198_107):
        raise EV1Error("V19 n600 Seg replay differs from sealed control/candidate receipt")
    if abs(before_d_pose - 163.061327281443) > 5.0e-10 or abs(after_d_pose - 163.061610203945) > 5.0e-10:
        raise EV1Error("V19 n600 Pose replay differs from sealed control/candidate receipt")
    rows = []
    for pair_id in range(PAIR_COUNT):
        before_error = before_cells[pair_id] != labels[pair_id]
        after_error = after_cells[pair_id] != labels[pair_id]
        per_stratum = {}
        for class_id, stratum in SEMANTIC_CLASSES.items():
            mask = labels[pair_id] == class_id
            per_stratum[stratum] = {
                "errors_before": int(np.count_nonzero(before_error & mask)),
                "errors_after": int(np.count_nonzero(after_error & mask)),
                "sites": int(np.count_nonzero(mask)),
            }
        rows.append(
            {
                "source_pair_id": pair_id,
                "errors_before": int(np.count_nonzero(before_error)),
                "errors_after": int(np.count_nonzero(after_error)),
                "sites": int(before_error.size),
                "d_seg_before": float(np.mean(before_error)),
                "d_seg_after": float(np.mean(after_error)),
                "d_pose_before": float(
                    np.mean(
                        np.square(before_pose[pair_id] - poses[pair_id]),
                        dtype=np.float64,
                    )
                ),
                "d_pose_after": float(
                    np.mean(
                        np.square(after_pose[pair_id] - poses[pair_id]),
                        dtype=np.float64,
                    )
                ),
                "changed_argmax_cells": int(np.count_nonzero(before_cells[pair_id] != after_cells[pair_id])),
                "helpful_flips": int(np.count_nonzero(before_error & ~after_error)),
                "harmful_flips": int(np.count_nonzero(~before_error & after_error)),
                "per_stratum": per_stratum,
                "receiver_closed": True,
                "per_pair_byte_allocation": None,
                "score_claim": False,
            }
        )
    return {
        "schema": V19_SCHEMA,
        "pair_count": PAIR_COUNT,
        "global_replay": {
            "control_errors": before_errors,
            "candidate_errors": after_errors,
            "control_d_pose": before_d_pose,
            "candidate_d_pose": after_d_pose,
            "sealed_receipt_replay_exact": True,
        },
        "rows": rows,
        "shared_rate_home": {
            "home_id": "v19_n600_archive_delta",
            "before_archive_bytes": 133_941,
            "after_archive_bytes": 135_529,
            "delta_counted_bytes": 1_588,
            "counted_exactly_once": True,
            "per_pair_allocation": None,
            "status": "MEASURED_EXACT_GLOBAL_ARCHIVE_DELTA_HOME",
        },
        "evidence_axis": AXIS,
        "score_claim": False,
    }


def _rd1_join(
    context: Context,
    root: Path,
    edges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bucket_rows: list[dict[str, Any]] = []
    edge_summaries = []
    for edge in edges:
        dual = int(edge["dual_index"])
        with np.load(edge["measurement_npz"]["path"]) as arrays:
            weights: dict[tuple[int, str, str, str], float] = {}
            raw: dict[tuple[int, str, str, str], tuple[list[int], float]] = {}
            for stratum in STRATA:
                for visibility in VISIBILITIES:
                    for g4 in G4_CLASSES:
                        key = (dual, stratum, visibility, g4)
                        hist = arrays["hist_" + "__".join((stratum, visibility, g4))].astype(np.uint64)
                        if stratum == "POSE6_GLOBAL" and visibility == "pose-visible":
                            delta_D = float(arrays[f"pose_D_{g4}"][0])
                        elif stratum != "POSE6_GLOBAL" and visibility == "seg-visible":
                            delta_D = float(arrays["semantic_D_" + "__".join((stratum, g4))][0])
                        else:
                            delta_D = 0.0
                        raw[key] = (hist.astype(object).tolist(), delta_D)
                        weights[key] = abs(delta_D)
            if sum(weights.values()) <= 0:
                raise EV1Error(f"RD1 edge {dual} has no scorer-metric byte-home weight")
            homes = allocate_exclusive_byte_homes(
                delta_counted_bytes=int(edge["delta_counted_bytes"]),
                weights=weights,
            )
            for key in sorted(raw):
                _dual, stratum, visibility, g4 = key
                hist, delta_D = raw[key]
                reuse = context.g4_reuse_profiles[g4]
                home = homes[key]
                bucket_rows.append(
                    {
                        "dual_index": dual,
                        "stratum": stratum,
                        "scorer_visibility": visibility,
                        "g4_temporal_class": g4,
                        "metric_id": METRIC_ID,
                        "delta_D_dimension": delta_D,
                        "receiver_uint8_abs_step_histogram": hist,
                        "receiver_changed_channel_values": int(sum(hist)),
                        "receiver_uint8_abs_step_sum": int(sum(index * value for index, value in enumerate(hist))),
                        **home,
                        **reuse,
                        "amortized_bytes_per_frame": (
                            home["delta_counted_bytes_dimension"]
                            * reuse["k_denominator"]
                            / reuse["k_numerator"]
                        ),
                        "lambda_bytes_per_D_dimension": None,
                        "pricing_owner": PRICE_OWNER,
                        "score_claim": False,
                    }
                )
        edge_summaries.append(
            {
                **dict(edge),
                "assigned_counted_bytes": int(edge["delta_counted_bytes"]),
                "counted_exactly_once": True,
                "accounting_home_not_physical_zip_separability": True,
            }
        )
    packed = pack_histogram_rows(bucket_rows)
    compressed = brotli.compress(packed, quality=11)
    codec_path = root / "outputs" / "rd1_uint8_histograms.bin.br"
    codec_path.parent.mkdir(parents=True, exist_ok=True)
    if codec_path.exists():
        if codec_path.read_bytes() != compressed:
            raise EV1Error("immutable RD1 histogram coder output differs")
    else:
        temporary = codec_path.with_name(f".{codec_path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(compressed)
        os.replace(temporary, codec_path)
    restored = brotli.decompress(compressed)
    keys = sorted(bucket_key(row) for row in bucket_rows)
    parsed = unpack_histogram_rows(restored, keys)
    expected = {bucket_key(row): row["receiver_uint8_abs_step_histogram"] for row in bucket_rows}
    if parsed != expected:
        raise EV1Error("Brotli Q11 histogram parse-back differs")
    return {
        "schema": RD1_SCHEMA,
        "bucket_rows": bucket_rows,
        "edge_summaries": edge_summaries,
        "amortization_custody": {
            "source": context.sources["g4_stationarity_ledger"],
            "profiles": context.g4_reuse_profiles,
            "single_owner_across_reach": True,
            "shared_clip_bucket_count": 0,
            "shared_clip_absence_reason": "NO_AGGREGATED_G4_BUCKET_IS_EXCLUSIVELY_K_EQ_600",
            "application_scope": (
                "G4_CLASS_LEVEL_REUSE_PRIOR_APPLIED_TO_TYPED_HOMES_NOT_EDGE_SPECIFIC_K_REMEASUREMENT"
            ),
            "scope": (
                "G4 exact-k event mass per recurrent locus is the effective shared-field reach; "
                "the aggregated static bucket is not promoted to shared_clip"
            ),
        },
        "histogram_coder": {
            "codec": "BROTLI_Q11",
            "quality": 11,
            "uncompressed_bytes": len(packed),
            "compressed": _output_row(codec_path),
            "parse_back_identical": True,
        },
        "pricing_status": "PENDING_MS2R_0_OF_162_ACTIONABLE",
        "score_claim": False,
    }


def _run(config: Config) -> Path:
    root = Path(config.checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < config.required_free_bytes:
        raise EV1Error(f"storage preflight REFUSE: {free} < {config.required_free_bytes}")
    context = Context(config)
    _atomic_json(
        root / "stage_checkpoints" / "00_sources_validated.json",
        {
            "schema": "ddm_ev1_sources_validated.v1",
            "typed_config_sha256": config.typed_hash(),
            "sources": context.sources,
            "storage_preflight": {
                "tier": "/Volumes/VertigoDataTier/pact",
                "required_free_bytes": config.required_free_bytes,
                "free_space_gate_satisfied": True,
                "status": "PASS",
            },
            "status": "complete",
        },
    )
    endpoints = (
        "v19_control",
        "v19_candidate",
        "v19c_admit_0005",
        "scalar_gain_bias_12b_frame1",
        "statistics_hard_analytic_composed_frame1",
        "c1_exact_solved_n600",
    )
    for endpoint in endpoints:
        _score_endpoint(context, root, endpoint)
    v19 = _v19_join(context, root)
    edges = [
        _measure_rd1_edge(context, root, dual, before, after)
        for dual, (before, after) in enumerate(EDGE_NAMES, start=1)
    ]
    rd1 = _rd1_join(context, root, edges)
    receipt = {
        "schema": SCHEMA,
        "run_id": config.run_id,
        "pair_count": PAIR_COUNT,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_hash(),
        "source_custody": context.sources,
        "implementation_custody": {
            "git_head_at_measurement": ("UNCOMMITTED_DELEGATED_WORKTREE; LANDING_COMMIT_REQUIRED"),
            "source_files": [
                _output_row(Path(__file__).resolve()),
                _output_row(REPO / "src/tac/ddm_campaign_evidence_join.py"),
            ],
        },
        "runtime_custody": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": __import__("torch").__version__,
            "isolated_ssd_venv_required": True,
            "evidence_axis": AXIS,
        },
        "scorer_custody": context.scorer_custody,
        "metric_custody": {
            "metric_id": METRIC_ID,
            "margin_fisher_gram_bound": True,
            "composite_R_hessian_bound": True,
            "euclidean_naive_rows_admitted": False,
            "semantic_component": "100 * exact receiver-closed argmax error delta",
            "pose_component": "sqrt(10 * exact receiver-closed Pose6 MSE)",
            "rate_component": "exact counted archive bytes",
        },
        "rd1_endpoint_remeasurement": {
            "status": "FRESH_CURRENT_RECEIVER_REPLAY_MEASURED",
            "prior_menu_score_batches_reused": False,
            "reason": (
                "sealed MENU1 camera hashes do not replay under the current "
                "receiver source; mixing prior scores with current uint8 cameras "
                "was refused"
            ),
            "endpoint_ids": [EDGE_NAMES[0][0], *[edge[1] for edge in EDGE_NAMES]],
            "receiver_scope": (
                "EXACT_MEASUREMENT_HARNESS_PAYLOAD_DECODE_THROUGH_UINT8_R; "
                "MENU1_PAYLOAD_ENDPOINTS_ARE_NOT_CONTEST_ARCHIVES"
            ),
            "evidence_axis": AXIS,
        },
        "v19_pair_join": v19,
        "rd1_evidence": rd1,
        "free_interpreter_custody": {
            "generic_decoder_and_transport_code_counted_bytes": 0,
            "irreducible_video_statistic_bytes_counted": True,
            "xi_video_parameters_free": False,
            "independent_physical_bev_claim": False,
            "law": (
                "generic decoder and xi-transport interpreter code is free; each video-derived "
                "residual or xi statistic is counted once unless already present in the decoded state"
            ),
        },
        "triality": {
            "dsl": config.schema_,
            "dag": (
                "V19 receivers -> frozen scorer -> 600 pair join; RD1 endpoints -> composite R -> "
                "G4 exact-k amortization -> 162 exclusive homes/hists"
            ),
            "equations": (
                "sum_h b_h=DeltaB; interiors(h) are disjoint; "
                "k_g=sum event_mass/sum locus_count; b_amortized=b_h/k_g; "
                "DeltaD=100*DeltaErrors/N+Delta sqrt(10*d_pose)"
            ),
        },
        "stores_consulted": [
            "V19 sealed control/candidate archives",
            "RD1 v19c/menu/C1 receiver endpoints",
            "G4 recurrence and xi-proxy custody",
            "G4 exact-k amortization table",
            "frozen SegNet/PoseNet target cache",
        ],
        "storage_preflight": {
            "tier": "/Volumes/VertigoDataTier/pact",
            "required_free_bytes": config.required_free_bytes,
            "free_space_gate_satisfied": True,
            "status": "PASS",
        },
        "resumability": {
            "per_batch_checkpoints": True,
            "batch_size": config.batch_size,
            "all_preserved": True,
        },
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "pointer_moved": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
        "verdict": (
            "MEASURED: V19 receiver-closed n600 pair join 600/600 and RD1 "
            "exclusive amortized byte homes plus uint8 histograms 162/162; "
            "prices remain ms2r-owned."
        ),
        "verdict_scope": "local n600 evidence join only; not a contest score or promotion",
    }
    validate_campaign_evidence_join(receipt)
    output = _resolve(config.output_receipt_path)
    _atomic_json(output, receipt)
    _atomic_json(
        root / "stage_checkpoints" / "99_complete.json",
        {
            "schema": "ddm_ev1_complete.v1",
            "receipt": _output_row(output),
            "validation_counts": validate_campaign_evidence_join(receipt),
            "status": "complete",
        },
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=".omx/research/configs/ddm_ev1_campaign_evidence_joins_20260724.json",
    )
    args = parser.parse_args()
    config = _load_config(_resolve(args.config))
    output = _run(config)
    print(f"[EV1] complete {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
