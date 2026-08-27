#!/usr/bin/env python3
"""BD1 first rung: scorer-free token-topology flow on the exact BS3 body.

The decode-side mechanism is a deterministic reaction--diffusion flow.  It sees
only the born-small semantic tokens, the born-small rendered frame, generic
four-neighbour topology, and one counted four-bit rung id per pair.  SegNet is
loaded only by the measurement stage, after every candidate frame has been
retained.  It selects rung ids for the counted constraint stream; it is never a
receiver dependency.

This runner is deliberately additive and resumable.  Every candidate frame,
every scorer logit/argmax payload, every coder contender/repeat, the parsed body
tokens, and every stage receipt are retained below ``--output``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import lzma
import math
import os
import platform
import random
import shutil
import struct
import sys
import time
import zipfile
import zlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np
import torch
from scipy import ndimage

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_bs4y_stage_executor as bs4y
from experiments import ddm_po1_t4_error_feedback_pose_compensation as po1
from experiments import ddm_rb1_born_small_receiver as rb1
from experiments import ddm_rj2_joint_renderer_object_change as rj2
from experiments import ddm_wd3_scorer_aware_width_distillation as wd3
from tac.scorer import load_default_segnet

DEFAULT_OUTPUT: Final = Path("/Volumes/APDataStore/pact/ddm_bd1_decode_time_structure")
BODY_ARCHIVE: Final = Path(
    "/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/retained/body/"
    "born_small_inherited_carrier.zip"
)
BODY_ARCHIVE_BYTES: Final = 101_150
BODY_ARCHIVE_SHA256: Final = (
    "5743f0ac7e8881e970ef8ba53c4bee3fd2a7a6157d2a50d381fd609ae624fea6"
)
BODY_RESULT: Final = Path(
    "/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/BODY_RESULT.json"
)
BODY_TOKENS_SHA256: Final = (
    "2884c5701dc2b2059df0e9f8e4ee3ed81809457b127a48ad3fd3fb6f7a17152b"
)
BO2_RAW: Final = bs4y.BO2_RAW
BO2_RAW_BYTES: Final = bs4y.BO2_RAW_BYTES
BO2_RAW_SHA256: Final = bs4y.BO2_RAW_SHA256
GT_CACHE: Final = rj2.GT_CACHE
GT_CACHE_BYTES: Final = 5_078_017_610
GT_CACHE_SHA256: Final = (
    "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
)
RECEIVER_SHA256: Final = (
    "917c2f3f53e8e3fcba8b9c26dbd120e300b6273731919d3a6d5a4d008b659239"
)
SEGNET_WEIGHTS: Final = REPO / "upstream/models/segnet.safetensors"
SEGNET_WEIGHTS_BYTES: Final = 38_502_892
SEGNET_WEIGHTS_SHA256: Final = (
    "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
)
DX2_RENDERER_SOURCE: Final = rj2.DX2_RUNTIME / "cpr1/inflate.py"
DX2_RENDERER_SOURCE_BYTES: Final = 13_792
DX2_RENDERER_SOURCE_SHA256: Final = (
    "ff446edd9237148bdc898be2f8f8c4782bf231a50cf3830c4b0b21a4474a736b"
)
PAIR_COUNT: Final = 600
PAIR_SAMPLE: Final = 32
SEED: Final = 20260827
EVAL_H: Final = 384
EVAL_W: Final = 512
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
RATE_DENOMINATOR: Final = 37_545_489
AXIS: Final = "[macOS-CPU frozen-scorer advisory, seeded-stratified random n32]"
FIXED_CHARTER_DSEG_GATE: Final = 1.2316650424494517 / 200.0
CONSTRAINT_MAGIC: Final = b"BD1C"
CONSTRAINT_HEADER: Final = struct.Struct("<4sBBH32s32s")
ARCHIVE_MAGIC: Final = b"BD1A"
ARCHIVE_HEADER: Final = struct.Struct("<4sBBHIII32s32s")
CODEC_IDS: Final = {"raw": 0, "zlib9": 1, "lzma2_extreme": 2, "brotli_q11": 3}
CODEC_NAMES: Final = {value: key for key, value in CODEC_IDS.items()}


class BD1Error(RuntimeError):
    """A source pin, retained payload, receiver, or measurement differs."""


@dataclasses.dataclass(frozen=True)
class FlowConfig:
    name: str
    iterations: int = 0
    diffusion: float = 0.0
    sharpening: float = 0.0
    component_pull: float = 0.0
    boundary_push: float = 0.0
    max_delta: float = 48.0


FLOW_CONFIGS: Final = (
    FlowConfig("identity"),
    FlowConfig("diffuse_2", iterations=2, diffusion=0.25),
    FlowConfig("diffuse_8", iterations=8, diffusion=0.25),
    FlowConfig("diffuse_32", iterations=32, diffusion=0.20),
    FlowConfig("sharpen_1", iterations=1, sharpening=0.25),
    FlowConfig("sharpen_4", iterations=4, sharpening=0.125),
    FlowConfig("component_pull_0125", component_pull=0.125),
    FlowConfig("component_pull_0500", component_pull=0.50),
    FlowConfig("boundary_push_0125", boundary_push=0.125),
    FlowConfig("boundary_push_0500", boundary_push=0.50),
    FlowConfig(
        "smooth_joint",
        iterations=8,
        diffusion=0.20,
        component_pull=0.125,
        boundary_push=0.125,
    ),
    FlowConfig(
        "sharp_joint",
        iterations=4,
        sharpening=0.10,
        component_pull=0.125,
        boundary_push=0.25,
    ),
    FlowConfig(
        "strong_joint",
        iterations=32,
        diffusion=0.15,
        component_pull=1.0,
        boundary_push=0.75,
        max_delta=96.0,
    ),
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def source_tree_record(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
            or path.name == "archive.zip"
        ):
            continue
        record = file_record(path)
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
        )
    manifest = canonical_json(files)
    return {
        "path": str(root.resolve()),
        "file_count": len(files),
        "bytes": sum(int(row["bytes"]) for row in files),
        "tree_sha256": sha256_bytes(manifest),
        "files": files,
    }


def checked_file(path: Path, expected_bytes: int, expected_sha256: str, label: str) -> dict[str, Any]:
    record = file_record(path)
    if record["bytes"] != expected_bytes or record["sha256"] != expected_sha256:
        raise BD1Error(f"pinned {label} differs: {record}")
    return record


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_bytes_once(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256_bytes(payload)
    if path.exists():
        record = file_record(path)
        if record["sha256"] != expected or record["bytes"] != len(payload):
            raise BD1Error(f"immutable retained payload differs: {path}")
        return record
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_record(path)


def atomic_json_once(path: Path, value: Any) -> dict[str, Any]:
    return atomic_bytes_once(path, canonical_json(value))


def atomic_npy_once(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
    return atomic_bytes_once(path, buffer.getvalue())


def configure_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)


def _current_git_head() -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _source_binding() -> dict[str, Any]:
    return {
        "body_archive": checked_file(
            BODY_ARCHIVE, BODY_ARCHIVE_BYTES, BODY_ARCHIVE_SHA256, "body archive"
        ),
        "body_result": file_record(BODY_RESULT),
        "bo2_raw": checked_file(BO2_RAW, BO2_RAW_BYTES, BO2_RAW_SHA256, "BO2 raw"),
        "gt_cache": checked_file(GT_CACHE, GT_CACHE_BYTES, GT_CACHE_SHA256, "GT cache"),
        "receiver": checked_file(Path(rb1.__file__), 11_482, RECEIVER_SHA256, "RB1 receiver"),
        "segnet_weights": checked_file(
            SEGNET_WEIGHTS,
            SEGNET_WEIGHTS_BYTES,
            SEGNET_WEIGHTS_SHA256,
            "SegNet weights",
        ),
        "generic_renderer_source": checked_file(
            DX2_RENDERER_SOURCE,
            DX2_RENDERER_SOURCE_BYTES,
            DX2_RENDERER_SOURCE_SHA256,
            "generic renderer source",
        ),
        "generic_runtime_tree": source_tree_record(rj2.DX2_RUNTIME),
        "runner": file_record(Path(__file__)),
    }


def storage_preflight(output: Path) -> dict[str, Any]:
    candidate_frame = CAMERA_H * CAMERA_W * 3 + 128
    logits = 5 * EVAL_H * EVAL_W * 4 + 128
    argmax = EVAL_H * EVAL_W + 128
    expected = PAIR_SAMPLE * len(FLOW_CONFIGS) * (candidate_frame + logits + argmax)
    expected += 512 * 1024**2
    reserve = 8 * 1024**3
    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    result = {
        "schema": "ddm_bd1_storage_preflight.v1",
        "tier": str(output),
        "free_bytes": free,
        "expected_materialized_bytes": expected,
        "reserve_bytes": reserve,
        "required_free_bytes": expected + reserve,
        "passed": free >= expected + reserve,
        "cleanup_policy": "certify-or-block; no retained payload is deleted or moved",
    }
    checkpoint = output / "checkpoints/stage_00_storage_preflight.json"
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        invariant_keys = (
            "schema",
            "tier",
            "expected_materialized_bytes",
            "reserve_bytes",
            "required_free_bytes",
            "cleanup_policy",
        )
        if any(prior.get(key) != result.get(key) for key in invariant_keys):
            raise BD1Error("storage preflight checkpoint geometry differs")
        if not prior.get("passed"):
            raise BD1Error("retained storage preflight did not pass")
        return prior
    atomic_json_once(checkpoint, result)
    if not result["passed"]:
        raise BD1Error(f"storage preflight refused: {result}")
    return result


def _dominant_strata() -> np.ndarray:
    dominant = np.empty(PAIR_COUNT, dtype=np.int64)
    for pair in range(PAIR_COUNT):
        target = rj2.read_stored_npz_pair(
            GT_CACHE,
            "lstars",
            pair,
            expected_shape=(PAIR_COUNT, EVAL_H, EVAL_W),
            expected_dtype=np.dtype("<i8"),
        )
        dominant[pair] = int(
            np.argmax(np.bincount(target.reshape(-1), minlength=5))
        )
    return (np.arange(PAIR_COUNT, dtype=np.int64) // 60) * 5 + dominant


def stage_selection(output: Path, source_binding: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = output / "checkpoints/stage_10_selection.json"
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        if prior.get("source_binding_sha256") != sha256_bytes(canonical_json(source_binding)):
            raise BD1Error("selection checkpoint source binding differs")
        return prior
    strata = _dominant_strata()
    selected = wd3.stratified_random_indices(strata, count=PAIR_SAMPLE, seed=SEED)
    strata_record = atomic_npy_once(output / "retained/selection/strata.int64.npy", strata)
    selected_record = atomic_npy_once(
        output / "retained/selection/pair_ids.int64.npy", selected
    )
    result = {
        "schema": "ddm_bd1_selection.v1",
        "seed": SEED,
        "population_pairs": PAIR_COUNT,
        "sample_pairs": PAIR_SAMPLE,
        "selection_mode": "seeded stratified random without replacement; temporal-block x dominant-class strata",
        "prefix": False,
        "pair_ids": selected.tolist(),
        "selected_strata": strata[selected].tolist(),
        "stratum_counts_population": dict(sorted(Counter(map(int, strata)).items())),
        "stratum_counts_sample": dict(sorted(Counter(map(int, strata[selected])).items())),
        "strata": strata_record,
        "selection": selected_record,
        "source_binding_sha256": sha256_bytes(canonical_json(source_binding)),
    }
    atomic_json_once(checkpoint, result)
    return result


def _same_label_mean(value: np.ndarray, labels: np.ndarray) -> np.ndarray:
    accum = value.astype(np.float32, copy=True)
    counts = np.ones(labels.shape, dtype=np.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        y0 = max(0, -dy)
        y1 = min(EVAL_H, EVAL_H - dy)
        x0 = max(0, -dx)
        x1 = min(EVAL_W, EVAL_W - dx)
        src_y = slice(y0, y1)
        src_x = slice(x0, x1)
        dst_y = slice(y0 + dy, y1 + dy)
        dst_x = slice(x0 + dx, x1 + dx)
        match = labels[dst_y, dst_x] == labels[src_y, src_x]
        accum[dst_y, dst_x] += value[src_y, src_x] * match[..., None]
        counts[dst_y, dst_x] += match
    return accum / counts[..., None]


def _component_mean_field(value: np.ndarray, labels: np.ndarray) -> np.ndarray:
    field = np.empty_like(value, dtype=np.float32)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    for class_id in range(5):
        components, count = ndimage.label(labels == class_id, structure=structure)
        if count == 0:
            continue
        component_ids = np.arange(1, count + 1, dtype=np.int64)
        class_mask = components > 0
        for channel in range(3):
            means = np.asarray(
                ndimage.mean(value[..., channel], components, component_ids),
                dtype=np.float32,
            )
            field[..., channel][class_mask] = means[components[class_mask] - 1]
    return field


def _different_label_mean(value: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    accum = np.zeros_like(value, dtype=np.float32)
    counts = np.zeros(labels.shape, dtype=np.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        y0 = max(0, -dy)
        y1 = min(EVAL_H, EVAL_H - dy)
        x0 = max(0, -dx)
        x1 = min(EVAL_W, EVAL_W - dx)
        src_y = slice(y0, y1)
        src_x = slice(x0, x1)
        dst_y = slice(y0 + dy, y1 + dy)
        dst_x = slice(x0 + dx, x1 + dx)
        differs = labels[dst_y, dst_x] != labels[src_y, src_x]
        accum[dst_y, dst_x] += value[src_y, src_x] * differs[..., None]
        counts[dst_y, dst_x] += differs
    mean = np.divide(
        accum,
        np.maximum(counts, 1.0)[..., None],
        out=np.zeros_like(accum),
    )
    return mean, counts > 0


def solve_token_topology_flow(
    master_hwc: np.ndarray, labels: np.ndarray, config: FlowConfig
) -> np.ndarray:
    """Run one genuine token-conditioned reaction--diffusion solve."""

    if master_hwc.shape != (CAMERA_H, CAMERA_W, 3) or master_hwc.dtype != np.uint8:
        raise BD1Error("master frame geometry differs")
    if labels.shape != (EVAL_H, EVAL_W) or labels.dtype != np.uint8:
        raise BD1Error("semantic token geometry differs")
    if config.name == "identity":
        return master_hwc.copy()

    master_chw = torch.from_numpy(np.ascontiguousarray(master_hwc)).permute(2, 0, 1)[None].float()
    low = torch.nn.functional.interpolate(
        master_chw, size=(EVAL_H, EVAL_W), mode="bilinear", align_corners=False
    )[0].permute(1, 2, 0).numpy()
    value = low.copy()
    for _ in range(config.iterations):
        neighbour = _same_label_mean(value, labels)
        value += config.diffusion * (neighbour - value)
        value += config.sharpening * (value - neighbour)
        value = np.clip(value, 0.0, 255.0)
    if config.component_pull:
        component = _component_mean_field(value, labels)
        value += config.component_pull * (component - value)
    if config.boundary_push:
        other, boundary = _different_label_mean(value, labels)
        reaction = np.clip(value - other, -32.0, 32.0)
        value[boundary] += config.boundary_push * reaction[boundary]
    delta = np.clip(value - low, -config.max_delta, config.max_delta)
    delta_chw = torch.from_numpy(np.ascontiguousarray(delta)).permute(2, 0, 1)[None]
    delta_camera = torch.nn.functional.interpolate(
        delta_chw, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False
    )[0].permute(1, 2, 0).numpy()
    return np.clip(np.rint(master_hwc.astype(np.float32) + delta_camera), 0, 255).astype(np.uint8)


def _body_semantic_renderer(body_archive: bytes) -> torch.nn.Module:
    """Load the counted renderer weights from the exact nested body."""

    sections, _packet = rb1.parse_body_packet(body_archive)
    residual_module, _repack, _coefficient, _selector = po1._runtime_modules(
        rj2.DX2_RUNTIME
    )
    semantic_body = residual_module._decompress_brotli(
        sections["semantic_renderer"]
    )
    semantic_body = residual_module._ck2_uninterleave_planes(semantic_body)
    renderer_module = po1._load_renderer(rj2.DX2_RUNTIME)
    model = renderer_module.SemanticTokenRenderer(96)
    state = renderer_module.unpack_variant_semantic_or_none(
        semantic_body, model.state_dict()
    )
    if state is None:
        if len(semantic_body) != residual_module.WANS_BODY_BYTES:
            raise BD1Error("nested body semantic section length differs")
        semantic_blob = residual_module.decode_f12_wans_body(
            semantic_body, residual_module.WANS_STREAM_ORDER
        )
        records = residual_module.decode_wans1(semantic_blob)
        state = {
            record.schema.name: torch.from_numpy(
                np.ascontiguousarray(record.values, dtype=np.float32)
            )
            for record in records
        }
    if tuple(state) != tuple(model.state_dict()) or len(state) != 38:
        raise BD1Error("nested body semantic renderer state differs")
    model.load_state_dict(state, strict=True)
    return model.eval()


def _render_body_master(
    model: torch.nn.Module, labels: np.ndarray, pair: int
) -> np.ndarray:
    token_tensor = torch.from_numpy(np.ascontiguousarray(labels))[None].long()
    pair_tensor = torch.tensor([pair], dtype=torch.long)
    rendered = rb1.render_camera_uint8(model, token_tensor, pair_tensor)
    return rendered[0].permute(1, 2, 0).cpu().numpy()


def stage_solve(
    output: Path, selection: Mapping[str, Any], decoded_tokens: Path, source_binding: Mapping[str, Any]
) -> dict[str, Any]:
    checkpoint = output / "checkpoints/stage_20_solve.json"
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        if prior.get("source_binding_sha256") != sha256_bytes(canonical_json(source_binding)):
            raise BD1Error("solve checkpoint source binding differs")
        return prior
    tokens = np.memmap(decoded_tokens, mode="r", dtype=np.uint8, shape=(PAIR_COUNT, EVAL_H, EVAL_W))
    raw = np.memmap(
        BO2_RAW, mode="r", dtype=np.uint8, shape=(2 * PAIR_COUNT, CAMERA_H, CAMERA_W, 3)
    )
    renderer = _body_semantic_renderer(BODY_ARCHIVE.read_bytes())
    rows: list[dict[str, Any]] = []
    for pair in map(int, selection["pair_ids"]):
        labels = np.asarray(tokens[pair]).copy()
        master = _render_body_master(renderer, labels, pair)
        if not np.array_equal(master, np.asarray(raw[2 * pair + 1])):
            raise BD1Error(
                f"body-derived master does not reproduce the retained exact raw: pair={pair}"
            )
        pair_rows: list[dict[str, Any]] = []
        for config_id, config in enumerate(FLOW_CONFIGS):
            root = output / f"retained/solve/pair_{pair:04d}/rung_{config_id:02d}_{config.name}"
            candidate_checkpoint = root / "SOLVE.json"
            if candidate_checkpoint.exists():
                row = json.loads(candidate_checkpoint.read_text())
                if row.get("config") != dataclasses.asdict(config):
                    raise BD1Error(f"retained solve config differs: pair={pair} rung={config_id}")
                pair_rows.append(row)
                continue
            started = time.perf_counter()
            candidate = solve_token_topology_flow(master, labels, config)
            seconds = time.perf_counter() - started
            frame = atomic_npy_once(root / "frame1.uint8.npy", candidate)
            changed_values = int(np.count_nonzero(candidate != master))
            max_abs_delta = int(
                np.max(np.abs(candidate.astype(np.int16) - master.astype(np.int16)))
            )
            row = {
                "schema": "ddm_bd1_solve_member.v1",
                "pair": pair,
                "config_id": config_id,
                "config": dataclasses.asdict(config),
                "frame": frame,
                "decode_seconds": seconds,
                "changed_uint8_values_vs_body": changed_values,
                "max_abs_delta_vs_body": max_abs_delta,
                "score_claim": False,
            }
            atomic_json_once(candidate_checkpoint, row)
            pair_rows.append(row)
        rows.append({"pair": pair, "members": pair_rows})
    del raw, tokens, renderer
    result = {
        "schema": "ddm_bd1_solve_stage.v1",
        "status": "RETAINED",
        "mechanism": "token-conditioned screened reaction-diffusion with component and boundary topology",
        "extra_structure_source": (
            "the counted body token field supplies class cells and adjacency, and the counted body "
            "semantic renderer supplies the starting RGB; generic spatial continuity and component "
            "topology are the only added rule-118-free structure"
        ),
        "master_reconstructed_from_counted_body": True,
        "bo2_raw_role": "validation oracle only; byte-identical to body-derived masters before solving",
        "configs": [dataclasses.asdict(config) for config in FLOW_CONFIGS],
        "rows": rows,
        "all_materialized_payloads_retained": True,
        "source_binding_sha256": sha256_bytes(canonical_json(source_binding)),
        "score_claim": False,
    }
    atomic_json_once(checkpoint, result)
    return result


def _score_candidate(
    segnet: torch.nn.Module, frame: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float()
    pair = torch.stack((tensor, tensor), dim=0)[None]
    started = time.perf_counter()
    with torch.inference_mode():
        logits = segnet(segnet.preprocess_input(pair))
    seconds = time.perf_counter() - started
    logits_np = logits[0].cpu().numpy().astype("<f4", copy=False)
    argmax = logits_np.argmax(axis=0).astype(np.uint8)
    d_seg = float(np.mean(argmax != target))
    return logits_np, argmax, d_seg, seconds


def stage_measure(
    output: Path,
    selection: Mapping[str, Any],
    solve: Mapping[str, Any],
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = output / "checkpoints/stage_30_measurement.json"
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        if prior.get("source_binding_sha256") != sha256_bytes(canonical_json(source_binding)):
            raise BD1Error("measurement checkpoint source binding differs")
        return prior
    segnet = load_default_segnet(REPO / "upstream", device="cpu").eval()
    rows: list[dict[str, Any]] = []
    solve_by_pair = {int(row["pair"]): row for row in solve["rows"]}
    for pair in map(int, selection["pair_ids"]):
        target_i64 = rj2.read_stored_npz_pair(
            GT_CACHE,
            "lstars",
            pair,
            expected_shape=(PAIR_COUNT, EVAL_H, EVAL_W),
            expected_dtype=np.dtype("<i8"),
        )
        target = target_i64.astype(np.uint8)
        target_record = atomic_npy_once(
            output / f"retained/measurement/pair_{pair:04d}/target_argmax.uint8.npy", target
        )
        measured: list[dict[str, Any]] = []
        for member in solve_by_pair[pair]["members"]:
            config_id = int(member["config_id"])
            root = output / f"retained/measurement/pair_{pair:04d}/rung_{config_id:02d}"
            candidate_checkpoint = root / "MEASUREMENT.json"
            if candidate_checkpoint.exists():
                measured.append(json.loads(candidate_checkpoint.read_text()))
                continue
            frame = np.load(member["frame"]["path"], allow_pickle=False)
            logits, argmax, d_seg, seconds = _score_candidate(segnet, frame, target)
            logits_record = atomic_npy_once(root / "segnet_logits.float32.npy", logits)
            argmax_record = atomic_npy_once(root / "segnet_argmax.uint8.npy", argmax)
            row = {
                "schema": "ddm_bd1_measurement_member.v1",
                "pair": pair,
                "config_id": config_id,
                "config_name": member["config"]["name"],
                "d_seg": d_seg,
                "seg_s": 100.0 * d_seg,
                "scorer_seconds": seconds,
                "frame": member["frame"],
                "target_argmax": target_record,
                "segnet_logits": logits_record,
                "segnet_argmax": argmax_record,
                "axis": AXIS,
                "score_claim": False,
                "promotion_eligible": False,
            }
            atomic_json_once(candidate_checkpoint, row)
            measured.append(row)
        best = min(measured, key=lambda row: (float(row["d_seg"]), int(row["config_id"])))
        baseline = next(row for row in measured if int(row["config_id"]) == 0)
        pair_row = {
            "pair": pair,
            "baseline": baseline,
            "selected": best,
            "recovery_factor": (
                float(baseline["d_seg"]) / float(best["d_seg"])
                if float(best["d_seg"]) > 0.0
                else math.inf
            ),
            "members": measured,
        }
        atomic_json_once(
            output / f"retained/measurement/pair_{pair:04d}/PAIR_ROW.json", pair_row
        )
        rows.append(pair_row)
    baseline_dseg = float(np.mean([row["baseline"]["d_seg"] for row in rows]))
    selected_dseg = float(np.mean([row["selected"]["d_seg"] for row in rows]))
    matched_gate = baseline_dseg / 2.0
    fixed_gate_passed = selected_dseg <= FIXED_CHARTER_DSEG_GATE
    matched_gate_passed = selected_dseg <= matched_gate
    result = {
        "schema": "ddm_bd1_measurement_stage.v1",
        "status": "MEASURED",
        "axis": AXIS,
        "selection_pairs": selection["pair_ids"],
        "per_pair": rows,
        "aggregate": {
            "baseline_d_seg": baseline_dseg,
            "baseline_seg_s": 100.0 * baseline_dseg,
            "selected_d_seg": selected_dseg,
            "selected_seg_s": 100.0 * selected_dseg,
            "recovery_factor": baseline_dseg / selected_dseg if selected_dseg > 0.0 else math.inf,
            "matched_two_x_gate_d_seg": matched_gate,
            "fixed_charter_two_x_gate_d_seg": FIXED_CHARTER_DSEG_GATE,
            "matched_gate_passed": matched_gate_passed,
            "fixed_charter_gate_passed": fixed_gate_passed,
            "first_rung_passed": matched_gate_passed and fixed_gate_passed,
        },
        "scorer_loaded_at_decode": False,
        "scorer_forwards": PAIR_SAMPLE * len(FLOW_CONFIGS),
        "all_scorer_outputs_retained": True,
        "all_materialized_payloads_retained": True,
        "source_binding_sha256": sha256_bytes(canonical_json(source_binding)),
        "score_claim": False,
        "promotion_eligible": False,
    }
    atomic_json_once(checkpoint, result)
    return result


def pack_constraint_stream(rungs: Sequence[int]) -> bytes:
    if len(rungs) != PAIR_COUNT or any(not 0 <= int(value) < 16 for value in rungs):
        raise BD1Error("constraint rungs must be n600 four-bit ids")
    body = bytearray(PAIR_COUNT // 2)
    for index in range(0, PAIR_COUNT, 2):
        body[index // 2] = int(rungs[index]) | (int(rungs[index + 1]) << 4)
    body_bytes = bytes(body)
    return CONSTRAINT_HEADER.pack(
        CONSTRAINT_MAGIC,
        1,
        4,
        PAIR_COUNT,
        bytes.fromhex(BODY_ARCHIVE_SHA256),
        hashlib.sha256(body_bytes).digest(),
    ) + body_bytes


def parse_constraint_stream(payload: bytes) -> list[int]:
    if len(payload) < CONSTRAINT_HEADER.size:
        raise BD1Error("constraint stream is truncated")
    magic, version, bits, pairs, body_sha, payload_sha = CONSTRAINT_HEADER.unpack_from(payload)
    body = payload[CONSTRAINT_HEADER.size :]
    if (
        magic != CONSTRAINT_MAGIC
        or version != 1
        or bits != 4
        or pairs != PAIR_COUNT
        or body_sha.hex() != BODY_ARCHIVE_SHA256
        or len(body) != PAIR_COUNT // 2
        or hashlib.sha256(body).digest() != payload_sha
    ):
        raise BD1Error("constraint stream identity or grammar differs")
    result: list[int] = []
    for value in body:
        result.extend((value & 0x0F, value >> 4))
    return result


def _encode_constraint(codec: str, raw: bytes) -> bytes:
    if codec == "raw":
        return raw
    if codec == "zlib9":
        return zlib.compress(raw, level=9)
    if codec == "lzma2_extreme":
        return lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    if codec == "brotli_q11":
        return brotli.compress(raw, quality=11)
    raise BD1Error(f"unknown constraint codec: {codec}")


def _decode_constraint(codec: str, payload: bytes) -> bytes:
    if codec == "raw":
        return payload
    if codec == "zlib9":
        return zlib.decompress(payload)
    if codec == "lzma2_extreme":
        return lzma.decompress(payload, format=lzma.FORMAT_XZ)
    if codec == "brotli_q11":
        return brotli.decompress(payload)
    raise BD1Error(f"unknown constraint codec: {codec}")


def _zip_member(payload: bytes) -> bytes:
    destination = io.BytesIO()
    info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(destination, mode="w") as archive:
        archive.writestr(info, payload)
    return destination.getvalue()


def pack_candidate_archive(body_archive: bytes, codec: str, raw: bytes, coded: bytes) -> bytes:
    header = ARCHIVE_HEADER.pack(
        ARCHIVE_MAGIC,
        1,
        CODEC_IDS[codec],
        0,
        len(body_archive),
        len(raw),
        len(coded),
        hashlib.sha256(body_archive).digest(),
        hashlib.sha256(raw).digest(),
    )
    return _zip_member(header + body_archive + coded)


def parse_candidate_archive(payload: bytes) -> tuple[bytes, list[int], str]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.namelist() != ["p"]:
            raise BD1Error("candidate archive must contain exactly member p")
        member = archive.read("p")
    if len(member) < ARCHIVE_HEADER.size:
        raise BD1Error("candidate archive is truncated")
    magic, version, codec_id, reserved, body_n, raw_n, coded_n, body_sha, raw_sha = (
        ARCHIVE_HEADER.unpack_from(member)
    )
    if magic != ARCHIVE_MAGIC or version != 1 or reserved or codec_id not in CODEC_NAMES:
        raise BD1Error("candidate archive header differs")
    if len(member) != ARCHIVE_HEADER.size + body_n + coded_n:
        raise BD1Error("candidate archive sections do not close")
    body = member[ARCHIVE_HEADER.size : ARCHIVE_HEADER.size + body_n]
    coded = member[ARCHIVE_HEADER.size + body_n :]
    codec = CODEC_NAMES[codec_id]
    raw = _decode_constraint(codec, coded)
    if (
        len(raw) != raw_n
        or hashlib.sha256(body).digest() != body_sha
        or hashlib.sha256(raw).digest() != raw_sha
        or sha256_bytes(body) != BODY_ARCHIVE_SHA256
    ):
        raise BD1Error("candidate archive section identity differs")
    return body, parse_constraint_stream(raw), codec


def stage_packet(
    output: Path,
    measurement: Mapping[str, Any],
    solve: Mapping[str, Any],
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = output / "checkpoints/stage_40_packet.json"
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        if prior.get("source_binding_sha256") != sha256_bytes(canonical_json(source_binding)):
            raise BD1Error("packet checkpoint source binding differs")
        return prior
    selected = [0] * PAIR_COUNT
    for row in measurement["per_pair"]:
        selected[int(row["pair"])] = int(row["selected"]["config_id"])
    raw = pack_constraint_stream(selected)
    if parse_constraint_stream(raw) != selected:
        raise BD1Error("constraint stream parse-back differs")
    raw_record = atomic_bytes_once(output / "retained/constraint/constraint.raw", raw)
    coder_rows: list[dict[str, Any]] = []
    for codec in CODEC_IDS:
        coded = _encode_constraint(codec, raw)
        repeat = _encode_constraint(codec, raw)
        coded_record = atomic_bytes_once(
            output / f"retained/constraint/coders/{codec}/constraint.coded", coded
        )
        repeat_record = atomic_bytes_once(
            output / f"retained/constraint/coders/{codec}/constraint.repeat.coded", repeat
        )
        if coded_record["sha256"] != repeat_record["sha256"]:
            raise BD1Error(f"constraint codec repeat differs: {codec}")
        if _decode_constraint(codec, coded) != raw:
            raise BD1Error(f"constraint codec parse-back differs: {codec}")
        coder_rows.append(
            {"codec": codec, "coded": coded_record, "repeat": repeat_record, "parseback_exact": True}
        )
    winner = min(coder_rows, key=lambda row: (int(row["coded"]["bytes"]), row["codec"]))
    winner_payload = Path(winner["coded"]["path"]).read_bytes()
    body = BODY_ARCHIVE.read_bytes()
    archive = pack_candidate_archive(body, winner["codec"], raw, winner_payload)
    repeat_archive = pack_candidate_archive(body, winner["codec"], raw, winner_payload)
    archive_record = atomic_bytes_once(output / "retained/archive/archive.zip", archive)
    repeat_record = atomic_bytes_once(
        output / "retained/archive/archive.repeat.zip", repeat_archive
    )
    parsed_body, parsed_rungs, parsed_codec = parse_candidate_archive(archive)
    if parsed_body != body or parsed_rungs != selected or parsed_codec != winner["codec"]:
        raise BD1Error("candidate archive parse-back differs")
    rb1.parse_body_packet(parsed_body)
    mutation = bytearray(raw)
    mutation[-1] ^= 1
    mutation_refused = False
    try:
        parse_constraint_stream(bytes(mutation))
    except BD1Error:
        mutation_refused = True
    if not mutation_refused:
        raise BD1Error("constraint mutation was not refused")

    decode_seconds_by_pair: dict[int, float] = {}
    solve_by_pair = {int(row["pair"]): row for row in solve["rows"]}
    for pair, config_id in enumerate(selected):
        if pair not in solve_by_pair:
            continue
        member = solve_by_pair[pair]["members"][config_id]
        decode_seconds_by_pair[pair] = float(member["decode_seconds"])
    projected_structure_seconds_n600 = float(np.mean(list(decode_seconds_by_pair.values()))) * PAIR_COUNT
    aggregate = measurement["aggregate"]
    rate_s = 25.0 * archive_record["bytes"] / RATE_DENOMINATOR
    perfect_pose_subset_s = rate_s + float(aggregate["selected_seg_s"])
    result = {
        "schema": "ddm_bd1_packet_stage.v1",
        "status": "BYTE_CLOSED",
        "constraint_raw": raw_record,
        "constraint_coder_rows": coder_rows,
        "constraint_winner": winner,
        "archive": archive_record,
        "archive_repeat": repeat_record,
        "archive_repeat_equal": archive_record["sha256"] == repeat_record["sha256"],
        "body_parseback_exact": True,
        "constraint_parseback_exact": True,
        "constraint_mutation_refused": mutation_refused,
        "measured_nonidentity_pair_count": int(sum(value != 0 for value in selected)),
        "dense_n600_rung_capacity_bytes_raw": len(raw),
        "projected_structure_seconds_n600": projected_structure_seconds_n600,
        "ceiling_only": projected_structure_seconds_n600 > 1800.0,
        "full_integrated_decode_wall_clock": "UNMEASURED",
        "rate_s_from_exact_candidate_archive": rate_s,
        "perfect_pose_subset_s": perfect_pose_subset_s,
        "perfect_pose_subset_s_scope": (
            "DERIVED from n32 selected d_seg plus exact body+constraint archive bytes; not an n600 score"
        ),
        "all_materialized_payloads_retained": True,
        "source_binding_sha256": sha256_bytes(canonical_json(source_binding)),
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
    }
    atomic_json_once(checkpoint, result)
    return result


def stage_final(
    output: Path,
    storage: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    selection: Mapping[str, Any],
    solve: Mapping[str, Any],
    measurement: Mapping[str, Any],
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = output / "RESULT.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    passed = bool(measurement["aggregate"]["first_rung_passed"])
    result = {
        "schema": "ddm_bd1_decode_time_structure_result.v1",
        "status": "LIVE_FIRST_RUNG" if passed else "BD_CELL_CLOSED_FIRST_RUNG",
        "verdict_scope": (
            "FORMULATION: exact BS3 body plus one scorer-free token-topology reaction-diffusion "
            "family, per-pair four-bit rung selection, seeded-stratified random n32"
        ),
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "body_archive": source_binding["body_archive"],
        "selection": selection,
        "aggregate": measurement["aggregate"],
        "packet": packet,
        "storage": storage,
        "source_binding": source_binding,
        "provenance": {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "git_head": _current_git_head(),
            "python": sys.version,
            "platform": platform.platform(),
            "seed": SEED,
        },
        "mechanism": {
            "name": "token-topology reaction-diffusion rung family",
            "receiver_inputs": [
                "exact BS3 decoded semantic token field",
                "exact BS3 counted semantic-renderer state",
                "generic four-neighbour continuity/component topology",
                "counted four-bit per-pair rung stream",
            ],
            "receiver_forbidden_inputs": ["SegNet", "PoseNet", "GT argmax", "scorer logits"],
            "outer_operation": "measurement-side scorer selection, explicitly not called a solver",
            "inner_operation": "deterministic reaction-diffusion solve",
        },
        "stage_receipts": {
            "solve": file_record(output / "checkpoints/stage_20_solve.json"),
            "measurement": file_record(output / "checkpoints/stage_30_measurement.json"),
            "packet": file_record(output / "checkpoints/stage_40_packet.json"),
        },
        "all_materialized_payloads_retained": True,
    }
    atomic_json_once(result_path, result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    resume_from = args.resume_from.resolve()
    if resume_from != output / "RESULT.json":
        raise BD1Error("--resume-from must be the output RESULT.json path")
    configure_determinism(SEED)
    storage = storage_preflight(output)
    source_binding = _source_binding()
    body_bytes = BODY_ARCHIVE.read_bytes()
    rb1.parse_body_packet(body_bytes)
    decoded_tokens = output / "retained/body_decode/tokens.u8"
    decode_checkpoint = output / "checkpoints/stage_05_body_decode.json"
    if decode_checkpoint.exists():
        decode_row = json.loads(decode_checkpoint.read_text())
        checked_file(
            decoded_tokens,
            int(decode_row["tokens"]["bytes"]),
            str(decode_row["tokens"]["sha256"]),
            "retained decoded tokens",
        )
    else:
        decoded_tokens.parent.mkdir(parents=True, exist_ok=True)
        decode = rb1.decode_body_tokens(body_bytes, decoded_tokens)
        tokens_record = checked_file(
            decoded_tokens, PAIR_COUNT * EVAL_H * EVAL_W, BODY_TOKENS_SHA256, "body tokens"
        )
        decode_row = {
            "schema": "ddm_bd1_body_decode.v1",
            "body_archive": source_binding["body_archive"],
            "tokens": tokens_record,
            "decoder_result": decode,
            "parseback_exact": True,
        }
        atomic_json_once(decode_checkpoint, decode_row)
    selection = stage_selection(output, source_binding)
    solve = stage_solve(output, selection, decoded_tokens, source_binding)
    measurement = stage_measure(output, selection, solve, source_binding)
    packet = stage_packet(output, measurement, solve, source_binding)
    return stage_final(output, storage, source_binding, selection, solve, measurement, packet)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-from", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
