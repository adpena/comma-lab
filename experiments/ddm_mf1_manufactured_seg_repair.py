#!/usr/bin/env python3
"""Retained CPU instrument for the DX2 manufactured-Seg repair charter.

The instrument has two independently resumable stages:

``localize``
    Replays the retained MST1/MS9 fields without launching a scorer.  It
    localizes every final manufactured pixel by earliest observed stage,
    class, vertical band, frozen-head margin, and decoded-token boundary
    distance.  It also derives decoder-computable per-class interior RGB
    fallback prototypes and a seeded stratified-random n=32 evaluation set.

``probe``
    Measures three counted-mask, frame-local prototype pulls on only the exact
    native-stage manufactured support in that n=32 set.  Each candidate is passed through the real bilinear camera
    lift, uint8 rounding, evaluator resize, frozen CPU SegNet, and frozen CPU
    PoseNet.  Every materialized per-pair field is retained before its scalar
    metrics are admitted.  This is a scope-reduced advisory measurement, not
    a full-n600 scorer job and never a contest score.

The shipped archive and ``upstream/`` are read-only. Candidate colors use only
the already-decoded token field and native render. The GT-derived repair
address is video-derived, so its full-n600 raw and Brotli-q11 forms are retained
and counted; it is never embedded in free code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter

N = 600
H = 384
W = 512
CAM_H = 874
CAM_W = 1164
PIXELS_PER_PAIR = H * W
TOTAL_PIXELS = N * PIXELS_PER_PAIR
CLASSES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
VERTICAL_BANDS = (
    ("top_0_95", 0, 96),
    ("horizon_96_191", 96, 192),
    ("roadfield_192_287", 192, 288),
    ("nearfield_288_383", 288, 384),
)
MARGIN_BINS = (
    ("deep_le_neg1", -math.inf, -1.0),
    ("moderate_neg1_to_neg0p25", -1.0, -0.25),
    ("hairline_neg0p25_to_zero", -0.25, 0.0),
)
BOUNDARY_RADII = (0, 1, 2, 4, 9)
PROBE_CANDIDATES = (
    ("native_oracle_alpha025", 0.25),
    ("native_oracle_alpha050", 0.50),
    ("native_oracle_alpha100", 1.00),
)
SEED = 20260823
AXIS = "[macOS-CPU advisory / stratified-random n32 scope reduction]"
RATE_S_PER_BYTE = 6.658590e-7
ARCHIVE_SHA256 = "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"
TOKEN_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
GT_SHA256 = "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"
CUDA_ARGMAX_SHA256 = "e89e1ac083e5964975a1b4121cd1bc8bd91236256d6922f66c650246d7783c34"
SEGNET_SHA256 = "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
POSENET_SHA256 = "0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576"
GT_POSE6_SHA256 = "8d5cfa83df55b89493ba43b1e5386d792c836c32791666192499a089068e7eff"
CPU_RAW_SHA256 = "7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7"

REPO = Path(__file__).resolve().parents[1]
MST1 = REPO / ".omx/tmp/arm_receipts_local/ddm_mst1_manufactured_stage_split/capture_r2_local"
DEFAULT_STORE = Path("/Volumes/APDataStore/pact/ddm_mf1_manufactured_seg_repair/measurement_v3")
CPU_RAW = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/decode_r1/inflated/0.raw")
GT_POSE6 = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/"
    "direct_v6/full_n600_eval/retained/pose_vectors/gt_first6_dali_n600.npy"
)
AP_ROOT = Path("/Volumes/APDataStore/pact")


class Mf1Error(RuntimeError):
    """Fail-closed instrument error."""


def require_apdatastore_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(AP_ROOT)
    except ValueError as error:
        raise Mf1Error(f"{label} must stay under {AP_ROOT}: {resolved}") from error
    return resolved


def sha256_file(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise Mf1Error(f"required file is absent: {path}")
    fact = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if expected_sha256 is not None and fact["sha256"] != expected_sha256:
        raise Mf1Error(f"source hash drift for {path}: expected {expected_sha256}, got {fact['sha256']}")
    return fact


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return file_fact(path)


def atomic_npy(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return file_fact(path)


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return file_fact(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return file_fact(path)


def recover_incomplete_temps(store: Path) -> list[dict[str, Any]]:
    """Cold-preserve interrupted atomic writes instead of silently deleting them."""
    rows: list[dict[str, Any]] = []
    recovery = store / "incomplete_atomic_writes"
    for path in sorted(store.rglob(".*.tmp-*")) if store.exists() else []:
        if not path.is_file():
            continue
        fact = file_fact(path)
        recovery.mkdir(parents=True, exist_ok=True)
        target = recovery / f"{path.parent.name}__{path.name.lstrip('.')}"
        suffix = 0
        while target.exists():
            suffix += 1
            target = recovery / f"{path.parent.name}__{path.name.lstrip('.')}__{suffix}"
        os.replace(path, target)
        fact["cold_preserved_as"] = str(target)
        fact["reason"] = "incomplete atomic write from an interrupted run; preserved, not deleted"
        rows.append(fact)
    if rows:
        atomic_json(recovery / "RECOVERY_MANIFEST.json", {"rows": rows})
    return rows


def storage_preflight(store: Path, *, required_free_bytes: int) -> dict[str, Any]:
    store = require_apdatastore_path(store, label="store")
    store.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(store)
    if usage.free < required_free_bytes:
        raise Mf1Error(f"APDataStore free-space blocker: {usage.free} B free < {required_free_bytes} B required")
    row = {
        "schema": "ddm_mf1.storage_preflight.v1",
        "path": str(store),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "required_free_bytes": required_free_bytes,
        "retention_policy": "all scientific payloads durable; atomic scratch success-renamed; interrupted scratch cold-preserved",
        "checked_at_unix": time.time(),
    }
    atomic_json(store / "STORAGE_PREFLIGHT.json", row)
    return row


def load_packbits(path: Path) -> np.ndarray:
    packed = np.fromfile(path, dtype=np.uint8)
    if packed.size * 8 != TOTAL_PIXELS:
        raise Mf1Error(f"packed-mask size drift for {path}: {packed.size * 8} bits")
    return np.unpackbits(packed, bitorder="little").reshape(N, H, W).astype(bool, copy=False)


def token_boundary(tokens: np.ndarray) -> np.ndarray:
    boundary = np.zeros((H, W), dtype=bool)
    boundary[:, 1:] |= tokens[:, 1:] != tokens[:, :-1]
    boundary[:, :-1] |= tokens[:, :-1] != tokens[:, 1:]
    boundary[1:, :] |= tokens[1:, :] != tokens[:-1, :]
    boundary[:-1, :] |= tokens[:-1, :] != tokens[1:, :]
    return boundary


def dilate_boundary(boundary: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return boundary
    return maximum_filter(boundary, size=2 * radius + 1, mode="constant", cval=0).astype(bool, copy=False)


def chunk_bounds(pair: int) -> tuple[int, int, int]:
    start = (pair // 16) * 16
    end = min(start + 16, N)
    return start, end, pair - start


def chunk_dir(pair: int) -> Path:
    start, end, _ = chunk_bounds(pair)
    return MST1 / f"retained/chunks/{start:04d}_{end - 1:04d}"


def source_facts(*, include_probe: bool) -> dict[str, Any]:
    facts = {
        "dx2_archive": file_fact(MST1 / "retained/provenance_sources/archive.zip", expected_sha256=ARCHIVE_SHA256),
        "decoded_tokens": file_fact(
            MST1 / "retained/inputs/tokens_cpu_stage_complete.u8", expected_sha256=TOKEN_SHA256
        ),
        "dali_gt": file_fact(MST1 / "retained/inputs/gt_argmax_n600.npy", expected_sha256=GT_SHA256),
        "cuda_terminal_argmax": file_fact(
            MST1 / "retained/inputs/cuda_terminal_argmax_n600.npy",
            expected_sha256=CUDA_ARGMAX_SHA256,
        ),
        "segnet": file_fact(REPO / "upstream/models/segnet.safetensors", expected_sha256=SEGNET_SHA256),
        "posenet": file_fact(REPO / "upstream/models/posenet.safetensors", expected_sha256=POSENET_SHA256),
        "renderer_source": file_fact(MST1 / "retained/provenance_sources/renderer_source.py"),
        "mst1_result": file_fact(MST1 / "MST1_RESULT.json"),
        "instrument_source": file_fact(Path(__file__).resolve()),
    }
    if include_probe:
        facts["cpu_raw"] = file_fact(CPU_RAW, expected_sha256=CPU_RAW_SHA256)
        facts["dali_gt_pose6"] = file_fact(GT_POSE6, expected_sha256=GT_POSE6_SHA256)
    return facts


def stratified_selection(native_counts: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(SEED)
    selected: list[int] = []
    strata: list[dict[str, Any]] = []
    for time_quartile in range(4):
        lo = time_quartile * 150
        hi = lo + 150
        ordered = np.arange(lo, hi)[np.argsort(native_counts[lo:hi], kind="stable")]
        for burden_quartile, members in enumerate(np.array_split(ordered, 4)):
            if members.size < 2:
                raise Mf1Error("stratification cell unexpectedly has fewer than two members")
            chosen = np.sort(rng.choice(members, size=2, replace=False))
            selected.extend(int(value) for value in chosen)
            strata.append(
                {
                    "time_quartile": time_quartile,
                    "pair_range": [lo, hi - 1],
                    "native_manufactured_rank_quartile": burden_quartile,
                    "cell_size": int(members.size),
                    "chosen_pairs": chosen.tolist(),
                    "chosen_native_manufactured_counts": native_counts[chosen].tolist(),
                }
            )
    array = np.asarray(sorted(selected), dtype=np.int16)
    if array.size != 32 or np.unique(array).size != 32:
        raise Mf1Error(f"selection closure failed: {array.tolist()}")
    return array, strata


@dataclass
class PrototypeAccumulator:
    sums: np.ndarray
    counts: np.ndarray

    @classmethod
    def create(cls) -> PrototypeAccumulator:
        return cls(
            sums=np.zeros((len(BOUNDARY_RADII), 5, 3), dtype=np.float64),
            counts=np.zeros((len(BOUNDARY_RADII), 5), dtype=np.int64),
        )

    def add(self, native_chw: np.ndarray, tokens: np.ndarray, boundary: np.ndarray) -> None:
        rgb = np.moveaxis(native_chw, 0, -1).astype(np.float64, copy=False)
        for radius_index, radius in enumerate(BOUNDARY_RADII):
            interior = ~dilate_boundary(boundary, radius)
            for class_index in range(5):
                mask = interior & (tokens == class_index)
                count = int(mask.sum())
                if count:
                    self.sums[radius_index, class_index] += rgb[mask].sum(axis=0)
                    self.counts[radius_index, class_index] += count

    def finish(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        prototypes = np.zeros((5, 3), dtype=np.float32)
        choices: list[dict[str, Any]] = []
        for class_index, class_name in enumerate(CLASSES):
            selected_radius_index = None
            for radius_index in reversed(range(len(BOUNDARY_RADII))):
                if self.counts[radius_index, class_index] >= 10_000:
                    selected_radius_index = radius_index
                    break
            if selected_radius_index is None:
                raise Mf1Error(f"no decoder-derived prototype support for {class_name}")
            count = int(self.counts[selected_radius_index, class_index])
            prototypes[class_index] = (self.sums[selected_radius_index, class_index] / count).astype(np.float32)
            choices.append(
                {
                    "class": class_name,
                    "interior_radius": BOUNDARY_RADII[selected_radius_index],
                    "support_pixels": count,
                    "rgb": prototypes[class_index].astype(float).tolist(),
                    "derivation": "global mean of native RGB where decoded token has no unlike token inside selected Chebyshev radius",
                }
            )
        return prototypes, choices


def margin_bin_counts(values: np.ndarray) -> dict[str, int]:
    rows: dict[str, int] = {}
    for name, low, high in MARGIN_BINS:
        mask = values <= high if math.isinf(low) else (values > low) & (values <= high)
        rows[name] = int(mask.sum())
    return rows


def localize(store: Path, *, resume_from: Path | None = None) -> None:
    storage_preflight(store, required_free_bytes=1_500_000_000)
    recover_incomplete_temps(store)
    facts = source_facts(include_probe=False)

    tokens = np.memmap(
        MST1 / "retained/inputs/tokens_cpu_stage_complete.u8",
        dtype=np.uint8,
        mode="r",
        shape=(N, H, W),
    )
    gt = np.load(MST1 / "retained/inputs/gt_argmax_n600.npy", mmap_mode="r")
    terminal = np.load(MST1 / "retained/inputs/cuda_terminal_argmax_n600.npy", mmap_mode="r")
    native_argmax = np.load(MST1 / "retained/assembled/argmax_native_n600.npy", mmap_mode="r")
    preuint8_argmax = np.load(MST1 / "retained/assembled/argmax_preuint8_n600.npy", mmap_mode="r")
    uint8_argmax = np.load(MST1 / "retained/assembled/argmax_uint8_n600.npy", mmap_mode="r")

    mask_root = MST1 / "retained/attribution_masks"
    earliest_paths = {
        "native_render_head": mask_root / "earliest_manufactured_native_render_head.n600.packbits",
        "preuint8_roundtrip_head": mask_root / "earliest_manufactured_preuint8_roundtrip_head.n600.packbits",
        "uint8_roundtrip_head": mask_root / "earliest_manufactured_uint8_roundtrip_head.n600.packbits",
        "cpu_to_cuda_terminal_unseparated_head": mask_root
        / "earliest_manufactured_cpu_to_cuda_terminal_unseparated_head.n600.packbits",
    }
    earliest = {name: load_packbits(path) for name, path in earliest_paths.items()}
    expected_stage_counts = {
        "native_render_head": 16_917,
        "preuint8_roundtrip_head": 4_030,
        "uint8_roundtrip_head": 544,
        "cpu_to_cuda_terminal_unseparated_head": 2,
    }
    observed_stage_counts = {name: int(mask.sum()) for name, mask in earliest.items()}
    if observed_stage_counts != expected_stage_counts:
        raise Mf1Error(f"MST1 stage gate drift: expected {expected_stage_counts}, got {observed_stage_counts}")
    final_manufactured = (np.asarray(terminal) != np.asarray(gt)) & (np.asarray(tokens) == np.asarray(gt))
    if int(final_manufactured.sum()) != 21_493:
        raise Mf1Error("MS9 final manufactured gate did not reproduce 21,493")
    if not np.array_equal(np.logical_or.reduce(list(earliest.values())), final_manufactured):
        raise Mf1Error("earliest-stage masks do not partition final manufactured support")

    stage_argmax = {
        "native_render_head": native_argmax,
        "preuint8_roundtrip_head": preuint8_argmax,
        "uint8_roundtrip_head": uint8_argmax,
    }
    stage_logits_name = {
        "native_render_head": "logits_native.float32.npy",
        "preuint8_roundtrip_head": "logits_preuint8.float32.npy",
        "uint8_roundtrip_head": "logits_uint8.float32.npy",
    }
    stage_names = tuple(stage_logits_name)
    source_binding = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
    checkpoint_path = resume_from if resume_from is not None else store / "LOCALIZE_CHECKPOINT.npz"
    if resume_from is not None and not checkpoint_path.is_file():
        raise Mf1Error(f"explicit localization resume checkpoint is absent: {checkpoint_path}")
    checkpoint_path = require_apdatastore_path(checkpoint_path, label="localization checkpoint")
    if checkpoint_path.parent != store.resolve():
        raise Mf1Error("localization --resume-from must name the checkpoint inside --store")
    event_dtypes = {
        "stage": np.uint8,
        "class": np.uint8,
        "band": np.uint8,
        "margin": np.float32,
        "predicted": np.uint8,
        "boundary_r0": np.uint8,
        "boundary_r9": np.uint8,
    }
    if checkpoint_path.is_file():
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            if str(checkpoint["source_binding_sha256"]) != source_binding:
                raise Mf1Error("localization checkpoint source binding drifted")
            next_start = int(checkpoint["next_start"])
            boundary_payloads = {
                radius: bytearray(checkpoint[f"boundary_r{radius}_packbits"].tobytes()) for radius in BOUNDARY_RADII
            }
            counts = checkpoint["boundary_population_counts"]
            boundary_counts = {radius: int(counts[index]) for index, radius in enumerate(BOUNDARY_RADII)}
            stage_counts = checkpoint["boundary_stage_counts"]
            boundary_stage_counts = {
                stage: {
                    radius: int(stage_counts[stage_index, radius_index])
                    for radius_index, radius in enumerate(BOUNDARY_RADII)
                }
                for stage_index, stage in enumerate(earliest)
            }
            native_counts = checkpoint["native_counts"].astype(np.int64, copy=True)
            prototypes = PrototypeAccumulator(
                sums=checkpoint["prototype_sums"].astype(np.float64, copy=True),
                counts=checkpoint["prototype_counts"].astype(np.int64, copy=True),
            )
            events = {name: [checkpoint[f"event_{name}"].copy()] for name in event_dtypes}
        if next_start < 0 or next_start > N or (next_start != N and next_start % 16):
            raise Mf1Error(f"invalid localization resume boundary: {next_start}")
        print(f"resuming localization at pair {next_start}", flush=True)
    else:
        next_start = 0
        boundary_payloads = {radius: bytearray(TOTAL_PIXELS // 8) for radius in BOUNDARY_RADII}
        boundary_counts = dict.fromkeys(BOUNDARY_RADII, 0)
        boundary_stage_counts = {stage: dict.fromkeys(BOUNDARY_RADII, 0) for stage in earliest}
        native_counts = np.zeros(N, dtype=np.int64)
        prototypes = PrototypeAccumulator.create()
        events = {name: [] for name in event_dtypes}

    for start in range(next_start, N, 16):
        end = min(start + 16, N)
        directory = MST1 / f"retained/chunks/{start:04d}_{end - 1:04d}"
        native_rgb = np.load(directory / "native_rgb.float32.npy", mmap_mode="r")
        logits = {stage: np.load(directory / filename, mmap_mode="r") for stage, filename in stage_logits_name.items()}
        for offset, pair in enumerate(range(start, end)):
            pair_tokens = np.asarray(tokens[pair])
            boundary = token_boundary(pair_tokens)
            prototypes.add(np.asarray(native_rgb[offset]), pair_tokens, boundary)
            native_counts[pair] = int(earliest["native_render_head"][pair].sum())
            bit_lo = pair * PIXELS_PER_PAIR // 8
            bit_hi = bit_lo + PIXELS_PER_PAIR // 8
            expanded: dict[int, np.ndarray] = {}
            for radius in BOUNDARY_RADII:
                expanded[radius] = dilate_boundary(boundary, radius)
                packed = np.packbits(expanded[radius].reshape(-1), bitorder="little")
                boundary_payloads[radius][bit_lo:bit_hi] = packed.tobytes()
                boundary_counts[radius] += int(expanded[radius].sum())
                for stage, stage_mask in earliest.items():
                    boundary_stage_counts[stage][radius] += int((expanded[radius] & stage_mask[pair]).sum())

            pair_gt = np.asarray(gt[pair], dtype=np.int64)
            for stage in stage_logits_name:
                stage_mask = earliest[stage][pair]
                if not stage_mask.any():
                    continue
                pair_logits = np.asarray(logits[stage][offset])
                target_logit = np.take_along_axis(pair_logits, pair_gt[None], axis=0)[0]
                max_other = np.max(
                    np.where(
                        np.arange(5)[:, None, None] == pair_gt[None],
                        -np.inf,
                        pair_logits,
                    ),
                    axis=0,
                )
                signed_margin = target_logit - max_other
                predicted = np.asarray(stage_argmax[stage][pair])
                ys, xs = np.nonzero(stage_mask)
                count = ys.size
                events["stage"].append(np.full(count, stage_names.index(stage), dtype=np.uint8))
                events["class"].append(pair_gt[ys, xs].astype(np.uint8, copy=True))
                events["band"].append((ys // 96).astype(np.uint8, copy=False))
                events["margin"].append(signed_margin[ys, xs].astype(np.float32, copy=True))
                events["predicted"].append(predicted[ys, xs].astype(np.uint8, copy=True))
                events["boundary_r0"].append(expanded[0][ys, xs].astype(np.uint8, copy=True))
                events["boundary_r9"].append(expanded[9][ys, xs].astype(np.uint8, copy=True))

        event_values = {
            name: np.concatenate(values) if values else np.empty(0, dtype=event_dtypes[name])
            for name, values in events.items()
        }
        atomic_npz(
            checkpoint_path,
            source_binding_sha256=np.asarray(source_binding),
            next_start=np.asarray(end, dtype=np.int16),
            **{
                f"boundary_r{radius}_packbits": np.frombuffer(payload, dtype=np.uint8)
                for radius, payload in boundary_payloads.items()
            },
            boundary_population_counts=np.asarray(
                [boundary_counts[radius] for radius in BOUNDARY_RADII], dtype=np.int64
            ),
            boundary_stage_counts=np.asarray(
                [[boundary_stage_counts[stage][radius] for radius in BOUNDARY_RADII] for stage in earliest],
                dtype=np.int64,
            ),
            native_counts=native_counts,
            prototype_sums=prototypes.sums,
            prototype_counts=prototypes.counts,
            **{f"event_{name}": value for name, value in event_values.items()},
        )
        events = {name: [value] for name, value in event_values.items()}
        print(f"localized and checkpointed pairs {end}/{N}", flush=True)

    prototype_values, prototype_choices = prototypes.finish()
    event_values = {
        name: np.concatenate(values) if values else np.empty(0, dtype=event_dtypes[name])
        for name, values in events.items()
    }
    if event_values["stage"].size != sum(expected_stage_counts[stage] for stage in stage_names):
        raise Mf1Error("localization event checkpoint does not close the non-terminal stage denominator")
    for stage_index, stage in enumerate(stage_names):
        count = int((event_values["stage"] == stage_index).sum())
        if count != expected_stage_counts[stage]:
            raise Mf1Error(f"localization event checkpoint stage drift for {stage}: {count}")
    expected_native_counts = earliest["native_render_head"].reshape(N, -1).sum(axis=1).astype(np.int64)
    if not np.array_equal(native_counts, expected_native_counts):
        raise Mf1Error("localization checkpoint native per-pair counts drifted")
    for radius, payload in boundary_payloads.items():
        unpacked = (
            np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little").reshape(N, H, W).astype(bool)
        )
        if int(unpacked.sum()) != boundary_counts[radius]:
            raise Mf1Error(f"localization checkpoint boundary population drift at radius {radius}")
        for stage, stage_mask in earliest.items():
            observed = int((unpacked & stage_mask).sum())
            if observed != boundary_stage_counts[stage][radius]:
                raise Mf1Error(f"localization checkpoint boundary/stage join drift for {stage} radius {radius}")
    cluster_rows: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(stage_names):
        for class_index, class_name in enumerate(CLASSES):
            for band_index, (band_name, _, _) in enumerate(VERTICAL_BANDS):
                cell = (
                    (event_values["stage"] == stage_index)
                    & (event_values["class"] == class_index)
                    & (event_values["band"] == band_index)
                )
                if not cell.any():
                    continue
                margins = event_values["margin"][cell]
                predicted_counts = np.bincount(event_values["predicted"][cell], minlength=5)
                dominant_index = int(np.argmax(predicted_counts))
                cluster_rows.append(
                    {
                        "stage": stage,
                        "class_index": class_index,
                        "class": class_name,
                        "spatial_band": band_name,
                        "pixel_count": int(cell.sum()),
                        "exact_token_boundary_count": int(event_values["boundary_r0"][cell].sum()),
                        "within_radius9_count": int(event_values["boundary_r9"][cell].sum()),
                        "margin_bins": margin_bin_counts(margins),
                        "signed_target_margin_quantiles": {
                            str(quantile): float(np.quantile(margins, quantile))
                            for quantile in (0.0, 0.1, 0.5, 0.9, 1.0)
                        },
                        "dominant_wrong_class": CLASSES[dominant_index],
                        "dominant_wrong_class_count": int(predicted_counts[dominant_index]),
                        "predicted_class_counts": {
                            CLASSES[index]: int(value) for index, value in enumerate(predicted_counts)
                        },
                        "new_vs_amplified": "NEW_FROM_CORRECT_TRANSMITTED_LABEL",
                        "amplified_existing_error_count": 0,
                    }
                )

    terminal_rows: list[dict[str, Any]] = []
    terminal_mask = earliest["cpu_to_cuda_terminal_unseparated_head"]
    for pair, y, x in np.argwhere(terminal_mask):
        target = int(gt[pair, y, x])
        terminal_rows.append(
            {
                "stage": "cpu_to_cuda_terminal_unseparated_head",
                "pair": int(pair),
                "y": int(y),
                "x": int(x),
                "class_index": target,
                "class": CLASSES[target],
                "spatial_band": next(name for name, lo, hi in VERTICAL_BANDS if lo <= y < hi),
                "pixel_count": 1,
                "margin": "UNMEASURED: retained CUDA logits are absent",
                "new_vs_amplified": "NEW_FROM_CORRECT_TRANSMITTED_LABEL",
                "amplified_existing_error_count": 0,
            }
        )

    selected, strata = stratified_selection(native_counts)
    boundary_facts = {}
    boundary_dir = store / "retained/boundary_masks"
    for radius, payload in boundary_payloads.items():
        path = boundary_dir / f"decoded_token_boundary_chebyshev_r{radius}.n600.packbits"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        boundary_facts[str(radius)] = file_fact(path)

    import brotli

    native_repair_raw = bytes(np.packbits(earliest["native_render_head"].reshape(-1), bitorder="little"))
    native_repair_compressed = brotli.compress(native_repair_raw, mode=brotli.MODE_GENERIC, quality=11)
    native_repair_repeat = brotli.compress(native_repair_raw, mode=brotli.MODE_GENERIC, quality=11)
    if native_repair_compressed != native_repair_repeat:
        raise Mf1Error("native repair-mask Brotli repeat is not deterministic")
    if brotli.decompress(native_repair_compressed) != native_repair_raw:
        raise Mf1Error("native repair-mask Brotli decode does not reproduce the raw mask")
    candidate_payload = {
        "raw": atomic_bytes(store / "retained/native_manufactured_mask.n600.packbits", native_repair_raw),
        "compressed": atomic_bytes(
            store / "retained/native_manufactured_mask.n600.brotli_q11", native_repair_compressed
        ),
        "compressed_repeat": atomic_bytes(
            store / "retained/native_manufactured_mask.n600.repeat.brotli_q11", native_repair_repeat
        ),
    }

    prototype_fact = atomic_npy(store / "retained/decoder_derived_prototypes.float32.npy", prototype_values)
    selection_fact = atomic_npz(
        store / "retained/stratified_random_n32_selection.npz",
        pair_indices=selected,
        native_manufactured_counts=native_counts[selected],
    )
    clusters_fact = atomic_json(
        store / "CLUSTERS.json",
        {
            "schema": "ddm_mf1.clusters.v1",
            "axis": "[retained exact field replay; macOS-CPU advisory intermediate logits]",
            "cluster_rows": cluster_rows,
            "terminal_rows": terminal_rows,
        },
    )
    stage_margin_summary = {}
    for stage_index, stage in enumerate(stage_names):
        values = event_values["margin"][event_values["stage"] == stage_index]
        stage_margin_summary[stage] = {
            "count": int(values.size),
            "bins": margin_bin_counts(values),
            "quantiles": {
                str(quantile): float(np.quantile(values, quantile))
                for quantile in (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
            },
        }

    localization = {
        "schema": "ddm_mf1.localization.v1",
        "axis": "[contest-CUDA T4 exact field support + macOS-CPU advisory intermediate logits]",
        "score_claim": False,
        "n_pairs": N,
        "denominator_pixels": TOTAL_PIXELS,
        "source_facts": facts,
        "final_manufactured_pixels": 21_493,
        "stage_counts": observed_stage_counts,
        "stage_margin_summary": stage_margin_summary,
        "boundary": {
            "definition": "decoded token has an unlike four-neighbor; radius uses Chebyshev dilation",
            "population_counts": boundary_counts,
            "earliest_stage_counts": boundary_stage_counts,
            "payloads": boundary_facts,
        },
        "prototypes": {
            "rule": "decoder-derived from current decoded token field and native render only; no stored video side information",
            "choices": prototype_choices,
            "payload": prototype_fact,
        },
        "selection": {
            "seed": SEED,
            "mode": "seeded stratified-random by time quartile x native-manufactured burden rank quartile",
            "pair_indices": selected.tolist(),
            "strata": strata,
            "payload": selection_fact,
        },
        "mechanism_source_adjudication": {
            "paint_ordering": "REFUTED_AT_SOURCE: SemanticTokenRenderer is one simultaneous embedding/CNN forward; no paint loop or overdraw exists",
            "fixed_prototype_color": "REFUTED_AT_SOURCE_AS_CURRENT_MECHANISM: the current renderer has learned token/frame embeddings, four dilated residual blocks, and a learned RGB head; no fixed palette exists",
            "native_boundary_context_spill": "CONFIRMED_AT_SOURCE_IF_JOIN_ENRICHED: four dilated 3x3 blocks plus 3x3 head mix unlike-token context over nominal radius 9",
            "pre_R_subpixel_placement": "CONFIRMED_OPERATOR_FOR_PREUINT8_STAGE: only bilinear lift/downsample lies between native and preuint8 observations; R is frozen and net-repairing globally",
            "uint8_amplitude_floor": "CONFIRMED_OPERATOR_FOR_UINT8_STAGE: round-to-uint8 is the only operation between preuint8 and uint8 observations",
        },
        "candidate_family": {
            "name": "counted oracle native-manufactured mask + decoder-derived frame-local interior-prototype pull",
            "address_scope": "exact 16,917-pixel native-stage manufactured support; oracle upper bound, not free addressing",
            "payload": candidate_payload,
            "archive_byte_delta": "UNMEASURED until receiver/container integration; compressed payload bytes are a strict lower bound",
            "variants": [{"tag": tag, "alpha": alpha} for tag, alpha in PROBE_CANDIDATES],
            "probe_status": "PENDING",
        },
    }
    result_fact = atomic_json(store / "LOCALIZATION.json", localization)
    manifest = {
        "schema": "ddm_mf1.localization_manifest.v1",
        "result": result_fact,
        "clusters": clusters_fact,
        "boundary_payloads": boundary_facts,
        "prototype_payload": prototype_fact,
        "selection_payload": selection_fact,
        "localize_checkpoint": file_fact(checkpoint_path),
        "candidate_payload": candidate_payload,
        "complete": True,
    }
    atomic_json(store / "LOCALIZATION_MANIFEST.json", manifest)


def load_scorers() -> tuple[Any, Any, Any]:
    import torch
    from safetensors.torch import load_file

    upstream = REPO / "upstream"
    sys.path.insert(0, str(upstream))
    try:
        import modules as upstream_modules
    finally:
        sys.path.pop(0)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    segnet = upstream_modules.SegNet().eval()
    posenet = upstream_modules.PoseNet().eval()
    segnet.load_state_dict(load_file(str(upstream / "models/segnet.safetensors"), device="cpu"))
    posenet.load_state_dict(load_file(str(upstream / "models/posenet.safetensors"), device="cpu"))
    return torch, segnet, posenet


def pose6_from_pair(torch: Any, posenet: Any, pair_hwc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pair = torch.from_numpy(pair_hwc.copy()).permute(0, 3, 1, 2).float()[None]
    with torch.inference_mode():
        pose_input = posenet.preprocess_input(pair)
        output = posenet(pose_input)["pose"][0, :6]
    return pose_input[0].detach().cpu().numpy().astype(np.float32), output.detach().cpu().numpy().astype(np.float32)


def frame_local_prototypes(
    native_hwc: np.ndarray,
    tokens: np.ndarray,
    boundary: np.ndarray,
    global_fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive class colors from this decoded frame, falling back to decoded-global means."""
    values = global_fallback.astype(np.float32, copy=True)
    support = np.zeros(5, dtype=np.int64)
    selected_radius = np.full(5, -1, dtype=np.int8)
    expanded = {radius: dilate_boundary(boundary, radius) for radius in BOUNDARY_RADII}
    for class_index in range(5):
        for radius in reversed(BOUNDARY_RADII):
            mask = (tokens == class_index) & ~expanded[radius]
            count = int(mask.sum())
            if count >= 16:
                values[class_index] = native_hwc[mask].mean(axis=0, dtype=np.float64).astype(np.float32)
                support[class_index] = count
                selected_radius[class_index] = radius
                break
    return values, support, selected_radius


def candidate_fields(
    torch: Any,
    segnet: Any,
    posenet: Any,
    *,
    native_chw: np.ndarray,
    tokens: np.ndarray,
    slave_hwc: np.ndarray,
    global_prototypes: np.ndarray,
    edit_mask: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    from torch.nn import functional

    boundary = token_boundary(tokens)
    native_hwc = np.moveaxis(native_chw, 0, -1).astype(np.float32, copy=True)
    prototypes, prototype_support, prototype_radius = frame_local_prototypes(
        native_hwc,
        tokens,
        boundary,
        global_prototypes,
    )
    target_rgb = prototypes[tokens]
    native_hwc[edit_mask] = (1.0 - alpha) * native_hwc[edit_mask] + alpha * target_rgb[edit_mask]
    native = torch.from_numpy(np.moveaxis(native_hwc, -1, 0).copy())[None]
    camera_float = functional.interpolate(native, size=(CAM_H, CAM_W), mode="bilinear", align_corners=False).clamp(
        0.0, 255.0
    )
    camera_u8 = camera_float.round().to(torch.uint8)
    with torch.inference_mode():
        resized = segnet.preprocess_input(camera_u8.float()[:, None])
        logits = segnet(resized)
        argmax = logits.argmax(dim=1)
    master_hwc = camera_u8[0].permute(1, 2, 0).cpu().numpy()
    pair_hwc = np.stack([slave_hwc, master_hwc], axis=0)
    pose_input, pose6 = pose6_from_pair(torch, posenet, pair_hwc)
    return {
        "native_rgb_float32": native[0].cpu().numpy().astype(np.float32),
        "camera_preuint8_rgb_float32": camera_float[0].cpu().numpy().astype(np.float32),
        "camera_uint8_rgb": camera_u8[0].cpu().numpy().astype(np.uint8),
        "evaluator_resized_rgb_float32": resized[0].detach().cpu().numpy().astype(np.float32),
        "segnet_logits_float32": logits[0].detach().cpu().numpy().astype(np.float32),
        "segnet_argmax_uint8": argmax[0].detach().cpu().numpy().astype(np.uint8),
        "pose_input_yuv6_float32": pose_input,
        "pose6_float32": pose6,
        "edit_mask_uint8": edit_mask.astype(np.uint8),
        "prototype_rgb_float32": prototypes,
        "prototype_support_int64": prototype_support,
        "prototype_radius_int8": prototype_radius,
    }


def existing_pair_fact(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    fact = file_fact(path, expected_sha256=expected_sha256)
    with np.load(path, allow_pickle=False) as values:
        fact["keys"] = sorted(values.files)
    return fact


def probe(store: Path, *, resume_from: Path | None = None) -> None:
    storage_preflight(store, required_free_bytes=4_000_000_000)
    recover_incomplete_temps(store)
    localization_path = store / "LOCALIZATION.json"
    if not localization_path.is_file():
        raise Mf1Error("localize stage must complete before probe")
    localization = json.loads(localization_path.read_text())
    facts = source_facts(include_probe=True)
    localization_instrument_sha = localization["source_facts"]["instrument_source"]["sha256"]
    if localization_instrument_sha != facts["instrument_source"]["sha256"]:
        raise Mf1Error("instrument source changed after localization; rerun localize before probe")
    prototype_path = store / "retained/decoder_derived_prototypes.float32.npy"
    selection_path = store / "retained/stratified_random_n32_selection.npz"
    prototypes = np.load(prototype_path)
    with np.load(selection_path) as selection:
        pairs = selection["pair_indices"].astype(np.int64)
    if pairs.size != 32:
        raise Mf1Error("probe selection is not n=32")

    tokens = np.memmap(
        MST1 / "retained/inputs/tokens_cpu_stage_complete.u8",
        dtype=np.uint8,
        mode="r",
        shape=(N, H, W),
    )
    gt = np.load(MST1 / "retained/inputs/gt_argmax_n600.npy", mmap_mode="r")
    terminal = np.load(MST1 / "retained/inputs/cuda_terminal_argmax_n600.npy", mmap_mode="r")
    gt_pose6 = np.load(GT_POSE6, mmap_mode="r")
    raw = np.memmap(CPU_RAW, dtype=np.uint8, mode="r", shape=(N * 2, CAM_H, CAM_W, 3))
    final_manufactured = (np.asarray(terminal) != np.asarray(gt)) & (np.asarray(tokens) == np.asarray(gt))
    native_manufactured = load_packbits(
        MST1 / "retained/attribution_masks/earliest_manufactured_native_render_head.n600.packbits"
    )
    candidate_payload_bytes = int(localization["candidate_family"]["payload"]["compressed"]["bytes"])
    probe_source_binding = hashlib.sha256(
        json.dumps(
            {
                "source_facts": facts,
                "localization_sha256": sha256_file(localization_path),
                "prototype": file_fact(prototype_path),
                "selection": file_fact(selection_path),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    torch, segnet, posenet = load_scorers()
    baseline_dir = store / "probe_n32/baseline/pairs"
    candidate_root = store / "probe_n32/candidates"
    checkpoint_path = resume_from if resume_from is not None else store / "PROBE_CHECKPOINT.json"
    if resume_from is not None and not checkpoint_path.is_file():
        raise Mf1Error(f"explicit probe resume checkpoint is absent: {checkpoint_path}")
    checkpoint_path = require_apdatastore_path(checkpoint_path, label="probe checkpoint")
    if checkpoint_path.parent != store.resolve():
        raise Mf1Error("probe --resume-from must name the checkpoint inside --store")
    checkpoint = (
        json.loads(checkpoint_path.read_text())
        if checkpoint_path.is_file()
        else {
            "schema": "ddm_mf1.probe_checkpoint.v1",
            "instrument_sha256": facts["instrument_source"]["sha256"],
            "source_binding_sha256": probe_source_binding,
            "baseline": [],
            "candidates": {},
        }
    )
    if checkpoint.get("instrument_sha256") != facts["instrument_source"]["sha256"]:
        raise Mf1Error("probe checkpoint belongs to another instrument source; refusing mixed-source resume")
    if checkpoint.get("source_binding_sha256") != probe_source_binding:
        raise Mf1Error("probe checkpoint source binding drifted; refusing mixed-input resume")
    baseline_admitted = {int(row["pair"]) for row in checkpoint["baseline"]}
    baseline_expected = {int(row["pair"]): row["payload"]["sha256"] for row in checkpoint["baseline"]}

    for pair in pairs:
        pair = int(pair)
        output = baseline_dir / f"pair_{pair:04d}.npz"
        if output.is_file():
            fact = existing_pair_fact(output, expected_sha256=baseline_expected.get(pair))
            if pair not in baseline_admitted:
                checkpoint["baseline"].append({"pair": pair, "payload": fact})
                baseline_admitted.add(pair)
                atomic_json(checkpoint_path, checkpoint)
            continue
        _, _, offset = chunk_bounds(pair)
        directory = chunk_dir(pair)
        cached_logits = np.asarray(np.load(directory / "logits_uint8.float32.npy", mmap_mode="r")[offset]).astype(
            np.float32, copy=True
        )
        cached_argmax = np.asarray(np.load(directory / "argmax_uint8.uint8.npy", mmap_mode="r")[offset]).astype(
            np.uint8, copy=True
        )
        native = np.asarray(np.load(directory / "native_rgb.float32.npy", mmap_mode="r")[offset]).astype(
            np.float32, copy=True
        )
        pair_raw = np.asarray(raw[2 * pair : 2 * pair + 2]).copy()
        fields = candidate_fields(
            torch,
            segnet,
            posenet,
            native_chw=native,
            tokens=np.asarray(tokens[pair]),
            slave_hwc=pair_raw[0],
            global_prototypes=prototypes,
            edit_mask=np.zeros((H, W), dtype=bool),
            alpha=0.0,
        )
        replay_master = np.moveaxis(fields["camera_uint8_rgb"], 0, -1)
        if not np.array_equal(replay_master, pair_raw[1]):
            difference = int(np.count_nonzero(replay_master != pair_raw[1]))
            raise Mf1Error(f"pair {pair} unchanged renderer replay differs from retained raw by {difference} channels")
        if not np.array_equal(fields["segnet_argmax_uint8"], cached_argmax):
            difference = int(np.count_nonzero(fields["segnet_argmax_uint8"] != cached_argmax))
            raise Mf1Error(f"pair {pair} unchanged CPU SegNet replay differs from MST1 by {difference} pixels")
        max_difference = float(np.max(np.abs(fields["segnet_logits_float32"] - cached_logits)))
        # MST1 evaluates cached logits in batch-16 chunks; this matched candidate
        # instrument evaluates one pair at a time. RGB and argmax must remain
        # exact, while the batch-geometry float diagnostic admits only this
        # bounded numerical envelope.
        if max_difference > 1.0e-4:
            raise Mf1Error(f"pair {pair} unchanged CPU logits differ from MST1; max abs {max_difference}")
        fact = atomic_npz(
            output,
            **fields,
            raw_pair_uint8=pair_raw,
            gt_argmax_uint8=np.asarray(gt[pair]).astype(np.uint8, copy=True),
            decoded_tokens_uint8=np.asarray(tokens[pair]).astype(np.uint8, copy=True),
            final_manufactured_mask_uint8=final_manufactured[pair].astype(np.uint8),
            gt_pose6_float32=np.asarray(gt_pose6[pair]).astype(np.float32, copy=True),
        )
        checkpoint["baseline"].append({"pair": pair, "payload": fact})
        baseline_admitted.add(pair)
        atomic_json(checkpoint_path, checkpoint)
        print(f"baseline retained pair {pair}", flush=True)

    for tag, alpha in PROBE_CANDIDATES:
        admitted = {int(row["pair"]) for row in checkpoint["candidates"].get(tag, [])}
        expected = {int(row["pair"]): row["payload"]["sha256"] for row in checkpoint["candidates"].get(tag, [])}
        for pair in pairs:
            pair = int(pair)
            output = candidate_root / tag / "pairs" / f"pair_{pair:04d}.npz"
            if output.is_file():
                existing_pair_fact(output, expected_sha256=expected.get(pair))
                if pair not in admitted:
                    checkpoint["candidates"].setdefault(tag, []).append({"pair": pair, "payload": file_fact(output)})
                    atomic_json(checkpoint_path, checkpoint)
                continue
            _, _, offset = chunk_bounds(pair)
            native = np.asarray(np.load(chunk_dir(pair) / "native_rgb.float32.npy", mmap_mode="r")[offset]).astype(
                np.float32, copy=True
            )
            fields = candidate_fields(
                torch,
                segnet,
                posenet,
                native_chw=native,
                tokens=np.asarray(tokens[pair]),
                slave_hwc=np.asarray(raw[2 * pair]),
                global_prototypes=prototypes,
                edit_mask=native_manufactured[pair],
                alpha=alpha,
            )
            fact = atomic_npz(output, **fields)
            checkpoint["candidates"].setdefault(tag, []).append({"pair": pair, "payload": fact})
            atomic_json(checkpoint_path, checkpoint)
            print(f"{tag} retained pair {pair}", flush=True)

    baseline_argmax = []
    baseline_pose6 = []
    candidate_rows = []
    for pair in pairs:
        with np.load(baseline_dir / f"pair_{int(pair):04d}.npz") as values:
            baseline_argmax.append(values["segnet_argmax_uint8"].copy())
            baseline_pose6.append(values["pose6_float32"].copy())
    baseline_argmax_array = np.stack(baseline_argmax)
    baseline_pose6_array = np.stack(baseline_pose6)
    subset_gt = np.asarray(gt[pairs])
    subset_tokens = np.asarray(tokens[pairs])
    subset_manufactured = final_manufactured[pairs]
    subset_native_manufactured = native_manufactured[pairs]
    subset_gt_pose6 = np.asarray(gt_pose6[pairs])
    baseline_error = baseline_argmax_array != subset_gt
    baseline_dseg = float(baseline_error.mean())
    baseline_dpose = float(np.mean((baseline_pose6_array - subset_gt_pose6) ** 2))
    baseline_metrics = {
        "d_seg": baseline_dseg,
        "d_pose": baseline_dpose,
        "seg_errors": int(baseline_error.sum()),
        "cpu_manufactured_errors": int((baseline_error & (subset_tokens == subset_gt)).sum()),
        "t4_final_manufactured_support": int(subset_manufactured.sum()),
        "t4_final_manufactured_still_wrong_cpu": int((subset_manufactured & baseline_error).sum()),
        "native_manufactured_support": int(subset_native_manufactured.sum()),
        "native_manufactured_still_wrong_cpu": int((subset_native_manufactured & baseline_error).sum()),
        "pose_contribution": math.sqrt(10.0 * baseline_dpose),
    }

    for tag, alpha in PROBE_CANDIDATES:
        candidate_argmax = []
        candidate_pose6 = []
        payloads = []
        for pair in pairs:
            path = candidate_root / tag / "pairs" / f"pair_{int(pair):04d}.npz"
            payloads.append(file_fact(path))
            with np.load(path) as values:
                candidate_argmax.append(values["segnet_argmax_uint8"].copy())
                candidate_pose6.append(values["pose6_float32"].copy())
        argmax = np.stack(candidate_argmax)
        pose6 = np.stack(candidate_pose6)
        error = argmax != subset_gt
        dseg = float(error.mean())
        dpose = float(np.mean((pose6 - subset_gt_pose6) ** 2))
        fixed = baseline_error & ~error
        introduced = ~baseline_error & error
        manufactured_baseline_wrong = subset_manufactured & baseline_error
        manufactured_fixed = manufactured_baseline_wrong & ~error
        manufactured_persisting = manufactured_baseline_wrong & error
        manufactured_introduced = subset_manufactured & ~baseline_error & error
        native_baseline_wrong = subset_native_manufactured & baseline_error
        native_fixed = native_baseline_wrong & ~error
        native_persisting = native_baseline_wrong & error
        per_class = []
        for class_index, class_name in enumerate(CLASSES):
            class_support = subset_gt == class_index
            per_class.append(
                {
                    "class": class_name,
                    "support_pixels": int(class_support.sum()),
                    "baseline_errors": int((baseline_error & class_support).sum()),
                    "candidate_errors": int((error & class_support).sum()),
                    "fixed": int((fixed & class_support).sum()),
                    "introduced": int((introduced & class_support).sum()),
                    "manufactured_fixed": int((manufactured_fixed & class_support).sum()),
                }
            )
        delta_seg_s = 100.0 * (dseg - baseline_dseg)
        delta_pose_s = math.sqrt(10.0 * dpose) - math.sqrt(10.0 * baseline_dpose)
        rate_delta_s_lower_bound = RATE_S_PER_BYTE * candidate_payload_bytes
        candidate_rows.append(
            {
                "tag": tag,
                "alpha": alpha,
                "compressed_payload_bytes": candidate_payload_bytes,
                "archive_byte_delta": "UNMEASURED: receiver/container not integrated",
                "rate_delta_s_lower_bound": rate_delta_s_lower_bound,
                "d_seg": dseg,
                "delta_d_seg": dseg - baseline_dseg,
                "seg_delta_s": delta_seg_s,
                "d_pose": dpose,
                "delta_d_pose": dpose - baseline_dpose,
                "pose_delta_s": delta_pose_s,
                "distortion_delta_s": delta_seg_s + delta_pose_s,
                "joint_delta_s_lower_bound": delta_seg_s + delta_pose_s + rate_delta_s_lower_bound,
                "errors_fixed": int(fixed.sum()),
                "errors_introduced": int(introduced.sum()),
                "net_error_delta": int(error.sum() - baseline_error.sum()),
                "manufactured_fixed": int(manufactured_fixed.sum()),
                "manufactured_persisting": int(manufactured_persisting.sum()),
                "manufactured_introduced_on_cpu": int(manufactured_introduced.sum()),
                "native_manufactured_fixed": int(native_fixed.sum()),
                "native_manufactured_persisting": int(native_persisting.sum()),
                "per_class": per_class,
                "payloads": payloads,
            }
        )

    result = {
        "schema": "ddm_mf1.probe_result.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "verdict_population": "seeded stratified-random n32 subset of current DX2 object; not a prefix and not n600",
        "pair_indices": pairs.tolist(),
        "source_facts": facts,
        "baseline": baseline_metrics,
        "candidates": candidate_rows,
        "byte_cost": {
            "compressed_payload_bytes": candidate_payload_bytes,
            "archive_delta_bytes": "UNMEASURED: receiver/container not integrated",
            "basis": "exact full-n600 GT-derived native-stage address mask compressed with real Brotli q11; payload bytes are a strict lower bound before container overhead",
            "archive_sha256": ARCHIVE_SHA256,
            "receiver_integration": "UNMEASURED: research instrument only; no shipping inflate.py was edited",
        },
        "authority_boundary": (
            "All component fields are measured through real R, uint8, frozen CPU SegNet and PoseNet "
            "on n32. No result is a full-n600 scorer row, contest-CPU row, contest-CUDA row, or pointer move."
        ),
        "localization_sha256": sha256_file(localization_path),
    }
    result_fact = atomic_json(store / "PROBE_RESULT.json", result)
    manifest = {
        "schema": "ddm_mf1.probe_manifest.v1",
        "result": result_fact,
        "checkpoint": file_fact(checkpoint_path),
        "baseline_payloads": [file_fact(baseline_dir / f"pair_{int(pair):04d}.npz") for pair in pairs],
        "candidate_payload_count": len(PROBE_CANDIDATES) * len(pairs),
        "complete": True,
    }
    atomic_json(store / "PROBE_MANIFEST.json", manifest)


def verify(store: Path) -> None:
    localization = json.loads((store / "LOCALIZATION.json").read_text())
    current_source_sha = sha256_file(Path(__file__).resolve())
    if localization["source_facts"]["instrument_source"]["sha256"] != current_source_sha:
        raise Mf1Error("instrument source changed after localization; verification is not source-closed")
    manifest = json.loads((store / "LOCALIZATION_MANIFEST.json").read_text())
    for fact in manifest["boundary_payloads"].values():
        file_fact(Path(fact["path"]), expected_sha256=fact["sha256"])
    for fact in manifest["candidate_payload"].values():
        file_fact(Path(fact["path"]), expected_sha256=fact["sha256"])
    for key in ("prototype_payload", "selection_payload", "clusters", "result", "localize_checkpoint"):
        fact = manifest[key]
        file_fact(Path(fact["path"]), expected_sha256=fact["sha256"])
    if localization["final_manufactured_pixels"] != 21_493:
        raise Mf1Error("localization final manufactured count drift")
    receipt: dict[str, Any] = {
        "schema": "ddm_mf1.completed_verification.v1",
        "localization_sha256": sha256_file(store / "LOCALIZATION.json"),
        "localization_manifest_sha256": sha256_file(store / "LOCALIZATION_MANIFEST.json"),
        "probe_complete": False,
    }
    probe_result = store / "PROBE_RESULT.json"
    if probe_result.is_file():
        probe_manifest = json.loads((store / "PROBE_MANIFEST.json").read_text())
        file_fact(Path(probe_manifest["result"]["path"]), expected_sha256=probe_manifest["result"]["sha256"])
        for fact in probe_manifest["baseline_payloads"]:
            file_fact(Path(fact["path"]), expected_sha256=fact["sha256"])
        checkpoint_path = Path(probe_manifest["checkpoint"]["path"])
        file_fact(checkpoint_path, expected_sha256=probe_manifest["checkpoint"]["sha256"])
        checkpoint = json.loads(checkpoint_path.read_text())
        baseline_pairs = [int(row["pair"]) for row in checkpoint["baseline"]]
        if len(baseline_pairs) != 32 or len(set(baseline_pairs)) != 32:
            raise Mf1Error("probe baseline checkpoint is not 32 unique pairs")
        expected_tags = {tag for tag, _ in PROBE_CANDIDATES}
        if set(checkpoint["candidates"]) != expected_tags:
            raise Mf1Error("probe checkpoint candidate tags drifted")
        candidate_rows = [row for rows in checkpoint["candidates"].values() for row in rows]
        candidate_keys = [(tag, int(row["pair"])) for tag, rows in checkpoint["candidates"].items() for row in rows]
        if len(candidate_rows) != 96 or len(set(candidate_keys)) != 96:
            raise Mf1Error("probe candidate checkpoint is not 3 x 32 unique pair payloads")
        if any(len(rows) != 32 for rows in checkpoint["candidates"].values()):
            raise Mf1Error("probe candidate checkpoint has an incomplete candidate")
        for rows in checkpoint["candidates"].values():
            for row in rows:
                file_fact(Path(row["payload"]["path"]), expected_sha256=row["payload"]["sha256"])
        receipt.update(
            {
                "probe_complete": True,
                "probe_result_sha256": sha256_file(probe_result),
                "probe_manifest_sha256": sha256_file(store / "PROBE_MANIFEST.json"),
                "candidate_payloads_verified": sum(len(rows) for rows in checkpoint["candidates"].values()),
            }
        )
    receipt["verified_at_unix"] = time.time()
    receipt["status"] = "COMPLETE"
    atomic_json(store / "COMPLETED_VERIFICATION.json", receipt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("localize", "probe", "verify"))
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if platform.system() != "Darwin":
        raise Mf1Error("this charter authorizes the macOS CPU advisory lane only")
    if args.stage == "localize":
        localize(args.store, resume_from=args.resume_from)
    elif args.stage == "probe":
        probe(args.store, resume_from=args.resume_from)
    else:
        if args.resume_from is not None:
            raise Mf1Error("verify discovers checkpoint paths from manifests; --resume-from is not accepted")
        verify(args.store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
