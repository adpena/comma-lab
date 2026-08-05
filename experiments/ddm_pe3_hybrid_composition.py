#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""pe3 hybrid per-regime composition and all-signal PE1 decomposition.

This is scorer-free. It reuses PE1's n600 class-pair component extraction and
real coder race, builds a distinct PE3 counted section, and appends the measured
section to the qo1 IX2 payload for byte closure. It does not patch PE2's runtime
receiver or run SegNet/PoseNet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

_REPO: Final = Path(__file__).resolve().parents[1]
for _path in (_REPO / "src", _REPO / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import ddm_pe1_per_edge_partition_race as pe1

from tac.optimization.ddm_ix2_archive_container import (
    build_payload,
    build_single_member_zip,
    parse_payload,
)

PE3_MAGIC: Final = b"PE3EDGE1"
PE3_VERSION: Final = 1
PE3_KIND_HYBRID: Final = 1
PE3_HEADER: Final = struct.Struct("<8sBHHHBBII32s")
PE3_MODE_CURVE: Final = 1
PE3_MODE_GENERATOR: Final = 2
MODE_NAMES: Final = {
    PE3_MODE_CURVE: "depth_conditioned_curve",
    PE3_MODE_GENERATOR: "generator_pair_bisector",
}

DEFAULT_RESEARCH_DIR: Final = _REPO / ".omx/research/ddm_pe3_20260805"
DEFAULT_SSD_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_pe3_20260805")
DEFAULT_PE1_RECEIPT: Final = _REPO / ".omx/research/ddm_pe1_20260805/ddm_pe1_repr_race_receipt.json"
DEFAULT_G4_RECURRENCE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_g4_spatial_stationarity_n600_20260722T212138Z/stage_checkpoints/01_recurrence_arrays.npz"
)
BASELINE_S: Final = pe1.BASELINE_S
BASELINE_BYTES: Final = pe1.BASELINE_BYTES
BASELINE_AXIS: Final = pe1.BASELINE_AXIS
RATE_DENOM: Final = pe1.RATE_DENOM


class PE3Error(ValueError):
    """PE3 failed a typed custody, parse-back, or accounting invariant."""


@dataclass(frozen=True, slots=True)
class ComponentInfo:
    component: pe1.Component
    depth_band: str
    flicker_pixels: int
    g4_static_pixels: int
    g4_transient_pixels: int
    curve_record: bytes
    curve_mask: np.ndarray
    generator_record: bytes
    generator_mask: np.ndarray
    policy_mode: int | None
    policy_reason: str
    density_value: int


@dataclass(frozen=True, slots=True)
class HybridBuild:
    surface_id: str
    raw: bytes
    frame_records: tuple[bytes, ...]
    component_masks: dict[int, np.ndarray]
    component_modes: dict[int, int]
    component_bytes: dict[int, int]
    selected_ids: frozenset[int]
    policy: str
    target_section_bytes: int | None
    prefix_counts: dict[str, int]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pe1.CoderResult):
        return {
            "codec": value.codec,
            "bytes": value.bytes,
            "sha256": value.sha256,
            "artifact_path": value.artifact_path,
        }
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    return value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise PE3Error(f"expected object JSON at {path}")
    return data


def depth_band(y: float, y1: float, y2: float) -> str:
    if y < y1:
        return "far_row"
    if y < y2:
        return "mid_row"
    return "near_row"


def depth_stride(band: str) -> int:
    if band == "far_row":
        return 4
    if band == "mid_row":
        return 8
    if band == "near_row":
        return 16
    raise PE3Error(f"unknown depth band {band!r}")


def build_flicker_masks(lstars: np.ndarray) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for pair in range(1, pe1.N_PAIRS - 1):
        lab = np.asarray(lstars[pair], dtype=np.uint8)
        prev = np.asarray(lstars[pair - 1], dtype=np.uint8)
        nxt = np.asarray(lstars[pair + 1], dtype=np.uint8)
        masks[pair] = (lab != prev) & (lab != nxt)
    return masks


def transition_counts_for_components(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arrays = np.load(path)
    if "transition_counts" not in arrays:
        return None
    counts = np.asarray(arrays["transition_counts"])
    if tuple(counts.shape) != (25, pe1.SEG_H, pe1.SEG_W):
        raise PE3Error(f"unexpected g4 transition_counts shape {counts.shape}")
    return counts


def component_static_transient_counts(
    comp: pe1.Component,
    current: np.ndarray,
    lstars: np.ndarray,
    transition_counts: np.ndarray | None,
) -> tuple[int, int]:
    if transition_counts is None:
        return (0, 0)
    flat = comp.flat.astype(np.int64)
    ys = flat // pe1.SEG_W
    xs = flat % pe1.SEG_W
    target = np.asarray(lstars[comp.pair], dtype=np.uint8).reshape(-1)[flat]
    pred = np.asarray(current[comp.pair], dtype=np.uint8).reshape(-1)[flat]
    codes = pred.astype(np.int16) * 5 + target.astype(np.int16)
    recurrence = transition_counts[codes, ys, xs]
    static = int(np.count_nonzero(recurrence >= 2))
    transient = int(np.count_nonzero(recurrence < 2))
    return static, transient


def prepare_component_infos(
    *,
    components: list[pe1.Component],
    lstars: np.ndarray,
    current: np.ndarray,
    generator_params: dict[int, pe1.GeneratorParams],
    y1: float,
    y2: float,
    transition_counts: np.ndarray | None,
) -> tuple[list[ComponentInfo], dict[str, Any]]:
    flicker_masks = build_flicker_masks(lstars)
    infos: list[ComponentInfo] = []
    centroid_y = np.asarray([comp.centroid_yx[0] for comp in components], dtype=np.float64)
    flip_sorted = sorted(components, key=lambda c: c.centroid_yx[0])
    flip_total = sum(comp.flip_mass for comp in flip_sorted)
    weighted_quantiles: dict[str, float] = {}
    if flip_total:
        cumulative = 0
        targets = {"q25": 0.25 * flip_total, "q50": 0.50 * flip_total, "q75": 0.75 * flip_total}
        pending = dict(targets)
        for comp in flip_sorted:
            cumulative += comp.flip_mass
            for name, target in list(pending.items()):
                if cumulative >= target:
                    weighted_quantiles[name] = float(comp.centroid_yx[0])
                    pending.pop(name)

    for comp in components:
        band = depth_band(comp.centroid_yx[0], y1, y2)
        curve_params = pe1.fit_curve_params(comp, depth_stride(band))
        curve_record = pe1.encode_curve_params(curve_params)
        curve_mask = pe1.rasterize_curve(pe1.decode_curve_params(curve_record))
        gen_params = generator_params[comp.uid]
        generator_record = pe1.encode_generator_params(gen_params)
        generator_mask = pe1.rasterize_generator(gen_params)
        flicker = 0
        if comp.pair in flicker_masks:
            flicker = int(flicker_masks[comp.pair].reshape(-1)[comp.flat.astype(np.int64)].sum())
        g4_static, g4_transient = component_static_transient_counts(
            comp,
            current=current,
            lstars=lstars,
            transition_counts=transition_counts,
        )
        flicker_dominated = comp.flip_mass > 0 and flicker >= comp.flip_mass
        tiny_dust = comp.flat.size <= 8 and comp.flip_mass <= 2
        if comp.flip_mass == 0:
            mode = None
            reason = "zero_flip_context_omission"
            density_value = 0
        elif tiny_dust:
            mode = None
            reason = "tiny_dust_priced_omission"
            density_value = 0
        elif flicker_dominated and comp.flat.size <= 64 and comp.flip_mass <= 4:
            mode = None
            reason = "fl1_flicker_dominated_priced_omission"
            density_value = 0
        elif comp.flat.size >= 24 and not (flicker_dominated and comp.flip_mass < 12):
            mode = PE3_MODE_CURVE
            reason = f"coherent_edge_depth_curve_{band}_k{depth_stride(band)}"
            source = pe1.component_mask(comp.flat)
            density_value = max(comp.flip_mass, int(np.count_nonzero(source & curve_mask)))
        else:
            mode = PE3_MODE_GENERATOR
            reason = "surgical_generator_pair_bisector"
            source = pe1.component_mask(comp.flat)
            density_value = max(comp.flip_mass, int(np.count_nonzero(source & generator_mask)))
        infos.append(
            ComponentInfo(
                component=comp,
                depth_band=band,
                flicker_pixels=flicker,
                g4_static_pixels=g4_static,
                g4_transient_pixels=g4_transient,
                curve_record=curve_record,
                curve_mask=curve_mask,
                generator_record=generator_record,
                generator_mask=generator_mask,
                policy_mode=mode,
                policy_reason=reason,
                density_value=density_value,
            )
        )
    meta = {
        "depth_y_cutpoints": [y1, y2],
        "centroid_y_quantiles_unweighted": {
            f"q{q}": float(np.percentile(centroid_y, q)) for q in (0, 10, 25, 50, 75, 90, 100)
        },
        "centroid_y_quantiles_flip_weighted": weighted_quantiles,
        "depth_stride_policy": {"far_row": 4, "mid_row": 8, "near_row": 16},
    }
    return infos, meta


def hybrid_record(info: ComponentInfo, mode: int) -> tuple[bytes, np.ndarray]:
    if mode == PE3_MODE_CURVE:
        return bytes([PE3_MODE_CURVE]) + info.curve_record, info.curve_mask
    if mode == PE3_MODE_GENERATOR:
        return bytes([PE3_MODE_GENERATOR]) + info.generator_record, info.generator_mask
    raise PE3Error(f"unknown PE3 mode {mode}")


def frame_records_from_hybrid(
    components: list[pe1.Component],
    records_by_uid: dict[int, bytes],
) -> tuple[bytes, ...]:
    by_pair: list[list[bytes]] = [[] for _ in range(pe1.N_PAIRS)]
    for comp in components:
        record = records_by_uid.get(comp.uid)
        if record is not None:
            by_pair[comp.pair].append(pe1.varint(len(record)) + record)
    return tuple(pe1.varint(len(records)) + b"".join(records) for records in by_pair)


def build_hybrid(
    *,
    surface_id: str,
    components: list[pe1.Component],
    selected_modes: dict[int, int],
    infos_by_uid: dict[int, ComponentInfo],
    policy: str,
    target_section_bytes: int | None,
    prefix_counts: dict[str, int],
) -> HybridBuild:
    records: dict[int, bytes] = {}
    masks: dict[int, np.ndarray] = {}
    modes: dict[int, int] = {}
    component_bytes: dict[int, int] = {}
    for uid, mode in selected_modes.items():
        info = infos_by_uid[uid]
        record, mask = hybrid_record(info, mode)
        records[uid] = record
        masks[uid] = mask
        modes[uid] = mode
        component_bytes[uid] = len(record)
    frame_records = frame_records_from_hybrid(components, records)
    return HybridBuild(
        surface_id=surface_id,
        raw=b"".join(frame_records),
        frame_records=frame_records,
        component_masks=masks,
        component_modes=modes,
        component_bytes=component_bytes,
        selected_ids=frozenset(records),
        policy=policy,
        target_section_bytes=target_section_bytes,
        prefix_counts=prefix_counts,
    )


def decode_body(codec: str, payload: bytes, expected_len: int) -> bytes:
    if codec == "brotli-q11":
        raw = brotli.decompress(payload)
        if len(raw) != expected_len:
            raise PE3Error("brotli PE3 body length mismatch")
        return raw
    return pe1.decode_body(codec, payload, expected_len)


def parse_pe3_section(section: bytes) -> dict[str, Any]:
    if len(section) < PE3_HEADER.size:
        raise PE3Error("PE3 section header truncated")
    magic, version, seg_h, seg_w, n_pairs, kind, codec_id, raw_len, frame_count, raw_sha = PE3_HEADER.unpack_from(
        section, 0
    )
    if magic != PE3_MAGIC or version != PE3_VERSION:
        raise PE3Error("PE3 magic/version mismatch")
    if (seg_h, seg_w, n_pairs) != (pe1.SEG_H, pe1.SEG_W, pe1.N_PAIRS):
        raise PE3Error("PE3 geometry mismatch")
    if kind != PE3_KIND_HYBRID:
        raise PE3Error(f"unknown PE3 kind {kind}")
    codec = pe1.ID_CODECS.get(int(codec_id))
    if codec is None:
        raise PE3Error(f"unknown codec id {codec_id}")
    raw = decode_body(codec, section[PE3_HEADER.size :], int(raw_len))
    if hashlib.sha256(raw).digest() != raw_sha:
        raise PE3Error("PE3 raw SHA mismatch")
    offset = 0
    records = 0
    mode_counts: Counter[str] = Counter()
    for _pair in range(n_pairs):
        count, offset = pe1.read_varint(raw, offset)
        for _ in range(count):
            length, offset = pe1.read_varint(raw, offset)
            record = raw[offset : offset + length]
            if len(record) != length:
                raise PE3Error("PE3 component record truncated")
            if not record:
                raise PE3Error("PE3 empty component record")
            mode = int(record[0])
            if mode not in MODE_NAMES:
                raise PE3Error(f"unknown PE3 component mode {mode}")
            mode_counts[MODE_NAMES[mode]] += 1
            offset += length
            records += 1
    if offset != len(raw):
        raise PE3Error("PE3 raw body has trailing bytes")
    if frame_count != n_pairs:
        raise PE3Error("PE3 frame count mismatch")
    return {
        "codec": codec,
        "kind": int(kind),
        "kind_name": "hybrid_per_regime",
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
        "section_bytes": len(section),
        "section_sha256": sha256_bytes(section),
        "frame_records": int(frame_count),
        "component_records": records,
        "mode_counts": dict(mode_counts),
        "schema": "pe3_section_parse_back.v1",
    }


def race_hybrid_coders(
    build: HybridBuild,
    *,
    artifact_dir: Path,
    store_best: bool,
) -> tuple[tuple[pe1.CoderResult, ...], str, bytes]:
    return pe1.race_coders(
        surface_id=build.surface_id,
        raw=build.raw,
        frame_records=build.frame_records,
        artifact_dir=artifact_dir,
        store_best=store_best,
    )


def build_pe3_section(build: HybridBuild, codec: str, body: bytes) -> bytes:
    header = PE3_HEADER.pack(
        PE3_MAGIC,
        PE3_VERSION,
        pe1.SEG_H,
        pe1.SEG_W,
        pe1.N_PAIRS,
        PE3_KIND_HYBRID,
        pe1.CODEC_IDS[codec],
        len(build.raw),
        len(build.frame_records),
        hashlib.sha256(build.raw).digest(),
    )
    section = header + body
    parsed = parse_pe3_section(section)
    if parsed["raw_sha256"] != sha256_bytes(build.raw):
        raise PE3Error("PE3 section parse-back raw SHA mismatch")
    return section


def component_metrics(components: list[pe1.Component], build: HybridBuild) -> dict[str, Any]:
    rep = pe1.RepresentationBuild(
        surface_id=build.surface_id,
        kind=pe1.PE1_CURVE,
        raw=build.raw,
        frame_records=build.frame_records,
        component_masks=build.component_masks,
        component_dims=dict.fromkeys(build.component_masks, 1),
        component_bytes=build.component_bytes,
        scope_note=build.policy,
        falls_out=True,
        selected_component_ids=build.selected_ids,
        metadata={},
    )
    return pe1.component_metrics(components, rep)


def price_hybrid(
    build: HybridBuild,
    *,
    components: list[pe1.Component],
    base_payload: bytes,
    artifact_dir: Path,
    store_best: bool,
) -> tuple[dict[str, Any], bytes]:
    coders, codec, body = race_hybrid_coders(build, artifact_dir=artifact_dir, store_best=store_best)
    section = build_pe3_section(build, codec, body)
    parsed = parse_pe3_section(section)
    metrics = component_metrics(components, build)
    bulk, sections = parse_payload(base_payload)
    projected_payload = build_payload(bulk, [*sections, section])
    projected_archive = build_single_member_zip(projected_payload, name="0.bin")
    selected_flip = sum(comp.flip_mass for comp in components if comp.uid in build.selected_ids)
    total_flip = sum(comp.flip_mass for comp in components)
    mode_counts = Counter(MODE_NAMES[mode] for mode in build.component_modes.values())
    best = min(coders, key=lambda row: row.bytes)
    row = {
        "surface_id": build.surface_id,
        "policy": build.policy,
        "target_section_bytes": build.target_section_bytes,
        "prefix_counts": build.prefix_counts,
        "section_bytes": len(section),
        "section_sha256": sha256_bytes(section),
        "raw_bytes": len(build.raw),
        "raw_sha256": sha256_bytes(build.raw),
        "best_codec": best.codec,
        "best_body_bytes": best.bytes,
        "best_body_sha256": best.sha256,
        "coder_race": list(coders),
        "parse_back": parsed,
        "mode_counts": dict(mode_counts),
        "selected_flip_mass": selected_flip,
        "total_flip_mass": total_flip,
        "flip_recall": selected_flip / total_flip if total_flip else 1.0,
        "mask_domain_fidelity": metrics,
        "section_bits_per_band_px": 8.0 * len(section) / max(1, metrics["source_band_pixels"]),
        "archive_projection": {
            "projected_archive_bytes": len(projected_archive),
            "projected_archive_sha256": sha256_bytes(projected_archive),
            "payload_bytes": len(projected_payload),
            "payload_sha256": sha256_bytes(projected_payload),
            "joint_section_count": len(sections) + 1,
        },
        "claim_label": "MEASURED n600 mask-domain coder bytes and PE3 section parse-back; scorer-free",
    }
    return row, section


def candidate_modes_curve_waterfill(
    infos: list[ComponentInfo],
    components: list[pe1.Component],
    count: int,
) -> dict[int, int]:
    by_edge: dict[tuple[int, int], list[ComponentInfo]] = defaultdict(list)
    for info in infos:
        if info.policy_mode is not None and info.density_value > 0:
            by_edge[info.component.edge].append(info)
    for edge in by_edge:
        by_edge[edge].sort(
            key=lambda item: (
                -item.density_value
                / max(1, len(item.curve_record) + len(pe1.varint(len(item.curve_record) + 1)) + 1),
                -item.density_value,
                item.component.uid,
            )
        )
    indexes = dict.fromkeys(by_edge, 0)
    selected: dict[int, int] = {}
    while len(selected) < count:
        best: ComponentInfo | None = None
        best_edge: tuple[int, int] | None = None
        best_density = -1.0
        for edge, edge_infos in by_edge.items():
            idx = indexes[edge]
            if idx >= len(edge_infos):
                continue
            item = edge_infos[idx]
            cost = len(item.curve_record) + len(pe1.varint(len(item.curve_record) + 1)) + 1
            density = item.density_value / max(1, cost)
            if density > best_density:
                best = item
                best_edge = edge
                best_density = density
        if best is None or best_edge is None:
            break
        indexes[best_edge] += 1
        selected[best.component.uid] = PE3_MODE_CURVE
    return selected


def rank_curve_residual_after_generator(
    infos_by_uid: dict[int, ComponentInfo],
    ranked_components: list[pe1.Component],
    generator_prefix: set[int],
) -> list[ComponentInfo]:
    residual: list[ComponentInfo] = []
    for comp in ranked_components:
        if comp.uid in generator_prefix or comp.flip_mass <= 0:
            continue
        info = infos_by_uid[comp.uid]
        cost = len(info.curve_record) + len(pe1.varint(len(info.curve_record) + 1)) + 1
        if cost <= 0:
            continue
        residual.append(info)
    residual.sort(
        key=lambda item: (
            -item.component.flip_mass
            / max(1, len(item.curve_record) + len(pe1.varint(len(item.curve_record) + 1)) + 1),
            -item.component.flip_mass,
            item.component.uid,
        )
    )
    return residual


def build_generator_plus_curve_modes(
    *,
    ranked_components: list[pe1.Component],
    generator_count: int,
    curve_residual: list[ComponentInfo],
    curve_count: int,
) -> dict[int, int]:
    modes: dict[int, int] = {
        comp.uid: PE3_MODE_GENERATOR for comp in ranked_components[:generator_count]
    }
    for info in curve_residual[:curve_count]:
        modes.setdefault(info.component.uid, PE3_MODE_CURVE)
    return modes


def choose_best_at_target(
    rows_and_sections: list[tuple[dict[str, Any], bytes]],
    target: int,
) -> tuple[dict[str, Any], bytes]:
    under = [item for item in rows_and_sections if item[0]["section_bytes"] <= target]
    if under:
        return max(under, key=lambda item: (item[0]["flip_recall"], item[0]["section_bytes"]))
    return min(rows_and_sections, key=lambda item: abs(item[0]["section_bytes"] - target))


def knee_point(rows_and_sections: list[tuple[dict[str, Any], bytes]]) -> tuple[dict[str, Any], bytes]:
    ordered = sorted(rows_and_sections, key=lambda item: item[0]["section_bytes"])
    if len(ordered) <= 2:
        return ordered[-1]
    x0 = ordered[0][0]["section_bytes"]
    y0 = ordered[0][0]["flip_recall"]
    x1 = ordered[-1][0]["section_bytes"]
    y1 = ordered[-1][0]["flip_recall"]
    denom = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 or 1.0
    best = ordered[0]
    best_dist = -1.0
    for item in ordered[1:-1]:
        x = item[0]["section_bytes"]
        y = item[0]["flip_recall"]
        dist = abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
        if dist > best_dist:
            best = item
            best_dist = dist
    best[0]["knee_detection"] = {
        "method": "max_distance_to_endpoint_line",
        "distance": best_dist,
        "candidate_count": len(ordered),
    }
    return best


def write_candidate(
    *,
    base_sub: Path,
    candidate_dir: Path,
    section: bytes,
    section_name: str,
) -> dict[str, Any]:
    if candidate_dir.exists():
        raise PE3Error(f"candidate dir already exists: {candidate_dir}")
    pe1.bd1.copy_runtime_tree(base_sub, candidate_dir)
    base_payload = pe1.bd1.read_archive_payload(base_sub / "archive.zip")
    bulk, sections = parse_payload(base_payload)
    payload = build_payload(bulk, [*sections, section])
    archive_zip = build_single_member_zip(payload, name="0.bin")
    (candidate_dir / "archive" / "0.bin").write_bytes(payload)
    (candidate_dir / "archive.zip").write_bytes(archive_zip)
    parse_back = pe1.bd1.build_local_ledger(
        candidate_dir / "archive.zip",
        ("config", "renderer", "selector", "pose_warp", "frame0_pose_repair", section_name),
    )
    pe3_parse = parse_pe3_section(parse_payload(payload)[1][-1])
    archive_bytes = (candidate_dir / "archive.zip").stat().st_size
    return {
        "submission_dir": str(candidate_dir),
        "archive_bytes": archive_bytes,
        "archive_sha256": sha256_file(candidate_dir / "archive.zip"),
        "payload_sha256": sha256_bytes(payload),
        "delta_bytes_vs_qo1": archive_bytes - BASELINE_BYTES,
        "rate_delta_S_vs_qo1": 25.0 * (archive_bytes - BASELINE_BYTES) / RATE_DENOM,
        "parse_back": parse_back,
        "pe3_section_parse_back": pe3_parse,
        "label": "BYTE-CLOSED / PE3-SECTION-PARSE-BACK / RUNTIME-SURVIVAL-UNMEASURED / score_claim=false",
    }


def decompose_transport(
    *,
    components: list[pe1.Component],
    generator_params: dict[int, pe1.GeneratorParams],
    infos_by_uid: dict[int, ComponentInfo],
    max_distance_px: float,
) -> dict[str, Any]:
    selected_ids = frozenset(comp.uid for comp in components)
    tracks = pe1.track_generator_components(
        components,
        generator_params,
        max_distance_px=max_distance_px,
    )
    by_pair: list[list[pe1.Component]] = [[] for _ in range(pe1.N_PAIRS)]
    for comp in components:
        by_pair[comp.pair].append(comp)
    previous_by_track: dict[int, tuple[int, ...]] = {}
    per_track: dict[int, dict[str, Any]] = {}
    for comps in by_pair:
        for comp in comps:
            if comp.uid not in selected_ids:
                continue
            params = generator_params[comp.uid]
            fields = pe1._absolute_generator_fields(params)
            track_id = tracks[comp.uid]
            gen_record = pe1.encode_generator_params(params)
            independent_len = len(pe1.varint(len(gen_record))) + len(gen_record)
            record = bytearray([params.edge[0], params.edge[1]])
            record += pe1.varint(track_id)
            previous = previous_by_track.get(track_id)
            if previous is None:
                record.append(0)
                for value in fields:
                    record += pe1.write_zigzag(value)
            else:
                record.append(1)
                for value, prev in zip(fields, previous, strict=True):
                    record += pe1.write_zigzag(value - prev)
            previous_by_track[track_id] = fields
            transport_len = len(pe1.varint(len(record))) + len(record)
            info = infos_by_uid[comp.uid]
            row = per_track.setdefault(
                track_id,
                {
                    "track_id": track_id,
                    "components": 0,
                    "independent_record_bytes": 0,
                    "transport_record_bytes": 0,
                    "flip_mass": 0,
                    "source_band_pixels": 0,
                    "depth_counts": Counter(),
                    "static_pixels": 0,
                    "transient_pixels": 0,
                },
            )
            row["components"] += 1
            row["independent_record_bytes"] += independent_len
            row["transport_record_bytes"] += transport_len
            row["flip_mass"] += comp.flip_mass
            row["source_band_pixels"] += int(comp.flat.size)
            row["depth_counts"][info.depth_band] += 1
            row["static_pixels"] += info.g4_static_pixels
            row["transient_pixels"] += info.g4_transient_pixels

    rows = list(per_track.values())
    for row in rows:
        row["transport_won"] = row["transport_record_bytes"] < row["independent_record_bytes"]
        row["majority_depth_band"] = row["depth_counts"].most_common(1)[0][0]
        row["depth_counts"] = dict(row["depth_counts"])
        total_stratum = row["static_pixels"] + row["transient_pixels"]
        row["g4_static_fraction"] = row["static_pixels"] / total_stratum if total_stratum else None
    winners = [row for row in rows if row["transport_won"]]
    return {
        "schema": "pe3_transport_decomposition.v1",
        "tracks": len(rows),
        "transport_won_tracks": len(winners),
        "transport_won_fraction": len(winners) / len(rows) if rows else 0.0,
        "component_denominator": len(components),
        "aggregate_independent_record_bytes": sum(row["independent_record_bytes"] for row in rows),
        "aggregate_transport_record_bytes": sum(row["transport_record_bytes"] for row in rows),
        "by_track_length": summarize_by_key(rows, lambda row: length_bucket(row["components"])),
        "by_depth": summarize_by_key(rows, lambda row: row["majority_depth_band"]),
        "by_g4_stratum": summarize_by_key(
            rows,
            lambda row: "static_majority" if (row["g4_static_fraction"] or 0.0) >= 0.5 else "transient_majority",
        ),
        "conditional_transport_hybrid": conditional_transport_price(rows),
        "sample_winning_tracks": sorted(winners, key=lambda row: (-row["flip_mass"], row["track_id"]))[:10],
        "verdict_scope": "FORMULATION-scoped to PE1 local generator-parameter transport; compressed aggregate remains worse unless the conditional subset price is separately adopted.",
    }


def length_bucket(length: int) -> str:
    if length <= 1:
        return "len1"
    if length <= 3:
        return "len2_3"
    if length <= 8:
        return "len4_8"
    if length <= 32:
        return "len9_32"
    return "len33_plus"


def summarize_by_key(rows: Iterable[dict[str, Any]], key_fn: Any) -> dict[str, Any]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(key_fn(row))
        bucket = out.setdefault(
            key,
            {
                "tracks": 0,
                "transport_won_tracks": 0,
                "components": 0,
                "flip_mass": 0,
                "independent_record_bytes": 0,
                "transport_record_bytes": 0,
            },
        )
        bucket["tracks"] += 1
        bucket["transport_won_tracks"] += int(bool(row["transport_won"]))
        bucket["components"] += row["components"]
        bucket["flip_mass"] += row["flip_mass"]
        bucket["independent_record_bytes"] += row["independent_record_bytes"]
        bucket["transport_record_bytes"] += row["transport_record_bytes"]
    for bucket in out.values():
        bucket["transport_won_fraction"] = bucket["transport_won_tracks"] / bucket["tracks"] if bucket["tracks"] else 0.0
    return out


def conditional_transport_price(rows: list[dict[str, Any]]) -> dict[str, Any]:
    independent = 0
    conditional = 0
    selector_bits = 0
    selected_tracks = 0
    for row in rows:
        independent += row["independent_record_bytes"]
        if row["transport_won"]:
            conditional += row["transport_record_bytes"]
            selected_tracks += 1
        else:
            conditional += row["independent_record_bytes"]
        selector_bits += 1
    selector_bytes = (selector_bits + 7) // 8
    return {
        "independent_record_bytes": independent,
        "conditional_record_bytes_with_1bit_selector": conditional + selector_bytes,
        "selector_bytes": selector_bytes,
        "selected_transport_tracks": selected_tracks,
        "delta_record_bytes": independent - (conditional + selector_bytes),
        "score_claim": False,
        "compression_scope": "record-byte model before global entropy coding; not promoted as archive-byte win",
    }


def decompose_missing_recall(
    *,
    components: list[pe1.Component],
    infos_by_uid: dict[int, ComponentInfo],
) -> dict[str, Any]:
    total_source = 0
    total_missing = 0
    by_depth: dict[str, dict[str, Any]] = defaultdict(lambda: {"source_band_pixels": 0, "missing_pixels": 0, "components": 0, "flicker_pixels": 0})
    by_regime: dict[str, dict[str, Any]] = defaultdict(lambda: {"source_band_pixels": 0, "missing_pixels": 0, "components": 0, "flicker_pixels": 0})
    by_edge: dict[str, dict[str, Any]] = defaultdict(lambda: {"source_band_pixels": 0, "missing_pixels": 0, "components": 0, "flicker_pixels": 0})
    for comp in components:
        info = infos_by_uid[comp.uid]
        source = pe1.component_mask(comp.flat)
        k8_params = pe1.fit_curve_params(comp, 8)
        decoded = pe1.rasterize_curve(pe1.decode_curve_params(pe1.encode_curve_params(k8_params)))
        source_px = int(source.sum())
        missing = int(np.count_nonzero(source & ~decoded))
        total_source += source_px
        total_missing += missing
        regime = "dust_fragment" if source_px <= 16 else "coherent_edge_tail"
        for table, key in (
            (by_depth, info.depth_band),
            (by_regime, regime),
            (by_edge, pe1.edge_name(comp.edge)),
        ):
            row = table[key]
            row["source_band_pixels"] += source_px
            row["missing_pixels"] += missing
            row["components"] += 1
            row["flicker_pixels"] += info.flicker_pixels
    return {
        "schema": "pe3_missing_recall_decomposition.v1",
        "reference": "PE1 explicit_curve_k8 full row",
        "source_band_pixels": total_source,
        "missing_pixels": total_missing,
        "recall": 1.0 - total_missing / total_source if total_source else 1.0,
        "by_depth": finalize_fraction_table(by_depth, "missing_pixels", total_missing),
        "by_regime": finalize_fraction_table(by_regime, "missing_pixels", total_missing),
        "by_edge": finalize_fraction_table(by_edge, "missing_pixels", total_missing),
        "fl1_flicker_overlap_scope": "component-local GT flicker pixels inside source band; not a scorer delta",
    }


def decompose_surgical_residual(
    *,
    components: list[pe1.Component],
    infos_by_uid: dict[int, ComponentInfo],
    surgical_selected: set[int],
) -> dict[str, Any]:
    total = sum(comp.flip_mass for comp in components)
    residual_components = [comp for comp in components if comp.uid not in surgical_selected]
    residual = sum(comp.flip_mass for comp in residual_components)
    by_edge: dict[str, dict[str, Any]] = defaultdict(lambda: {"components": 0, "flip_mass": 0, "source_band_pixels": 0, "flicker_pixels": 0})
    by_depth: dict[str, dict[str, Any]] = defaultdict(lambda: {"components": 0, "flip_mass": 0, "source_band_pixels": 0, "flicker_pixels": 0})
    by_class: dict[str, dict[str, Any]] = defaultdict(lambda: {"components": 0, "flip_mass": 0, "source_band_pixels": 0, "flicker_pixels": 0})
    for comp in residual_components:
        info = infos_by_uid[comp.uid]
        edge_name = pe1.edge_name(comp.edge)
        for table, key in (
            (by_edge, edge_name),
            (by_depth, info.depth_band),
        ):
            row = table[key]
            row["components"] += 1
            row["flip_mass"] += comp.flip_mass
            row["source_band_pixels"] += int(comp.flat.size)
            row["flicker_pixels"] += info.flicker_pixels
        for cls in comp.edge:
            row = by_class[pe1.CLASS_NAMES[cls]]
            row["components"] += 1
            row["flip_mass"] += comp.flip_mass
            row["source_band_pixels"] += int(comp.flat.size)
            row["flicker_pixels"] += info.flicker_pixels
    return {
        "schema": "pe3_surgical_residual_decomposition.v1",
        "reference": "PE1 generator_pair_bisector_waterfill_75kb",
        "total_flip_mass": total,
        "selected_flip_mass": total - residual,
        "residual_flip_mass": residual,
        "residual_fraction": residual / total if total else 0.0,
        "by_edge": finalize_fraction_table(by_edge, "flip_mass", residual),
        "by_depth": finalize_fraction_table(by_depth, "flip_mass", residual),
        "by_class_incident": finalize_fraction_table(by_class, "flip_mass", max(1, sum(row["flip_mass"] for row in by_class.values()))),
    }


def finalize_fraction_table(
    table: dict[str, dict[str, Any]],
    numerator_key: str,
    denominator: int,
) -> dict[str, dict[str, Any]]:
    out = {}
    for key, row in sorted(table.items(), key=lambda item: (-item[1].get(numerator_key, 0), item[0])):
        copied = dict(row)
        copied["fraction"] = copied.get(numerator_key, 0) / denominator if denominator else 0.0
        out[key] = copied
    return out


def pure_generator_marginal_rows(
    *,
    components: list[pe1.Component],
    generator_params: dict[int, pe1.GeneratorParams],
    base_payload: bytes,
    artifact_dir: Path,
    counts: list[int],
) -> list[dict[str, Any]]:
    ranked = sorted(components, key=lambda c: (-c.flip_mass, c.edge, c.pair, c.uid))
    rows: list[dict[str, Any]] = []
    for count in counts:
        selected = frozenset(comp.uid for comp in ranked[: min(count, len(ranked))])
        rep = pe1.build_generator_representation_from_params(
            components=components,
            params_by_uid=generator_params,
            selected_ids=selected,
            surface_id=f"generator_pair_bisector_prefix_{count}",
        )
        row, _section, _codec, _body = pe1.build_row(
            rep,
            components=components,
            base_payload=base_payload,
            artifact_dir=artifact_dir,
            store_best=False,
        )
        rows.append(row)
    return rows


def write_markdown(path: Path, receipt: dict[str, Any]) -> None:
    lines = [
        "# ddm_pe3 hybrid per-regime composition - 2026-08-05",
        "",
        "Status: **BYTE-CLOSED / PE3-SECTION-PARSE-BACK / RUNTIME-SURVIVAL-UNMEASURED / score_claim=false**.",
        "",
        f"Own-vehicle baseline: `S = {BASELINE_S} @ {BASELINE_BYTES:,} B {BASELINE_AXIS}`. No scorer job was run.",
        "",
        "## Typed Hybrid Table",
        "",
        "| point | section B | archive B | flip recall | band recall | modes | vs pe1 surgical flip | vs pe1 full bytes |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    pe1_surgical = receipt["pe1_references"]["surgical_winner"]
    pe1_full = receipt["pe1_references"]["full_winner"]
    for row in receipt["hybrid_points"]:
        lines.append(
            f"| `{row['surface_id']}` | `{row['section_bytes']}` | "
            f"`{row['archive_projection']['projected_archive_bytes']}` | "
            f"`{row['flip_recall']:.6f}` | `{row['mask_domain_fidelity']['recall']:.6f}` | "
            f"`{row['mode_counts']}` | "
            f"`{row['flip_recall'] - pe1_surgical['flip_recall']:+.6f}` | "
            f"`{row['section_bytes'] - pe1_full['section_bytes']:+d}` |"
        )
    lines.extend(
        [
            "",
            "## Candidate",
            "",
            f"- 75KB hybrid candidate: `{receipt['candidate_75kb']['archive_bytes']}` B, sha256 `{receipt['candidate_75kb']['archive_sha256']}`, path `{receipt['candidate_75kb']['submission_dir']}`.",
            f"- Knee hybrid projection: `{receipt['hybrid_knee']['section_bytes']}` section B, `{receipt['hybrid_knee']['archive_projection']['projected_archive_bytes']}` projected archive B.",
            "",
            "## RECALL EVIDENCE",
            "",
        ]
    )
    for item in receipt["recall_evidence"]:
        lines.append(f"- `{item['source']}`: {item['finding']} Plan impact: {item['plan_impact']}")
    lines.extend(["", "## ALL-SIGNAL DECOMPOSITION", ""])
    transport = receipt["all_signal_decomposition"]["transport"]
    lines.append(
        f"- xi transport: `{transport['transport_won_tracks']}/{transport['tracks']}` tracks won by raw record bytes; conditional record model delta `{transport['conditional_transport_hybrid']['delta_record_bytes']}` B before global entropy coding."
    )
    missing = receipt["all_signal_decomposition"]["missing_recall"]
    lines.append(
        f"- PE1 full missing recall: `{missing['missing_pixels']}/{missing['source_band_pixels']}` band px missing (`{1 - missing['recall']:.6f}`), decomposed in JSON by depth/regime/edge."
    )
    surgical = receipt["all_signal_decomposition"]["surgical_residual"]
    lines.append(
        f"- PE1 surgical residual: `{surgical['residual_flip_mass']}/{surgical['total_flip_mass']}` flip mass (`{surgical['residual_fraction']:.6f}`), decomposed in JSON by edge/depth/class."
    )
    row106 = receipt["all_signal_decomposition"]["row_106465"]
    lines.append(
        f"- 106,465 B row: `{row106['surface_id']}` traded band recall `{row106['recall_trade_vs_full']:.6f}` vs PE1 full; PE3 treats it as a depth-conditioned component source, not as a full-row winner."
    )
    lines.extend(["", "## Boundaries", ""])
    for boundary in receipt["boundaries"]:
        lines.append(f"- {boundary}")
    lines.extend(
        [
            "",
            "## NEXT-IF-RESUMED",
            "",
            receipt["next_if_resumed"],
            "",
            f"Own-vehicle frontier line: `S = {BASELINE_S} @ {BASELINE_BYTES:,} B {BASELINE_AXIS}`; pe3 did not run a scorer and did not move the contest pointer.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sub", type=Path, default=pe1.DEFAULT_BASE_SUB)
    parser.add_argument("--gt-cache", type=Path, default=pe1.DEFAULT_GT_CACHE)
    parser.add_argument("--current-argmax", type=Path, default=pe1.DEFAULT_CURRENT_ARGMAX)
    parser.add_argument("--pe1-receipt", type=Path, default=DEFAULT_PE1_RECEIPT)
    parser.add_argument("--g4-recurrence", type=Path, default=DEFAULT_G4_RECURRENCE)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    parser.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--depth-y1", type=float, default=190.0)
    parser.add_argument("--depth-y2", type=float, default=230.0)
    parser.add_argument("--transport-max-distance-px", type=float, default=24.0)
    parser.add_argument("--store-best", action="store_true")
    args = parser.parse_args(argv)

    args.research_dir.mkdir(parents=True, exist_ok=True)
    args.ssd_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    run_ssd = args.ssd_dir / run_id
    run_ssd.mkdir(parents=True, exist_ok=False)
    artifact_dir = run_ssd / "payloads"
    free = shutil.disk_usage(args.ssd_dir).free
    if free < 512 * 1024 * 1024:
        raise PE3Error(f"SSD storage preflight failed: {free} bytes free at {args.ssd_dir}")

    base_archive = args.base_sub / "archive.zip"
    base_sha = sha256_file(base_archive)
    if base_sha != pe1.BASELINE_ARCHIVE_SHA256:
        raise PE3Error(f"qo1 base archive SHA drift: {base_sha}")
    base_payload = pe1.bd1.read_archive_payload(base_archive)
    bulk, sections = parse_payload(base_payload)
    if len(sections) != 5:
        raise PE3Error(f"expected qo1 base with 5 sections, got {len(sections)}")

    pe1_receipt = load_json(args.pe1_receipt)
    lstars = pe1.open_stored_npy_memmap(args.gt_cache, "lstars")
    current = pe1.load_current_argmax(args.current_argmax)
    components, extraction = pe1.extract_components(lstars, current)
    all_ids = frozenset(comp.uid for comp in components)
    gen_rep, generator_params = pe1.build_generator_representation(
        components=components,
        lstars=lstars,
        selected_ids=all_ids,
    )
    transition_counts = transition_counts_for_components(args.g4_recurrence)
    infos, depth_meta = prepare_component_infos(
        components=components,
        lstars=lstars,
        current=current,
        generator_params=generator_params,
        y1=args.depth_y1,
        y2=args.depth_y2,
        transition_counts=transition_counts,
    )
    infos_by_uid = {info.component.uid: info for info in infos}
    ranked = sorted(components, key=lambda c: (-c.flip_mass, c.edge, c.pair, c.uid))
    generator_count = int(pe1_receipt["surgical_winner"]["mask_domain_fidelity"]["selected_components"])
    generator_prefix = {comp.uid for comp in ranked[:generator_count]}
    curve_residual = rank_curve_residual_after_generator(
        infos_by_uid=infos_by_uid,
        ranked_components=ranked,
        generator_prefix=generator_prefix,
    )

    hybrid_candidates: list[tuple[dict[str, Any], bytes]] = []
    for curve_count in (0, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000, 12000):
        modes = build_generator_plus_curve_modes(
            ranked_components=ranked,
            generator_count=generator_count,
            curve_residual=curve_residual,
            curve_count=curve_count,
        )
        build = build_hybrid(
            surface_id=f"pe3_hybrid_g{generator_count}_c{curve_count}",
            components=components,
            selected_modes=modes,
            infos_by_uid=infos_by_uid,
            policy="generator_pair_bisector surgical prefix + depth-conditioned curve residual waterfill",
            target_section_bytes=None,
            prefix_counts={"generator": generator_count, "curve_residual": curve_count},
        )
        row, section = price_hybrid(
            build,
            components=components,
            base_payload=base_payload,
            artifact_dir=artifact_dir,
            store_best=args.store_best,
        )
        hybrid_candidates.append((row, section))

    hybrid_75, section_75 = choose_best_at_target(hybrid_candidates, 75_000)
    hybrid_75["surface_id"] = f"{hybrid_75['surface_id']}_at_75kb"
    hybrid_75["target_section_bytes"] = 75_000
    hybrid_knee, section_knee = knee_point(hybrid_candidates)
    hybrid_knee["surface_id"] = f"{hybrid_knee['surface_id']}_knee"
    hybrid_points = [hybrid_75, hybrid_knee]

    curve_waterfill_modes = candidate_modes_curve_waterfill(infos, components, 10_000)
    curve_build = build_hybrid(
        surface_id="pe3_curve_only_waterfill_10000_control",
        components=components,
        selected_modes=curve_waterfill_modes,
        infos_by_uid=infos_by_uid,
        policy="all-signal control: per-edge curve-only waterfill, not the operator hybrid law",
        target_section_bytes=None,
        prefix_counts={"curve": len(curve_waterfill_modes)},
    )
    curve_control, _curve_section = price_hybrid(
        curve_build,
        components=components,
        base_payload=base_payload,
        artifact_dir=artifact_dir,
        store_best=False,
    )

    candidate_75 = write_candidate(
        base_sub=args.base_sub,
        candidate_dir=run_ssd / "sub_auto_pairbit_pe3_hybrid_75kb",
        section=section_75,
        section_name="pe3_hybrid_75kb",
    )
    candidate_knee = write_candidate(
        base_sub=args.base_sub,
        candidate_dir=run_ssd / "sub_auto_pairbit_pe3_hybrid_knee",
        section=section_knee,
        section_name="pe3_hybrid_knee",
    )

    surgical_selected = generator_prefix
    explicit_106 = next(row for row in pe1_receipt["representation_rows"] if row["surface_id"] == "explicit_curve_k16")
    explicit_full = pe1_receipt["full_winner"]
    decomposition = {
        "transport": decompose_transport(
            components=components,
            generator_params=generator_params,
            infos_by_uid=infos_by_uid,
            max_distance_px=args.transport_max_distance_px,
        ),
        "missing_recall": decompose_missing_recall(components=components, infos_by_uid=infos_by_uid),
        "surgical_residual": decompose_surgical_residual(
            components=components,
            infos_by_uid=infos_by_uid,
            surgical_selected=surgical_selected,
        ),
        "pure_generator_marginal_rows": pure_generator_marginal_rows(
            components=components,
            generator_params=generator_params,
            base_payload=base_payload,
            artifact_dir=artifact_dir,
            counts=[generator_count, generator_count + 1000, generator_count + 2000, generator_count + 4000],
        ),
        "row_106465": {
            "surface_id": explicit_106["surface_id"],
            "section_bytes": explicit_106["section_bytes"],
            "recall": explicit_106["mask_domain_fidelity"]["recall"],
            "full_reference": explicit_full["surface_id"],
            "full_reference_recall": explicit_full["mask_domain_fidelity"]["recall"],
            "recall_trade_vs_full": explicit_full["mask_domain_fidelity"]["recall"] - explicit_106["mask_domain_fidelity"]["recall"],
            "verdict": "hybrid component source for near/depth-conditioned curve rows; not the full-coverage winner because recall is lower.",
        },
        "curve_only_control": curve_control,
    }

    pe1_refs = {
        "full_winner": {
            "surface_id": explicit_full["surface_id"],
            "section_bytes": explicit_full["section_bytes"],
            "flip_recall": explicit_full["flip_recall"],
            "band_recall": explicit_full["mask_domain_fidelity"]["recall"],
        },
        "surgical_winner": {
            "surface_id": pe1_receipt["surgical_winner"]["surface_id"],
            "section_bytes": pe1_receipt["surgical_winner"]["section_bytes"],
            "flip_recall": pe1_receipt["surgical_winner"]["flip_recall"],
            "band_recall": pe1_receipt["surgical_winner"]["mask_domain_fidelity"]["recall"],
        },
        "full_120577_B": pe1_receipt["full_winner"],
        "surgical_67607_B": pe1_receipt["surgical_winner"],
    }

    hybrid_beats_surgical = hybrid_75["flip_recall"] > pe1_refs["surgical_winner"]["flip_recall"]
    receipt = {
        "schema": "ddm_pe3_hybrid_composition.v1",
        "built_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory / scorer-free mask-domain byte custody]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
        "storage_preflight": {
            "path": str(args.ssd_dir),
            "observed_free_bytes": free,
            "required_free_bytes": 512 * 1024 * 1024,
            "status": "PASS",
        },
        "base": {
            "submission_dir": str(args.base_sub),
            "archive_bytes": base_archive.stat().st_size,
            "archive_sha256": base_sha,
            "payload_sha256": sha256_bytes(base_payload),
            "joint_section_count": len(sections),
            "own_vehicle_S": BASELINE_S,
            "axis": BASELINE_AXIS,
        },
        "inputs": {
            "pe1_receipt": str(args.pe1_receipt),
            "gt_cache": str(args.gt_cache),
            "current_argmax": str(args.current_argmax),
            "g4_recurrence": str(args.g4_recurrence),
            "selection_mode": "n600 all pairs; no prefix; scorer-free",
            "shape": [pe1.N_PAIRS, pe1.SEG_H, pe1.SEG_W],
            "class_order": dict(enumerate(pe1.CLASS_NAMES)),
        },
        "section_format": {
            "magic": PE3_MAGIC.decode("ascii"),
            "version": PE3_VERSION,
            "kind": "hybrid_per_regime",
            "coordination": "No ddm_pe2_20260805 receipt was present at run time; PE3 claimed a distinct section magic/version and did not edit PE2 receiver code.",
        },
        "edge_extraction": extraction,
        "depth_policy": depth_meta,
        "pe1_references": pe1_refs,
        "hybrid_points": hybrid_points,
        "hybrid_75kb": hybrid_75,
        "hybrid_knee": hybrid_knee,
        "candidate_75kb": candidate_75,
        "candidate_knee": candidate_knee,
        "staged_batch_note_for_main": {
            "hybrid_beats_pe1_surgical_flip_recall": hybrid_beats_surgical,
            "fire_order": "If pe2 runtime receiver consumption lands for PE sections, add the PE3 75KB hybrid candidate to the staged scorer batch behind sq2; do not run a scorer from this arm.",
        },
        "all_signal_decomposition": decomposition,
        "recall_evidence": [
            {
                "source": ".omx/tmp/codex_runs/_common_contract.md",
                "finding": "PE3 is scorer-free because sq2 owns the scorer slot; original recall and real-coder evidence are required.",
                "plan_impact": "no SegNet/PoseNet forward was run; all rows are mask-domain byte custody with score_claim=false.",
            },
            {
                "source": ".omx/research/operator_directive_per_edge_optimality_criteria_20260805.md",
                "finding": "addendum 3 requires hybrid/per-level composition and all-signal decomposition of positives and negatives.",
                "plan_impact": "PE3 composes generator surgical prefix plus curve residuals and decomposes transport, recall misses, surgical residual, and the 106465 B row.",
            },
            {
                "source": ".omx/research/ddm_pe1_20260805/ddm_pe1_repr_race_receipt.json",
                "finding": "PE1 measured explicit_curve_k8 full at 120577 B / 0.984425 band recall and generator_pair_bisector_waterfill_75kb at 67607 B / 0.807339 flip recall.",
                "plan_impact": "these are the fixed pure-row references and the generator prefix size denominator.",
            },
            {
                "source": ".omx/research/ddm_g4_spatial_stationarity_n600_20260722T212138Z/ddm_g4_spatial_stationarity_receipt.json plus recurrence arrays",
                "finding": "static-in-image dominates aggregate flip stationarity; xi-proxy is scoped and not free physical BEV.",
                "plan_impact": "transport decomposition joins tracks to image-static/transient counts and does not promote xi transport globally.",
            },
            {
                "source": ".omx/research/ddm_fl1_perclass_flicker_floors_20260731.md",
                "finding": "GT flicker is formulation-scoped, not a hard floor, but ranks phase-faithfulness debt.",
                "plan_impact": "tiny/flicker-dominated fragments are priced omission candidates and counted in the missing/residual decomposition.",
            },
            {
                "source": ".venv/bin/python tools/list_canonical_equations.py --json and .omx/research/CANONICAL_RESEARCH_INDEX_20260629.md",
                "finding": "boundary/waterfill/transport laws exist, but no current PE3 hybrid section was already settled in the registry or index.",
                "plan_impact": "PE3 measures the section bytes instead of promoting a prior equation or a charter assumption.",
            },
        ],
        "boundaries": [
            "No SegNet/PoseNet scorer forward was run; sq2 owns the scorer slot.",
            "No upstream/ files were edited.",
            "No /tmp evidence is cited; durable candidate archives and payloads are on the SSD tier.",
            "PE3 candidate archives append counted video-derived sections; no data is hidden in free code.",
            "PE3 proves section parse-back and IX2 byte closure, not runtime RGB consumption; PE2 owns inflate_runner consumption.",
            "The all-signal curve-only control is a decomposition signal, not the operator hybrid-law candidate.",
            "All byte counts are measured on real serialized coder outputs over n600 cached masks.",
        ],
        "follow_on_disposition": [
            {
                "id": "pe3-hybrid-scorer-batch",
                "status": "QUEUED-WITH-FIRE-ORDER" if hybrid_beats_surgical else "FOLDED",
                "action": "After PE2 receiver consumption lands and sq2 releases the scorer slot, include the PE3 75KB candidate in the one-batch survival measurement if MAIN accepts the extra entrant.",
            },
            {
                "id": "pe3-curve-only-control",
                "status": "QUEUED-WITH-FIRE-ORDER",
                "action": "If the operator wants the all-signal surprise followed, build a PE4 explicit-curve surgical candidate; do not relabel it as the addendum-3 hybrid law.",
            },
            {
                "id": "pe3-transport-conditional",
                "status": "FOLDED",
                "action": "Record-byte conditional transport saves only in a subset before global entropy coding; archive-byte promotion requires a new coder, not a PE3 claim.",
            },
        ],
        "next_if_resumed": (
            "Start from PE3_RECEIPT_20260805.md and ddm_pe3_hybrid_receipt.json. "
            "Do not run a scorer while sq2 owns the slot. If PE2 lands runtime PE section consumption, add "
            "the PE3 75KB candidate under the SSD run directory to MAIN's staged survival batch or spawn a PE4 "
            "curve-only surgical control from the recorded decomposition."
        ),
    }

    json_path = args.research_dir / "ddm_pe3_hybrid_receipt.json"
    md_path = args.research_dir / "PE3_RECEIPT_20260805.md"
    json_path.write_text(json.dumps(jsonable(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, receipt)
    print(
        json.dumps(
            {
                "receipt": str(json_path),
                "markdown": str(md_path),
                "candidate_75kb": candidate_75["submission_dir"],
                "hybrid_75_section_bytes": hybrid_75["section_bytes"],
                "hybrid_75_flip_recall": hybrid_75["flip_recall"],
                "own_vehicle_frontier": f"S = {BASELINE_S} @ {BASELINE_BYTES} B {BASELINE_AXIS}",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
