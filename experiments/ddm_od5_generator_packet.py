#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""OD5 generator-coordinate Weak Stage-1 packet pricing.

This is a scorer-free OD5 bridge from OD4's sparse per-flip packet negative to
the surviving generator-coordinate formulation.  It reconstructs PE1/PE3
generator masks, applies an ST2-style context targeter only as a residual
selector, measures retained OD2 Stage-1 fixes on the same n32 rows, and prices
real section packets.  It does not run SegNet, PoseNet, upstream/evaluate.py, a
receiver-closed inflate, or full n600 scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "src", REPO / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ddm_pe1_per_edge_partition_race as pe1  # noqa: E402
import ddm_pe3_hybrid_composition as pe3  # noqa: E402
import ddm_st2_scorer_native_student as st2  # noqa: E402
from ddm_et1_ph1_block16_on_our_vehicle import solve_blocks, translate_blocks  # noqa: E402
from tac.optimization import ddm_od4_weak_stage1_packet as od4  # noqa: E402

DEFAULT_OD2_DIR: Final = REPO / ".omx/research/ddm_od2_20260805"
DEFAULT_OD2_JSON: Final = DEFAULT_OD2_DIR / "od2_js1_n32_cprime_k4.json"
DEFAULT_PAIR_SELECTION: Final = DEFAULT_OD2_DIR / "PAIR_SELECTION.json"
DEFAULT_ARGMAX_CACHE: Final = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
DEFAULT_GT_CACHE: Final = pe1.DEFAULT_GT_CACHE
DEFAULT_PE1_RECEIPT: Final = REPO / ".omx/research/ddm_pe1_20260805/ddm_pe1_repr_race_receipt.json"
DEFAULT_PE3_RECEIPT: Final = REPO / ".omx/research/ddm_pe3_20260805/ddm_pe3_hybrid_receipt.json"
DEFAULT_ST2_RECEIPT: Final = REPO / ".omx/research/ddm_st2_20260805/ddm_st2_receipt.json"
DEFAULT_G4_RECURRENCE: Final = pe3.DEFAULT_G4_RECURRENCE
DEFAULT_GT_MARGIN_F16: Final = st2.DEFAULT_GT_MARGIN_F16
DEFAULT_HOPE_CAPACITY_TABLE: Final = st2.DEFAULT_HOPE_CAPACITY_TABLE
DEFAULT_RESEARCH_DIR: Final = REPO / ".omx/research/ddm_od5_20260805"
DEFAULT_SSD_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_od5_20260805")
OD4_RATE_COST_OVER_SEG_WIN: Final = 0.7106


class OD5Error(ValueError):
    """OD5 scorer-free packet build failed a typed invariant."""


@dataclass(frozen=True, slots=True)
class TargetPair:
    pair: int
    target_argmax: np.ndarray
    full_record: od4.SparsePairCorrections
    build_row: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, od4.CoderRow):
        return value.as_json()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise OD5Error(f"JSON root is not an object: {path}")
    return data


def _storage_preflight(path: Path, required_free_bytes: int) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "required_free_bytes": int(required_free_bytes),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "ok": bool(usage.free >= required_free_bytes),
    }


def _best_coder(rows: tuple[od4.CoderRow, ...]) -> od4.CoderRow:
    candidates = [row for row in rows if row.parseback_exact and row.bytes > 0]
    if not candidates:
        raise OD5Error("no OD5 packet coder row survived parse-back")
    return min(candidates, key=lambda row: row.bytes)


def _subset_frame_records(frame_records: tuple[bytes, ...], pairs: list[int]) -> bytes:
    out = bytearray()
    out += pe1.varint(len(pairs))
    for pair in pairs:
        record = frame_records[pair]
        out += pe1.varint(int(pair))
        out += pe1.varint(len(record))
        out += record
    return bytes(out)


def _masks_by_pair(
    *,
    components: list[pe1.Component],
    component_masks: dict[int, np.ndarray],
    pairs: list[int],
) -> dict[int, np.ndarray]:
    wanted = set(pairs)
    out = {pair: np.zeros((od4.SEG_H, od4.SEG_W), dtype=bool) for pair in pairs}
    for comp in components:
        if comp.pair not in wanted:
            continue
        mask = component_masks.get(comp.uid)
        if mask is not None:
            out[comp.pair] |= np.asarray(mask, dtype=bool)
    return out


def _derive_target_pair(
    *,
    pair: int,
    od2_row: dict[str, Any],
    current: np.ndarray,
    gt: np.ndarray,
    block: int,
    rmax: int,
) -> TargetPair:
    before = int((current != gt).sum())
    if before != int(od2_row["flips_before"]):
        raise OD5Error(f"pair {pair}: cached current flips {before} != OD2 {od2_row['flips_before']}")
    offsets = solve_blocks(current, gt, block, rmax)
    target = translate_blocks(current, offsets.reshape(-1, 2), block)
    n_described = before - int((target != gt).sum())
    if n_described != int(od2_row["n_described"]):
        raise OD5Error(f"pair {pair}: n_described {n_described} != OD2 {od2_row['n_described']}")
    desired = before - int(od2_row["stage1"]["flips_after"])
    full_record = od4.select_sparse_corrections(
        pair=pair,
        current_argmax=current,
        gt_argmax=gt,
        target_argmax=target,
        desired_fix_count=desired,
        fraction=1.0,
    )
    return TargetPair(
        pair=pair,
        target_argmax=target,
        full_record=full_record,
        build_row={
            "pair": pair,
            "flips_before": before,
            "od2_flips_after": int(od2_row["stage1"]["flips_after"]),
            "od2_fix_count": desired,
            "n_described": n_described,
            "full_od2_stage1_record_count": full_record.count,
            "offsets_sha256": _sha256_bytes(np.ascontiguousarray(offsets.astype(np.int8)).tobytes()),
        },
    )


def _selected_flats_from_mask(
    target_pairs: dict[int, TargetPair],
    masks: dict[int, np.ndarray],
    pairs: list[int],
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for pair in pairs:
        full = np.asarray(target_pairs[pair].full_record.flat_indices, dtype=np.int64)
        if full.size == 0:
            out[pair] = full
            continue
        mask = np.asarray(masks[pair], dtype=bool).reshape(-1)
        out[pair] = np.sort(full[mask[full]])
    return out


def _union_selected(
    pairs: list[int],
    *selected: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for pair in pairs:
        chunks = [np.asarray(sel.get(pair, np.array([], dtype=np.int64)), dtype=np.int64) for sel in selected]
        if chunks:
            out[pair] = np.unique(np.concatenate(chunks)).astype(np.int64)
        else:
            out[pair] = np.array([], dtype=np.int64)
    return out


def _residual_selected(
    *,
    target_pairs: dict[int, TargetPair],
    base_selected: dict[int, np.ndarray],
    pairs: list[int],
    fraction: float,
) -> dict[int, np.ndarray]:
    if not 0.0 <= fraction <= 1.0:
        raise OD5Error("innovation fraction must be in [0, 1]")
    out: dict[int, np.ndarray] = {}
    for pair in pairs:
        full = np.asarray(target_pairs[pair].full_record.flat_indices, dtype=np.int64)
        base = np.asarray(base_selected.get(pair, np.array([], dtype=np.int64)), dtype=np.int64)
        remaining = np.setdiff1d(full, base, assume_unique=True)
        keep = int(np.floor(remaining.size * fraction + 1e-9))
        out[pair] = np.sort(remaining[:keep])
    return out


def _records_from_selected(
    *,
    target_pairs: dict[int, TargetPair],
    selected: dict[int, np.ndarray],
    pairs: list[int],
) -> list[od4.SparsePairCorrections]:
    records: list[od4.SparsePairCorrections] = []
    for pair in pairs:
        flats = np.sort(np.asarray(selected.get(pair, np.array([], dtype=np.int64)), dtype=np.int64))
        target_flat = target_pairs[pair].target_argmax.reshape(-1)
        labels = tuple(int(value) for value in target_flat[flats])
        records.append(od4.SparsePairCorrections(pair, tuple(int(value) for value in flats), labels))
    return records


def _decompress_st2_payload(best: dict[str, Any], *, raw_len: int) -> bytes:
    payload = Path(str(best["artifact_path"])).read_bytes()
    codec = str(best["codec"])
    if codec == "brotli-q11":
        raw = brotli.decompress(payload)
    elif codec == "zlib-9":
        raw = zlib.decompress(payload)
    elif codec == "lzma1-raw":
        raw = od4.unlzma1_raw(payload, raw_len)
    else:
        raise OD5Error(f"unknown ST2 codec {codec!r}")
    if len(raw) != raw_len:
        raise OD5Error(f"ST2 raw payload length mismatch: {len(raw)} != {raw_len}")
    return raw


def _context_selected_flats(
    *,
    st2_receipt: dict[str, Any],
    target_pairs: dict[int, TargetPair],
    pairs: list[int],
    current: np.ndarray,
    gt: np.ndarray,
    margins: np.ndarray,
    capacity_prior: dict[str, Any],
) -> tuple[dict[int, np.ndarray], dict[str, Any], bytes]:
    selected = st2_receipt["leg_st2_scorer_native_student"]["selected"]
    payload = selected["payload"]
    raw = _decompress_st2_payload(payload["best"], raw_len=int(payload["raw_bytes"]))
    header, qlogits = st2.decode_scorer_native_payload(raw)
    feature = selected["feature"]
    feature_mode = str(feature["mode"])
    bucket_count = int(selected["bucket_count"])
    context_radius = int(feature["context_radius"])
    qscale = float(header["qscale"])
    threshold = float(selected["threshold_from_train_holdout"]["threshold"])
    road_lane_frequency, all_flip_frequency = st2.st1.compute_frequency_maps(gt, current)

    pair_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    x_chunks: list[np.ndarray] = []
    flat_chunks: list[np.ndarray] = []
    for pair in pairs:
        full = np.asarray(target_pairs[pair].full_record.flat_indices, dtype=np.int64)
        if full.size == 0:
            continue
        pair_chunks.append(np.full(full.size, int(pair), dtype=np.int16))
        y_chunks.append((full // od4.SEG_W).astype(np.int16))
        x_chunks.append((full % od4.SEG_W).astype(np.int16))
        flat_chunks.append(full)
    if not pair_chunks:
        return {pair: np.array([], dtype=np.int64) for pair in pairs}, {}, raw

    p_arr = np.concatenate(pair_chunks)
    y_arr = np.concatenate(y_chunks)
    x_arr = np.concatenate(x_chunks)
    flats = np.concatenate(flat_chunks)
    hashes = st2.model_hashes(
        feature_mode=feature_mode,
        current=current,
        margins=margins,
        pairs=p_arr,
        y=y_arr,
        x=x_arr,
        road_lane_frequency=road_lane_frequency,
        all_flip_frequency=all_flip_frequency,
        capacity_prior=capacity_prior,
        bucket_count=bucket_count,
        context_radius=context_radius,
    )
    probs = st2.st1.sigmoid(qlogits[hashes].astype(np.float32) / np.float32(qscale))
    keep = probs >= np.float32(threshold)
    out = {pair: [] for pair in pairs}
    for pair, flat, kept in zip(p_arr.tolist(), flats.tolist(), keep.tolist(), strict=True):
        if kept:
            out[int(pair)].append(int(flat))
    selected_by_pair = {pair: np.asarray(sorted(values), dtype=np.int64) for pair, values in out.items()}
    meta = {
        "feature_mode": feature_mode,
        "context_radius": context_radius,
        "bucket_count": bucket_count,
        "threshold": threshold,
        "qscale": qscale,
        "payload_best": payload["best"],
        "payload_raw_bytes": int(payload["raw_bytes"]),
        "payload_raw_sha256": _sha256_bytes(raw),
        "candidate_od2_fix_points": int(p_arr.size),
        "selected_fix_points": int(sum(arr.size for arr in selected_by_pair.values())),
        "receiver_legality": (
            "targeter-only in OD5: this ST2 selected row mixes scorer-native cached fields into the hash, "
            "so it is not claimed as a decoder-legal feature surface."
        ),
    }
    return selected_by_pair, meta, raw


def _build_generator_surfaces(
    *,
    components: list[pe1.Component],
    lstars: np.ndarray,
    current: np.ndarray,
    pe1_receipt: dict[str, Any],
    g4_recurrence: Path,
    depth_y1: float,
    depth_y2: float,
) -> dict[str, Any]:
    ranked = sorted(components, key=lambda comp: (-comp.flip_mass, comp.edge, comp.pair, comp.uid))
    generator_count = int(pe1_receipt["surgical_winner"]["mask_domain_fidelity"]["selected_components"])
    all_ids = frozenset(comp.uid for comp in components)
    _all_gen_rep, generator_params = pe1.build_generator_representation(
        components=components,
        lstars=lstars,
        selected_ids=all_ids,
    )
    generator_ids = frozenset(comp.uid for comp in ranked[:generator_count])
    generator_rep = pe1.build_generator_representation_from_params(
        components=components,
        params_by_uid=generator_params,
        selected_ids=generator_ids,
        surface_id="od5_generator_pair_bisector_prefix",
    )
    transition_counts = pe3.transition_counts_for_components(g4_recurrence)
    infos, depth_meta = pe3.prepare_component_infos(
        components=components,
        lstars=lstars,
        current=current,
        generator_params=generator_params,
        y1=depth_y1,
        y2=depth_y2,
        transition_counts=transition_counts,
    )
    infos_by_uid = {info.component.uid: info for info in infos}
    generator_prefix = {comp.uid for comp in ranked[:generator_count]}
    curve_residual = pe3.rank_curve_residual_after_generator(
        infos_by_uid=infos_by_uid,
        ranked_components=ranked,
        generator_prefix=generator_prefix,
    )
    hybrid_builds: dict[str, pe3.HybridBuild] = {}
    for curve_count, name in ((750, "hybrid75"), (6000, "hybrid_knee")):
        modes = pe3.build_generator_plus_curve_modes(
            ranked_components=ranked,
            generator_count=generator_count,
            curve_residual=curve_residual,
            curve_count=curve_count,
        )
        hybrid_builds[name] = pe3.build_hybrid(
            surface_id=f"od5_pe3_hybrid_g{generator_count}_c{curve_count}",
            components=components,
            selected_modes=modes,
            infos_by_uid=infos_by_uid,
            policy="generator_pair_bisector prefix + depth-conditioned curve residual",
            target_section_bytes=75_000 if curve_count == 750 else None,
            prefix_counts={"generator": generator_count, "curve_residual": curve_count},
        )
    return {
        "ranked_components": ranked,
        "generator_count": generator_count,
        "generator_rep": generator_rep,
        "hybrid75": hybrid_builds["hybrid75"],
        "hybrid_knee": hybrid_builds["hybrid_knee"],
        "depth_meta": depth_meta,
    }


def _store_packet(path: Path, packet: bytes, best: od4.CoderRow) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(packet)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "selected_coder": best.codec,
        "selected_coder_bytes": best.bytes,
        "selected_coder_sha256": best.sha256,
    }


def _rung_receipt(
    *,
    name: str,
    sections: list[od4.OD5Section],
    selected_flats: dict[int, np.ndarray],
    target_pairs: dict[int, TargetPair],
    od2_rows_by_pair: dict[int, dict[str, Any]],
    pairs: list[int],
    current_argmax: np.ndarray,
    gt_argmax: np.ndarray,
    projected_n600_packet_bytes: int,
    projection_scope: str,
    ssd_dir: Path,
) -> dict[str, Any]:
    records = _records_from_selected(target_pairs=target_pairs, selected=selected_flats, pairs=pairs)
    sparse_packet = od4.serialize_sparse_packet(records)
    sparse_parsed = od4.parse_sparse_packet(sparse_packet)
    fidelity = od4.fidelity_for_packet(
        current_argmax=current_argmax,
        gt_argmax=gt_argmax,
        packet=sparse_parsed,
        od2_rows_by_pair=od2_rows_by_pair,
    )
    packet = od4.serialize_od5_packet(sections)
    parsed = od4.parse_od5_packet(packet)
    if od4.serialize_od5_packet(parsed.sections) != packet:
        raise OD5Error(f"{name}: OD5 section packet did not reserialize exactly")
    coder_rows = od4.race_packet_coders(packet)
    best = _best_coder(coder_rows)
    artifact = _store_packet(ssd_dir / "packets" / f"{name}.od5.raw_packet", packet, best)
    totals = fidelity["totals"]
    projections = {
        "seg_only": od4.projection_rows_with_projected_packet_bytes(
            n32_packet_bytes=best.bytes,
            n600_packet_bytes_projected=projected_n600_packet_bytes,
            n_pairs=len(pairs),
            retained_fix_count=int(totals["retained_fix_count"]),
            include_od2_pose_credit=False,
            projection_scope=projection_scope,
        ),
        "with_od2_pose_credit": od4.projection_rows_with_projected_packet_bytes(
            n32_packet_bytes=best.bytes,
            n600_packet_bytes_projected=projected_n600_packet_bytes,
            n_pairs=len(pairs),
            retained_fix_count=int(totals["retained_fix_count"]),
            include_od2_pose_credit=True,
            projection_scope=projection_scope,
        ),
    }
    return {
        "name": name,
        "packet": {
            "schema": od4.OD5_PACKET_SCHEMA,
            "raw_packet_bytes": len(packet),
            "raw_packet_sha256": od4.sha256_bytes(packet),
            "section_count": parsed.section_count,
            "sections": [
                {
                    "name": section.name,
                    "payload_bytes": len(section.payload),
                    "payload_sha256": od4.sha256_bytes(section.payload),
                }
                for section in sections
            ],
            "artifact": artifact,
            "coder_race": [row.as_json() for row in coder_rows],
            "best_coder": best.as_json(),
        },
        "constraint_replay": {
            "sparse_packet_schema": od4.PACKET_SCHEMA,
            "sparse_packet_bytes": len(sparse_packet),
            "sparse_packet_sha256": od4.sha256_bytes(sparse_packet),
            "sparse_constraint_count": sparse_parsed.correction_count,
        },
        "fidelity": fidelity,
        "projected_n600_packet_bytes": int(projected_n600_packet_bytes),
        "projection_scope": projection_scope,
        "projection_seg_only": projections["seg_only"],
        "projection_with_od2_stage2_pose_credit": projections["with_od2_pose_credit"],
    }


def _price_table_markdown(receipt: dict[str, Any]) -> str:
    lines = [
        "| rung | exact n32 bytes | projected n600 bytes | retained fixes | retained eta | S seg-only | S w/ OD2 pose credit | rate/seg win | vs OD4 71% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in receipt["rungs"]:
        totals = row["fidelity"]["totals"]
        proj0 = row["projection_seg_only"]
        proj1 = row["projection_with_od2_stage2_pose_credit"]
        ratio = float(proj1["rate_cost_over_seg_win"])
        comparison = "better" if ratio < OD4_RATE_COST_OVER_SEG_WIN else "worse"
        lines.append(
            "| "
            f"{row['name']} | "
            f"{row['packet']['best_coder']['bytes']} | "
            f"{proj1['packet_bytes_n600_projected']} | "
            f"{totals['retained_fix_count']} | "
            f"{totals['eta_receiver']:.6f} | "
            f"{proj0['projected_s']:.9f} | "
            f"{proj1['projected_s']:.9f} | "
            f"{ratio:.3f} | "
            f"{comparison} |"
        )
    return "\n".join(lines)


def _write_gate_script(path: Path) -> None:
    content = """#!/usr/bin/env bash
set -euo pipefail

# OD5 queued scorer gate. od3 owns the scorer slot at OD5 build time, so this
# is a fire-order artifact only. Bind SUB_DIR after a receiver-closed staged
# submission exists.
SUB_DIR="${SUB_DIR:?set SUB_DIR to the receiver-closed staged submission directory}"
OUT="${OUT:-.omx/research/ddm_od5_20260805/od5_receiver_gate_receipt.json}"

.venv/bin/python experiments/ddm_fz2_byteclose_and_eval.py \\
  --sub-dir "${SUB_DIR}" \\
  --out "${OUT}" \\
  --inflate-out "${SUB_DIR}/inflated" \\
  --device cpu \\
  --batch-size 16 \\
  --num-threads 6
"""
    _atomic_write_text(path, content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_markdown(path: Path, receipt: dict[str, Any]) -> None:
    best = receipt["best_rung_by_projected_s_with_od2_pose_credit"]
    best_projection = best["projection_with_od2_stage2_pose_credit"]
    delta = best_projection["projected_s"] - od4.CURRENT_OWN_S
    verdict = "below" if delta < 0 else "above"
    receipt_json = Path(receipt["receipt_json_path"])
    md = f"""# OD5 generator-coordinate Weak Stage-1 packet receipt - 2026-08-05

Status: `SCORER_FREE_GENERATOR_COORDINATE_PACKET_PRICED / NO FRONTIER MOVE`.

Axis: `[macOS-CPU cache-derived advisory / scorer-free mask-domain replay]`.
`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`, `scorer_forwards_run=0`.

## Answer First

OD5 priced the generator-coordinate survivor from the OD4 pivot on OD2's same n32 rows. The best projected row is `{best['name']}`: `{best['packet']['best_coder']['bytes']}` exact n32 OD5 packet bytes, `{best_projection['packet_bytes_n600_projected']}` projected n600 packet bytes, `{best['fidelity']['totals']['retained_fix_count']}` retained OD2 Stage-1 fixes, eta `{best['fidelity']['totals']['eta_receiver']:.6f}`.

The best row projects to `S = {best_projection['projected_s']:.9f}` with OD2 Stage-2 pose credit, `{verdict}` the live own line by `{delta:.9f}`. This is not a score and not promotion-eligible: it is mask-domain, n32, uses measured PE1/PE3/ST2 component bytes plus linear residual projections, and has no receiver-closed RGB/inflate/scorer survival.

## Price Table

{_price_table_markdown(receipt)}

OD4 sparse per-flip packet ratio was `0.7106` rate-cost/seg-win. OD5 improves that ratio only when the generator-coordinate section retains enough OD2 fixes; the ST2 selected row is recorded as targeter-only because its best 3.6 KB hash mixes scorer-native cached fields.

## RECALL EVIDENCE

| source | recalled fact | plan impact |
|---|---|---|
| `_common_contract.md`, `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `.omx/state/main_hot_state.md` | od3 owns scorer; current own line is `S=0.7539807296911207 @ 357,836 B`; protected files and staged index are off limits. | Built scorer-free OD5 only and queued the scorer gate. |
| `OD4_WEAK_PACKET_RECEIPT.md` | Sparse per-flip OD4 best projected `S=0.761509399`; projected 104,775 B cost ate about 71% of the seg win. | Used OD4 as the falsifier baseline and avoided per-flip where-tax as the primary packet. |
| `GC17_CONVOCATION_RECEIPT.md` and `operator_directive_per_edge_optimality_criteria_20260805.md` | Weak Stage-1 packet must carry generator/worldsheet coordinates and residual innovation, not dense solved fields; per-edge address tax is the live crux. | Packet sections carry generator/hybrid record streams and residual sparse innovation only. |
| `ddm_pe1_20260805`, `ddm_pe3_20260805`, `ddm_st2_20260805`, `ddm_g4` | PE1 generator prefix is 67,607 B for 80.73% n600 flip mass; PE3 hybrid75 is 74,408 B for 83.15%; ST2 selected table is 3,602 B but scorer-native. | Used measured component bytes for n600 projections and rederived OD2 retained eta rather than transferring flip recall. |
| searches over `.omx/research`, `.omx/state`, and canonical equations for `worldsheet`, `Weak Stage-1`, `receiver-close`, `address law`, `#941` | Existing evidence says receiver closure and counted bytes dominate; ST2 is targeter/prior until decoder-legal features exist. | Labeled the context row targeter-only and left scorer/inflate fire-order queued. |

## SHA Table

| artifact | bytes | sha256 |
|---|---:|---|
| `{receipt['source_files']['od2_json']['path']}` | {receipt['source_files']['od2_json']['bytes']} | `{receipt['source_files']['od2_json']['sha256']}` |
| `{receipt['source_files']['pair_selection']['path']}` | {receipt['source_files']['pair_selection']['bytes']} | `{receipt['source_files']['pair_selection']['sha256']}` |
| `{receipt['source_files']['pe1_receipt']['path']}` | {receipt['source_files']['pe1_receipt']['bytes']} | `{receipt['source_files']['pe1_receipt']['sha256']}` |
| `{receipt['source_files']['pe3_receipt']['path']}` | {receipt['source_files']['pe3_receipt']['bytes']} | `{receipt['source_files']['pe3_receipt']['sha256']}` |
| `{receipt['source_files']['st2_receipt']['path']}` | {receipt['source_files']['st2_receipt']['bytes']} | `{receipt['source_files']['st2_receipt']['sha256']}` |
| `{receipt_json}` | {receipt_json.stat().st_size if receipt_json.exists() else 0} | `{_sha256_file(receipt_json) if receipt_json.exists() else 'pending'}` |

## NEXT_IF_RESUMED

1. Replace the ST2 scorer-native targeter row with a decoder-legal context feature, or keep it only as an encoder-side residual ordering prior.
2. Re-run the OD5 script against od3 terminal fields once the scorer slot is free and the terminal field receipt exists.
3. Build a receiver-closed RGB/inflate OD5 candidate only after the n32 mask-domain row projects below the live line and the feature surface is decoder-legal.
4. Fire `.omx/research/ddm_od5_20260805/OD5_SCORER_GATE_FIRE_ORDER.sh` only with `SUB_DIR` bound to that exact receiver-closed staged submission.

## Boundaries

- No `upstream/evaluate.py`, SegNet, PoseNet, full n600 scorer job, contest-CPU, or contest-CUDA run.
- PE1/PE3 n600 bytes are measured prior component section bytes; OD5 composite n600 bytes are projected component sums, not exact archive bytes.
- OD2 pose credit is inherited for the same-row projection only and was not remeasured by OD5.
- ST2 context is targeter-only in the best selected row because it mixes scorer-native cached fields into the hash.
- This is not a legal archive row and does not move the frontier.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
"""
    _atomic_write_text(path, md)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--od2-json", type=Path, default=DEFAULT_OD2_JSON)
    ap.add_argument("--pair-selection", type=Path, default=DEFAULT_PAIR_SELECTION)
    ap.add_argument("--argmax-cache", type=Path, default=DEFAULT_ARGMAX_CACHE)
    ap.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    ap.add_argument("--pe1-receipt", type=Path, default=DEFAULT_PE1_RECEIPT)
    ap.add_argument("--pe3-receipt", type=Path, default=DEFAULT_PE3_RECEIPT)
    ap.add_argument("--st2-receipt", type=Path, default=DEFAULT_ST2_RECEIPT)
    ap.add_argument("--g4-recurrence", type=Path, default=DEFAULT_G4_RECURRENCE)
    ap.add_argument("--gt-margin-f16", type=Path, default=DEFAULT_GT_MARGIN_F16)
    ap.add_argument("--hope-capacity-table", type=Path, default=DEFAULT_HOPE_CAPACITY_TABLE)
    ap.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    ap.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--rmax", type=int, default=5)
    ap.add_argument("--depth-y1", type=float, default=190.0)
    ap.add_argument("--depth-y2", type=float, default=230.0)
    ap.add_argument("--innovation-fractions", default="0.25,0.50,1.00")
    args = ap.parse_args(argv)

    args.research_dir.mkdir(parents=True, exist_ok=True)
    storage = _storage_preflight(args.ssd_dir, required_free_bytes=512 * 1024 * 1024)
    if not storage["ok"]:
        raise OD5Error(f"SSD storage preflight failed: {storage}")
    run_id = args.run_id or datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    run_ssd = args.ssd_dir / run_id
    run_ssd.mkdir(parents=True, exist_ok=False)

    od2_json = _load_json(args.od2_json)
    pair_selection = _load_json(args.pair_selection)
    pe1_receipt = _load_json(args.pe1_receipt)
    pe3_receipt = _load_json(args.pe3_receipt)
    st2_receipt = _load_json(args.st2_receipt)
    rows = od2_json.get("rows")
    if not isinstance(rows, list) or not rows:
        raise OD5Error("OD2 JSON has no rows")
    od2_rows_by_pair = {int(row["pair"]): row for row in rows}
    pairs = [int(pair) for pair in pair_selection["pairs"]]
    missing = [pair for pair in pairs if pair not in od2_rows_by_pair]
    if missing:
        raise OD5Error(f"OD2 JSON missing selected pairs: {missing}")

    current_argmax = np.load(args.argmax_cache / "cx1_argmax_n600.npy", mmap_mode="r")
    gt_argmax = np.load(args.argmax_cache / "gt_argmax_n600.npy", mmap_mode="r")
    lstars = pe1.open_stored_npy_memmap(args.gt_cache, "lstars")
    if not np.array_equal(np.asarray(lstars[pairs], dtype=np.uint8), np.asarray(gt_argmax[pairs], dtype=np.uint8)):
        raise OD5Error("PE1 GT cache and OD2 argmax cache differ on selected n32 pairs")

    target_pairs = {
        pair: _derive_target_pair(
            pair=pair,
            od2_row=od2_rows_by_pair[pair],
            current=np.asarray(current_argmax[pair], dtype=np.uint8),
            gt=np.asarray(gt_argmax[pair], dtype=np.uint8),
            block=args.block,
            rmax=args.rmax,
        )
        for pair in pairs
    }
    target_denominators = {
        "od2_n_described": int(sum(tp.build_row["n_described"] for tp in target_pairs.values())),
        "od2_stage1_fix_count": int(sum(tp.full_record.count for tp in target_pairs.values())),
        "pairs": len(pairs),
    }

    components, extraction = pe1.extract_components(lstars, current_argmax)
    surfaces = _build_generator_surfaces(
        components=components,
        lstars=lstars,
        current=current_argmax,
        pe1_receipt=pe1_receipt,
        g4_recurrence=args.g4_recurrence,
        depth_y1=args.depth_y1,
        depth_y2=args.depth_y2,
    )
    generator_rep: pe1.RepresentationBuild = surfaces["generator_rep"]
    hybrid75: pe3.HybridBuild = surfaces["hybrid75"]
    hybrid_knee: pe3.HybridBuild = surfaces["hybrid_knee"]

    generator_masks = _masks_by_pair(components=components, component_masks=generator_rep.component_masks, pairs=pairs)
    hybrid75_masks = _masks_by_pair(components=components, component_masks=hybrid75.component_masks, pairs=pairs)
    hybrid_knee_masks = _masks_by_pair(components=components, component_masks=hybrid_knee.component_masks, pairs=pairs)
    generator_selected = _selected_flats_from_mask(target_pairs, generator_masks, pairs)
    hybrid75_selected = _selected_flats_from_mask(target_pairs, hybrid75_masks, pairs)
    hybrid_knee_selected = _selected_flats_from_mask(target_pairs, hybrid_knee_masks, pairs)

    margins = st2.load_margin_memmap(args.gt_margin_f16)
    capacity_prior = st2.load_road_lane_capacity_prior(args.hope_capacity_table)
    context_selected, context_meta, context_raw = _context_selected_flats(
        st2_receipt=st2_receipt,
        target_pairs=target_pairs,
        pairs=pairs,
        current=current_argmax,
        gt=gt_argmax,
        margins=margins,
        capacity_prior=capacity_prior,
    )
    hybrid75_context_selected = _union_selected(pairs, hybrid75_selected, context_selected)

    pe1_generator_projected = int(pe1_receipt["surgical_winner"]["parse_back"]["section_bytes"])
    pe3_hybrid75_projected = int(pe3_receipt["hybrid_75kb"]["section_bytes"])
    pe3_hybrid_knee_projected = int(pe3_receipt["hybrid_knee"]["section_bytes"])
    st2_context_projected = int(
        st2_receipt["leg_st2_scorer_native_student"]["selected"]["payload"]["best"]["bytes"]
    )

    generator_subset_raw = _subset_frame_records(generator_rep.frame_records, pairs)
    hybrid75_subset_raw = _subset_frame_records(hybrid75.frame_records, pairs)
    hybrid_knee_subset_raw = _subset_frame_records(hybrid_knee.frame_records, pairs)

    rungs: list[dict[str, Any]] = []
    rungs.append(
        _rung_receipt(
            name="generator_only_pe1_prefix",
            sections=[od4.OD5Section("pe1_generator_coords_n32", generator_subset_raw)],
            selected_flats=generator_selected,
            target_pairs=target_pairs,
            od2_rows_by_pair=od2_rows_by_pair,
            pairs=pairs,
            current_argmax=current_argmax,
            gt_argmax=gt_argmax,
            projected_n600_packet_bytes=pe1_generator_projected,
            projection_scope="PE1 measured n600 generator-prefix section bytes; OD5 exact n32 subset packet for selected pairs",
            ssd_dir=run_ssd,
        )
    )
    rungs.append(
        _rung_receipt(
            name="generator_hybrid75",
            sections=[od4.OD5Section("pe3_hybrid75_coords_n32", hybrid75_subset_raw)],
            selected_flats=hybrid75_selected,
            target_pairs=target_pairs,
            od2_rows_by_pair=od2_rows_by_pair,
            pairs=pairs,
            current_argmax=current_argmax,
            gt_argmax=gt_argmax,
            projected_n600_packet_bytes=pe3_hybrid75_projected,
            projection_scope="PE3 measured n600 hybrid75 section bytes; OD5 exact n32 subset packet for selected pairs",
            ssd_dir=run_ssd,
        )
    )
    rungs.append(
        _rung_receipt(
            name="generator_hybrid_knee",
            sections=[od4.OD5Section("pe3_hybrid_knee_coords_n32", hybrid_knee_subset_raw)],
            selected_flats=hybrid_knee_selected,
            target_pairs=target_pairs,
            od2_rows_by_pair=od2_rows_by_pair,
            pairs=pairs,
            current_argmax=current_argmax,
            gt_argmax=gt_argmax,
            projected_n600_packet_bytes=pe3_hybrid_knee_projected,
            projection_scope="PE3 measured n600 hybrid-knee section bytes; OD5 exact n32 subset packet for selected pairs",
            ssd_dir=run_ssd,
        )
    )
    rungs.append(
        _rung_receipt(
            name="generator_context_targeter",
            sections=[
                od4.OD5Section("pe3_hybrid75_coords_n32", hybrid75_subset_raw),
                od4.OD5Section("st2_context_targeter_table", context_raw),
            ],
            selected_flats=hybrid75_context_selected,
            target_pairs=target_pairs,
            od2_rows_by_pair=od2_rows_by_pair,
            pairs=pairs,
            current_argmax=current_argmax,
            gt_argmax=gt_argmax,
            projected_n600_packet_bytes=pe3_hybrid75_projected + st2_context_projected,
            projection_scope="PE3 measured hybrid75 bytes plus ST2 measured scorer-native targeter table bytes; context is targeter-only",
            ssd_dir=run_ssd,
        )
    )

    base_selected = hybrid75_context_selected
    innovation_fractions = [float(item) for item in args.innovation_fractions.split(",") if item.strip()]
    for fraction in innovation_fractions:
        residual = _residual_selected(target_pairs=target_pairs, base_selected=base_selected, pairs=pairs, fraction=fraction)
        residual_records = _records_from_selected(target_pairs=target_pairs, selected=residual, pairs=pairs)
        residual_packet = od4.serialize_sparse_packet(residual_records)
        residual_best = _best_coder(od4.race_packet_coders(residual_packet))
        residual_projected = od4.projected_n600_packet_bytes(residual_best.bytes, len(pairs))
        combined = _union_selected(pairs, base_selected, residual)
        tag = f"{int(round(fraction * 100)):03d}"
        rungs.append(
            _rung_receipt(
                name=f"generator_context_targeter_innovation_{tag}",
                sections=[
                    od4.OD5Section("pe3_hybrid75_coords_n32", hybrid75_subset_raw),
                    od4.OD5Section("st2_context_targeter_table", context_raw),
                    od4.OD5Section(f"sparse_innovation_{tag}", residual_packet),
                ],
                selected_flats=combined,
                target_pairs=target_pairs,
                od2_rows_by_pair=od2_rows_by_pair,
                pairs=pairs,
                current_argmax=current_argmax,
                gt_argmax=gt_argmax,
                projected_n600_packet_bytes=pe3_hybrid75_projected + st2_context_projected + residual_projected,
                projection_scope=(
                    "PE3 measured hybrid75 bytes plus ST2 measured targeter bytes plus linear n600 projection "
                    f"of exact n32 sparse innovation residual at fraction {fraction:.2f}"
                ),
                ssd_dir=run_ssd,
            )
        )
        rungs[-1]["innovation_residual"] = {
            "fraction": fraction,
            "n32_sparse_packet_best_bytes": residual_best.bytes,
            "n600_sparse_linear_projection_bytes": residual_projected,
            "selected_residual_fixes": int(sum(arr.size for arr in residual.values())),
        }

    best = min(rungs, key=lambda row: row["projection_with_od2_stage2_pose_credit"]["projected_s"])
    receipt_json_path = args.research_dir / "ddm_od5_generator_packet_receipt.json"
    gate_script = args.research_dir / "OD5_SCORER_GATE_FIRE_ORDER.sh"
    md_path = args.research_dir / "OD5_GENERATOR_PACKET_RECEIPT.md"
    receipt: dict[str, Any] = {
        "schema": od4.OD5_RECEIPT_SCHEMA,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "axis": "[macOS-CPU cache-derived advisory / scorer-free mask-domain replay]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
        "scorer_forwards_run": 0,
        "frontier_moved": False,
        "current_own_vehicle_frontier": {
            "S": od4.CURRENT_OWN_S,
            "archive_bytes": od4.CURRENT_OWN_BYTES,
            "axis": od4.CURRENT_OWN_AXIS,
        },
        "denominators": {
            "selection_mode": pair_selection["selection"]["pair_selection"],
            "selection_seed": pair_selection["seed"],
            "n_pairs": len(pairs),
            "population_pairs": od4.N_PAIRS,
            "height": od4.SEG_H,
            "width": od4.SEG_W,
            "rate_denominator_bytes": od4.RATE_DENOMINATOR_BYTES,
            **target_denominators,
        },
        "storage_preflight": storage,
        "run_ssd_dir": str(run_ssd),
        "source_files": {
            "od2_json": {"path": str(args.od2_json), "bytes": args.od2_json.stat().st_size, "sha256": _sha256_file(args.od2_json)},
            "pair_selection": {"path": str(args.pair_selection), "bytes": args.pair_selection.stat().st_size, "sha256": _sha256_file(args.pair_selection)},
            "pe1_receipt": {"path": str(args.pe1_receipt), "bytes": args.pe1_receipt.stat().st_size, "sha256": _sha256_file(args.pe1_receipt)},
            "pe3_receipt": {"path": str(args.pe3_receipt), "bytes": args.pe3_receipt.stat().st_size, "sha256": _sha256_file(args.pe3_receipt)},
            "st2_receipt": {"path": str(args.st2_receipt), "bytes": args.st2_receipt.stat().st_size, "sha256": _sha256_file(args.st2_receipt)},
            "argmax_cache": {
                "path": str(args.argmax_cache),
                "cx1_sha256": _sha256_file(args.argmax_cache / "cx1_argmax_n600.npy"),
                "gt_sha256": _sha256_file(args.argmax_cache / "gt_argmax_n600.npy"),
            },
            "g4_recurrence": {"path": str(args.g4_recurrence), "bytes": args.g4_recurrence.stat().st_size, "sha256": _sha256_file(args.g4_recurrence)},
        },
        "component_extraction": extraction,
        "generator_surface": {
            "generator_count": int(surfaces["generator_count"]),
            "generator_n32_subset_raw_bytes": len(generator_subset_raw),
            "hybrid75_n32_subset_raw_bytes": len(hybrid75_subset_raw),
            "hybrid_knee_n32_subset_raw_bytes": len(hybrid_knee_subset_raw),
            "depth_meta": surfaces["depth_meta"],
        },
        "context_targeter": context_meta,
        "od2_pair_build_rows": [target_pairs[pair].build_row for pair in pairs],
        "rungs": rungs,
        "best_rung_by_projected_s_with_od2_pose_credit": best,
        "baseline_comparison": {
            "od4_sparse_per_flip_rate_cost_over_seg_win": OD4_RATE_COST_OVER_SEG_WIN,
            "od4_sparse_best_projected_s_with_pose_credit": 0.761509399,
            "current_own_s": od4.CURRENT_OWN_S,
        },
        "queued_scorer_gate_script": str(gate_script),
        "boundaries": [
            "No upstream/evaluate.py run",
            "No SegNet/PoseNet scorer job",
            "No full n600 dispatch",
            "No receiver-closed RGB/inflate archive",
            "ST2 context row is targeter-only because selected feature mode mixes scorer-native cached fields",
            "n600 composite bytes are projected component sums, not exact archive bytes",
        ],
        "frontier_line": "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.",
        "receipt_json_path": str(receipt_json_path),
        "receipt_md_path": str(md_path),
    }
    _atomic_write_json(receipt_json_path, receipt)
    _write_gate_script(gate_script)
    _write_markdown(md_path, receipt)

    print(_price_table_markdown(receipt))
    print(f"receipt_json={receipt_json_path}")
    print(f"receipt_md={md_path}")
    print(f"gate_script={gate_script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
