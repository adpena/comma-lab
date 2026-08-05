#!/usr/bin/env python
"""RZ1 PE3 grammar x frozen-head regional solve measurement.

Consumes the measured PE3EDGE1 section as WHERE/WHICH constraints, not as
direct RGB paint.  The RGB realization is solved against the frozen SegNet head
on the described scorer-lattice band, then reduced to regional prototype slots
keyed by the PE3 component ownership.  All scoring here is local frozen-scorer
advisory; no upstream/evaluate.py or pointer authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import sys
import time
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "experiments", REPO / "src", REPO / "src" / "tac" / "optimization"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import inflate_runner_v4d as receiver
from ddm_sq1_eta_seg_realization import (
    CAM_H,
    CAM_W,
    N_PAIRS_TOTAL,
    SEG_H,
    SEG_W,
    Scorer,
    decode_gt_frames,
    seq_len,
)
from ddm_sq1_stage_decomposition_and_solved_paint import (
    realize_scorer_paint_to_camera,
    solve_margin_optimal_paint,
)

from tac.optimization.ddm_ix2_archive_container import parse_payload

AXIS = "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE"
LIVE_FRONTIER_S = 0.7539807296911207
LIVE_FRONTIER_BYTES = 357_836
LIVE_FRONTIER_DSEG = 0.00431179
LIVE_FRONTIER_DPOSE = 0.00071459
PE3_OD9_COMPOSED_PROJECTION_BYTES = 114_852
CONTEST_DENOMINATOR_BYTES = 37_545_489
SCORE_DENOMINATOR_PIXELS = N_PAIRS_TOTAL * SEG_H * SEG_W


@dataclass(frozen=True)
class PE3Component:
    mode: int
    mode_name: str
    indices: np.ndarray
    classes: np.ndarray
    record_bytes: int


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score_from_components(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return (
        100.0 * float(d_seg)
        + math.sqrt(10.0 * float(d_pose))
        + 25.0 * float(archive_bytes) / CONTEST_DENOMINATOR_BYTES
    )


def load_archive_payload(path: Path) -> bytes:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            names = [name for name in zf.namelist() if not name.endswith("/")]
            for name in ("archive/0.bin", "0.bin"):
                if name in names:
                    return zf.read(name)
            if len(names) == 1:
                return zf.read(names[0])
            raise SystemExit(f"cannot identify archive payload member in {path}: {names}")
    return path.read_bytes()


def extract_pe3_section(path: Path) -> tuple[bytes, dict]:
    payload = load_archive_payload(path)
    _bulk, sections = parse_payload(payload)
    matches = [bytes(section) for section in sections if bytes(section).startswith(receiver.PE3_EDGE_MAGIC)]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one PE3EDGE1 section in {path}, found {len(matches)}")
    blob = matches[0]
    return blob, receiver._pe3_parse_edge_field(blob)


def parse_pe3_components(blob: bytes) -> tuple[list[list[PE3Component]], dict]:
    if len(blob) < receiver.PE3_EDGE_HEADER.size:
        raise SystemExit("PE3 section header truncated")
    (
        magic,
        version,
        seg_h,
        seg_w,
        n_pairs,
        kind,
        codec,
        raw_len,
        frame_record_count,
        raw_sha,
    ) = receiver.PE3_EDGE_HEADER.unpack_from(blob, 0)
    if magic != receiver.PE3_EDGE_MAGIC or version != receiver.PE3_EDGE_VERSION:
        raise SystemExit("PE3 section magic/version differs")
    if int(kind) != receiver._PE3_HYBRID:
        raise SystemExit("PE3 section kind differs")
    if int(frame_record_count) != int(n_pairs):
        raise SystemExit("PE3 frame record count differs")
    raw = receiver._pe1_decode_body(int(codec), blob[receiver.PE3_EDGE_HEADER.size:])
    if len(raw) != int(raw_len):
        raise SystemExit("PE3 raw body length differs")
    if hashlib.sha256(raw).digest() != raw_sha:
        raise SystemExit("PE3 raw body SHA differs")

    rows: list[list[PE3Component]] = []
    offset = 0
    mode_counts: dict[str, int] = {}
    component_count = 0
    for _pair in range(int(n_pairs)):
        count, offset = receiver._pe1_read_varint(raw, offset)
        pair_rows: list[PE3Component] = []
        for _ in range(int(count)):
            length, offset = receiver._pe1_read_varint(raw, offset)
            record = raw[offset:offset + int(length)]
            if len(record) != int(length):
                raise SystemExit("PE3 component record truncated")
            offset += int(length)
            if not record:
                raise SystemExit("PE3 empty component record")
            mode = int(record[0])
            payload = record[1:]
            if mode == receiver._PE3_MODE_CURVE:
                indices, classes = receiver._pe1_curve_indices(payload, int(seg_h), int(seg_w))
            elif mode == receiver._PE3_MODE_GENERATOR:
                indices, classes = receiver._pe1_generator_indices(payload, int(seg_h), int(seg_w))
            else:
                raise SystemExit(f"unknown PE3 component mode {mode}")
            if int(indices.size) != int(classes.size):
                raise SystemExit("PE3 component index/class length differs")
            mode_name = receiver._PE3_MODE_NAMES[mode]
            mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1
            component_count += 1
            pair_rows.append(
                PE3Component(
                    mode=mode,
                    mode_name=mode_name,
                    indices=np.asarray(indices, dtype=np.int32),
                    classes=np.asarray(classes, dtype=np.uint8),
                    record_bytes=int(length),
                )
            )
        rows.append(pair_rows)
    if offset != len(raw):
        raise SystemExit("PE3 raw body has trailing bytes")
    meta = {
        "seg_h": int(seg_h),
        "seg_w": int(seg_w),
        "n_pairs": int(n_pairs),
        "codec": int(codec),
        "raw_bytes": int(raw_len),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "section_bytes": len(blob),
        "section_sha256": hashlib.sha256(blob).hexdigest(),
        "component_records": int(component_count),
        "mode_counts": mode_counts,
    }
    return rows, meta


def effective_component_ownership(
    components: list[PE3Component],
    total_slots: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    owner = np.full(total_slots, -1, dtype=np.int32)
    owner_class = np.full(total_slots, -1, dtype=np.int16)
    for comp_index, component in enumerate(components):
        if component.indices.size == 0:
            continue
        owner[component.indices.astype(np.int64)] = int(comp_index)
        owner_class[component.indices.astype(np.int64)] = component.classes.astype(np.int16)

    slots: list[dict] = []
    final_indices = np.flatnonzero(owner >= 0)
    for comp_index, component in enumerate(components):
        hit = final_indices[owner[final_indices] == int(comp_index)]
        if hit.size == 0:
            continue
        for cls in np.unique(owner_class[hit]):
            cls_hit = hit[owner_class[hit] == int(cls)]
            if cls_hit.size == 0:
                continue
            slots.append(
                {
                    "component_index": int(comp_index),
                    "mode_name": component.mode_name,
                    "class": int(cls),
                    "effective_pixels": int(cls_hit.size),
                }
            )
    return owner, owner_class, slots


def select_pairs_from_na3(selection_json: Path, n: int) -> tuple[list[int], dict]:
    payload = json.loads(selection_json.read_text())
    selection = payload["selection"]
    indices = [int(x) for x in selection["indices"]]
    block_count = int(selection["params"].get("block_count", 10))
    population = int(selection.get("population", N_PAIRS_TOTAL))
    seed = int(selection["seed"])
    if n > len(indices):
        raise SystemExit(f"requested n={n} exceeds NA3 selected n={len(indices)}")
    if n == len(indices):
        return indices, {
            "source": str(selection_json),
            "source_schema": payload.get("schema"),
            "mode": "na3_full_selection",
            "n_requested": int(n),
            "seed": seed,
            "block_count": block_count,
        }

    rng = np.random.default_rng(seed + int(n))
    block_width = population // block_count
    by_block: list[list[int]] = []
    for block in range(block_count):
        lo = block * block_width
        hi = population if block == block_count - 1 else (block + 1) * block_width
        members = [p for p in indices if lo <= p < hi]
        by_block.append(members)
    base = n // block_count
    rem = n % block_count
    allocation = [base] * block_count
    if rem:
        nonempty = [i for i, members in enumerate(by_block) if len(members) > base]
        chosen = rng.choice(nonempty, size=rem, replace=False)
        for block in chosen.tolist():
            allocation[int(block)] += 1
    selected: list[int] = []
    block_rows = []
    for block, (members, take) in enumerate(zip(by_block, allocation, strict=True)):
        if take <= 0:
            block_rows.append({"block": block, "take": 0, "available": len(members), "pairs": []})
            continue
        if take > len(members):
            raise SystemExit(f"block {block} has {len(members)} members, cannot take {take}")
        picks = sorted(rng.choice(members, size=take, replace=False).astype(int).tolist())
        selected.extend(picks)
        block_rows.append({"block": block, "take": int(take), "available": len(members), "pairs": picks})
    selected = sorted(selected)
    return selected, {
        "source": str(selection_json),
        "source_schema": payload.get("schema"),
        "mode": "stratified_blocks_from_na3_n120_seeded_without_prefix",
        "n_requested": int(n),
        "seed": seed,
        "derived_seed": int(seed + int(n)),
        "block_count": block_count,
        "population": population,
        "block_allocation": block_rows,
    }


def target_from_pe3(lstar: np.ndarray, band: np.ndarray, classes: np.ndarray) -> np.ndarray:
    target = np.asarray(lstar, dtype=np.uint8).reshape(-1).copy()
    target[np.asarray(band, dtype=np.int64)] = np.asarray(classes, dtype=np.uint8)
    return target.reshape(SEG_H, SEG_W)


def prototype_payload_for_pair(
    paint_u8: np.ndarray,
    owner: np.ndarray,
    owner_class: np.ndarray,
    slots: list[dict],
) -> tuple[np.ndarray, bytes, list[dict]]:
    proto = np.zeros((SEG_H, SEG_W, 3), dtype=np.uint8)
    payload = bytearray()
    rows: list[dict] = []
    for slot in slots:
        comp = int(slot["component_index"])
        cls = int(slot["class"])
        flat_idx = np.flatnonzero((owner == comp) & (owner_class == cls))
        if flat_idx.size == 0:
            continue
        yy = flat_idx // SEG_W
        xx = flat_idx % SEG_W
        rgb = np.rint(np.median(paint_u8[yy, xx], axis=0)).clip(0, 255).astype(np.uint8)
        proto[yy, xx] = rgb
        payload.extend(rgb.tobytes())
        rows.append(
            {
                "component_index": comp,
                "mode_name": slot["mode_name"],
                "class": cls,
                "effective_pixels": int(flat_idx.size),
                "rgb": [int(x) for x in rgb.tolist()],
            }
        )
    return proto, bytes(payload), rows


def compression_summary(payload: bytes) -> dict:
    candidates = {
        "raw": bytes(payload),
        "zlib9": zlib.compress(payload, 9),
        "brotli_q11": brotli.compress(payload, quality=11, lgwin=24),
        "lzma_raw": lzma.compress(
            payload,
            format=lzma.FORMAT_RAW,
            filters=[{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 0, "lp": 0, "pb": 0}],
        ),
    }
    sizes = {name: len(body) for name, body in candidates.items()}
    best = min(sizes, key=sizes.get)
    return {"sizes": sizes, "best": best, "best_bytes": int(sizes[best])}


def empty_aggregate(name: str) -> dict:
    return {
        "name": name,
        "flips_after": 0,
        "fixed": 0,
        "introduced": 0,
        "net_fixed": 0,
        "offband_argmax_changed": 0,
        "offband_introduced_errors": 0,
        "d_pose_after_sum": 0.0,
    }


def add_variant_metrics(
    agg: dict,
    tag: str,
    lam: np.ndarray,
    lgt: np.ndarray,
    lstar: np.ndarray,
    band_mask: np.ndarray,
    flips0: np.ndarray,
    pose_after: float,
    rec: dict,
) -> None:
    fa = lam != lgt
    fixed = int((flips0 & ~fa).sum())
    introduced = int((~flips0 & fa).sum())
    offband = ~band_mask
    payload = {
        "flips_after": int(fa.sum()),
        "fixed": fixed,
        "introduced": introduced,
        "net_fixed": int(flips0.sum()) - int(fa.sum()),
        "offband_argmax_changed": int(((lam != lstar) & offband).sum()),
        "offband_introduced_errors": int(((~flips0) & fa & offband).sum()),
        "d_pose_after": float(pose_after),
    }
    rec[tag] = payload
    agg["flips_after"] += int(payload["flips_after"])
    agg["fixed"] += fixed
    agg["introduced"] += introduced
    agg["net_fixed"] += int(payload["net_fixed"])
    agg["offband_argmax_changed"] += int(payload["offband_argmax_changed"])
    agg["offband_introduced_errors"] += int(payload["offband_introduced_errors"])
    agg["d_pose_after_sum"] += float(pose_after)


def build_aggregate(
    rows: list[dict],
    n_pairs: int,
    total_effective_slots_n600: int,
    total_band_px_n600: int,
    subset_prototype_payload: bytes,
) -> dict:
    denom = int(n_pairs) * SEG_H * SEG_W
    flips_before = int(sum(r["flips_before"] for r in rows))
    target_flips = int(sum(r["target_flips"] for r in rows))
    d_pose_before = float(sum(r["d_pose_before"] for r in rows) / max(1, n_pairs))
    prototype_slots_subset = int(sum(r["prototype_slots"] for r in rows))
    compressed = compression_summary(subset_prototype_payload)

    variants = {}
    for name in ("dense_headsolve", "regional_prototype"):
        flips_after = int(sum(r[name]["flips_after"] for r in rows))
        d_pose_after = float(sum(r[name]["d_pose_after"] for r in rows) / max(1, n_pairs))
        variants[name] = {
            "d_seg": float(flips_after / denom),
            "d_pose": d_pose_after,
            "flips_after": flips_after,
            "fixed": int(sum(r[name]["fixed"] for r in rows)),
            "introduced": int(sum(r[name]["introduced"] for r in rows)),
            "net_fixed": int(flips_before - flips_after),
            "offband_argmax_changed": int(sum(r[name]["offband_argmax_changed"] for r in rows)),
            "offband_introduced_errors": int(sum(r[name]["offband_introduced_errors"] for r in rows)),
        }

    if prototype_slots_subset:
        scale = float(total_effective_slots_n600) / float(prototype_slots_subset)
        projected_best_bytes = math.ceil(compressed["best_bytes"] * scale)
    else:
        scale = 0.0
        projected_best_bytes = 0
    prototype_bytes = {
        "effective_component_class_slots_subset": prototype_slots_subset,
        "effective_component_class_slots_n600": int(total_effective_slots_n600),
        "raw_bytes_subset": len(subset_prototype_payload),
        "raw_bytes_n600_exact_from_pe3_slots": int(total_effective_slots_n600 * 3),
        "compressed_subset": compressed,
        "compressed_projection_scale_n600_over_subset_slots": scale,
        "compressed_best_bytes_n600_linear_projection": int(projected_best_bytes),
        "composed_projection_bytes_raw": int(PE3_OD9_COMPOSED_PROJECTION_BYTES + total_effective_slots_n600 * 3),
        "composed_projection_bytes_compressed": int(PE3_OD9_COMPOSED_PROJECTION_BYTES + projected_best_bytes),
    }

    for key in ("raw", "compressed"):
        archive_bytes = prototype_bytes[f"composed_projection_bytes_{key}"]
        variants["regional_prototype"][f"score_projection_{key}_bytes"] = score_from_components(
            variants["regional_prototype"]["d_seg"],
            variants["regional_prototype"]["d_pose"],
            archive_bytes,
        )
        variants["regional_prototype"][f"delta_vs_live_projection_{key}_bytes"] = (
            variants["regional_prototype"][f"score_projection_{key}_bytes"] - LIVE_FRONTIER_S
        )

    return {
        "n_pairs": int(n_pairs),
        "seg_denominator_pixels": int(denom),
        "full_n600_denominator_pixels": int(SCORE_DENOMINATOR_PIXELS),
        "flips_before": flips_before,
        "target_flips": target_flips,
        "label_ceiling_net_fixed": int(flips_before - target_flips),
        "d_seg_before": float(flips_before / denom),
        "d_pose_before": d_pose_before,
        "variants": variants,
        "prototype_bytes": prototype_bytes,
        "dense_upper_bound_bytes_not_receiver_priced": {
            "band_px_subset": int(sum(r["band_px"] for r in rows)),
            "band_px_n600_from_pe3": int(total_band_px_n600),
            "note": "Dense per-pixel paint is an upper-bound mechanism only here; RZ1 gating uses regional_prototype pricing.",
        },
    }


def write_output(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-sub-dir", type=Path, required=True)
    ap.add_argument("--pe3-archive", type=Path, required=True)
    ap.add_argument("--gt-mkv", type=Path, required=True)
    ap.add_argument("--pairs-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=4.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    pe3_blob, pe3_field = extract_pe3_section(args.pe3_archive)
    components, pe3_meta = parse_pe3_components(pe3_blob)
    total_slots = SEG_H * SEG_W
    if int(pe3_meta["n_pairs"]) != N_PAIRS_TOTAL:
        raise SystemExit(f"PE3 section n_pairs={pe3_meta['n_pairs']} != {N_PAIRS_TOTAL}")

    effective_slots_n600 = 0
    band_px_n600 = 0
    for pair_components in components:
        owner, _owner_class, slots = effective_component_ownership(pair_components, total_slots)
        effective_slots_n600 += len(slots)
        band_px_n600 += int((owner >= 0).sum())

    pairs, selection_meta = select_pairs_from_na3(args.pairs_json, args.n)
    raw_path = args.base_sub_dir / "inflated" / "0.raw"
    raw = np.memmap(raw_path, dtype=np.uint8, mode="r",
                    shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3))
    wanted = {seq_len * p + t for p in pairs for t in (0, 1)}
    gt_frames = decode_gt_frames(args.gt_mkv, wanted)
    sc = Scorer(args.threads)
    segnet = sc.net.segnet

    rows: list[dict] = []
    subset_prototype_payload = b""
    if args.resume and args.out.exists():
        prior = json.loads(args.out.read_text())
        rows = prior.get("rows", [])
        subset_prototype_payload = bytes.fromhex(prior.get("subset_prototype_payload_hex", ""))
        done = {int(row["pair"]) for row in rows}
        pairs_to_run = [p for p in pairs if int(p) not in done]
        print(f"[rz1] resume loaded {len(rows)} rows, remaining {len(pairs_to_run)}", flush=True)
    else:
        pairs_to_run = pairs

    static_payload = {
        "schema": "ddm_rz1_pe3_headsolve.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "frontier_reference": {
            "S": LIVE_FRONTIER_S,
            "archive_bytes": LIVE_FRONTIER_BYTES,
            "d_seg": LIVE_FRONTIER_DSEG,
            "d_pose": LIVE_FRONTIER_DPOSE,
            "axis": "[macOS-CPU advisory]",
            "contest_pointer": "borrowed/unmoved",
        },
        "inputs": {
            "base_raw_path": str(raw_path),
            "base_raw_sha256": sha256_path(raw_path),
            "pe3_archive": str(args.pe3_archive),
            "pe3_archive_sha256": sha256_path(args.pe3_archive),
            "gt_mkv": str(args.gt_mkv),
            "pairs_json": str(args.pairs_json),
        },
        "command": sys.argv,
        "solver": {
            "mechanism": "PE3 constraints target a frozen-head margin-optimal paint; regional prototype payload is then keyed by effective PE3 component/class ownership.",
            "steps": int(args.steps),
            "lr": float(args.lr),
            "eval_every": int(args.eval_every),
            "starts": ["decoded", "truth"],
        },
        "selection": selection_meta,
        "pe3_parse": {
            **pe3_meta,
            "receiver_field_raster_sha256": pe3_field["raster_sha256"],
            "receiver_field_pair_count_sum": int(sum(int(x) for x in pe3_field["pair_counts"])),
            "effective_component_class_slots_n600": int(effective_slots_n600),
            "band_px_n600": int(band_px_n600),
        },
    }

    print(
        f"[rz1] ready n={args.n} pairs={pairs} pe3_slots_n600={effective_slots_n600} "
        f"band_px_n600={band_px_n600} t={time.time() - t0:.1f}s",
        flush=True,
    )

    for ordinal, pair in enumerate(pairs_to_run, start=len(rows) + 1):
        tp = time.time()
        dec = np.stack([raw[seq_len * pair], raw[seq_len * pair + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * pair], gt_frames[seq_len * pair + 1]])
        lstar = sc.seg_argmax(dec)
        lgt = sc.seg_argmax(gt)
        pose_gt = sc.pose_out(gt)
        pose_before = sc.d_pose(pose_gt, sc.pose_out(dec))
        band = np.asarray(pe3_field["bands"][pair], dtype=np.int32)
        classes = np.asarray(pe3_field["classes"][pair], dtype=np.uint8)
        band_mask = np.zeros((SEG_H * SEG_W,), dtype=bool)
        band_mask[band.astype(np.int64)] = True
        band_mask = band_mask.reshape(SEG_H, SEG_W)
        target = target_from_pe3(lstar, band, classes)
        flips0 = lstar != lgt
        target_flips = target != lgt

        owner, owner_class, slots = effective_component_ownership(components[pair], total_slots)
        nbad, dense_paint, solve_tag, solve_diag = solve_margin_optimal_paint(
            segnet,
            dec[1],
            gt[1],
            band_mask,
            target,
            steps=args.steps,
            lr=args.lr,
            eval_every=args.eval_every,
        )
        dense_cam = realize_scorer_paint_to_camera(dec[1], band_mask, dense_paint)
        dense_pair = np.stack([dec[0], dense_cam])
        dense_lam = sc.seg_argmax(dense_pair)

        proto_paint, pair_payload, slot_rows = prototype_payload_for_pair(
            dense_paint,
            owner,
            owner_class,
            slots,
        )
        proto_cam = realize_scorer_paint_to_camera(dec[1], band_mask, proto_paint)
        proto_pair = np.stack([dec[0], proto_cam])
        proto_lam = sc.seg_argmax(proto_pair)
        subset_prototype_payload += pair_payload

        rec = {
            "pair": int(pair),
            "selection_ordinal": int(ordinal),
            "flips_before": int(flips0.sum()),
            "target_flips": int(target_flips.sum()),
            "label_ceiling_net_fixed": int(flips0.sum()) - int(target_flips.sum()),
            "band_px": int(band.size),
            "component_records": len(components[pair]),
            "prototype_slots": len(slot_rows),
            "prototype_raw_bytes": len(pair_payload),
            "d_pose_before": float(pose_before),
            "solve": {
                "tag": solve_tag,
                "proxy_flips_scorer_lattice": int(nbad),
                "selected": solve_diag["selected"],
            },
            "prototype_slots_detail": slot_rows,
        }
        dense_agg = empty_aggregate("dense_headsolve")
        proto_agg = empty_aggregate("regional_prototype")
        add_variant_metrics(
            dense_agg,
            "dense_headsolve",
            dense_lam,
            lgt,
            lstar,
            band_mask,
            flips0,
            sc.d_pose(pose_gt, sc.pose_out(dense_pair)),
            rec,
        )
        add_variant_metrics(
            proto_agg,
            "regional_prototype",
            proto_lam,
            lgt,
            lstar,
            band_mask,
            flips0,
            sc.d_pose(pose_gt, sc.pose_out(proto_pair)),
            rec,
        )
        rows.append(rec)
        aggregate = build_aggregate(
            rows,
            len(rows),
            effective_slots_n600,
            band_px_n600,
            subset_prototype_payload,
        )
        write_output(
            args.out,
            {
                **static_payload,
                "elapsed_s": time.time() - t0,
                "rows": rows,
                "subset_prototype_payload_hex": subset_prototype_payload.hex(),
                "aggregate": aggregate,
            },
        )
        print(
            f"[rz1] pair {pair:3d} ({len(rows)}/{args.n}) "
            f"before={rec['flips_before']:5d} target={rec['target_flips']:5d} "
            f"dense={rec['dense_headsolve']['flips_after']:5d} "
            f"regional={rec['regional_prototype']['flips_after']:5d} "
            f"proto_slots={rec['prototype_slots']:3d} "
            f"pose {rec['d_pose_before']:.6g}->{rec['regional_prototype']['d_pose_after']:.6g} "
            f"dt={time.time() - tp:.1f}s",
            flush=True,
        )

    final = json.loads(args.out.read_text()) if args.out.exists() else {**static_payload, "rows": rows}
    aggregate = final.get("aggregate") or build_aggregate(
        rows,
        len(rows),
        effective_slots_n600,
        band_px_n600,
        subset_prototype_payload,
    )
    regional = aggregate["variants"]["regional_prototype"]
    smoke_green = (
        int(regional["net_fixed"]) > 0
        and int(regional["introduced"]) <= int(regional["fixed"])
        and float(regional["d_pose"]) <= max(1.1 * float(aggregate["d_pose_before"]), aggregate["d_pose_before"] + 1e-4)
    )
    n32_green = (
        args.n >= 32
        and float(regional["delta_vs_live_projection_compressed_bytes"]) < 0.0
        and float(regional["d_pose"]) <= max(1.1 * LIVE_FRONTIER_DPOSE, LIVE_FRONTIER_DPOSE + 1e-4)
    )
    final["gate_verdict"] = {
        "smoke_green": bool(smoke_green),
        "n32_green_for_one_n600": bool(n32_green),
        "criteria": {
            "smoke_green": "regional net_fixed > 0, introduced <= fixed, and no >10%/1e-4 pose regression vs subset base",
            "n32_green": "n>=32, regional compressed-byte score projection delta-S<0 vs live, and pose survives vs live",
        },
    }
    final["elapsed_s"] = time.time() - t0
    write_output(args.out, final)
    print(f"[rz1] DONE -> {args.out}", flush=True)
    print(json.dumps(final["gate_verdict"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
