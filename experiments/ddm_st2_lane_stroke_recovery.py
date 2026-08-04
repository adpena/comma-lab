#!/usr/bin/env python
# ruff: noqa: I001
"""ddm_st2 -- Lane stroke-production survival curve + payload codec race.

This is the recovery landing for the st1/stroke-production arm.  It deliberately
reuses QA92/SQ1 surfaces:

* target set: QA92 erased super-nucleus Lane components (>5 px, 8-connected,
  erased iff <50% recovered as Lane in the base realized argmax)
* decode/scoring path: camera RGB -> frozen CPU DistortionNet SegNet, with GT
  decoded only through frame_utils.yuv420_to_rgb
* realizer: camera-resolution pre-R anti-aliased stroke compositing using the
  existing frozen-head solved Lane prototype colour [77.43, 86.71, 118.53]

Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE.  score_claim=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "upstream", REPO / "src", REPO / "experiments"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

try:
    import brotli
except ImportError:  # pragma: no cover - optional race arm
    brotli = None

import scipy.ndimage as ndi

from ddm_qa92_carrier_discriminator import (
    ERASED_RECOVER_THRESH,
    LANE_CLASS,
    NUCLEUS_PX,
    PROTO_LANE_SOLVED,
    erased_super_nucleus_mask,
)
from ddm_r7_token_coder import (
    VERIFY_CANONICAL,
    decode_token_codes,
    encode_token_codes,
    pack_nibbles,
)
from ddm_sq1_eta_seg_realization import (
    CAM_H,
    CAM_W,
    CLASS_NAMES,
    N_PAIRS_TOTAL,
    SEG_H,
    SEG_W,
    Scorer,
    decode_gt_frames,
    seq_len,
)


DEFAULT_SUB_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2")
DEFAULT_GT_MKV = Path("/Volumes/VertigoDataTier/pact/ddm_de1_20260803/0.mkv")
DEFAULT_ARGMAX_CACHE = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
DEFAULT_PAIRS_JSON = Path("/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_pair_selection.json")
DEFAULT_OUT = REPO / ".omx/research/ddm_st2_lane_stroke_recovery_20260804_receipt.json"

LIVE_OWN_VEHICLE_S = 0.7910689
LIVE_OWN_VEHICLE_BYTES = 353_805
PR130_FLOOR_S = 0.172141
GAP_S = LIVE_OWN_VEHICLE_S - PR130_FLOOR_S
S_PER_BYTE = 25.0 / 37_545_489.0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data)
    tmp.replace(path)


def _parse_csv_floats(text: str) -> list[float]:
    out = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("empty float list")
    return out


def _load_pairs(path: Path, limit: int) -> tuple[list[int], dict]:
    doc = json.loads(path.read_text())
    pairs = [int(p) for p in doc["pairs"]]
    if limit:
        pairs = pairs[:limit]
    return pairs, doc


def _seg_argmax_batch(sc: Scorer, pairs: list[np.ndarray], batch: int) -> list[np.ndarray]:
    import einops

    out: list[np.ndarray] = []
    for i in range(0, len(pairs), batch):
        chunk = np.stack(pairs[i : i + batch])
        x = torch.from_numpy(np.ascontiguousarray(chunk))
        x = einops.rearrange(x, "b t h w c -> b t c h w").float()
        with torch.inference_mode():
            logits = sc.net.segnet(sc.net.segnet.preprocess_input(x))
            pred = logits.argmax(dim=1).numpy().astype(np.uint8)
        out.extend([pred[j] for j in range(pred.shape[0])])
    return out


def _alpha_scorer_from_target(target: np.ndarray, width_extra_px: float) -> np.ndarray:
    """Soft scorer-lattice stroke alpha.

    width_extra_px is the expansion beyond the GT Lane target support in scorer
    pixels.  width=0 paints only target pixels; fractional widths give an
    anti-aliased edge before camera upsampling.
    """

    if not target.any():
        return np.zeros(target.shape, dtype=np.float32)
    dist = ndi.distance_transform_edt(~target)
    alpha = np.clip(width_extra_px + 1.0 - dist, 0.0, 1.0).astype(np.float32)
    alpha[target] = 1.0
    return alpha


def _alpha_to_camera(alpha_scorer: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(alpha_scorer))[None, None].float()
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(x, size=(CAM_H, CAM_W), mode="bilinear", align_corners=False)
    return np.clip(up[0, 0].numpy(), 0.0, 1.0).astype(np.float32)


def _composite_stroke(dec_f1: np.ndarray, alpha_cam: np.ndarray, amplitude: float) -> np.ndarray:
    blend = np.clip(float(amplitude) * alpha_cam[..., None], 0.0, 1.0)
    proto = PROTO_LANE_SOLVED.astype(np.float32)[None, None, :]
    base = dec_f1.astype(np.float32)
    edited = (1.0 - blend) * base + blend * proto
    return np.ascontiguousarray(np.clip(np.rint(edited), 0, 255).astype(np.uint8))


def _score_pose_subset(sc: Scorer, gt_pairs: dict[int, np.ndarray], base_pairs: dict[int, np.ndarray],
                       edited_f1: dict[int, np.ndarray]) -> dict:
    before = []
    after = []
    for p, gt in gt_pairs.items():
        pose_gt = sc.pose_out(gt)
        before.append(sc.d_pose(pose_gt, sc.pose_out(base_pairs[p])))
        after_pair = np.stack([base_pairs[p][0], edited_f1[p]])
        after.append(sc.d_pose(pose_gt, sc.pose_out(after_pair)))
    before_mean = float(np.mean(before)) if before else 0.0
    after_mean = float(np.mean(after)) if after else 0.0
    return {
        "n_pairs": len(before),
        "d_pose_before_mean": before_mean,
        "d_pose_after_mean": after_mean,
        "d_pose_delta_mean": after_mean - before_mean,
        "pose_S_delta_at_subset_baseline_slope": (
            float(np.sqrt(10.0 * after_mean) - np.sqrt(10.0 * before_mean))
            if before_mean >= 0.0 and after_mean >= 0.0 else None
        ),
    }


def _target_stats_for_pairs(lstars: np.ndarray, base: np.ndarray, pairs: Iterable[int]) -> dict[int, dict]:
    structure = np.ones((3, 3), dtype=bool)
    out: dict[int, dict] = {}
    for p in pairs:
        p = int(p)
        target, n_super, n_erased, target_px = erased_super_nucleus_mask(
            lstars[p] == LANE_CLASS,
            base[p],
            structure,
        )
        flips = base[p] != lstars[p]
        out[p] = {
            "target": target,
            "n_super": int(n_super),
            "n_erased": int(n_erased),
            "target_px": int(target_px),
            "base_flip_T": int((flips & target).sum()),
            "base_flip_total": int(flips.sum()),
        }
    return out


def _aggregate_target_stats(stats: dict[int, dict]) -> dict:
    n = len(stats)
    denom = float(max(n, 1) * SEG_H * SEG_W)
    base_flip_T = sum(int(s["base_flip_T"]) for s in stats.values())
    base_flip_total = sum(int(s["base_flip_total"]) for s in stats.values())
    target_px = sum(int(s["target_px"]) for s in stats.values())
    return {
        "n_pairs": n,
        "n_erased_super_nucleus": int(sum(int(s["n_erased"]) for s in stats.values())),
        "n_super_nucleus": int(sum(int(s["n_super"]) for s in stats.values())),
        "target_px": int(target_px),
        "base_flip_T": int(base_flip_T),
        "base_flip_total": int(base_flip_total),
        "target_pool_S_units": 100.0 * base_flip_T / denom,
        "base_total_S_units": 100.0 * base_flip_total / denom,
    }


def _race_payload(alpha_codes: np.ndarray) -> dict:
    codes = np.ascontiguousarray(alpha_codes[..., None].astype(np.uint8))
    if np.any(codes >= 16):
        raise ValueError("alpha codes must be 4-bit")
    raw_nibbles = pack_nibbles(codes)
    race: dict[str, dict] = {
        "semantic_shape": list(codes.shape),
        "semantic_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
        "raw_nibble_bytes": len(raw_nibbles),
        "code_nonzero_frac": float((codes != 0).mean()),
    }
    codec_bytes: dict[str, int | None] = {}
    for codec in ("smevr", "brotli11", "lzma1"):
        try:
            frame = encode_token_codes(codes, levels=16, codec=codec)
            restored = decode_token_codes(frame, verify=VERIFY_CANONICAL)
            if not np.array_equal(restored, codes):
                raise RuntimeError(f"{codec} roundtrip mismatch")
            codec_bytes[codec] = len(frame)
        except Exception as exc:
            race[f"{codec}_error"] = repr(exc)
            codec_bytes[codec] = None
    race["r7_framed_codec_bytes"] = codec_bytes

    raw_controls: dict[str, int | None] = {}
    if brotli is not None:
        b = bytes(brotli.compress(raw_nibbles, quality=11))
        if bytes(brotli.decompress(b)) != raw_nibbles:
            raise RuntimeError("raw brotli roundtrip mismatch")
        raw_controls["raw_nibbles_brotli11"] = len(b)
    else:
        raw_controls["raw_nibbles_brotli11"] = None
    lz = lzma.compress(
        raw_nibbles,
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}],
    )
    dec = lzma.LZMADecompressor(
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}],
    ).decompress(lz)
    if dec != raw_nibbles:
        raise RuntimeError("raw lzma1 roundtrip mismatch")
    raw_controls["raw_nibbles_lzma1_x9e"] = len(lz)
    race["raw_packed_controls_bytes"] = raw_controls
    valid = {k: v for k, v in codec_bytes.items() if isinstance(v, int)}
    race["r7_winner"] = min(valid, key=lambda k: valid[k]) if valid else None
    return race


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-dir", type=Path, default=DEFAULT_SUB_DIR)
    ap.add_argument("--gt-mkv", type=Path, default=DEFAULT_GT_MKV)
    ap.add_argument("--argmax-cache", type=Path, default=DEFAULT_ARGMAX_CACHE)
    ap.add_argument("--pairs-json", type=Path, default=DEFAULT_PAIRS_JSON)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seg-batch", type=int, default=4)
    ap.add_argument("--amplitudes", default="0.125,0.25,0.375,0.5,0.75,1.0")
    ap.add_argument("--widths", default="0,0.5,1.0,1.5,2.0,3.0")
    args = ap.parse_args()

    t0 = time.time()
    pairs, pair_doc = _load_pairs(args.pairs_json, args.limit)
    amplitudes = _parse_csv_floats(args.amplitudes)
    widths = _parse_csv_floats(args.widths)

    raw = np.memmap(
        args.sub_dir / "inflated" / "0.raw",
        dtype=np.uint8,
        mode="r",
        shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3),
    )
    base_cache = np.load(args.argmax_cache / "cx1_argmax_n600.npy", mmap_mode="r")
    gt_cache = np.load(args.argmax_cache / "gt_argmax_n600.npy", mmap_mode="r")

    population_stats = _target_stats_for_pairs(gt_cache, base_cache, range(N_PAIRS_TOTAL))
    selected_stats = _target_stats_for_pairs(gt_cache, base_cache, pairs)
    pop_agg = _aggregate_target_stats(population_stats)
    sel_agg = _aggregate_target_stats(selected_stats)

    wanted = set()
    for p in pairs:
        wanted.update({seq_len * p, seq_len * p + 1})
    gt_frames = decode_gt_frames(args.gt_mkv, wanted)
    gt_pairs = {
        p: np.stack([gt_frames[seq_len * p], gt_frames[seq_len * p + 1]]).astype(np.uint8)
        for p in pairs
    }
    base_pairs = {
        p: np.stack([raw[seq_len * p], raw[seq_len * p + 1]]).astype(np.uint8)
        for p in pairs
    }

    sc = Scorer(args.threads)
    controls = {"C2_lstar_matches_cache": 0, "C3_lgt_matches_cache": 0}
    for p in pairs:
        lstar = sc.seg_argmax(base_pairs[p])
        lgt = sc.seg_argmax(gt_pairs[p])
        controls["C2_lstar_matches_cache"] += int(np.array_equal(lstar, np.asarray(base_cache[p])))
        controls["C3_lgt_matches_cache"] += int(np.array_equal(lgt, np.asarray(gt_cache[p])))

    # Precompute alpha fields once per pair/width.
    alpha_cam: dict[tuple[int, float], np.ndarray] = {}
    alpha_scorer: dict[tuple[int, float], np.ndarray] = {}
    for p in pairs:
        target = selected_stats[p]["target"]
        for w in widths:
            asc = _alpha_scorer_from_target(target, w)
            alpha_scorer[(p, w)] = asc
            alpha_cam[(p, w)] = _alpha_to_camera(asc)

    curve_rows = []
    combo_argmax: dict[tuple[float, float], list[np.ndarray]] = {}
    for w in widths:
        for a in amplitudes:
            edited_pairs = []
            for p in pairs:
                edited = _composite_stroke(base_pairs[p][1], alpha_cam[(p, w)], a)
                edited_pairs.append(np.stack([base_pairs[p][0], edited]))
            preds = _seg_argmax_batch(sc, edited_pairs, args.seg_batch)
            combo_argmax[(a, w)] = preds

            after_total = 0
            after_T = 0
            lane_px_after_T = 0
            for p, pred in zip(pairs, preds, strict=True):
                gt = np.asarray(gt_cache[p])
                T = selected_stats[p]["target"]
                after_total += int((pred != gt).sum())
                after_T += int(((pred != gt) & T).sum())
                lane_px_after_T += int(((pred == LANE_CLASS) & T).sum())
            base_total = sel_agg["base_flip_total"]
            base_T = sel_agg["base_flip_T"]
            denom = float(len(pairs) * SEG_H * SEG_W)
            recovered = base_T - after_T
            target_recovered_S = 100.0 * recovered / denom
            net_delta_S = 100.0 * (after_total - base_total) / denom
            curve_rows.append({
                "amplitude_blend": a,
                "width_extra_scorer_px": w,
                "after_flip_total": int(after_total),
                "after_flip_T": int(after_T),
                "target_recovered_flips": int(recovered),
                "target_recovery_frac": float(recovered / base_T) if base_T else 0.0,
                "target_recovered_S_units_sample": target_recovered_S,
                "off_target_collateral_S_units_sample": net_delta_S + target_recovered_S,
                "net_delta_S_units_sample": net_delta_S,
                "lane_px_after_T_frac": float(lane_px_after_T / max(sel_agg["target_px"], 1)),
            })
            print(
                f"[st2] width {w:.2f} amp {a:.3f} "
                f"recovery {curve_rows[-1]['target_recovery_frac']:.4f} "
                f"net_dS {net_delta_S:+0.5f} elapsed {time.time() - t0:.1f}s",
                flush=True,
            )
            _atomic_write_json(args.out, {
                "schema": "ddm_st2_lane_stroke_recovery.v1.partial",
                "rows_completed": curve_rows,
                "elapsed_s": round(time.time() - t0, 2),
            })

    positive_rows = [r for r in curve_rows if r["target_recovered_flips"] > 0]
    min_survival = min(
        positive_rows,
        key=lambda r: (r["width_extra_scorer_px"], r["amplitude_blend"]),
    ) if positive_rows else None
    net_positive_rows = [r for r in curve_rows if r["net_delta_S_units_sample"] < 0.0]
    best_row = min(curve_rows, key=lambda r: r["net_delta_S_units_sample"])
    best_a = float(best_row["amplitude_blend"])
    best_w = float(best_row["width_extra_scorer_px"])

    def _payload_codes_for(row: dict) -> np.ndarray:
        row_a = float(row["amplitude_blend"])
        row_w = float(row["width_extra_scorer_px"])
        codes = []
        for pair_id in pairs:
            codes.append(
                np.rint(15.0 * row_a * alpha_scorer[(pair_id, row_w)])
                .clip(0, 15)
                .astype(np.uint8)
            )
        return np.stack(codes)

    edited_best: dict[int, np.ndarray] = {}
    for p in pairs:
        edited_best[p] = _composite_stroke(base_pairs[p][1], alpha_cam[(p, best_w)], best_a)
    edited_survival: dict[int, np.ndarray] | None = None
    if min_survival:
        survival_a = float(min_survival["amplitude_blend"])
        survival_w = float(min_survival["width_extra_scorer_px"])
        edited_survival = {
            p: _composite_stroke(base_pairs[p][1], alpha_cam[(p, survival_w)], survival_a)
            for p in pairs
        }
    payload_race_best = _race_payload(_payload_codes_for(best_row))
    payload_race_survival = _race_payload(_payload_codes_for(min_survival)) if min_survival else None
    pose_subset_best = _score_pose_subset(sc, gt_pairs, base_pairs, edited_best)
    pose_subset_survival = (
        _score_pose_subset(sc, gt_pairs, base_pairs, edited_survival)
        if edited_survival is not None else None
    )

    def _rate_s_units(race: dict | None) -> dict[str, float | None] | None:
        if race is None:
            return None
        return {
            k: (None if v is None else float(v * S_PER_BYTE))
            for k, v in race.get("r7_framed_codec_bytes", {}).items()
        }

    # Sample-to-population target-pool alignment is measured from cache, not inferred.
    target_pool_ratio = (
        sel_agg["target_pool_S_units"] / pop_agg["target_pool_S_units"]
        if pop_agg["target_pool_S_units"] else None
    )
    inferred_n600_target_recovered_S = (
        pop_agg["target_pool_S_units"] * best_row["target_recovery_frac"]
        if best_row["target_recovery_frac"] >= 0.0 else None
    )
    inferred_n600_survival_target_recovered_S = (
        pop_agg["target_pool_S_units"] * min_survival["target_recovery_frac"]
        if min_survival is not None else None
    )

    result = {
        "schema": "ddm_st2_lane_stroke_recovery.v1",
        "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "own_vehicle_frontier": {
            "S": LIVE_OWN_VEHICLE_S,
            "bytes": LIVE_OWN_VEHICLE_BYTES,
            "axis": "[macOS-CPU advisory]",
            "moved": False,
        },
        "class_order": CLASS_NAMES,
        "lane_class_index": LANE_CLASS,
        "method": {
            "target": "QA92 erased super-nucleus Lane components, 8-connectivity",
            "nucleus_px_threshold": NUCLEUS_PX,
            "erased_recover_thresh": ERASED_RECOVER_THRESH,
            "realizer": "camera-res pre-R AA stroke, blend toward frozen-head solved Lane prototype",
            "proto_lane_solved_rgb": [float(x) for x in PROTO_LANE_SOLVED],
            "amplitude_definition": "normalized blend strength: edited=(1-a*alpha)*decoded + a*alpha*proto",
            "width_definition": "extra scorer-pixel soft expansion beyond erased GT Lane support; width=0 paints target support only",
            "gt_decode": "frame_utils.yuv420_to_rgb only",
            "sample_rule": pair_doc.get("method"),
            "sample_source": str(args.pairs_json),
        },
        "inputs": {
            "sub_dir": str(args.sub_dir),
            "gt_mkv": str(args.gt_mkv),
            "argmax_cache": str(args.argmax_cache),
            "sub_archive_sha256": _sha256_file(args.sub_dir / "archive.zip"),
            "gt_mkv_sha256": _sha256_file(args.gt_mkv),
        },
        "denominators": {
            "n_pairs_measured": len(pairs),
            "pairs": pairs,
            "population_n_pairs": N_PAIRS_TOTAL,
            "gap_S": GAP_S,
            "S_per_byte": S_PER_BYTE,
            "sample_pair_selection_total_flip_ratio": pair_doc.get("ratio"),
            "target_pool_sample_to_population_ratio": target_pool_ratio,
            "population_target_pool": pop_agg,
            "sample_target_pool": sel_agg,
        },
        "positive_controls": controls,
        "amplitudes": amplitudes,
        "widths_extra_scorer_px": widths,
        "survival_curve": curve_rows,
        "minimum_nonzero_survival": min_survival,
        "minimum_net_positive": (
            min(net_positive_rows, key=lambda r: (r["width_extra_scorer_px"], r["amplitude_blend"]))
            if net_positive_rows else None
        ),
        "best_sample_net_row": best_row,
        "best_row_inferred_population_target_recovered_S_units": inferred_n600_target_recovered_S,
        "payload_race_for_best_net_row": payload_race_best,
        "payload_race_for_minimum_survival_row": payload_race_survival,
        "payload_rate_S_units_best_net_sample": _rate_s_units(payload_race_best),
        "payload_rate_S_units_minimum_survival_sample": _rate_s_units(payload_race_survival),
        "pose_subset_for_best_row": pose_subset_best,
        "pose_subset_for_minimum_survival_row": pose_subset_survival,
        "minimum_survival_row_inferred_population_target_recovered_S_units": (
            inferred_n600_survival_target_recovered_S
        ),
        "minimum_survival_row_sample_S_price": (
            {
                "target_recovered_S_units_sample": min_survival["target_recovered_S_units_sample"],
                "seg_net_delta_S_units_sample": min_survival["net_delta_S_units_sample"],
                "smevr_rate_S_units_sample": _rate_s_units(payload_race_survival)["smevr"],
                "pose_delta_S_units_sample": (
                    pose_subset_survival["pose_S_delta_at_subset_baseline_slope"]
                    if pose_subset_survival is not None else None
                ),
                "seg_plus_rate_delta_S_units_sample": (
                    min_survival["net_delta_S_units_sample"]
                    + _rate_s_units(payload_race_survival)["smevr"]
                ),
                "seg_plus_rate_plus_pose_delta_S_units_sample": (
                    min_survival["net_delta_S_units_sample"]
                    + _rate_s_units(payload_race_survival)["smevr"]
                    + pose_subset_survival["pose_S_delta_at_subset_baseline_slope"]
                ) if pose_subset_survival is not None else None,
            }
            if min_survival is not None else None
        ),
        "verdict_scope": {
            "stroke_amplitude_width_curve": "MEASURED on bounded n32 stratified-systematic sample; sample target-pool ratio reported",
            "population_target_pool": "MEASURED from n600 cached argmax arrays; no scorer forwards",
            "n600_realized_recovery": "INFERRED only when multiplying sample recovery fraction by population target pool",
            "payload_race": "MEASURED on the n32 best-net and minimum-survival alpha payloads; not a full n600 payload byte claim",
        },
        "state_boundaries": [
            "No contest-CPU/CUDA exact eval run.",
            "No full n600 scorer job run.",
            "No MPS authority used.",
            "No protected files or staged index touched by this script.",
        ],
        "elapsed_s": round(time.time() - t0, 2),
    }
    _atomic_write_json(args.out, result)
    print(json.dumps({
        "out": str(args.out),
        "min_survival": min_survival,
        "best": best_row,
        "payload_winner_best_net": payload_race_best.get("r7_winner"),
        "payload_winner_min_survival": (
            payload_race_survival.get("r7_winner") if payload_race_survival else None
        ),
        "elapsed_s": result["elapsed_s"],
    }, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
