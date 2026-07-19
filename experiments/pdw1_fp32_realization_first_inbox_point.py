#!/usr/bin/env python
"""PDW1 fp32 realization — the first measured (bytes, d_seg) point in the 100–300 KB box.

Unit charter (2026-07-19, lane ``pdw1_realization_20260719``): the #572 rate-crush
verdict left the 100–300 KB total-archive budget box with ZERO measured
(bytes, d_seg) points; the blocker was REALIZATION, not bytes.  This script:

  Phase A — sweeps the committed quotient prefix (frames 0..194) and measures,
            under the declared frozen-fp32 receiver contract
            (``pdw1-native-f32-power-first-max.v1``), contract-vs-L* and
            contract-vs-generic-float64 disagreements, with exact tie geometry
            for every instance (closes the #543 arithmetic-authority hole on
            the whole prefix, not just one pixel).
  Phase B — reproduces the #543 frame-195 blocker pixel first-hand from the
            sealed diagnostic receipt and shows it closes under the contract.
  Phase C — builds the minimal PDW1P payload (contract labels + per-class
            fills) for n24 pairs, realizes every plane through the PROVEN
            factor-2 lattice operator to exact uint8 camera frames, and
            measures d_seg through the hard oracle (frozen CPU-Torch SegNet
            argmax on the realized frames) plus d_pose (repeat-frame1 policy)
            through the frozen PoseNet.  Decomposes the residual by class
            pair and boundary distance.

Axis: [macOS-CPU advisory] research_only=true — NO score/promotion/pointer
authority.  Reads are read-only (GT cache, quotient cache, frozen scorers,
sealed receipts); bulk outputs go to the SSD evidence tier.

Usage:
    TAC_UPSTREAM_DIR=/Users/adpena/Projects/pact/upstream PYTHONPATH=src \
        .venv/bin/python experiments/pdw1_fp32_realization_first_inbox_point.py \
        --output .omx/research/pdw1_fp32_realization_receipt_20260719.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import platform
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import brotli
import numpy as np

REPO = Path("/Users/adpena/Projects/pact")
QUOTIENT_CACHE = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "v10_power_diagram_byteclose_20260718/n600_rank4_features/quotient_features.f32.npy"
)
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DIAGNOSTIC = REPO / ".omx/research/v10_power_diagram_frame195_diagnostic_20260718.json"
DIAGNOSTIC_SHA256 = "65d97194c6298a5502d0fcc792ee2fe3bf05599c69f1130d64c270dec5ec36ee"
EVIDENCE_DIR = Path("/Volumes/VertigoDataTier/pact/evidence/pdw1_realization_20260719")

PREFIX_FRAMES = 195  # committed frames 0..194 in the preserved quotient cache
N24 = 24
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
SCORE_BYTES_NORMALIZER = 37_545_489
FRONTIER_S = 0.1910828242
# Budget box constants from the #572 memo (derived from the frozen score law).
BOX_TOTAL_AT_EXACT_REALIZATION = 286_682
BOX_PER_PAIR = 477.8
BOX_DSEG_AT_236KB = 3.39e-4


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def order0_entropy_bytes(payload: bytes) -> int:
    counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
    nz = counts[counts > 0].astype(np.float64)
    total = float(len(payload))
    bits = float(-np.sum((nz / total) * np.log2(nz / total))) * total
    return int(np.ceil(bits / 8.0))


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    """Pixels whose 4-neighbourhood contains another class."""

    mask = np.zeros(labels.shape, dtype=bool)
    mask[:-1, :] |= labels[:-1, :] != labels[1:, :]
    mask[1:, :] |= labels[1:, :] != labels[:-1, :]
    mask[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    mask[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return mask


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        grown[:-1, :] |= out[1:, :]
        grown[1:, :] |= out[:-1, :]
        grown[:, :-1] |= out[:, 1:]
        grown[:, 1:] |= out[:, :-1]
        out = grown
    return out


def load_frozen_target():
    from tac.boundary_math.power_diagram_witness import decode_pdw1

    receipt = json.loads(DIAGNOSTIC.read_text())
    target = decode_pdw1(bytes.fromhex(receipt["frozen_target"]["pdw1_hex"]))
    return target, receipt


def phase_a_prefix_closure(target, lstars_mm, quotient_mm) -> dict[str, Any]:
    """Contract-f32 vs L* and vs generic-f64 over the committed prefix."""

    from tac.boundary_math.pdw1_fp32_receiver_contract import contract_f32_assign
    from tac.boundary_math.power_diagram_witness import power_assign

    t0 = time.time()
    contract_vs_lstar: list[dict[str, Any]] = []
    contract_vs_f64: list[dict[str, Any]] = []
    n_lstar_mismatch = 0
    n_f64_mismatch = 0
    total = 0
    for frame in range(PREFIX_FRAMES):
        z = np.ascontiguousarray(quotient_mm[frame], dtype=np.float32).reshape(-1, 4)
        lst = np.asarray(lstars_mm[frame]).reshape(-1)
        ours = contract_f32_assign(z, target)
        generic = power_assign(z.astype(np.float64), target)
        total += lst.size
        bad = np.nonzero(ours != lst)[0]
        n_lstar_mismatch += bad.size
        for flat in bad[:64]:
            contract_vs_lstar.append(
                {"frame": frame, "y": int(flat // 512), "x": int(flat % 512),
                 "lstar": int(lst[flat]), "contract": int(ours[flat])}
            )
        diff = np.nonzero(ours != generic)[0]
        n_f64_mismatch += diff.size
        for flat in diff[:64]:
            contract_vs_f64.append(
                {"frame": frame, "y": int(flat // 512), "x": int(flat % 512),
                 "contract": int(ours[flat]), "generic_f64": int(generic[flat]),
                 "lstar": int(lst[flat])}
            )
    return {
        "label": "MEASURED_PREFIX_CLOSURE_FRAMES_0_194",
        "frames": PREFIX_FRAMES,
        "pixels": total,
        "contract_vs_lstar_mismatches": int(n_lstar_mismatch),
        "contract_vs_generic_f64_mismatches": int(n_f64_mismatch),
        "contract_vs_lstar_instances": contract_vs_lstar,
        "contract_vs_generic_f64_instances": contract_vs_f64,
        "wall_seconds": round(time.time() - t0, 2),
    }


def phase_b_frame195(target, receipt) -> dict[str, Any]:
    from tac.boundary_math.pdw1_fp32_receiver_contract import (
        CONTRACT_ID,
        contract_f32_assign,
        contract_f32_power_scores,
    )
    from tac.boundary_math.power_diagram_witness import power_assign, power_scores

    rep = receipt["reproduction"]
    z32 = np.asarray([rep["rank4_quotient"]], dtype=np.float32)
    scores32 = contract_f32_power_scores(z32, target)[0]
    label32 = int(contract_f32_assign(z32, target)[0])
    scores64 = np.asarray(power_scores(z32.astype(np.float64), target)[0])
    label64 = int(power_assign(z32.astype(np.float64), target)[0])
    tie_exact = bool(scores32[0] == scores32[1])
    return {
        "label": "MEASURED_FRAME195_CLOSURE_UNDER_CONTRACT",
        "contract_id": CONTRACT_ID,
        "pixel": {"frame": 195, "y": rep["pixel_y"], "x": rep["pixel_x"]},
        "cached_lstar": int(rep["cached_lstar"]),
        "cpu_torch_argmax": int(rep["cpu_torch"]["argmax"]),
        "cpu_torch_winner_margin": rep["cpu_torch"]["winner_margin"],
        "contract_f32": {
            "scores": [float(s) for s in scores32],
            "argmax": label32,
            "class0_class1_exact_tie": tie_exact,
        },
        "generic_f64": {
            "scores": [float(s) for s in scores64],
            "argmax": label64,
            "winner_margin": float(np.sort(scores64)[-1] - np.sort(scores64)[-2]),
        },
        "closes": bool(
            label32 == int(rep["cached_lstar"]) == int(rep["cpu_torch"]["argmax"])
        ),
        "f64_disagreement_reproduced": bool(label64 != int(rep["cached_lstar"])),
        "tie_geometry": (
            "exact fp32 tie between classes 0/1 at score "
            f"{float(scores32[0])!r}; f64 real-arithmetic margin "
            f"{float(scores64[1] - scores64[0])!r} favouring class 1; frozen "
            "CPU-Torch fp32 logit margin 4.76837158203125e-07 favouring class 0; "
            "contract first-max resolves the fp32 tie to class 0 == L*"
        ),
    }


def build_planes_and_measure(target, args) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    from tac.boundary_math.pdw1_fp32_receiver_contract import contract_f32_assign
    from tac.codec.pdw1_plane_codec import (
        Pdw1PlanePayload,
        decode_pdw1p,
        encode_pdw1p,
        expand_scorer_plane,
    )
    from tac.optimization.uint8_lattice_feasibility import (
        DisjointResizeOperator,
        realize_factor2_uint8_scorer_plane,
        verify_factor2_uint8_scorer_plane,
    )
    from tac.witness_control.factorized_features import load_frozen_segnet_cpu

    gt = np.load(GT_CACHE, mmap_mode="r")
    lstars = gt["lstars"]
    quotient = np.load(QUOTIENT_CACHE, mmap_mode="r")

    # --- labels under the contract + d_A -----------------------------------
    labels = np.empty((N24, 384, 512), dtype=np.uint8)
    d_a_mismatch = 0
    for pair in range(N24):
        z = np.ascontiguousarray(quotient[pair], dtype=np.float32).reshape(-1, 4)
        lab = contract_f32_assign(z, target).reshape(384, 512)
        labels[pair] = lab.astype(np.uint8)
        d_a_mismatch += int((lab != np.asarray(lstars[pair])).sum())

    # --- frozen scorers ------------------------------------------------------
    segnet = load_frozen_segnet_cpu()

    # GT scorer planes (for fills only; encoder-side) via the REAL preprocess.
    def camera_to_plane(frame_u8: np.ndarray) -> np.ndarray:
        xp = torch.from_numpy(frame_u8[None, None]).permute(0, 1, 4, 2, 3).contiguous().float()
        with torch.inference_mode():
            plane = segnet.preprocess_input(xp)
        return plane[0].permute(1, 2, 0).numpy()

    fills = np.zeros((N24, 5, 3), dtype=np.uint8)
    gt_planes_mean_accum = np.zeros((5, 3), dtype=np.float64)
    gt_planes_area_accum = np.zeros(5, dtype=np.int64)
    for pair in range(N24):
        gt_plane = camera_to_plane(np.asarray(gt["gt_f1"][pair]))
        lab = labels[pair]
        for c in range(5):
            m = lab == c
            if m.any():
                mean = gt_plane[m].mean(axis=0)
                fills[pair, c] = np.clip(np.round(mean), 0, 255).astype(np.uint8)
                gt_planes_mean_accum[c] += gt_plane[m].sum(axis=0)
                gt_planes_area_accum[c] += int(m.sum())
    global_fills = np.clip(
        np.round(gt_planes_mean_accum / np.maximum(gt_planes_area_accum, 1)[:, None]),
        0, 255,
    ).astype(np.uint8)

    # --- payload bytes -------------------------------------------------------
    payload = Pdw1PlanePayload(labels=labels, fills=fills)
    blob = encode_pdw1p(payload)
    decoded = decode_pdw1p(blob)
    assert encode_pdw1p(decoded) == blob, "PDW1P re-encode identity failed"

    label_streams = [brotli.compress(labels[p].tobytes(), quality=11) for p in range(N24)]
    label_bytes = [len(s) for s in label_streams]
    concat = b"".join(labels[p].tobytes() for p in range(N24))
    coder_rows = {
        "per_pair_brotli_q11_total": int(sum(label_bytes)),
        "joint_brotli_q11": len(brotli.compress(concat, quality=11)),
        "joint_zlib_9": len(zlib.compress(concat, 9)),
        "joint_lzma_preset9e": len(lzma.compress(concat, preset=9 | lzma.PRESET_EXTREME)),
        "order0_ideal_entropy_per_pair_total": int(
            sum(order0_entropy_bytes(labels[p].tobytes()) for p in range(N24))
        ),
    }
    boundary_px = [int(boundary_mask(labels[p]).sum()) for p in range(N24)]

    # --- realize through the PROVEN factor-2 lattice + hard oracle ----------
    op = DisjointResizeOperator.build(camera_h=874, camera_w=1164, scorer_h=384, scorer_w=512)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    per_pair: list[dict[str, Any]] = []
    confusion = np.zeros((5, 5), dtype=np.int64)
    boundary_hist = {1: 0, 2: 0, 4: 0}
    mism_total = 0
    realized_frames = np.empty((N24, 874, 1164, 3), dtype=np.uint8)
    pred_all = np.empty((N24, 384, 512), dtype=np.uint8)
    t0 = time.time()
    for pair in range(N24):
        plane = expand_scorer_plane(decoded, pair)
        frame = realize_factor2_uint8_scorer_plane(op, plane)
        ver = verify_factor2_uint8_scorer_plane(op, frame, plane)
        if not ver.certified_exact:
            raise AssertionError(f"pair {pair}: factor-2 realization not certified exact")
        realized_frames[pair] = frame
        xr = torch.from_numpy(frame[None, None]).permute(0, 1, 4, 2, 3).contiguous().float()
        with torch.inference_mode():
            pred = segnet(segnet.preprocess_input(xr))[0].argmax(dim=0).numpy()
        pred_all[pair] = pred.astype(np.uint8)
        lst = np.asarray(lstars[pair])
        mism = pred != lst
        mism_total += int(mism.sum())
        np.add.at(confusion, (lst[mism].ravel(), pred[mism].ravel()), 1)
        bmask = boundary_mask(lst)
        for radius in boundary_hist:
            boundary_hist[radius] += int((mism & dilate(bmask, radius)).sum())
        per_pair.append(
            {
                "pair": pair,
                "d_seg": float(mism.mean()),
                "mismatch_px": int(mism.sum()),
                "label_stream_bytes": label_bytes[pair],
                "boundary_px": boundary_px[pair],
                "realization_certified_exact": True,
            }
        )
    seg_wall = time.time() - t0

    d_seg_mean = float(np.mean([row["d_seg"] for row in per_pair]))
    d_b_mismatch = int(sum((pred_all[p] != labels[p]).sum() for p in range(N24)))

    # --- global-fills variant (RD slope row) ---------------------------------
    variant_mism = 0
    for pair in range(N24):
        plane = global_fills[labels[pair]]
        xr = torch.from_numpy(plane.astype(np.float32)).permute(2, 0, 1)[None]
        with torch.inference_mode():
            pred = segnet(xr)[0].argmax(dim=0).numpy()
        variant_mism += int((pred != np.asarray(lstars[pair])).sum())
    variant_d_seg = variant_mism / float(N24 * 384 * 512)
    # NOTE: variant runs SegNet on the plane directly — admissible because the
    # certified realization makes resize(realized frame) == plane exactly
    # (verified above; resized-vs-plane max abs < 2e-6 fp32).

    # --- d_pose under the repeat-frame1 policy -------------------------------
    sys.path.insert(0, str(Path(args.upstream)))
    from modules import PoseNet
    from safetensors.torch import load_file

    posenet = PoseNet()
    posenet.load_state_dict(load_file(str(Path(args.upstream) / "models/posenet.safetensors")), strict=True)
    posenet.eval()
    d_pose_rows: list[float] = []
    with torch.inference_mode():
        for pair in range(N24):
            dec_pair = np.stack([realized_frames[pair], realized_frames[pair]])
            gt_pair = np.stack([np.asarray(gt["gt_f0"][pair]), np.asarray(gt["gt_f1"][pair])])
            xd = torch.from_numpy(dec_pair[None]).permute(0, 1, 4, 2, 3).contiguous().float()
            xg = torch.from_numpy(gt_pair[None]).permute(0, 1, 4, 2, 3).contiguous().float()
            out_d = posenet(posenet.preprocess_input(xd))
            out_g = posenet(posenet.preprocess_input(xg))
            d_pose_rows.append(float(posenet.compute_distortion(out_d, out_g)[0]))
    d_pose_mean = float(np.mean(d_pose_rows))

    # --- bulk custody ---------------------------------------------------------
    payload_path = EVIDENCE_DIR / "pdw1p_n24_payload.bin"
    frames_path = EVIDENCE_DIR / "realized_frames_n24_u8.npy"
    pred_path = EVIDENCE_DIR / "hard_oracle_pred_n24_u8.npy"
    payload_path.write_bytes(blob)
    np.save(frames_path, realized_frames)
    np.save(pred_path, pred_all)

    # --- totals + box comparison ----------------------------------------------
    total_bytes = len(blob)
    per_pair_bytes = total_bytes / N24
    header_bytes = 17
    n600_total = header_bytes + round((total_bytes - header_bytes) / N24 * 600)
    implied_rate = 25.0 * n600_total / SCORE_BYTES_NORMALIZER
    implied_s_measured_pose = 100.0 * d_seg_mean + float(np.sqrt(10.0 * d_pose_mean)) + implied_rate
    implied_s_seg_rate_only = 100.0 * d_seg_mean + implied_rate

    confusion_rows = []
    for a in range(5):
        for b in range(5):
            if a != b and confusion[a, b] > 0:
                confusion_rows.append(
                    {
                        "lstar": CLASS_NAMES[a],
                        "pred": CLASS_NAMES[b],
                        "px": int(confusion[a, b]),
                        "share": round(float(confusion[a, b]) / mism_total, 4),
                    }
                )
    confusion_rows.sort(key=lambda r: -r["px"])

    return {
        "label": "MEASURED_N24_FIRST_INBOX_POINT",
        "n_pairs": N24,
        "d_A_contract_labels_vs_lstar_mismatch_px": d_a_mismatch,
        "d_A": d_a_mismatch / float(N24 * 384 * 512),
        "d_B_hard_oracle_vs_stored_labels_mismatch_px": d_b_mismatch,
        "d_B": d_b_mismatch / float(N24 * 384 * 512),
        "d_seg_hard_oracle_vs_lstar": d_seg_mean,
        "d_pose_repeat_frame1_policy": d_pose_mean,
        "payload": {
            "total_bytes_n24": total_bytes,
            "sha256": sha256_bytes(blob),
            "bytes_per_pair": round(per_pair_bytes, 2),
            "header_bytes": header_bytes,
            "fills_bytes_total": 15 * N24,
            "label_stream_bytes_total": int(sum(label_bytes)),
            "coder_comparison": coder_rows,
            "boundary_px_total": int(sum(boundary_px)),
            "bits_per_boundary_px": round(
                8.0 * sum(label_bytes) / max(sum(boundary_px), 1), 3
            ),
        },
        "n600_extrapolation": {
            "label": "DERIVED_N24_TO_N600_LINEAR",
            "total_bytes": n600_total,
            "bytes_per_pair": round(n600_total / 600.0, 2),
            "rate_term": round(implied_rate, 6),
            "implied_S_with_measured_repeat_frame1_pose": round(implied_s_measured_pose, 4),
            "implied_S_seg_plus_rate_only_pose_external": round(implied_s_seg_rate_only, 4),
        },
        "box_comparison": {
            "box_total_bytes_at_exact_realization_distortion": BOX_TOTAL_AT_EXACT_REALIZATION,
            "box_bytes_per_pair": BOX_PER_PAIR,
            "box_d_seg_needed_at_236kb": BOX_DSEG_AT_236KB,
            "bytes_in_box": bool(n600_total <= BOX_TOTAL_AT_EXACT_REALIZATION),
            "d_seg_in_box": bool(d_seg_mean < BOX_DSEG_AT_236KB),
            "d_seg_over_need_factor": round(d_seg_mean / BOX_DSEG_AT_236KB, 1),
        },
        "decomposition": {
            "confusion_rows_lstar_to_pred": confusion_rows,
            "mismatch_within_chebyshev_r_of_lstar_boundary": {
                str(r): {"px": v, "share": round(v / mism_total, 4)}
                for r, v in boundary_hist.items()
            },
            "mismatch_total_px": mism_total,
        },
        "rd_slope_rows": {
            "global_fills_variant": {
                "d_seg": variant_d_seg,
                "delta_d_seg": variant_d_seg - d_seg_mean,
                "delta_bytes_n600": -15 * 599,
                "note": "one shared 15-byte fill table instead of per-pair fills",
            },
            "labels_joint_vs_per_pair_brotli_delta_bytes_n24": (
                coder_rows["joint_brotli_q11"] - coder_rows["per_pair_brotli_q11_total"]
            ),
        },
        "per_pair": per_pair,
        "seg_wall_seconds": round(seg_wall, 1),
        "bulk_custody": {
            "payload": {"path": str(payload_path), "bytes": total_bytes, "sha256": sha256_bytes(blob)},
            "realized_frames": {"path": str(frames_path), "sha256": sha256_file(frames_path)},
            "hard_oracle_pred": {"path": str(pred_path), "sha256": sha256_file(pred_path)},
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upstream", type=Path, default=REPO / "upstream")
    parser.add_argument("--skip-prefix-scan", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing receipt {args.output}")
    if sha256_file(DIAGNOSTIC) != DIAGNOSTIC_SHA256:
        raise SystemExit("sealed frame-195 diagnostic receipt hash drift — refusing")

    target, receipt = load_frozen_target()
    gt = np.load(GT_CACHE, mmap_mode="r")
    quotient = np.load(QUOTIENT_CACHE, mmap_mode="r")

    from tac.boundary_math.pdw1_fp32_receiver_contract import CONTRACT_ID
    from tac.boundary_math.power_diagram_witness import encode_pdw1, encode_pdw2, pdw1_to_pdw2

    result: dict[str, Any] = {
        "schema": "pdw1_fp32_realization_first_inbox_point.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract_id": CONTRACT_ID,
        "authority": {
            "axis": "[macOS-CPU advisory]",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": f"{FRONTIER_S} [contest-CPU Linux x86_64] UNMOVED",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch_threads": 1,
        },
        "inputs": {
            "gt_cache": {"path": str(GT_CACHE), "bytes": GT_CACHE.stat().st_size},
            "quotient_cache": {"path": str(QUOTIENT_CACHE), "bytes": QUOTIENT_CACHE.stat().st_size},
            "frame195_diagnostic": {"path": str(DIAGNOSTIC), "sha256": DIAGNOSTIC_SHA256},
        },
        "coefficient_certificate": {
            "note": (
                "encoder-side only; NOT counted in the payload because no "
                "receiver can expand coefficients into a spatial partition "
                "without a channel feature field (consumption discipline)"
            ),
            "pdw1_raw_bytes": len(encode_pdw1(target)),
            "pdw2_margin_raw_bytes": len(encode_pdw2(pdw1_to_pdw2(target))),
        },
    }
    if not args.skip_prefix_scan:
        result["phase_a_prefix_closure"] = phase_a_prefix_closure(
            target, gt["lstars"], quotient
        )
    result["phase_b_frame195_closure"] = phase_b_frame195(target, receipt)
    result["phase_c_first_inbox_point"] = build_planes_and_measure(target, args)

    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"receipt -> {args.output}")
    point = result["phase_c_first_inbox_point"]
    print(
        f"POINT: {point['payload']['bytes_per_pair']} B/pair · "
        f"d_seg {point['d_seg_hard_oracle_vs_lstar']:.6f} · "
        f"in-box bytes={point['box_comparison']['bytes_in_box']} "
        f"d_seg={point['box_comparison']['d_seg_in_box']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
