#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""AMC-warm-start (arXiv 2607.10109) — per-ROW margin-saliency-TIERED code bit allocation.

MEANS. Pointer 0.19108282 [contest-CPU] UNMOVED. Authority: [macOS-CPU advisory;
NumPy-fp32 receiver; CPU frozen scorers] NON-PROMOTABLE — score_claim=false everywhere.
NO-FAKE: real frozen V9-CGauge EMA-best checkpoint, real #202 byte-close grammar (the
EXACT shipped inflate runtime is executed per candidate), real frozen CPU-torch SegNet
argmax + PoseNet through the contest R on all real-GT pairs.

WHY THIS EXISTS (the PAPER_WARM_START_FROM_DIVERGENCE deliverable for the AMC seed):
AMC's mechanism = multi-tier saliency detection over per-TOKEN units + a small per-tier
{rank, bit-width} menu. Its saliency signal (runtime activation L1) and its resource
(45nm CMOS energy) diverge from ours at the fork: our units are the witness `code` ROWS
(1200 per-frame latents = the exact per-token analog), our saliency is MEASURED
(per-pair baseline d_seg from the #336 n600 response artifact + the structural
frame-role fact), and our resource is ARCHIVE BYTES (the contest rate term). What
survives the fork is the TIER STRUCTURE: a small per-row bit-width menu assigned by
saliency percentile — the sub-tensor granularity that the settled #336 per-TENSOR
formulation (REJECT via joint non-additivity, memo
.omx/research/witness_sensitivity_bitalloc_336_20260713.md) never expressed.

STRUCTURAL FACTS this design leans on (verified against the render path in
tools/probe_witness_sensitivity_bitalloc.py::_render_pair):
  * code row 2i   = frame_0 of pair i  -> feeds PoseNet ONLY (d_pose exposure);
  * code row 2i+1 = frame_1 of pair i  -> feeds SegNet (d_seg) AND PoseNet; the
    self-orient feature loop also reads ONLY row 2i+1;
  * per-row QDQ on the TENSOR-GLOBAL absmax grid needs ZERO receiver changes and ZERO
    side-info bytes: the shipped int8 grammar re-quantizes the dequantized tensor with
    one scale, and low-bit rows simply produce fewer distinct int8 symbols (real brotli
    savings, measured on the exact blob).

REUSE, not re-derivation: candidate build (`_candidate`), realization (`_realize` /
`_intn_qdq_numpy`), the byte-closed inflate+score loop (`_measure_runtime_candidate`),
state handling and hashing are all imported AS-IS from
tools/probe_witness_sensitivity_bitalloc.py (the #336 n600 producer). This tool adds
ONLY: (1) the per-row tiered QDQ, (2) the tier-map construction (role tiers + AMC-style
3-tier saliency percentiles + a random-assignment falsifier control), (3) anchor
custody re-derivation so the already-MEASURED n600 baseline/uniform rows from the #336
artifact are reused by byte identity (same archive sha => same frames => same scores
under the identical scorer config), never re-asserted.

Honest bounds: this is a compress-half rate instrument on a frozen checkpoint; it
changes archive bytes, never trained weights. All rows are advisory; upstream
evaluate.py is operator-GO and is NOT run here.

Usage (chunked foreground; rc=7 = resumable boundary, re-run to continue):
  TAC_GOVERNED_ADMISSION=1 .venv/bin/python tools/safe_run.py -- \
    .venv/bin/python tools/apply_amc_saliency_tiered_bitalloc_witness.py \
      --ckpt-dir experiments/results/v9_cgauge_432_coherent_arm_20260711 \
      --out-dir experiments/results/amc_saliency_tiered_bitalloc_20260714 \
      --prior-dir experiments/results/witness_sensitivity_bitalloc_336_20260713T042157Z \
      --torch-threads 1 --chunk-seconds 240
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "tools", _REPO / "upstream"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from probe_witness_sensitivity_bitalloc import (  # noqa: E402
    _atomic_json,
    _candidate,
    _measure_runtime_candidate,
    _realize,
    _sha256_bytes,
    _sha256_file,
)

RATE_DENOM = 37_545_489.0
SCHEMA = "amc_saliency_tiered_code_bitalloc.v1"

#: AMC (arXiv 2607.10109) tier fractions: High=top 20%, Mid=next 30%, Low=bottom 50%.
AMC_TIER_FRACTIONS = (0.20, 0.30, 0.50)
#: Our bit menu for the three AMC tiers on frame_1 rows (AMC used {16,8,4}; our int8
#: grammar caps the top tier at 8 and the MEASURED n600 code curve places the knee
#: between int3 and int2, so the menu is {8,5,3}).
AMC_TIER_BITS = (8, 5, 3)

ARM_NAMES = (
    "role_f0int3_f1int8",
    "role_f0int2_f1int8",
    "role_f0int3_f1int4",
    "amc3_salient",
    "amc3_random",
    "pairkkt_f0int3",
)

#: Rungs available for the exact per-pair KKT arm (per-pair d_seg at each rung is
#: MEASURED in the prior #336 uniform-code units; int2 excluded — below the knee the
#: first-order model breaks and d_pose blows).
PAIRKKT_RUNGS = (8, 7, 6, 5, 4, 3)
#: Per-row byte proxy (B/row/bit) DERIVED from the MEASURED uniform code-stream brotli
#: ladder ((18459-4171)/(1200*5) ≈ 2.38); used ONLY to pick the Lagrange multiplier —
#: real bytes are measured on the exact blob afterwards.
PAIRKKT_BYTES_PER_ROW_BIT = 2.38
#: Target mean f1 bits for the KKT arm (matches the AMC tier mixture 0.2*8+0.3*5+0.5*3
#: so the two arms land in the same byte class and are directly comparable).
PAIRKKT_TARGET_MEAN_F1_BITS = 4.6


def tiered_code_qdq(code_fp: np.ndarray, row_bits: np.ndarray) -> np.ndarray:
    """Per-row symmetric int-N QDQ on the TENSOR-GLOBAL absmax grid.

    A constant ``row_bits == b`` map reproduces ``_intn_qdq_numpy(code, b)`` exactly
    (same global absmax scale), so the uniform rows of the #336 artifact are the
    degenerate case of this transform — the tiering demonstrably CHANGES the
    allocation only through the per-row map.
    """
    code = np.asarray(code_fp, np.float32)
    row_bits = np.asarray(row_bits, np.int64)
    if row_bits.shape != (code.shape[0],):
        raise ValueError(f"row_bits shape {row_bits.shape} != ({code.shape[0]},)")
    if not np.all((row_bits >= 2) & (row_bits <= 8)):
        raise ValueError("row bits must be within [2, 8]")
    absmax = float(np.abs(code).max())
    if absmax <= 0.0:
        return code.copy()
    out = np.empty_like(code)
    for bits in np.unique(row_bits):
        qmax = float(2 ** (int(bits) - 1) - 1)
        scale = absmax / qmax
        rows = row_bits == bits
        out[rows] = (np.clip(np.round(code[rows] / scale), -qmax, qmax) * scale).astype(
            np.float32
        )
    return out


def build_row_bits(
    arm: str, saliency: np.ndarray, n_pairs: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """Tier map per code row. Row 2i = frame_0 (pose-only), row 2i+1 = frame_1 (d_seg)."""
    row_bits = np.full(2 * n_pairs, 8, dtype=np.int64)
    meta: dict[str, Any] = {"arm": arm}
    if arm == "role_f0int3_f1int8":
        row_bits[0::2] = 3
    elif arm == "role_f0int2_f1int8":
        row_bits[0::2] = 2
    elif arm == "role_f0int3_f1int4":
        row_bits[0::2] = 3
        row_bits[1::2] = 4
    elif arm in {"amc3_salient", "amc3_random"}:
        row_bits[0::2] = 3
        if arm == "amc3_salient":
            # Descending MEASURED per-pair baseline d_seg; ties broken by pair id
            # (np.argsort stable on the negated key).
            order = np.argsort(-saliency, kind="stable")
            meta["saliency_source"] = "prior #336 baseline_repeat_a per-pair d_seg (MEASURED n600)"
        else:
            order = rng.permutation(n_pairs)
            meta["saliency_source"] = "random permutation control (falsifier; seed recorded)"
        n_hi = round(AMC_TIER_FRACTIONS[0] * n_pairs)
        n_mid = round(AMC_TIER_FRACTIONS[1] * n_pairs)
        f1_bits = np.full(n_pairs, AMC_TIER_BITS[2], dtype=np.int64)
        f1_bits[order[:n_hi]] = AMC_TIER_BITS[0]
        f1_bits[order[n_hi : n_hi + n_mid]] = AMC_TIER_BITS[1]
        row_bits[1::2] = f1_bits
        meta["tier_fractions"] = list(AMC_TIER_FRACTIONS)
        meta["tier_bits"] = list(AMC_TIER_BITS)
        meta["tier_counts"] = {
            str(AMC_TIER_BITS[0]): int(n_hi),
            str(AMC_TIER_BITS[1]): int(n_mid),
            str(AMC_TIER_BITS[2]): int(n_pairs - n_hi - n_mid),
        }
    elif arm == "pairkkt_f0int3":
        raise ValueError("pairkkt_f0int3 is built by build_pairkkt_row_bits (needs per-pair curves)")
    else:
        raise ValueError(f"unknown arm {arm!r}")
    meta["mean_bits_all_rows"] = float(np.mean(row_bits))
    meta["mean_bits_f0"] = float(np.mean(row_bits[0::2]))
    meta["mean_bits_f1"] = float(np.mean(row_bits[1::2]))
    meta["row_bits_sha256"] = _sha256_bytes(row_bits.tobytes())
    return row_bits, meta


def build_pairkkt_row_bits(
    per_pair_dseg: dict[int, dict[int, float]], n_pairs: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """EXACT per-pair KKT allocation of frame_1 bits from MEASURED per-pair responses.

    Structural additivity (the reason this escapes the 07-13 cross-tensor joint-REJECT):
    pair i's d_seg depends ONLY on its own frame_1 code row (params + self-orient both
    read row 2i+1 only), so mean d_seg over pairs composes EXACTLY from the per-pair
    rows of the prior uniform-code units. The Lagrangian
        cost_i(b) = 100 * dseg_i(b) / n_pairs + lam * PAIRKKT_BYTES_PER_ROW_BIT * b * 25 / RATE_DENOM
    is minimized per pair over PAIRKKT_RUNGS; ``lam`` is bisected so the mean f1 bits
    hits PAIRKKT_TARGET_MEAN_F1_BITS (same byte class as the amc3 arms).
    """

    def mean_bits(lam: float) -> tuple[float, np.ndarray]:
        picks = np.empty(n_pairs, dtype=np.int64)
        for i in range(n_pairs):
            best_b, best_c = None, None
            for b in PAIRKKT_RUNGS:
                c = 100.0 * per_pair_dseg[i][b] / n_pairs + (
                    lam * PAIRKKT_BYTES_PER_ROW_BIT * b * 25.0 / RATE_DENOM
                )
                if best_c is None or c < best_c:
                    best_b, best_c = b, c
            picks[i] = best_b
        return float(np.mean(picks)), picks

    lo, hi = 0.0, 1.0
    while mean_bits(hi)[0] > PAIRKKT_TARGET_MEAN_F1_BITS and hi < 1e9:
        hi *= 4.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mean_bits(mid)[0] > PAIRKKT_TARGET_MEAN_F1_BITS:
            lo = mid
        else:
            hi = mid
    lam = hi
    achieved, picks = mean_bits(lam)
    row_bits = np.full(2 * n_pairs, 8, dtype=np.int64)
    row_bits[0::2] = 3
    row_bits[1::2] = picks
    meta: dict[str, Any] = {
        "arm": "pairkkt_f0int3",
        "saliency_source": "EXACT per-pair MEASURED d_seg response at each rung (prior #336 "
        "uniform-code units); allocation = per-pair Lagrangian argmin",
        "lambda": float(lam),
        "target_mean_f1_bits": PAIRKKT_TARGET_MEAN_F1_BITS,
        "achieved_mean_f1_bits": achieved,
        "rung_histogram": {str(b): int(np.sum(picks == b)) for b in PAIRKKT_RUNGS},
        "predicted_d_seg_recombination": float(
            np.mean([per_pair_dseg[i][int(picks[i])] for i in range(n_pairs)], dtype=np.float64)
        ),
        "mean_bits_all_rows": float(np.mean(row_bits)),
        "mean_bits_f0": 3.0,
        "mean_bits_f1": achieved,
    }
    meta["row_bits_sha256"] = _sha256_bytes(row_bits.tobytes())
    return row_bits, meta


def load_per_pair_dseg_curves(
    prior_dir: Path, n_pairs: int
) -> dict[int, dict[int, float]]:
    """Per-pair d_seg at every uniform code rung from the prior #336 resume state."""
    resume = json.loads((prior_dir / "resume_state.json").read_text())
    units = resume["units"]
    curves: dict[int, dict[int, float]] = {i: {} for i in range(n_pairs)}
    for bits in PAIRKKT_RUNGS:
        label = "baseline_repeat_a" if bits == 8 else f"code:int{bits}"
        pairs = units[label]["pairs"]
        for i in range(n_pairs):
            curves[i][bits] = float(pairs[str(i)]["d_seg"])
    return curves


def load_prior_anchors(prior_dir: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Load the #336 n600 artifact: uniform-code anchor rows + per-pair baseline d_seg."""
    response = json.loads((prior_dir / "section_precision_response_curves.json").read_text())
    resume = json.loads((prior_dir / "resume_state.json").read_text())
    baseline_pairs = resume["units"]["baseline_repeat_a"]["pairs"]
    n_pairs = len(baseline_pairs)
    saliency = np.array(
        [float(baseline_pairs[str(i)]["d_seg"]) for i in range(n_pairs)], dtype=np.float64
    )
    anchors: dict[str, Any] = {"baseline": response["baseline"]}
    for row in response["rows"]:
        if row.get("tensor") == "code" and row.get("operation", "").startswith("int"):
            anchors[f"uniform_code_{row['operation']}"] = {
                key: row[key]
                for key in (
                    "archive_bytes",
                    "archive_sha256",
                    "blob_sha256",
                    "d_seg",
                    "d_pose",
                    "operation",
                )
            }
    return anchors, saliency


def uniform_rd_interpolate(anchors: dict[str, Any], bytes_at: float) -> dict[str, float] | None:
    """Piecewise-linear d_seg/d_pose on the MEASURED uniform-code RD curve at ``bytes_at``."""
    points = sorted(
        (
            (float(v["archive_bytes"]), float(v["d_seg"]), float(v["d_pose"]))
            for k, v in anchors.items()
            if k.startswith("uniform_code_int")
        ),
        key=lambda p: p[0],
    )
    base = anchors["baseline"]
    points.append((float(base["archive_bytes"]), float(base["d_seg"]), float(base["d_pose"])))
    points = sorted(set(points), key=lambda p: p[0])
    if not points or bytes_at < points[0][0] or bytes_at > points[-1][0]:
        return None
    for (b0, s0, p0), (b1, s1, p1) in itertools.pairwise(points):
        if b0 <= bytes_at <= b1:
            t = 0.0 if b1 == b0 else (bytes_at - b0) / (b1 - b0)
            return {
                "d_seg": s0 + t * (s1 - s0),
                "d_pose": p0 + t * (p1 - p0),
                "segment_bytes": [b0, b1],
            }
    return None


def delta_s(
    row: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, float]:
    dseg = float(row["d_seg"]) - float(baseline["d_seg"])
    droot = float(np.sqrt(10.0 * float(row["d_pose"]))) - float(
        np.sqrt(10.0 * float(baseline["d_pose"]))
    )
    dbytes = float(row["archive_bytes"]) - float(baseline["archive_bytes"])
    return {
        "delta_d_seg": dseg,
        "delta_sqrt10dpose": droot,
        "delta_archive_bytes": dbytes,
        "delta_S_advisory": 100.0 * dseg + droot + 25.0 * dbytes / RATE_DENOM,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--npz-name", default="levelset_witness_ema_BEST.npz")
    ap.add_argument(
        "--gt-cache",
        type=Path,
        default=Path("experiments/results/mlx_fleet_gt_cache/gt_n600.npz"),
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--prior-dir",
        type=Path,
        required=True,
        help="the #336 n600 artifact root (response curves + resume state)",
    )
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--arms", default=",".join(ARM_NAMES))
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--verdict-batch", type=int, default=4)
    ap.add_argument("--inflate-workers", type=int, default=16)
    ap.add_argument("--so-freq-across", type=float, default=32.0)
    ap.add_argument("--so-freq-along", type=float, default=8.0)
    ap.add_argument("--so-tau", type=float, default=4.0)
    ap.add_argument("--so-iters", type=int, default=4)
    ap.add_argument("--rand-seed", type=int, default=20260714)
    ap.add_argument("--max-new-pairs", type=int, default=0)
    ap.add_argument("--chunk-seconds", type=float, default=0.0)
    ap.add_argument(
        "--bytes-only",
        action="store_true",
        help="light mode: custody checks + REAL archive bytes per arm (brotli on the exact "
        "blob), no inflate/scoring — sub-GiB footprint; distortions stay owed",
    )
    ap.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("experiments/.scratch/amc_saliency_tiered_bitalloc"),
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (1 <= int(args.n_pairs) <= 600):
        raise ValueError("--n-pairs must be in [1,600]")
    out_dir = args.out_dir.resolve()
    if str(out_dir).startswith(("/tmp/", "/private/tmp/", "/var/tmp/")):
        raise ValueError("durable --out-dir may not be temporary")
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [arm.strip() for arm in str(args.arms).split(",") if arm.strip()]
    unknown = sorted(set(arms) - set(ARM_NAMES))
    if unknown:
        raise ValueError(f"unknown arms {unknown}; known: {ARM_NAMES}")

    source = (args.ckpt_dir / args.npz_name).resolve()
    frozen = out_dir / "frozen_source_checkpoint.npz"
    if not frozen.exists():
        shutil.copy2(source, frozen)
    source_sha = _sha256_file(source)
    if _sha256_file(frozen) != source_sha:
        raise ValueError("frozen checkpoint copy SHA mismatch")

    anchors, saliency = load_prior_anchors(args.prior_dir.resolve())
    if len(saliency) < int(args.n_pairs):
        raise ValueError("prior per-pair saliency does not cover --n-pairs")

    import levelset_byte_close_and_eval as bc
    import torch
    from train_witness_realized_through_R_mlx import load_gt_from_cache

    torch.set_num_threads(int(args.torch_threads))
    loaded, cfg = bc._load_levelset_ckpt(out_dir, frozen.name)
    code_fp = np.asarray(loaded.pop("code"), np.float32)
    params_fp = {k: np.asarray(v, np.float32) for k, v in loaded.items()}
    so = bc.detect_self_orient(
        cfg,
        {
            "freq_across": args.so_freq_across,
            "freq_along": args.so_freq_along,
            "tau": args.so_tau,
            "iters": args.so_iters,
        },
    )

    # ---- anchor custody re-derivation: byte identity, never assertion ------------------
    custody_checks: dict[str, Any] = {}
    p0, c0 = _realize(params_fp, code_fp, None, "int8")
    baseline_candidate = _candidate(bc, cfg, so, p0, c0)
    custody_checks["baseline_int8"] = {
        "rederived_archive_sha256": baseline_candidate["archive_sha256"],
        "prior_archive_sha256": anchors["baseline"]["archive_sha256"],
        "byte_identical": baseline_candidate["archive_sha256"]
        == anchors["baseline"]["archive_sha256"],
    }
    # The all-8 tier map must reproduce the baseline blob exactly (degenerate-case pin).
    all8 = tiered_code_qdq(code_fp, np.full(code_fp.shape[0], 8, dtype=np.int64))
    p_a8, c_a8 = _realize({**params_fp}, all8, None, "int8")
    all8_candidate = _candidate(bc, cfg, so, p_a8, c_a8)
    custody_checks["tiermap_all8_equals_baseline"] = {
        "byte_identical": all8_candidate["archive_sha256"]
        == baseline_candidate["archive_sha256"],
    }
    for bits in (6, 5, 4, 3):
        key = f"uniform_code_int{bits}"
        if key not in anchors:
            continue
        uni = tiered_code_qdq(code_fp, np.full(code_fp.shape[0], bits, dtype=np.int64))
        p_u, c_u = _realize({**params_fp}, uni, None, "int8")
        cand_u = _candidate(bc, cfg, so, p_u, c_u)
        custody_checks[key] = {
            "rederived_archive_sha256": cand_u["archive_sha256"],
            "prior_archive_sha256": anchors[key]["archive_sha256"],
            "byte_identical": cand_u["archive_sha256"] == anchors[key]["archive_sha256"],
        }
    failed = sorted(k for k, v in custody_checks.items() if not v["byte_identical"])
    if failed:
        _atomic_json(out_dir / "custody_checks.json", custody_checks)
        raise RuntimeError(
            "anchor custody re-derivation failed byte identity; refusing to reuse prior "
            f"measured rows: {failed}"
        )
    _atomic_json(out_dir / "custody_checks.json", custody_checks)

    # ---- tiered candidates --------------------------------------------------------------
    rng = np.random.default_rng(int(args.rand_seed))
    pairkkt_curves = (
        load_per_pair_dseg_curves(args.prior_dir.resolve(), int(args.n_pairs))
        if "pairkkt_f0int3" in arms
        else None
    )
    plan: list[dict[str, Any]] = []
    for arm in arms:
        if arm == "pairkkt_f0int3":
            row_bits, meta = build_pairkkt_row_bits(pairkkt_curves, int(args.n_pairs))
        else:
            row_bits, meta = build_row_bits(
                arm, saliency[: int(args.n_pairs)], int(args.n_pairs), rng
            )
        code_t = tiered_code_qdq(code_fp, row_bits)
        p_t, c_t = _realize({**params_fp}, code_t, None, "int8")
        candidate = _candidate(bc, cfg, so, p_t, c_t)
        plan.append({"label": arm, "meta": meta, "candidate": candidate,
                     "row_bits": row_bits.tolist()})

    if args.bytes_only:
        bytes_rows = []
        for spec in plan:
            row = {
                "label": spec["label"],
                "kind": "bytes_MEASURED_real_brotli_blob; distortions OWED (through-R scoring "
                "pending governed admission)",
                "archive_bytes": int(spec["candidate"]["archive_bytes"]),
                "archive_sha256": spec["candidate"]["archive_sha256"],
                "blob_sha256": spec["candidate"]["blob_sha256"],
                "delta_archive_bytes": int(spec["candidate"]["archive_bytes"])
                - int(anchors["baseline"]["archive_bytes"]),
                "tier_meta": spec["meta"],
            }
            bytes_rows.append(row)
            print(
                f"[amc-tiered bytes-only] {row['label']:24s} bytes={row['archive_bytes']:6d} "
                f"delta_vs_baseline={row['delta_archive_bytes']:+d}",
                flush=True,
            )
        _atomic_json(
            out_dir / "amc_tiered_bytes_only.json",
            {
                "schema": SCHEMA + ".bytes_only",
                "score_claim": False,
                "baseline": anchors["baseline"],
                "rows": bytes_rows,
                "custody_checks": custody_checks,
            },
        )
        return 0

    fingerprint_payload = {
        "schema": SCHEMA,
        "ckpt_sha256": source_sha,
        "gt_cache_sha256": _sha256_file(Path(args.gt_cache)),
        "prior_response_sha256": _sha256_file(
            args.prior_dir / "section_precision_response_curves.json"
        ),
        "n_pairs": int(args.n_pairs),
        "arms": arms,
        "arm_row_bits_sha256": {spec["label"]: spec["meta"]["row_bits_sha256"] for spec in plan},
        "rand_seed": int(args.rand_seed),
        "so": [args.so_freq_across, args.so_freq_along, args.so_tau, args.so_iters],
        "torch_threads": int(args.torch_threads),
        "verdict_batch": int(args.verdict_batch),
        "inflate_workers": int(args.inflate_workers),
    }
    fingerprint = _sha256_bytes(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    )
    state_path = out_dir / "resume_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("fingerprint") != fingerprint:
            raise ValueError("resume fingerprint mismatch; refusing to mix evidence")
    else:
        state = {
            "schema": SCHEMA + ".resume",
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "units": {},
            "final_inflate_chunks": {},
        }
        _atomic_json(state_path, state)

    gt, seg_cpu, pose_cpu = load_gt_from_cache(args.gt_cache, int(args.n_pairs))
    batch_control_path = out_dir / "scorer_batch_axis_control.json"
    deadline = time.monotonic() + args.chunk_seconds if args.chunk_seconds > 0 else None
    new_pairs = 0
    for spec in plan:
        label = spec["label"]
        unit = state["units"].setdefault(label, {"pairs": {}})
        candidate = spec["candidate"]
        custody = {
            "archive_bytes": candidate["archive_bytes"],
            "archive_sha256": candidate["archive_sha256"],
            "blob_sha256": candidate["blob_sha256"],
        }
        for key, value in custody.items():
            if key in unit and unit[key] != value:
                raise ValueError(f"{label}: candidate custody drift in {key}")
            unit[key] = value
        unit["tier_meta"] = spec["meta"]
        complete, new_pairs = _measure_runtime_candidate(
            bc=bc,
            label=label,
            candidate=candidate,
            unit=unit,
            state=state,
            state_path=state_path,
            out_dir=out_dir,
            scratch_root=args.scratch_root,
            gt=gt,
            seg_cpu=seg_cpu,
            pose_cpu=pose_cpu,
            n_pairs=int(args.n_pairs),
            verdict_batch=int(args.verdict_batch),
            inflate_workers=int(args.inflate_workers),
            deadline=deadline,
            max_new_pairs=int(args.max_new_pairs),
            new_pairs=new_pairs,
            batch_control_path=batch_control_path,
        )
        if not complete:
            print(
                f"[amc-tiered] resumable boundary after {new_pairs} new pair measurements; "
                f"state={state_path}"
            )
            return 7
        ordered = [unit["pairs"][str(i)] for i in range(int(args.n_pairs))]
        dseg = float(np.mean([r["d_seg"] for r in ordered], dtype=np.float64))
        dpose = float(np.mean([r["d_pose"] for r in ordered], dtype=np.float64))
        print(
            f"[amc-tiered] complete {label}: n={args.n_pairs} d_seg={dseg:.9g} "
            f"d_pose={dpose:.9g} archive={unit['archive_bytes']}B",
            flush=True,
        )
        if deadline is not None and time.monotonic() >= deadline:
            print(f"[amc-tiered] resumable boundary after completed arm {label}")
            return 7

    # ---- report --------------------------------------------------------------------------
    baseline = anchors["baseline"]
    rows: list[dict[str, Any]] = []
    for spec in plan:
        label = spec["label"]
        unit = state["units"][label]
        ordered = [unit["pairs"][str(i)] for i in range(int(args.n_pairs))]
        row = {
            "label": label,
            "kind": f"tiered_MEASURED_n{int(args.n_pairs)}",
            "archive_bytes": int(unit["archive_bytes"]),
            "archive_sha256": unit["archive_sha256"],
            "d_seg": float(np.mean([r["d_seg"] for r in ordered], dtype=np.float64)),
            "d_pose": float(np.mean([r["d_pose"] for r in ordered], dtype=np.float64)),
            "tier_meta": spec["meta"],
        }
        row.update(delta_s(row, baseline))
        matched = uniform_rd_interpolate(anchors, float(row["archive_bytes"]))
        if matched is not None:
            row["uniform_curve_at_same_bytes"] = matched
            row["d_seg_vs_uniform_at_same_bytes"] = row["d_seg"] - matched["d_seg"]
            row["d_pose_vs_uniform_at_same_bytes"] = row["d_pose"] - matched["d_pose"]
        rows.append(row)
    anchor_rows = []
    for key, val in anchors.items():
        entry = dict(val)
        entry["label"] = key
        entry["kind"] = "prior_MEASURED_anchor_byte_identity_verified"
        if key != "baseline":
            entry.update(delta_s(entry, baseline))
        anchor_rows.append(entry)
    report = {
        "schema": SCHEMA,
        "fingerprint": fingerprint,
        "authority": "[macOS-CPU advisory; NumPy-fp32 receiver; CPU frozen scorers] NON-PROMOTABLE",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "eval_pairs": int(args.n_pairs),
        "baseline": baseline,
        "anchor_rows": anchor_rows,
        "tiered_rows": rows,
        "custody_checks": custody_checks,
        "seed_paper": "arXiv 2607.10109 (AMC) — tier structure warm-started; saliency + "
        "resource substituted at the divergence fork",
        "verdict_scope": "INSTANCE x FORMULATION: frozen V9 ep150 EMA-best; post-hoc per-row "
        "tiered QDQ on the code tensor through the unchanged int8 grammar",
    }
    _atomic_json(out_dir / "amc_tiered_report.json", report)
    print(json.dumps({k: report[k] for k in ("schema", "eval_pairs")}, indent=1))
    for row in rows:
        print(
            f"  {row['label']:24s} bytes={row['archive_bytes']:6d} d_seg={row['d_seg']:.9f} "
            f"d_pose={row['d_pose']:.6f} dS={row['delta_S_advisory']:+.6f}"
            + (
                f" d_seg_vs_uniform@bytes={row['d_seg_vs_uniform_at_same_bytes']:+.9f}"
                if "d_seg_vs_uniform_at_same_bytes" in row
                else ""
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
