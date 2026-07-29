# SPDX-License-Identifier: MIT
"""ddm_xi1 — carried-ξ token INTER-prediction race harness (QA39; HOTZ-constrained).

Pure CODER race: no scorer, no evaluator, no dispatch, no pointer mutation.  The
counted object is the frozen endpoint token array
``[600,24,32,4]`` (SHA-bound to the pb1 composed archive).  Every row is a
COMPLETE materialized token frame that decodes to the exact input and re-encodes
byte-identically before admission.

The falsifier (charter): if BOTH the warp-context expert AND the innovation
alphabet produce a complete-frame total >= the SMEVR baseline, the negative is
INSTANCE-scoped to this warp chart + token alphabet and QA39 flips FIRED.

Axis: [macOS-CPU advisory, rate-only] lossless byte measurement.
Pointer 0.1910828242 [contest-CPU] UNMOVED.  score_claim=false.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/adpena/Projects/pact")
for entry in (str(REPO), str(REPO / "src"), str(REPO / "experiments")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from tac.optimization import ddm_tr1_runtime as rt  # noqa: E402

import ddm_r7_token_coder as r7  # noqa: E402
import ddm_xi1_carried_xi_coder as xi1  # noqa: E402

SEG_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_pb1_20260729/p2c_aimed_archive.zip")
TARGETS = Path(
    "/Volumes/VertigoDataTier/pact/"
    "ddm_ms4_metric_producers_and_measurement_20260724T042005Z/"
    "pose_metric_n600_batch32.json"
)
ST_JSONL = Path("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/d1_warp_solve.partial.jsonl")
SSD_OUT = Path("/Volumes/VertigoDataTier/pact/ddm_xi1_20260729")
POINTER = "0.1910828242 [contest-CPU] UNMOVED"
CHARTER_FALSIFIER = 557_238  # r7 memo SMEVR endpoint complete frame
CEILING_0172 = 190_334
CEILING_015 = 157_294
LEVELS = 16


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_carried_pose(n: int) -> tuple[np.ndarray, np.ndarray]:
    rows = json.loads(TARGETS.read_text())["rows"]
    tp = np.stack([np.asarray(rows[i]["center"], np.float64) for i in range(n)], 0)
    tp = tp.astype(np.float16).astype(np.float64)  # SHIPPED f16 quantization
    per: dict[int, int] = {}
    for line in ST_JSONL.read_text().splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            per[int(rec["pair"])] = int(rec["s_t_idx"])
    st = np.asarray([per[i] for i in range(n)], dtype=np.int64)
    return tp, st


def _cond_entropy_bytes(sym: np.ndarray, *ctx: np.ndarray) -> float:
    """Plug-in H(sym|ctx) in TOTAL bytes over all symbols (orientation only)."""
    sym = sym.reshape(-1).astype(np.int64)
    key = np.zeros_like(sym)
    for context in ctx:
        flat = context.reshape(-1).astype(np.int64)
        key = key * (int(flat.max()) + 1) + flat
    order = np.argsort(key, kind="stable")
    ks = key[order]
    ss = sym[order]
    n = len(ks)
    _uniq, starts = np.unique(ks, return_index=True)
    bounds = list(starts) + [n]
    total_bits = 0.0
    for a in range(len(bounds) - 1):
        block = ss[bounds[a] : bounds[a + 1]]
        counts = np.bincount(block, minlength=LEVELS).astype(np.float64)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        total_bits += block.size * float(-(probs * np.log2(probs)).sum())
    return total_bits / 8.0


def main() -> int:
    np.seterr(all="ignore")
    SSD_OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    parsed = rt.parse_archive(SEG_ARCHIVE.read_bytes())
    codes = np.ascontiguousarray(parsed.packet.token_codes, dtype=np.uint8)
    pairs, height, width, channels = codes.shape
    tp, st_idx = _load_carried_pose(pairs)
    base, delta = r7.factor_mode_delta(codes, LEVELS)

    # --- SMEVR baseline (same-object), reproduced + roundtrip-verified ---
    smevr_frame = r7.encode_token_codes(codes, levels=LEVELS, codec="smevr")
    smevr_rb = r7.decode_token_codes(smevr_frame)
    if not np.array_equal(np.asarray(smevr_rb, dtype=np.uint8), codes):
        raise SystemExit("SMEVR baseline roundtrip differs")
    smevr_acc = r7.frame_accounting(smevr_frame)
    baseline_bytes = smevr_acc.framed_bytes

    rows: list[dict] = []

    def race(variant: str, forward: bool) -> dict:
        frame = xi1.encode_token_codes(
            codes, tp, st_idx, levels=LEVELS, variant=variant, forward=forward
        )
        restored = xi1.decode_token_codes(frame, tp, st_idx)
        exact = bool(np.array_equal(np.asarray(restored, dtype=np.uint8), codes))
        if not exact:
            raise SystemExit(f"{variant} forward={forward} roundtrip differs")
        acc = xi1.frame_accounting(frame)
        row = {
            "variant": variant,
            "warp_direction": acc.warp_direction,
            "complete_frame_bytes": acc.framed_bytes,
            "header_bytes": acc.header_bytes,
            "base_bytes": acc.base_bytes,
            "delta_bytes": acc.delta_bytes,
            "roundtrip_exact": exact,
            "frame_sha256": acc.sha256,
            "delta_bytes_vs_smevr": acc.delta_bytes - smevr_acc.delta_bytes,
            "complete_vs_smevr": acc.framed_bytes - baseline_bytes,
            "complete_vs_charter_falsifier": acc.framed_bytes - CHARTER_FALSIFIER,
        }
        rows.append(row)
        print(
            f"[{variant:22s} {acc.warp_direction:8s}] complete={acc.framed_bytes} "
            f"(delta={acc.delta_bytes}) vs SMEVR {baseline_bytes} "
            f"(Δ{row['complete_vs_smevr']:+d}) vs charter {CHARTER_FALSIFIER} "
            f"(Δ{row['complete_vs_charter_falsifier']:+d})",
            flush=True,
        )
        return row

    print(f"SMEVR baseline complete frame: {baseline_bytes} "
          f"(header {smevr_acc.header_bytes} base {smevr_acc.base_bytes} "
          f"delta {smevr_acc.delta_bytes})", flush=True)
    for variant in ("smevr_warp_context", "smevr_warp_innovation"):
        for forward in (False, True):  # backward first (better plug-in)
            race(variant, forward)

    # --- Wyner floor: H(tokens_t | ξ-warp(tokens_{t-1})) plug-in, both directions ---
    base_b = np.broadcast_to(base[None], codes.shape)
    prev_coloc = np.zeros_like(delta)
    prev_coloc[1:] = delta[:-1]
    wyner: dict[str, float] = {
        "H_delta": _cond_entropy_bytes(delta),
        "H_delta_given_base": _cond_entropy_bytes(delta, base_b),
        "H_delta_given_base_prevcoloc": _cond_entropy_bytes(delta, base_b, prev_coloc),
    }
    K, Kinv = xi1._token_grid_intrinsics(height, width)
    grid = xi1._target_grid(height, width)
    for forward in (False, True):
        pred_code = np.zeros_like(codes)
        for i in range(1, pairs):
            s_t = float(xi1.ST_GRID[int(st_idx[i])])
            H_mat = xi1._ground_homography(tp[i], K, Kinv, s_t)
            for c in range(channels):
                w = xi1._warp_channel(
                    codes[i - 1, :, :, c].astype(np.float64),
                    H_mat, grid, height, width, forward=forward,
                )
                pred_code[i, :, :, c] = np.clip(np.rint(w), 0, LEVELS - 1).astype(np.uint8)
        pred_code[0] = base
        # conditional entropy of the CURRENT CODE given the warped previous code
        tag = "bwd" if not forward else "fwd"
        wyner[f"H_code_given_warpprev_{tag}"] = _cond_entropy_bytes(codes, pred_code)
        wyner["H_code"] = _cond_entropy_bytes(codes)
        # innovation order-0 (replace-predictor) and warp-as-extra-context on delta
        innov = ((codes.astype(np.int16) - pred_code.astype(np.int16)) % LEVELS).astype(np.uint8)
        wyner[f"H_innov_order0_{tag}"] = _cond_entropy_bytes(innov)
        wdelta = ((pred_code.astype(np.int16) - base_b.astype(np.int16)) % LEVELS).astype(np.uint8)
        wyner[f"H_delta_given_base_prevcoloc_warp_{tag}"] = _cond_entropy_bytes(
            delta, base_b, prev_coloc, wdelta
        )

    best = min(rows, key=lambda r: r["complete_frame_bytes"])
    falsifier_fired = best["complete_frame_bytes"] >= baseline_bytes
    receipt = {
        "schema": "ddm_xi1_carried_xi_race.v1",
        "axis": "[macOS-CPU advisory, rate-only] lossless byte measurement",
        "pointer": POINTER,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "counted_object": "endpoint token array [600,24,32,4] levels 16",
        "seg_archive": str(SEG_ARCHIVE),
        "seg_archive_sha256": _file_sha(SEG_ARCHIVE),
        "token_shape": list(codes.shape),
        "token_raw_bytes": int(codes.size),
        "smevr_baseline": {
            "complete_frame_bytes": baseline_bytes,
            "header_bytes": smevr_acc.header_bytes,
            "base_bytes": smevr_acc.base_bytes,
            "delta_bytes": smevr_acc.delta_bytes,
            "frame_sha256": smevr_acc.sha256,
            "roundtrip_exact": True,
        },
        "charter_falsifier_bytes": CHARTER_FALSIFIER,
        "carried_pose": {
            "tp_source": str(TARGETS),
            "tp_source_sha256": _file_sha(TARGETS),
            "st_source": str(ST_JSONL),
            "st_source_sha256": _file_sha(ST_JSONL),
            "note": "t_p (600x6 f16) + s_t index are the archive's OWN pose member "
                    "payload (pfs1 grammar v3), already counted for frame_0; used "
                    "here as ZERO-marginal-byte token context (wr1 pool: COMPETE, "
                    "never sum).",
        },
        "rows": rows,
        "best_row": best,
        "falsifier_fired": falsifier_fired,
        "wyner_conditional_entropy_floor_bytes": wyner,
        "wyner_note": "PLUG-IN order-0 conditional entropies (total bytes), "
                      "orientation only — NOT a theorem, score, or achievable "
                      "coder bound; the adaptive coder pays model-learning + "
                      "context-dilution cost the plug-in ignores.",
        "ceilings": {"target_0172": CEILING_0172, "target_015": CEILING_015},
        "wall_seconds": time.time() - t0,
    }
    out = SSD_OUT / "ddm_xi1_race_receipt.json"
    out.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({
        "best": best,
        "falsifier_fired": falsifier_fired,
        "wyner_H_code_given_warpprev_bwd": wyner.get("H_code_given_warpprev_bwd"),
        "wyner_H_delta_given_base_prevcoloc": wyner["H_delta_given_base_prevcoloc"],
        "wyner_H_delta_given_base_prevcoloc_warp_bwd": wyner.get(
            "H_delta_given_base_prevcoloc_warp_bwd"),
    }, indent=1), flush=True)
    print(f"receipt: {out} sha {_file_sha(out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
