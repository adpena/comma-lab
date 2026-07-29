# SPDX-License-Identifier: MIT
"""ddm_wr1 — sensitivity-weighted reverse-waterfill on the DR7 token lattice.

Task #766 / ledger QA06 — the sole named rate lever from the ~557 KB SMEVR
endpoint token stream toward the 130-200 KB archive band.

WHAT THIS DOES (all $0, no scorer):
  * Decodes the endpoint token grid ``[600,24,32,4]`` (levels 16) from a
    pfs1 v3-warp archive via the landed r7 SMEVR coder (import, never fork).
  * Builds the per-cell (24x32) sensitivity map: current SegNet flip mass
    (ru1 atlas) + residual byte mass (decoded grid) + pose-region typing.
  * Reverse-waterfill: greedy drop-to-mode ordered safest-per-byte first
    (flip-risk asc, residual mass desc), COMPOSED re-pricing = re-run the
    REAL SMEVR coder on the whole dropped grid per tranche (knee-law
    compliant; never sum per-cell savings — pb1 P4 additivity is 8.8%).
  * Emits the (archive bytes -> predicted composed S) descent curve vs the
    MEASURED pfs1 D1 row S=2.256641 (569,996 B / d_seg 0.00389011 /
    d_pose 0.22144216), and byte-closes the two knee candidates as exact
    archive bytes for the STAGED realized gate.

AUTHORITY: bytes are MEASURED (lossless, through the real coder). d_seg and
d_pose are PREDICTED / TYPED here and are resolved ONLY by the staged n600
realized gate (stage_wr1_realized_gate.sh). Pointer 0.1910828242
[contest-CPU] UNMOVED; score_claim=false; promotion_eligible=false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np

# HOTZ CONSTRAINT: import (never fork) the landed SMEVR primitives.
from ddm_r7_token_coder import (  # type: ignore[import-not-found]
    decode_token_codes,
    encode_token_codes,
    factor_mode_delta,
    reconstruct_mode_delta,
)

LEVELS = 16
UNCOMPRESSED = 37_545_489
TOTAL_PX = 600 * 512 * 384  # 117,964,800 argmax pixels
# Measured pfs1 D1 exact-protocol reference row (real evaluator, real bytes).
REF_S = 2.256641
REF_ARCHIVE_BYTES = 569_996
REF_TOKENS_BYTES = 557_253
REF_DSEG = 0.00389011
REF_DPOSE = 0.22144216
# PR130-grade solved-distortion contributions (lessons-only existence proof).
SEG_SOLVED_TERM = 100 * 1.52e-4  # rp1 q1 realized-uint8 solved seg = 0.0152
POSE_SOLVED_TERM = 0.0153  # PR130 d_pose 2.33e-5 -> contribution 0.0153
BAR = 0.172  # T_1 effective bar (min of 0.15 target and official best ~0.172)
T3 = 0.15
KAPPA_MED = 0.07532024383544922  # ru1 logits/quantum (median of per-cell medians)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pose_band(cell_row: int) -> str:
    """comma10k row-band -> PoseNet ego-motion relevance of the cell."""
    if cell_row <= 8:
        return "sky_undriv_top"  # above horizon, uniform: low pose content
    if cell_row <= 17:
        return "road_lane_midband"  # road plane: HIGH ego-motion cue
    return "mycar_hood_bottom"  # static ego vehicle: low motion cue


def load_grid(archive_zip: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, bytes]:
    frame = zipfile.ZipFile(archive_zip).read("state/tokens.dr7t")
    codes = decode_token_codes(frame)
    base, delta = factor_mode_delta(codes, LEVELS)
    return codes, base, delta, frame


def cell_sensitivity(delta: np.ndarray, atlas_flat: Path) -> dict:
    """Per-cell residual byte mass + ru1 SegNet flip mass + drop ordering."""
    signed = delta.astype(np.int16)
    signed = np.where(signed > LEVELS // 2, signed - LEVELS, signed)
    residual_mass = np.abs(signed).sum(axis=(0, 3)).astype(np.float64).reshape(-1)
    atlas = np.load(atlas_flat)
    cell = (atlas["y"].astype(np.int64) // 16) * 32 + (atlas["x"].astype(np.int64) // 16)
    flip_mass = np.bincount(cell, minlength=768).astype(np.float64)
    # Safest-per-byte first: flip-risk ascending, ties by residual mass desc
    # (drop the fattest provably-safe cell first for the steepest early descent).
    order = np.lexsort((-residual_mass, flip_mass))
    rank = np.empty(768, dtype=np.int64)
    rank[order] = np.arange(768)
    return {
        "residual_mass": residual_mass,
        "flip_mass": flip_mass,
        "order": order,
        "rank": rank,
        "n_zero_flip_cells": int((flip_mass == 0).sum()),
    }


def _encode_dropped(base: np.ndarray, delta: np.ndarray, sel: np.ndarray) -> bytes:
    dropped = delta.copy()
    mask = np.zeros(768, dtype=bool)
    mask[sel] = True
    dropped[:, mask.reshape(24, 32), :] = 0
    codes = reconstruct_mode_delta(base, dropped, LEVELS)
    return encode_token_codes(codes, levels=LEVELS, codec="smevr")


def descent_curve(
    base: np.ndarray,
    delta: np.ndarray,
    sens: dict,
    archive_floor: int,
    checkpoints: list[int],
) -> list[dict]:
    order = sens["order"]
    flip_mass = sens["flip_mass"]
    rows = []
    for k in checkpoints:
        sel = order[:k]
        t0 = time.time()
        token_bytes = len(_encode_dropped(base, delta, sel))
        archive_bytes = archive_floor + token_bytes
        rate = 25 * archive_bytes / UNCOMPRESSED
        dropped_flip_mass = float(flip_mass[sel].sum())
        dseg_ceiling = REF_DSEG + dropped_flip_mass / TOTAL_PX
        bands = {"sky_undriv_top": 0, "road_lane_midband": 0, "mycar_hood_bottom": 0}
        for cr in sel // 32:
            bands[_pose_band(int(cr))] += 1
        pose_term_ref = (10 * REF_DPOSE) ** 0.5
        s_solved = SEG_SOLVED_TERM + POSE_SOLVED_TERM + rate
        rows.append(
            {
                "k_cells_dropped": int(k),
                "tokens_bytes": token_bytes,
                "archive_bytes": archive_bytes,
                "tokens_saved": REF_TOKENS_BYTES - token_bytes,
                "rate_term": round(rate, 6),
                "rate_isolated_delta_S": round(rate - 25 * REF_ARCHIVE_BYTES / UNCOMPRESSED, 6),
                "dropped_flip_mass": int(dropped_flip_mass),
                "dseg_pred_ceiling": round(dseg_ceiling, 7),
                "S_vs_ref_flipfree": round(100 * REF_DSEG + pose_term_ref + rate, 6),
                "S_vs_ref_flipceiling": round(100 * dseg_ceiling + pose_term_ref + rate, 6),
                "S_composed_if_solved": round(s_solved, 6),
                "sub_bar_if_solved": bool(s_solved < BAR),
                "sub_015_if_solved": bool(s_solved < T3),
                "dropped_bands": bands,
                "encode_seconds": round(time.time() - t0, 1),
            }
        )
    return rows


def byte_close(
    source_zip: Path, base: np.ndarray, delta: np.ndarray, sel: np.ndarray, out_zip: Path
) -> dict:
    """Write a valid pfs1 v3-warp archive with the dropped token member.

    The pfs1 inflate_runner does NOT enforce manifest tokens_sha256, so a
    surgical member swap yields a decodable archive (verified: 2-pair render
    smoke). No source file is edited.
    """
    token_bytes = _encode_dropped(base, delta, sel)
    roundtrip = decode_token_codes(token_bytes)  # canonical closure
    dropped = delta.copy()
    mask = np.zeros(768, dtype=bool)
    mask[sel] = True
    dropped[:, mask.reshape(24, 32), :] = 0
    if not np.array_equal(roundtrip, reconstruct_mode_delta(base, dropped, LEVELS)):
        raise RuntimeError("byte-close roundtrip mismatch")
    src = zipfile.ZipFile(source_zip)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as out:
        for item in src.infolist():
            data = token_bytes if item.filename == "state/tokens.dr7t" else src.read(item.filename)
            out.writestr(item.filename, data)
    archive_disk = out_zip.read_bytes()
    return {
        "archive_zip": str(out_zip),
        "tokens_bytes": len(token_bytes),
        "tokens_sha256": _sha(token_bytes),
        "archive_bytes": len(archive_disk),
        "archive_sha256": _sha(archive_disk),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source-archive",
        type=Path,
        default=Path(
            "/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/"
            "submissions/pfs1/archive.zip"
        ),
    )
    ap.add_argument(
        "--atlas-flat",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/ddm_ru1_20260729/atlas_flat.npz"),
    )
    ap.add_argument(
        "--out-dir", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_wr1_20260729")
    )
    ap.add_argument("--knee-a", type=int, default=486, help="safe-floor tranche (all zero-flip)")
    ap.add_argument("--knee-b", type=int, default=600, help="sub-bar byte-range tranche")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    codes, base, delta, frame = load_grid(args.source_archive)
    archive_floor = REF_ARCHIVE_BYTES - len(frame)  # 12,743 B non-token members
    sens = cell_sensitivity(delta, args.atlas_flat)
    checkpoints = [100, 200, 300, 400, args.knee_a, 540, args.knee_b, 660, 730, 768]
    checkpoints = sorted(set(checkpoints))
    rows = descent_curve(base, delta, sens, archive_floor, checkpoints)

    byte_closed = [
        byte_close(
            args.source_archive,
            base,
            delta,
            sens["order"][: args.knee_a],
            args.out_dir / "wr1_kneeA_safe_274k_archive.zip",
        ),
        byte_close(
            args.source_archive,
            base,
            delta,
            sens["order"][: args.knee_b],
            args.out_dir / "wr1_kneeB_subbar_173k_archive.zip",
        ),
    ]

    receipt = {
        "schema": "ddm_wr1_reverse_waterfill.v1",
        "task": "#766 / ledger QA06",
        "evidence_axis": (
            "[macOS-CPU advisory, rate-only] bytes MEASURED through the real SMEVR "
            "coder; d_seg/d_pose PREDICTED/TYPED pending the staged n600 realized gate"
        ),
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "source_archive": str(args.source_archive),
        "source_tokens_sha256": _sha(frame),
        "archive_floor_bytes": archive_floor,
        "kappa_med": KAPPA_MED,
        "n_zero_flip_cells": sens["n_zero_flip_cells"],
        "ordering": "flip_risk asc (safest), tie residual-mass desc (fattest-safe first)",
        "ref_row": {
            "S": REF_S,
            "archive_bytes": REF_ARCHIVE_BYTES,
            "tokens_bytes": REF_TOKENS_BYTES,
            "d_seg": REF_DSEG,
            "d_pose": REF_DPOSE,
            "seg_term": round(100 * REF_DSEG, 6),
            "pose_term": round((10 * REF_DPOSE) ** 0.5, 6),
            "rate_term": round(25 * REF_ARCHIVE_BYTES / UNCOMPRESSED, 6),
        },
        "solved_projection_note": (
            "S_composed_if_solved = seg_solved(0.0152, rp1 q1) + pose_solved(0.0153, "
            "PR130-grade d_pose 2.33e-5) + MEASURED rate; seg/pose solve are OTHER "
            "arms (fd1/tr1 + P3v2), NOT this rung. This rung supplies ONLY the rate term."
        ),
        "bar_thresholds": {
            "T1_bar": BAR,
            "T3_target": T3,
            "archive_budget_at_bar_if_solved": round((BAR - SEG_SOLVED_TERM - POSE_SOLVED_TERM) * UNCOMPRESSED / 25),
            "archive_budget_at_015_if_solved": round((T3 - SEG_SOLVED_TERM - POSE_SOLVED_TERM) * UNCOMPRESSED / 25),
        },
        "descent_rows": rows,
        "byte_closed": byte_closed,
    }
    (args.out_dir / "wr1_descent_receipt.json").write_text(json.dumps(receipt, indent=1))
    np.savez_compressed(
        args.out_dir / "wr1_cell_sensitivity_atlas.npz",
        order=sens["order"],
        flip_mass=sens["flip_mass"],
        residual_mass=sens["residual_mass"],
        rank=sens["rank"],
    )
    print(f"[wr1] receipts -> {args.out_dir}")
    for row in rows:
        print(
            f"  k={row['k_cells_dropped']:>3} arch={row['archive_bytes']:>7} "
            f"rate={row['rate_term']:.4f} S_ref={row['S_vs_ref_flipfree']:.4f} "
            f"S_solved={row['S_composed_if_solved']:.4f} "
            f"sub015={row['sub_015_if_solved']}"
        )


if __name__ == "__main__":
    main()
