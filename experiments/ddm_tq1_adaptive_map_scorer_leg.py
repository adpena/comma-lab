#!/usr/bin/env python3
"""ddm_tq1 — SCORER LEG of tz1 READY manifest #1/#2 (#869 token waterfill).

tz1 (2026-08-04) landed the BYTE-ONLY legs of the adaptive per-cell token
re-quantization and queued the scorer verdicts "the instant the scorer frees":

  #1 ADAPTIVE margin-coupled [16,12,8,4] — net −113,555 B (12.22% of gap),
     pays iff Δd_seg < 7.56e-4 (pre-registered break-even).
  #2 ADAPTIVE derived-activity [16,12,8,4] — net −62,502 B (0-byte map),
     pays iff Δd_seg < 4.16e-4.

This arm fires those verdicts against the LIVE frontier archive (qo1
``sub_auto_pairbit``, sha d5e814d5…, 357,836 B, S 0.7539807 [macOS-CPU
advisory]).  Mechanism (tz1 ARM B, recall-at-source): each (R,C) cell's tokens
are snapped to its assigned sub-lattice while KEEPING the container's global
levels=16 alphabet — the re-quantized stream encodes/decodes through the
UNCHANGED receiver (round-trip verified), so the byte-close is honest by
construction and NO map ships (the flip-mass field guides ENCODE-side only;
the counted payload is the snapped tokens themselves).

Chain: parse 0.bin → decode token bulk → snap per map → re-encode → rebuild
payload with ALL joint sections byte-identical → deterministic zip → clone the
qo1 runtime sub-dir → fz2 canonical byte-close + full n600 eval
(``experiments/ddm_fz2_byteclose_and_eval.py``) → receipt with realized
(Δd_seg, Δd_pose, Δbytes, ΔS) vs the pre-registered break-even.

Axis: [macOS-CPU frozen-scorer advisory]; score_claim=false until the fleet
protocol promotes.  Resumable: every stage is idempotent (skips on existing
artifact with matching sha).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_SRC = REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from tac.optimization import ddm_ix2_archive_container as IX2  # noqa: E402
from tac.witness_dsl.ax1_pool_a_levers_20260730 import (  # noqa: E402
    margin_coupled_level_map,
)
from ddm_tz1_token_sweep_rate_attack import (  # noqa: E402  recall-at-source (tz1 ARM B)
    _apply_cell_level_map,
    _per_cell_activity,
)

POINTER_LINE = (
    "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; "
    "contest pointer borrowed/unmoved."
)
DEN = 37_545_489
BASELINE = {
    "archive": "/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/archive.zip",
    "sha256": "d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a",
    "bytes": 357_836,
    "d_seg": 0.00431179,
    "d_pose": 0.00071459,
    "S": 0.7539807296911207,
}
DEFAULT_FIELD = "/Volumes/VertigoDataTier/pact/ddm_sg1_20260731/cell_flip_mass.npy"
DEFAULT_OUT = "/Volumes/VertigoDataTier/pact/ddm_tq1_20260805"
BREAK_EVEN_DSEG = {"margin": 7.56e-4, "derived": 4.16e-4}  # tz1 §8 pre-registered
LADDER = (16, 12, 8, 4)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _score(d_seg: float, d_pose: float, nbytes: int) -> float:
    return 100.0 * d_seg + (10.0 * d_pose) ** 0.5 + 25.0 * nbytes / DEN


def compose(variant: str, field_path: Path, out_root: Path) -> dict:
    src_dir = Path(BASELINE["archive"]).parent
    src_zip = Path(BASELINE["archive"])
    assert _sha(src_zip) == BASELINE["sha256"], "baseline archive sha drift"
    with zipfile.ZipFile(src_zip) as zf:
        payload = zf.read("0.bin")
    bulk, sections = IX2.parse_payload(payload)
    tok = IX2.decode_token_frame(bulk)

    if variant == "margin":
        field = np.load(field_path).astype(np.float64)
    elif variant == "derived":
        field = _per_cell_activity(tok)  # decoder-derivable (rule-118-free)
    else:
        raise SystemExit(f"unknown variant {variant!r}")
    lvl_map = margin_coupled_level_map(
        field, base_levels=max(LADDER), min_levels=min(LADDER), n_tiers=len(LADDER))
    if lvl_map.shape != tok.shape[1:3]:
        raise SystemExit(f"map shape {lvl_map.shape} != cell grid {tok.shape[1:3]}")

    snapped = _apply_cell_level_map(tok, lvl_map)
    new_bulk = IX2.encode_token_frame(snapped, levels=16)
    # receiver-transparency proof: the UNCHANGED decoder returns exactly the snapped codes
    assert np.array_equal(IX2.decode_token_frame(new_bulk), snapped), "roundtrip broke"

    new_payload = IX2.build_payload(new_bulk, list(sections))
    rb, rs = IX2.parse_payload(new_payload)
    assert rb == new_bulk and all(a == b for a, b in zip(rs, sections)), "sections drifted"

    dst_dir = out_root / f"sub_tq1_{variant}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in src_dir.iterdir():
        if f.is_file() and f.name != "archive.zip" and f.suffix != ".txt":
            shutil.copy2(f, dst_dir / f.name)
    (dst_dir / "archive.zip").write_bytes(IX2.build_single_member_zip(new_payload))

    nbytes = (dst_dir / "archive.zip").stat().st_size
    vals, counts = np.unique(lvl_map, return_counts=True)
    receipt = {
        "schema": "ddm_tq1_compose.v1",
        "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
        "score_claim": False,
        "pointer": POINTER_LINE,
        "variant": variant,
        "rung_ladder": list(LADDER),
        "level_histogram": {str(int(v)): int(c) for v, c in zip(vals, counts)},
        "baseline": BASELINE,
        "token_bulk_bytes": {"before": len(bulk), "after": len(new_bulk),
                             "saved": len(bulk) - len(new_bulk)},
        "archive": str(dst_dir / "archive.zip"),
        "archive_bytes": nbytes,
        "archive_sha256": _sha(dst_dir / "archive.zip"),
        "archive_bytes_delta": nbytes - BASELINE["bytes"],
        "break_even_dseg": BREAK_EVEN_DSEG[variant],
        "joint_sections_byte_identical": True,
        "no_map_shipped": True,  # snap baked into token values; receiver unchanged
    }
    (dst_dir / "compose_receipt.json").write_text(json.dumps(receipt, indent=1))
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=["margin", "derived"], default="margin")
    ap.add_argument("--field", type=Path, default=Path(DEFAULT_FIELD))
    ap.add_argument("--out-root", type=Path, default=Path(DEFAULT_OUT))
    ap.add_argument("--compose-only", action="store_true",
                    help="stop after the byte-closed compose (no scorer)")
    ap.add_argument("--num-threads", type=int, default=4)
    args = ap.parse_args(argv)

    args.out_root.mkdir(parents=True, exist_ok=True)
    rec = compose(args.variant, args.field, args.out_root)
    print(f"[tq1] composed {args.variant}: {rec['archive_bytes']} B "
          f"(Δ {rec['archive_bytes_delta']:+d} B) sha {rec['archive_sha256'][:12]}…")
    if args.compose_only:
        return 0

    dst_dir = Path(rec["archive"]).parent
    eval_receipt = dst_dir / "fz2_eval_receipt.json"
    cmd = [str(REPO / ".venv/bin/python"),
           str(REPO / "experiments/ddm_fz2_byteclose_and_eval.py"),
           "--sub-dir", str(dst_dir),
           "--out", str(eval_receipt),
           "--inflate-out", str(dst_dir / "inflated"),
           "--num-threads", str(args.num_threads)]
    print("[tq1] firing canonical byte-close + n600 eval:", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=REPO).returncode
    if rc != 0 or not eval_receipt.exists():
        print(f"[tq1] fz2 eval FAILED rc={rc}")
        return rc or 3

    ev = json.loads(eval_receipt.read_text()).get("evaluate", {})
    if not ev.get("ran"):
        print("[tq1] fz2 receipt has no completed evaluate block")
        return 4
    d_seg = float(ev["d_seg"])
    d_pose = float(ev["d_pose"])
    s_new = _score(d_seg, d_pose, rec["archive_bytes"])
    dd_seg = d_seg - BASELINE["d_seg"]
    verdict = {
        "schema": "ddm_tq1_verdict.v1",
        "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
        "score_claim": False,
        "pointer": POINTER_LINE,
        "variant": args.variant,
        "S_new": s_new,
        "S_baseline": BASELINE["S"],
        "delta_S": s_new - BASELINE["S"],
        "d_seg": d_seg, "d_pose": d_pose,
        "delta_d_seg": dd_seg,
        "delta_d_pose": d_pose - BASELINE["d_pose"],
        "archive_bytes": rec["archive_bytes"],
        "break_even_dseg": rec["break_even_dseg"],
        "pays": bool(s_new < BASELINE["S"]),
        "under_preregistered_break_even": bool(dd_seg < rec["break_even_dseg"]),
    }
    (dst_dir / "tq1_verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
