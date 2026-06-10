#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""RD sweep for the amortized luma carrier — the #57 empirical crux (honest RD curve).

Sweeps the carrier capacity (hidden_dim / mod_dim / n_fourier / quant_bits) and, per operating
point, trains the carrier score-aware vs the EXACT PoseNet and measures (d_pose, carrier_bytes) on
the EXACT frozen scorer (numpy-decoded inflate-time frame, GT via yuv420_to_rgb). Produces the RD
curve {capacity → (d_pose, bytes, rate, pose_term)} so the waterfilling operating-point solve can
pick the S-minimising point.

This is the DECISIVE measurement: does an AMORTIZED learned carrier reach a d_pose at a byte budget
that lets the FULL S beat the frontier / hit sub-0.15? The pre-registered KILL criterion (design
memo) fires if no operating point does.

Authority ``[local CPU-torch advisory]``. NO MPS. $0. Non-promotable. Runs as a detached daemon
(durable SSD state) per CLAUDE.md "Durable detached daemons" — each operating point checkpoints to
disk so a SIGURG kill leaves the converged points behind.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "upstream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tac.boundary_math.amortized_luma_carrier import LumaCarrierConfig, carrier_param_count  # noqa: E402

_CONTEST_TOTAL_BYTES = 37_545_489
_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# capacity grid: (hidden_dim, mod_dim, n_fourier, n_hidden, fourier_sigma)
DEFAULT_GRID = [
    ("tiny", 48, 16, 16, 3, 6.0),
    ("small", 96, 32, 32, 3, 8.0),
    ("medium", 160, 48, 48, 4, 10.0),
    ("large", 256, 64, 64, 4, 12.0),
]


def run_sweep(
    targets_dir: Path,
    out_dir: Path,
    *,
    n_pairs: int,
    epochs: int,
    lr: float,
    grid: list[tuple],
    both_frames: bool,
    seed: int,
) -> dict[str, Any]:
    from tools.score_native_train_luma_carrier import train

    if any(str(out_dir).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"out_dir={out_dir!r} is /tmp-class; use SSD tier.")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for name, hidden, mod, nfour, nhid, sigma in grid:
        cfg = LumaCarrierConfig(num_pairs=n_pairs, n_fourier=nfour, hidden_dim=hidden,
                                n_hidden=nhid, mod_dim=mod, fourier_sigma=sigma, quant_bits=8)
        pt_dir = out_dir / f"pt_{name}"
        print(f"\n[sweep] {name} hidden={hidden} mod={mod} params={carrier_param_count(cfg)} "
              f"t={time.time()-t0:.0f}s", flush=True)
        res = train(targets_dir, pt_dir, cfg, n_pairs=n_pairs, epochs=epochs, lr=lr,
                    both_frames=both_frames, seed=seed, eval_every=max(10, epochs // 4))
        d_pose = res["exact_mean_d_pose"]
        bytes_ = res["byte_account"]["total_bytes"]
        rate = 25.0 * bytes_ / _CONTEST_TOTAL_BYTES
        row = {
            "name": name, "hidden_dim": hidden, "mod_dim": mod, "n_fourier": nfour,
            "param_count": res["param_count"], "carrier_bytes": bytes_, "rate_term": rate,
            "exact_mean_d_pose": d_pose, "pose_term_sqrt10": float(np.sqrt(10.0 * d_pose)),
            "parity_pass": res["portability_parity"]["parity_pass"],
        }
        rows.append(row)
        print("[sweep-row] " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                           for k, v in row.items()}), flush=True)
        # checkpoint sweep state every point (durable).
        (out_dir / "rd_sweep.json").write_text(json.dumps({
            "subagent": "task57_pose_carrier", "utc": _utc(),
            "evidence_grade": "[local CPU-torch advisory]", "promotion_eligible": False,
            "score_claim": False, "n_pairs": n_pairs, "epochs": epochs,
            "mode": "both_frames" if both_frames else "frame0_only",
            "wall_s": round(time.time() - t0, 1), "rows": rows,
        }, indent=2))
    # monotonicity check + best operating point (carrier-only S contribution).
    by_bytes = sorted(rows, key=lambda r: r["carrier_bytes"])
    monotone = all(by_bytes[i]["exact_mean_d_pose"] >= by_bytes[i + 1]["exact_mean_d_pose"] - 1e-9
                   for i in range(len(by_bytes) - 1))
    result = {
        "subagent": "task57_pose_carrier", "utc": _utc(),
        "evidence_grade": "[local CPU-torch advisory]", "promotion_eligible": False,
        "score_claim": False, "n_pairs": n_pairs, "epochs": epochs,
        "mode": "both_frames" if both_frames else "frame0_only",
        "wall_s": round(time.time() - t0, 1), "rows": rows,
        "rd_monotone_decreasing": bool(monotone),
        "best_pose_term": min((r["pose_term_sqrt10"] for r in rows), default=None),
    }
    (out_dir / "rd_sweep.json").write_text(json.dumps(result, indent=2))
    print("\n=== RD SWEEP COMPLETE ===")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    base = "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610"
    ap.add_argument("--targets-dir", type=Path, default=Path(base) / "targets_n600")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--both-frames", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    run_sweep(args.targets_dir, args.out_dir, n_pairs=args.n_pairs, epochs=args.epochs,
              lr=args.lr, grid=DEFAULT_GRID, both_frames=args.both_frames, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
