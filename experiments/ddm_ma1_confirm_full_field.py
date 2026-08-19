# SPDX-License-Identifier: MIT
"""ddm_ma1 - full n600 confirmation of the within-miss law, with its nesting control.

The sector race (``ddm_ma1_race_within_miss.py``) is exact for the sector by the
separability theorem, but a theorem is not a measurement of the thing that ships.
This runs the REAL corrector over all 117,964,800 positions through the shipped
decode order and reports:

* **C1** ``ddm_fx2`` D1, the live law -- reproduces the inherited code length.
* **C2** this module with ``within_miss=False`` -- must be **bit-identical** to
  C1, which is what makes every delta below the new sector and never the
  plumbing.
* **the candidate** -- and its delta must equal the sector race's delta, because
  the hit-event term cannot move.  A disagreement falsifies the separability
  argument and voids the race.

Wall-clock is measured SERIALLY here (nothing else running), because on this
lineage the decode budget, not the byte count, is what picks the candidate:
``ddm_fx1`` measured the real parse-back at 1,639.78 s against 1,800 s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tac.micro_edit.coder_replay import ReplayAssets, replay_code_length  # noqa: E402

ASSETS = ReplayAssets(
    logits_i16=Path(
        "/Volumes/APDataStore/pact/ddm_hm1_20260816/retained/base_logits_int16_n600.i16"
    ),
    tokens_u8=Path(
        "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2/inflated/"
        ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
    ),
    boundary_u8=Path(
        "/Volumes/APDataStore/pact/ddm_hm1_20260816/retained/boundary_bucket_n600.u8"
    ),
    group_index_u8=Path("/Volumes/APDataStore/pact/ddm_hm1_20260816/retained/group_index.u8"),
    table_values_npy=Path("/Volumes/APDataStore/pact/ddm_me1/table_values.npy"),
)

LIVE_RR4_BITS = 884090.2210952122
"""The shipped rr4 law, reproduced by ddm_fx1 and ddm_fx2 at delta 0.000000."""

FX2_D1_DELTA_BYTES = -710.84
"""ddm_fx2's frozen D1 build, which the LIVE ck1/ck2 body carries."""

RETAIN = Path("/Volumes/APDataStore/pact/ddm_ma1/retained")


def _run(label: str, factory, frames: int) -> dict:
    started = time.time()
    result = replay_code_length(ASSETS, label=label, corrector_factory=factory, frames=frames)
    elapsed = time.time() - started
    RETAIN.mkdir(parents=True, exist_ok=True)
    payload = RETAIN / f"bits_{label}_n{frames}.npy"
    np.save(payload, result.bits_per_frame)  # ALWAYS KEEP THE PAYLOAD
    sha = hashlib.sha256(payload.read_bytes()).hexdigest()
    row = {
        "label": label,
        "frames": frames,
        "code_bits": result.code_bits,
        "code_bytes": result.code_bytes,
        "delta_bytes_vs_live_rr4": result.code_bytes - LIVE_RR4_BITS / 8.0,
        "elapsed_s": elapsed,
        "payload": str(payload),
        "payload_sha256": sha,
    }
    print(
        f"{label:34s} {result.code_bytes:12.4f} B   vs_rr4 {row['delta_bytes_vs_live_rr4']:+9.3f}"
        f"   [{elapsed:7.1f}s]  {sha[:16]}"
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--miss-cell", default="nb3_prev1")
    ap.add_argument("--out", default="/Volumes/APDataStore/pact/ddm_ma1/race/confirm_full_field.json")
    ap.add_argument("--skip-controls", action="store_true")
    args = ap.parse_args()

    from experiments.ddm_fx2_model_axis_corrector import (
        SHIPPED_CONFIG as FX2_CONFIG,
        Fx2ModelAxisMixer,
    )
    from experiments.ddm_ma1_within_miss_corrector import Ma1WithinMissCorrector

    rows: list[dict] = []

    if not args.skip_controls:
        rows.append(
            _run("C1_fx2_D1_live_law", lambda p: Fx2ModelAxisMixer(p, **FX2_CONFIG), args.frames)
        )
        rows.append(
            _run(
                "C2_ma1_within_miss_OFF",
                lambda p: Ma1WithinMissCorrector(p, within_miss=False, **FX2_CONFIG),
                args.frames,
            )
        )

    rows.append(
        _run(
            f"ma1_{args.miss_cell}",
            lambda p: Ma1WithinMissCorrector(
                p, within_miss=True, miss_cell=args.miss_cell, **FX2_CONFIG
            ),
            args.frames,
        )
    )

    out = {"frames": args.frames, "rows": rows}
    if not args.skip_controls:
        c1, c2 = rows[0], rows[1]
        nesting = c2["code_bits"] - c1["code_bits"]
        out["nesting_control_delta_bits"] = nesting
        out["nesting_control_payload_identical"] = (
            c1["payload_sha256"] == c2["payload_sha256"]
        )
        out["c1_vs_fx2_published_delta_bytes"] = (
            c1["delta_bytes_vs_live_rr4"] - FX2_D1_DELTA_BYTES
        )
        cand = rows[-1]
        out["candidate_delta_bytes_vs_fx2_D1"] = cand["code_bytes"] - c1["code_bytes"]
        print()
        print(f"C1 vs ddm_fx2's published D1 (-710.84 B): "
              f"{out['c1_vs_fx2_published_delta_bytes']:+.4f} B")
        print(f"NESTING CONTROL  C2 - C1 = {nesting:+.6f} bits   "
              f"payload identical: {out['nesting_control_payload_identical']}")
        print(f"CANDIDATE vs the live D1 law: "
              f"{out['candidate_delta_bytes_vs_fx2_D1']:+.4f} B")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
