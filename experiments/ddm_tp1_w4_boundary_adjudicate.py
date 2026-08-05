#!/usr/bin/env python3
"""ddm_tp1 w4/w4m ep1363 boundary adjudication — the #925 margin A/B at window scale.

Reads BOTH arms' telemetry (a1 gate rows, `realized_gate_dseg_mean`, EMA shadow,
n=36 gate pairs), runs the canonical `adjudicate_tail_slope` on each POST-RESUME
tail, and emits ONE typed receipt with: per-arm tail verdicts, the matched-epoch
A/B table, the margin-weighted-loss window verdict, and the pre-registered
routing (jd1 #366 ticket 6564914a regenerates vs the WINNER; extension is the
alternative if the winner's tail still dominates the pose axis's expected rate).

Gate d_seg here is the n36 EMA-shadow ADVISORY gate metric — NOT an n600 verdict
and NOT a score claim. Axis: [macOS-MLX research-signal], score_claim=false.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.optimization.trajectory_stopping import adjudicate_tail_slope  # noqa: E402

BASE = Path("/Volumes/VertigoDataTier/pact/ddm_tp1_20260805")
ARMS = {"w4_margin_off": "full_birth_lane_on_w4", "w4m_margin_on": "full_birth_lane_on_w4m"}
RESUME_EPOCH = 1223  # w3 endpoint both arms resumed from (chain min 0.0038515726725260415)
EPOCH_MINUTES = 1.26  # measured cadence this window (ep1250->1288 in 48 min)
OUT = BASE / "w4_boundary_adjudication.json"


def arm_tail(run_dir: Path) -> list[tuple[float, float]]:
    rows = []
    with open(run_dir / "telemetry.jsonl") as fh:
        for line in fh:
            if "realized_gate_dseg_mean" not in line:
                continue
            r = json.loads(line)
            ep = r.get("epoch")
            v = r.get("realized_gate_dseg_mean")
            if ep is not None and v is not None and ep > RESUME_EPOCH:
                rows.append((float(ep), float(v)))
    rows.sort()
    return rows


def main() -> int:
    receipt: dict = {
        "schema": "tp1_w4_boundary_adjudication.v1",
        "adjudicator": "tac.optimization.trajectory_stopping.adjudicate_tail_slope",
        "axis": "[macOS-MLX research-signal] n36 EMA-shadow gate metric, NON-PROMOTABLE",
        "score_claim": False,
        "resume_epoch": RESUME_EPOCH,
        "parent_endpoint_dseg": 0.0038515726725260415,
        "arms": {},
    }
    endpoints: dict[str, float] = {}
    for name, sub in ARMS.items():
        tail = arm_tail(BASE / sub)
        steps = [t[0] for t in tail]
        vals = [t[1] for t in tail]
        verdict = adjudicate_tail_slope(steps, vals)
        fits = getattr(verdict, "fits", None)
        # S-units conversion: d(S)/dt = 100 * d(d_seg)/d(epoch) / (EPOCH_MINUTES/60)
        slope_s_per_hour = None
        try:
            last_fit = fits[-1] if fits else None
            if last_fit is not None:
                sl = last_fit["slope"] if isinstance(last_fit, dict) else last_fit.slope
                slope_s_per_hour = 100.0 * float(sl) * (60.0 / EPOCH_MINUTES)
        except Exception:
            pass
        endpoints[name] = vals[-1] if vals else float("nan")
        receipt["arms"][name] = {
            "n_gate_rows": len(tail),
            "endpoint": {"epoch": steps[-1] if steps else None, "dseg": vals[-1] if vals else None},
            "min": {"epoch": steps[vals.index(min(vals))] if vals else None,
                    "dseg": min(vals) if vals else None},
            "verdict": str(getattr(verdict, "verdict", verdict)),
            "fits": repr(fits),
            "advisory_slope_S_per_hour": slope_s_per_hour,
            "tail_last6": tail[-6:],
        }
    off, on = endpoints.get("w4_margin_off"), endpoints.get("w4m_margin_on")
    if off is not None and on is not None:
        winner = "w4m_margin_on" if on < off else "w4_margin_off"
        receipt["margin_ab"] = {
            "endpoint_off": off, "endpoint_on": on,
            "delta_on_minus_off": on - off,
            "winner_by_endpoint": winner,
            "single_variable": "margin_weighted_loss (en1 #925), identical resume/epochs",
        }
        receipt["routing_pre_registered"] = (
            f"jd1 #366 ticket 6564914a regenerates vs {winner} endpoint checkpoint; "
            "extension-of-winner is the alternative iff its advisory seg slope (S/hour) "
            "dominates the pose axis's expected opening; MHAR #956 probe + jb1 full "
            "derived-basis race also fire vs the SAME winner checkpoint."
        )
    OUT.write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
