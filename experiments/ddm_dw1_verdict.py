"""ddm_dw1 — QA75 distill-window VERDICT analysis (guards 4/5/6/8 + falsifier).

Consumes the matched A/B/C window telemetry + receipts and produces the fork verdict:

  * gate d_seg trajectory + slope d(gate_dseg)/d(ep) per window (EMA-shadow gates; matched
    shadow dynamics => the A-vs-B slope DIFFERENCE is the real distill effect, shadow-warming
    is common-mode and cancels).
  * NOISE FLOOR (guard 6): B's gate-to-gate residual std about its own linear trend — the split
    below which A-vs-B is NOT a verdict.  Stated BEFORE reading the split.
  * UNDER-DRIVE (guard 4): |B slope| ~ 0 (both flat) => PROVISIONAL-UNDERDRIVEN; the named next
    step is an LR-rewarmup rerun (#518 beta2-derived laws).  ax1 measured QA24 schedule-truncated
    with COUPLED_DESCENT so a live B slope is expected; a dead B is an apparatus signal.
  * JOINT seg+rate (guard 5): endpoint realized n600 d_seg (full_confirm) AND total_counted_bytes
    (SMEVR) per window — distill could buy d_seg by spending token entropy; reported separately.
  * s/ep (guard 8): wall per epoch per window (matched-cost check; if not matched the budget axis
    is wall-clock not epochs).
  * FALSIFIER (preregistered QA75): distilled endpoint NOT clearly better than CE (split <= noise)
    => amortization gap NOT distillation-curable (QA24 form fixes lead); distill clearly better
    => burn-3 distill-opening GO.

score_claim=false; advisory [macOS-CPU]; pointer 0.1910828242 [contest-CPU] UNMOVED.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT_ROOT = "/Volumes/VertigoDataTier/pact/ddm_dw1_20260730"


def _load(run_dir: Path) -> dict:
    tel = run_dir / "telemetry.jsonl"
    rcpt = run_dir / "tr1_window_receipt.json"
    rows = [json.loads(ln) for ln in tel.read_text().splitlines() if ln.strip()] \
        if tel.is_file() else []
    receipt = json.loads(rcpt.read_text()) if rcpt.is_file() else {}
    # A window that stopped on the pre-registered A1 refuse has no in-run full_confirm; the
    # zero-step resume confirm (SAME trainer EMA-shadow surface) lands in a sibling
    # ``<name>_endpoint_confirm`` dir — consume its full_confirm as the endpoint.
    if not (receipt.get("full_confirm") or {}).get("realized_dseg_mean"):
        sib = run_dir.parent / f"{run_dir.name}_endpoint_confirm" / "tr1_window_receipt.json"
        if sib.is_file():
            fc_sib = (json.loads(sib.read_text()).get("full_confirm") or {})
            if fc_sib.get("realized_dseg_mean") is not None:
                receipt = {**receipt, "full_confirm": fc_sib}
    gates = [r for r in rows if r.get("event") == "a1_gate"]
    eps = [r for r in rows if r.get("event") == "epoch"]
    reanchor = [r for r in rows if r.get("event") == "resume_form_reanchor"]
    # EMA-shadow gates only (the basis the endpoint confirm uses); ep + d_seg + bytes.
    sg = [(g["epoch"], g["realized_gate_dseg_mean"], g.get("total_counted_bytes"))
          for g in gates if g.get("gate_params") == "ema_shadow"]
    fc = receipt.get("full_confirm") or {}
    # s/ep from epoch-row t_wall deltas.
    twalls = [r.get("t_wall") for r in eps if r.get("t_wall") is not None]
    s_per_ep = float(np.median(np.diff(twalls))) if len(twalls) >= 2 else None
    return {
        "n_epochs": len(eps),
        "resume_form_reanchor_fired": bool(reanchor),
        "shadow_gates": sg,
        "endpoint_realized_dseg_n600": fc.get("realized_dseg_mean"),
        "endpoint_dseg_max": fc.get("realized_dseg_max"),
        "endpoint_bytes": (gates[-1].get("total_counted_bytes") if gates else None),
        "endpoint_bytes_smevr": (gates[-1].get("tokens_bytes_smevr") if gates else None),
        "final_gate_dseg": (gates[-1]["realized_gate_dseg_mean"] if gates else None),
        "s_per_epoch": s_per_ep,
        "stop_reason": receipt.get("stop_reason"),
        "a1_refused": any(r.get("event") == "a1_stage_exit_refuse" for r in rows),
    }


def _slope_and_resid(sg: list) -> tuple[float | None, float | None]:
    """Linear fit gate_dseg vs epoch over the shadow gates; return (slope, residual std)."""
    if len(sg) < 3:
        return None, None
    x = np.array([e for e, _, _ in sg], dtype=float)
    y = np.array([d for _, d, _ in sg], dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    (m, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - (m * x + b)
    return float(m), float(np.std(resid))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--rate-ref-bytes", type=float, default=37_545_489.0)
    args = ap.parse_args()
    root = Path(args.out_root)

    W = {name: _load(root / name) for name in ("control", "distill", "distill_head_relax")}
    B, A, C = W["control"], W["distill"], W["distill_head_relax"]

    b_slope, b_resid = _slope_and_resid(B["shadow_gates"])
    a_slope, _ = _slope_and_resid(A["shadow_gates"])
    c_slope, _ = _slope_and_resid(C["shadow_gates"])
    # Noise floor (guard 6): B's residual std about its trend (fallback: gate-to-gate std).
    noise_floor = b_resid
    if noise_floor is None and B["shadow_gates"]:
        ys = np.array([d for _, d, _ in B["shadow_gates"]])
        noise_floor = float(np.std(np.diff(ys))) if len(ys) >= 2 else None

    def _S(dseg, bytes_):  # advisory composed seg+rate (pose common-mode / not in this arm)
        if dseg is None or bytes_ is None:
            return None
        return 100.0 * float(dseg) + 25.0 * float(bytes_) / args.rate_ref_bytes

    ep_b, ep_a, ep_c = (B["endpoint_realized_dseg_n600"], A["endpoint_realized_dseg_n600"],
                        C["endpoint_realized_dseg_n600"])
    split_ab = (ep_b - ep_a) if (ep_b is not None and ep_a is not None) else None  # >0 => A better
    split_ca = (ep_a - ep_c) if (ep_a is not None and ep_c is not None) else None  # >0 => C better

    # Under-drive (guard 4), DIMENSIONALLY CONSISTENT: compare B's TOTAL window descent
    # |slope x gate-span| (a d_seg quantity) against the d_seg noise floor — a per-epoch
    # slope vs a level-noise floor is a unit conflation (caught in self-review; the first
    # emitted verdict used the conflated test and mis-fired PROVISIONAL-UNDERDRIVEN).
    b_span = (B["shadow_gates"][-1][0] - B["shadow_gates"][0][0]) if len(B["shadow_gates"]) >= 2 else 0
    b_total_descent = abs(b_slope * b_span) if (b_slope is not None and b_span) else None
    under_driven = (b_total_descent is not None and noise_floor is not None
                    and b_total_descent < (noise_floor if noise_floor > 0 else 1e-9))

    # Falsifier (endpoint d_seg split vs noise floor; slope corroborates).
    if ep_a is None or ep_b is None:
        verdict = "INCOMPLETE (missing endpoint confirm)"
    elif under_driven:
        verdict = ("PROVISIONAL-UNDERDRIVEN: B slope below noise floor at the resumed LR — "
                   "next step = LR-rewarmup rerun (#518 beta2-derived); do NOT read the split as "
                   "a fork verdict")
    elif noise_floor is not None and split_ab is not None and split_ab > 2.0 * noise_floor:
        verdict = ("DISTILL CLEARLY BETTER: amortization gap is (partly) DISTILLATION-CURABLE => "
                   "burn-3 distill-opening GO (QA75 leads)")
    elif noise_floor is not None and split_ab is not None and split_ab < -2.0 * noise_floor:
        verdict = ("DISTILL WORSE than CE: falsifier FIRES — gap NOT distillation-curable "
                   "(optimization/capacity leads; QA24 form fixes lead; class-change leg strengthens)")
    else:
        verdict = ("NO SPLIT (|A-B| <= 2x noise floor): falsifier FIRES — amortization gap NOT "
                   "distillation-curable at matched budget (QA24 form fixes lead)")

    report = {
        "schema": "ddm_dw1_verdict.v1",
        "score_claim": False, "authority": "advisory [macOS-CPU]",
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "noise_floor_gate_dseg": noise_floor,
        "resume_form_reanchor": {k: W[k]["resume_form_reanchor_fired"] for k in W},
        "windows": {
            "B_control": {"endpoint_dseg_n600": ep_b, "slope_per_ep": b_slope,
                          "endpoint_bytes": B["endpoint_bytes"], "s_per_ep": B["s_per_epoch"],
                          "n_ep": B["n_epochs"], "stop": B["stop_reason"], "S": _S(ep_b, B["endpoint_bytes"])},
            "A_distill": {"endpoint_dseg_n600": ep_a, "slope_per_ep": a_slope,
                          "endpoint_bytes": A["endpoint_bytes"], "s_per_ep": A["s_per_epoch"],
                          "n_ep": A["n_epochs"], "stop": A["stop_reason"], "S": _S(ep_a, A["endpoint_bytes"])},
            "C_head_relax": {"endpoint_dseg_n600": ep_c, "slope_per_ep": c_slope,
                             "endpoint_bytes": C["endpoint_bytes"], "s_per_ep": C["s_per_epoch"],
                             "n_ep": C["n_epochs"], "stop": C["stop_reason"],
                             "advisory_non_deployable": True, "S": _S(ep_c, C["endpoint_bytes"])},
        },
        "slope_ratio_A_over_B": (a_slope / b_slope if (a_slope is not None and b_slope not in (None, 0.0)) else None),
        "slope_ratio_C_over_A": (c_slope / a_slope if (c_slope is not None and a_slope not in (None, 0.0)) else None),
        "endpoint_split_B_minus_A_pos_means_A_better": split_ab,
        "endpoint_split_A_minus_C_pos_means_C_better": split_ca,
        "under_driven": under_driven,
        "b_total_window_descent_gate_dseg": b_total_descent,
        "verdict": verdict,
        "verdict_scope": "INSTANCE (this endpoint/scorer/window)" if "NO SPLIT" in verdict
                         or "WORSE" in verdict else "n/a",
    }
    (root / "verdict.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
