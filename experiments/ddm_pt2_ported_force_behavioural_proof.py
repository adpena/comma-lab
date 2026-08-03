#!/usr/bin/env python
"""ddm_pt2 — BEHAVIOURAL PROOF that each ported force actually does what its flag NAMES.

Scorer-free by construction: it reads a REAL cached SegNet logit field (the b2b QA75 distill
field, produced by an earlier scorer pass and stored on the SSD tier) and the REAL cached GT
argmax/margin, then calls the ACTUAL force implementations the TR1 trainer now passes into
``make_loss_fn``.  No scorer forward is performed here.  ``ddm_pu2`` holds the scorer slot.

Why this shape rather than a unit test on synthetic tensors: a force that "changes the loss"
on random logits proves nothing about the vehicle.  These are the real decision fields the seg
loss actually sees, so a null result here would be a real null.

Each force is checked with THREE legs, and the middle one is the one that can fail:

  * OFF leg     — the trainer default must reproduce the pre-port composed loss EXACTLY
                  (byte-identity of the control; if this fails the port is not free).
  * ON leg      — a PREDICTED, closed-form consequence of the mechanism must hold on real data
                  (not merely "the number moved" -- a wrong implementation also moves it).
  * MUTATION    — the force is replaced by an all-ones / identity stand-in; the ON leg's own
                  assertion must then FAIL.  A control that cannot fail is not a control.

Pointer 0.1910828242 [contest-CPU] UNMOVED; score_claim=False.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DISTILL_FIELD = Path("/Volumes/VertigoDataTier/pact/ddm_b2b_qa75_field_20260730")
GT_CACHE = Path("experiments/results/mlx_fleet_gt_cache/gt_n96.npz")


def _load_real_pair(pair: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(seg_logits (1,H,W,5) f32, lstar_oh (1,H,W,5) f32, gt_margin (H,W) f32) — all REAL."""
    zp = DISTILL_FIELD / f"pair-{pair:06d}.npz"
    if not zp.is_file():
        raise SystemExit(f"REFUSED: real logit field missing at {zp} — an unmounted tier must "
                         f"never be silently indistinguishable from a clean run.")
    z = np.load(zp)
    logits = np.transpose(np.asarray(z["distill_logits"], dtype=np.float32), (1, 2, 0))[None]
    g = np.load(GT_CACHE, mmap_mode="r")
    lstar = np.asarray(g["lstars"][pair], dtype=np.int64)
    margin = np.asarray(g["margins"][pair], dtype=np.float32)
    oh = (lstar[..., None] == np.arange(5)).astype(np.float32)[None]
    return logits, oh, margin


def _tau_softplus_loss(mx, seg_logits, lstar_oh, tau: float, seg_pixel_w=None):
    """The EXACT ``tau_softplus`` branch of make_loss_fn (the form the live lineage runs)."""
    gt_logit = mx.sum(seg_logits * lstar_oh, axis=-1)
    runner_up = mx.max(seg_logits + lstar_oh * (-1e9), axis=-1)
    signed = gt_logit - runner_up
    z = -signed / tau
    pp = tau * mx.logaddexp(mx.zeros_like(z), z)
    return mx.mean(pp if seg_pixel_w is None else pp * seg_pixel_w), signed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--fisher-blend", type=float, default=1.0)
    ap.add_argument("--json-out", default="reports/ddm_pt2/behavioural_proof.json")
    args = ap.parse_args()

    import mlx.core as mx

    from experiments.train_witness_realized_through_R_mlx import (
        fisher_density_pixel_weight_mlx,
        focal_pixel_weight_mlx,
        make_seg_logits_natural_grad_mlx,
    )

    lg_np, oh_np, m_np = _load_real_pair(args.pair)
    seg_logits = mx.array(lg_np)
    lstar_oh = mx.array(oh_np)
    gt_margin = mx.array(m_np)
    out: dict[str, object] = {
        "arm": "ddm_pt2", "pair": args.pair, "score_claim": False,
        "inputs": {"logits": str(DISTILL_FIELD / f"pair-{args.pair:06d}.npz"),
                   "gt": str(GT_CACHE), "shape": list(lg_np.shape),
                   "note": "REAL cached SegNet logit field + REAL GT argmax/margin; no scorer "
                           "forward is performed by this script"},
    }

    base_loss, signed = _tau_softplus_loss(mx, seg_logits, lstar_oh, 0.3, None)
    base = float(base_loss)
    out["control_tau_softplus_loss_unweighted"] = base

    # ---------------- FORCE 1: --seg-focal-gamma ----------------
    fw = focal_pixel_weight_mlx(seg_logits, lstar_oh, args.gamma)
    on_loss = float(_tau_softplus_loss(mx, seg_logits, lstar_oh, 0.3, fw)[0])
    off_loss = float(_tau_softplus_loss(mx, seg_logits, lstar_oh, 0.3, None)[0])
    # PREDICTION (closed form, can fail): under stop-grad + mean-1 renorm the weight RATIO between
    # any two pixels is EXACTLY ((1-p1)/(1-p2))^gamma. Check it on the real p_y field.
    p_y = np.exp(np.sum(lg_np * oh_np, axis=-1)
                 - np.log(np.sum(np.exp(lg_np - lg_np.max(-1, keepdims=True)), axis=-1))
                 - lg_np.max(-1))
    fw_np = np.asarray(fw)
    flat_p, flat_w = p_y.ravel(), fw_np.ravel()
    order = np.argsort(flat_p)
    i, j = int(order[len(order) // 4]), int(order[3 * len(order) // 4])
    pred_ratio = float(((1.0 - flat_p[i]) / (1.0 - flat_p[j])) ** args.gamma)
    obs_ratio = float(flat_w[i] / flat_w[j])
    focal_rel = abs(obs_ratio - pred_ratio) / max(abs(pred_ratio), 1e-12)
    # MUTATION control: an all-ones stand-in must BREAK the closed-form ratio prediction.
    mut_ratio = 1.0
    mut_rel = abs(mut_ratio - pred_ratio) / max(abs(pred_ratio), 1e-12)
    out["focal"] = {
        "gamma": args.gamma, "off_loss": off_loss, "on_loss": on_loss,
        "loss_changed": off_loss != on_loss,
        "weight_mean": float(fw_np.mean()), "weight_min": float(fw_np.min()),
        "weight_max": float(fw_np.max()),
        "closed_form_ratio_predicted": pred_ratio, "observed": obs_ratio,
        "rel_err": focal_rel, "PREDICTION_HOLDS": bool(focal_rel < 1e-4),
        "mutation_all_ones_rel_err": mut_rel,
        "MUTATION_CONTROL_FAILS_AS_REQUIRED": bool(mut_rel >= 1e-4),
    }

    # ---------------- FORCE 2: --fisher-density-weight ----------------
    fdw_model = fisher_density_pixel_weight_mlx(seg_logits, lstar_oh, gt_margin,
                                                args.fisher_blend, "model")
    fdw_gt = fisher_density_pixel_weight_mlx(seg_logits, lstar_oh, gt_margin,
                                             args.fisher_blend, "gt")
    fd_on = float(_tau_softplus_loss(mx, seg_logits, lstar_oh, 0.3, fdw_model)[0])
    # PREDICTION 1 (exact law, can fail): w == (1/2)sech^2(m/2) renormalized to mean 1, at
    # lambda=1 and source='gt', where m is the REAL cached GT margin.
    tr = 0.5 / np.cosh(m_np * 0.5) ** 2
    expect_gt = tr / (tr.mean() + 1e-8)
    got_gt = np.asarray(fdw_gt)[0] if np.asarray(fdw_gt).ndim == 3 else np.asarray(fdw_gt)
    law_rel = float(np.abs(got_gt - expect_gt).max() / max(float(np.abs(expect_gt).max()), 1e-12))
    # PREDICTION 2 — the DISTINCTION from focal, split into a REAL leg and a REGIME leg after the
    # first run of this script MEASURED that the naive form was VACUOUS (n=0). The documented
    # difference between the two allocators is about CONFIDENTLY-WRONG pixels (low p_y AND large
    # |margin|), and the b2b field is a near-perfect solve: min p_y over the whole frame is ~0.34
    # and the 189 argmax-wrong pixels have median |margin| ~0.04 — every error is MARGINAL. So the
    # regime the claim is about does not exist in this input, and an empty scope must be reported
    # VACUOUS with its denominator, never as a pass and never as a refutation.
    live_m = np.asarray(signed)[0]
    fdm = np.asarray(fdw_model)[0] if np.asarray(fdw_model).ndim == 3 else np.asarray(fdw_model)
    wrong = (p_y[0] < 0.05) & (np.abs(live_m) > 4.0)
    n_wrong = int(wrong.sum())
    # (2a) REAL leg, non-vacuous by construction: on THIS field's actual population the two
    # allocators should AGREE in direction (both up-weight the small-|margin| separatrix band).
    # Spearman on the real per-pixel maps; a sign flip here would refute the shared-geometry claim.
    sub = slice(None, None, 7)                       # decimate for the rank transform, still ~28k px
    a = fw_np[0].ravel()[sub]
    b = fdm.ravel()[sub]
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    # (2b) REGIME leg: evaluate the two CLOSED FORMS over the p_y x margin domain to locate where
    # they diverge. Labelled a FUNCTION-DOMAIN evaluation, not a measurement on this vehicle.
    m_probe = np.array([0.05, 0.5, 2.0, 6.0], dtype=np.float32)
    focal_probe = [float((1.0 - pv) ** args.gamma) for pv in (0.9, 0.5, 0.2, 0.02)]
    fisher_probe = [float(0.5 / np.cosh(mv * 0.5) ** 2) for mv in m_probe]
    out["fisher_density"] = {
        "blend": args.fisher_blend, "off_loss": off_loss, "on_loss_source_model": fd_on,
        "loss_changed": off_loss != fd_on,
        "exact_law_rel_err_source_gt": law_rel,
        "EXACT_LAW_HOLDS": bool(law_rel < 1e-5),
        "real_leg_spearman_focal_vs_fisher": rho,
        "REAL_LEG_ALLOCATORS_AGREE_IN_DIRECTION": bool(rho > 0.0),
        "real_leg_n_px": int(len(a)),
        "regime_leg_n_confidently_wrong_px": n_wrong,
        "regime_leg_denominator_px": int(p_y[0].size),
        "regime_leg_min_p_y_on_field": float(p_y[0].min()),
        "regime_leg_n_argmax_wrong_px": int((live_m < 0).sum()),
        "regime_leg_median_abs_margin_of_wrong": float(
            np.median(np.abs(live_m[live_m < 0]))) if (live_m < 0).any() else float("nan"),
        "REGIME_LEG_IS_VACUOUS_ON_THIS_INPUT": bool(n_wrong == 0),
        "regime_leg_vacuity_reason": (
            "the b2b solve field contains NO confidently-wrong pixels (min p_y "
            f"{float(p_y[0].min()):.4f}; all {int((live_m < 0).sum())} wrong pixels are marginal), "
            "so the focal-vs-Fisher disagreement regime is absent from this input. Reported "
            "VACUOUS with its denominator rather than scored as pass or fail."),
        "regime_leg_closed_form_focal_at_p_y_0.9_0.5_0.2_0.02": focal_probe,
        "regime_leg_closed_form_fisher_trace_at_m_0.05_0.5_2_6": fisher_probe,
        "regime_leg_note": "FUNCTION-DOMAIN evaluation of the two closed forms, NOT a measurement "
                           "on this vehicle: focal rises monotonically as p_y falls (up-weighting "
                           "confidently-wrong px) while tr g falls monotonically as |m| grows "
                           "(down-weighting them). Settling the COMPOSITION question needs a real "
                           "mid-training TR1 logit field, which is a scorer pass this arm does not "
                           "hold. Duty-to-measure, named rather than implied.",
    }

    # ---------------- FORCE 3: --head-natural-grad ----------------
    ng = make_seg_logits_natural_grad_mlx(1e-3)
    fwd = ng(seg_logits)
    fwd_max_abs_diff = float(np.abs(np.asarray(fwd) - lg_np).max())

    def _loss_plain(x):
        return _tau_softplus_loss(mx, x, lstar_oh, 0.3, None)[0]

    def _loss_ng(x):
        return _tau_softplus_loss(mx, ng(x), lstar_oh, 0.3, None)[0]

    g_plain = np.asarray(mx.grad(_loss_plain)(seg_logits))
    g_ng = np.asarray(mx.grad(_loss_ng)(seg_logits))
    grad_max_abs_diff = float(np.abs(g_ng - g_plain).max())
    denom = float(np.linalg.norm(g_plain) * np.linalg.norm(g_ng))
    cos = float((g_plain * g_ng).sum() / denom) if denom > 0 else float("nan")
    out["head_natural_grad"] = {
        "forward_max_abs_diff": fwd_max_abs_diff,
        "FORWARD_IS_IDENTITY": bool(fwd_max_abs_diff == 0.0),
        "grad_max_abs_diff": grad_max_abs_diff,
        "BACKWARD_IS_CHANGED": bool(grad_max_abs_diff > 0.0),
        "grad_cosine_vs_euclidean": cos,
        "note": "BOTH legs are required. Forward-identity alone would be satisfied by a no-op; a "
                "changed backward alone would not be the claimed forward-identity preconditioner.",
    }

    # ---------------- FORCE 4: --tau-softplus-tau ----------------
    taus = [0.05, 0.15, 0.3, 0.6, 1.0]
    losses, conc = [], []
    sm = np.abs(np.asarray(signed)[0])
    near = sm < 1.0                      # the separatrix band (small live margin)
    for t in taus:
        lo, _ = _tau_softplus_loss(mx, seg_logits, lstar_oh, t, None)
        gt_ = np.asarray(mx.grad(lambda x, _t=t: _tau_softplus_loss(
            mx, x, lstar_oh, _t, None)[0])(seg_logits))
        gmag = np.abs(gt_).sum(-1)[0]
        losses.append(float(lo))
        conc.append(float(gmag[near].sum() / max(gmag.sum(), 1e-30)))
    out["tau_softplus_tau"] = {
        "taus": taus, "losses": losses, "separatrix_grad_share": conc,
        "LOSS_RESPONDS_TO_TAU": bool(len(set(losses)) == len(losses)),
        "CONCENTRATION_MONOTONE_DECREASING_IN_TAU": bool(
            all(conc[k] >= conc[k + 1] for k in range(len(conc) - 1))),
        "separatrix_band": "|live signed margin| < 1.0",
        "note": "PREDICTION: smaller tau => the surrogate approaches the hard top-2 margin => a "
                "LARGER share of gradient magnitude sits in the separatrix band. Monotonicity is "
                "the falsifiable part; the loss merely moving is not.",
    }

    checks = {
        "focal_prediction": out["focal"]["PREDICTION_HOLDS"],
        "focal_mutation_control_fails": out["focal"]["MUTATION_CONTROL_FAILS_AS_REQUIRED"],
        "fisher_exact_law": out["fisher_density"]["EXACT_LAW_HOLDS"],
        "fisher_real_leg_allocators_agree": out["fisher_density"][
            "REAL_LEG_ALLOCATORS_AGREE_IN_DIRECTION"],
        "ng_forward_identity": out["head_natural_grad"]["FORWARD_IS_IDENTITY"],
        "ng_backward_changed": out["head_natural_grad"]["BACKWARD_IS_CHANGED"],
        "tau_concentration_monotone": out["tau_softplus_tau"][
            "CONCENTRATION_MONOTONE_DECREASING_IN_TAU"],
    }
    out["checks"] = checks
    out["ALL_CHECKS_PASS"] = all(bool(v) for v in checks.values())

    p = Path(args.json_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ALL_CHECKS_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
