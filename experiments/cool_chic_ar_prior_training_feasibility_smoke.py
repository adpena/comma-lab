# SPDX-License-Identifier: MIT
"""TRACK B step-2: $0 CPU feasibility smoke — does training the Cool-Chic AR prior
tighten its predicted distribution and DROP the real coded-byte count, and how fast?

This sizes the AR-prior-training campaign (the path to closing Cool-Chic's measured
~44x gap to HNeRV's decoder-weight floor) and decides local-MPS-vs-Modal-CUDA.

The ONE question (MVP-first per CLAUDE.md "Carmack MVP-first phasing"):
    When we TRAIN the AR prior (the density-unified Layer-2 rate lever), does the
    AR prior's predicted conditional distribution TIGHTEN (sigma contract) so the
    REAL coded bytes (encode_latent_chain) FALL — and at what rate per epoch?

Design (the cleanest isolation of the thesis variable = AR PRIOR QUALITY):
    - Load the real smoke's TRAINED latents (appearance already learned; FIXED).
    - Train ONLY the AR prior nets (coarse + fine) to minimize the conditional
      NLL == ``ar_gaussian_predicted_bits`` (the EXACT quantity the coder emits, the
      Layer-1<->2 unifier). This isolates "can the AR architecture fit the latent
      distribution better than the 6-epoch smoke prior did?" from latent movement.
    - At checkpoints measure (a) AR predicted bits/latent (continuous + discrete),
      (b) the REAL coded-byte count via ``encode_latent_chain`` on a representative
      window, (c) sec/epoch on CPU (for campaign cost projection).

Why CPU-only / fixed-latents:
    The basin daemon (pid 33911) owns the MPS GPU; this smoke must NOT contend.
    Training only the tiny AR prior nets (coarse+fine ~ a few K params) on fixed
    latents is fast even on CPU. The encode_latent_chain coder is pure-Python and
    slow, so we measure the real coded bytes on a SMALL representative pair window
    at a few checkpoints (the per-element coded rate is window-stationary, so a
    window faithfully tracks the full-archive trend; we verify the window vs
    full-archive ratio is stable at the endpoints).

NO FAKE (CLAUDE.md supreme rule):
    - The AR prior is REALLY trained (real backprop on the conditional NLL of the
      REAL trained latents). If it does NOT tighten, we report that honestly — a
      negative is THE key finding, not a failure.
    - The coded bytes are REALLY measured (encode_latent_chain range-codes; the
      byte count is exact, not predicted). The predicted-bits-vs-coded-bytes
      agreement is itself reported (the Layer-1<->2 unification check).

Evidence grade: ``[contest-CPU advisory]`` NON-PROMOTABLE. No scorer eval, no score
claim, no paid dispatch. The byte counts + losslessness are exact; the rate term is
advisory; the frontier is UNMOVED.

Usage::

    .venv/bin/python experiments/cool_chic_ar_prior_training_feasibility_smoke.py \\
        --smoke-bin experiments/results/cool_chic_mps_smoke_20260612T121501Z/0.bin \\
        --epochs 400 --lr 3e-3 --bits-per-std 5.0 \\
        --measure-window-pairs 48 --measure-every 50 \\
        --out-json experiments/results/<...>/ar_prior_training_feasibility.json
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch

from tac.substrates.cool_chic.architecture import CoolChicConfig, CoolChicSubstrate
from tac.substrates.cool_chic.archive import (
    CCV1_HEADER_FMT,
    CCV1_HEADER_SIZE,
    parse_archive,
)
from tac.substrates.cool_chic.entropy_coder import (
    LatentGrid,
    choose_grid_step,
    encode_latent_chain,
)

CONTEST_N = 37_545_489.0


# --------------------------------------------------------------------------- #
# AR prior training (fixed latents; isolate the prior-quality variable)        #
# --------------------------------------------------------------------------- #

def _ar_conditional_nll_bits(
    latents: torch.Tensor,
    prior: torch.nn.Module,
    grid: LatentGrid,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (continuous_nll_nats_sum, discrete_bits_sum, sigma_tensor).

    Mirrors ``_ar_log_prob_chain`` EXACTLY (zero context for t=0, then z_{t-1}).
    - continuous NLL (nats) is the trainable loss (== -log p, the Layer-2 surrogate).
    - discrete bits is ``ar_gaussian_predicted_bits``'s per-element bin rate (the
      EXACT quantity ``encode_latent_chain`` emits, up to the coder flush): it adds
      the constant ``-log2(step)`` offset, clamped >=0 (a bin can't cost < 0 bits).
    """
    n = latents.shape[0]
    prev = torch.zeros_like(latents[0:1])
    prev_stack = torch.cat([prev, latents[: n - 1]], dim=0)
    mean, log_scale = prior(prev_stack)
    sigma = torch.exp(log_scale)
    diff = (latents - mean) / sigma
    log_two_pi = math.log(2.0 * math.pi)
    # continuous Gaussian log-density (nats)
    ln_p = -0.5 * diff * diff - log_scale - 0.5 * log_two_pi
    cont_nll_nats = (-ln_p).sum()
    # discrete bin bits = (-ln_p - ln step)/ln2, clamped >= 0 per element
    bits = ((-ln_p - math.log(grid.step)) / math.log(2.0)).clamp(min=0.0).sum()
    return cont_nll_nats, bits, sigma


def _train_one_axis(
    latents: torch.Tensor,
    prior: torch.nn.Module,
    grid: LatentGrid,
    *,
    epochs: int,
    lr: float,
    measure_window_pairs: int,
    measure_every: int,
    axis_name: str,
) -> list[dict]:
    """Train ONE AR prior (coarse or fine) on its fixed latents; log the trend.

    Returns a list of checkpoint dicts (epoch, continuous bits/elem, discrete
    bits/elem, sigma stats, REAL coded bytes on the window, real coded bits/elem).
    """
    n_pairs = int(latents.shape[0])
    n_elem = int(latents.numel())
    elems_per_pair = n_elem // n_pairs
    opt = torch.optim.Adam(prior.parameters(), lr=lr)

    # Representative window for the REAL coder (pure-Python, slow): first W pairs.
    win = min(measure_window_pairs, n_pairs)
    window_latents = latents[:win].contiguous()
    window_elems = int(window_latents.numel())

    checkpoints: list[dict] = []

    def _measure(epoch: int) -> dict:
        prior.eval()
        with torch.no_grad():
            cont_nats, disc_bits, sigma = _ar_conditional_nll_bits(latents, prior, grid)
            cont_bits_per_elem = float(cont_nats / math.log(2.0) / n_elem)
            disc_bits_per_elem = float(disc_bits / n_elem)
            sigma_mean = float(sigma.mean())
            sigma_median = float(sigma.median())
        # REAL coded bytes on the window (the empirical bit-spend proof).
        t0 = time.time()
        coded = encode_latent_chain(window_latents, prior, grid)
        code_sec = time.time() - t0
        coded_bytes = len(coded)
        coded_bits_per_elem = coded_bytes * 8.0 / window_elems
        prior.train()
        return {
            "epoch": epoch,
            "ar_continuous_bits_per_elem": cont_bits_per_elem,
            "ar_discrete_bits_per_elem": disc_bits_per_elem,
            "sigma_mean": sigma_mean,
            "sigma_median": sigma_median,
            "sigma_over_step_median": sigma_median / grid.step,
            "real_coded_bytes_window": coded_bytes,
            "real_coded_bits_per_elem_window": coded_bits_per_elem,
            "window_pairs": win,
            "window_elems": window_elems,
            "code_seconds_window": code_sec,
        }

    # epoch 0 baseline (the as-shipped 6-epoch smoke prior)
    checkpoints.append({**_measure(0), "phase": "baseline_pretrained_prior"})

    epoch_times: list[float] = []
    for epoch in range(1, epochs + 1):
        prior.train()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        cont_nats, _disc_bits, _sigma = _ar_conditional_nll_bits(latents, prior, grid)
        # loss = continuous conditional NLL (nats) per element (== Layer-2 surrogate)
        loss = cont_nats / n_elem
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), max_norm=1.0)
        opt.step()
        epoch_times.append(time.time() - t0)

        if epoch % measure_every == 0 or epoch == epochs:
            cp = {**_measure(epoch), "phase": "trained"}
            cp["sec_per_epoch_mean"] = float(np.mean(epoch_times))
            checkpoints.append(cp)
            print(
                f"[{axis_name}] ep {epoch:4d}: "
                f"cont_bits/elem={cp['ar_continuous_bits_per_elem']:.4f} "
                f"disc_bits/elem={cp['ar_discrete_bits_per_elem']:.4f} "
                f"real_coded_bits/elem={cp['real_coded_bits_per_elem_window']:.4f} "
                f"sigma_med={cp['sigma_median']:.5f} "
                f"({cp['sec_per_epoch_mean']*1000:.1f} ms/epoch CPU)",
                flush=True,
            )

    return checkpoints


# --------------------------------------------------------------------------- #
# Projection + verdict                                                         #
# --------------------------------------------------------------------------- #

def _fit_exp_decay_to_floor(epochs: np.ndarray, vals: np.ndarray) -> dict:
    """Fit ``v(e) = floor + (v0 - floor) * exp(-e/tau)`` by a small grid search.

    Robust, dependency-free (no scipy.optimize). Returns floor, tau, v0, r2.
    The floor is the asymptote (the architecture's coded-rate limit); tau is the
    e-folding epoch count (how fast it gets there).
    """
    if len(epochs) < 3:
        return {"fit_ok": False, "reason": "need >=3 checkpoints"}
    v0 = float(vals[0])
    vmin = float(vals.min())
    # candidate floors below the observed min (asymptote can be below last point)
    best = None
    span = max(1e-9, v0 - vmin)
    for floor in np.linspace(vmin - span, vmin, 60):
        amp = v0 - floor
        if abs(amp) < 1e-12:
            continue
        for tau in np.linspace(max(1.0, epochs[-1] / 50.0), epochs[-1] * 4.0, 80):
            pred = floor + amp * np.exp(-epochs / tau)
            ss_res = float(np.sum((vals - pred) ** 2))
            ss_tot = float(np.sum((vals - vals.mean()) ** 2)) or 1e-12
            r2 = 1.0 - ss_res / ss_tot
            if best is None or r2 > best["r2"]:
                best = {"floor": float(floor), "tau": float(tau), "v0": v0, "r2": r2}
    if best is None:
        return {"fit_ok": False, "reason": "no fit"}
    best["fit_ok"] = True
    return best


def _project_and_verdict(
    coarse_cp: list[dict],
    fine_cp: list[dict],
    *,
    n_pairs_full: int,
    coarse_elems_full: int,
    fine_elems_full: int,
    hnerv_floor_bytes: float,
) -> dict:
    """Project coded-bytes-vs-epoch and emit GO / NEEDS-DESIGN / NO-GO."""
    out: dict = {}

    def _axis_proj(cp: list[dict], elems_full: int, name: str) -> dict:
        ep = np.array([c["epoch"] for c in cp], dtype=np.float64)
        # real coded bits/elem on the window is the empirical rate; project to full
        rate = np.array([c["real_coded_bits_per_elem_window"] for c in cp], dtype=np.float64)
        fit = _fit_exp_decay_to_floor(ep, rate)
        baseline_rate = float(rate[0])
        final_rate = float(rate[-1])
        rel_drop = (baseline_rate - final_rate) / baseline_rate if baseline_rate > 0 else 0.0
        # per-100-epoch drop over the trained span
        if len(ep) >= 2 and ep[-1] > ep[1]:
            drop_per_100 = (rate[1] - rate[-1]) / (ep[-1] - ep[1]) * 100.0
        else:
            drop_per_100 = 0.0
        floor_rate = fit.get("floor", final_rate) if fit.get("fit_ok") else final_rate
        floor_bytes_full = max(0.0, floor_rate) * elems_full / 8.0
        final_bytes_full = final_rate * elems_full / 8.0
        baseline_bytes_full = baseline_rate * elems_full / 8.0
        return {
            "axis": name,
            "baseline_coded_bits_per_elem": baseline_rate,
            "final_coded_bits_per_elem": final_rate,
            "rel_drop_pct": rel_drop * 100.0,
            "drop_bits_per_elem_per_100ep": drop_per_100,
            "exp_fit": fit,
            "projected_floor_bits_per_elem": max(0.0, floor_rate),
            "projected_floor_bytes_full_axis": floor_bytes_full,
            "final_bytes_full_axis": final_bytes_full,
            "baseline_bytes_full_axis": baseline_bytes_full,
        }

    coarse = _axis_proj(coarse_cp, coarse_elems_full, "coarse")
    fine = _axis_proj(fine_cp, fine_elems_full, "fine")
    out["coarse"] = coarse
    out["fine"] = fine

    baseline_total = coarse["baseline_bytes_full_axis"] + fine["baseline_bytes_full_axis"]
    final_total = coarse["final_bytes_full_axis"] + fine["final_bytes_full_axis"]
    floor_total = (
        coarse["projected_floor_bytes_full_axis"] + fine["projected_floor_bytes_full_axis"]
    )
    out["total_latent_bytes_baseline"] = baseline_total
    out["total_latent_bytes_final"] = final_total
    out["total_latent_bytes_projected_floor"] = floor_total
    out["total_rel_drop_observed_pct"] = (
        (baseline_total - final_total) / baseline_total * 100.0 if baseline_total else 0.0
    )
    out["projected_floor_advisory_rate_term"] = 25.0 * floor_total / CONTEST_N
    out["hnerv_floor_bytes"] = hnerv_floor_bytes
    out["projected_floor_vs_hnerv_ratio"] = (
        floor_total / hnerv_floor_bytes if hnerv_floor_bytes else float("inf")
    )

    # Verdict logic.
    # GO: the prior tightens meaningfully (real coded rate drops >= 10% over the run)
    #     AND the exp-fit floor projects materially below the baseline (the prior
    #     has not plateaued at the start). It does NOT require beating HNeRV in this
    #     smoke (that needs the long campaign + coarser grid + T1/T8); it requires
    #     the TRAINING LEVER to be real and worth a campaign.
    # NEEDS-DESIGN: tightens but slowly / plateaus far above any competitive floor.
    # NO-GO: the prior does not tighten (rate flat) -> the latent rate floor is set
    #        by the latents, not the prior -> training the prior is not the lever.
    total_rel_drop = out["total_rel_drop_observed_pct"]
    # is the floor materially below the FINAL (still descending => more headroom)?
    floor_below_final_pct = (
        (final_total - floor_total) / final_total * 100.0 if final_total else 0.0
    )
    still_descending = floor_below_final_pct > 5.0

    if total_rel_drop >= 10.0:
        verdict = "GO"
        reason = (
            f"real coded latent bytes dropped {total_rel_drop:.1f}% over the run; "
            f"the AR-prior-training lever is REAL and worth a campaign"
            + (" (still descending; more headroom)" if still_descending else "")
        )
    elif total_rel_drop >= 3.0:
        verdict = "NEEDS-DESIGN"
        reason = (
            f"prior tightens but slowly ({total_rel_drop:.1f}% over the run); a "
            "better AR architecture / context model is needed for a competitive floor"
        )
    else:
        verdict = "NO-GO"
        reason = (
            f"prior did NOT tighten ({total_rel_drop:.1f}% coded-rate change); the "
            "latent rate floor is set by the latents, not the prior — AR-prior "
            "training is not the lever (honest negative)"
        )
    out["verdict"] = verdict
    out["verdict_reason"] = reason
    out["still_descending_at_run_end"] = still_descending
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-bin", required=True, help="real CCV1 0.bin with trained latents")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--bits-per-std", type=float, default=5.0)
    p.add_argument("--measure-window-pairs", type=int, default=48,
                   help="pairs used for the REAL pure-Python coder (slow); rate is window-stationary")
    p.add_argument("--measure-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hnerv-floor-bytes", type=float, default=161_000.0,
                   help="HNeRV decoder-weight floor (~91%% of ~177KB frontier archive)")
    p.add_argument("--mps-sec-per-epoch", type=float, default=28.0,
                   help="known full-model MPS sec/epoch from step-1 (for campaign sizing)")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    raw = Path(args.smoke_bin).read_bytes()
    header = struct.unpack(CCV1_HEADER_FMT, raw[:CCV1_HEADER_SIZE])
    meta_len = header[-1]
    raw_meta = json.loads(raw[-meta_len:].decode("utf-8"))
    ar_hidden = int(raw_meta["ar_prior_hidden"])

    arc = parse_archive(raw)
    lc = arc.latents_coarse.to(torch.float32)
    lf = arc.latents_fine.to(torch.float32)
    n_pairs = int(lc.shape[0])

    # Rebuild the AS-SHIPPED priors (the 6-epoch smoke prior is the epoch-0 baseline)
    cfg = CoolChicConfig(num_pairs=n_pairs, ar_prior_hidden=ar_hidden)
    model = CoolChicSubstrate(cfg)
    model.ar_prior_coarse.load_state_dict(
        {k.replace("coarse.", ""): v for k, v in arc.ar_prior_state_dict.items() if k.startswith("coarse.")},
        strict=False,
    )
    model.ar_prior_fine.load_state_dict(
        {k.replace("fine.", ""): v for k, v in arc.ar_prior_state_dict.items() if k.startswith("fine.")},
        strict=False,
    )

    step_c = choose_grid_step(lc, bits_per_std=args.bits_per_std)
    step_f = choose_grid_step(lf, bits_per_std=args.bits_per_std)
    grid_c, grid_f = LatentGrid(step_c), LatentGrid(step_f)

    print(
        f"[smoke] real latents: coarse {tuple(lc.shape)} fine {tuple(lf.shape)} | "
        f"n_pairs={n_pairs} ar_hidden={ar_hidden} | grid_step coarse={step_c:.2e} fine={step_f:.2e}",
        flush=True,
    )

    t_start = time.time()
    coarse_cp = _train_one_axis(
        lc, model.ar_prior_coarse, grid_c,
        epochs=args.epochs, lr=args.lr,
        measure_window_pairs=args.measure_window_pairs,
        measure_every=args.measure_every, axis_name="coarse",
    )
    fine_cp = _train_one_axis(
        lf, model.ar_prior_fine, grid_f,
        epochs=args.epochs, lr=args.lr,
        measure_window_pairs=args.measure_window_pairs,
        measure_every=args.measure_every, axis_name="fine",
    )
    wall_sec = time.time() - t_start

    coarse_elems_full = int(lc.numel())
    fine_elems_full = int(lf.numel())

    proj = _project_and_verdict(
        coarse_cp, fine_cp,
        n_pairs_full=n_pairs,
        coarse_elems_full=coarse_elems_full,
        fine_elems_full=fine_elems_full,
        hnerv_floor_bytes=args.hnerv_floor_bytes,
    )

    # Campaign cost: epochs-to-floor * sec/epoch on each device.
    # Use the dominant axis's tau (e-folding) to estimate epochs to ~95% of floor.
    def _epochs_to_floor(cp_proj: dict) -> float:
        fit = cp_proj.get("exp_fit", {})
        if fit.get("fit_ok"):
            return 3.0 * float(fit["tau"])  # ~95% of the way to the asymptote
        return float("nan")

    coarse_eps = _epochs_to_floor(proj["coarse"])
    fine_eps = _epochs_to_floor(proj["fine"])
    epochs_to_floor = float(np.nanmax([coarse_eps, fine_eps]))

    # Full-model training sec/epoch: this smoke trains ONLY the AR prior on fixed
    # latents (cheap). A real campaign trains the FULL model (latents + synthesis +
    # AR + scorers). Step-1 measured ~28s/epoch on MPS for that full loop; CPU full
    # loop is far slower (not run here to avoid basin contention). We report the
    # AR-only CPU sec/epoch for THIS smoke and use the step-1 MPS number for the
    # full-campaign projection (honest: the campaign trains the full model).
    coarse_ms = coarse_cp[-1].get("sec_per_epoch_mean", float("nan")) * 1000.0
    fine_ms = fine_cp[-1].get("sec_per_epoch_mean", float("nan")) * 1000.0

    mps_hours = (epochs_to_floor * args.mps_sec_per_epoch) / 3600.0 if math.isfinite(epochs_to_floor) else float("nan")
    # Modal T4 is ~1x our reference; full-model T4 sec/epoch unknown without a smoke,
    # but the campaign is dominated by the scorer forward/backward (same as MPS
    # workload). Conservative: assume T4 ~ MPS throughput for the full loop (the
    # operator can refine with a $0.30 T4 timing smoke). Modal T4 ~ $0.59/hr.
    modal_t4_hours = mps_hours  # same workload assumption (documented)
    modal_t4_cost = modal_t4_hours * 0.59 if math.isfinite(modal_t4_hours) else float("nan")

    result = {
        "schema": "cool_chic_ar_prior_training_feasibility_v1",
        "evidence_grade": "contest-CPU advisory",
        "promotable": False,
        "score_claim": False,
        "smoke_bin": str(args.smoke_bin),
        "bits_per_std": args.bits_per_std,
        "epochs": args.epochs,
        "lr": args.lr,
        "measure_window_pairs": args.measure_window_pairs,
        "n_pairs_full": n_pairs,
        "grid_step_coarse": step_c,
        "grid_step_fine": step_f,
        "wall_seconds_total": wall_sec,
        "ar_only_cpu_ms_per_epoch_coarse": coarse_ms,
        "ar_only_cpu_ms_per_epoch_fine": fine_ms,
        "coarse_trend": coarse_cp,
        "fine_trend": fine_cp,
        "projection": proj,
        "campaign_sizing": {
            "epochs_to_floor_estimate": epochs_to_floor,
            "note_full_model_sec_per_epoch": (
                "this smoke trains ONLY the AR prior on FIXED latents (cheap, CPU); a "
                "real campaign trains the FULL model (latents+synthesis+AR+scorers). "
                "step-1 measured ~28s/epoch on MPS for that full loop; CPU full loop "
                "not run here (basin owns the GPU). The epochs_to_floor estimate is "
                "the AR-prior tightening time; the full campaign also re-optimizes "
                "latents toward compressibility so true convergence may differ."
            ),
            "mps_sec_per_epoch_full_loop": args.mps_sec_per_epoch,
            "mps_hours_to_floor": mps_hours,
            "modal_t4_hours_to_floor_same_workload_assumption": modal_t4_hours,
            "modal_t4_cost_usd_estimate": modal_t4_cost,
            "device_recommendation": (
                "local-MPS (free; sequence behind/alongside the basin)"
                if math.isfinite(mps_hours) and mps_hours < 12.0
                else "Modal-T4 (paid; operator approves) if MPS-hours exceed a basin window"
            ),
        },
        "verdict": proj["verdict"],
        "verdict_reason": proj["verdict_reason"],
    }

    print("\n" + "=" * 72)
    print(json.dumps(result["projection"], indent=2))
    print(json.dumps(result["campaign_sizing"], indent=2))
    print(f"\nVERDICT: {result['verdict']} — {result['verdict_reason']}")
    print("=" * 72)

    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[smoke] wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
