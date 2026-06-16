#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Byte-neutral d_seg-aware TAPER waterfill SOLVE — upgrade the hand-tuned heuristic to a
genuine marginal-equalization solve driven by the MEASURED per-stage d_seg-sensitivity map.

INPUT: ``reports/dseg_sensitivity_map_n600.json`` (the REAL exact_eval sensitivity map from
``probe_dseg_sensitivity_map_basin_n600.py``). Per STAGE s (0..6), the map gives the measured
positive Δd_seg attributable to that stage's width and the stage's current param mass; the
sensitivity DENSITY = Δd_seg-share / param-share is the marginal value of a param at that stage.

THE SOLVE (waterfilling / marginal equalization):
  maximize  total predicted Δd_seg-reduction = Σ_s density_s · Δparams_s
  s.t.      Σ_s params_s(channels) ≈ vendored_param_count  (byte-neutral, within ~3%)
            channels has exactly 7 positive entries, channels[6] (final) near baseline
            (raising the final inflates ``refine`` QUADRATICALLY — a rate-blowup, so we cap it).

Because conv param mass COUPLES adjacent stages (block i = c_i·c_{i+1}·36+..., so widening
stage j raises BOTH block_{j-1} and block_j), the marginal cost of a +1 channel at stage j is
not constant — it is ``∂params/∂c_j`` evaluated at the current schedule. The honest waterfill is
therefore a GREEDY marginal-equalization on the discrete channel lattice: repeatedly move one
channel-unit from the lowest marginal-VALUE-density stage to the highest, where
  marginal value-density at stage j = density_j / (∂params/∂c_j),
until the densities equalize (no move improves the predicted total at <=byte-neutral cost) OR a
constraint binds (final cap / positivity / byte band). This is a first-order/marginal solve: the
per-stage density is the measured first-order sensitivity; the DOWNSTREAM from-scratch A/B is the
authority that validates the realized Δd_seg (the solve PROPOSES; the exact_eval DISPOSES).

AUTHORITY: ``[contest-CPU advisory] NON-PROMOTABLE``. The density inputs are sensitivity
MEASUREMENTS, not score claims; the solved channels are a PROPOSAL for the from-scratch A/B.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.torch_vehicle.configurable_taper_decoder import (
    decoder_param_count,
    dseg_aware_taper,
    vendored_taper,
)

_MAP_PATH = Path("reports/dseg_sensitivity_map_n600.json")


def _param_cost_marginal(channels: list[int], stage: int, delta: int, latent_dim: int) -> int:
    """Exact Δparams from changing channels[stage] by ``delta`` (couples adjacent stages via the
    block convs). Computed by the canonical ``decoder_param_count`` difference — no re-derivation,
    so it cannot drift from the module construction."""
    base = decoder_param_count(channels, latent_dim=latent_dim)
    cand = list(channels)
    cand[stage] = max(1, cand[stage] + delta)
    return decoder_param_count(cand, latent_dim=latent_dim) - base


def _tensor_masses(c: list[int], latent_dim: int) -> dict[str, float]:
    """Exact per-WEIGHT-TENSOR param mass for schedule ``c`` (same tensors the probe perturbed).
    Keyed by the probe's tensor names so per-tensor measured densities multiply the right mass.
    Sums to ``decoder_param_count`` (verified by the self-consistency test)."""
    L = latent_dim
    bh, bw = 6, 8
    m: dict[str, float] = {}
    m["stem.weight"] = L * (c[0] * bh * bw) + (c[0] * bh * bw)
    for i in range(6):
        ci, co = c[i], c[i + 1]
        m[f"blocks.{i}.weight"] = ci * (co * 4) * 9 + (co * 4)
        if ci != co:
            m[f"skips.{i}.weight"] = ci * co + co
    cf = c[6]
    m["refine.0.weight"] = cf * (cf // 2) * 9 + (cf // 2)
    m["refine.1.weight"] = (cf // 2) * cf * 9 + cf
    m["rgb_0.weight"] = cf * 3 * 9 + 3
    m["rgb_1.weight"] = cf * 3 * 9 + 3
    return m


def waterfill_solve(
    stage_density: dict[int, float],
    *,
    tensor_density: dict[str, float] | None = None,
    base_channels: int = 20,
    latent_dim: int = 28,
    final_cap: int | None = None,
    byte_band: float = 0.03,
    trust_region: float = 0.40,
    max_iters: int = 100000,
) -> dict:
    """Greedy marginal-equalization waterfill with a TRUST REGION. ``stage_density[s]`` = measured
    d_seg-sensitivity density of stage s (Δ-share / param-share). Moves one channel-unit at a time
    from the lowest-marginal-value stage to the highest, holding total params within ``byte_band``
    of the vendored target AND each stage within ``trust_region`` (±frac of its vendored width).

    WHY the trust region: the per-stage density is a FIRST-ORDER marginal measured AT the vendored
    schedule (a 20%-RMS perturbation). The locally-linear waterfill objective is only valid in a
    NEIGHBOURHOOD of that point — extrapolating to a 1-channel early stage (which would break the
    decoder's early representation) violates the model. The trust region keeps the solve inside the
    regime where the measured density is a faithful gradient, so the SOLVED taper is a defensible
    first-order step the downstream from-scratch A/B then validates (the solve PROPOSES a local
    improvement; the exact_eval DISPOSES). Without it the linear program degenerates to collapsing
    all low-density stages to 1 channel — outside the measured regime and architecturally invalid."""
    vend = vendored_taper(base_channels)
    target = decoder_param_count(vend, latent_dim=latent_dim)
    if final_cap is None:
        final_cap = vend[-1]  # hold final at baseline → bound refine's quadratic blowup

    ch = list(vend)  # start from the vendored schedule (byte-neutral by construction)
    lo = target * (1.0 - byte_band)
    hi = target * (1.0 + byte_band)
    # per-stage trust-region bounds: within ±trust_region of the vendored width, floor 1.
    tr_lo = [max(1, round(v * (1.0 - trust_region))) for v in vend]
    tr_hi = [max(1, round(v * (1.0 + trust_region))) for v in vend]
    tr_hi[6] = min(tr_hi[6], final_cap)  # final stage also bounded by the refine-quadratic cap

    def can(c):
        p = decoder_param_count(c, latent_dim=latent_dim)
        return (lo <= p <= hi and all(x >= 1 for x in c) and c[6] <= final_cap
                and all(tr_lo[s] <= c[s] <= tr_hi[s] for s in range(7)))

    # Stage→tensor coupling: each weight tensor's mass scales with its OUTPUT-width stage (the
    # stage whose density was measured by perturbing that tensor). A channel c_j appears in TWO
    # tensors (blocks.{j-1} output AND blocks.j input), so per-TENSOR densities credit the coupling
    # faithfully — strictly better than per-stage attribution (which double-counts a shared width).
    def _stage_of(name: str) -> int:
        if name == "stem.weight":
            return 0
        if name.startswith(("blocks.", "skips.")):
            return int(name.split(".")[1]) + 1  # output width = stage i+1
        return 6  # refine.* / rgb_* draw the final width

    def objective(c):
        """The byte-neutral waterfill OBJECTIVE: predicted total d_seg-reduction credit
        ``J(c) = Σ_tensor density_tensor · params_tensor(c)`` (per-TENSOR when ``tensor_density`` is
        given; else per-STAGE fallback). ``density_tensor`` (Δd_seg-per-param measured by the probe
        perturbing that tensor) × the tensor's actual param mass = its predicted d_seg-reduction
        credit. Reallocating params from low-density tensors to high-density tensors RAISES J —
        maximizing J at fixed total params IS the waterfill. Each accepted move requires a STRICT J
        increase and the lattice is finite+bounded (trust region) ⇒ the greedy converges, no cycles."""
        masses = _tensor_masses(c, latent_dim)
        if tensor_density:
            return sum(tensor_density.get(t, 0.0) * masses.get(t, 0.0) for t in masses)
        # per-stage fallback: aggregate tensor masses to stages, weight by stage_density.
        stage_mass = dict.fromkeys(range(7), 0.0)
        for t, mt in masses.items():
            stage_mass[_stage_of(t)] += mt
        return sum(stage_density.get(s, 0.0) * stage_mass[s] for s in range(7))

    moves = 0
    cur_J = objective(ch)
    for _ in range(max_iters):
        # Pick the single byte-neutral +1/-1 pair that MOST raises the objective J (strict climb).
        best = None  # (new_J, cand)
        for down_s in range(7):
            if ch[down_s] - 1 < 1:
                continue
            for up_s in range(7):
                if up_s == down_s:
                    continue
                if up_s == 6 and ch[up_s] + 1 > final_cap:
                    continue
                cand = list(ch)
                cand[down_s] -= 1
                cand[up_s] += 1
                if not can(cand):
                    continue
                new_J = objective(cand)
                if new_J > cur_J + 1e-9 and (best is None or new_J > best[0]):
                    best = (new_J, cand)
        if best is None:
            break  # marginal-equalization converged: no byte-neutral pair strictly raises J
        cur_J = best[0]
        ch = best[1]
        moves += 1

    solved_params = decoder_param_count(ch, latent_dim=latent_dim)
    return {
        "solved_channels": ch,
        "vendored_channels": vend,
        "target_params": target,
        "solved_params": solved_params,
        "byte_gap_pct": 100.0 * (solved_params - target) / target,
        "moves": moves,
        "final_cap": final_cap,
        "trust_region": trust_region,
        "trust_lo": tr_lo,
        "trust_hi": tr_hi,
        "objective_kind": "per_tensor" if tensor_density else "per_stage",
        "stage_density": {str(s): stage_density.get(s, 0.0) for s in range(7)},
        "tensor_density": dict(tensor_density) if tensor_density else None,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", type=Path, default=_MAP_PATH,
                   help="measured sensitivity-map JSON (default reports/dseg_sensitivity_map_n600.json)")
    p.add_argument("--base-channels", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=28)
    p.add_argument("--byte-band", type=float, default=0.03, help="byte-neutrality tolerance (frac)")
    p.add_argument("--trust-region", type=float, default=0.40,
                   help="per-stage trust region (±frac of vendored width); keeps the first-order "
                        "marginal solve inside the measured regime.")
    p.add_argument("--final-cap", type=int, default=None,
                   help="cap on the final-stage channel (default = vendored final, bounds refine).")
    args = p.parse_args(argv)

    if not args.map.exists():
        raise FileNotFoundError(
            f"sensitivity map {args.map} missing — run probe_dseg_sensitivity_map_basin_n600.py "
            "first. Refusing to solve against a fabricated map (NO-FAKE)."
        )
    m = json.loads(args.map.read_text())
    stage_summary = m.get("stage_summary", {})
    if not stage_summary:
        raise ValueError(f"{args.map} has no stage_summary — re-run the n600 probe.")
    stage_density = {int(s): float(v["density"]) for s, v in stage_summary.items()}

    # Per-TENSOR density = Δ-share / param-share (positive Δ only), the faithful per-tensor marginal
    # the objective consumes (a channel appears in two tensors; per-tensor credit handles that).
    rows = m.get("rows", [])
    tot_d = sum(max(r["delta"], 0.0) for r in rows) or 1e-9
    tot_p = sum(r["params"] for r in rows) or 1
    tensor_density = {
        r["tensor"]: (max(r["delta"], 0.0) / tot_d) / (r["params"] / tot_p) for r in rows
    } if rows else None

    sol = waterfill_solve(
        stage_density,
        tensor_density=tensor_density,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        final_cap=args.final_cap,
        byte_band=args.byte_band,
        trust_region=args.trust_region,
    )

    vend = sol["vendored_channels"]
    heur = dseg_aware_taper(args.base_channels, latent_dim=args.latent_dim)
    heur_params = decoder_param_count(heur, latent_dim=args.latent_dim)
    solved = sol["solved_channels"]

    print("=== TAPER WATERFILL SOLVE (n600 measured map) [contest-CPU advisory] NON-PROMOTABLE ===")
    print("  measured per-stage density: " +
          ", ".join(f"st{s}={stage_density[s]:.2f}" for s in range(7)))
    print(f"  vendored : {vend}  params={sol['target_params']}")
    print(f"  heuristic: {heur}  params={heur_params} "
          f"(gap {100*(heur_params-sol['target_params'])/sol['target_params']:+.2f}%)")
    print(f"  SOLVED   : {solved}  params={sol['solved_params']} "
          f"(gap {sol['byte_gap_pct']:+.2f}%, {sol['moves']} moves, final_cap={sol['final_cap']})")
    agree = "AGREE (within 1 ch/stage)" if all(
        abs(a - b) <= 1 for a, b in zip(solved, heur, strict=True)) else "DIFFER"
    print(f"  solved vs heuristic: {agree}")
    for s in range(7):
        print(f"    stage{s}: vendored {vend[s]:>3} | heuristic {heur[s]:>3} | "
              f"SOLVED {solved[s]:>3}  (density {stage_density[s]:.2f})")

    out = {
        **sol,
        "heuristic_channels": heur,
        "heuristic_params": heur_params,
        "solved_vs_heuristic": agree,
        "map_path": str(args.map),
        "authority": "contest-CPU advisory NON-PROMOTABLE",
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/taper_waterfill_solve_n600.json").write_text(json.dumps(out, indent=2))
    print("  wrote reports/taper_waterfill_solve_n600.json")
    print(f"\n  --taper-channels \"{','.join(str(c) for c in solved)}\"  (for launch_taper_ab --arm custom)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
