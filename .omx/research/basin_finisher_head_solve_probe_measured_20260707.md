# Quadratic basin finisher — Stage-0 head-solve probe MEASURED (#341, council §16.1) — 2026-07-07

**Axis:** `[macOS-CPU advisory]` NON-PROMOTABLE (numpy-fp32 deploy forward + frozen CPU-torch SegNet,
chunked vbatch 12). Pointer contest-CPU **0.19110 UNMOVED** (this is a means/measurement row).
Machine-readable row (costate SENSE consumable): `reports/basin_finisher_probe_20260707.json`.
Tool: `tools/quadratic_basin_finisher_probe.py` (+ tests) — predecessor-authored, successor-verified,
committed `f307b5dcf`. Durable artifacts: `experiments/results/basin_finisher_probe_20260707/`.

## What was measured

On the **mod32cap ep650 EMA-best** checkpoint (tau-saturated; in-trainer logged n600 d_seg
**0.0033662**; the live run at ep930, deep in the Muon era, has still not beaten it):

1. **Self-orient reconstruction gap** (co-evolved state not persisted in checkpoints): probe
   fixed-point reconstruction (GT-seeded, 1 iter, all 600 pairs) gives baseline n600 d_seg
   **0.0035103** = **+4.3%** over the in-run verdict. 24-pair convergence probes: gt-seed
   0.003522→0.003395→0.003411 (iters 0/1/2), zero-seed iter-2 0.003419 — the state is
   reconstructible to ~0.7% from either seed by 1–2 iterations. All probe A/Bs below use the probe's
   own 0.0035103 baseline (identical codepath both arms — apples-to-apples).
2. **Stage 0, head-only damped Newton-CG** (Levenberg; HVP = `mx.vjp` of `mx.grad`, MLX-CPU fp32) on
   the exactly-affine head subset `{out_sdf.{weight,bias}, out_tex.{weight,bias}, palette}` (~791
   params; FiLM gains excluded — not affine), solving THE live tau-stage loss
   (`100·mean(τ·softplus(−signed/τ))`, τ=0.3, + 0.001·length; w_pose=0, eikonal=0 — verified against
   the live launch argv), on a seeded 8-pair subset (seed 0 → pairs [9,24,45,160,183,304,378,504]):
   - 2 LM rounds, **both accepted**, ρ = **0.847, 0.868**; λ: 0.1 → 0.0333 → 0.0111; 16 CG iters
     (cap 8/round, truncated-Newton); gnorm 0.249 → 0.161.
   - Subset proxy loss **0.22837 → 0.22078 (−3.3%)**.
3. **n600 chunked verdict on solved θ** (the real deploy path: int8-dequant → numpy fwd → R →
   uint8 → frozen CPU SegNet argmax): **0.0036878 = +5.1% WORSE** than the 0.0035103 baseline.

## The decisive attribution (per-pair deltas, both arms' n600 jsonls)

| slice | baseline | solved | Δ |
|---|---|---|---|
| 8 solve-subset pairs | 0.003345 | 0.003232 | **−3.4%** |
| 592 held-out pairs | 0.003513 | 0.003694 | **+5.2%** |
| pairs worse / better | | | 546 / 54 |

- **Proxy→verdict transfer is 1:1 in-subset** (−3.4% verdict vs −3.3% proxy): the τ=0.3 softplus
  surrogate tracks the realized argmax flip rate through int8 deploy quantization essentially
  exactly. The surrogate is NOT the problem.
- **The chart is genuinely near-quadratic** (ρ≈0.85–0.87 across both LM rounds): the Morse-lemma
  premise of §16.1 HOLDS at the ep650 head chart.
- **The failure is k=8 SUBSET OVERFIT**: 791 affine params vs 8 pairs = deep interpolation regime;
  the solve specializes to the subset at the expense of 546/600 pairs. NOT a basin-radius bound,
  NOT proxy mismatch, NOT quantization.

## Stage 1 (full-mask GN/CG at K=8): REFUSED with measured reason

The measured failure mode (subset overfit) strictly worsens with parameter count at fixed K.
Full-mask (~300K+ params) at K=8 reproduces a predictable negative at ~40× the cost. Not run —
that would be measurement theater, not measurement.

## Measured costs (CPU-MLX, co-resident with the live GPU run)

HVP ≈ **19 s/pair**, grad ≈ **10 s/pair**, verdict ≈ **1.4 s/pair**. Full-P (n600) CG iteration ≈
**3.2 h CPU** / ≈ **11 min** at the 17× GPU grouped-backward (GPU = gradient throughput only, never
verdict authority).

## Schedule consequences (the contract's §14 / TerminalSolve deliverables)

1. **§14 stage-boundary PRIMING:** subset-K head-solve is **REFUSED** as a priming primitive — it
   injects +5.2% held-out d_seg damage precisely where a boundary needs stability. The only
   admissible form is **full-P**: the grad phase then equals one epoch-equivalent of the trainer's
   existing accum loop, i.e. priming must be an **in-trainer GPU stage**, not a post-run CPU tool.
2. **TerminalSolve Phase-2 go/no-go: NO-GO** for the designed post-run subset-solve tool form.
   Conditionally open ONLY as full-P GPU in-trainer. Its three fire-time conditions vs this probe:
   (a) quadratic regime — **CONFIRMED** (ρ 0.847/0.868); (b) topology stability — **NOT measured
   here** (persistence-diagram detector still required); (c) no-transitions-remaining — **FAILS at
   ep650** (Muon ep726 was pending); this probe is the priming context, not the terminal basin.
3. **Equation registration: NONE** beyond this honest anchor (negative-verdict rule); the
   FEED-07t registration stays deferred. The positive sub-findings worth carrying forward as
   PRIORS (not laws): near-quadratic head chart at tau-saturation; 1:1 τ-softplus→argmax transfer
   through deploy; ~0.7%-reconstructible self-orient state.

## Confound/validity checks (L3 discipline)

- Apparatus liveness: both verdict arms ran the identical probe codepath on the identical
  reconstructed-feats state (tag `main_gt1`); baseline reproduces the in-run verdict to +4.3% with
  the gap itself measured and attributed (self-orient reconstruction).
- Positive control: the in-subset −3.4% verdict improvement IS the canary — the instrument
  registers a known-direction effect where theory predicts it.
- Duplicate cg_iter-7 rows in `solve_log_head.jsonl` (two racing chunk invocations resumed the same
  saved state): deterministic identical math, atomic state writes — no corruption; noted for
  forensics.
