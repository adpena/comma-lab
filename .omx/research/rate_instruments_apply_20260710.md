# #394 UNIT B — rate instruments APPLIED to the mod32cap witness (2026-07-10)

**Pointer 0.19110 UNMOVED.** Both instruments are MEANS; both landed a MEASURED verdict, neither moves
the exact pointer. Authority: `[macOS-CPU advisory]` NON-PROMOTABLE (bounded evenly-spaced subsets,
witness-alone surface). Ckpt: `levelset_witness_ema_BEST.npz` (mod32cap ep650, d_seg 0.003366),
READ-ONLY copy at `experiments/results/perclass_bitalloc_witness_20260710/mod32cap_ep650_BEST.npz`.
git sha `a6e0fad66`.

STORES CONSULTED: CLAUDE.md §WITNESS CAPSTONE + §GOAL · `papers_checked_stac_sparc_taskaware_
compression_20260709.md` (the SPARC grain) · FEED-07k (prior aggregate #336, `sensitivity_bitalloc_
witness_n96_20260707.md`) · `#121` lever ledger (`re-validate-at-convergence`) · `frontier_exact_
bitalloc.py` (#157 KKT) · MEMORY.md L25 (basis-before-capacity), L2 (mod32cap = council clean
baseline), L57 (per-class attribution). Verdict-scope ladder + P2/P10/P12 honored.

---

## Item 1 — #336 SPARC-grain: per-class-weighted vs aggregate #157 KKT waterfill

**DECISION (ΔS sign): POSITIVE for all four allocations — NOTHING PAYS. And the SPARC per-class tilt
is strictly DOMINATED by aggregate by +0.45 to +0.53 S.** MEASURED, not assumed.

### What was applied
The shipped #336 apply (`apply_sensitivity_bitalloc_witness.py`, FEED-07k) drives the #157 KKT
reverse-waterfill off the AGGREGATE d_seg response. The SPARC grain (folded into #336's task by
`papers_checked_stac_sparc_...`) requires the sensitivity be per-class-weighted / margin-aware to keep
the equal-marginal allocation from starving rare-class weights (Lane = 0.64% of pixels). New tool
`tools/apply_perclass_bitalloc_witness.py` measures BOTH functionals head-to-head through the SAME
allocator and re-measures the TRUE aggregate d_seg (the contest metric) for each:
- **aggregate** `D_agg = misses/pixels` (the contest metric = uniform per-pixel argmax-mean).
- **per-class** `D_pc = mean_k missrate_k` (each of 5 classes weight 1/5 → Lane counts 20% not 0.64%).

REUSE (not re-derivation): `waterfill_bit_allocation` / `lam_for_target_mean_bits` /
`CombinedTensorSensitivity` (#157), `_realize_alloc` / `_brotli_bytes` (#202 grammar, real brotli-11
bytes), the `measure_contour_string_flip_coding` render authority — all imported as-is.

### Baseline int8 (n48 eval)
weights_total 82,197 B · aggregate d_seg 0.003458 · per-class miss-rate
[Road 0.00498, **Lane 0.23073**, Undrivable 0.00072, Movable 0.03596, MyCar 0.00065]. The SPARC
premise is REAL: Lane's within-class miss rate is 0.23 (25% of Lane pixels flip) while aggregate is
0.0035 — Lane dominates the per-class functional exactly as SPARC warns.

### Per-tensor sensitivity re-ranking (n8 probe, int8→int5)
| tensor | c_agg | c_pc | note |
|---|---:|---:|---|
| in_proj.weight | 0.194 | 2.27 | #1 in both |
| code | 0.099 | 1.33 | per-class boosts (Lane-supporting latent) |
| hidden.0.weight | 0.073 | 1.28 | per-class boosts |
| hidden.1.weight | 0.039 | 0.74 | per-class boosts |
| film.weight | 0.034 | 0.46 | |
| hidden.2/3 | 0.018/0.016 | 0.23/0.34 | |
| palette | 0.012 | 0.13 | |
| out_tex.weight | 0.005 | 0.08 | |
| **out_sdf.weight** | **0.0013** | **0.000** | per-class ZEROED it (Δpc noise-negative on n8) |

Per-class genuinely RE-RANKS — it boosts `code`/`hidden.*` (Lane-supporting depth) and, critically,
**zeroes `out_sdf` (the class-logit head)** because its int5 probe response on the tiny n8 subset gave
a negative per-class delta (rare-class noise).

### Head-to-head (n48 eval)
| mb | arm | bytes | agg d_seg | Δseg S | Δrate S | **net ΔS** | Lane miss | out_sdf bits |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 6.0 | AGG | 55,027 | 0.005361 | +0.190 | −0.018 | **+0.172** | 0.340 | 6 |
| 6.0 | PC  | 56,090 | 0.010671 | +0.721 | −0.017 | **+0.704** | 0.499 | **2** |
| 5.0 | AGG | 41,131 | 0.011345 | +0.789 | −0.027 | **+0.761** | 0.594 | 5 |
| 5.0 | PC  | 43,719 | 0.015811 | +1.235 | −0.026 | **+1.210** | 0.637 | **2** |

**PC − AGG net ΔS = +0.532 (mb6), +0.448 (mb5): per-class is WORSE at BOTH operating points**, and
worse even on the TRUE Lane miss-rate (0.499 vs 0.340 at mb6). The mechanism: the per-class functional
mis-attributed `out_sdf` (the SDF/class-logit head) as insensitive, so waterfill starved it to
b_min=2 → catastrophic aggregate AND per-class collapse. The rate saving is nearly identical
(both ~−0.018 S) so the win is entirely rate-capped and the d_seg cost dominates.

### Verdict (verdict-scope: FORMULATION — the SPARC per-class-tilt formulation on a frozen-uniform
scorer; the #157 allocator + #336 aggregate apply are untouched)
1. **The SPARC grain is a MEASURED NO for the witness regime.** SPARC's tilt is correct for a
   TRANSPORT codec with a per-class-caring downstream task; the witness's frozen scorer computes
   d_seg as a UNIFORM per-pixel argmax-mean, so the tilt optimizes a NON-contest objective and is
   dominated by the aggregate functional that directly targets the contest metric. Deep-math
   expectation confirmed: optimizing the tilted objective gives a worse contest-objective result.
2. **Confirms FEED-07k**: int8 is past this witness's RD knee — NO sub-int8 weight allocation pays
   (all four net ΔS positive). The grain is honored and closed, not re-opened.
3. **P2 noise-floor**: subset noise on net ΔS is ~±0.06 (AGG mb6 = +0.114 at n96 vs +0.172 at n48).
   The PC−AGG gap (+0.45 to +0.53) is ~8× the subset noise → the DOMINATION verdict is robust to
   subset noise. Confound noted: `out_sdf`'s per-class starvation partly rides n8-probe noise
   (per-class functional AMPLIFIES rare-class probe noise — itself a fragility of the instrument).
4. **What n600 needs** (only promotable path): a full #202 byte-close of the chosen allocation +
   n600 verdict through `tools/levelset_byte_close_and_eval.py`, then contest-CPU/CUDA exact eval.
   Given the sign and magnitude, n600 is NOT warranted for the per-class arm (dominated); the
   aggregate apply's n600 was already deemed not-warranted in FEED-07k.

Artifacts: `experiments/results/perclass_bitalloc_witness_20260710/perclass_bitalloc_n48.json`.

---

## Item 2 — #121 d_seg-aware byte-neutral Fourier taper: re-grade on the witness vehicle

**VERDICT: FLAT on the witness → cross-vehicle CONFIRMATION (P10 — the frontier negative
GENERALIZES). Do NOT spec an A/B.** The witness's Fourier-column saliency is EVEN FLATTER than the
frontier's 5.5×.

### Two distinct "saliency profiles" — do NOT conflate (they give opposite answers)
The #121 lever downgrade ("frontier saliency 5.5× flat") was on the HNeRV channel schedule. On the
witness there are TWO different domains, and the taper lives in the second:

1. **Per-TENSOR bit-alloc saliency** (item-1 probe, the #336/#157 domain): NON-flat — c_t spread
   7.28× median-relative, 86× min-relative, `in_proj` dominant. This is the allocator's domain, NOT
   the taper's.
2. **Per-Fourier-COLUMN spectral saliency** (the #121 taper's ACTUAL decision domain): computed at
   $0 from the witness's generic curvelet bank (`curvelet_directional_B` → `curvelet_feats`, 80 of
   96 feature columns = 83%) + the GT top1-top2 margin saliency field (`gt_n600.npz['margins']`,
   n16 pairs), via `saliency_from_margins` + `compute_dseg_aware_fourier_taper`:
   - `r_k` (boundary-annulus concentration, unit-mean): min 0.955, max 1.045, **max/min = 1.09×**.
   - taper `w` (strength 1, floor 0.05): max/min 1.09×; **L1 reallocation |w−1|.mean() = 0.0027**
     — a near-no-op (the frontier's 5.5× is 60× more concentrated than the witness's 1.09×).

### Why the taper is flat here (deep-math — the generalizing constraint)
The #121 taper reweights GLOBAL-support Fourier/curvelet columns, but d_seg saliency is a LOCAL,
codim-1 boundary-annulus field (annulus law: ~97% of d_seg in ~4.7% of area). A global-support basis
column integrates the annulus near-uniformly → its saliency-weighted-energy ratio r_k ≈ 1 for every
column → the byte-neutral reallocation has essentially nothing to move. This is a BASIS-STRUCTURAL
reason (global basis vs local saliency), which is why it was flat on the frontier (channel schedule)
AND flat on the witness (Fourier columns) — the negative generalizes across vehicles that use
global-support bases. SECOND, distinct reason: a fixed linear feature taper before a TRAINED linear
`in_proj` is REDUNDANT at convergence (in_proj already learned per-column scaling; a fixed `w[k]` is
a reparametrization the trained weights absorb) — so a post-hoc taper on the frozen ep650 ckpt cannot
lower d_seg regardless, and a training-time taper only changes the trajectory/prior.

### Verdict (verdict-scope: FORMULATION — measured on the generic curvelet bank, 83% of columns)
- **Cross-vehicle confirmation registered (P10 constraint row): the #121 taper's flat-saliency
  negative generalizes from the HNeRV frontier to the mod32cap witness** (1.09× vs 5.5×; even
  flatter). Do NOT spec a fresh byte-neutral taper A/B on the strength of saliency concentration.
- **Named residual (honest bound)**: the 16 self-orient DIRECTIONAL columns (17% of 96; they orient
  per-pixel to the GT tangent so COULD concentrate more) were NOT measured (needs the self-orient
  front-end). Queued micro-measurement below if #121 is ever re-opened. The dominant 83% of columns
  are flat and the linear-absorption-at-convergence argument holds for all columns, so the residual
  does not overturn the confirmation — it bounds it.

Artifact: item-2 profile reproducible at $0 via the snippet in the DAG FEED (no render, no SegNet).

---

## Queued commands (governed; owed-16 A/B owns the machine)
```bash
# Item 1 — full n600 per-class-vs-aggregate (only promotable if the sign ever flips; NOT warranted
# given the measured domination, queued for completeness):
TAC_GOVERNED_ADMISSION=1 .venv/bin/python tools/safe_run.py --rss-mb 8192 --projected-gib 4 -- \
  .venv/bin/python tools/apply_perclass_bitalloc_witness.py \
    --ckpt-dir experiments/results/perclass_bitalloc_witness_20260710 \
    --npz-name mod32cap_ep650_BEST.npz \
    --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
    --probe-pairs 16 --eval-pairs 600 --mean-bits 6 5 --torch-threads 2 --chunk-seconds 300 \
    --out experiments/results/perclass_bitalloc_witness_20260710/perclass_bitalloc_n600.json

# Item 2 — self-orient directional-column residual micro-measurement (only if #121 re-opened): build
# the 16 self-orient cols via the render front-end and re-run compute_dseg_aware_fourier_taper on the
# full 96-col feats; expected flat by the global-support argument.
```

## Triality
- **DAG**: FEED-u394b (below, appended to `sub015_DAG_*`).
- **DSL**: no gauge change — both are default-OFF measurement rows on the compress-half (rate)
  instrument, not new trainer levers. `DsegAwareTaper` DSL lever unchanged (verdict sharpened, ledger
  updated). `#336` allocator is a tool, not a DSL lever.
- **equations**: two MEASURED `EmpiricalAnchor`s appended —
  `witness_perclass_vs_aggregate_waterfill_n48_ep650_20260710` →
  `rate_mdl_cosmological_constant_reverse_waterfill_v1`; and
  `witness_fourier_taper_saliency_flat_n16_ep650_20260710` → `dseg_aware_fourier_taper_reweight_v1`.

**HARD GATE: pointer 0.19110 UNMOVED. Both instruments = MEANS; both MEASURED NO / confirmation.**
