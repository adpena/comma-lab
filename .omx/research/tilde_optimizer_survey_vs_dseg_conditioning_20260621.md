# Tilde Research optimizer survey, judged against the κ≈19 shallow-boundary d_seg conditioning (2026-06-21)

**Operator ask (2026-06-21):** survey Tilde Research's optimizers/optimization papers + OSS (site + GitHub),
and judge which (if any) suit the capstone's **stage-8 d_seg-finisher** role — specifically against the deep-math
we just derived: the d_seg loss has a **boundary-dominated Gauss-Newton Hessian** that is
**conditioning-WIDE (κ-proxy ≈ 19) but SHALLOW** (66.5% of flips lost by <0.5 logit), the plateau is
**optimizer-conditioning-limited NOT capacity-limited**, and **plain Muon's Newton-Schulz `polar(M)=UVᵀ` is the
principled κ-buster** (O(ln 1/ε) vs O(κ·ln 1/ε)) whose magnitude-decoupling ALSO makes its LR batch-invariant.

**RESEARCH/ADVISORY MEMO — $0, no training files edited, no dispatch, no clone.** Authority: `[contest-CPU
advisory]` / `[external-claim]` (all Tilde perf numbers are blog/README claims). NON-PROMOTABLE
(`score_claim=false`, `promotable=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`).
Pointer UNMOVED 0.19110. Only `upstream/evaluate.py` on paired CUDA + Linux-x86_64 CPU produces a score.

Cross-links (the deep-math this judges against): `dseg_boundary_hessian_conditioning_20260621.md` (κ≈19,
shallow boundary, Muon = spectral preconditioner) · `capstone_batch_size_fixed_point_B64_launch_spec_20260621.md`
§2.1 (Muon LR batch-invariance via the same magnitude-decoupling) ·
`decoder_weight_rate_axis_and_shallow_boundary_synthesis_20260621.md` (shallow boundary is quant/perturbation-fragile).
Prior Tilde work this builds on (banked, not re-derived): `tilde_research_optimizers_survey_and_aurora_build_20260609.md`
(task #33 — full survey + the built `tac.optimization.aurora_mlx` kernel + 26 NO-FAKE tests) ·
`comp_muon_release_applicability_research_20260609.md` · `tilde_optimizers_for_inert_loop_20260610T193200Z.md`
(task #77) · `codex_findings_tilde_research_parallax_nerv_intake_20260603T174229Z_codex.md`.

---

## 1. What we already knew (banked — 1 paragraph)

We surveyed Tilde three times (tasks #33/#77 + the codex parallax intake). The standing conclusion: **plain Muon
(Keller-Jordan; PR95 L15 stage-8) is the correct stage-8 optimizer and is already MLX-live**
(`tac.local_acceleration.pr95_hnerv_mlx.zeropower_via_newtonschulz5_mlx`, wired into `apply_pr95_mlx_optimizer_step`).
Of Tilde's named optimizers: **Aurora** (MIT, `aurora-release`) fixes Muon "neuron death" — row-leverage
anisotropy on **tall** matrices under activations with `φ(0)=0 ∧ φ'(0)≈0` — but our Muon-eligible HNeRV partition
is **11 WIDE / 0 TALL / 0 SQUARE** and HNeRV uses **sin** (`sin'(0)=1≠0`, precondition broken), so Aurora
**reduces to Muon** here (verified `max diff 0.0` in our own `test_square_reduces_to_muon_polar`); predicted-null,
DEFER-with-reactivation behind a $0 MLX A/B (and we already BUILT the kernel for that A/B). **Compositional Muon**
(Apache-2.0) optimizes transformer **attention QK/OV factor-pairs** — HNeRV has no attention → structurally
inapplicable (cannot run). **Parallax** is NOT a Tilde-org optimizer and NOT an optimizer at all — it is an
attention primitive (Local Linear Attention) under `Yifei-Zuo/*` → category error. Nitrobrew/NSA/MoMoE/Wall-Attention
= regime mismatch (large-vocab KL / sparse-attention / MoE / attention decay) → DEFER. Net banked verdict: KEEP plain
Muon; Aurora is a cheap falsifiable arm only.

## 2. Fresh web survey (2026-06-21) — what is NEW since the 2026-06-10 pass

**GitHub org `github.com/tilde-research` (fetched 2026-06-21):** NO new optimizer repo since the last survey.
Optimizer/core repos unchanged: `aurora-release` (147★, updated 2026-05-27, MIT), `comp-muon-release` (22★,
updated 2026-06-05, Apache-2.0), `nitrobrew-release` (Apr 28, Apache-2.0). Attention/infra unchanged
(`wall-attention-release` Jun 3, `nsa-release`, `momoe-release`, `wall-flash-linear-attention`, `activault`,
`nitrobrew-trl/verl`). **The org has shipped no new optimizer code in the ~11 days since 2026-06-10.**

**Tilde research index `tilderesearch.com/research` — two entries newer-or-relevant than the last pass:**

| Entry | Date | What it is | Code/license | Mechanism (vs Muon) | Batch behavior |
|---|---|---|---|---|---|
| **Parallax** | 2026-06-09 (Tilde index) | NOW listed on Tilde's index, but it is an **attention primitive** (Parameterized Local Linear Attention; "single additive correction on top of Softmax"), NOT an optimizer. Public impl is `Yifei-Zuo/Parallax` (MIT), torch 2.9.1 + triton 3.5.1 + optional Hopper CuTeDSL. | code under Yifei-Zuo (MIT); no Tilde-org repo | n/a — not an optimizer (no orthogonalized update; replaces SDPA) | n/a |
| **Compositional Muon** | 2026-06-05 | Partner-whitened Muon for transformer **QK/OV** composed matrices; `cm_qk(q,k,head_dim,…)`/`cm_ov(v,o,…)` REQUIRE q/k/v/o `nn.Linear` + `head_dim`. Home-regime gain small (−0.005..−0.01 LM val loss, no downstream win). | `comp-muon-release`, Apache-2.0, PyTorch | whitens each attention factor by partner inverse-Gram-root before/after spectral sign | no batch-size studies |
| **Aurora** | 2026-05-08 | Leverage-aware optimizer; row-norm-uniform polar on **tall** matrices; fixes Muon neuron-death. | `aurora-release`, MIT, PyTorch (we have MLX port) | alternating projection Stiefel ∩ row-oblique; ≈Muon on wide/square | **none** (verified §3) |
| **Gram-Space Manifold Muon** | 2025-10-13 (vignette) | **THEORY ONLY, no code/pip.** A unifying Gram-matrix-constraint framework; derives Stiefel/Diagonal-Gram/Oblique manifold-Muon variants. Authors Keigwin/Pai/Chen. | **no repo, no package** | Stiefel-Muon = **condition number 1 by construction** (== plain Muon's polar); the new DGram/Oblique variants **RELAX** that, accepting MORE singular-value spread for "wiggle room". | not studied (MLP toy only) |

**Adjacent NON-Tilde Muon-conditioning work surfaced** (recorded for completeness; NOT Tilde, so out of scope for
this directive — flagged as separate-reactivation if we ever want a κ-busting variant beyond plain Muon):
`Mousse` (arXiv 2603.09697 — Shampoo-structural curvature-aware preconditioning on top of the Stiefel constraint),
`Muon2` (arXiv 2604.09967 — Adam-style adaptive scaling of the momentum matrix BEFORE polar, to fix the
ill-conditioned momentum spectrum), `Preconditioning Benefits of Spectral Orthogonalization in Muon` (arXiv
2601.13474 — the theory anchor our conditioning memo already cites). These are the genuinely κ-targeted second-order
Muon extensions, but none are Tilde, none have a maintained MLX path, and Muon2/Mousse add Adam/Shampoo state +
hyperparameters. Out of scope here; possible future arm.

## 3. The decisive judgment: does ANY Tilde optimizer beat PLAIN MUON at κ-busting the SHALLOW boundary?

This is the new question the prior surveys could not ask (the κ≈19 / shallow-boundary derivation post-dates them).
The capstone's binding need (restated): **a final-stage optimizer that busts the κ≈19 boundary-Hessian
conditioning to resolve the shallow d_seg flips, batch-robust (B=64 spec), stable (no Muon pose-kick), drop-in for
PR95 stage-8, and boundary-preserving (no aggressive perturbation that flips the <0.5-logit pixels).**

The load-bearing distinction: **our problem is LOSS-Hessian conditioning (κ of `H ≈ (1/P)Σ ℓ''(m_p) j_p j_pᵀ`),
NOT weight-matrix row-leverage.** Plain Muon's `polar(M)` flattens the *update* spectrum → it is the κ-buster for
the loss geometry (the conditioning memo §4 derivation; arXiv 2601.13474). Aurora flattens the *weight-matrix row
norms* → a DIFFERENT object. These are orthogonal; Aurora's fix does not touch loss κ. (Verified from the Aurora
blog 2026-06-21: *"steepest descent under the joint constraint of row-norm uniformity and orthogonality"*,
*"rows with low leverage… tend to also have low leverage in the update"* — **a weight-matrix geometry problem, not a
loss-landscape conditioning problem; no batch-size experiments; no convergence claim under ill-conditioned losses.**)

Per-optimizer suitability for the stage-8 d_seg-finisher, scored on (a) κ-busting vs plain Muon, (b) batch
robustness, (c) stability, (d) drop-in + boundary-preserving, (e) license/inflate (optimizer is compress-time only
→ never touches the numpy-portable inflate; noted once, not re-scored):

| Rank | Optimizer | (a) κ-bust vs plain Muon | (b) batch-robust | (c) stable | (d) drop-in + boundary-safe | Verdict |
|---|---|---|---|---|---|---|
| **1** | **Plain Muon (Keller-Jordan, PR95 L15)** | **IS the κ-buster** (polar→σ=1; O(ln 1/ε) regardless of κ≈19) — the baseline everything else is judged against | **YES** — magnitude-decoupling ⇒ LR batch-invariant (our B=64 §2.1 derivation); FLOP overhead Θ(m/B) shrinks at large B | known pose-kick at stage transitions / weak-pose term (the equimarginal instability) — managed by fp32-exact pose + extending stage-8 epochs, not by swapping optimizer | already live + MLX-native + canonical; preserves the shallow boundary (controlled spectral step, not magnitude-driven) | **KEEP — already optimal** |
| 2 | **Aurora** (MIT) | **NO** — fixes weight-matrix row-leverage, NOT loss κ (blog-confirmed: no conditioning/convergence claim). On our 11-WIDE/0-TALL/sin partition it **reduces to Muon** (diff 0.0) | no batch results; same NS core as Muon so ~same | ~Muon (square/wide path identical) | drop-in (we built `tac.optimization.aurora_mlx`); boundary-safe (≈Muon step). But ≈Muon ⇒ no marginal κ gain | **DEFER-with-reactivation** — $0 A/B already specced; predicted-null on κ |
| 3 | **Gram-Space Manifold Muon** (no code) | **NO / WRONG DIRECTION** — its variants RELAX Stiefel's condition-number-1 (the property that makes plain Muon the κ-buster). DGram/Oblique accept MORE spread | not studied | n/a (theory) | no repo/pip ⇒ not drop-in; would need a from-scratch build of a variant that is worse-conditioned than plain Muon | **DEFER (theory; counter-indicated)** — relaxing conditioning is the opposite of what κ≈19 needs |
| 4 | **Compositional Muon** (Apache-2.0) | n/a — **cannot run** (no attention QK/OV in HNeRV) | n/a | n/a | structurally inapplicable | **DEFER (regime mismatch)** — reactivate only with an attention witness backend |
| 5 | **Parallax** (Yifei-Zuo, MIT) | **category error** — an attention primitive, not an optimizer; produces no orthogonalized update | n/a | n/a | not an optimizer; torch+triton+Hopper runtime ⇒ also inflate-hostile if ever mis-used as a backend | **NOT-RELEVANT** |
| — | Nitrobrew / NSA / MoMoE / Wall-Attention | n/a | n/a | n/a | regime mismatch (KL/sparse-attn/MoE/attn-decay) | **DEFER** |

**(e) inflate note (applies to all):** the optimizer runs at **compress time only**; it never enters `inflate.sh`,
so optimizer license/runtime does NOT touch the contest archive's numpy-portable inflate requirement. The only
contest-relevant property of the optimizer is the weights it produces. (Tilde licenses are clean to vendor anyway:
Aurora MIT, comp-muon Apache-2.0.)

## 4. Recommendation — KEEP PLAIN MUON (do not adopt any Tilde variant for the κ problem)

**KEEP plain Muon as the stage-8 d_seg-finisher. Do NOT adopt Aurora / Gram-Space / comp-muon for the κ≈19
shallow-boundary problem.** The reasoning ties directly to the three deep-math findings:

1. **κ≈19 is a LOSS-Hessian conditioning problem, and plain Muon already IS the κ-buster** (polar flattens the
   update spectrum → O(ln 1/ε); conditioning memo §4 + arXiv 2601.13474). No Tilde optimizer improves *loss*
   conditioning over plain Muon — Aurora targets a different object (weight row-leverage), and Gram-Space's
   variants RELAX the very condition-number-1 property that makes Stiefel-Muon the buster. **The Tilde family's own
   theory (Gram-Space) confirms plain Stiefel-Muon is the conditioning optimum within their Muon design space.**
2. **The shallow boundary is perturbation-fragile** (66.5% of flips <0.5 logit; synthesis memo §4 — int5 quant
   already caps S~0.49 by flipping these pixels). This *argues against* any optimizer that perturbs the boundary
   more aggressively than plain Muon's controlled spectral step. None of the Tilde variants offer a gentler
   boundary-preserving step; they offer orthogonal (Aurora) or relaxed-conditioning (Gram-Space) or inapplicable
   (comp-muon) changes. Plain Muon's controlled, magnitude-decoupled step is the boundary-safe choice.
3. **Batch-invariance (B=64) is a plain-Muon property, not a Tilde-variant feature.** Our B=64 spec's whole
   premise (§2.1) is that Muon's magnitude-decoupling makes its LR batch-invariant; Aurora reports no batch
   results and shares Muon's NS core, so it neither adds nor subtracts here. There is no Tilde optimizer that is
   *more* batch-robust than plain Muon for our regime.

The honest meta-point (Assumption-Adversary): the assumption "a fancier optimizer beats plain Muon at d_seg" is
**CARGO-CULTED** for this problem. The conditioning math says plain Muon is already at the principled optimum for
the loss-Hessian κ; the score moves on **training to convergence with Muon (free, §4 of the synthesis memo) +
structural architecture (weight-tie/low-rank, rate)**, not on an optimizer swap. The capstone spec should keep
Muon as stage-8 and NOT swap it (conditioning memo §7.2 reaches the same verdict independently).

## 5. The smallest $0 falsification smoke (if a trial is ever wanted — it ISN'T, but here is the cheapest one)

We do NOT recommend a trial — the math predicts null and the kernel A/B was already specced. But the cheapest
$0 local falsification that would *test* "does any Tilde variant beat plain Muon at busting κ≈19" is the **stage-8
A/B we already built the kernel for**, re-purposed to measure the κ-relevant quantity rather than just loss:

- **Setup ($0, MLX-local, `[macOS-MLX research-signal]`):** from ONE fixed PR95 stage-1..7 checkpoint, run stage-8
  twice, identical epochs/LR/seed: arm A = plain Muon (`zeropower_via_newtonschulz5_mlx`), arm B = Aurora
  (`aurora_leverage_uniform_polar_mlx`, already built + tested).
- **The κ-relevant readout (the falsification metric, not loss):** at stage-8 ep{50,250,end} measure the **flipped-pixel
  |GT-class margin| distribution** (reuse probe `a688…` / `dseg_margin_distribution_capstone_ema_shadow_n24` machinery)
  and the **exact live-render d_seg** (argmax disagreement). κ-busting shows up as the near-zero-margin mass (p10
  side, the κ-driver) clearing FASTER — i.e. p10 |margin| rising and exact d_seg dropping faster in one arm.
- **Pre-registered prediction `[extrapolation]`:** arm B ≈ arm A (Aurora reduces to Muon on the all-wide/sin
  partition; it does not touch loss κ). **Falsification trigger:** if arm B clears the p10 near-zero-margin mass
  measurably faster AND drops exact d_seg faster than plain Muon, AND it survives byte-close + (eventually) paired
  CUDA+CPU re-eval → reactivate Aurora. Otherwise the DEFER stands with an empirical artifact.
- **Cost = $0** (kernel + tests + probe machinery all exist; this is a config-only A/B, no new code, no dispatch).

This is strictly a *falsification* smoke, not a capstone change — it does not gate or touch the live faithful run.

## NO-FAKE ledger
- VERIFIED (web, 2026-06-21): Tilde org has no new optimizer repo since comp-muon (2026-06-05); Aurora explicitly
  targets weight-row-leverage NOT loss-Hessian conditioning, with no batch-scaling / no ill-conditioning convergence
  claim (blog quotes §3); Gram-Space Manifold Muon is theory-only (no code) and its variants RELAX condition-number-1;
  Parallax is an attention primitive on the Tilde index but coded under Yifei-Zuo (not an optimizer).
- BANKED (prior surveys, re-used not re-derived): Aurora reduces to Muon on our 11-WIDE/0-TALL/sin partition
  (test diff 0.0); comp-muon inapplicable (no attention); plain Muon is MLX-live + canonical.
- REASONED (against the deep-math): loss-κ ⊥ weight-row-leverage ⇒ no Tilde variant beats plain Muon at κ-busting;
  shallow boundary is perturbation-fragile ⇒ argues for plain Muon's controlled step; batch-invariance is a
  plain-Muon property.
- NOT claimed: no score moved; pointer UNMOVED 0.19110; all Tilde perf numbers are `[external-claim]`; the §5 A/B is
  a $0 falsification smoke, not run here, predicted-null.

## 6-hook wire-in (Catalog #125; research/advisory memo)
- **#1 Sensitivity-map / #2 Pareto / #3 Bit-allocator:** N/A — optimizer survey, no per-axis byte sensitivity,
  non-binding on the rate/seg/pose feasible region, not a codec primitive.
- **#4 Cathedral autopilot dispatch:** N/A — research_only; the §5 A/B (if ever run) is operator-routed.
- **#5 Continual-learning posterior:** PENDING — emit an anchor only if the §5 stage-8 A/B produces an empirical
  margin/d_seg trajectory (none run here).
- **#6 Probe-disambiguator:** the §5 stage-8 A/B (re-readout on the flipped-margin distribution) **is** the
  disambiguator for "does any Tilde variant beat plain Muon at κ-busting" — resolved empirically; the conditioning
  argument is the provisional verdict until run.

`council_predicted_mission_contribution: frontier_protecting` — this memo prevents a low-EV optimizer detour
(adopting Aurora/Gram-Space/comp-muon against the κ-shallow-boundary problem they do not address) and keeps the
capstone on plain Muon, the principled κ-buster, per the just-derived conditioning math.

## Sources
- Tilde org (GitHub): https://github.com/tilde-research
- Tilde research index: https://tilderesearch.com/research
- Aurora blog: https://blog.tilderesearch.com/blog/aurora · repo (MIT): https://github.com/tilde-research/aurora-release
- Compositional Muon blog: https://blog.tilderesearch.com/blog/compositional-muon · repo (Apache-2.0): https://github.com/tilde-research/comp-muon-release
- Gram-Space Manifold Muon (vignette, theory-only): https://blog.tilderesearch.com/vignettes/gram-space
- Parallax (NOT Tilde-org optimizer): https://arxiv.org/abs/2605.29157 · https://github.com/Yifei-Zuo/Parallax
- Muon (Keller-Jordan): https://kellerjordan.github.io/posts/muon/
- Preconditioning Benefits of Spectral Orthogonalization in Muon (theory anchor): https://arxiv.org/pdf/2601.13474
- Adjacent NON-Tilde κ-targeted Muon work (out of scope): Mousse https://arxiv.org/abs/2603.09697 · Muon2 https://arxiv.org/html/2604.09967v1
