# Tilde optimizers vs the #75 inert-loop grad pathology — research verdict (task #77)

- **Date:** 2026-06-10 (UTC) · **Agent:** claude (subagent `task77_tilde_optimizers`)
- **Lane lineage:** `lane_tilde_opt_aurora_20260609` (research_only) · feeds #76 (loop fix) + #74/#68 (retraining)
- **research_only:** `true`. Every Tilde performance/quality number is `[external-claim]`
  (Tilde blog / GitHub README / MarkTechPost). Any MLX A/B is `[macOS-MLX research-signal]`.
  `score_claim=false`, `promotable=false`, `promotion_eligible=false`,
  `ready_for_exact_eval_dispatch=false`. Only `upstream/evaluate.py` on paired CUDA +
  Linux-x86_64 CPU produces a contest score. This memo recommends an experiment; it claims no win.
- **Authority tier:** mechanism/research. `mechanism_update_eligible=true`,
  `score_roadmap_update_eligible=false`.

---

## LEAD ANSWER

**Does a Tilde optimizer (compositional-muon / parallax / aurora) plausibly fix the #75 inert-loop
grad pathology? — PARTLY YES, but the fix is plain MUON, NOT the three named variants.**

- **Y — Muon (Keller-Jordan; the Tilde lineage's root, and PR95's own final-stage optimizer)** plausibly
  fixes the #75 pathology. **Mechanism:** the #75 failure is exploding/ill-conditioned gradients
  (`grad_norm` 5e4→6.8e6, **whole-tree clip to global-norm 1.0 fires 100% of steps**, `loss_seg` drifts
  UP, exact d_seg pinned 0.50). Muon's Newton-Schulz orthogonalization sets **all singular values of the
  update to ≈1**, so the **update magnitude is independent of the incoming gradient norm**
  (`[external-claim]`, Keller-Jordan / Bernstein / arXiv:2601.13474). That is precisely the property the
  inert loop lacks: it would replace the microscopic ~1e-7-scaled AdamW step with a controlled,
  well-scaled spectral step every step, regardless of whether the raw grad norm is 5e4 or 6.8e6.
- **N — compositional-muon, parallax, aurora specifically do NOT add value over plain Muon for THIS
  pathology** on our HNeRV decoder:
  - **compositional-muon** optimizes transformer **attention QK/OV factor pairs**; HNeRV has no
    attention → structurally inapplicable (cannot run).
  - **parallax** is **not an optimizer** — it is an attention primitive (Local Linear Attention),
    owned by `Yifei-Zuo`, not the Tilde org → inapplicable.
  - **aurora** fixes Muon **row-norm anisotropy / neuron-death on TALL matrices** under activations with
    `φ(0)=0 ∧ φ'(0)≈0`. Our Muon-eligible partition is **11 WIDE, 0 TALL, 0 SQUARE** and HNeRV uses
    **sin** (`sin'(0)=1≠0`, precondition broken). For wide/square, Aurora **reduces to Muon** (verified
    `max diff 0.0` in our own test `test_square_reduces_to_muon_polar`). So Aurora ≈ Muon here; it does
    not address grad-norm explosion (its fix is anisotropy, **not** raw gradient magnitude — Tilde blog,
    quoted §2.3).

**The decisive new evidence (reframes the prior memo):** the #75 inert run **was AdamW, not Muon.**

---

## 1. The decisive new finding: the inert run never actually ran Muon on the decoder weights

The #75 audit (`pr95_elephant_audit_20260610T185556Z.md`) describes the pathology as "AdamW + hard-clip-
to-1.0 stalled." The B1 inert run's own `training_artifact.json`
(`/Volumes/VertigoDataTier/pact/b1_229k_clean_20260609T085348Z/training_artifact.json`) confirms it
**verbatim**:

```json
"optimizer_class": "adamw",
"muon_active": false,
"pr95_muon_policy": "faithful_stage8_only",
"optimizer_kind_by_group": {
  "matrix_decoder_weights": "adamw",   <-- the hidden conv weights trained on AdamW
  "stem_rgb_head": "adamw", "latents": "adamw",
  "biases_norms_scalars": "adamw", "entropy_qat_params": "adamw"
}
```

- The launch manifest's `pr95_muon_policy = "faithful_stage8_only"` means **Muon activates ONLY in stage 8**
  (the final stage). Through stages 1–7 — i.e. the entire span where d_seg is pinned at 0.50 and grad_norm
  explodes (ep0→ep2750 in the #75 table) — the 191,104 "matrix_decoder_weights" were optimized by **AdamW**.
  `ep250_campaign_decision.json` confirms `"muon_active_at_ep249": false`.
- This is **source-faithful to PR95** (PR95 also runs AdamW stages 1–7, Muon only stage 8 — per PR95 L14/L15).
  But PR95's grad norms stay *sane* under AdamW because **PR95's loss is well-conditioned**. Ours produces
  `grad_norm` ≈ 1e6 under the identical AdamW configuration → the inert loop is an AdamW conditioning failure
  in stages 1–7, **before** Muon is ever scheduled.

### Why the clip makes AdamW inert (the exact arithmetic)

The live MLX clip (`src/tac/local_acceleration/pr95_hnerv_mlx.py:2440`) is a **global-norm** clip:

```python
scale = mx.minimum(mx.array(1.0), mx.array(float(max_norm)) / (norm + 1e-6))   # max_norm = 1.0
```

When `norm = 6.8e6` and `max_norm = 1.0`, `scale ≈ 1.47e-7`. The **entire** AdamW update (already direction-
normalized by AdamW's per-coordinate `1/√v`) is then multiplied by ~1.5e-7 → an effectively zero step in a
direction that, at 1e6 grad norm, is dominated by whatever few coordinates are exploding. The objective is
COMPUTED; the optimization is INERT. (`STRICT_SCRUTINY_ep614_finding.md`: "grad-clip fires 100% of steps;
nan_inf=0 — no actual divergence (containment holds)." Containment ≠ progress.)

### Why Muon structurally breaks this (the mechanism that matters)

Muon's update is `−lr · scale · NewtonSchulz5(momentum)` where NS drives the singular values of the
(momentum) matrix to ≈1. So:

- the **output magnitude does not depend on the input gradient norm** — a 5e4-norm gradient and a 6.8e6-norm
  gradient both produce an orthogonalized step of comparable, controlled spectral scale (`[external-claim]`,
  WebSearch corpus: "This normalization of singular values to 1 means the update direction is independent of
  the gradient norm"; "naturally mitigates exploding gradient problems since the spectral norm of updates is
  controlled by the orthogonalization process").
- the global-norm clip therefore **rarely fires** (the post-NS update norm is ~`max(1,√(m/n))`, O(1)), so the
  effective step is the real spectral step, not a 1e-7 fraction of it.
- Muon's convergence is governed by the **spectral-norm** Lipschitz constant (arXiv:2601.13474,
  "Preconditioning Benefits of Spectral Orthogonalization in Muon"), which on an ill-conditioned matrix loss
  can be far smaller than the Euclidean one — exactly the regime where AdamW + Euclidean-norm clip stalls.

**This is a genuinely different conclusion from the prior `tilde_research_optimizers_survey_and_aurora_build`
memo**, which assessed Aurora-vs-Muon *marginal* gains and predicted-null because vanilla Muon "suffices."
That framing implicitly assumed Muon was already running and working. It was not running during the inert
span. The high-EV move is not "Aurora vs Muon" — it is **"Muon (or sane conditioning) vs the AdamW stages-1–7
that are actually inert."**

---

## 2. What each Tilde method is (cited) + whether it addresses the #75 pathology

| Method | What it precisely is | Source (primary) | Addresses #75 grad pathology? |
|---|---|---|---|
| **Muon** (root; PR95 L15) | MomentUm Orthogonalized by Newton-Schulz: `W ← W − lr·max(1,√(m/n))·NS5(momentum)`; NS5 = 5-step quintic driving all singular values→1. | kellerjordan.github.io/posts/muon; Bernstein "Deriving Muon"; arXiv:2601.13474 | **YES (mechanistically).** Update magnitude is grad-norm-independent → defeats the exploding-grad / clip-to-1e-7 inert mode. This is the relevant fix. |
| **aurora** (Tilde, MIT, `aurora-release` @ `7303d8cb`, updated 2026-05-27, 144★) | Leverage-aware optimizer: alternating projection onto Stiefel ∩ row-oblique manifold (damped diagonal preconditioner, `pp_iterations=2`, `pp_beta=0.5`) so the orthogonalized update has uniform row leverage on **tall** matrices. | blog.tilderesearch.com/blog/aurora; github.com/tilde-research/aurora-release | **NO marginal value here.** Fixes row-norm **anisotropy** (NOT gradient magnitude — quoted §2.3), only on **tall** matrices, only under `φ(0)=0∧φ'(0)≈0`. HNeRV = all-wide + sin → Aurora **reduces to Muon** (verified diff 0.0). It does not add to Muon's grad-norm-independence. |
| **compositional-muon** (Tilde, Apache-2.0, `comp-muon-release`, updated 2026-06-05, 20★) | Partner-whitened Muon for transformer attention: whitens each of QK / OV by the inverse-Gram-root of its partner before/after the spectral sign. Public API is `cm_qk(q_proj, k_proj, head_dim, …)` / `cm_ov(v_proj, o_proj, …)` — **requires** q/k/v/o `nn.Linear` + `head_dim`. | blog.tilderesearch.com/blog/compositional-muon; github.com/tilde-research/comp-muon-release | **NO — cannot run.** HNeRV has no attention / no q/k/v/o / no heads. Structurally inapplicable. (Reactivation only if we adopt an attention-bearing witness backend.) |
| **parallax** (NOT Tilde-org; `Yifei-Zuo/Parallax`, MIT) | **Not an optimizer.** Parameterized Local Linear Attention primitive for LMs (`parallax_func(q,r,k,v,scale)` replacing SDPA); CUDA/Triton/Hopper kernels. | arXiv:2605.29157; github.com/Yifei-Zuo/Parallax | **NO — category error.** It is an attention kernel, not an optimizer; produces no orthogonalized update and does not touch the grad pathology. |

Tilde org current state (WebFetch github.com/tilde-research, 2026-06-10): no NEW optimizer release since the
2026-06-03 parallax intake — aurora (2026-05-27) and comp-muon (2026-06-05) are the only two optimizer repos;
the rest are attention/MoE/distillation (nsa, momoe, wall-attention, nitrobrew) — all DEFER (regime mismatch).

---

## 3. MLX-portable / reproducible? Our task-#33 arm re-checked

**Yes — and our prior arm already implements Newton-Schulz orthogonalization in MLX, ready to use now.**

- **`tac.optimization.muon`** — torch `MuonOptimizer` + `zeropower_via_newtonschulz5` (Keller-Jordan 5-step,
  tuned coeffs `(3.4445,-4.7750,2.0315)`) + `partition_params_for_muon`. Tested (`test_muon_optimizer.py`).
- **`tac.local_acceleration.pr95_hnerv_mlx.zeropower_via_newtonschulz5_mlx`** — the **canonical MLX Muon NS
  kernel**, already wired into the live stage-8 dispatch `apply_pr95_mlx_optimizer_step` (this IS the kernel
  the `faithful_stage8_only` policy uses when Muon activates). MLX-native, numpy/torch-portable.
- **`tac.optimization.aurora_mlx`** — source-faithful MLX Aurora kernel
  (`aurora_simple_quintic_polar_mlx` 12-step quintic + `aurora_leverage_uniform_polar_mlx` leverage-uniform
  projection), **26 NO-FAKE tests pass** incl. byte-identical PyTorch-reference parity and the headline
  `test_tall_leverage_uniformity_beats_plain_polar` (row-‖U_i‖² std 0.0059 vs plain-Muon 0.0594). It is a
  drop-in for the NS kernel via the documented wire-in SPEC (`muon_kernel="aurora_leverage_uniform"`), but —
  per §2 — predicted ≈Muon on our all-wide partition.
- **Duplication noted (still open):** `adapter.py::_build_aurora_like_mlx_optimizer` carries a *private*
  `_polar_simple_quintic` (Wave-N+11 `aurora_like` optimizer object) duplicating the kernel; the SPEC routes
  both through the single tested `aurora_mlx` kernel. Low priority for #76 (the fix is Muon, not Aurora).

So Newton-Schulz orthogonalization is **already MLX-portable and live** — the #76 fix does not require new
optimizer code, only a **policy/config change** (run Muon on the matrix-decoder partition during the stages
where AdamW is currently inert, or otherwise fix the conditioning) + the existing kernel.

---

## 4. RECOMMENDATION (feeds #76 loop-fix + #74/#68 retraining)

**Primary (highest EV, $0, cheapest falsifiable): change the optimizer POLICY so the decoder matrices are
optimized by the grad-norm-independent Muon step during the inert span — do NOT reach for Aurora/comp-muon/
parallax.** Concretely, as the #76 loop-fix probe (MLX-local, `[macOS-MLX research-signal]`):

1. **Run the existing canonical MLX Muon NS kernel on `matrix_decoder_weights` from stage 1**, not only
   stage 8 (i.e. relax `pr95_muon_policy` from `faithful_stage8_only` to a Muon-on-hidden-weights-throughout
   policy for the conditioning probe). Keep AdamW for stem / rgb head / latents / 1-D params (Keller-Jordan
   canon, unchanged). This is the minimal change that converts the inert AdamW span into a controlled
   spectral-step span.
2. **The decisive cheap re-probe (per #75 §3, $0, mechanism):** at ep50 / ep250 measure (a) `grad_norm`
   (expect O(1)–O(10) post-NS, clip rarely firing — vs 1e6 today), (b) `loss_seg` **descending** (vs drifting
   UP today), (c) the **exact** live-render d_seg (argmax disagreement, NOT PSNR/proxy) dropping below ~0.50.
   - If grad_norm normalizes AND loss_seg descends AND exact d_seg < 0.50 → **the inert loop is fixed by
     Muon-conditioning**; continue to the #74/#68 retraining.
   - If grad_norm normalizes but exact d_seg stays 0.50 → the defect is the **objective**, not the optimizer
     (margin-surrogate-to-argmax misalignment per #75 hypothesis path), and the loop fix must also re-derive
     the exact margin loss + STE-round + preprocess (#75 fix item 2). Muon-conditioning is necessary but not
     sufficient — record and pivot to the objective.
3. **Do NOT spend on Aurora as the #76 fix.** Aurora ≈ Muon on our partition (verified). Keep the Aurora arm
   as the *separate* DEFER-with-reactivation A/B (stage-8 marginal, arbitrated by the exact evaluator-action
   waterfiller) from the prior memo — it is not the loop fix and not on the #76 critical path. Reactivation:
   only if a future witness backend introduces tall Muon-eligible weights (guarded by
   `test_hnerv_muon_eligible_partition_has_no_tall_matrices`).
4. **comp-muon / parallax: DEFER (regime mismatch), do not wire.** Reactivate only with an attention-bearing
   witness backend.

**For #74/#68 retraining:** once the #76 probe confirms Muon-conditioning makes loss_seg descend, the
retraining should adopt **Muon on the hidden decoder-weight partition throughout the curriculum** (not just
stage 8), with AdamW retained for stem/rgb/latents/biases. This is the optimizer change predicted to let the
score-aware loss actually descend where AdamW + global-norm-clip-to-1.0 stalled. Predicted effect on the grad
pathology `[extrapolation]`: grad_norm collapses from ~1e6 to O(1)–O(10); clip stops firing 100% of steps;
the EMA-best checkpoint moves past the stage-1 `best_epoch000286` it was frozen at. **This is a mechanism
prediction, NOT a score claim** — the exact d_seg trajectory and the paired CUDA+CPU auth-eval on the
byte-closed archive are the only authorities.

### Honest caveat (Assumption-Adversary)

The shared assumption "fixing the optimizer fixes the score" is only PARTLY hard-earned. Muon-conditioning is
near-certain to fix the **exploding-grad / inert-step** half of #75 (the update-magnitude mechanism is
established). It is **unproven** that, once steps are well-scaled, the **margin-surrogate** then drives the
**exact argmax** d_seg down on our carrier — #75 explicitly flags the "loss_seg descends but exact d_seg
stays 0.50" branch. The §4.2 ep50/ep250 exact-d_seg probe is the disambiguator; it must be run before any
retraining-budget commitment. If the optimizer fix alone does not move exact d_seg, the loop fix also needs
the objective re-derivation (#75 fix item 2) and/or the vendor-PR95-loop fallback (#75 fix item 4).

---

## 5. Wire-in pillars (Catalog #125) + bookkeeping

- **#1 Sensitivity-map / #2 Pareto / #3 Bit-allocator:** N/A — optimizer-policy research, no per-axis byte
  sensitivity, non-binding on the rate/seg/pose feasible region, not a codec primitive.
- **#4 Cathedral autopilot dispatch:** N/A — research_only; the #76 probe is operator/B1-routed.
- **#5 Continual-learning posterior:** PENDING — emit an anchor only when the §4.2 ep50/ep250 exact-d_seg
  probe produces an empirical grad_norm/loss_seg/d_seg trajectory (none run in this research pass).
- **#6 Probe-disambiguator:** the §4.2 ep50/ep250 exact-d_seg measurement **is** the disambiguator between
  "optimizer-conditioning fixes the loop" vs "the objective is still misaligned" — resolved empirically, not
  by prose.
- **No code edited, no dispatch, no clone** in this pass (the Tilde clone + MLX kernels + tests already exist
  from task #33). Disk hygiene: nothing new written to SSD/tmp; this memo is the only durable artifact.

## Cross-refs
`pr95_elephant_audit_20260610T185556Z.md` (#75 — the inert-loop audit this memo answers) ·
`tilde_research_optimizers_survey_and_aurora_build_20260609.md` (the prior task-#33 Aurora build/survey;
this memo REFRAMES its predicted-null from "Aurora≈Muon marginal" to "the inert span was AdamW, Muon's
grad-norm-independence is the fix") · `comp_muon_release_applicability_research_20260609.md` (comp-muon
inapplicable) · `codex_findings_tilde_research_parallax_nerv_intake_20260603T174229Z_codex.md` (parallax is
not a Tilde-org optimizer) · `src/tac/optimization/{muon,aurora_mlx}.py` +
`tac.local_acceleration.pr95_hnerv_mlx.zeropower_via_newtonschulz5_mlx` (the live MLX NS kernel) ·
`/Volumes/VertigoDataTier/pact/b1_229k_clean_20260609T085348Z/training_artifact.json` (the
`optimizer_class=adamw / muon_active=false` evidence).

## Sources (external, tagged `[external-claim]`)
- Aurora blog: https://blog.tilderesearch.com/blog/aurora
- Aurora repo (MIT, `7303d8cb`, 2026-05-27): https://github.com/tilde-research/aurora-release
- Compositional Muon blog: https://blog.tilderesearch.com/blog/compositional-muon
- comp-muon repo (Apache-2.0, 2026-06-05): https://github.com/tilde-research/comp-muon-release
- Tilde org: https://github.com/tilde-research
- Muon (Keller Jordan): https://kellerjordan.github.io/posts/muon/
- Deriving Muon (Bernstein): https://jeremybernste.in/writing/deriving-muon
- Preconditioning Benefits of Spectral Orthogonalization in Muon: https://arxiv.org/pdf/2601.13474
- Parallax (Yifei-Zuo, NOT Tilde org): https://arxiv.org/abs/2605.29157 · https://github.com/Yifei-Zuo/Parallax
