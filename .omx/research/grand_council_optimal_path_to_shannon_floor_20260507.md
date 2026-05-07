# Grand Council — optimal path to lowest possible score (no constraint deliberation)

**Date**: 2026-05-07T19:35Z
**Trigger**: user directive — "consult the grand council now what are the optimal designs using everything we know and have developed... determine the true path forward to lowest score possible no time limit or expense constraints or coding time constraints"
**Council**: inner-ten + grand-council on demand
**Mandate**: synthesize the entire cathedral + all research queues + alternative paradigms into one canonical design that targets the lowest contest score achievable.

## Frozen facts

- **Contest score**: `S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489`
- **Public medal band**: PR101 (gold) 0.193, PR103 (silver) 0.195, PR102 (bronze) 0.195 — all on PR100 substrate
- **Our anchor**: PR103-on-PR106 standalone 0.20898 `[contest-CUDA T4]`, 5/5 gates GREEN
- **Gap decomposition** (PR103-on-PR106 vs PR101 gold): pose=69%, rate=31%, seg≈0%
- **Shannon floors on PR106 substrate**: H₀=167,570 B, H₂=88,983 B → 1.91× context-aware headroom
- **Yousfi-Fridrich floor** (paper §6.7): strictly below Shannon for task-aware coding through scorer blind spots
- **Importance-flip threshold**: d_pose = 2.5e-4 (analytic); below = pose-dominated regime
- **Realistic Shannon floor estimate**: ≈ 0.155 (CLAUDE.md grand council Wave-Ω derivation)

## The optimal system (bilevel optimization)

The lowest-score-achievable system is a **bilevel optimization** with three nested loops:

```
OUTER:  δεζ joint training reshapes substrate θ to maximize score-coupled compressibility
MIDDLE: meta-Lagrangian search over (codec, architecture, paradigm) atoms
INNER:  Joint-ADMM allocates rate-distortion budget across (decoder, latent, mask, pose)
                                                            streams at fixed substrate
```

Each layer feeds the next:

- **Inner ADMM** produces a Pareto-optimal per-stream byte allocation given the
  current substrate + atom set.
- **Middle meta-Lagrangian** picks atoms that move the Pareto frontier inward
  (per `tools.meta_lagrangian_search_cli`).
- **Outer δεζ training** updates the substrate so that atoms which currently
  lose become wins (e.g., AC regression on tensors 0/2/4 disappears when
  weights are trained to be conditionally uniform, per Path B Shannon analysis).

This is the **Yousfi-Fridrich-floor-targeting** structure: outer training
reshapes the substrate to live closer to the YF floor; middle/inner pick the
codec configuration that realizes the gain.

---

## Inner-ten council positions (all 10, expressive)

### Shannon (LEAD — information theory)

The contest's R(D) function is composite:

  R(D) = R_seg(d_seg) + R_pose(d_pose) + R_compose(d_seg, d_pose)

where R_compose captures the joint coding gain. **Today** we operate at
brotli ≈ 1.015× H₀ — within 1.5% of the i.i.d. Shannon floor. The 1.91×
headroom to H₂ is the **provable** rate prize from any context-aware
coder. Beyond H₂ lies the scorer-task-aware floor: bytes spent encoding
patterns the scorer cannot see are wasted bytes (this is the YF-floor
inequality `R_YF(D) ≤ R_Sh(D)` that's strict whenever the scorer has
blind spots).

**Optimal action**: train substrate to push R toward R_YF, not R_Sh.
Operationally that means joint training with the **scorer-task-aware
loss** alongside the rate proxy — exactly what δεζ training does.
Predicted floor at `d_seg ≈ 0.0006, d_pose ≈ 1e-5, B ≈ 80,000`:
**S ≈ 0.060 + 0.010 + 0.053 ≈ 0.123**. That's 35% below the current
public medal band.

**ENDORSE bilevel optimization. Train against R_YF, not R_Sh.**

### Dykstra (CO-LEAD — alternating projections / convex optimization)

The achievable region in `(d_seg, d_pose, R)` space is the convex hull
of empirical (θ, score) points. The **Pareto frontier** is the lower
envelope. Tracing it requires sweeping multipliers and solving the inner
Lagrangian each time — exactly what `tools.apogee_intN_pareto` already
does for one axis.

**Optimal action**: extend the Pareto tracer to a 3-axis sweep over
(λ_seg, λ_pose, λ_rate) using the existing
`tac.joint_admm_coordinator.run_admm`'s adaptive-ρ. The achievable
region's vertices are the candidate atoms for the meta-Lagrangian.

**Bilevel framing**: outer-loop δεζ training is itself an alternating
projection — between (a) feasible-distortion set defined by the renderer
+ scorer, and (b) feasible-rate set defined by the codec + archive
budget. The fixed point is the rate-distortion optimum.

**ENDORSE. The cathedral's `joint_admm_coordinator` is exactly the
inner-loop machinery; we just need the outer-loop training driver.**

### Yousfi (steganalysis / contest scorer architect)

The contest IS inverse steganalysis. The right codec hits the YF-floor
by exploiting the scorer's **blind spots**:

- **SegNet stride-2 stem** loses half-resolution detail before any class
  computation. Sub-(256,192) artifacts are invisible to SegNet.
- **PoseNet YUV6 + FastViT-T12** coarsens chroma; chroma noise below
  some threshold is invisible to pose head.

**Optimal action**: encode coarse pixels at full fidelity; encode fine
pixels at LOW bits via wavelet residual or SIREN coordinate-MLP. The
encoded artifacts will be invisible to the scorer.

**Per Path-B forensic, pose carries 69% of our gap.** Pose distortion
on PR106 is `3.36e-5`; on PR100 medal band it's lower. Pose-TTO drift
catastrophes (350× MPS-vs-CUDA, eval_roundtrip=False default) are
prime suspects. **Wrap RAFT-derived poses** (Lane FL has existing
dispatch evidence) as the pose-stream ProximalCodec. This alone could
recover 0.011 score points.

**ENDORSE. RAFT poses as Op-PoseStream. Wavelets + SIREN for
SegNet-blind-spot exploitation.**

### Fridrich (DDE Lab / inverse steganalysis)

Cover-source-mismatch (PR101's byte_maps regressing on PR106) is the
canonical steganalysis defeat: a code tuned to one cover source fails
on a different one. **Auto-select algorithms (`auto_select_byte_maps`,
`ac_auto_fallback`) ARE the cover-source-aware codec** — they are
already correct.

The deeper move: **train the substrate to be its own cover source**.
δεζ joint training optimizes weights so that the codec's prior matches
the empirical posterior — the Bayes-optimal codec under MDL.

UNIWARD's textured-region-prior insight maps to: weight quantization
noise should be CONCENTRATED in textured regions (where SegNet's
blind spot lives) and ABSENT from class boundaries (where SegNet
attention concentrates). This is the **score-aware quantization** the
cathedral's β-paradigm sensitivity preprocessing already implements.

**ENDORSE. δεζ training + score-aware quantization (β paradigm) +
auto-select gates = the Fridrich-floor codec.**

### Contrarian (challenges weak arguments)

The bilevel optimization is mathematically optimal but **risks the
"plan no one executes" failure mode** (May 4 race postmortem). Each
loop has a wall-clock cost:

- **Outer δεζ training**: multi-day GPU training run
- **Middle meta-Lagrangian**: hours-to-days per substrate
- **Inner ADMM**: minutes per atom

If we run the FULL bilevel naively, we exhaust GPU budget before any
contest-CUDA score lands. **Don't run the outer loop on synthetic
substrates** — anchor it on the proven PR100 substrate so each outer
iteration produces a deployable archive.

**Demand grouped milestones**:

1. PR100-substrate Op1+Op2+Op2.5 stack → contest-CUDA score, target
   ~0.190 [predicted-band]. Ship-or-fail decision in 1 day.
2. RAFT-poses + Op2 stack on PR100 → contest-CUDA, target ~0.183.
   Ship-or-fail decision in 1 week.
3. δεζ training on PR100 → contest-CUDA, target ~0.165. Ship-or-fail
   decision in 1 month.

**Each milestone must produce a contest-CUDA score**, not just a paper.
DISSENT against any plan that defers contest-CUDA to "after the full
bilevel converges."

### Quantizr (adversarial reverse-engineering)

The leaderboard rewards engineering velocity at this score band. Three
teams hit medal-band by adding 241 LOC bolt-ons to one prior
submission. **Don't out-think them; out-ship them.**

Concrete:

1. Replay PR101's gold codec on PR101's NATIVE substrate to verify we
   reproduce 0.193 [contest-CUDA] — if we do, the codec is correct;
   any drop on PR106 is substrate-side.
2. Compose PR101's split-Brotli + PR103's AC bolt-on + PR102's
   inference tuning on PR101's substrate as one stack → predicted
   <0.190 (orthogonal compositions on shared substrate).

**ENDORSE Path A first**, novel research second. The cathedral's
canonical CodecOp Protocol means each replay is one wrap + one test.

### Hotz (engineering instinct)

The cathedral is fine. Stop adding paradigms; **wire what we have**.
Specifically:

1. `tac.codec_pipeline.Op1_PR101SplitBrotli(auto_select=True)` is the
   default. Ship it.
2. `Op2_PR103ArithmeticCodec(ac_auto_fallback=True)` — landed by
   bug-hunter v1 fix. Ship it.
3. `Op3_ApogeeIntN_Substrate(bits=6)` — already L1, gates ready.
4. `tac.joint_admm_coordinator.run_admm` — production. Plug it in.
5. The Pareto sweep at `tools.apogee_intN_pareto` — extend to 3-axis
   sweep, that's a 50-LOC fork.

**Don't write more code; PROVE the existing code on PR100 substrate.**

### Selfcomp / szabolcs-cs (PR #56 author)

PR #56 hit 0.36 with grayscale-LUT + 8-block SegMap + xz-int8. The
cathedral's α paradigm wraps these but they've never composed with Op1+Op2.
**The right OPTIMAL design doesn't pick HNeRV OR SegMap — it composes
both**: PR101's HNeRV decoder for the renderer's structured weights,
SegMap's grayscale-LUT for masks (which HNeRV's renderer doesn't even
encode!), and a unified arithmetic terminal across both.

The mask-encoder bakeoff (`tac.codec_pipeline_mask`) already wraps four
mask codecs (NeRV, wavelet, VQ-VAE, grayscale-LUT). On PR100/PR106 the
masks live in `masks.mkv`; on Selfcomp's PR56 they live in a 1.24 MB
xz-int8 binary. **Bakeoff on each substrate** to pick the best.

**ENDORSE α-paradigm bakeoff per-substrate**.

### MacKay (memorial seat — IT + Bayesian + MDL)

Under Minimum Description Length, the right metric is **total description
length**: archive bytes + decoder code bytes + runtime metadata. SIREN's
sinusoidal coordinate-MLP saves DECODER bytes (smaller code), which counts
toward archive size in the contest.

For the rate-distortion R(D), the variational hyperprior (Ballé 2018)
provides an end-to-end-trainable entropy model. **Replace the static
Brotli/AC model with a learned hyperprior** trained jointly with the
weights — this is the Ballé prescription that the γ-paradigm's STUB
documents but doesn't yet implement.

**ENDORSE Ballé hyperprior + δεζ joint training as the canonical
neural-compression frontier.**

### Ballé (modern neural compression)

The hyperprior + GDN nonlinearity + scale-conditional Gaussian is the
2018 reference; the 2020s extensions (channel-conditional, autoregressive
priors) push deeper. **Ship the 2018 hyperprior first** as Op-Hyperprior
in the codec_pipeline; it's a drop-in replacement for Op2 with a
learnable entropy model.

The γ-paradigm's STUB on `Op_GammaJointADMM(substrate_aware_init=True)`
is intentionally inert because the underlying `tac.balle_hyperprior_codec`
doesn't yet expose a trainable prior_init API. **Add that API**:

```python
class BalleHyperpriorCodec:
    def fit_prior(self, substrate_state_dict, n_steps=1000): ...
```

Then the γ-paradigm can substrate-tune. Predicted gain on PR100 substrate:
−10 to −15 KB at fixed distortion (the 1.91× headroom from H₀ to H₂).

**ENDORSE Ballé prior-fit API + δεζ joint training.**

### Boyd (grand council — convex optimization at operational level)

Adaptive-ρ ADMM (already in `tac.joint_admm_coordinator`) is convergent
under the 4-stream waterline equilibration condition (`dScore/dByte`
equal across active streams). **The KKT residual is the convergence
witness**: zero residual means we've found the rate-distortion frontier
point at the chosen multiplier.

The bilevel optimization's **outer loop** (δεζ training) breaks
convexity — the substrate-dependent codec is non-convex in θ. But ADMM
under non-convexity still converges to a stationary point under mild
conditions; the council should accept that the global optimum may not
be reachable but the stationary points reachable from PR100/PR106
warm-starts are still better than the manual-engineering local optima.

**ENDORSE bilevel; accept stationary-point convergence; use warm-starts.**

---

## Council verdict (10/10 ENDORSE, 1 PARTIAL DISSENT integrated)

**Unanimous on the bilevel structure**. Contrarian's dissent is
integrated as: each milestone produces a contest-CUDA score, not a paper.

## The canonical optimal design (concrete + buildable)

### Phase 1 — PR100 anchor + Op1+Op2+Op2.5 stack [predicted ~0.190 contest-CUDA]

1. Replay PR101 standalone on PR101's native substrate; verify 0.193
   reproduction.
2. Compose Op1 (auto_select) + Op2 (ac_auto_fallback) + Op2.5 (PR102
   inference tuning) on PR100 substrate → contest-CUDA dispatch.
3. Promote via 5-gate ladder; ship if score ≤ 0.193.

### Phase 2 — RAFT-poses + Path-B forensic [predicted ~0.183]

1. Wrap RAFT-derived poses (Lane FL signal) as
   `Op_RAFTPoseStream(StreamProximalCodec)` per Boyd interface.
2. Replace pose archive in PR100-stack with RAFT poses.
3. Joint-ADMM coordinator allocates byte budget between renderer +
   masks + RAFT poses.
4. Contest-CUDA dispatch; target d_pose drop from 3.36e-5 → 1e-5.

### Phase 3 — δεζ joint training on PR100 [predicted ~0.165]

1. Build training driver that uses
   `tac.codec_pipeline_deltaepszeta_callback` for end-of-epoch
   archive-bytes signal AND `tac.shannon_h2_loss` for in-epoch
   differentiable rate proxy.
2. Loss: `L = score_loss(SegNet, PoseNet) + λ · shannon_h2_loss + ν · render_fidelity`
3. Per-tensor weights from
   `tools.build_deltaepszeta_training_targets`: focus auxiliary loss on
   top 5 `blocks.*` (78.5% of H₀-H₂ headroom).
4. Train on PR100 substrate (warm-start from PR100 weights).
5. Multi-day GPU run on H100 SXM ($2.47/hr × ~48h ≈ $120 budget).

### Phase 4 — Ballé hyperprior + γ paradigm [predicted ~0.155]

1. Add `BalleHyperpriorCodec.fit_prior` API (Ballé council prescription).
2. Wrap as `Op_GammaJointADMM(substrate_aware_init=True)` un-stub.
3. Replace Op2 with γ in the canonical stack.
4. δεζ + γ together: joint substrate + entropy model training.

### Phase 5 — α mask paradigm + wavelet residual + SIREN MLP [predicted ~0.145]

1. α mask bakeoff per-substrate: pick smallest of NeRV/wavelet/VQ-VAE/
   grayscale-LUT for the masks.mkv slot.
2. Wavelet residual sidechannel for SegNet-blind-spot exploitation
   (Yousfi prescription).
3. SIREN coordinate-MLP for decoder-byte-saving (MacKay prescription).
4. Compose all via Joint-ADMM + meta-Lagrangian search.

### Phase 6 — telescopic foveation + Fridrich UNIWARD + RL/ADMM coupling [predicted ~0.135]

1. Telescopic foveation (subagent a124fb91 in flight) decorates renderer
   output with high-fidelity center / low-fidelity periphery.
2. UNIWARD textured-region weighting moves quantization noise to
   SegNet-invisible regions.
3. RL/PufferLib (per Council #271) explores the meta-Lagrangian atom
   space.

### Phase 7 — sub-Shannon territory [predicted 0.125 - 0.155 YF floor]

1. SJ-KL basis (score-Jacobian Karhunen-Loève) for task-aware residual
   coding; Wave-Ω-V3 launch-ready.
2. arXiv 2604.24658 paradigm research (subagent F).
3. Cosmos / Transfer 2.5 / FP4 recipe integration.
4. Telescope every paradigm wrap into one canonical archive.

## Predicted score trajectory

| Phase | Target | Cumulative GPU $ | Wall-clock | Risk |
|---|---:|---:|---:|---|
| 1 | 0.190 | $5 | 1 day | low (canonical-winner replay) |
| 2 | 0.183 | $25 | 1 week | medium (RAFT integration) |
| 3 | 0.165 | $145 | 1 month | medium (δεζ training stability) |
| 4 | 0.155 | $250 | 1.5 months | high (Ballé prior-fit API) |
| 5 | 0.145 | $400 | 2 months | high (paradigm composition risk) |
| 6 | 0.135 | $700 | 3 months | very high (RL exploration) |
| 7 | 0.125-0.155 | $1500+ | 4-6 months | research-grade |

## Integration + wiring (what gets built next)

1. **`tools/run_bilevel_optimization.py`** — outer δεζ training loop
   with cathedral-aware archive-size signal.
2. **Extend `tools/apogee_intN_pareto.py`** to 3-axis sweep over
   (λ_seg, λ_pose, λ_rate).
3. **Extend `tools/meta_lagrangian_search_cli.py`** atom ledger to
   include cathedral CodecOp variants.
4. **Wire `contest_score_marginals`** into
   `joint_admm_coordinator.run_admm`'s adaptive-ρ update so the
   coordinator queries operating-point-aware sensitivities each
   iteration.
5. **Add `BalleHyperpriorCodec.fit_prior` API** (Ballé prescription).
6. **Wrap RAFT-derived poses** as `Op_RAFTPoseStream`.
7. **Wrap MAE-pretrained backbone** when GPU returns.
8. **Wrap SIREN coordinate-MLP** as decoder-byte-saving alternative.

## How this consolidates everything

| Existing primitive | Role in optimal design |
|---|---|
| `tac.codec_pipeline.CodecOp` | substrate-codec interface |
| `tac.codec_pipeline*` (4 paradigms + Op4) | atom set for meta-Lagrangian |
| `tac.joint_admm_coordinator` | inner ADMM loop |
| `tac.shannon_h2_loss` | δεζ training rate proxy |
| `tac.contest_rate_distortion_system` | contest-objective coupling |
| `tac.codec_pipeline_deltaepszeta_callback` | epoch-boundary signal |
| `tools.apogee_intN_pareto` | Pareto frontier tracer |
| `tools.meta_lagrangian_search_cli` | atom-ledger search |
| `tools.build_deltaepszeta_training_targets` | per-tensor weight bridge |
| `tools.contest_score_gap_decomposition` | Path-B forensic |
| `experiments/run_cathedral_autopilot.py` | end-to-end Pareto demonstrator |
| `experiments/run_paradigm_chorus.py` | paradigm health check |
| Alternative paradigms (RAFT/MAE/SIREN/etc.) | new atoms for ledger |

The cathedral consolidates into the bilevel optimization as: **each
existing primitive is a node in the dataflow**. Outer δεζ updates θ;
middle meta-Lagrangian picks atoms from the cathedral's `CodecOp`
registry; inner ADMM allocates rate budget; contest_score_marginals
guides the operating-point-aware sensitivity oracle each iteration.

## What is intentionally NOT in this design

- **Token efficiency for the agent**: this is a no-constraint
  deliberation per user mandate. Council has not optimized for agent
  cost.
- **Single-archive shipping**: each phase produces ONE archive that
  gets contest-CUDA evaluated. We do NOT skip phases or batch.
- **MNeRV / CLADE / SPADE / LA-pose**: Contrarian's dissent retained
  these as deferred (speculative or limited-leverage). Reconsider after
  Phase 4.

## Cross-references

- Cathedral: `.omx/research/INDEX_session_2026_05_07_codec_pipeline_canonicalization.md`
- Composition contract: `.omx/research/four_way_stack_composition_contract_20260507_claude.md`
- Path-B pose forensic: `tools/contest_score_gap_decomposition.py`
- Dual-path memo: `feedback_pr106_substrate_is_below_medal_band_20260507.md`
- Alternative-paradigms queue: `feedback_alternative_paradigms_research_queue_20260507.md`
- δεζ session memos: `feedback_canonical_codec_pipeline_session_complete_20260507.md`
- May 4 race postmortem: `feedback_may_4_hnerv_race_postmortem_20260505.md`
- Yousfi-Fridrich floor (paper §6.7): `docs/paper/06_related_work.md`

## Authoritative verdict

**The council unanimously prescribes the bilevel optimization** with
Phases 1-7 trajectory and the integration wiring above. The lowest
score achievable in the YF-floor regime is 0.125-0.155 [predicted].
Phase 1 (PR100 canonical-winner replay) is the next concrete dispatch
when GPU billing is available; everything else flows from there.

---

## Codex senior-engineering review addendum

**Verdict**: keep this memo as a strategic synthesis, but do not treat it as
an executable or evidence-complete control plane until the following
corrections are satisfied.

### Required corrections before canonical execution

1. **Evidence-grade all score numbers.**
   The PR103-on-PR106 anchor is exact CUDA T4, but the strict contest formula
   score is `0.2089810755823297`; the report-reconstructed score is
   `0.20898105277982337`. Roadmaps, inventory rows, and proof gates should
   cite both only when they also explain the rounded-rate delta. Phase targets
   such as `0.190`, `0.165`, `0.155`, `0.145`, and `0.125` remain
   `[prediction]` until exact archive eval exists.

2. **Do not encode 0.155 as a floor or ceiling.**
   The current mandate is lowest score possible with no artificial lower
   limit. Treat `0.155` as a useful waypoint from earlier Wave-Omega council
   reasoning, not a stopping condition and not a proven lower bound.

3. **Replace loose "sub-Shannon" wording with task-aware rate-distortion.**
   The scientifically precise claim is: for the contest scorer, a task-aware
   sufficient-statistic code can beat generic pixel/video/independent-entropy
   proxy rates by not spending bits on scorer-invisible information. It is not
   evidence for violating an information-theoretic Shannon bound once the
   correct sufficient statistic and distortion functional are specified.

4. **PR100-107 1:1 reproduction is Phase 0.**
   Before using PR100/101/102/103/105/106/107 as stack atoms, each must have:
   archive SHA, member SHA, source/runtime custody, README/report capture,
   `compress`/`inflate` behavior, binary grammar notes, decode/re-encode
   parity status, no-op controls where relevant, and smallest missing proof.
   Detached public clones are forensic inputs only; promoted artifacts land on
   `main` through explicit review.

5. **Phases should run in parallel where write scopes are independent.**
   The sequential table is useful for narrative, but wall-clock optimal work is
   parallel: PR100-107 custody, delta-epsilon-zeta target-to-candidate bridge,
   alpha categorical/VQ mask bridge, PR91/HPM1 categorical recovery, HDC2/HDM3
   entropy closure, and RAFT/LA/telescopic foveation runtime proof can advance
   concurrently with disjoint files.

6. **Do not prematurely defer CLADE/SPADE/LA-pose.**
   The memo's "intentionally not in this design" section is too strong. These
   lanes should be categorized as active sidecars pending runtime-visible
   proof, especially categorical labeling/self-compression, CLADE/SPADE-style
   semantic priors, RAFT-derived pose, LA-pose, and telescopic foveation. They
   should not block the delta-epsilon-zeta path, but they remain candidates for
   stacking or substitution.

7. **Every phase needs a dispatchable artifact contract.**
   A phase is not complete when a model, memo, or callback exists. It is
   complete when it emits a deterministic byte-different candidate archive or
   a fail-closed blocker artifact with: old/new SHA-256, charged bytes,
   runtime tree SHA, changed payload proof, no-op control, lane-claim status,
   strict compliance JSON, and exact CUDA readiness.

### Revised immediate implementation order

1. Stabilize and commit the current PR103-on-PR106 strict-score anchor,
   categorical label atoms, entropy-frontier selector, H2 target audit, and
   preflight hardening.
2. Build the PR100-107 reproduction/deconstruction ledger so every public
   frontier atom has byte-level custody and a precise missing-proof row.
3. Build the delta-epsilon-zeta PR106 candidate bridge: consume
   `targets.json`, prove nonzero H2 pressure on top tensors, emit changed-byte
   requirements and fail-closed readiness until a trained payload/archive
   exists.
4. In parallel, push the largest stackable sidecars:
   alpha VQ/categorical mask replacement, PR91/HPM1 categorical parity,
   HDC2/HDM3 entropy closure, RAFT/LA/telescopic foveation runtime consumer,
   and Balle/JCSP runtime parity.
5. Extend meta-Lagrangian/Pareto to rank only artifacts with explicit custody
   state, evidence grade, expected information gain, and KKT/Pareto status.

This addendum supersedes roadmap wording that treats predictions as measured
facts, treats 0.155 as a hard floor, or defers semantic/pose/foveation sidecars
without runtime-visible proof.
