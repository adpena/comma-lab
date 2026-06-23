# Muon-jump conditioning deep-math + re-route staging (the 1e-3-verdict prediction) — 2026-06-23

**Status:** `[contest-CPU advisory]` NON-PROMOTABLE · `research_only=true` · ANALYSIS-ONLY ($0,
NO new training, reuse measured data) · pointer **UNMOVED 0.19110** · NO score claim · NO kill.
**Subagent:** `reroute_deepmath_20260623`. **Sister memos:**
`muon_vs_adamw_from_stage4_convergence_arm_20260622.md` (probe-2, N=16),
`APPARATUS_FIX_muon_lr_floor_resume_20260622.md` (the BUG-B floor fix).

> This is a **prediction to TEST**, not a verdict. The decisive arbiter is the live 1e-3 run's
> real 600-pair Muon d_seg slope. NO premature kill of any of the three hypotheses.

---

## 0. The measured state (reused, not re-run)

| source | fact |
|---|---|
| probe-2 `result_higher.json` (N=16, Muon 3e-3 / AdamW 3e-4) | Muon Δd_seg_quant **−0.001404** vs AdamW **−0.001065**; gap **−0.000340** (≈6.8× the 5e-5 discrimination band); Muon strictly below at every step, gap **widening** −0.000258→−0.000280→−0.000353 |
| probe-2 `result_stage8.json` (N=16, faithful Muon 2e-4 / AdamW 1e-5) | Muon Δd_seg_quant **−0.000437** vs AdamW **+0.000284**; gap **−0.000721**. At faithful LR **AdamW does NOT descend d_seg (it rises post-quant); Muon does** — the cleanest conditioning contrast |
| live faithful run (`yousfi_r3_taper_marginhinge_e5_20260620`, summary) | 600-pair d_seg **0.002075** PARKED across stages 2–5 (all AdamW); stage_index=4 (`stage5_c1a_l7`), epoch_in_stage 5050; paused/resumable; train MPS, authority CPU |
| live disambiguation run (`yousfi_r3_MUONJUMP_stage8_lr1e3_…`, PID 16938) | Muon jump at **1e-3** forked from the stage-5 jump point; LIVE; flat-on-d_seg over the early peak-LR window per the task brief |
| scheduler (`driver.py:1827`,`1852`) | shared-λ floor `eta_min=lr_floor_ratio/adamw_lr = 5e-6/1e-5 = 0.5×peak` (BUG-B: Muon stuck at ≥0.5× its large peak). Fixed Muon floor `= max(5e-6/muon_lr, 1e-3)` → abs floor **5e-6** for muon_lr∈{2e-4,1e-3}; both anneal to the fine-polish regime |
| targets (this run's measured pose+rate 0.108) | **pointer-beat: d_seg < 8.31e-4** (2.50× from 0.002075); **sub-0.15: d_seg < 4.20e-4** (4.94×) |

---

## 1. Why N=16 Muon descended but 600-pair (2e-4) is flat — the spectral-step-alignment mechanism

Muon orthogonalizes each per-layer gradient: replace `G` by its polar factor `UV^T` (Newton-Schulz,
5 iters), i.e. **set all singular values to 1**, keep singular *vectors*. One Muon step on a conv weight
is `W ← W − η·UV^T` — a maximally-conditioned step (κ=1) in the gradient's row/column subspace.

The d_seg gradient is the SegNet argmax-flip signal pulled back to decoder weights. It is a **sum over the
boundary-flip pixel set**: `G = Σ_p g_p`, where each `g_p` is the per-pixel-flip outer-product contribution.
Decompose the aggregate gradient by its cross-pair / cross-pixel **alignment**:

- **Coherent flip set (N=16).** The 16 fixed pairs have a SMALL flip set whose per-pixel `g_p` are
  highly aligned (the same handful of boundary segments flip across the memorized pairs). `G ≈ N·ḡ`
  is **effectively low-rank** (effective rank `r_eff` ≈ a few): its leading singular vectors ARE the
  flip direction. The orthogonalized step `UV^T` then **aligns with the flip-reduction direction**, so
  one Muon step buys a large d_seg drop — and it is **LR-saturated-but-not-LR-limited**: even the
  faithful 2e-4 descends (probe-2 stage8: Muon −0.000437 while AdamW +0.000284). Muon bites because
  the *direction* is right and orthogonalization removes the κ≈19 stretch AdamW's diagonal can't.

- **Diverse flip set (600 pairs).** The full set's flip pixels span many road-scene boundaries; the per-pair
  `g_p` point in **conflicting** decoder-weight directions. `G = Σ g_p` suffers **destructive averaging**:
  the aggregate is HIGHER effective-rank with NO dominant flip axis (`r_eff` large, leading σ-gap small).
  Orthogonalizing a high-`r_eff`, near-isotropic `G` produces a step that is a **whitened average** —
  it spreads the (small) η across many singular directions, each getting `η/√r_eff`-scale motion, much of
  it canceling across pairs. The d_seg-reduction component of the orthogonalized step is then
  **suppressed by 1/(cross-pair alignment)**, and at faithful 2e-4 it can fall *below* the per-step
  d_seg noise → reads FLAT.

**Formalize the discriminant.** Let `a = ‖Σ_p g_p‖ / Σ_p‖g_p‖` ∈ (0,1] be the cross-pair *alignment*
(a=1 perfectly coherent, a→0 isotropic). The orthogonalized step's first-order d_seg reduction scales as

```
Δd_seg_Muon  ∝  η · a · σ̄,      with σ̄ ≈ ‖G‖/√r_eff   (whitened scale)
```

so for the SAME η, **N=16 (a≈1, r_eff small) gets a large drop; 600 (a small, r_eff large) gets a
small drop**. This is the SINGLE mechanism that produces "Muon descends on 16, flat on 600 at 2e-4."

**Crucially, this also says how to break the flatness:** since `Δd_seg_Muon ∝ η·a`, a smaller `a` (600
pairs) is **partially recoverable by larger η**. Muon's orthogonalization already removes the κ≈19
conditioning penalty AdamW pays — the residual loss is the alignment factor `a`, which is **linear in
η to first order**. So a higher Muon LR should move the 600-pair d_seg *unless* `a` is so small that the
needed η destabilizes the shared trunk (Muon-chaos regime) or the basin is genuinely exhausted (capacity).

This is **memorization-flavored but NOT a pure artifact**: the N=16 *absolute* drop is optimistic (small
fixed set is easy), yet the *mechanism* (orthogonalization + alignment) is real and operates on 600 too —
just at reduced `a`. The probe-2 caveat (small-N memorization bias) bounds the *magnitude*, not the
*existence*, of the 600-pair Muon bite.

---

## 2. THE PREDICTION for the live 1e-3 verdict

**Prediction: at Muon LR 1e-3 the 600-pair d_seg DESCENDS — modestly, NOT to the N=16 rate — with a
slope that is SMALL early (peak-LR, where the trunk is hottest and the whitened step is most diffuse) and
STEEPENS as the cosine anneals the LR toward the 5e-6 floor and the alignment-limited step becomes a clean
fine-polish.** Concretely I expect d_seg to leave the 0.00207 park within the peak-LR→mid-anneal window
and trend toward (but likely not fully reach in one stage) the **8.31e-4 pointer-beat** line; the
*direction* should be unambiguously down by the time the LR is in the lower half of its cosine.

Falsifiable signature (advisory, on the live CPU-authority eval curve):
- **DESCENDS (prediction holds):** d_seg drops below ~0.00190 (≈ outside the live run's parked band +
  the ~5e-5 discrimination band) within the 1e-3 run's stage, and the slope is negative through the
  anneal. `a` is small-but-nonzero; higher η recovered the alignment-limited bite. → Muon-jump-early is
  viable on 600; the **faithful resume + this 1e-3 jump bracket the operating LR**.
- **FLAT-AT-1e-3 (prediction wrong):** d_seg stays pinned at 0.00207 ± noise through the full anneal,
  no negative trend even as LR enters the fine-polish floor.

> **NOTE — the "1e-3-early-flat" the brief reports is NOT yet a flat verdict.** Section 1 predicts the
> 1e-3 slope is *smallest at peak LR* (hottest trunk, most diffuse whitened step). Early-window flatness
> is the EXPECTED shape, not the disconfirmation. The verdict needs the **post-anneal** slope.

### What FLAT-at-1e-3 (through the full anneal) would imply — the three-way disambiguation

`Δd_seg_Muon ∝ η·a·σ̄`. If 1e-3 (5× the faithful 2e-4) is STILL flat on 600:

1. **NOT "2e-4 too timid" (hypothesis 1 falsified).** A 5× LR with no movement means the bottleneck is
   not η. (If it WERE timid, 1e-3 descends — that's the prediction above.)
2. **`a` (600-pair alignment) is effectively ~0 → the N=16 bite was a memorization/coherence artifact
   that does NOT generalize from this fork (hypothesis 2 supported).** The orthogonalized step has no
   net d_seg direction once the flip set is diverse; Muon doesn't bite on 600 from the stage-5 fork.
   This does **NOT** prove capacity (see §3 confound) — it proves *this fork is not Muon-ready for 600*.
3. **Residual candidate: needs-EVEN-higher-LR (3e-3+).** Possible but **bounded by Muon-chaos**: the
   2026-06-12 MPS-pose verdict + the optimizer-chaos finding mean a too-large Muon η diverges the shared
   trunk (pose blows up). 3e-3 is the probe-2 `higher` LR that bit on N=16; on 600 it risks chaos. This
   is a *low-priority* branch, gated on the 1e-3 slope shape (if 1e-3 shows a faint-but-real negative
   slope, more LR helps; if dead-flat, more LR likely just adds chaos).
4. **The remaining live hypothesis is then hypothesis 3 (fork too early)** — Muon needs
   stage-7-conditioned weights → routes to the faithful resume below.

**No single result kills capacity.** Per probe-2 caveat 2: a flat Muon arm from an early fork could be
"fork not Muon-ready," not "capacity floor." Capacity is concluded ONLY if the faithful resume (stage-7
weights → stage-8 Muon at the faithful 2e-4 *and* a 1e-3 retry) ALSO goes flat through the full anneal.

---

## 3. The confound that keeps capacity OPEN (binding)

Both probe-2 and the live 1e-3 run fork from **pre-stage-7** weights (stage-4 snapshot / stage-5 jump
point). The faithful curriculum reserves stage 5 (C1a-L7, 9000ep) + stage 6 (λ-sweep) + stage 7
(σ-sweep) BEFORE stage 8 Muon. If those stages **condition the weights into the basin where the d_seg
flip set becomes coherent enough** (`a` rises) for the orthogonalized step to bite on 600, then an
early-fork flat result is a *fork* artifact, and the faithful resume (§4) is the only test that
disambiguates fork-vs-capacity. **This is why the faithful resume is staged regardless of the 1e-3
outcome.**

---

## 4. TASK B — staged faithful resume (the fork-too-early / hypothesis-3 test)

**The faithful run is paused (trainer process gone; only a dashboard renderer reads the dir), clean and
resumable** at `stage_index=4` (`stage5_c1a_l7`), epoch_in_stage 5050, `has_muon=False`. Resuming its
**own out-dir** auto-resumes from the checkpoint and runs stages 5→6→7→8 NATURALLY, so Muon (stage 8)
enters from **stage-7-conditioned weights** — exactly hypothesis 3.

**Resume is CLEAN:** the driver auto-resumes from the out-dir checkpoint (decoder+latents+EMA+AdamW+
schedulers+RNG). `has_muon=False` → the `muon_lr_floor_fix` and `taper`/`warmup` resume guards
(`driver.py:3216-3260`) all PASS (the Muon scheduler is built fresh at stage 8). The manifest already
carries `muon_lr_floor_fix=true` (the BUG-B fix was applied mid-burn), so the resume is floor-fix-faithful.

### The exact faithful-resume command (byte-identical to the live faithful argv; FAITHFUL Muon 2e-4)

```bash
cd /Users/adpena/projects/pact
# MPS must be FREE first (the live 1e-3 run owns it now → this is the POST-1e-3 move).
nohup .venv/bin/python -u experiments/launch_split_by_head_basin.py \
  --no-split-by-head --train-device mps --device cpu \
  --base-channels 20 --latent-dim 28 --n-pairs 600 \
  --targets-cache experiments/results/capstone_gt_targets_cache \
  --async-eval --eval-every 25 --checkpoint-every-epochs 25 \
  --muon-lr-floor-fix --seg-margin-hinge \
  --stage-lr-warmup-frac 0.03 \
  --taper-channels 16,16,17,19,19,14,10 \
  --defer-batch-sync \
  --out-dir experiments/results/yousfi_r3_taper_marginhinge_e5_20260620 \
  < /dev/null \
  > experiments/results/yousfi_r3_taper_marginhinge_e5_20260620/launch_faithful_resume_$(date -u +%Y%m%dT%H%M%SZ).outer.log 2>&1 &
disown
```

- **NO `--muon-lr` override** ⇒ vendored faithful **2e-4** for stage 8 (the faithful arm). Every other
  token is byte-identical to the recovered live faithful argv (which differs from the 1e-3 MUONJUMP run
  ONLY by `--muon-lr 0.001` and the out-dir).
- **Option to ALSO retry at 1e-3:** add `--muon-lr 0.001` to test stage-7-conditioned weights at the
  higher LR (this is the strongest single test: stage-7 conditioning × the η that bit on N=16). Best run
  as a SECOND resume to a COPY of the out-dir so the faithful 2e-4 arm is preserved — OR sequentially.
- **Resume verification (post-launch):** confirm a FRESH manifest (`age < 10s`) with `epoch_in_stage`
  advancing past 5050, `stage_index` still 4 then climbing, `n_records` CONTINUING (not reset),
  trajectory/summary appending. Per the apparatus-fix verification pattern.

### Cost

Remaining = stage5 tail (3950) + stage6 (2000) + stage7 (3000) + stage8 (5000) = **13,950 epochs**.
At the faithful ~5.84 s/epoch (measured under eval-contention + shared MPS) ≈ **22.6 h ≈ ~1 day**;
with MPS contention or slower eval cadence the brief's **~2-day** figure is the right conservative upper
bound. **Needs MPS** (the 104× scorer lever; CPU-only would be ~weeks) → it is the **post-1e-3 move**
(MPS is busy with the live 1e-3 run now).

---

## 5. Ranked re-route EV (the STILL-FLAT-at-1e-3 branch)

Ranked by expected info-per-day toward the d_seg<8.31e-4 pointer-beat:

1. **[HIGHEST EV] Faithful resume → stages 5→6→7→8 (§4), faithful 2e-4.** The ONLY test that
   disambiguates fork-too-early (hyp 3) from alignment-artifact (hyp 2) from capacity. ~1–2 days, MPS,
   $0 (local). Decisive: a flat stage-8 Muon from stage-7 weights is the strongest capacity signal we
   can get locally (still not a kill — see §3 + probe-2 caveat 2). **Do this regardless of the 1e-3
   verdict** (the brief stages it as the still-flat branch, but its value is unconditional).
2. **[HIGH EV, cheap rider] Add `--muon-lr 0.001` on a COPY of the faithful resume** (stage-7
   conditioning × higher η). Tests hyp 1 and hyp 3 jointly. Same ~1–2 day cost but strictly more
   informative than 2e-4 alone IF the 1e-3-from-stage5 run is flat (isolates "fork" from "η").
3. **[MED EV] Even-higher Muon LR 3e-3 from the stage-5/stage-7 fork** — the probe-2 `higher` LR that
   bit N=16. Gated on the 1e-3 slope SHOWING a faint negative trend (η-limited, not dead). If 1e-3 is
   dead-flat, 3e-3 risks Muon-chaos (2026-06-12 verdict) with low upside → skip.
4. **[MED EV] Different/earlier fork is NOT recommended** — the data points the opposite way (later =
   more conditioned = higher `a`). Forking earlier than stage 5 only makes the alignment worse.
5. **[FALLBACK, if Muon truly walls on 600] Abandon-Muon-for-other-d_seg-levers.** The d_seg knobs that
   are NOT optimizer-conditioning: (a) the **margin-hinge** seg surrogate is already on (`--seg-margin-hinge`,
   0.643× CE residual d_seg) — push its hinge harder; (b) the **d_seg-aware taper** is already on
   (`16,16,17,19,19,14,10`) — re-waterfill toward the high-res boundary band; (c) **capacity** via
   base_channels 20→24 with KD-warm-start (`--kd-warm-start-dir`) — the capacity-RD lever from the
   small-basis memos (bc24 floor 0.1353, marginal d_seg headroom), trading a little rate for d_seg.
   These are the post-Muon-wall pivot, NOT a kill of Muon (Catalog #307: implementation-level, paradigm
   intact).

---

## 6. 6-hook wire-in (research_only)

(1) sensitivity-map — N/A (optimizer-conditioning analysis, not per-byte). (2) Pareto — N/A (no bytes).
(3) bit-allocator — N/A. (4) cathedral autopilot — N/A (no archive artifact). (5) continual-learning —
the §2 prediction + §5 ranking feed the live-run curriculum decision as a candidate prior (NOT a
posterior score anchor). (6) probe-disambiguator — THIS memo + the live 1e-3 run + the staged faithful
resume ARE the disambiguator for fork-vs-alignment-vs-capacity on the stage-2-5 d_seg flatness.

**Decisive arbiter:** the live 1e-3 run's post-anneal 600-pair Muon d_seg slope, then (unconditionally)
the faithful resume's stage-8 Muon slope from stage-7 weights. NO kill; pointer UNMOVED 0.19110.
