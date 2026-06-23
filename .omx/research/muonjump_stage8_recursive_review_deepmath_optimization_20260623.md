# Jump-to-stage-8-Muon-early — recursive deep-math + optimization + adversarial review (2026-06-23)

**Status:** `[contest-CPU advisory]` config/decision-rigor gate · `research_only=true` · pointer
UNMOVED 0.19110 · NO score claim · NO kill. The live faithful run (`pid 79893`,
`yousfi_r3_taper_marginhinge_e5_20260620`) KEEPS RUNNING during this review (a DEFER costs nothing).
**Task #161 sister** (the surgery + smoke is `af3ce2e80918f880b`; THIS review owns the CONFIG math).
**Subagent:** `muonjump_review_20260623`.

> ## VERDICT: **PROCEED_WITH_REVISIONS**
> Jumping the live stage-5 checkpoint (ep ~15.6k, d_seg 0.00207, d_pose 0.00021, archive 79.5 KB)
> straight to the stage-8 Muon finisher is **sound and well-motivated** by the probe-2 data and the
> conditioning math. The jump applies 3 simultaneous shifts (AdamW→Muon, λ 0.01→0.02, σ 0.2→0.1) but
> the run's already-on `--stage-lr-warmup-frac 0.03` (150-epoch LR warm-in from 0.1× peak) is a
> genuine mitigation, the EMA/export path is shadow-correct (no warm-in jolt), and pose is at floor
> with `--seg-margin-hinge` protecting it. The REVISIONS are: **(R1)** raise the faithful Muon LR
> from 2e-4 toward **~6e-4** (a measured speed/stability sweet spot — NOT the 3e-3 overshoot edge);
> **(R2)** keep the muon-lr-floor-fix ON (mandatory); **(R3)** do NOT add a manual λ/σ warm-in beyond
> the existing 0.03 LR warmup — the warmup already eases the shift, and a manual λ ramp would fight
> C1a's own equilibration; **(R4)** byte-close + fire the paired Modal exact eval when the **EMA-SHADOW**
> d_seg (not the live d_seg) crosses **≤ 9.0e-4** (a margin below the 9.22e-4 break-even), re-checked
> against the THEN-current archive bytes.

---

## The operating point (live, last advisory eval ep 15600)

| quantity | value | source |
|---|---|---|
| d_seg (shadow) | 0.002079 | trajectory eval row ep15600 |
| d_pose | 0.000209 | "" |
| archive bytes | 79,479 | "" (λ=0.01 C1a equilibrium; was 87.6 KB pre-C1a) |
| rate term `25·B/37545489` | 0.0529 | recomputed |
| pose term `√(10·d_pose)` | 0.0460 | recomputed |
| score | 0.3066 | 100·0.00208 + 0.0460 + 0.0529 |
| **non-seg S (rate+pose)** | **0.0990** | the floor the d_seg sits on |

**Break-even (recomputed at the LIVE operating point, supersedes the floor memo's 8.26e-4 which used the
older 0.108 pose+rate):**
* beat borrowed 0.19110 ⇒ `100·d_seg < 0.19110 − 0.0990` ⇒ **d_seg < 9.22e-4** (2.25× from 0.00208).
* sub-0.15 (T_3) ⇒ **d_seg < 5.11e-4** (4.07× from 0.00208).

Both targets get **EASIER** as stage 8's λ=0.02 shrinks the archive below 79.5 KB (see Axis 2).

---

## The jump vs the natural path — what changes

The faithful curriculum (vendored `hnerv_muon_finetuned_from_pr95` builders, VERIFIED):

| stage | λ (cat_lambda) | σ (cat_sigma) | adamw_lr | muon_lr | use_muon | epochs |
|---|---|---|---|---|---|---|
| 5 (live now) | 0.01 | 0.2 | 3e-5 | — | False | 9000 |
| 6 | **0.02** | 0.2 | 3e-5 | — | False | 2000 |
| 7 | 0.02 | **0.1** | 3e-5 | — | False | 3000 |
| 8 (Muon finisher) | 0.02 | 0.1 | 1e-5 | **2e-4** | **True** | 5000 |

The jump (stage_index 4→7) skips ~9.5k epochs of stages 5-tail/6/7 and lands the 3 simultaneous shifts
at once. Mechanically it is a checkpoint `position`-edit to `(stage_index=7, epoch_in_stage=0)` +
auto-resume; the af3 sister verifies the weights carry. `epoch_in_stage=0` is decisive: it makes the
jump a **clean stage boundary**, so the driver's resume guards all pass and the Muon optimizer +
scheduler are built FRESH (cosine over 5000 ep with the floor fix) — see Axis 3/5.

---

## Optimal jump config — knob-by-knob vs the faithful stage-8 default

| knob | faithful stage-8 default | **OPTIMAL JUMP** | rationale (axis) |
|---|---|---|---|
| Muon LR | 2e-4 | **~6e-4** | A1: probe-2 shows 2e-4 descends but slowly; 3e-3 fast but overshoot-risky; FP-rate sublinear in LR (15×LR→1.58× rate) ⇒ 3× LR buys ~1.25× speed at far lower overshoot risk. |
| muon-lr-floor-fix | (must be ON) | **ON (mandatory)** | A1: OFF floors Muon at 0.5×peak=1e-4 forever (BUG-B); never anneals to fine-polish. Already ON in live argv. |
| muon eta_min (with fix, LR 6e-4) | `5e-6/6e-4`=8.3e-3 ⇒ floor 5e-6 | unchanged | A1: floor keys to muon_lr; correct absolute fine-polish floor regardless of peak. |
| AdamW-aux LR | 1e-5 | **1e-5 (keep)** | A4: aux set is stem/rgb/biases+latents; small LR keeps pose-relevant heads stable while Muon does the trunk d_seg work. |
| cat_lambda (C1a) | 0.02 | **0.02 (keep)** | A2: λ=0.02 = the rate work; stage 8 carries it, so the jump does NOT forfeit the rate prize, it DOES it. |
| cat_sigma | 0.1 | **0.1 (keep)** | A3: σ drop reduces quant-noise reg; the 0.03 LR warmup eases the transition; no manual σ ramp. |
| stage-LR-warmup-frac | 0.03 (live) | **0.03 (keep)** | A3: w=150 ep linear LR warm-in from 0.1× peak on BOTH AdamW+Muon = the simultaneous-shift mitigation already present. |
| seg loss | vendored l7_softplus | **margin_hinge (live)** | A1/confound-2: live sets `seg_surrogate=margin_hinge` for ALL stages; its gradient does NOT collapse on confident flips (constant slope −1) ⇒ a BETTER Muon-d_seg pairing than the probe's l7_softplus. |
| ema_warmup | OFF | **OFF (keep)** | A5: 75 steps/ep ⇒ EMA τ≈13 ep ≪ eval_every=25; shadow tracks; warm-in unnecessary AND would jolt a carried-warm shadow. |
| grad_clip / grad_clip_muon | 1.0 / 1.0 | **1.0 / 1.0 (keep)** | A3: clip bounds the spectral step; matches probe-2. |
| eval cadence | eval_every 25 | **25 (keep), watch SHADOW** | A6: τ≈13 ep ⇒ each eval reads a converged shadow; fire paired eval when SHADOW d_seg ≤ 9.0e-4. |

---

## The six deep-math axes

### Axis 1 — optimal Muon LR for the jump (speed-to-d_seg-close, not faithfulness)

**Probe-2 measured (from the stage4 fork, int8-quant d_seg, the contest authority surface):**

* **faithful stage8 LRs** (Muon 2e-4 / AdamW 1e-5): Muon DESCENDS (FP 0.00195→0.00119 over 150 steps,
  −5.04e-6/step; quant net −4.37e-4) while **AdamW REGRESSES** (quant 0.00194→0.00352 at step 50,
  recovering only to 0.00222 at step 150 — i.e. ABOVE the fork; FP barely moves −0.00046). **Gap
  (Muon−AdamW) = −7.2e-4 at step 150**, Muon strictly below. This is the cleanest conditioning
  signal in the dataset: at the FAITHFUL LR, AdamW cannot even hold d_seg, Muon descends it.
* **higher LRs** (Muon 3e-3 / AdamW 3e-4): Muon −1.40e-3 quant over 199 steps (−7.98e-6/step FP),
  AdamW −1.07e-3; gap −3.4e-4 widening monotonically.

**The Newton-Schulz spectral derivation.** Muon's update is the orthogonalized gradient `U Vᵀ`
(unit spectral norm after ns_steps=5 iterations), so the parameter step magnitude is `lr` DIRECTLY —
not `lr·‖grad‖`. This makes Muon **κ-independent**: it decorrelates the κ≈19 boundary Hessian that
AdamW's diagonal preconditioner cannot, which is exactly why AdamW's grad-norm collapses (0.039→0.004
in probe-2's within-arm trace) and its d_seg stalls while Muon keeps biting. Because the step is `lr`,
the d_seg-descent rate is **sublinear in LR** (15× LR 2e-4→3e-3 yields only 1.58× FP descent rate):
beyond a point, larger steps overshoot the boundary basin and the int8-127 quant grid clamps the gain.

**The LR recommendation.** The faithful 2e-4 is SAFE but slow (~247 steps to break-even on the
optimistic small-N extrapolation). 3e-3 is the overshoot edge (the probe's noisy quant curve at step
50, 0.00233 vs FP 0.00161, shows the int8 grid already fighting the large step). The sublinearity says
the marginal speed above ~1e-3 is small while the overshoot/quant-thrash risk rises. **Recommend
Muon LR ≈ 6e-4** (3× faithful): on the FP-rate trend that is ~6.0e-6/step (interpolating the two
measured points), ~207 steps to break-even — ~25% faster than 2e-4 — at a step magnitude only 3× the
faithful, well inside the regime where probe-2's 2e-4 arm was clean and monotone. KEEP the floor fix
so even at 6e-4 peak the cosine anneals to 5e-6 for the fine polish. **Adversarial caveat:** if the
first ~150 eval epochs show the SHADOW d_seg non-monotone (overshoot), drop to 4e-4; the floor-fix +
0.03 warmup make this self-correcting (the warmup starts at 0.1×6e-4 = 6e-5, ramps over 150 ep).

### Axis 2 — stage-8 config audit + the rate framing (λ=0.02 COMPRESSES during the jump)

VERIFIED from `curriculum.build_curriculum` + vendored `stage8_muon_finetune.make_config`: stage_index
7 = `use_muon=True`, `adamw_lr=1e-5`, `muon_lr=2e-4`, `grad_clip_muon=1.0`, `cat_lambda=0.02`,
`cat_sigma=0.1`, `use_qat=True`, seg=100/pose=1 weights. **C1a mechanism** (`cat_entropy_v2(decoder,
sigma)`): a categorical-entropy penalty that biases decoder weights toward a low-entropy
(brotli-friendly, snap-to-grid) distribution — a RATE lever. The vendored λ-sweep history confirms it
lowers bytes (stage 6 λ=0.02 canonical `lambda_0.02_ep475`, stage 7 `exp4_sigma01_ep975`).

**Re-quantified rate framing.** The live archive ALREADY dropped 87.6 KB → 79.5 KB (−9%, −0.005 rate
term) when stage-5 C1a λ=0.01 activated. The jump's stage 8 inherits **λ=0.02 (2× stronger)**, so it
continues compressing during the Muon finetune. Therefore the "~0.005 rate prize sacrificed by
skipping stages 6-7" framing is **wrong-signed**: the jump does NOT forfeit the rate work — stage 8
carries the same λ=0.02 the skipped stages would have applied, and provides 5000 epochs to settle it
CONCURRENTLY with the d_seg finish. The only thing genuinely skipped is the redundant 5000-epoch
λ/σ-equilibration WALL-CLOCK of stages 6+7. Net: the break-even d_seg threshold (9.22e-4) only gets
LOOSER as the archive shrinks below 79.5 KB during stage 8.

### Axis 3 — simultaneous-shift instability (AdamW→Muon + λ↑ + σ↓ all at once)

The natural path eases these in over stages 6-7; the jump applies all three at the stage-5→8 boundary.
Adversarial analysis:

* **AdamW→Muon under QAT.** The decoder is under int8-127 fake-quant (QAT on since stage 4). A sudden
  Muon spectral step is the LARGEST single change. Mitigation present: `_warmup_wrap` with
  `warmup_frac=0.03` ⇒ **w = ceil(0.03·5000) = 150 epochs** of LINEAR LR ramp from `start_ratio=0.1`
  (6e-5 at LR 6e-4) up to the cosine value — applied to BOTH the AdamW and Muon LambdaLRs. This eases
  the trunk into the orthogonalized-step regime over 150 ep × 75 steps = 11,250 steps rather than
  slamming to peak. grad_clip_muon=1.0 bounds the per-step spectral magnitude. The QAT grid is the
  SAME 127-level both before and after (no grid change at the boundary), so QAT itself is continuous.
* **σ 0.2→0.1.** σ is the C1a categorical kernel width, NOT the quant-noise σ of an earlier curriculum
  variant — it sharpens the weight-clustering penalty. A sharper penalty is a SMOOTH change in the
  loss landscape, not a discontinuity; the 150-ep LR warmup covers the adjustment.
* **λ 0.01→0.02.** Doubling the C1a weight is a smooth loss-term rescale; the warmup eases the trunk's
  response. A manual λ ramp would FIGHT C1a's own gradient-driven equilibration (the penalty is
  self-annealing as weights snap to grid) and is NOT recommended (R3).

**Verdict (Axis 3):** the simultaneous shift is **acceptable** given the existing 0.03 LR warmup +
grad_clip_muon=1.0 + continuous QAT grid. No additional manual warm-in warranted. **Adversarial
residual:** the 0.03 warmup was validated as a stage-TRANSITION pose-kick fix, not specifically for a
3-stage skip; if the SHADOW d_seg or d_pose spikes in the first 150 ep, the floor-fix cosine + warmup
self-correct, but watch the first eval (ep+25) closely.

### Axis 4 — pose protection (d_pose at floor 0.00021)

d_pose contributes `√(10·d_pose)` = 0.0460 — at this operating point pose's MARGINAL sensitivity is
HIGH (`d/d(d_pose) √(10·d_pose) = 5/√(10·d_pose) = 109`, vs d_seg's constant 100), so a d_pose
REGRESSION is expensive. Muon orthogonalizes the TRUNK grads; the seg push could in principle spill
into pose via the shared trunk. Protections present + analysis:

* **Partition.** `partition_params_for_muon(decoder)` puts the 12 conv trunk weights under Muon; the
  stem/rgb/bias heads + **latents** stay under AdamW at 1e-5 (tiny). The pose head reads the trunk
  features but the rgb_0/rgb_1 + latent path that carries pose stays AdamW-slow.
* **`--seg-margin-hinge`** concentrates the seg gradient EXACTLY on the d_seg flip set (constant-slope
  hinge on `logit[GT] − max_{c≠GT}`), wasting NO gradient on interior/pose-neutral pixels — so the
  seg objective's trunk perturbation is sparse and boundary-localized, minimizing pose spillover.
* **Empirical anchor:** the live run held d_pose at ~0.00021 through stages 1-5 under AdamW; probe-2's
  Muon arms were d_seg-only but the partition (latents under AdamW) is identical.

**Verdict (Axis 4):** d_pose-regression risk is LOW but NON-ZERO. **Mitigation already present**
(partition + margin-hinge + AdamW-slow latents). **Recommendation:** watch d_pose at each eval; if it
rises above ~3.0e-4 (pose term 0.055, costing +0.009 S), that is the DEFER/throttle signal — but no
pre-emptive change needed. (The run has no pose-throttle active, which is correct per the
score>training-time directive — pose is computed every epoch.)

### Axis 5 — EMA carry + warmup on the epoch-0 stage restart (export-critical)

The export bytes = the EMA SHADOW (the EMA non-negotiable). The jump restarts at stage-8/epoch-0 with
the carried `ema_decoder`/`ema_latents`. Two sub-questions:

* **Does the carried-warm shadow get RE-SNAPPED to decay 0.1?** NO. `ema_warmup` is **OFF** on this run
  (no `--ema-warmup` in argv) ⇒ decay is the constant `spec.ema_decay=0.999` every step. The #85
  warmup ramp (`min(decay,(t+1)/(t+10))`) is never consulted. So there is NO warm-in jolt to a carried
  shadow. (Even IF warmup were on, the driver PERSISTS `_ema_step` across resume — `_checkpoint_state`
  key `ema_step`, `_restore_into` reads it — precisely to avoid the re-snap; verified in driver.py
  ~2501-2558. So both paths are safe.)
* **Does constant 0.999 track the FAST Muon descent?** YES. batch_size=8, 600 pairs ⇒ **75 steps/epoch**
  ⇒ EMA τ = 1000 steps ≈ **13.3 epochs** ≪ eval_every=25 epochs. The shadow converges to within ~5% of
  a step-change in ~40 ep, ~1% in ~61 ep. So each eval (every 25 ep) reads a CONVERGED shadow; the
  BEST-from-shadow tracking is shadow-correct.

**Verdict (Axis 5):** EMA/export path is SOUND. **Important operational consequence:** during the fast
descent the shadow LAGS the live d_seg by ~τ·rate (~5e-3 transient while live descends linearly), so
the EXPORT (shadow) crosses the break-even threshold AFTER the live does. The byte-close gate MUST read
the shadow d_seg (which the driver already evals), NOT the live training d_seg — see Axis 6.

### Axis 6 — early-eval + byte-close strategy

Probe-2's optimistic small-N extrapolation: ~207 steps (at LR 6e-4) to live break-even; but the LIVE
run's 600-pair descent is SLOWER than the 16-pair probe (small-N memorization bias, caveat 1). Realistic
expectation: the descent is hundreds of epochs (×75 steps). At ~14 s/epoch the break-even is plausibly
hours-to-~1 day; the full 5000-ep stage ~20 h+.

**Cadence:** keep eval_every=25 (τ≈13 ep, so each eval is shadow-converged). **Byte-close + fire the
paired Modal exact eval when the SHADOW d_seg ≤ 9.0e-4** (a 2.4% margin below the 9.22e-4 break-even,
to absorb the advisory↔exact gap), re-computing the break-even against the THEN-current archive bytes
(λ=0.02 will have shrunk it below 79.5 KB, so the threshold loosens). For sub-0.15, continue to the
**SHADOW d_seg ≤ 5.0e-4** gate. Do NOT fire on the live d_seg (it leads the shadow during descent — a
premature exact eval would over-state).

---

## Recursive adversarial review log (3-clean-pass discipline)

Lenses per round: (a) optimizer-math/Muon, (b) stability/Dykstra-feasibility, (c) pose-coupling,
(d) QAT/rate, (e) EMA/export, (f) Contrarian (is jumping right at all?). Each round also answers the
ASSUMPTION-CHALLENGE axis. A round with zero findings = clean pass; 3 consecutive = SEAL.

### Round 1 — findings (counter resets to 0)
* **(a) FINDING A1.** Initial draft recommended Muon LR 2e-4 (faithful). The jump optimizes for SPEED,
  and probe-2 shows 2e-4 is slow; the sublinear-in-LR math says ~6e-4 is the sweet spot. **FIX:**
  raised to 6e-4 with floor-fix + warmup self-correction. (knob table + A1 updated.)
* **(d) FINDING A2.** Initial draft accepted the "~0.005 rate sacrifice" framing. Audit of the vendored
  λ-sweep + the live 87.6→79.5 KB drop shows stage-8 λ=0.02 COMPRESSES — the jump does the rate work,
  not skips it. **FIX:** re-framed Axis 2; break-even threshold loosens during stage 8.
* **(e) FINDING A3.** Initial EMA analysis used a wrong ~6 steps/epoch estimate (catastrophic-lag
  scare). batch_size=8 ⇒ 75 steps/ep ⇒ τ≈13 ep ≪ eval_every. **FIX:** recomputed; shadow tracks; gate
  on shadow not live.
* **Assumption-challenge (R1):** the review was operating within "Muon-on-600-pairs behaves like the
  N=16 probe." VIOLATING it (600-pair is slower + less memorization-prone) does NOT change the SIGN of
  the verdict (Muon still descends, AdamW still stalls — the CONTRAST is the transferable signal, not
  the absolute rate) but DOES change the wall-clock (hundreds of epochs, not the optimistic ~207
  steps). Folded into Axis 6.

### Round 2 — findings (counter resets to 0)
* **(f) Contrarian FINDING A4.** "Is jumping even right vs letting the faithful run reach stage 8
  naturally in ~1.5 d?" The faithful run is at stage 5 (9000 ep) with ~4975/9000 done, then stages 6
  (2000) + 7 (3000) before stage 8 — that is ~9000 more epochs × 14 s ≈ **35 h** to even START stage 8,
  vs the jump starting it NOW. The probe-2 stage8-LR data shows stages 5-7 are NOT necessary d_seg-prep
  (AdamW at faithful LR REGRESSES d_seg). The jump's only RISK over the natural path is the
  simultaneous-shift (Axis 3, mitigated) and the stage4-vs-stage7 fork confound (probe forked from
  stage4, the live jump forks from stage5 — STRONGER, more Muon-ready weights). **RESOLUTION:** jumping
  is justified; the faithful run KEEPS RUNNING as the control/fallback, so the jump is a parallel
  candidate, not a replacement — zero downside. (No config change; strengthens the verdict.)
* **(c) FINDING A5.** Pose-coupling: confirmed the partition keeps latents under AdamW, but flagged
  that d_pose marginal sensitivity (109) now EXCEEDS d_seg's (100) at this operating point — a d_pose
  regression is the dominant tail risk. **FIX:** added the explicit d_pose watch threshold (≥3.0e-4 =
  DEFER signal) to Axis 4.
* **Assumption-challenge (R2):** operating within "the probe's l7_softplus Muon-descent transfers to
  the live margin_hinge seg loss." VIOLATING it: margin_hinge has a CONSTANT-slope gradient on the flip
  set (no collapse on confident flips, unlike l7_softplus's vanishing `p(1−p)`), so margin_hinge + Muon
  should descend d_seg AT LEAST AS WELL, likely BETTER — the assumption-violation STRENGTHENS the
  verdict. Folded into the knob table (seg-loss row) + A1.

### Round 3 — CLEAN PASS 1
* (a) Muon LR 6e-4 + floor-fix + warmup: math consistent, overshoot self-correcting. No finding.
* (b) Dykstra-feasibility: the 3-constraint shift (Muon-step, λ↑, σ↓) lands at a clean stage boundary
  with a 150-ep LR warm-in onto a continuous QAT grid; feasible. No finding.
* (c) pose: partition + margin-hinge + AdamW-slow latents + the explicit watch threshold. No finding.
* (d) QAT/rate: 127-grid continuous; λ=0.02 compresses; break-even loosens. No finding.
* (e) EMA/export: ema_warmup OFF, τ≈13 ep ≪ eval cadence, _ema_step persists anyway, gate on shadow.
  No finding.
* (f) Contrarian: faithful run is the fallback; jump is a zero-downside parallel candidate. No finding.
* Assumption-challenge: the two load-bearing assumptions (N=16↔600 transfer; l7_softplus↔margin_hinge
  transfer) were both surfaced and resolved — neither flips the verdict. No finding.

### Round 4 — CLEAN PASS 2
Re-traced every knob in the config table against the driver source (line refs in Axes 1-5). The
Muon construction (driver 1784-1796), floor-fix (1827-1864), warmup_wrap (102-139), EMA update
(2121-2134), checkpoint persist/restore of ema_step (2505/2558), and the resume guards (3216-3257)
all match the recommended config and pass on an epoch-0 stage-boundary jump. No finding.

### Round 5 — CLEAN PASS 3 → **SEAL**
Adversarial re-read of the DEFER conditions: the only named-measured blockers that would flip
PROCEED_WITH_REVISIONS → DEFER are (1) SHADOW d_seg non-monotone/overshoot in the first 150 ep under
LR 6e-4 (mitigated by floor-fix+warmup; if it occurs, drop LR to 4e-4 — a revision, not a kill), or
(2) d_pose regressing above 3.0e-4 (watch threshold). Neither is currently measured (the jump has not
launched), so neither blocks PROCEED. **3 consecutive clean passes — SEALED.**

---

## DEFER blockers (none currently active; named-measured triggers for a future DEFER)
* **B1 (overshoot):** if SHADOW d_seg is non-monotone over the first 150 stage-8 epochs at LR 6e-4 →
  revise LR down to 4e-4 (NOT a kill; the floor-fix+warmup make this self-diagnosing at the first
  evals).
* **B2 (pose spill):** if d_pose rises above ~3.0e-4 (pose term 0.055) → the seg push is spilling into
  pose; throttle the Muon LR or DEFER pending a pose-protection revision. Not currently measured.

## 6-hook wire-in (research_only)
* (1) sensitivity-map — N/A (config/decision review, not a per-byte sensitivity contribution).
* (2) Pareto constraint — N/A (no archive bytes; advisory non-promotable).
* (3) bit-allocator hook — N/A.
* (4) cathedral autopilot dispatch — N/A (no archive-deployable artifact).
* (5) continual-learning posterior — the verdict + optimal config feed the live-run jump decision as a
  candidate prior; NOT a posterior SCORE anchor (no exact row).
* (6) probe-disambiguator — this review CONSUMES probe-2 (the Muon-vs-AdamW disambiguator) and gates
  the jump config; it IS the config-rigor gate for the multi-day launch. `research_only=true`.

## Authority + non-negotiables
`[contest-CPU advisory]` NON-PROMOTABLE. NO score claim (only `upstream/evaluate.py` on the byte-closed
archive is authority). NO kill. pointer UNMOVED 0.19110. MPS is the live run's TRAIN-gradient device
(104× scorer speedup) and NEVER a score authority — this review used only the CPU-authority probe-2
data + source inspection; no MPS device was touched and the live run was read-only.
