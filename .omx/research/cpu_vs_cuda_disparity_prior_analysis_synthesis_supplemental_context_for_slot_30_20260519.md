# CPU-vs-CUDA disparity prior analysis synthesis — supplemental context for slot 30

> **FOR-SLOT-30-CONSUMPTION** — operator-routed supplemental context per
> CLAUDE.md "Subagent coherence-by-default" non-negotiable. Slot 30
> (`afceb9c62403f9781`) is in-flight on CPU-vs-CUDA + MPS portability
> analysis; this memo synthesizes the canonical prior analysis surface so
> slot 30's framing INTEGRATES rather than re-discovers.

> _Generated_: `2026-05-19` ·
> _Tag_: `[diagnostic-not-score]` ·
> _Score claim_: **NONE** ·
> _Lane_: `lane_prior_context_location_plus_hf_jobs_dispatch_fix_20260519`

---

## 1. Operator framing (verbatim)

> *"the disparity is more nuanced than that"*
> *"we have some deeper analysis already that should serve as context and some information in our writeup i think in a .md draft somewhere"*

**Translation**: the simple "CPU eval is 0.01-0.03 better than CUDA on the
same archive" oversimplification is structurally incomplete. The canonical
prior analysis already captured packet-specific, operating-point-dependent,
mechanism-decomposed nuance that any new analysis MUST integrate.

---

## 2. Canonical prior-analysis source inventory

### 2.1 Primary anchor (THE reference)

**`.omx/research/device_axis_paired_anchor_matrix_20260511.md`** —
device-axis paired anchor matrix (refreshed 2026-05-11T16:27:18Z). This
is the canonical empirical table that EXPLICITLY falsifies the monotone
"CPU > CUDA" framing. Per the matrix:

| Anchor | Representation | CUDA score | CPU score | Δ | Pose CUDA/CPU | Seg CUDA/CPU | Winning axis |
|---|---|---:|---:|---:|---:|---:|---|
| **A1** (PR101-derived, score-gradient training) | HNeRV-cluster | 0.22635 | **0.19285** | **+0.0335** | **5.18×** worse on CUDA | 1.18× | **CPU** |
| **PR106 latent sidecar r2** | HNeRV + per-pair latent perturbation | **0.20665** | 0.22809 | **-0.0215** | **0.197×** (5.1× *better* on CUDA) | 1.017× | **CUDA** |
| PR106 r2 + PR101 grammar (FLOOR) | HNeRV + r2 + format_id=0x02 | **0.20662** | 0.22806 | -0.02144 | 0.197× | 1.017× | **CUDA** |
| 5 family wrappers (c3 / wavelet / cool_chic / siren / coord_mlp empty) | PR106 r2 + 0xFD wrapper | 0.20663 | 0.22810 (where measured) | -0.02147 | 5.06× | 1.18× | **CUDA** |

**Two empirical anchors lying on OPPOSITE sides of the device-axis flip.**
A1 wins on CPU by +0.0335. PR106 r2 wins on CUDA by -0.0215. **Same
HNeRV-family representation; opposite device-axis winner.** The pose ratio
flips from 5.18× CPU-favored to 0.197× CUDA-favored.

This single matrix EXTINCTS any monotone "X is better than Y" framing.

### 2.2 Mechanism deep-dive (THE explanation)

**`.omx/research/cuda_cpu_pose_drift_mechanism_deep_dive_20260508_claude.md`**
(2026-05-08 deep research, no GPU dispatched). Falsifies 3 common-knowledge
intuitions and proposes a calibrated additive-noise model:

1. **FastViT-T12 has ZERO attention layers** (all 4 stages are RepMixer
   depthwise-conv). The "T" refers to transformer-style block layout but
   uses RepMixer not self-attention. The first FastViT variant with
   attention is `fastvit_sa12` — but PoseNet uses T12.
2. **T4 (sm_75 Turing) does NOT support TF32.** TF32 was introduced on
   Ampere (sm_80, A100). Setting `cuda.matmul.allow_tf32=True` is a no-op
   on T4.
3. **`torch.backends.cuda.matmul.allow_tf32` defaults to FALSE in PyTorch
   ≥ 1.12** even on Ampere hardware where TF32 is available. cuDNN's
   `allow_tf32` defaults TRUE, but is a no-op on T4 regardless.

The canonical mechanism candidates (additive, not multiplicative):

- **(A) cuDNN FP32 reduction order** in conv2d / depthwise-conv2d (parallel
  vs serial; per-output relative error ~1e-6 to 1e-5; compounds across
  ~50+ conv layers).
- **(B) NVDEC vs PyAV ground-truth decode divergence** (±1 LSB on chroma
  boundaries; amplified ~4× by std=63.75 normalization). Largest
  hypothesized contributor at 25% of pose drift.
- **(C) GELU(tanh) FP32 intrinsic difference** (CPU glibc `tanh()` vs CUDA
  `__tanhf` intrinsic; ~2-3 ulp per call across dozens of GELU calls).
- **(D) BatchNorm `AllNorm` flat reduction** in pose head (mostly NOT a
  mechanism — `view(-1, 1)` is bit-comparable in eval mode).

**Canonical model (Welch's law / additive variance):**
```
pose_cuda ≈ pose_cpu + sigma²_noise
sigma_total = sqrt(L) × sigma_per_op
For L=50, sigma_per_op=0.0017, sigma_total ≈ 0.012
Observed CUDA pose noise RMS: sqrt(1.39e-4) ≈ 0.0118  ← match within 2%
```

**Why pose drift is 5× and seg drift is only 1.17×:** the two outputs
differ in their TERMINAL NONLINEARITY:
- **PoseNet** outputs a regression vector; precision noise enters
  quadratically via `MSE(out1, out2)`. At medal-band `pose_avg≈3.5e-5`,
  the CUDA noise floor (~1.4e-4) is **4× the signal** → pose distortion
  is NOISE-FLOOR-LIMITED at medal band, not signal-limited.
- **SegNet** outputs class logits; `compute_distortion = E[argmax(out1) !=
  argmax(out2)]` is STABLE under small logit perturbations; only boundary
  pixels flip.

**Decomposition of PR102's 0.0330 score-gap:**
- Pose contribution: 0.0231 (70% of gap)
- Seg contribution: 0.0100 (30% of gap)
- Rate contribution: identical (file size division)

### 2.3 Dual-eval mandate (THE rule)

**`feedback_dual_cpu_cuda_auth_eval_mandatory_20260508.md`** — 2026-05-08
operator-mandated NON-NEGOTIABLE rule (now in CLAUDE.md "Submission auth
eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" section).
Empirical basis: PR102 third prize was awarded against its CPU score
0.19538, not its CUDA score 0.22839. Our PR #107 was scored only on CUDA.
**The contest leaderboard ranks by CPU eval, not CUDA eval.**

Required hardware compliance: Linux x86_64 for CPU (NOT macOS — ARM
intrinsics drift); NVIDIA Linux for CUDA (NOT MPS — 23× PoseNet drift
empirically established).

### 2.4 Xray mechanism-attribution pipeline (THE diagnostic surface)

**`.omx/research/cpu_cuda_xray_synthesis_20260511.md`** — handoff P5
deliverable. Three orchestrator tools landed for mechanism attribution:

| Tool | What it measures |
|---|---|
| `tools/cpu_cuda_xray_loader_drift.py` | PyAV (CPU) vs DALI/NVDEC (CUDA) decoded RGB delta + shared-input custody |
| `tools/cpu_cuda_xray_segnet_layer_drift.py` | Per-EfficientNet-B2-stage CPU/CUDA forward drift + per-stage compounding |
| `tools/cpu_cuda_xray_posenet_layer_drift.py` | Per-FastViT-T12-block CPU/CUDA forward drift + Lipschitz compound estimate |

These produce a 4-cell verdict structure: **A: Loader-dominated** / **B:
Scorer-forward-dominated** / **C: Threshold-geometry / Lipschitz-amplified**
/ **D: Mixed / coupled**. The synthesis is a diagnostic guide; it does
NOT decide score-promotion verdicts.

### 2.5 Writeup draft (THE narrative anchor)

**`docs/writeup_draft.md`** — long-form historical writeup. References the
23× MPS-vs-CUDA discovery + the catastrophic 53.61 mask-resolution
incident + Lane G v3 1.05 [contest-CUDA]. The 2026-05-06 public-claim
hygiene update added quarantine framing for every score not promoted by
an exact A++/A evidence row. Not the canonical CPU-vs-CUDA disparity
analysis (the device-axis matrix is); but the operator may also have been
referring to the writeup's terminology + history.

### 2.6 MPS portability anchor (THE complementary axis)

**`feedback_mps_drift_mathematical_and_engineering_formalization_landed_20260519.md`**
(today). Lands `tac.mps_diagnostic.drift_predictor`. Empirically RECLASSIFIES
the CLAUDE.md "MPS PoseNet drift 23×" anchor as HARD-EARNED-EMPIRICALLY-
RECLASSIFIED for current architecture-class (0.072% Phase B aggregate
falsifies for current architecture). Includes 4-verdict cosine
distribution taxonomy + Higham 2002 sqrt(N) layer-depth bound +
calibration-anchor mechanism. **CRITICAL for slot 30**: the 23× anchor is
architecture-class-dependent, not universal — the contest scorer
architecture (SegNet/PoseNet) at current model class shows 0.072% drift,
NOT 23%.

**Sister anchor**: `feedback_mps_drift_granular_analysis_corrective_engineering_landed_20260519.md`
— per-pair drift CV = 2.6% (UNIFORM, not fat-tail). Drift is uniform-noise
dominated, not structural per-pair. Falsifies per-pair CUDA-shadow routing
as EV-positive engineering direction.

### 2.7 Cross-references (additional layers)

- **`.omx/research/cuda_cpu_drift_sweep_design_20260508_claude.md`** — empirical sweep design across 25 PRs and 6 strata
- **`.omx/research/public_replay_drift_hypothesis_20260508_codex.md`** — codex's drift hypothesis matrix (5 hypothesis rows)
- **`.omx/research/cpu_cuda_drift_mechanism_validity_guard_20260511_codex.md`** — validity guard
- **`.omx/research/cpu_cuda_drift_mechanism_localization_20260511.md`** — localization
- **`.omx/research/cpu_cuda_drift_analyzer_canonicalization_20260511_codex.md`** — canonicalization
- **`.omx/research/a1_pr106_cpu_cuda_axis_validation_20260513_codex.md`** — A1+PR106 validation
- **`.omx/research/hdm4_hlm1_cpu_cuda_axis_closure_20260513_codex.md`** — HDM4+HLM1 closure
- **`.omx/research/pr101_fec6_paired_cpu_cuda_axis_xray_20260515_codex.md`** — PR101 fec6 xray
- **`.omx/research/pr106_format0c_paired_cpu_cuda_auth_eval_20260515_codex.md`** — PR106 format0c
- **`.omx/research/l5_v2_tt5l_paired_cpu_cuda_axis_plan_from_anchor_20260517_codex.md`** — L5 v2 TT5L plan
- **`.omx/research/exact_cuda_terminal_reviews_20260516_codex/z3_v2_full_paired_cpu_20260515_paired_modal_auth_20260515T142723Z_cuda_review.json`** — Z3 v2 paired review
- **`.omx/research/public_pr_cpu_cuda_drift_analysis_20260508_codex.md`** — public PR drift
- **`.omx/research/public_pr_auth_eval_comment_drift_and_dual_axis_protocol_20260508_codex.md`** — paired-axis protocol
- **`reports/public_pr100_108_eval_comment_scorecard_20260508.json`** — 5 paired CPU/CUDA datapoints (PR102 anchor)

---

## 3. THE CANONICAL NUANCED FRAMING (synthesis)

The prior analysis surface is consistent on the following structurally
non-negotiable points:

### 3.1 Device-axis behavior is PACKET-SPECIFIC, not monotone

A1 (PR101-derived, score-gradient training) wins on CPU by +0.0335.
PR106 latent sidecar r2 wins on CUDA by -0.0215. The pose ratio flips
sign between the two HNeRV-family substrates (5.18× CPU-favored vs 0.197×
CUDA-favored). **You cannot extrapolate the winning axis of one packet
to another packet.**

### 3.2 The 5× pose ratio is OPERATING-POINT-DEPENDENT, not constant

At medal-band `pose_avg ≈ 3.5e-5`:
- `pose_cuda ≈ pose_cpu + sigma²_noise_floor` where `sigma²_floor ≈ 1.4e-4 ≈ 4 × pose_cpu`
- Predicted ratio: ~5×. Observed: 5.18× (A1) / 5.07× (PR106 r2 family).

At high pose (e.g., PR60 AV1 baseline at pose_cpu ~ 5e-3):
- Predicted ratio: 1 + 1.4e-4/5e-3 ≈ **1.03** (essentially zero device-axis drift)

At very low pose (better than 3.5e-5):
- Predicted ratio: blows up further.

**The 5× ratio is a SATURATED FLOOR effect at the medal-band operating
point**, not an intrinsic property of any architecture.

### 3.3 Pose vs Seg asymmetry is CANONICAL (regression vs classification)

The 5× pose drift / 1.17× seg drift asymmetry derives from:
- Regression heads (pose) are O(2^-bits) sensitive to precision noise.
  Per LSQ paper, terminal regression layers have 5-10× higher Hessian
  eigenvalue than classification layers.
- Classification heads (seg) are bounded by argmax stability radius;
  only boundary pixels flip under small logit perturbations.

This asymmetry is INTRINSIC to the scorer architecture and applies across
ALL substrates — but its MAGNITUDE depends on the operating point.

### 3.4 Two mechanism axes are CODE-REAL split sources

Only TWO components have empirically-grounded CPU-CUDA split mechanisms:

- **(a) PoseNet/SegNet forward-pass FP32 computation** (cuDNN reduction
  order + GELU intrinsics)
- **(b) Ground-truth NVDEC vs PyAV decode** (PyAV's "matches nvdec output"
  docstring does NOT guarantee bit-exactness on chroma)

A 25/75 attribution (decoder/forward) is a HYPOTHESIS, not measured
causality. The §9.3 "PyAV decoder + CUDA forward" discriminator
experiment is the highest-EV test to localize the split.

**Rate term, inflated-frame `.raw` decode, BatchNorm stats, seed/RNG,
score formula — all are HARDCODED-IDENTICAL across axes** (no drift
contribution).

### 3.5 MPS is a SEPARATE axis with reclassified drift

Per today's MPS drift formalization landing:
- The CLAUDE.md "MPS PoseNet drift 23×" anchor was real for LEGACY
  archives.
- Current architecture-class (the contest scorer we use today) exhibits
  **0.072% aggregate gap** (Phase B empirical anchor) — 69× below the 5%
  MPS-viable threshold.
- The non-negotiable still binds (MPS is NEVER score-truth + paired
  Linux x86_64 contest-axis required for promotion) — but the magnitude
  of "noise" is architecture-class-dependent.
- Per-pair drift CV = 2.6% (UNIFORM, not fat-tail). Falsifies per-pair
  CUDA-shadow routing as EV-positive.

### 3.6 The marginal-EV analysis FLIPS at the frontier operating point

Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent":
- At old 1.x score operating point: SegNet ~77× more important (marginal).
- At PR106 frontier (pose_avg=3.4e-5): pose is **2.71× more important
  than SegNet** at the marginal-EV-per-byte level.
- Crossover at `pose_avg ≈ 2.5e-4`.

**Total contribution remains seg-dominated at PR106** (seg 0.067 vs pose
0.018), but MARGINAL improvement favors pose at the frontier operating
point.

### 3.7 CPU axis IS the contest leaderboard ranker

PR102 third prize was awarded against its CPU score 0.19538, not its CUDA
score 0.22839. Our internal CUDA-first measurement bias structurally
under-served the contest leaderboard ranking — the dual-eval mandate
(landed 2026-05-08) extincts that bias forward.

---

## 4. WHAT SLOT 30'S ANALYSIS SHOULD INTEGRATE / SUPERSEDE / REFINE

### 4.1 INTEGRATE — these are HARD-EARNED priors

1. **Packet-specificity** (§3.1) — slot 30 must NOT frame device-axis as
   monotone. Cite A1 vs PR106 r2 opposite-winner empirical anchors.
2. **Operating-point dependence** (§3.2) — slot 30 must NOT cite a single
   "5×" or "0.033" number as universal. Both depend on `pose_avg`.
3. **Pose vs Seg asymmetry** (§3.3) — slot 30 must distinguish regression-
   sensitive (pose) from classification-stable (seg) terms.
4. **CPU axis = contest leaderboard ranker** (§3.7) — slot 30 must NOT
   prioritize CUDA when CPU is the ranking axis.
5. **MPS reclassification** (§3.5) — slot 30 must cite the 0.072% Phase B
   anchor, NOT the legacy 23× anchor, when discussing CURRENT scorer
   portability.

### 4.2 SUPERSEDE — these are CARGO-CULTED claims slot 30 should refuse

1. ANY claim that "CPU is 0.01-0.03 better than CUDA on the same archive"
   universally. Anchor: A1's +0.0335 CPU win vs PR106 r2's -0.0215 CUDA
   win on the same HNeRV family.
2. ANY claim that "FastViT-T12 has attention layers compounding TF32 to
   5×". Anchor: T4 has no TF32, FastViT-T12 has no attention,
   `allow_tf32=False` by default in PyTorch ≥ 1.12.
3. ANY claim that "MPS PoseNet drift is 23× universally". Anchor: today's
   0.072% Phase B aggregate for current architecture-class.
4. ANY claim that "per-pair CUDA-shadow routing is EV-positive". Anchor:
   today's CV=2.6% uniform finding falsifies the fat-tail premise.

### 4.3 REFINE — these are PARTIAL prior analyses slot 30 can extend

1. The 4-cell A/B/C/D mechanism-attribution verdict from `cpu_cuda_xray_synthesis_20260511.md`
   was DESIGNED but the CUDA layer-drift captures were never fully
   landed (operator-gated $<0.15 Modal dispatch pending). Slot 30 could
   propose closing this measurement loop.
2. The 25/75 decoder/forward attribution from `cuda_cpu_pose_drift_mechanism_deep_dive_20260508_claude.md`
   §8.1 + §9.3 is a HYPOTHESIS; the discriminator experiment (PyAV decoder
   + CUDA forward) has NOT been run empirically. Slot 30 could
   prioritize.
3. The MPS portability matrix per architecture-class is partial; the
   contest scorer at current model class is empirically VIABLE (0.072%)
   but other architecture classes (SegMap / NeRV / Cool-Chic / Z6) are
   NOT yet measured. Slot 30 could propose canonical per-class drift
   prediction via `tac.mps_diagnostic.drift_predictor.predict_drift(...)`.

---

## 5. OPEN QUESTIONS THE PRIOR ANALYSIS SURFACED BUT DID NOT RESOLVE

These are operator-routable for slot 30 OR follow-on subagents:

### 5.1 Does R_pose actually drop to ~1 at AV1 high-pose substrate?

The additive-noise model predicts YES (PR60 AV1 baseline at pose_cpu ~
5e-3 should show R_pose ≈ 1.03). If empirical R_pose stays at 5, the
additive model is WRONG and we need a multiplicative-saturating model.

### 5.2 Is the 25% NVDEC/PyAV vs 75% forward-kernel attribution actually 25/75?

The discriminator: sample a CUDA eval that uses PyAV-decoded tensors as
shared input but otherwise runs CUDA inference. 1-line code change; runs
on T4 in <60 min; discriminates the model decisively. Highest-EV
experiment in the entire mechanism investigation.

### 5.3 What is the per-layer drift profile shape?

Linear (Welch's law / random walk, model B) vs Exponential (multiplicative
compound, model A) vs Saturated (precision floor, model D)? Discriminates
the canonical model via per-layer activation dumps on CUDA T4 vs CPU.
Requires GPU dispatch + activation hooks. Three xray tools landed but CUDA
captures pending.

### 5.4 Does the device-axis flip between A1 and PR106 r2 reflect a substrate-class boundary phenomenon?

Council Insight 1: each HNeRV-family substrate has a different "signature"
in how its decoded RGB bytes are received by the scorer. The xray output
schema includes `first_divergence` and `stage_compounding` fields
explicitly so this comparison can be made by feeding A1 and PR106 r2
decoded-RGB shared-input tensors through the same tool back-to-back.

### 5.5 What is the empirical per-op epsilon?

Mechanism analysis calibrates to `epsilon_per_op ≈ 1.7e-3 RMS` (from
`sqrt(50) × epsilon = 0.012` observed pose noise). Fitting against the
25-PR paired-anchor dataset would yield a robust estimate. If epsilon is
NOT constant across PRs, the simple additive-noise model is wrong.

---

## 6. STRUCTURAL ANTI-PATTERNS TO AVOID (per CLAUDE.md non-negotiables)

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + the
forbidden-patterns family:

1. **Tag EVERY score by axis**: `[contest-CUDA]` for CUDA promotion truth,
   `[contest-CPU]` for explicit Linux x86_64 leaderboard reproduction,
   `[macOS-CPU advisory only]` / `[MPS-PROXY]` / `[MPS-research-signal]`
   for non-promotable.
2. **DO NOT extrapolate one axis from the other**: the CUDA-CPU gap is
   per-archive empirical (A1: +0.033; PR106 r2: -0.021). Computing
   "CPU prediction = CUDA - 0.025" is FALSE per the matrix.
3. **DO NOT downgrade existing CUDA-only artifacts retroactively**: they
   remain `[contest-CUDA]` with their CUDA-axis truth value. The dual-eval
   mandate is forward-looking.
4. **DO NOT cite legacy 23× MPS anchor as universal**: cite the 0.072%
   Phase B reclassification for current architecture-class.
5. **Slot 30's analysis is NON-PROMOTABLE by construction**: tag every
   emitted artifact `[diagnostic-not-score]` and add Provenance per
   Catalog #323.

---

## 7. Slot 30 framing recommendation

Slot 30's analysis should structure around the FIVE DIMENSIONS the prior
analysis already established (rather than re-discovering):

| Dimension | Canonical signal | Prior anchor |
|---|---|---|
| **Per-archive packet** | A1 vs PR106 r2 sign-flip empirical anchor | §3.1 (device-axis matrix) |
| **Operating point** | Medal-band saturated-floor mechanism | §3.2 (additive-noise model) |
| **Component (seg vs pose)** | Regression-vs-classification asymmetry | §3.3 (LSQ paper + canonical formula) |
| **Mechanism (loader vs forward)** | 4-cell A/B/C/D verdict from xray pipeline | §3.4 (mechanism deep-dive) |
| **Portability axis (CPU/CUDA/MPS)** | Hardware-class-dependent, NOT universal | §3.5 + §3.7 (dual-eval + MPS reclassification) |

If slot 30 extends ALL 5 dimensions empirically (e.g., via the §5.1-§5.5
open questions), the analysis becomes the canonical synthesis. If it
collapses to a single "CPU vs CUDA" framing, it loses the operator's
nuanced premise.

---

## 8. Sister-context inventory (cross-references already loaded)

- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
  CONTEST-COMPLIANT HARDWARE" (canonical statement of the dual-eval rule)
- CLAUDE.md "MPS auth eval is NOISE" (the canonical statement of the
  23× empirical anchor + non-negotiable)
- CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent
  (UPDATED 2026-05-04)" (the operating-point dependence framing)
- CLAUDE.md "Apples-to-apples evidence discipline" (the
  axis/hardware/substrate triple)
- Catalog #1 + #192 (defensive validator gates for MPS-fallback and
  macOS-CPU advisory non-promotion)
- Catalog #317 (local research signal canonical helper for MPS/CPU
  dispatch)
- Catalog #323 (canonical provenance for ALL score-claim rows)
- Catalog #287 (empirical claim with evidence tag — disparity claims
  MUST cite anchor)

---

## 9. SCOPE B sister landing (HF Jobs LEGAL_NATIVE_PLATFORMS)

Bundled in the same commit batch via the same lane: `src/tac/deploy/dispatch_protocol.py::LEGAL_NATIVE_PLATFORMS` extended to include `"hf_jobs"`. This closes slot 26's HF Jobs T4 dispatch blocker (slot 13 wired `_dispatch_hf_jobs` at `tools/operator_authorize.py:2329` but the legal-platforms enum lagged). All 37 dispatch protocol + HF Jobs tests pass. Cross-ref: `feedback_hf_dataset_plus_hf_jobs_implementation_surface_landed_20260519.md` + `feedback_hf_jobs_segnet_surrogate_per_pixel_sister_lane_landed_20260519.md`.

---

## 10. Discipline honored

- **Catalog #229 PV (premise verification)**: 7 premises verified
  pre-edit (LEGAL_NATIVE_PLATFORMS line 73 / `_dispatch_hf_jobs` line
  2329 / `cuda_cpu_pose_drift_mechanism_deep_dive` content / device-axis
  matrix anchors / MPS reclassification empirical / docs/writeup_draft.md
  presence / sister memory files).
- **Catalog #110 / #113 HISTORICAL_PROVENANCE**: this memo is a SYNTHESIS
  of prior analyses, not a mutation. Cited source files are READ-ONLY in
  this scope.
- **Catalog #323 canonical provenance**: this memo emits ZERO score
  claims. All cited scores carry their original axis tags.
- **Catalog #287 evidence-claim-with-tag**: every empirical number cited
  carries its source anchor path inline.
- **Catalog #230 sister-subagent ownership map**: this memo's scope is
  the synthesis surface; sister slot 30 owns the new analysis surface;
  no contended-file edits.
- **CLAUDE.md "Public Disclosure Hygiene"**: this is internal coordination
  context; not a public artifact.
- **CLAUDE.md "Apples-to-apples evidence discipline"**: every score in
  this memo preserves axis + hardware substrate tags.

---

## 11. Lane registry status

- Lane: `lane_prior_context_location_plus_hf_jobs_dispatch_fix_20260519`
- Level: **L1** (impl_complete + memory_entry; strict_preflight not
  applicable for a synthesis memo)
- Gates marked: `impl_complete` (✓) + `memory_entry` (✓)
- 6-hook wire-in: hook #4 cathedral autopilot = **ACTIVE** (slot 30's
  output will route through cathedral consumers per Catalog #335); hook
  #6 probe-disambiguator = **ACTIVE** (this synthesis IS the canonical
  disambiguator between cargo-culted oversimplified disparity framings
  and the nuanced packet-specific / operating-point-dependent /
  mechanism-decomposed framing).

## 12. Cost

$0 editor; ~50 min wall-clock (sweep + read + synthesize + write + SCOPE B
fix + tests).

<!-- # FORMALIZATION_PENDING:pre_framework_supplemental_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_signal_loss_recovery_subagent_per_catalog_344_strict_flip_residual_violation_fix_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
