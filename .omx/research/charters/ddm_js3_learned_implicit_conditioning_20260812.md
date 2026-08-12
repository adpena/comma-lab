# ddm_js3 — learned implicit-conditioning actuator (the routed seg training leg)

Successor of ddm_js2b per its typed route
(/Volumes/VertigoDataTier/pact/ddm_js2b_20260812/training_route/ROUTE.json):
direct two-W4-code continuation on the CP135/F26 state is FORBIDDEN (F1+F2
receipt); the allowed successor is a REPRESENTATION-CHANGING learned
implicit-conditioning actuator. This charter is that successor's design+build
leg. The long training burn itself is MAIN-fired (governed launcher); this arm
builds to admission and runs bounded screens only.

## MISSION

Build and smoke a small LEARNED conditioning module for the cp135-family seg
engine that flips GT-disagreeing pixels PAST the measured robustness bar —
margin-robust flips that survive the CPU→CUDA instrument transfer — at a
counted-byte price that keeps joint ΔS negative. The seg leg owes ≥ −0.004 S
(≈ 4,700 CUDA flips) of the −0.011955 gap to sub-0.15.

## THE DESIGN KEY (consume js2b's calibration as the LOSS)

js2b measured the conservative robustness bar on the REAL T4 custody planes:
**δ = 0.08036041259765625** SegNet logit-margin units (the lowest-margin
15,421-of-50,389 rank edge), plus the full local-error margin distribution
(p50 0.1603, p90 0.5279). js2b's failure teaches the objective: raw flips are
worthless — 15/15 beneficial catalog flips were tie-fragile. Therefore the
training loss REWARDS post-correction margin beyond δ (hinge at δ, e.g.
sum over GT-disagreeing pixels of max(0, δ − m_correct(x)) with m_correct the
signed margin toward the GT class) and PRICES collateral at the same hinge.
Robustness is IN the objective, not a post-hoc filter.

## FORM (candidate family — derive, then build the smallest real one)

- An edge/context-conditioned modulation of the semantic render: a tiny conv
  or FiLM-generator net (counted weights, target ≤ ~4 KB fp16/int8-QAT)
  reading FREE decoder-side context (the decoded partition/edge geometry,
  margin field derivable at decode = rule-118 free compute) and emitting a
  bounded correction to the pre-R RGB planes or the renderer's conditioning.
- Trained through the REAL chain: receiver render → correction → R
  (bicubic/bilinear per upstream) → uint8-STE → frozen CPU-torch SegNet →
  hinge-at-δ objective + pose guard term on the custody pose planes.
  (hr1 realization law: round-trip in ALL transforms, in-loop.)
- Weights counted honestly: real coder (Brotli-q11 race) on the exported
  module; +bytes enters the joint ΔS arithmetic.

## BINDING LAWS (inherited + specific)

1. RELATIVE GAUGE (js2b law 1): baseline 50,389 local flips, batch 16 /
   8 threads fixed. Absolute local d_seg never claimed as progress
   [macOS-CPU advisory, instrument floor 0.0131 S].
2. ROBUST FLIPS ONLY: report Δflips at margin ≥ δ separately; tie-fragile
   flips are not progress.
3. POSE GUARD: local d_pose on custody pose planes; regression ≥ 2e-6
   disqualifies.
4. TRAINING FIRE IS MAIN'S: build the trainer resumable (state on disk,
   per-stage checkpoints P0), emit a sealed launch recipe; do NOT launch the
   long burn. Bounded smokes (≤ ~30 min wall-clock, n32 stratified-random)
   are in-scope for the arm.
5. T4 acceptance rows: QUEUED-WITH-A-FIRE-ORDER for MAIN when a trained
   checkpoint projects n600 robust Δflips ≤ −2,000 at ≤ +1,500 counted bytes.
6. Payload law P0: every trained checkpoint, exported module, candidate
   archive, argmax field persisted with sha256+bytes to
   /Volumes/VertigoDataTier/pact/ddm_js3_20260812/.
7. Serializer commits, --no-co-author, post-edit shas, tags
   [no-triality] [p0-ledger-ok]; no REVIEW_GATE_OVERRIDE on .py; blocked-git
   → commit_intent patch + receipt for MAIN.

## OPTIMAL FORM

- Reference form: scorer-in-loop trained correction through the real
  receiver + R + uint8-STE + frozen SegNet (the tb1/jd-chain training physics
  transplanted to the cp135 family), with the js2b δ-hinge objective.
  Receipts: ROUTE.json + FINAL_RESULT.json (js2b) ·
  .omx/research/ddm_hr1* (realization engineering) ·
  .omx/research/ddm_rvs1* (survival playbook).
- SCOPE reductions (legal): n32 stratified-random screens; short smokes;
  module capacity ladder starting tiny. Any queued fire-order row must be
  n600-projected on the local instrument.
- MECHANISM reductions (TOY-BRACKETED, no family verdict): correction NOT
  through the real R/uint8 chain; loss without the δ-hinge; entropy-estimate
  byte prices instead of real coder.
- Provenance pins: cp135 archive sha 6eb1a3b79cb167e03372339e07e93cae13b6
  ba3114a9eb917288bb038622edb6; custody manifest at
  /Volumes/VertigoDataTier/pact/ddm_js2_20260812/instrument_validation_cuda/
  scorer_input_cache_tensors/manifest.json; δ + margin distribution at
  ddm_js2b FINAL_RESULT.json (sha-pinned there).

## DELIVERABLES

1. Derivation note: the δ-hinge objective written down (facts → objective),
   module family chosen with capacity/byte ladder.
2. The trainer (resumable, per-stage ckpts, governed-launch recipe) + tests.
3. Bounded n32 smoke: does the gradient move robust flips at all? (existence
   signal, not a verdict) + measured s/step → derived burn schedule.
4. Sealed launch recipe for MAIN (config, memory preflight at real scale,
   wall-clock projection).
5. Honest negative if the gradient is dead through uint8-STE at this family:
   verdict_scope labeled, mechanism named.

## FALSIFIERS

- F1: bounded smoke shows zero robust-flip movement over ≥ 300 steps at any
  capacity rung while train hinge-loss descends → realization gap through
  uint8/R (report per-stage attribution).
- F2: capacity ladder shows robust flips scale so weakly that projected
  n600 cost > +1,500 B for −2,000 robust flips at every rung → family
  rate-dominated on this vehicle.
- F3: pose guard fails at every capacity → coupling wall; report map.
