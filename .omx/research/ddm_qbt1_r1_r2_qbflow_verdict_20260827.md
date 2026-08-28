# ddm_qbt1 R1+R2 QBFLOW verdict — pose leg ESCAPED the family wall (first ever); seg leg blocked by CLASS BIRTH (2 of 5 classes born, three never predicted); rate HELD through training; run ended by ENOSPC at step 4,865/5,000

STORES CONSULTED: qbt1 charter + build memo (755f31e9ef) · qbflow rate-rung memo
(c6dee964cb) · no2 §5 gate (d0fe0168b5) · SEALED_TRAINING_FIRE_ORDER.json ·
r1 receipts (governed_n32) · r2 checkpoints/manifests (governed_n32_r2) ·
m143 cross-regime-constant-transfer · m194 paint-not-partition · #315 CE→tau
derived schedule · #208 rare-class init · v14 realization-fidelity memo ·
tv1/tv2 τ-inversion (#1253) · m88/m96 subset-bias axis triple.

score_claim=false everywhere. All rows [macOS-MPS/CPU frozen-scorer advisory,
n32 seeded-stratified smoke scale]. Pointer UNMOVED.

## 1. The two windows (MEASURED)

- **R1** (130 steps, 383 s, rc=0, Metal, chunk 16): mechanism proof. Archive
  107,518 B repeat-identical; resume identity byte-exact; per-5-step
  checkpoints + per-checkpoint re-encode through the real coder all landed.
- **R2** (5,000 steps authorized, launch counter 685, config sha 2a98385ef711…,
  EMA re-resolved via `resolve_ema_law(5000)` → 0.9990793899844618): ran to
  **step 4,865 (97.3%)** at 2.29 s/step, then died rc=120 at 11,138 s.

## 2. R2 death: ENOSPC, not a training failure (MEASURED)

- rc=120 is CPython's failed-to-flush-std-streams exit. Three independent
  0-byte artifacts at the death instant (run.log, safe_run status tmp
  `.resource_safe_run_status.json.53696.tmp`, no traceback) = environment
  failure, not program failure.
- **APDataStore hit 100% full** (474 MB free of 2 TB at diagnosis). The r2
  tree measured 71 GB on disk but only ~8 GB logical: ExFAT 128 KB clusters ×
  ~500 small files per re-encode dir × 973 dirs amplified ~8×.
- Root cause of the preflight miss: the storage projection (8.6 GB) was
  calibrated on r1's 130-step geometry and never rescaled to 5,000 steps —
  **m143 cross-regime constant transfer on the STORAGE axis** (mine, at
  authorization time). The walltime and memory budgets were rescaled; storage
  was not, and the ExFAT cluster amplification was invisible to a logical-size
  model.
- Cure applied: lossless in-place repack (per-step tar, member-count verified
  before dir removal; receipt `REPACK_RECEIPT.json` in the reencoded root) —
  zero bytes lost, ~55 GB freed. Two-landing leg 2 (owed): the launcher's
  storage preflight must project ON-DISK bytes per checkpoint-cadence ×
  steps × filesystem cluster size, not logical bytes at the calibration run's
  step count.
- Loss from the crash: ~nothing. Per-5-step checkpoints (periodic_step_004865
  newest) + the full 4,865-row history + all 973 re-encode receipts survived.
  The P0 resumability discipline paid exactly as designed.

## 3. The trajectory (MEASURED, from checkpoint history — no resume needed)

| step | loss_total | seg flip REALIZED | seg flip NATIVE | pose_mse REALIZED | tau |
|---|---|---|---|---|---|
| 1 | 167.15 | 0.500018 | 0.825304 | 119.84 | 0.150 |
| 130 | 50.18 | 0.251114 | 0.249162 | 0.002264 | 0.147 |
| 1000 | 27.10 | 0.250420 | 0.019208 | 0.001991 | 0.130 |
| 3000 | 26.89 | 0.250400 | 0.017711 | 0.000579 | 0.090 |
| 4800 | 26.85 | 0.250399 | 0.017439 | 0.000454 | 0.054 |

- **POSE: the family wall is ESCAPED.** pose_mse_realized 119.84 → 4.5e-4
  through the REAL render→R→uint8→PoseNet path — the first trained carrier in
  this campaign whose pose descends through realization (every predecessor:
  born-small 66–209×, nr1 349×, W72 46×, W96 185–204× over budget at first
  fit). The charter's open question is answered YES for pose. Caveat: the
  plateau (~3.9e-4 at step 4000, wobble after) is still ~3× above the
  1.25e-4 absolute pose budget (m110), at n32 smoke scale, single seed
  (#1251).
- **SEG: frozen at 0.2504 realized for 4,670 consecutive steps** while the
  native interface field converged 0.825 → 0.0174 (14.4× native/realized
  gap). The realized term IS in the loss (it dominates it: 100×0.2504 ≈ 25
  of loss 26.85) and still never descended — a real wall, not a wiring gap.
- **RATE: HELD through training** — trained archives 107,518 B
  repeat-identical (r1) and per-checkpoint re-encodes stable ~107.5 KB /
  B_hat ~122.8 KB across all 973 checkpoints (r2), under the 137,986 B cap.
  First family whose rate gate survives training.

## 4. The seg mechanism: CLASS BIRTH stopped at 2 of 5 (MEASURED)

Per-class decomposition from r1's 32 retained stage-05 payloads
(`segnet_argmax_u8` vs `target_argmax_u8`; r1 endpoint realized flip 0.2511 ≈
r2's 0.2504 — the aggregate stayed flat to 4 decimals across r2's 4,670 steps,
so the structure below persisted [DERIVED for r2, MEASURED for r1]):

| GT class | area % | flip share % | within-class err % | predicted share % |
|---|---|---|---|---|
| Road | 23.12 | 92.56 | **100.00** | **0.00** |
| Lane | 0.60 | 2.39 | **100.00** | **0.00** |
| Undrivable | 49.58 | 0.00 | 0.00 | 59.42 |
| Movable | 1.25 | 4.99 | **100.00** | **0.00** |
| MyCar | 25.46 | 0.06 | 0.06 | 40.58 |

The realized output is a TWO-CLASS field: every pixel reads Undrivable or
MyCar. Road paints as Undrivable above (8.48% of all px) and MyCar below
(14.64%). The two largest classes were born in the first 130 steps
(0.50 → 0.25); no third class was ever born. The hood hypothesis (0.2504 ≈
MyCar area) is REFUTED — MyCar is nearly perfect.
verdict_scope: instance — the hood-area explanation of qbt1's frozen 0.2504
(stage-03 config, n32 seeded-stratified, single seed); refuted by the full
32-pair decomposition above (MyCar within-class error 0.06%), not a claim
about any other vehicle or scale.

Mechanism attribution (labeled):
- **[DERIVED] The expected-flip-margin law cannot BIRTH a class.** For
  confidently-wrong pixels the flip probability saturates → vanishing
  gradient exactly where the third class must appear (the fixed-β-hosc
  saturation genus). The #315/#686 derived schedule states this as law: CE
  births, tau/margin sharpens. qbt1 ran the margin law from step 0 — the
  w96b "aligned law" receipts were measured on already-born W96 fields;
  transferring them to from-scratch birth was m143 cross-regime transfer
  (my charter miss, named).
- **[DERIVED] The paint, not the partition (m194/v14).** The native interface
  knows the boundaries (0.0174); the interior head's RGB never evokes
  Road/Lane/Movable through the frozen SegNet. v14's margin-optimal prototype
  colors (closed-form from the frozen head) are the corpus's measured cure.
  **ERRATUM (appended 2026-08-28, from ddm_qbt2's source trace — d638d0c5ae):
  the parenthetical "closed-form from the frozen head" is WRONG. No closed-form
  RGB solve exists in the corpus: FP1's (5,3) palette is a 32-pair/100-step
  Adam/CE solve through the full frozen SegNet; SQ1's paint is an Adam solve;
  v14 hard-codes only the Movable triple. The genuine closed-form bank
  (build_frozen_rank4_prototype_bank) lives in the terminal head's 4-dim
  feature quotient, not RGB, and has no head-only RGB inverse. The CURE claim
  survives with corrected provenance: FP1's real-path CE-TRAINED palette is
  the measured paint precedent; "closed-form" does not.**
- **[INFERRED, unmeasured] Pose–seg interior conflict.** Pose descends by
  shaping interior photometry (tv1/tv2: pose lives in interiors); Road-class
  evocation also lives in interiors. The working pose gradient may pin the
  ground-plane paint. Discriminated by the r3 A/B below, not asserted.

## 5. Verdict + the named next rung

- verdict_scope: **INSTANCE (qbt1 stage-03 configuration at n32, single
  seed)** for the seg wall — NOT a family closure. The family record after
  this window: rate PROVEN through training · pose realization PROVEN ·
  seg blocked on class birth with the cure already measured elsewhere in the
  corpus.
- §5 gate: correctly refused (d_seg_hat 0.2504 dominates; control leg absent).
  No admission claim.
- **Next rung (ddm_qbt2): birth-first curriculum on the same vehicle** —
  (a) $0 closed-form prototype-color init of the interior head from the
  frozen SegNet head (v14 cure; #208 rare-class-protected init for
  Lane/Movable), (b) a short CE birth stage BEFORE the expected-flip-margin
  stage (the #315 event-triggered hand-off: exit CE when all 5 classes are
  born and stable), (c) then the existing margin law + pose exactly as built.
  Falsifier: if with prototype init + CE stage the realized field still
  cannot birth Road at n32, the QBFLOW seg leg fails at FORMULATION scope and
  the family table closes on distortion like its predecessors.
- Do NOT resume the last 135 r2 steps: the curve is flat on every axis that
  matters; the marginal information is zero.

## 6. Custody

- r1: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/governed_n32/`
  (checkpoints, re-encodes, 32 retained pair payloads, GATE.json).
- r2: `.../governed_n32_r2/` — 973 periodic checkpoints (newest
  periodic_step_004865.pt, 15.8 MB) + 973 re-encode receipts, repacked as
  per-step tars (lossless, count-verified; REPACK_RECEIPT.json).
- Decomposition inputs: the 32 r1 `pair_*.npz` (this memo §4 is reproducible
  from them with ~20 lines of numpy).

## Observability surface
Per-layer: checkpoint history rows (per-step objective decomposition) +
per-checkpoint REENCODE_MANIFEST (B_hat, shas) + retained per-pair npz
(camera/logits/argmax/pose/targets). Decomposable: §3 per-axis + §4 per-class.
Diffable: r1 vs r2 same schema. Queryable: all JSON/npz on AP custody.
Citeable: config shas 545662e6ad… (r1) / 2a98385ef7… (r2), launch counters
684/685. Counterfactual: checkpoints every 5 steps enable any-window replay.

— end —
