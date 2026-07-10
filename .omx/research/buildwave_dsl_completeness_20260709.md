# Build-wave #377 — DSL completeness leg: FusedRKernel gap + flagship #332 designed levers

**Date:** 2026-07-09 · **Charter:** #377 build-wave ("Build all unbuilt", operator GO 2026-07-09),
DSL-completeness agent — drive `lever_registry.completeness().unmapped` toward zero with **ZERO
silent leftovers**. **Triality:** DSL leg only (the 4 factories). Equations leg = **N/A —
transcription/compute levers, no new S_tau law** (FusedRKernel is score-neutral compute; the other
three activate already-derived mechanisms). DAG leg = FEED-buildwave-dsl block appended. **means !=
ends:** composition plumbing, the pointer (0.19110) moves only via a byte-closed exact row.

## STORES CONSULTED
- CLAUDE.md §"'Off' is a tracked queue" + §triality "DSL HOLDS every designed lever" (a lever is a
  `Lever` factory only WITH swept non-default intent; generic tuning knobs stay base config).
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md` §9 (BUILT vs ACTIVATED) +
  §10 (#363 coverage: "unmapped = 123 flags; **26 are genuine designed levers** ... the #332
  signal-loss surface exists to close") + open-items row (fused-R bit-identity, L70).
- `.omx/research/t5_crucible/position_S5_lever_ledger_20260707.md` (row 11: Muon raw hyperparams
  "stays in base config"; "113 unmapped trainer flags are orthogonal to the 36 [ledger levers]").
- MEMORY L70 (fused-R MLX-GPU bit-identity localization, #348) · L15/L56 (costate closed-loop) ·
  L76 (LEVER-4 through-R reachability S_R, #268) · L25 (mod-dim ladder = PR95-echo SECONDARY) ·
  L5 (spike-guard = confound-safety) · docs/operating_manual_craft_handoff.md (§8 anti-goldplating,
  §3 risk, §6 attack-own-fixes).
- `src/tac/tests/test_feed07_dsl_wirein.py` (the test pattern) · `lever_registry.emit_stub_lever`.

## What landed (the fold — 4 factories, 9 flags mapped)

`unmapped` **116 → 107** (−9), `stale == []` (no dead/typo drift). All 4 are DSL `Lever` factories
⇒ auto-derived into `lever_registry.completeness()` + `activation_ledger.known_levers()` ⇒ they
sit in `never_fired()`/`duty_to_measure()` for the #247 costate SENSE layer (VERIFIED, not asserted
— test D against an isolated empty ledger).

| Factory | flag(s) held | swept intent (value provenance) | verdict |
|---|---|---|---|
| **FusedRKernel** | `--fused-r-kernel` | True (BooleanOptionalAction). **THE gap P7 flagged.** | score-NEUTRAL compute lever; MEASURED bit-identical fwd + ~1 ULP VJP + ~8% faster + localizes GPU non-determinism 0/28 cross-proc N=10 (L70/#348). Requires `--mlx-device gpu`. |
| **ClosedLoopEikonalControl** | all 6 `--closed-loop-*` | control=True + bump 0.05 / max 0.20 / 2 bumps / 3+3 windows = **the trainer's OWN designed defaults** (verified argparse L10593–10603, not invented) | #292 build-3 costate controller (flagship #332-owed boolean-activation lever) |
| **CurriculumReanchorLevers** | `--curriculum-reanchor-levers` | True (BooleanOptionalAction) | #302 M1 tau-relative re-anchor; requires `--curriculum-event-triggered` |
| **MarginSaliencyReachability** | `--margin-saliency-reachability` | True (store_true, C2-satisfied) | LEVER-4 through-R S_R (#268/L76); distinct from the KKT-waterfill `MarginSaliency` weight lever |

**Scope discipline (why only these 4, not all 26):** the SPEC's 26 designed levers split into
*boolean-activation* mechanisms (swept intent = flip ON — folded faithfully here) and
*value-configured* clusters whose designed-ON magnitude is not a trainer default (e.g.
`--eikonal-steik-weight` defaults 0.0=off; the active weight is design-specific and the viscosity
axis was a poisoned confound, L4). Folding those with an invented magnitude would be a **fake lever**
(NO-FAKE #1 / the "swept non-default intent" discipline). Per operating-manual §8 (anti-goldplating)
+ the contended-file "small commit" constraint, they are documented FOLD-OWED-#332 below rather than
fake-folded — each is a real fold candidate for a sibling/#332 unit that can cite the designed value.

## DISPOSITION TABLE — every one of the 107 remaining unmapped flags (ZERO silent leftovers)

Accounting verified programmatically: 68 FOLD-OWED (17 clusters) + 38 BASE-CONFIG (8 clusters) +
1 MEASURED-EXCLUDED = 107; `unmapped − table = {}` and `table − unmapped = {}` (no missing, no extra).

### A. FOLD-OWED-#332 — genuine designed levers, value-configured (68 flags) — fold candidates, deferred to avoid inventing magnitudes

| cluster | flags | why owed (not folded here) |
|---|---|---|
| eikonal | `--eikonal-steik-weight/-normalized/-norm-eps`, `--eikonal-junction-relax/-tau`, `--eikonal-visco-ca-band/-ca-pairs/-eps-floor/-eps-upper/-margin-factor`, `--eikonal-viscosity-anneal` | weight default 0.0=off; active magnitude design-specific; viscosity a poisoned confound (L4) — needs a cited designed value |
| pose_carrier | `--pose-carrier-fit-pairs/-pitch/-residual-scale/-s-r/-s-t`, `--pose-eps` | companion params to the (mapped) `--pose-carrier` activation; s_r/s_t/pitch magnitudes are fit-specific |
| start_events | `--lane-band-start-event`, `--muon-start-event`, `--seg-chroma-boundary-start-event` | str-choice curriculum event gates; value = a specific event name (companion to event-triggered curriculum) |
| regime_companions | `--logit-adjust-classes`, `--persistence-classes`, `--persistence-recall-weight` | **law-derived** from the held DirectionalBasisRebalance/Persistence levers (SPEC §10) — belong emitted BY the regime lever |
| hardness_sampling | `--hardness-weighted`, `--hardness-band/-oversample/-power/-source` | store_true activation + value schedule (designed sampler) |
| mod_dim_ablation | `--mod-dim-ablation`, `--mod-dim-ablation-k`, `--mod-dim-dynamics` | designed ablation lever; mod-dim ladder = PR95-echo SECONDARY (L25) |
| amplify_margin | `--amplify-form/-margin-target/-persist`, `--additive-margin`, `--margin-target-end` | margin-amplification schedule; magnitudes design-specific |
| code_nuclear | `--code-nuclear-weight/-eps/-ns-iters` | nuclear-norm code regularizer; weight default off |
| seg_reweight | `--seg-spike-reweight`, `--seg-spike-downweight`, `--seg-coherent-upweight`, `--seg-loss`, `--score-domain-loss` | seg-loss reweighting lever (store_true + mode/value) |
| topology_loss | `--cldice-iters`, `--island-dilate-px` | clDice/island-birth companions |
| costate_probe | `--lambda-pre-probe-fd-eps/-iters`, `--annulus-plateau-dwell-windows/-min-epochs/-rel-eps` | costate lambda-probe + plateau-detect params (companion to ClosedLoop) |
| lane_prior | `--lane-prior-phi1-bias-scale`, `--lane-prior-phi1-source-pair` | lane-prior injection params |
| structured_init | `--structured-init-lr/-sdf-clip/-steps/-subsample/-thresh` | SDF structured warm-start init lever |
| seed_anneal | `--seed-anneal-epochs/-shape`, `--seed-blend`, `--seed-lr` | companion params to the held SeedIslandEased lever |
| aa_fine | `--aa-self-orient-fine-cache-cap/-mode` | companion to the held AACoverageRender |
| residual_carrier | `--residual-mode`, `--residual-target-npz` | residual-carrier lever (mode + payload path) |
| l7_defect | `--l7-mult`, `--l7-threshold` | l7 is a MEASURED DEFECT demoted from the default curriculum (CLAUDE.md) — fold only if re-opened |

### B. BASE-CONFIG — argparse-supplied, no swept-score intent (38 flags) — correctly NOT levers

| cluster | flags | rationale |
|---|---|---|
| arch | `--head`, `--n-hidden` | architecture dims — substrate base config |
| muon_raw_hyperparams | `--muon-lr`, `--muon-adamw-lr`, `--muon-momentum`, `--muon-ns-steps`, `--muon-weight-decay` | Muon raw hyperparams "stays in base config" (SPEC S5 row 11); MuonWarmStart holds the swept warm-start/final-frac |
| optimizer_base | `--weight-decay`, `--per-group-grad-clip`, `--hinge-weight` | generic optimizer/loss base knobs (per-group-grad-clip is WitnessStability-owned if swept) |
| basis_bank | `--bank-base/-f0/-n-iso/-n-orient0/-n-scales`, `--wire-s0/-w0` | curvelet bank substrate config; DirectionalBasisRebalance sweeps freq flags, these are the fixed bank |
| runtime_observability | `--profile-timing`, `--mlx-cache-clear-accum`, `--verdict-subprocess`, `--gpu-reorient`, `--tail-live-mq`, `--lane-band-would-fire-telemetry` | score-neutral runtime/observability (defaults-on class); no swept-score intent |
| resume_warmstart | `--resume-model-from`, `--resume-allow-lever-drift`, `--resume-clear-spike-guard`, `--warm-start-epoch`, `--warm-start-weights-only`, `--warmup-epochs`, `--freeze-decoder-fit-codes` | resume/warm-start machinery — launch config, not a score lever |
| spike_guard_safety | `--spike-guard-mode`, `--spike-factor`, `--spike-rollback-frac/-lr-cut/-max/-window` | confound-SAFETY config (L5: spike-guard-mode must default correctly) — safety, not a swept score lever |
| containment | `--containment-mode`, `--containment-damp` | blast-radius containment (the `Contain` object's territory) |

### C. MEASURED-EXCLUDED (1 flag)

| flag | rationale |
|---|---|
| `--mx-compile` | MEASURED NO-GO — mx.compile reintroduces fp-contraction that flips the uint8-STE d_seg argmax; the trainer help itself says "Prefer --fused-r-kernel". A documented dead alternative, not a swept-win lever. |

## Tests (src/tac/tests/test_buildwave_dsl_completeness.py — 8 passing)
composable + resolve==Lever · validate()==[] over BASELINE · FusedRKernel exact-flag · ClosedLoop
6-flag-at-trainer-defaults + typed override · sealed-205 composition parses through
`build_real_trainer_parser` · newly-mapped flags leave `unmapped` + `stale==[]` · activation-ledger
known/never-fired/duty-to-measure · MarginSaliencyReachability distinct from MarginSaliency.
Regression: `test_feed07_dsl_wirein` (4) + `test_lever_registry` (16) green. ruff F clean.

## Owed / next
- **#332**: the 68 FOLD-OWED flags (17 clusters) are the remaining signal-loss surface — each a real
  fold once a sibling/unit can cite the designed-ON magnitude. `emit_stub_lever` gives the review-
  and-accept starting point per cluster; the value flags need a MEASURED/DERIVED active value first.
- No GPU, no launch, $0. Pointer UNMOVED (this is DSL plumbing, a MEANS).
