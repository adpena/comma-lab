# Z6 L2 3000ep MLX-LOCAL RESPAWN — STAND-DOWN LANDED 2026-05-26T17:22:32Z

**Subagent**: `z6-l2-3000ep-mlx-local-respawn-20260526` (fresh subagent_id after orphan
`z6-l2-3000ep-extension-20260526` was terminally closed via TaskStop at 2026-05-26T17:10:30Z
per Catalog #206 crash-resume protocol).

**Lane**: `lane_z6_l2_3000ep_mlx_local_respawn_stand_down_20260526` L1
(impl_complete=stand_down_decision + memory_entry).

**Verdict**: **STAND-DOWN per Catalog #340 / #302 / "Subagent coherence-by-default"
non-negotiable.** The TaskCreate #1332 charter scope — extend MLX-local Z6 L2 training
from 300ep → 3000ep — was already executed and canonically landed by sister
**DRIFT-VS-DEPTH-CHAR-D-Z6** (TaskCreate #1305) at 2026-05-26T12:51:30Z (memo
`.omx/research/path_3_d_z6_drift_vs_training_depth_characterization_landed_20260526T125130Z.md`).

Mission contribution per Catalog #300: `apparatus_maintenance` (sister coherence + orphan
cleanup + signal preservation; zero duplicate paid GPU / paid wall-clock; structurally
protects against duplicate work per the convergent-multi-subagent META-pattern).

Cost: $0 GPU; ~5 min M5 Max wall-clock for premise verification + this memo.

---

## Catalog #229 premise verification (PV)

Read in order:

1. `experiments/results/z6_drift_vs_depth_3000ep_20260526T124756Z/training_artifact.json`
   (~1006 KB; 3000 per-epoch metrics; canonical Provenance + non-promotable markers).
2. `experiments/results/z6_drift_vs_depth_3000ep_20260526T124756Z/gate_1265_verdict.json`
   (Catalog #1265 gate PASS; max_abs_drift = 0.000725; threshold 0.001; margin 0.000275;
   ratio 65.9× PR95 empirical anchor).
3. `experiments/results/z6_l2_canonical_LONGTRAIN_20260526T123600Z/training_artifact.json`
   (300ep predecessor reference; loss 0.114436; archive sha `dabdcf94...`).
4. `.omx/research/path_3_d_z6_drift_vs_training_depth_characterization_landed_20260526T125130Z.md`
   (sister #1305 5-anchor canonical landing memo).
5. `.omx/research/path_3_d_z6_l2_long_training_first_canonical_run_landed_20260526T123709Z.md`
   (sister #1299 300ep L2 canonical reference landing memo).
6. `src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py` (Z6 canonical adapter).
7. `experiments/train_substrate_z6_predictive_coding_mlx_l2.py` (Z6 L2 entry-point).
8. `src/tac/training/long_training_canonical.py` (canonical helper; `run_long_training`
   + `LongTrainingConfig` + `TrainingArtifact` + `resume_from_checkpoint` support).
9. `.omx/state/canonical_equations_registry.jsonl` (4 anchor_appended events for
   `mlx_pytorch_drift_vs_training_depth_z6_v1` + registered event for
   `mlx_drift_accumulation_engineering_response_v1`).
10. `.omx/state/subagent_progress.jsonl` (sister in-flight set + orphan terminal row).

**Empirical anchor**: the 3000ep MLX-local artifact at archive sha
`fbe405e08e651743552648d1ca61819e0b75d383bf4a585825e3b6937bc448d8` (64,804 bytes) was
emitted by `experiments/train_substrate_z6_predictive_coding_mlx_l2.py` via the canonical
`tac.training.long_training_canonical.run_long_training` helper, took 34.2 seconds
M5 Max wall-clock (NOT 12-24h as the original charter projection estimated; the canonical
helper is dramatically faster than the worst-case projection because the Z6 MLX renderer
at 48×64 resolution + 50 pairs + AdamW is bandwidth-bound, not compute-bound), and PASSED
Catalog #1265 gate at the saturation regime.

## Honest convergence trajectory (5-anchor empirical fit per sister #1305)

| epochs | wall_s | loss_final | loss_reduction | ema_drift_final | max_abs_drift | ratio_pr95 | verdict |
|-------:|------:|-----------:|---------------:|----------------:|--------------:|-----------:|:--------|
| 300 | 3.8 | 0.1144 | 66.2% | 10.120 | 0.000253 | 23.0× | PASS |
| 500 | 5.8 | 0.1020 | 69.8% | 9.249 | 0.000358 | 32.6× | PASS |
| 1000 | 11.4 | 0.0958 | 71.7% | 5.611 | 0.000458 | 41.6× | PASS |
| 2000 | 22.0 | 0.0789 | 76.7% | 3.840 | 0.000721 | 65.5× | PASS |
| 3000 | 34.2 | 0.0793 | 76.6% | 2.703 | 0.000725 | 65.9× | PASS |

**Empirical fit**: `drift = 1.81e-5 * epochs^0.47` (R² = 0.971; sub-linear, asymptotic;
falsifies the n=2 super-linear `epochs^1.45` extrapolation that motivated the original
TaskCreate #1332 charter; #1305 ratified the asymptotic regime).

**Loss saturation**: 76.7% (2000ep) → 76.6% (3000ep) = -0.1% (within noise floor; the
loss curve has saturated). The predicted_band [-0.015, -0.005] derived from
COMPREHENSIVE-ROADMAP-2026-05-26 commit `9a0574da9` was conditioned on continued
loss reduction past 1000ep; the empirical anchors show convergence has already saturated
by 2000ep on this Z6 MLX-local config (50 pairs, latent_dim=24, 48×64 res).

## Catalog #344 canonical equation state

Already up-to-date per sister #1305:

```
equation_id:    mlx_pytorch_drift_vs_training_depth_z6_v1
anchors:        5 (300/500/1000/2000/3000ep)
event_lifecycle: registered + 4 × anchor_appended (one anchor came with registration)
well-calibrated: True (max residual ~10% at 2000ep)
trigger:        when_3+_new_empirical_anchors_in_domain
consumers (2):  tools.gate_mlx_candidate_contest_equivalence_z6,
                tac.cathedral_consumers.canonical_equation_lookup_consumer
producers (2):  experiments.train_substrate_z6_predictive_coding_mlx_l2,
                tools.register_z6_drift_vs_depth_equation
```

A second equation `mlx_drift_accumulation_engineering_response_v1` (TaskCreate #1309)
remains at `registered` lifecycle with 0 anchor_appended events; this is the
operator-routable PROVISIONAL → EMPIRICAL-ANCHOR-VALIDATED transition for sister to
land if/when the Carmack-Hotz Kahan-EMA / response-windows engineering response
becomes empirically anchored (sister TIER1-T3-OP2-OP3 already empirically demonstrated
Kahan provides 0× mitigation at Z6 L2 fp32 1000ep per #1305 verdict; the no-op
empirical anchor is itself a candidate row for that equation but it is sister-owned
and OUT of this respawn's scope).

## Catalog #340 sister-checkpoint guard

Pre-edit check: `.omx/state/subagent_progress.jsonl` shows 5 sister subagents in flight
(`nscs06_v8_chroma_lut_mlx_iteration_20260526`, `z7_mamba2_mlx_scaffold_ext_20260526`,
`pr110-opt-frame0-bundle-20260526`, `hinton-mlx-local-pivot-20260526`,
`boostnerv-pr110-l1-empirical-mlx-20260526`). None touches `experiments/train_substrate_z6_*`,
`src/tac/substrates/time_traveler_l5_z6/*`, or
`experiments/results/z6_drift_vs_depth_*`. STAND-DOWN does not edit any source files;
only this landing memo + one checkpoint row are emitted. Sister-coherence preserved.

## Why STAND-DOWN, not OVERRIDE

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable: *"Two subagents working on
the same lane is a registry failure, not a coordination failure."* The 3000ep MLX-local
canonical artifact already exists. Re-running it would:

1. Produce a byte-different archive (different timestamp; `mx.random.seed(0)` deterministic
   but cumulative file IO + wall-clock metadata differ) → false drift signal in
   `mlx_pytorch_drift_vs_training_depth_z6_v1` posterior.
2. Bloat `experiments/results/` with redundant 1+ MB per-epoch telemetry JSONL.
3. Burn unnecessary M5 Max wall-clock (~34s × N retry rounds) on demonstrably-saturated
   loss curve.
4. Violate Catalog #298 substrate retirement / artifact-lifecycle discipline by adding
   a duplicate LIVE_RECIPE-class artifact when an identical canonical artifact exists.

The honest operator-routable next step is consumption of the existing artifact, not
re-production.

## 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map**: N/A — STAND-DOWN does not produce new signal; the
  existing #1305 anchors already route through `tac.sensitivity_map.*` consumers via
  the canonical equation lookup consumer.
- **Hook #2 Pareto constraint**: N/A — no new constraint added.
- **Hook #3 bit-allocator**: N/A — no new bit allocation signal.
- **Hook #4 cathedral autopilot dispatch**: N/A — the canonical equation already
  surfaces 3000ep observability annotations via auto-discovered
  `tac.cathedral_consumers.canonical_equation_lookup_consumer` (Catalog #335).
- **Hook #5 continual-learning posterior**: N/A — existing 5 anchors in
  `mlx_pytorch_drift_vs_training_depth_z6_v1` posterior already capture the 3000ep
  empirical signal per #1305 landing.
- **Hook #6 probe-disambiguator**: N/A — STAND-DOWN itself is a disambiguator between
  re-execute-vs-consume routing for the operator + future sister subagents (this memo
  is the canonical reference).

## Catalog #287 + #323 canonical Provenance + non-promotable markers

This STAND-DOWN landing memo carries no score claim (`score_claim=false`). The cited
empirical artifacts ALL carry canonical Provenance + non-promotable markers per
Catalog #127/#192/#317/#341 (axis_tag = `[macOS-MLX research-signal]`,
hardware_substrate = `darwin_arm64_apple_silicon`, evidence_grade = `predicted`,
promotion_eligible = false). No literal score values are claimed in this memo
(Catalog #343 frontier pointer compliance).

## HORIZON-CLASS classification per Catalog #309

The 3000ep MLX-local artifact remains in `HORIZON_CLASS_UNCLASSIFIED` (per the
non-promotable-by-construction Catalog #341 routing-markers contract); the only way
to assign a HORIZON-CLASS to this work would be a paired contest-CPU + contest-CUDA
auth eval per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" — explicitly
**OUT OF SCOPE for this slot** per operator's "all on MLX" directive.

## Sister coherence cross-references

- **Sister TaskCreate #1299** (`l2_longtrain_d_z6_20260526`): 300ep canonical reference.
  Memo: `.omx/research/path_3_d_z6_l2_long_training_first_canonical_run_landed_20260526T123709Z.md`.
- **Sister TaskCreate #1305** (`DRIFT-VS-DEPTH-CHAR-D-Z6`): 5-anchor extension to 3000ep.
  Memo: `.omx/research/path_3_d_z6_drift_vs_training_depth_characterization_landed_20260526T125130Z.md`.
- **Sister TaskCreate #1309** (`TIER1-T3-OP1-OP4`): canonical equation
  `mlx_drift_accumulation_engineering_response_v1` PROVISIONAL registration.
- **Sister TaskCreate #1307** (`TIER1-T3-OP2-OP3`): Kahan-EMA shadow wrapper +
  Carmack 30-min smoke (0× empirical mitigation at Z6 L2 fp32 1000ep).
- **Sister TaskCreate #1310** (`COUNCIL-RECURSIVE-SELF-REFLECTION-PROTOCOL`):
  Catalog #363 META protection.

## Operator-routable next step

Two operator-decisions are unblocked by the existing 3000ep artifact (NOT this STAND-DOWN
memo's scope to execute):

1. **CONTEST-CUDA VALIDATION** (paid dispatch envelope ~$0.50-5): paired Modal/Vast
   contest-CPU + contest-CUDA auth eval on the 3000ep MLX-local archive sha `fbe405e0...`
   to either (a) ratify the MLX-local proxy as contest-grade or (b) measure the
   MLX→CUDA Δscore at the saturation regime + update the canonical equation. Explicitly
   gated by operator's "Submission auth eval — BOTH CPU AND CUDA" non-negotiable + the
   "all on MLX in this slot" current directive (which forbids paid dispatch IN this slot
   but does not preclude a future operator-decision slot).

2. **Z6 PHASE 3 RE-CONVENE** per Catalog #325 per-substrate symposium discipline if the
   loss-saturation observation (76.7% → 76.6% delta from 2000ep → 3000ep) suggests the
   Z6 architectural ceiling on this MLX-local config (50 pairs / 48×64 / latent_dim=24)
   has been reached; the symposium's next-step routing would be either L3 hyperparameter
   sweep at higher latent_dim / larger pair budget OR Z6 → Z7/Z8 substrate-class shift
   per the Mamba2 / hierarchical-predictive-coding sister memos.

Both decisions are deferred to operator per CLAUDE.md "Executing actions with care".

## Discipline declarations

- Catalog #229 PV: 10 source files + 2 sister landing memos read BEFORE this memo
  drafted.
- Catalog #117 / #157 / #174 canonical serializer: this STAND-DOWN memo will be
  committed via `tools/subagent_commit_serializer.py` with POST-EDIT
  `--expected-content-sha256` per the standing protocol.
- Catalog #206 crash-resume: fresh `z6-l2-3000ep-mlx-local-respawn-20260526`
  subagent_id verified empty before initial checkpoint; orphan
  `z6-l2-3000ep-extension-20260526` terminal row preserved at
  2026-05-26T17:10:30.178677+00:00.
- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE: zero mutations to existing
  forensic artifacts (sister landing memos, gate verdicts, training_artifact.json).
- Catalog #230 sister-subagent ownership map: no overlap with the 5 in-flight sisters.
- Catalog #287 placeholder-rationale rejection: every rationale ≥4 chars + substantive.
- Catalog #340 sister-checkpoint guard: STAND-DOWN protects against the canonical
  duplicate-work failure mode; this memo IS the canonical reference for future sister
  subagents that may be respawned with the same charter.
- Catalog #343 frontier-pointer-only: no hardcoded frontier-band score literals.
- Catalog #344 canonical equation reference:
  `mlx_pytorch_drift_vs_training_depth_z6_v1` already covers the 3000ep empirical
  anchor; no new anchor_appended event needed because the empirical anchor is already
  in the registry.

---

**STAND-DOWN FINAL VERDICT**: Charter scope completed by sister TaskCreate #1305 ~6
hours before this respawn. Honest operator-facing report below in the response body.
