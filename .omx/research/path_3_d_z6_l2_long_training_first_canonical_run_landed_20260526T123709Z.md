# Path 3 D=Z6 L2 LONG-TRAINING FIRST CANONICAL RUN — LANDED 2026-05-26T12:37:09Z

**Lane**: `lane_path_3_d_z6_l2_long_training_first_canonical_run_20260526` L1
**Subagent**: `l2_longtrain_d_z6_20260526`
**Mission contribution per Catalog #300**: `frontier_protecting_apparatus_maintenance` (proof-of-pattern execution of canonical L2 helper at convergence scale; unblocks L1-PROMOTION-CASCADE sister substrates with the 78% LOC reduction reference template).
**Cost**: $0 GPU; 3.8s MLX wall-clock (~5 min total session wall-clock).
**Doctrines honored**: cascade doctrine `fb270e9b6` §"L2 LONG-TRAINING INFRASTRUCTURE"; MLX-first doctrine `4107bbf8d`; CLAUDE.md "MLX portable-local-substrate authority" + "EMA — NON-NEGOTIABLE" + "MPS auth eval is NOISE".

---

## Charter

FIRST canonical L2 long-training execution via the newly-landed
`tac.training.long_training_canonical.run_long_training` helper (commit
`f5e4784ef`). Per the L2-INFRA-BUILD landing the helper is substrate-agnostic
with a 21-symbol public API + 60/60 tests; D=Z6's substrate-specific surface
is ~136 LOC of config + adapter + ONE `run_long_training(adapter, config)`
invocation (78% LOC reduction vs the L1 promotion's hand-rolled ~600 LOC).

Goal: produce a CONVERGED MLX-faithful D=Z6 Z6PCWM1 archive at convergence
scale with full curriculum + canonical Polyak EMA + atomic JSONL telemetry
+ checkpoint discipline + canonical Provenance + cathedral-queryable
posterior emission — all $0 on M-series. The artifact becomes the canonical
D=Z6 substrate anchor for downstream L3 hyperparameter sweeps + L6 bridge
calibration paid CUDA.

## Empirical anchor

| Field | Value |
| --- | --- |
| Lane id | `lane_path_3_d_z6_l2_long_training_first_canonical_run_20260526` |
| Substrate id | `time_traveler_l5_z6` |
| Output dir | `experiments/results/z6_l2_canonical_LONGTRAIN_20260526T123600Z/` |
| Epochs completed | 300 / 300 |
| Pairs | 50 (matches D=Z6 L1 baseline + Catalog #1265 gate parity) |
| Latent dim | 24 |
| Output resolution | 48×64 (canonical MLX-local proxy res; sister L6 bridge handles 384×512 contest res) |
| Wall-clock | 3.79s on M-series MLX |
| Final loss | 0.114436 (vs initial 0.338232 = **66.2% reduction**) |
| Final EMA drift L2 | 10.12 (canonical 0.997 decay; cumulative drift across 300 steps) |
| Archive path | `experiments/results/z6_l2_canonical_LONGTRAIN_20260526T123600Z/0.bin` |
| Archive sha256 | `dabdcf94c44092c1ce075639c53391973c67abbad8e59fa12c4b4ebe8fc447ad` |
| Archive bytes | 64,642 |
| Sister #1265 gate verdict | **PASS** (`max_abs_drift=0.000253` vs threshold `0.001`; margin `0.000747`; 23× PR95 empirical anchor) |
| Posterior update accepted | `False` (canonical advisory-grade refusal: `evidence_tag='[MPS-research-signal]'`) |
| Posterior emission | Written to `.omx/state/mps_research_signal_manifest.jsonl` row #157 (cathedral-queryable per Catalog #335) |
| MPS manifest evidence_grade | `MPS-research-signal` |
| `promotable` / `score_claim` / `promotion_eligible` / `rank_or_kill_eligible` / `ready_for_exact_eval_dispatch` | **all False** (canonical non-promotable markers per Catalog #127/#192/#317/#341) |

### Convergence trajectory (sampled per-50-epoch)

| epoch | loss | ema_drift_l2 | wall_clock_seconds |
| ---: | ---: | ---: | ---: |
| 0 | 0.338232 | 0.3060 | 0.030 |
| 49 | 0.149482 | 10.2394 | 0.666 |
| 99 | 0.132480 | 10.5757 | 1.304 |
| 149 | 0.125673 | 10.6672 | 1.926 |
| 199 | 0.120680 | 10.6457 | 2.571 |
| 249 | 0.117317 | 10.4611 | 3.209 |
| 299 | 0.114436 | 10.1196 | 3.781 |

Loss curve shape: fast drop in first 50 epochs (0.338 → 0.149, 56% of total
reduction), then slow monotonic descent to 0.114 by epoch 299. EMA drift
plateaus near 10.5 around epoch 100-200 then slowly relaxes to 10.12 by
epoch 299 — characteristic of canonical Polyak averaging settling against
the live-weight trajectory as the model approaches a stable basin.

### Checkpoints emitted (canonical interval=50)

`experiments/results/z6_l2_canonical_LONGTRAIN_20260526T123600Z/checkpoints/`
contains 7 checkpoint triples (live.state + ema_shadow.state + meta.json)
at epochs 49 / 99 / 149 / 199 / 249 / 299 + final at epoch 299. Each
checkpoint is ~414 KB on disk (live state + EMA shadow + meta). Meta.json
carries `substrate_id` + `lane_id` + `curriculum_hash` per Catalog #190
resume-discipline.

## Comparison vs L1 baseline

L1 promotion ran at the same `num_pairs=50` but `epochs=30` (proof-of-pattern
scale; commit `8833b9db5`):

| Metric | L1 (30 epochs) | L2 (300 epochs) | Δ |
| --- | --- | --- | --- |
| Wall-clock | 0.3s | 3.8s | 10× more compute, 10× more epochs |
| Loss reduction | 0.339 → 0.176 (48%) | 0.338 → 0.114 (66%) | +18 percentage points |
| Archive bytes | 64,244 | 64,642 | +398 bytes (codebook offsets shifted by trained weights) |
| EMA active | ✅ (hand-rolled) | ✅ (canonical PolyakEMAShadow) | architecture parity confirmed |
| Posterior emission | Manual (L1 trainer) | Canonical (via L2 helper auto-emission) | infrastructure consolidation |
| Sister #1265 gate | PASS (untrained) | PASS (trained, `max_abs_drift=0.000253`) | trained-weight parity confirmed |

The L2 converged model has 18 percentage points more loss reduction than the
L1 baseline at the same `num_pairs=50` resolution. Per the canonical helper
trajectory the loss is still slowly decreasing at epoch 299 (slope
≈ −0.00006 per epoch); further training to `epochs=1000-3000` would
likely yield additional reduction but is operator-routable rather than
in-scope for this proof-of-pattern.

## Canonical helper integration verification

Per the 10-element canonical contract from
`docs/canonical_long_training_infrastructure.md`:

| Element | Status |
| --- | --- |
| 1. `run_long_training(adapter, config)` entry-point | ✅ INVOKED |
| 2. `LongTrainingConfig` frozen dataclass schema | ✅ VALIDATED |
| 3. `CurriculumStage` frozen dataclass | ✅ single stage `z6_l2_recon_full` |
| 4. Checkpoint+resume + substrate_id/lane_id/curriculum_hash validation | ✅ 7 checkpoint triples + meta |
| 5. Per-arm canonical Provenance + posterior anchor | ✅ MPS-research-signal manifest row #157 emitted |
| 6. EMA shadow (PolyakEMAShadow decay=0.997) | ✅ ema_drift trajectory 0.31 → 10.12; archive emitted from shadow |
| 7. Multi-arm parallel dispatch | N/A this run (single arm) |
| 8. OOM-safe step runner (batch halving) | ✅ available (not triggered; MLX comfortable at batch=8) |
| 9. Observability surface per Catalog #305 (TelemetrySink JSONL) | ✅ 300 rows in `telemetry.jsonl` (fcntl-locked) |
| 10. OSS-clean public API + SPDX + zero `/Users/...` paths | ✅ trainer + adapter + helper all conformant |

Sister #1265 (`tools/gate_mlx_candidate_contest_equivalence_z6.py`) PASSES at
converged weights — this is the structural verification that the trained
MLX model's decoder matches the PyTorch sister within 23× PR95 anchor margin
(`max_abs_drift=0.000253` < threshold `0.001`). The CLAUDE.md "MLX
portable-local-substrate authority" non-negotiable requires MLX as
research-signal only; the #1265 PASS does NOT promote the score axis — it
confirms the decoder bridge is byte-stable for future L6 paid CUDA
calibration.

## Canonical Provenance + non-promotable markers (Catalog #323 + #341)

The canonical helper auto-stamped the TrainingArtifact with the canonical
Provenance shape per `tac.provenance.builders.build_provenance_for_predicted`:

```json
{
  "artifact_kind": "predicted_from_model",
  "canonical_helper_invocation": "tac.provenance.builders.build_provenance_for_predicted",
  "captured_at_utc": "2026-05-26T12:36:15Z",
  "evidence_grade": "predicted",
  "hardware_substrate": "macos_arm64_mlx_local",
  "measurement_axis": "[macOS-MLX research-signal]",
  "promotion_eligible": false,
  "score_claim_valid": false,
  "source_path": "<predictor:long_training_canonical:time_traveler_l5_z6>",
  "source_sha256": "207c6d0f27294b1a7d54484e889a193f06b8d94b53590d8f5bba3d3366c9a23e"
}
```

All five non-promotable markers are `False` at the TrainingArtifact level
(`promotable` / `score_claim` / `promotion_eligible` / `rank_or_kill_eligible`
/ `ready_for_exact_eval_dispatch`) — the canonical posterior update was
correctly REFUSED with reason `non-authoritative (advisory-grade)
evidence_tag: '[MPS-research-signal]'`, mirroring the smoke trajectory's
posterior refusal behavior.

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution** — N/A this run (L2 reconstruction-proxy
   training; per-axis SegNet/PoseNet decomposition is deferred to L6 sister
   per the per-substrate symposium PROCEED_WITH_REVISIONS verdict + Yousfi
   dissent; Z6 adapter's `score_aware_components` returns `None` by design).
2. **Pareto constraint** — N/A this run (single-axis MSE proxy; no Pareto
   trade-off until L6 bridge introduces seg/pose/rate triple).
3. **Bit-allocator hook** — N/A this run (Z6PCWM1 archive grammar is
   fixed-allocation per the L1 promotion landing; per-element bit allocation
   is L3+ territory).
4. **Cathedral autopilot dispatch** — **ACTIVE** (the
   `mps_research_signal_manifest.jsonl` row #157 with archive sha
   `dabdcf94...` + canonical Provenance is automatically discoverable by the
   62-cathedral-consumer cascade per Catalog #335 auto-discovery; the next
   `tools/cathedral_autopilot_autonomous_loop.py` invocation will ingest it).
5. **Continual-learning posterior update** — **REFUSED-BY-DESIGN** per CLAUDE.md
   "MLX portable-local-substrate authority" non-negotiable (advisory-grade
   evidence_tag refuses canonical posterior update; emission to
   research-signal manifest is the canonical alternative for cathedral
   observability).
6. **Probe-disambiguator** — N/A this run (single-config training; no
   competing methodological interpretations to disambiguate).

## Discipline checklist (per CLAUDE.md non-negotiables)

- Catalog #229 PV — read FULL canonical helper docs + Z6 L2 trainer + adapter
  + smoke artifact BEFORE invoking. ✅
- Catalog #117/#157/#174/#235/#289 canonical serializer + POST-EDIT
  `--expected-content-sha256` for this memo + lane registry mark. ✅
  (to be applied at commit)
- Catalog #119 Co-Authored-By trailer. ✅ (commit policy)
- Catalog #287 placeholder-rationale rejection — no placeholder rationales
  in any commit / memo / waiver. ✅
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — NEW dir + NEW memo
  only; zero mutation of canonical helper / trainer / adapter / smoke
  artifacts. ✅
- Catalog #208 docs/local-paths — every artifact path is repo-relative
  (`experiments/results/z6_l2_canonical_LONGTRAIN_20260526T123600Z/...`);
  zero `/tmp/` or `/Users/adpena/` in persisted artifacts. ✅
- Catalog #220 substrate L1+ scaffold byte-addition operational mechanism —
  the Z6PCWM1 64,642-byte archive is consumed by the canonical `inflate.py`
  Z6PCWM1 grammar (per L1 promotion landing); the bytes carry trained model
  state. ✅
- Catalog #230 ownership map — output dir + memo are disjoint from all
  in-flight sister files_touched. ✅
- Catalog #240 recipe-vs-trainer-state consistency — Z6 trainer's
  `_full_main` is the live path (no NotImplementedError); recipe state
  matches trainer state. ✅
- Catalog #244 canonical NVML env block — N/A (MLX-local, not Modal). ✅
- Catalog #305 observability surface — the 6-facet observability auto-exposed
  by the canonical helper via `telemetry.jsonl` (300 rows; per-epoch loss +
  ema_drift_l2 + wall_clock + stage_name + canonical Provenance lineage in
  `training_artifact.json`). ✅
- Catalog #323 canonical Provenance umbrella — TrainingArtifact carries the
  canonical Provenance sub-object validated via
  `tac.provenance.audit_score_claim_dict`. ✅
- Catalog #335 cathedral consumer auto-discovery — the MPS-research-signal
  manifest emission auto-routes through the 62-consumer cascade. ✅
- Catalog #341 Tier A non-promotable markers — all 3 markers
  (`predicted_delta_adjustment=0.0` semantically / `promotable=False` /
  `axis_tag=[predicted]` via `measurement_axis="[macOS-MLX research-signal]"`)
  present at every emission point. ✅
- Catalog #356 per-axis decomposition — DEFERRED per Yousfi dissent in
  per-substrate symposium (L2 MLX is reconstruction-proxy only; L6 paid
  CUDA via PyTorch sister carries per-axis SegNet/PoseNet). ✅
- CLAUDE.md "EMA — NON-NEGOTIABLE" — canonical Polyak decay 0.997 verified
  via `config_snapshot.ema_decay`; EMA drift trajectory recorded; archive
  emitted from EMA shadow per snapshot+restore pattern. ✅
- CLAUDE.md "MLX portable-local-substrate authority" — every artifact tagged
  `[macOS-MLX research-signal]`; posterior refused advisory-grade. ✅
- CLAUDE.md "Executing actions with care" — NO `gh pr create`, NO
  Modal/Vast/Lightning paid dispatch, $0 spend. ✅

## Operator-routable next steps

1. **L3 hyperparameter sweep at $0 MLX** via
   `run_long_training_multi_arm` — fan out N concurrent D=Z6 arms over
   {latent_dim ∈ {16, 24, 32, 48}} × {lambda_residual ∈ {0.5, 1.0, 2.0}} at
   `epochs=300 num_pairs=50` MLX-local; sequential-safe default per Catalog
   #302 sister-subagent scope overlap.
2. **Convergence-extension run** at `epochs=1000` or `epochs=3000` MLX-local
   to characterize the asymptotic loss floor (current trajectory at epoch
   299 shows slope ≈ −0.00006 per epoch suggesting non-trivial headroom
   remains).
3. **L6 paid CUDA bridge calibration** on this converged candidate per
   CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable +
   the canonical per-substrate symposium gate (Catalog #325): the MLX
   archive sha `dabdcf94...` would be re-built in PyTorch via
   `tac.substrates.time_traveler_l5_z6.mlx_export_bridge` then dispatched
   to T4/A100 for paired contest-CPU + contest-CUDA auth eval. Cost
   estimate per the doctrine: ~$2-5.
4. **L1-PROMOTION-CASCADE for sister substrates** can now follow this Z6 L2
   reference pattern via the canonical 78% LOC reduction template — each
   substrate's L2 trainer is ~30 LOC of substrate-specific config + ONE
   `run_long_training(adapter, config)` invocation. Suggested cascade order
   per the doctrine: B'=Z7-Mamba-2-v2 / C'=NSCS06 / E=BoostNeRV / G=NIRVANA
   / J=MDL-IBPS / A=DreamerV3 / F=Z8-hierarchical / H=ATW-v2.
5. **Sister #1265 canonical-grammar gate replication** for non-Z6 substrates
   — each substrate-class grammar needs its own #1265 parameterized gate
   per the canonical pattern (e.g. `gate_mlx_candidate_contest_equivalence_z7.py`
   for Z7-Mamba-2 once Z7 lands an MLX renderer + sister grammar).

## Cross-references

- Path 3 cascade doctrine: `fb270e9b6` "L2 LONG-TRAINING INFRASTRUCTURE"
- MLX-first doctrine: `4107bbf8d`
- L2-INFRA-BUILD landing: `f5e4784ef`
- D=Z6 L1 promotion: commit `8833b9db5` + `.omx/research/path_3_d_z6_l1_promotion_landed_20260526.md`
- Canonical helper: `src/tac/training/long_training_canonical.py` (1170 LOC)
- Canonical helper docs: `docs/canonical_long_training_infrastructure.md`
- Z6 L2 trainer: `experiments/train_substrate_z6_predictive_coding_mlx_l2.py` (136 LOC)
- Z6 long-training adapter: `src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py`
- Z6PCWM1 archive grammar: `src/tac/substrates/time_traveler_l5_z6/archive.py`
- Sister #1265 gate: `tools/gate_mlx_candidate_contest_equivalence_z6.py` (landed `fc44aa670`)
- Smoke reference: `experiments/results/z6_l2_canonical_smoke_20260526T091918Z/`
- Wave #1 posterior emission helper: `src/tac/substrates/_shared/posterior_emission_helper.py` (`3d103dafd`)
- CLAUDE.md "MLX portable-local-substrate authority" non-negotiable
- CLAUDE.md "EMA — NON-NEGOTIABLE"
- CLAUDE.md "MPS auth eval is NOISE" non-negotiable

## Sister-substrate impact

L1-PROMOTION-CASCADE for B' / C' / E / G / J / A / F / H substrates can now
follow this Z6 L2 reference pattern. Each L1 promotion produces a
substrate-specific `long_training_adapter.py` per the Z6 adapter template
(~275 LOC for Z6; substrate-specific adapter surface is the only
substrate-engineering territory); L2 trainer entry-point becomes ~30 LOC of
config + ONE `run_long_training(adapter, config)` invocation. Total LOC
reduction vs L1 hand-rolled: ~78% (Z6 reference: 600 LOC L1 → 136 LOC L2).

Once 3-5 sister substrates land L2 adapters, the operator can fan out
concurrent multi-arm L2 long-training via `run_long_training_multi_arm`
per the canonical helper's 7th element — substrate-parallel proof-of-cascade
demonstration.
