---
council_tier: T1
council_attendees: [phase-b-mps-gap-infrastructure-build-subagent]
council_quorum_met: false
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "MPS-trained weights produce CUDA-forward components within a usable tolerance for substrate training"
    classification: CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION
    rationale: "Predecessor falsified the Conv2d wrapper hypothesis empirically (drift 2.205e-3 with vs without wrapper). End-to-end SegNet drift on synthetic noise was 7.6e-5. This recipe's verdict on REAL contest frames is the open question."
  - assumption: "The build-then-await pattern correctly separates infrastructure build from operator-greenlight dispatch"
    classification: HARD-EARNED
    rationale: "Pattern proven by sister recipes (e.g. lane_time_traveler_l5_z6 declares research_only:true + dispatch_enabled:false at landing pending Phase 2 sextet-pact council consensus; operator_authorize.py refuses non-dispatchable recipes)"
council_decisions_recorded:
  - "op-routable #1: Operator reviews this memo + flips dispatch_enabled to true when ready"
  - "op-routable #2: Local MPS training runs first (tools/run_mps_gap_experiment.sh) which produces the checkpoint Modal needs"
  - "op-routable #3: Modal dispatch is one paired-env operator command; this subagent does NOT fire it"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
deferred_substrate_id: null
related_deliberation_ids:
  - mps_conv2d_wrap_fix_plus_real_frame_gap_experiment_20260518
  - mps_local_compute_frontier_diagnostic_20260518
---

# Phase B MPS-train CUDA-score gap experiment — infrastructure landed (no dispatch)

**Lane**: `lane_phase_b_mps_gap_experiment_infrastructure_build_20260518` L1
**Predecessor**: `lane_mps_conv2d_wrap_fix_plus_real_frame_gap_experiment_20260518` (commit `24278cf06`)
**Evidence grade**: infrastructure-only / no score claim
**Score claim**: false
**Promotion eligible**: false
**Axis tags**: every artifact emitted by this package tagged `[MPS-research-signal]` or `[diagnostic-CUDA Modal A10G]`
**Cost**: $0 (editor only; Modal $0.50 envelope RESERVED but NOT consumed)

## What was built

Per the build plan in this subagent's prompt:

| File | Role | LOC |
|---|---|---:|
| `src/tac/mps_gap_experiment/__init__.py` | Public API + package docstring | ~60 |
| `src/tac/mps_gap_experiment/tiny_renderer.py` | ~7K-param FiLM-conditioned RGB pair renderer | ~180 |
| `src/tac/mps_gap_experiment/train_on_mps.py` | MPS local training loop (EMA(0.997) + eval_roundtrip + canonical artifacts) | ~260 |
| `src/tac/mps_gap_experiment/train_on_mps_cli.py` | argparse front-end for the local training step | ~50 |
| `src/tac/mps_gap_experiment/harvest_and_verdict.py` | Per-component gap computer + verdict classifier | ~210 |
| `src/tac/tests/test_mps_gap_experiment_tiny_renderer.py` | 10 dedicated tests (all pass) | ~130 |
| `.omx/operator_authorize_recipes/mps_gap_experiment_tiny_renderer_modal_a10g_dispatch.yaml` | Modal A10G recipe with `dispatch_enabled: false` | ~140 |
| `experiments/mps_gap_experiment_a10g_dispatch.py` | Modal dispatch entry point with Catalog #151 manifest | ~140 |
| `scripts/remote_mps_gap_experiment_a10g.sh` | Modal A10G remote driver (canonical NVML block) | ~50 |
| `tools/run_mps_gap_experiment.sh` | Local-side harness (train + greenlight + dispatch) | ~80 |

## How to fire the dispatch (operator one-command pattern)

After reviewing this memo, the operator flips ONE line in the recipe:

```yaml
# .omx/operator_authorize_recipes/mps_gap_experiment_tiny_renderer_modal_a10g_dispatch.yaml
- dispatch_enabled: false
+ dispatch_enabled: true
```

Then a single command runs:

```bash
# Local MPS training first (~5-10 min wallclock)
bash tools/run_mps_gap_experiment.sh

# After training prints the checkpoint location, re-run with paired-env auth:
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=0.50 \
bash tools/run_mps_gap_experiment.sh
```

The first invocation produces the local checkpoint and stops at the paired-env gate (per Catalog #199). The second invocation reuses the local checkpoint and fires the Modal A10G dispatch.

## Expected wall-clock

- Phase 1 (local MPS training): ~5-10 minutes for 100 epochs on 10 real-frame pairs (verified empirically: 3 epochs / 4 pairs ran in 0.4s)
- Phase 2 (Modal A10G dispatch): ~10 minutes including image build + mount
- Phase 3 (local harvest + verdict): ~2 minutes
- **Total**: ~17-22 minutes wallclock; $0.50 GPU spend

## Expected output

`experiments/results/mps_gap_experiment_a10g_dispatch/gap_results.json` containing:

```json
{
  "target_device": "cuda",
  "mps_reference_device": "mps",
  "verdict": "LOCAL_MPS_TRAIN_VIABLE" | "LOCAL_MPS_TRAIN_VIABLE_ADVISORY_ONLY" | "LOCAL_MPS_TRAIN_NOT_VIABLE_PIVOT_MLX_OR_VTOOLBOX",
  "gap_relative_aggregate": <float>,
  "components": [...]
}
```

## Verdict thresholds (per build plan)

| `gap_relative_aggregate` | Verdict | Operator next step |
|---|---|---|
| `< 5%` | `LOCAL_MPS_TRAIN_VIABLE` | Recommend Catalog #317 scope-narrowing for MPS opt-in across substrate trainers |
| `5-20%` | `LOCAL_MPS_TRAIN_VIABLE_ADVISORY_ONLY` | Use local-MPS for advisory ranking only; not promotion-grade |
| `>= 20%` | `LOCAL_MPS_TRAIN_NOT_VIABLE_PIVOT_MLX_OR_VTOOLBOX` | Pivot to MLX or VideoToolbox-decode + CUDA-train |

## 9-dimension success checklist evidence

Per Catalog #294.

1. **UNIQUENESS** — Distinct from sister `tac.mps_diagnostic.layerwise_drift` (layer-by-layer drift); this package measures END-TO-END training-loop survival.
2. **BEAUTY + ELEGANCE** — 10 dedicated tests pass; tiny renderer is 7K params + ~180 LOC; the full package reviewable in 30 seconds.
3. **DISTINCTNESS** — Build-then-await pattern (`dispatch_enabled: false` at landing) is the canonical "infrastructure built but dispatch withheld" pattern (mirrors `lane_time_traveler_l5_z6` recipe).
4. **RIGOR** — Catalog #229 premise verification BEFORE every edit (10 PVs done). Catalog #117 / #157 / #174 canonical serializer commit pending.
5. **OPTIMIZATION PER TECHNIQUE** — N/A (diagnostic infrastructure, not a substrate).
6. **STACK-OF-STACKS-COMPOSABILITY** — `compute_gap_components` accepts arbitrary target device; downstream gap-classifier-consuming subagents can plug in.
7. **DETERMINISTIC REPRODUCIBILITY** — `build_tiny_renderer(seed=N)` is deterministic (tested); `torch.manual_seed` + `torch.Generator().manual_seed(seed + epoch)` per-epoch shuffle pinned.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — Training loop ~0.13s/epoch on MPS (3-epoch smoke).
9. **OPTIMAL MINIMAL CONTEST SCORE** — N/A (no direct score contribution). Indirect: if verdict is VIABLE, unlocks free local-MPS compute for ALL future substrate training (huge mission-positive signal).

## Observability surface

Per Catalog #305.

1. **Inspectable per layer** — every Conv2d / Linear in `TinyRenderer` is named + accessible via `model.named_modules()`.
2. **Decomposable per signal** — `GapManifest.components` carries per-component gap rows (pixel_l1 + optional SegNet/PoseNet outputs).
3. **Diff-able across runs** — `gap_results.json` is canonical JSON; sister runs can `jq`-diff component-by-component.
4. **Queryable post-hoc** — `training_metrics.json` + `gap_results.json` are JSON; consumable by autopilot ranker as advisory signal.
5. **Cite-able** — every manifest carries `(target_device, mps_reference_device, num_pairs, seed)` tuple + `evidence_grade` + `axis_tag`.
6. **Counterfactual-able** — `compute_gap_components` accepts arbitrary `target_device` for counterfactual probes (CPU vs CUDA, etc.).

## Canonical-vs-unique decision per layer

Per Catalog #290.

| Layer | Decision | Rationale |
|---|---|---|
| `TinyRenderer` architecture | UNIQUE | Trivial diagnostic renderer; no canonical exists; the FiLM-conditioned pattern is well-known but the specific 7K-param size + 16ch hidden is bespoke for the diagnostic |
| `EMA(0.997)` weight tracking | ADOPT_CANONICAL | `tac.training.EMA` per CLAUDE.md "EMA — NON-NEGOTIABLE" |
| Eval roundtrip | ADOPT_CANONICAL_PATTERN | `_eval_roundtrip` mirrors the canonical uint8 quantization round-trip |
| `load_default_scorers` | ADOPT_CANONICAL | `tac.scorer.load_default_scorers` per the slot 1 finding |
| Modal recipe schema | ADOPT_CANONICAL | mirrors sister `substrate_time_traveler_l5_z6_modal_t4_dispatch.yaml` shape |
| Catalog #244 NVML env block | ADOPT_CANONICAL | required for any `scripts/remote_*` driver |
| Catalog #199 paired-env discipline | ADOPT_CANONICAL | harness script enforces the paired env vars |
| Non-promotability manifest | ADOPT_CANONICAL | mirrors `tac.optimization.mps_research_signal` Catalog #192 pattern |

## Cargo-cult audit per assumption

Per Catalog #303.

| Assumption | Classification | Unwind path |
|---|---|---|
| The Modal dispatch will produce a usable MPS-vs-CUDA gap signal | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION | This recipe IS the unwind test |
| 100 epochs is the right MPS training budget | HARD-EARNED-WITH-NUANCE | Predecessor's drift was measured after 0 training; this recipe extends to a full training loop |
| Tiny renderer is small enough for the diagnostic | HARD-EARNED | 7K params; <15K budget; verified by `count_params` test |
| The build-then-await pattern is the right operator-spend-control surface | HARD-EARNED | Mirrored from sister `lane_time_traveler_l5_z6` recipe |

## Predicted ΔS band — N/A

This is a diagnostic recipe; the verdict is a `gap_relative_aggregate` float, not a contest ΔS. The recipe declares `predicted_band_validation_status: phantom_pre_training` per Catalog #324 + `research_only: true` + `dispatch_enabled: false` per Catalog #240 transparent non-promotable. The `predicted_band` field is `[0.18, 0.30]` as a placeholder; no Tier-C / Dykstra-feasibility check is required because the recipe never produces a contest score.

## 6-hook wire-in (Catalog #125)

1. Sensitivity-map contribution — **N/A**: diagnostic infrastructure; no score-axis contribution
2. Pareto constraint — **N/A**
3. Bit-allocator hook — **N/A**
4. **Cathedral autopilot dispatch hook — ACTIVE**: a VIABLE gap verdict will route into the autopilot's backend-selection logic (advisory signal for "this substrate can be trained locally"); the consumer will be added in a follow-on lane once the verdict empirically lands
5. Continual-learning posterior — **ACTIVE**: every dispatch registers a call_id in `.omx/state/modal_call_id_ledger.jsonl` via the canonical `register_dispatched_call_id` (Catalog #245); the gap verdict is appended as a probe outcome via `register_probe_outcome` (Catalog #313)
6. Probe-disambiguator — **ACTIVE**: this entire infrastructure IS the disambiguator between "MPS-train viable for substrate training" vs "MPS-train requires backend pivot"

## Lane registry evidence

- `impl_complete=true`: 5 modules + 10 dedicated tests + 1 recipe + 1 dispatch entry + 2 shell scripts
- `real_archive_empirical=false`: no archive (diagnostic only)
- `contest_cuda=false`: never run on contest-axis hardware
- `strict_preflight=N/A`: research infrastructure
- `three_clean_review=false`: not a contest substrate
- `memory_entry=true`: this memo + memory file
- `deploy_runbook=true`: `scripts/remote_mps_gap_experiment_a10g.sh` + `tools/run_mps_gap_experiment.sh`

Level 1 (impl_complete + memory_entry + deploy_runbook).

## Risk

- The wrapper-as-no-op finding (predecessor) means this gap experiment is the ONLY remaining path to either unlock local-MPS for substrate training or definitively rule it out
- If the verdict is `NOT_VIABLE`, the pivot suggestion is MLX (Apple's native ML framework, bypasses the PyTorch MPS abstraction) — but MLX is a 6+ month rewrite; the verdict has real strategic weight
- Modal A10G $0.50 envelope is firm; the recipe declares `dispatch_enabled: false` so no accidental spend can occur

## Mission alignment

`council_predicted_mission_contribution: frontier_protecting`. A VIABLE verdict unlocks the free local-MPS axis for substrate training (mission-positive: more parallelism for less $); an ADVISORY_ONLY or NOT_VIABLE verdict cleanly closes the path and reroutes operator attention to MLX or CUDA-train without further sunk cost.

## Cross-references

- `.omx/research/mps_conv2d_wrap_fix_empirical_finding_20260518.md` (predecessor)
- `.omx/research/mps_drift_mechanism_20260519T035310Z.md` (sister diagnostic)
- `src/tac/mps_diagnostic/layerwise_drift.py` (sister layerwise diagnostic)
- `src/tac/mps_gap_experiment/__init__.py` (THIS package public API)
- CLAUDE.md "MPS auth eval is NOISE" non-negotiable + Catalog #1 + #192 + #317 + #199 + #245 + #313 + #270
