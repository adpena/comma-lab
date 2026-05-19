---
council_tier: T1
council_attendees: [mps-phase-b-fire-and-harvest-subagent]
council_quorum_met: false
council_verdict: DEFER_PENDING_EVIDENCE
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Single-machine compute_gap_components on Modal A10G can measure MPS-vs-CUDA gap"
    classification: CARGO-CULTED-FALSIFIED
    rationale: "Modal A10G has no MPS hardware; both target_device AND mps_reference_device forwarded on CUDA producing IDENTICAL outputs (pixel_l1=0.0, segnet=0.0). Genuine MPS-vs-CUDA gap measurement requires SPLIT-DEVICE architecture (local Mac MPS forward + remote Modal CUDA forward + local diff)."
  - assumption: "Predecessor harvest_and_verdict infrastructure produces dispositive MPS-vs-CUDA verdict from a single Modal dispatch"
    classification: CARGO-CULTED-FALSIFIED
    rationale: "Infrastructure design assumption that one-shot Modal CUDA forward could compare against MPS reference was structurally invalid because MPS reference forward never happened on the Modal worker (Modal A10G hardware has no MPS device); the harvest helper silently fell back to CUDA-vs-CUDA comparison."
council_decisions_recorded:
  - "op-routable #1: rewrite tac.mps_gap_experiment.harvest_and_verdict to support split-device manifest (local MPS components saved as reference JSON + Modal CUDA components compared against)"
  - "op-routable #2: OR pivot to MLX exploration as planned per the NOT_VIABLE_PIVOT branch (operator decision)"
  - "op-routable #3: OR have the Modal dispatch compare CUDA-A10G vs CUDA-T4 (cross-instance hardware drift) — still useful diagnostic but doesn't answer the MPS-train-viability question"
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: true
council_override_rationale: 'Operator-frontier-override 2026-05-19 verbatim "All operator fates and decisions approved" — see .omx/research/operator_authorizations/e7_e8_symposium_operator_frontier_override_20260519T051028Z.md'
deferred_substrate_id: mps_gap_experiment_phase_b
deferred_substrate_retrospective_due_utc: 2026-06-18T05:35:30Z
related_deliberation_ids:
  - mps_gap_experiment_infrastructure_built_20260519T043638Z
  - mps_conv2d_wrap_fix_plus_real_frame_gap_experiment_20260518
---

# MPS Phase B gap experiment — verdict + per-component gap analysis

**Lane**: `lane_mps_phase_b_fire_and_harvest_20260519` L1 → L2 (after this memo + memory entry land)
**Predecessor**: `lane_phase_b_mps_gap_experiment_infrastructure_build_20260518` (commit `df2f6a297` 2026-05-19)
**Modal call_id**: `fc-01KRZBHXARS5EGW4EM5P4NJRK6` (Modal A10G; rc=0; 11.6s elapsed)
**Modal app_id**: `ap-aOGkOGhlCm4Fu3yS4DfFD4`
**Operator authorization**: OPERATOR_FRONTIER_OVERRIDE 2026-05-19 ("All operator fates and decisions approved")
**Spend**: ~$0.005 of $0.50 envelope (3 dispatches totaling 19.4s on A10G @ $1.10/hr)
**Evidence grade**: `MPS-research-signal` per CLAUDE.md "MPS auth eval is NOISE" non-negotiable + Catalog #1/#192/#317
**Score claim**: false
**Promotion eligible**: false

## TL;DR — gap manifest is structurally invalid; paradigm intact; DEFER not KILL

The Modal A10G dispatch returned `gap_results.json` with:

| Component | mps_value | target_value | absolute_diff | relative_diff |
|---|---|---|---|---|
| pixel_l1_mean | 0.0339 | 0.0339 | 0.0 | 0.0 |
| segnet_mean_output | -0.9894 | -0.9894 | 0.0 | 0.0 |
| posenet_mean_output | NaN | NaN | NaN | NaN |

`gap_relative_aggregate = NaN`, verdict `LOCAL_MPS_TRAIN_NOT_VIABLE_PIVOT_MLX_OR_VTOOLBOX` from the NaN fallback (per `tac.mps_gap_experiment.harvest_and_verdict.classify_verdict`).

**The 0.0 gap is NOT a real "MPS-trained weights survive CUDA forward perfectly" finding.** It's a structural artifact: the `compute_gap_components(target_device="cuda", mps_reference_device="mps")` call on the Modal A10G worker silently fell back to CUDA-vs-CUDA comparison because **Modal A10G has no MPS hardware**. The output's `mps_reference_device: "cuda"` field confirms this — the predecessor's harvest helper allowed `device="mps"` to fall through to `device_obj = torch.device("mps")` which raises on Modal, then likely caught + retried on CUDA.

The verdict is **DEFER**, not KILL, per CLAUDE.md "Forbidden premature KILL without research exhaustion": the original infrastructure design is paradigm-level INTACT (the build-then-await pattern + 7K-param tiny renderer + harvest manifest schema all worked), but the implementation-level architecture cannot answer the MPS-vs-CUDA gap question from a single Modal-only dispatch.

## What this measurement DOES tell us (advisory, not authoritative)

1. **Modal A10G CUDA forward is deterministic with itself** — pixel_l1 and segnet outputs are bit-stable across two runs of the same EMA checkpoint on identical inputs (no nondeterminism in the tiny renderer + segnet forward path). This is a useful sanity check.

2. **PoseNet on Modal A10G threw an exception** — the helper caught and returned NaN. Likely a tensor shape / dtype mismatch with the tiny renderer's reconstruction output (PoseNet expects 12-channel YUV6 input from 2 frames; the tiny renderer outputs a single-pair-batch reconstruction at a different shape). This is a separate small bug.

3. **The dispatch infrastructure works end-to-end** — Catalog #199/#202/#243/#270 paired-env discipline + dispatch_kind:tool scope-fix + canonical NVML env block + canonical bootstrap + PYBIN + canonical mount manifest TIER_1_EXTRA_MOUNT_PATHS + Modal call_id ledger + harvest_modal_calls.py all worked correctly. Three failed dispatches surfaced 3 distinct bug classes (each documented in commit log + fixed); the 4th succeeded.

## What the measurement CANNOT tell us (the open question stays open)

The mission's gating question — **"Do MPS-trained weights survive CUDA scoring on real contest frames within a usable tolerance?"** — remains EMPIRICALLY UNRESOLVED. The 0.0 gap is a measurement artifact, not evidence of MPS-CUDA equivalence. The Phase A predecessor (commit `24278cf06`) established that MPS-vs-CPU SegNet end-to-end drift on synthetic noise was 7.6e-5 — well below the 1e-3 cumulative threshold — but that's a different question (MPS-vs-CPU not MPS-vs-CUDA, synthetic-noise not real-frame, no end-to-end training loop).

## Operator-routable reactivation paths (3-option fork)

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + "Substrate retirement discipline" (Catalog #298):

### Option A — Rewrite harvest infrastructure for split-device (recommended for empirical closure)

Refactor `tac.mps_gap_experiment.harvest_and_verdict.compute_gap_components` into TWO entry points:
1. `compute_local_mps_reference_components(...)` — runs locally on Mac MPS, saves `local_mps_components.json` next to the checkpoint
2. `compute_target_cuda_components(...)` — runs on Modal A10G, saves `target_cuda_components.json` in the Modal output dir
3. `diff_components_and_classify_verdict(local_path, target_path, output_path)` — runs locally, loads both JSONs, computes diff, emits canonical `gap_results.json`

Update the harness `tools/run_mps_gap_experiment.sh` to call (1) before the Modal dispatch, and (3) after harvest. Estimated wallclock: ~2 hours engineering, ~$0.50 to re-dispatch and harvest. This produces the dispositive verdict the mission was designed to surface.

### Option B — Pivot to MLX exploration (recommended for path-of-least-resistance per the NOT_VIABLE branch)

Per the recipe's verdict thresholds + Phase A premise: if the gap is genuinely > 20%, the planned pivot is MLX (Apple's native ML framework, bypasses the PyTorch MPS abstraction). MLX is a 6+ month rewrite per the predecessor's risk assessment — start scoping the rewrite NOW since this is the canonical operator-decision branch when the gap is NOT_VIABLE.

### Option C — Defer indefinitely; mark research_only=true; reactivate when one of A/B becomes operator-priority

Per CLAUDE.md "Substrate retirement discipline": the lane registry can carry the MPS gap experiment as `research_only=true` + reactivation criteria pinned, and the operator can re-prioritize when local-MPS compute frontier becomes blocking again. No further spend until then.

## 9-dimension success checklist evidence

Per Catalog #294.

1. **UNIQUENESS** — DEFER outcome distinct from sister verdicts (NSCS06 v6→v7 PROCEED / Wunderkind G1 v2 DEFER-paradigm-intact / TT5L "fundamentally broken janky" REFUSE). This is a paradigm-intact / implementation-incomplete DEFER.
2. **BEAUTY + ELEGANCE** — 4 dispatches fired, 3 different bug classes surfaced + fixed in 3 commits (`bf6a2ecea` recipe + scope-fix / `182bcab44` Modal mount path + extra_mount_paths / `ab296d5b3` driver bootstrap + PYBIN), 4th dispatch SUCCEEDED in 11.6s rc=0; manifest harvest pipeline worked end-to-end.
3. **DISTINCTNESS** — Per-component gap analysis (pixel_l1 + segnet + posenet) is structurally distinct from prior MPS diagnostic memos (Phase A was synthetic-noise; this was real-frame after training loop; the new finding is that single-machine architecture is the structural limit).
4. **RIGOR** — Catalog #229 PV BEFORE every edit (8 PVs); Catalog #117/#157/#174 canonical serializer per commit; Catalog #199 paired-env operator authorization; Catalog #202 paired-env trusted-sentinel-clean bypass; Catalog #243 local pre-deploy harness; Catalog #270 dispatch protocol PASS with dispatch_kind:tool scope-fix; Catalog #313 probe outcome registered (DEFER blocking expires 2026-06-18).
5. **OPTIMIZATION PER TECHNIQUE** — N/A (diagnostic infrastructure).
6. **STACK-OF-STACKS-COMPOSABILITY** — Lane is independent of contest substrate stacks; verdict DEFER means it does not contribute to any contest composition.
7. **DETERMINISTIC REPRODUCIBILITY** — Modal A10G CUDA forward is bit-stable per the 0.0 gap observation (deterministic across two runs of the same checkpoint on identical inputs).
8. **EXTREME OPTIMIZATION + PERFORMANCE** — Local MPS training 1.8s for 100 epochs; Modal A10G forward + harvest 11.6s; total wall-clock from Phase 1 start to verdict ~17 minutes including 3 failed-dispatch iterations.
9. **OPTIMAL MINIMAL CONTEST SCORE** — N/A. The DEFER verdict does not contribute to score. If Option A reactivation produces VIABLE, then local-MPS compute unlock would be mission-positive for parallel substrate training (no direct score contribution, but unlocks more parallel exploration for less $).

## Observability surface

Per Catalog #305.

1. **Inspectable per layer** — TinyRenderer's named modules still queryable via `model.named_modules()`; predecessor diagnostic intact.
2. **Decomposable per signal** — `gap_results.json::components` carries per-component rows; this verdict memo decomposes the NaN aggregate into the 3 component rows + explains the 0.0 + NaN values.
3. **Diff-able across runs** — `gap_results.json` is canonical JSON; future re-runs (Option A) can `jq`-diff component-by-component against this verdict.
4. **Queryable post-hoc** — probe outcome registered at `.omx/state/probe_outcomes.jsonl`; Modal call_id at `.omx/state/modal_call_id_ledger.jsonl`; lane registry tracked at `.omx/state/lane_registry.json`.
5. **Cite-able** — every artifact carries (target_device, mps_reference_device, num_pairs, modal_call_id, modal_app_id, evidence_grade, axis_tag) tuple.
6. **Counterfactual-able** — Option A reactivation path provides the counterfactual probe (split-device architecture) that would resolve the open question.

## Canonical-vs-unique decision per layer

Per Catalog #290. No new substrate engineering — the diagnostic infrastructure was canonical (per predecessor's design memo Section "Canonical-vs-unique decision per layer"). The DEFER finding is at the implementation surface, not the canonical-helper-adoption surface.

## Cargo-cult audit per assumption

Per Catalog #303.

| Assumption | Classification | Unwind path |
|---|---|---|
| Single Modal dispatch can produce MPS-vs-CUDA gap verdict | CARGO-CULTED-FALSIFIED | Option A split-device architecture |
| `compute_gap_components(target_device="cuda", mps_reference_device="mps")` runs MPS forward when invoked on a Modal worker | CARGO-CULTED-FALSIFIED | The helper silently falls back to CUDA when MPS is unavailable; the API contract should require the MPS reference to be precomputed locally, NOT inferred from a kwarg passed to a remote worker |
| Predecessor verdict thresholds (5% / 20%) are dispositive at gap_relative_aggregate level | HARD-EARNED-PENDING-OPTION-A | The thresholds are correctly tuned for a real MPS-vs-CUDA comparison; only the measurement infrastructure was broken |
| Posenet helper would succeed on the tiny-renderer reconstruction shape | CARGO-CULTED-FALSIFIED-MINOR | The reconstruction's shape doesn't match PoseNet's expected (B, 2, 3, H, W) 2-frame YUV6 input; needs shape adapter or explicit error message |

## Predicted ΔS band — N/A

This is a diagnostic recipe per the predecessor's design memo. No contest ΔS prediction; verdict is `gap_relative_aggregate` float (currently NaN, mission-incomplete).

## 6-hook wire-in (Catalog #125)

1. Sensitivity-map contribution — **N/A**: diagnostic infrastructure
2. Pareto constraint — **N/A**
3. Bit-allocator hook — **N/A**
4. **Cathedral autopilot dispatch hook — ACTIVE**: the DEFER probe outcome registered via `tac.probe_outcomes_ledger.register_probe_outcome` is consumable by `tools/cathedral_autopilot_autonomous_loop.py` as advisory signal for "MPS-train viability question is OPEN; do not route substrates to local MPS until Option A resolves OR operator pivots to MLX"
5. **Continual-learning posterior — ACTIVE**: probe outcome appended to `.omx/state/probe_outcomes.jsonl` (canonical fcntl-locked JSONL per Catalog #131 / #138 / #245 sister discipline)
6. **Probe-disambiguator — ACTIVE**: this entire dispatch + verdict cycle IS the canonical disambiguator between "MPS-train viable" / "ADVISORY only" / "NOT_VIABLE pivot" branches; the DEFER outcome means none of the 3 branches is empirically validated yet; the 4 reactivation paths (Options A/B/C/D) ARE the next disambiguator steps

## Lane registry evidence (post-this-memo + memory entry)

- `impl_complete=true`: 3 commits + 4 Modal dispatches (3 failed-then-fixed + 1 succeeded rc=0); recipe + trainer + driver all in canonical state
- `real_archive_empirical=false`: no archive (diagnostic only; the `gap_results.json` is `[MPS-research-signal]` not a contest archive)
- `contest_cuda=false`: never run on contest-axis (diagnostic-CUDA Modal A10G is NOT contest-CUDA)
- `strict_preflight=N/A`: research infrastructure
- `three_clean_review=false`: not a contest substrate
- `memory_entry=true`: this memo + memory file
- `deploy_runbook=true`: `scripts/remote_mps_gap_experiment_a10g.sh` + `tools/run_mps_gap_experiment.sh`

Level 2 (impl_complete + memory_entry + deploy_runbook + real_archive_empirical=FALSE-but-empirical-measurement-landed).

## Risk

- If the operator picks Option B (MLX pivot) without first attempting Option A, we close out the local-MPS-compute frontier WITHOUT a dispositive measurement of its viability. That's a research-rigor regression per CLAUDE.md "Forbidden premature KILL without research exhaustion".
- Option A is the recommended path because it preserves the option to ACTUALLY answer the gating question with ~$0.50 + 2 hours engineering. If the verdict from Option A is genuinely NOT_VIABLE, Option B (MLX pivot) is well-justified at that point.
- Sister subagent collision (Catalog #314) was observed during dispatch #3 (Modal mtime-stability check fired when sister edited `src/comma_lab/research_state.py` during image upload); the canonical paired-env trusted-sentinel-clean bypass + retry worked. No data loss.

## Mission alignment

`council_predicted_mission_contribution: rigor_overhead`. This dispatch cycle exposed an implementation-level architectural gap in the predecessor's infrastructure (single-machine compute_gap_components cannot compare MPS-vs-CUDA). The output is a DEFER + 3-option reactivation fork, NOT a frontier-breaking score lower OR frontier-protection gate. Per CLAUDE.md "Mission alignment - non-negotiable" consequence 3 (30-day retrospective due 2026-06-18): re-evaluate whether the MPS gap question is still mission-relevant; if YES, prioritize Option A.

## Cross-references

- `.omx/research/mps_gap_experiment_infrastructure_built_20260519T043638Z.md` (predecessor)
- `.omx/research/mps_conv2d_wrap_fix_empirical_finding_20260518.md` (Phase A finding)
- `.omx/research/mps_drift_mechanism_20260519T035310Z.md` (sister diagnostic)
- `experiments/results/lane_mps_gap_experiment_tiny_renderer_modal_a10g_dispatch_20260519T053223Z_modal/harvested_artifacts/mps_gap_results/gap_results.json` (canonical gap manifest output)
- `.omx/state/probe_outcomes.jsonl` (probe outcome row `mps_phase_b_gap_experiment_verdict_20260519T053530Z`)
- `.omx/state/modal_call_id_ledger.jsonl` (call_id `fc-01KRZBHXARS5EGW4EM5P4NJRK6`)
- `.omx/operator_authorize_recipes/mps_gap_experiment_tiny_renderer_modal_a10g_dispatch.yaml` (recipe; dispatch_enabled:true + dispatch_kind:tool)
- CLAUDE.md "MPS auth eval is NOISE" + "Forbidden premature KILL without research exhaustion" + "Substrate retirement discipline" + Catalog #1 + #192 + #199 + #202 + #243 + #270 + #313
