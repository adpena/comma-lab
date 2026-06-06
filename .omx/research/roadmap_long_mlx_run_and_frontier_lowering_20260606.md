# Roadmap — groundwork to long MLX training + frontier score lowering

UTC: 2026-06-06T20:30:00Z · planning-only; no score claims; frontier numbers
live in `.omx/state/canonical_frontier_pointer.json` (pointer-only rule).

## Gate order (witness-readiness DAG = launch authority)

SHARED: source-boundary audit (ran today) → exact scorer-oracle cache →
distortion trace harness (servo kernel LANDED `fc340494d`+`a0f082435`) →
distortion-birth-before-rate-pressure (v4 receipt clears argmax-progress;
min-ratio now source-qualified `3de167ba5`) → joint trust (actuator joint gate
LANDED `c3e9bb1b8`; adapter-level rank-3 still open).
HINERV: short receiver-surface smoke (v4 ran; pose-threaded v5 pending) →
localized target-region actuator (LANDED L2) → launch gate.
SNERV: official MFU/HFR/TUB source-forward (codex v61) → LF/HF collapse smoke
→ launch gate.

## HiNeRV ladder (partner-adopted)

L3 (pose-trusted birth): thread runner `pose_scorer_teacher` (line ~9413) into
actuator callsite (~19120) + rerun v5 same pair/region → receipt with
old/new_d_pose_pair + delta_score_nonrate + pose_trust_pass. Actuator side
DONE (require_pose_trust + contest-res guard + joint gate).
L4 (survival): fakequant + parse-back receipts of the SAME accepted birth
(codex's parseback/inflate servo lift `d9fd5ce28`/`1c9c0d01b` is the consumer
surface) + hysteresis receipt (wins persist M steps).
L5 (representative): inflate CPU replay; value-per-byte for byte changes;
coverage via `lane_receiver_replay_scorer_hard_region_miner_20260606` (L0).

## SNeRV burndown (codex-owned, v61)

trained checkpoint state_dict load → temporal-encoder + output_2 portable
weight mapping → lineage verification → bit-flip falsification test →
LF/HF representation-collapse smoke under byte pressure → authority gate
`official_tub_lf_hf_decoder_replacement_ready=true`.

## Shared compiler groundwork (frontier lowering WITHOUT waiting on NeRV)

1. `lane_action_effect_thin_ir_20260606` — typed ActionEffect over the servo
   admission schema (+ ScoreProgramRuntime Python interface).
2. `lane_pr110_pairwise_commutator_ledger_20260606` — comm(a,b) at parse-back
   authority over PR110 catalog on the CURRENT frontier archive ($0 CPU on
   cached scorer atoms) → macro-actions/conflicts.
3. `lane_selector_menu_ilp_20260606` — per-tier/cluster menus vs global K=16,
   menu overhead priced in exact score units (L26/L31 encoders exist).
4. MISSING TOOL: `tools/validate_nerv_long_run_gate.py` — the approval-doc
   consumer (parse-back selection manifest + birth smoke + trust rows +
   value-per-byte ledger + receiver proof + full-video replay). Build skeleton.
5. Rank-5 section value-per-byte ledger; rank-6 full-video MLX replay gate.

## Long-run campaign mechanics (CLAUDE.md campaign contract)

lane_id + dispatch claim · timing smoke (sec/epoch, M5 Max) · resumable
full-run cmd + harvest path · storage waterfall to SSD + auto-clean hook ·
stop/continue thresholds (smoke/mid/export/exact) · parse-back selection
REQUIRED (landed `143b1b11a`) · PR95 8-stage curriculum binding (landed) ·
EMA + eval_roundtrip + differentiable YUV6 (standing) · end with paired
exact CPU+CUDA auth eval on the exact archive bytes.

## Hygiene backlog (small, high-leverage)

- `test_dag_remediation_argv_enables_blocked_actuator` — canonical argv STILL
  lacks teacher flags; cannot open the hard-birth gate it remediates. Update
  argv after v5 fixes the final flag set.
- `actuators_enabled_effective` receipt + configured-but-zero-weight test.
- Crux-trace auto-ingest of actuator receipts (producer rows).
- L0 lanes: hinerv_scorer_bootstrap_floor_preserve; snerv_lf_hf_runtime/
  bounded-training binding; renderer-unblock queue contract.

## Suggested order

NEXT SESSION: pose callsite + v5 receipt → fakequant/parse-back receipt of
same birth → hysteresis → validate_nerv_long_run_gate skeleton.
THEN ($0, frontier-direct): ActionEffect IR → commutator ledger on frontier
archive → menu ILP → candidate bolt-on → paid paired CPU+CUDA replay only at
promotion.
THEN: HiNeRV bounded long-run timing smoke → launch decision through the gate
tool; SNeRV long run follows TUB closure.
