# #438 CUDA smoke — smoke-regime rebuild, sealed fire daemon, cost-table contract (2026-07-15)

**Arm:** cuda_smoke_438_respawn (respawn of the session-limited CUDA timing-smoke arm)
**Lane:** `cuda_v9_fullrun_campaign_20260715` (coordination) / `lane_cloud_launcher_v9_cgauge_cuda_438_20260711` (launcher)
**Authority:** `[contest-CUDA training-advisory] NON-PROMOTABLE` — cost-regime measurement, never a science arm.
**Pointer:** UNMOVED (apparatus/means).

## 1. What is built + committed (HEAD e4c849ed50)

- **Factory fix (NO-FAKE):** `compile_v9_cgauge_432_smoke_regime_config` as landed at fde0e733cf was
  BROKEN at invocation — `--seg-temporal-screw-start-epoch` is Lever-resident
  (temporal_screw_consistency lever), not base-resident, so the base-only override raised
  `ValueError` on every call. The "33/33 spec tests" never invoked the factory. Fixed: the forced
  set now lands wherever the flag actually lives (base OR unique lever owner), refuses
  absence/ambiguity, and 3 new tests INVOKE the factory end-to-end (argv diffs vs launch config are
  exactly the 3 forced backstops → 1; typed hash distinct; validate_program clean).
- **Consumer-leg wiring (closes the drift-detector consumer leg for the smoke-regime factory):**
  - trainer `--dsl-config {v9_cgauge_432_launch, v9_cgauge_432_smoke_regime}` + receipt/result echo;
  - driver `WITNESS_DSL_CONFIG` env + **optional** `WITNESS_STOP_AFTER_EPOCHS` (empty = timeout-stop:
    the 1500 s child timeout is the stop; operator 2026-07-15 "3-warmup-epoch cutoff is toy-regime");
  - `build_plan(dsl_config=..., stop_after_epochs=None)` (schema `witness_cloud_plan.v8`);
  - `tools/launch_witness_cloud.py --dsl-config / --stop-after-epochs`;
  - `modal_train_lane` V9 env contract extended (WITNESS_DSL_CONFIG required; stop ∈ {"", "1".."3"}).
- **Real preflight PASS:** n600 CPU preflight at the smoke-regime config on the real GT cache:
  `runtime_epoch_window [1,3000]`, `runtime_stop_after_epochs null`, `dsl_config` echoed, scorer
  custody exact. 156 tests green across the three touched suites.

## 2. The sealed plan + fire daemon

- Plan (plan-only build at e4c849ed50): `plan_sha256 93756073f49f…`, **ceiling $3.296256**
  (< $5 internal refusal max; within the operator-pre-approved $3.30-class), label
  `v9-cgauge-cuda-438-smokereg-20260715`, H100, n600, epochs 3000 typed horizon, NO
  stop-after-epochs (1500 s child timeout is the stop), smoke-regime config.
- **Blocker at fire time:** the launcher requires a fully clean main worktree; sibling arms
  (m5_burndown_509, cohesive_package_507, merge_coherence) hold in-flight edits.
- **Fire daemon (Pattern A, charter-authorized):** `.omx/tmp/fire_438_smokereg_daemon.sh`
  (log `.omx/tmp/fire_438_smokereg_20260715.log`). Polls for a clean-main window (6 h deadline),
  then runs the runbook sealed one-shot: rebuild plan at fire-HEAD → guard
  (smoke_regime ∧ stop=None ∧ label ∧ cost ≤ $3.31 ∧ execution_allowed) → execute with
  `--expected-plan-sha256` + GO-CLOUD-381. Every launcher gate (clean-tree, HEAD adjacency,
  active-claim, custody, $5 max) remains fail-closed underneath. On dispatch it appends the
  claim-row update. Kill: `kill <pid in checkpoint step 5>`.
- Prior lane spend: one 2026-07-12 dispatch `fc-01KXBWNEBFHDKWESRPJ7ECHSD2` rc=13 (GPU-less
  `H100!` bug, since fixed), elapsed 0.0 s → ≈$0 actual. Envelope #381 headroom intact.

## 2b. DISPATCH RECORD (2026-07-15T17:03Z)

- **Fired:** attempt 2, sealed plan `90772392e40ab60fddddddfbb41452f4c4c321b7d927782625ba83164721be19`
  at HEAD `6d39a7dfc9471b26093e507f99c77ecdc844541d` (clean-main window; siblings landed).
- **Call ID:** `fc-01KXKBQ1B6NZ0YR7Z7TJB585YT` (H100 stage, detached; child ≤1500 s; CPU preflight
  passed same-image first; modal app ap-8CVdagVdtREOhPpd4uGiMm).
- Attempt 1 (17:57:44Z log): refused PRE-SPAWN by the modal-side Catalog #166 clean-head gate —
  2 mid-flight tracked-state edits (incl. this arm's own claim-row write; lesson: hold ALL tree
  writes between daemon clean-check and modal-side clean-check). $0 spent; recorded in the modal
  ledger as `pre_spawn_fatal`.
- Ceiling $3.296256 persisted against the call ID (`--expected-cost-usd`); envelope #381.
- Old stop=3 claim row terminated `stale_superseded_by_smokereg_timeout_stop_plan`.

## 3. Harvest contract (whoever harvests: follow exactly)

1. `.venv/bin/python tools/harvest_modal_calls.py --from-ledger --call-id <call-id> --execute`.
2. **Regime numbers** from `training_throughput_epoch` rows in the run-dir trajectory JSONL:
   - **post-event steady-state s/ep** = median of `seconds` over epochs ≥ 2 (epoch 1 =
     `compile_warmup_epoch`, EXCLUDE). In the smoke-regime config the post-event cost levers
     (lane_band / seg_chroma_boundary / seg_temporal_screw) are forced on from ep1, so
     steady-state here IS the expensive regime that binds ~99% of a full run.
   - **warmup/cheap-regime s/ep is intentionally NOT measured by this run** (that was the
     condemned toy-regime number). Muon (ep 726+) / phase-advect adders remain labeled
     UNMEASURED-extrapolation.
   - **verdict cost** = 0 additional s/ep on the torch path: the frozen-scorer forward over all
     n600 pairs is fused into every training epoch (controller emits per-epoch `d_seg` in
     `v9_controller_epoch` rows — train-mode, live weights, full n600; advisory, never a score).
3. **DRIFT column (CUDA-drift law, poison_taxonomy_event_recompute_and_cuda_drift_20260715):**
   - per-substrate determinism flags: CUDA path runs `use_deterministic_algorithms(False)`,
     `TF32=True`, `cudnn_benchmark=True`, amp bf16 (`select_torch_execution_policy`) — record the
     `amp_dtype` field from throughput rows + these policy flags in the report; same-seed bit-repeat
     is NOT expected on this substrate (drift-OK for training per 07-15 law).
   - in-run parity receipts: `cuda_numpy_forward_parity` (vs numpy-fp32 authority) +
     `backend_fp_reorder_probe` rows at startup — copy both into the report.
   - K-epoch trajectory divergence vs MLX: requires an MLX reference run at the SAME
     smoke-regime config + seed 0 (see §4). Compare per-epoch `d_seg` + loss-terms trajectories.
     **Confound guard (C0 finding 2026-07-15):** any gradient-level comparison must use the
     POST-`--grad-normalize` update direction (the config normalizes per-param AFTER clip → the
     effective update law is unit-norm×LR); raw pre-clip grad comparisons inherit the
     telemetry≠mechanism confound (`perparam_normalize_masks_all_norm_clipping_c0_confound_20260715`).
   - A lever measured on CUDA is a CUDA-arm finding only — NO cross-substrate lever verdicts.

## 4. Cost table (skeleton — fill from harvest; FULL RUN stays gated on operator GO)

| Tier | $/hr (ceiling) | post-event s/ep | epochs/hr | days→3000 ep | cost→3000 ep | provenance |
|---|---|---|---|---|---|---|
| Modal H100 | 5.00 | MEASURED@harvest | derive | derive | derive | this smoke |
| Modal A100 | 4.00 | UNMEASURED | — | — | — | needs own smoke; do NOT scale-guess |
| 4090-class (Vast) | ~0.35 | UNMEASURED | — | — | — | needs own smoke |
| M5 Max Metal (max-leverage) | $0 | PLACEHOLDER — m5_burndown_509 measured max-Metal number | — | — | — | burn-down arm |

Coordination ask to **m5_burndown_509**: (a) drop your measured max-Metal s/ep into the M5 row;
(b) if feasible, run K epochs of the MLX reference at `compile_v9_cgauge_432_smoke_regime_config`
seed 0 so the DRIFT trajectory column is computable (the local launcher does not yet expose the
smoke-regime named config — either add it beside your ISO configs in `tools/launch_witness_run.py`
or invoke the torch trainer's `--dsl-config` CPU path).

## 5. Consistency

- DAG leg: this ledger. DSL leg: the factory + `--dsl-config` selection (committed). Equations
  leg: no new law (cost-regime apparatus); the flicker/event laws referenced are pre-registered.
- NON-PROMOTABLE everywhere: `score_claim=false`, `promotion_eligible=false` on every row this
  smoke produces.
