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

## 6. r5 receipt + r6 staging (cuda-smokereg-r6 arm, 2026-07-15 ~19:5xZ)

- **r5 (fc-01KXKG3EQV92YZ4PRK1XMWHHCW) rc=124 @1451.8s, 0 epochs — root cause LOCALIZED from the
  harvested remote log** (`experiments/results/lane_v9-cgauge-cuda-438-smokereg-r5-20260715_modal/
  harvested_artifacts/modal_lane_...log`): the boot chain COMPLETED — `cuda_numpy_forward_parity`
  argmax_equal cosine_phi 0.9999999 → `backend_fp_reorder_probe` PASSED at ~80s (adoptable, cosine
  0.9999998, max-autotune on the 64-row closure) → dseg_aware_taper → pose_carrier → structured_init
  (sky IoU 0.976, hood IoU 0.993) → both cap_fired_before_event rows (smoke-regime forced starts
  WORKING) — then the FULL-P TRAINING-REGION `torch.compile` max-autotune benchmarking
  (`AUTOTUNE addmm(1572864x96,...)`, thousands of triton_mm variants) consumed every remaining
  second to the TERM cutoff. The 6dfb43c96b policy override was the right fix but was NOT THREADED:
  the trainer called `select_torch_execution_policy(device)` bare.
- **r6 wire-in LANDED (commit 5a0107cf72, 9 files, 147 tests green):** launcher
  `build_plan(torch_compile_mode="default")` (plan schema v9, hash-bound) →
  `WITNESS_TORCH_COMPILE_MODE` env (modal_train_lane exact-match allowlist + value gate) →
  `remote_v9_cgauge_cuda.sh` → trainer `--torch-compile-mode` →
  `select_torch_execution_policy(compile_mode=...)` AND `compile_identity_probe(mode=...)` (the
  probe now validates the SAME Inductor artifact training adopts). Plan-only rebuild verified:
  `WITNESS_TORCH_COMPILE_MODE=default` rides the real dispatch argv (plan_sha
  3afbe8d267288b9e..., ceiling $3.296256).
- **r6 NOT FIRED — spend tripwire (MEASURED):** ledger elapsed-billed campaign cumulative
  ≈ **$3.54** (r2 148s $0.22 + r3 324s $0.49 + r4 415s $0.63 + r5 1452s $2.20, H100 @$5/hr
  + cpu/mem), vs the ~$2.19 figure in the dispatch briefing. A timeout-stop r6 ALWAYS runs its
  full 1500s child window by design → realistic ~$2.2–2.5 → projected cumulative ~$5.7–6.0 >
  the $5 STOP line ("if projected r6 cost would push cumulative past $5, STOP and report").
  Secondary blocker at staging time: sister agents hold 7 dirty files incl. sentinel
  `src/tac/witness_dsl/spec_v9_cgauge.py`; the launcher fail-closes on any dirty main worktree.
- **One-command refire staged (operator GO required):**
  `nohup bash .omx/tmp/fire_438_smokereg_r6_daemon.sh & disown` — sealed-plan daemon with the
  r5-daemon guards PLUS `torch_compile_mode=default` field/env shape guard PLUS a single-flight
  `modal app list` running-task guard. Claims reconciled: phantom-active r5 campaign row
  terminal-rowed (`failed_rc124_training_region_max_autotune_window_exhausted`); no r6 traces on
  ledger/claims/Modal at staging.

## 7. Modal docs-facts table (operator-GO condition 1, verified 2026-07-15 from primary docs)

| # | Fact (verified) | r6 conformance | Source |
|---|---|---|---|
| a | Modal's own timeout raises `modal.exception.FunctionTimeoutError` to the caller — NO return value survives; functions "may run a handful of seconds longer"; docs recommend user-code timeout logic for precise control | CONFORMS BY CONSTRUCTION: we self-timeout INSIDE the function (GNU `timeout --signal=TERM 1440s` on the trainer child; wrapper survives, syncs volume, returns artifacts) with Modal's cap at 1800s (300s reserve). In-vivo proof: r5 rc=124 harvest returned manifest+log | https://modal.com/docs/guide/timeouts |
| b | `.spawn()` results retained "up to 7 days after completion" (`OutputExpiredError` after); `FunctionCall.from_id(...).get(timeout=...)` retrieval | harvest window ample (same-session); SUPERSEDES the ~24h TTL note in CLAUDE.md (docs now say 7 days) | https://modal.com/docs/guide/job-queue |
| c | H100 = $0.001097/s (≈$3.9492/hr); CPU $0.0000131/core/s; mem $0.00000222/GiB/s; per-second usage billing | our plan ceiling assumed $5.00/hr → 27% conservative. True-cost r6 ceiling: 1800s GPU=$1.97 + cpu/mem $0.22 + preflight $0.03 (+$0.50 staging allowance, unspent — GT asset already staged) ≈ $2.73 worst / ~$2.1 realistic | https://modal.com/pricing |
| d | env-var injection: our launcher passes env_overrides as plain `.spawn()` function kwargs merged into the lane-script subprocess env (modal_train_lane.py:1613) — NOT Modal Secrets/image env; no Modal-side size/charset constraint applies; our own `,`/`=` delimiter guards are the constraint surface | `WITNESS_TORCH_COMPILE_MODE=default` (alnum+underscore) safe | code path + https://modal.com/docs/guide (no Secret used) |
| e | queue/scheduling wait + cold start NOT billed ("You never pay for idle resources — just actual compute time"; billing guide: "only pay for the compute you use or request") | conforms; Modal H100 scheduling stalls cost $0 | https://modal.com/pricing + https://modal.com/docs/guide/billing |
| f | `retries` config would restart a fresh timeout per attempt (double-billing risk) | our function sets `retries=0` (modal_train_lane.py:2886,2989) — no retry double-bill | https://modal.com/docs/guide/timeouts + code |

## 8. r6 pre-fire recursive adversarial review (operator-GO condition 2)

- **Round 1 (2 findings → counter 0):**
  - **R1-1 (daemon):** single-flight `modal app list` guard ran once at daemon START — stale if the
    clean-tree window opens hours later. FIXED: moved inside the loop, checked at fire time.
  - **R1-2 (LAUNCH-INVALIDATING, trainer):** trajectory rows (`training_throughput_epoch`,
    `v9_controller_epoch`, epoch-final-chunk `loss_terms`) were buffered in memory and flushed ONLY
    at `--ckpt-every 25` checkpoints; no SIGTERM/atexit handler. A timeout-stop smoke killed at
    1440s before epoch 25 would burn ~$2 and return ZERO regime numbers (r5 masked this — it never
    reached epoch 1). FIXED: `_stage_epoch_row` mirrors every row to stdout at append time
    (`stdout_mirror: true` for harvest dedupe); the provider lane log captures stdout even at
    rc=124 (r5 in-vivo proof); the on-disk JSONL keeps its checkpoint-consistent resume invariant.
- **Round 2 (fixes re-reviewed + all 5 axes, 0 findings → counter 1):** helper buffers the
  UN-marked row (flush path unchanged); daemon `bash -n` clean; end-to-end trace re-verified;
  Modal-facts table above; budget recomputed at true $/s (cumulative-measured ≈$2.9 at
  $0.001097/s, r6 realistic ~$2.1 → projected ≈$5.0 of the $20 cap). Residual accepted risks,
  stated honestly: (i) default-mode first-compile at full-P shapes may still be minutes —
  attribution via `compile_warmup_epoch` + per-epoch mirror rows; fallback lever wired
  (`--torch-compile-mode off`) for r7 if compile still eats the window; (ii) Inductor cudagraph
  trees ride with default mode (same as sealed design; degrade-gracefully expected).
