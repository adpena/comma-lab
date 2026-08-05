# ddm_dy2 Receipt: Plateau-Anchored JD1 Tail EMA

Captured: 2026-08-05T22:16:26Z

## RECALL EVIDENCE

Searches run before implementation:

- `rg -n "tail[-_ ]average|tail average|Polyak|dynamic[-_ ]EMA|dynamic EMA|jd1.*ema|ema.*jd1|ema_decay_run_geometry|T3_LIVE_ADAPTED|scope-law|scope_law" .omx/research src/tac/witness_dsl experiments tools .omx/state --glob '!*.log' --glob '!*.jsonl'`
- `.venv/bin/python tools/list_canonical_equations.py --json | rg -n "ema|EMA|polyak|Polyak|tail|average|T3_LIVE|scope"`
- `rg -n "dy2|dynamic[-_ ]EMA|plateau.*tail|ema.*mode|tail.*anchor|T3_LIVE_ADAPTED" .omx/research/CANONICAL_RESEARCH_INDEX* .omx/research/sub015_DAG_* .omx/state .omx/research/ddm_tp1_boundary_receipt_20260805.md`
- `rg -n "EMA|ema|Polyak|tail average|plateau" .omx/research/ddm_tp1_boundary_receipt_20260805.md .omx/tmp/codex_runs/dy1_prompt.md .omx/tmp/codex_runs/tp1_prompt.md .omx/tmp/codex_runs/rr1_prompt.md`

Found beyond charter seeds:

- Existing `PolyakFinisher` / `PolyakTailAverager` precedent exists in `src/tac/witness_dsl/curriculum_dsl.py` and `src/tac/witness_control/polyak_finisher.py`, but it exports an extra candidate/finisher and does not replace TR1's shipping JD1 EMA shadow. Plan unchanged: build the mode switch inside `experiments/train_tr1_partition_renderer_mlx.py`.
- Canonical equation `ema_decay_run_geometry_v1` exists for the fixed geometric law. It remains the pre-anchor law.
- `dy1_prompt.md` references `T3_LIVE_ADAPTED` / scope-law work, but dy1's module is unmerged. I did not depend on it.
- `ddm_tp1_boundary_receipt_20260805.md` confirmed the dynamic-EMA A/B and MAIN-R5X basis correction: gate rows must label the actual basis, not collapse tail average back to `ema_shadow`.

Changed plan:

- Added a local TR1 helper family and DSL lever now, without depending on dy1.
- Added a `FORMALIZATION_PENDING` waiver for the tail-average law registration, with fire order below.

## Build

Fireable now:

- New trainer flags:
  - `--jd1-ema-mode {geometric,plateau_tail_average}`, default `geometric`.
  - `--jd1-ema-tail-anchor-epoch N`, default `-1`.
- Validation refuses inert or ambiguous shapes:
  - tail mode while JD1 is off;
  - tail anchor while JD1 is off;
  - geometric mode with a tail anchor;
  - tail mode without `--jd1-ema-stage-scope window`;
  - tail mode without explicit anchor.
- Default path keeps old checkpoint metadata shape: `jd1_ema_initial_state(args) == {}` and `jd1_ema_checkpoint_payload(args,{}) == {}` for geometric/default.
- Tail mode runs geometric EMA until the explicit anchor epoch, then resets the shipping EMA shadow to live trainable params and updates as a growing-horizon average:
  - anchor live params are sample 0;
  - first post-anchor settled live update uses weight `1/2`;
  - later updates use `1/(k+2)` where `k` is the count already folded after anchor.
- Resume metadata persists mode, configured anchor, actual anchor, active bit, update count, global step, reason, and last live weight through `jd1_pose_finish` checkpoint meta.
- Gate/confirm/boundary-positive-control basis selection now goes through `jd1_ema_gate_basis_label`; active tail average labels basis as `ema_tail_average`.
- DSL leg: `lever_jd1_plateau_tail_average_ema(anchor_epoch=...)` in `src/tac/witness_dsl/spec_tr1_renderer_20260728.py`.

## Triality

- DSL: `lever_jd1_plateau_tail_average_ema`.
- Equations: pre-anchor fixed law remains `ema_decay_run_geometry_v1`; tail-average law is `FORMALIZATION_PENDING` until dy1 scope-law merge.
- DAG/artifact: this receipt and `NEXT_IF_RESUMED.md`; no score/pointer DAG advancement claimed.

Formalization waiver:

- The implemented law is explicit and tested: `shadow_{n+1} = shadow_n + (live_n - shadow_n)/(n+2)` after an anchor sample.
- Canonical registration is queued, not skipped, because the charter says dy1's `T3_LIVE_ADAPTED` scope-law module is unmerged.

## Verification

Passed:

- `.venv/bin/python -m py_compile experiments/train_tr1_partition_renderer_mlx.py src/tac/witness_dsl/spec_tr1_renderer_20260728.py src/tac/tests/test_ddm_dy2_jd1_tail_average_ema.py src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py src/tac/tests/test_ddm_bp1_boundary_reset_race.py`
- `.venv/bin/python -m pytest src/tac/tests/test_ddm_dy2_jd1_tail_average_ema.py src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py src/tac/tests/test_ddm_bp1_boundary_reset_race.py::test_gate_basis_differs_between_fresh_and_resumed_runs`
  - Result: 18 passed.
  - Pytest printed a known MLX atexit Metal-device warning after a zero exit status.
- Review tracker: two `tools/review_tracker.py mark-file --status reviewed` passes for all changed Python files.

Attempted broader guard:

- `.venv/bin/python -m pytest src/tac/tests/test_ddm_bp1_boundary_reset_race.py`
  - Result: 36 passed, 6 failed.
  - All six failures are environment-coupled MLX `nn`/optimizer/module construction failures: `[metal::load_device] No Metal device available`.
  - The dy2 source-guard regression in that file was repaired and verified by the single targeted test above.

## MEASURED

- Parser validation for the new flags.
- Default helper/checkpoint-payload byte-shape identity at metadata level.
- Tail-average arithmetic closed form vs loop.
- Tail state persistence through the trainer's real `save_checkpoint` metadata writer.
- Gate-basis selector behavior for default live/warm, default EMA shadow, and active `ema_tail_average`.
- DSL lever compile and live trainer flag declaration.

## NOT-MEASURED

- No scorer, no `upstream/evaluate.py`, no n600 replay.
- No MLX training launch.
- No byte-closed archive row.
- No exact contest CPU/CUDA row.
- No true `load_checkpoint` resume round-trip in this sandbox: `load_checkpoint` calls `mx.array` and the session has no Metal device.
- No plateau classifier trigger; explicit epoch is the only built trigger.

## Boundaries

- Score claim: none.
- Axis claim: unit tests / parser / metadata only.
- Pointer: unmoved.
- No edits to common forbidden files.
- No scorer slot used.
- `T3_LIVE_ADAPTED` scope-law registration remains queued for dy1 merge.

## Follow-Ons

QUEUED-WITH-A-FIRE-ORDER:

1. When dy1 scope-law module merges, register the tail-average law under the `T3_LIVE_ADAPTED` scope and point this receipt at the canonical equation id.
2. MAIN at the jd4/tp1 boundary selects the explicit anchor epoch from the tp1 Case-0 plateau detector and fires the A/B via DSL, not hand flags.
3. On a Metal-capable host, run a true `save_checkpoint` -> `load_checkpoint` JD1 tail-state round-trip and compare resumed state before launch.
4. If classifier-triggered anchor is still desired, build it as a separate fully flagged, default-off, tested lever.

## File Hashes

- `experiments/train_tr1_partition_renderer_mlx.py` sha256 `612e5034192fdc438d48ee2c331fb3b5357820d461642b70d1593c3b4bdebe81`
- `src/tac/witness_dsl/spec_tr1_renderer_20260728.py` sha256 `ac4aa686db1a4056037ca5b2e55bf57212c8ada0a82744964e035b24f6ddbd41`
- `src/tac/tests/test_ddm_dy2_jd1_tail_average_ema.py` sha256 `b1ba1e81b0d39369be026c3a380727cc59c9d9c00cd06d12881b53852bbf5b2a`
- `src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py` sha256 `b0107c4a1ee575ab1c2795eb96e50ccc4b3817f3005bdea57f72d7543203fc7b`
- `src/tac/tests/test_ddm_bp1_boundary_reset_race.py` sha256 `942ebc47aa5d5e7b7f2556845796c29a93d03b1734bd49d4f0dd6135a1445eae`

Frontier line: S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
