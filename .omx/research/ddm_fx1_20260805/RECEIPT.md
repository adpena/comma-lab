# ddm_fx1 Receipt — Full-Stack Measured-Change Batch

Status: implementation complete in the working tree; commit landing blocked by managed-sandbox Git writes; no scorer runs, no launches, no live run dirs touched.

## Commit Landing Status

Serializer attempts:
- Patch-file intent mode reached the pre-commit hook, then the hook blocked on unrelated PG1/Q3 MLX tests in the dirty working tree (`src/tac/tests/test_ddm_tb1_tr1_renderer.py`), not on the fx1 staged patch.
- Patch-file intent mode with the hook's CI-blind skip then failed before commit because `git apply --cached` could not create Git backing-store temp files: `Operation not permitted`.
- Exact-OID temp-index `--no-stage` mode passed the staged preflight checks, then `git commit` failed opening `.git/COMMIT_EDITMSG`: `Operation not permitted`.

Commit SHA: none. `HEAD` remained `dd5d0c7e06` after the attempts, the shared index was clean, and the unrelated PG1/Q3 trainer working bytes were restored byte-for-byte (`sha256=2d2f61c7abd089704f9a9d2a9908c84e871560b7a77e38827aab42188454475e`).

## RECALL EVIDENCE

Sources and queries searched:
- Charter and contract: `.omx/tmp/codex_runs/fx1_prompt.md`; `.omx/tmp/codex_runs/_common_contract.md`.
- Governing files: `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`.
- Targeted corpus queries:
  - `rg -n "RR1-C2-R7-F1|RR1-C2-R7-F2|RR1-R7-F1|birth_amplify|pose term absent|resume-event pre-restore|parent-ACTIVE-decay|loss_term_keys|value-ledger" .omx/research .omx/state docs src/tac/tests experiments`
  - `rg -n "loss_terms|TR1_LOSS_TERM_KEYS|telemetry_v9_port|jd1_pose_finish|tr1_birth_amplify|boundary_jump|active_ema_decay|resume" experiments/train_tr1_partition_renderer_mlx.py`
  - `rg -n "levers|overrides|argv|regenerate|ticket|ema-decay|seg-hold-floor-source" experiments/ddm_jd1_ticket_regenerate.py`
  - `.venv/bin/python tools/list_canonical_equations.py --json | rg -n "JD1 plateau-tail EMA|w_pose_marginal_weight_law|pose_null_subspace|score.*d_pose|active_ema"`

Findings beyond the charter seeds:
- `.omx/state/main_hot_state.md` independently names the same three trainer telemetry debts as the next trainer-touching landing: pose+birth itemization, parent active decay, and resume-event pre-restore display.
- `.omx/research/ddm_rr1_20260805/RECEIPT.md` contains the load-bearing R7 findings for both the trainer defects and the regenerator value-ledger debt, including the measured parent cfg decay `0.999960019990005` vs active JD1 decay `0.9997777777777778`.
- `.omx/research/ddm_tp1_boundary_receipt_20260805.md` confirms the H8 telemetry-port gap and the resume-event gotcha on JD5.
- The canonical-equations registry reinforced the score law and JD1 EMA law; it did not contain a prior fix for these trainer telemetry defects.
- `experiments/ddm_jd1_ticket_regenerate.py` already contained `rebuild_lever_overrides_from_argv(...)` and `validate_lever_overrides_match_argv(...)` in the current checkout. Plan change: verify the regenerator debt with its focused tests rather than mutating already-emitted hash-custodied tickets or adding cosmetic churn.
- The trainer file also has unrelated same-file PG1/Q3-projector working-tree edits. Plan change: use serializer `--patch-file` intent mode for the commit so those unrelated hunks are not attributed to ddm_fx1.

## Changes

| Debt | Edit site | Regression killed | Proof |
|---|---|---|---|
| RR1-C2-R7-F1 / TP1 H8 | `experiments/train_tr1_partition_renderer_mlx.py`: active loss-term helpers, v9 telemetry declaration, `pair_loss(..., terms_out=...)`, and telemetry recompute. | `loss_terms` no longer leaves active JD1 `pose` and BI1 `birth_amplify` inside an unattributed residual. | `src/tac/tests/test_ddm_tp1_v9_telemetry_port.py::test_loss_terms_row_itemizes_active_jd1_pose_and_birth_amplify`; full TP1 suite passed. |
| RR1-C2-R7-F2 | `experiments/train_tr1_partition_renderer_mlx.py`: `parent_boundary_ema_decay_fields(...)`, `resume_ema_decay_fields(...)`, and `boundary_jump_row(..., parent_cfg_ema_decay=...)`. | First post-resume `boundary_jump` compares parent active JD1 decay against child active decay, not parent config decay against child active decay. | `src/tac/tests/test_ddm_bp1_boundary_reset_race.py::test_boundary_jump_uses_parent_active_jd1_decay_not_cfg_decay`. |
| Resume display debt | `experiments/train_tr1_partition_renderer_mlx.py`: resume event now carries `post_restore_active_ema_decay`, provenance, `child_cfg_ema_decay`, and separate parent cfg/active fields. | A resumed JD1 window has an event row naming the restored active decay instead of forcing readers to infer it from the first epoch row. | `src/tac/tests/test_ddm_bp1_boundary_reset_race.py::test_resume_event_fields_carry_post_restore_active_decay`. |
| RR1-R7-F1 regenerator value ledger | `experiments/ddm_jd1_ticket_regenerate.py` current checkout, verified unchanged by fx1. | Future emitted tickets rebuild `levers[*].overrides` from final argv and refuse declared-vs-argv mismatches. | `src/tac/tests/test_ddm_jd1_ticket_regenerate.py` passed; existing tests cover stale base-ticket repair and forced mismatch refusal. |

Implementation choice for Change 3: added explicit post-restore fields to the resume event rather than moving the event. This is less invasive to resume ordering and preserves the existing restore sequence while making the restored active value machine-readable.

## Verification

Commands run:
- `.venv/bin/python -m py_compile experiments/train_tr1_partition_renderer_mlx.py experiments/ddm_jd1_ticket_regenerate.py src/tac/tests/test_ddm_tp1_v9_telemetry_port.py src/tac/tests/test_ddm_bp1_boundary_reset_race.py src/tac/tests/test_ddm_jd1_ticket_regenerate.py` — passed.
- `git diff --check -- experiments/train_tr1_partition_renderer_mlx.py src/tac/tests/test_ddm_tp1_v9_telemetry_port.py src/tac/tests/test_ddm_bp1_boundary_reset_race.py` — passed.
- `.venv/bin/python -m pytest src/tac/tests/test_ddm_tp1_v9_telemetry_port.py` — 18 passed.
- `.venv/bin/python -m pytest src/tac/tests/test_ddm_jd1_ticket_regenerate.py` — 5 passed.
- `.venv/bin/python -m pytest src/tac/tests/test_ddm_dy2_jd1_tail_average_ema.py` — 12 passed.
- `.venv/bin/python -m pytest src/tac/tests/test_ddm_bp1_boundary_reset_race.py -k 'parent_active_jd1_decay or resume_event_fields'` — 2 passed.
- `.venv/bin/python -m pytest src/tac/tests/test_ddm_bp1_boundary_reset_race.py` — 38 passed, 6 failed on local MLX Metal-device availability (`[metal::load_device] No Metal device available`). This is the known non-Metal/#856 class; the failures are in MLX optimizer/checkpoint tests, not the new pure decay tests.
- `tools/review_tracker.py mark-file ... --status reviewed` twice for each edited Python file: `experiments/train_tr1_partition_renderer_mlx.py`, `src/tac/tests/test_ddm_tp1_v9_telemetry_port.py`, `src/tac/tests/test_ddm_bp1_boundary_reset_race.py`.

No `REVIEW_GATE_OVERRIDE` was used.

## Boundaries

- No scorer job ran.
- No archive was built.
- No live run dir was touched.
- No already-emitted ticket was mutated.
- No contest score or advisory score changed in this landing.

Own-vehicle frontier line unchanged: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; contest pointer `0.1910828242 [contest-CPU]` remains borrowed and unmoved.
