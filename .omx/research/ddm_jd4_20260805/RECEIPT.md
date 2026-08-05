# ddm_jd4 continuation-prep receipt

Date: 2026-08-05

Status: BUILD arm complete. No scorer job, no Metal/MLX training run, no archive build, no exact
evaluation, and no launch. The emitted continuation ticket is `launch_now=false`; MAIN fires it
only after the n600 both-bases endpoint probe completes.

## RECALL EVIDENCE

| Surface | Query / artifact | Result beyond charter seeds | Plan change |
|---|---|---|---|
| Memory registry | `rg -n "jd4|jd1|jd3|rr1|tp1|cross_regime|force-ema|regenerator|jd1-force" ~/.codex/memories/MEMORY.md` | No jd4-specific memory row found. Existing Pact memory reinforced serializer/review discipline and report-only pointer honesty. | Treat live artifacts as authority; do not infer a prior jd4 cure. |
| Governing files | `_common_contract.md`, `jd4_prompt.md`, `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | `CLAUDE.md` and `AGENTS.md` are byte-identical (`65da6dd8...`). Hot state says jd4 is scorer-free and the n600 endpoint probe owns the scorer slot. | No launch, no scorer, no protected-file edits, commit via serializer with post-edit hashes and review marks. |
| TP1 boundary receipt | `.omx/research/ddm_tp1_boundary_receipt_20260805.md` lines around RR1 R4/R6/R7 and endpoint | Confirmed all five regenerator debts: argv interpreter, unique out-dir, force full-window reanchor, recursive path repair, and lever override rebuild. Endpoint is ep1405, final checkpoint sha starts `2c3bd24455...`. | Implement all five debts in the regenerator factory and emit a continuation from the endpoint snapshot. |
| RR1 review receipt | `.omx/research/ddm_rr1_20260805/RECEIPT.md` | RR1-R4-F1/R6-F1/R7-F1 are the evidence rows for the state-flag, recursive-template, and lever-ledger inheritance failures. | Added fail-closed tests for each refuse surface; do not mutate the old fired ticket. |
| Endpoint snapshot | `/Volumes/VertigoDataTier/pact/ddm_jd3_20260805/full_v3_endpoint_ep1405_snapshot/` | `stage_joint_pose_finish_final.npz` sha256 `2c3bd24455eedeeb015ce7304375a643dd9ad5d691726961707ef357bf2fe048`; checkpoint field epoch `1406`, telemetry tail last epoch `1405`, active EMA provenance `U=1200`. | The forced resume helper uses tail-derived next epoch `1406` for window geometry and records the checkpoint epoch-field caveat. |
| Canonical equation recall | `tools/list_canonical_equations.py --json` filtered for EMA, plus `src/tac/canonical_equations/evaluators.py` | EMA law is executable: `decay_from_warmup_fraction`, so `U=18000` gives `0.9997777777777778`. | Ticket records derived U and decay; trainer re-derives at forced resume rather than copying the smoke latch. |
| Broader corpus search | Targeted `rg` over `.omx/research`, `.omx/state`, docs, reports for jd4/jd3/recursive/lever/EMA terms | Found only the live hot-state/TP1/RR1 surfaces already consumed; no additional jd4 cure found in searched scope. | No extra side tasks; keep scope to trainer, regenerator, tests, ticket, and receipt. |

## Fix Table

| Debt | Fix | Verification |
|---|---|---|
| Trainer force reanchor | Added args-only `--jd1-force-ema-reanchor-on-resume`. When set on JD1 resume, carried `stage_ema_reanchored=true` no longer suppresses the reanchor; telemetry records old carried decay/provenance and new derived decay/provenance. | `pytest src/tac/tests/test_ddm_bp1_boundary_reset_race.py -q -k 'jd3 or jd4'`: `7 passed, 35 deselected` (known no-Metal atexit warning after pass). |
| Terminal checkpoint epoch geometry | Under the force flag only, resume geometry uses parent telemetry tail (`last_tail_epoch+1`) when the final checkpoint stores an exclusive epoch. Off path remains `saved_epoch+1`. | `test_jd4_forced_resume_start_epoch_uses_terminal_tail_geometry`. |
| Regenerator argv[0] rc=126 class | Finalization normalizes emitted tickets to `/Users/adpena/Projects/pact/.venv/bin/python` at argv[0] and the trainer script at argv[1]. | `test_jd4_continuation_emission_repairs_all_debt_surfaces`; emitted ticket jq check shows argv0/argv1 correct. |
| Unique child out-dir | Regenerator refuses to emit if the child out-dir already has checkpoint NPZs. jd4 child is `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/tr1_jd4_cont_ep1406`. | `test_child_out_dir_checkpoint_reuse_refuses`. |
| Geometry-change force flag | Regenerator compares parent checkpoint stage EMA `U=1200` to new window `U=18000`; mismatch automatically adds `--jd1-force-ema-reanchor-on-resume`. | Emitted ticket `regenerated_from.parent_stage_ema_u=1200`, `new_window_u=18000`, `force_ema_reanchor_on_resume=true`. |
| Recursive resume template | Finalization rewrites `recursive_encode_pass_loop.continue_policy.next_resume_from_template` under the actual child out-dir and validates containment. | `test_recursive_template_must_stay_under_child_or_declared_new_dir`; emitted ticket template resolves under jd4 child out-dir. |
| Lever override ledger | Finalization rebuilds `levers[*].overrides` from final argv, then refuses any declared-vs-argv mismatch or missing flag. | `test_declared_lever_override_mismatch_refuses`, `test_missing_declared_lever_override_refuses_during_rebuild`, emitted ticket validation helper returned `validated=true`, `lever_count=23`. |

## Ticket

| Field | Value |
|---|---|
| Ticket | `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd4_ticket_cont_ep1406.json` |
| File sha256 | `a22783a9340c13e60fc8e79dc6f186d0570e0054f43cb79fe1a89c15ab171130` |
| Internal ticket_hash | `51c64222b432b1abfac8cdb0d72ba39622573ce8a27c8e868e7144df26f93076` |
| Resume checkpoint | `/Volumes/VertigoDataTier/pact/ddm_jd3_20260805/full_v3_endpoint_ep1405_snapshot/stage_joint_pose_finish_final.npz` |
| Resume checkpoint sha256 | `2c3bd24455eedeeb015ce7304375a643dd9ad5d691726961707ef357bf2fe048` |
| Child out-dir | `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/tr1_jd4_cont_ep1406` |
| Epoch limit | `1526` |
| Window geometry | `120 epochs x 150 steps/epoch = U=18000` |
| Derived stage EMA decay | `0.9997777777777778` |
| Wall cap | `165` minutes (`55 s/epoch x 120 x 1.5`) |
| Launch status | `launch_now=false`; MAIN fires after endpoint probe completion |

## Verification

- `py_compile`: PASS for `experiments/train_tr1_partition_renderer_mlx.py`, `experiments/ddm_jd1_ticket_regenerate.py`, `src/tac/tests/test_ddm_bp1_boundary_reset_race.py`, and `src/tac/tests/test_ddm_jd1_ticket_regenerate.py`.
- `pytest src/tac/tests/test_ddm_jd1_ticket_regenerate.py -q`: PASS, `5 passed`.
- `pytest src/tac/tests/test_ddm_bp1_boundary_reset_race.py -q -k 'jd3 or jd4'`: PASS, `7 passed, 35 deselected`; MLX emitted the known no-Metal atexit warning after test completion.
- Emitted ticket JSON validated by regenerator helpers: recursive template containment and lever overrides match final argv.

## SHA-256

| Artifact | sha256 |
|---|---:|
| `experiments/train_tr1_partition_renderer_mlx.py` | `ebe2eadcc0d2dee8ad387cbc63ee1c3949e67e767f6918ff9fcd15bbfabd02cb` |
| `experiments/ddm_jd1_ticket_regenerate.py` | `5606d9c9395d1d8d853e90c7440dbd0228a04fbd4c205ca619392516ad04fc58` |
| `src/tac/tests/test_ddm_bp1_boundary_reset_race.py` | `836053559f0212eef828d5f63f0e6a39ce0432bac16995b80f91d932767a7b9e` |
| `src/tac/tests/test_ddm_jd1_ticket_regenerate.py` | `3a57462ccc8ab5603fb01455e59bfca5c21f0d8ef3a930c7f47014eefb064822` |
| `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd4_ticket_cont_ep1406.json` | `a22783a9340c13e60fc8e79dc6f186d0570e0054f43cb79fe1a89c15ab171130` |

## Boundaries

- No score was measured by this arm.
- No scorer slot was used; the n600 both-bases endpoint probe remains the owner of the scorer slot.
- No Metal run, no trainer launch, no archive, no byte-close, no contest CPU/CUDA eval.
- The emitted ticket inherits the fired v3 argv except for the required continuation fields and the force-reanchor repair.
- The final checkpoint has `meta::epoch=1406` and telemetry tail last epoch `1405`; jd4 records and handles this as terminal-checkpoint exclusive-epoch geometry under the force flag.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
