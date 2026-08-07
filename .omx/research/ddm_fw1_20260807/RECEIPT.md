# ddm_fw1 receipt - rr8-F2 false-success receipts plus #983 residual

## Verdicts

FIXED rr8-F2 driver rc propagation:
- `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_n600_driver.sh` now captures the final stage rc in `final_rc`, writes `final rc: $final_rc`, and exits `"$final_rc"`.
- `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/hpac_our_labels_driver.sh` now records stage2/stage3/stage4 rc values into `overall_rc`, skips dependent stages after failures, writes `all done rc=$overall_rc`, and exits `"$overall_rc"`.
- `et4_repair_rows_and_final.sh` was checked and already exits `$rc` after its repaired final stage.

FIXED rr8-F2 guard coverage:
- Added warn-only staged shell scan in `tools/preflight_hook.py`: `_staged_shell_files`, `_shell_driver_rc_receipt_warnings`, and `run_shell_driver_rc_receipt_scan`.
- Guard warning class: a shell script logs `rc:` or `rc=` but has an unconditional `exit 0` or no variable `exit`/`return` propagation.
- Same-line waiver: `DRIVER_RC_EXIT0_OK:<reason>`.
- Added same-line waivers to the intentional detached/status paths in `tools/codex_companion_spawn.sh`.

FIXED #983 residual test coverage:
- `tools/preflight_hook.py` already contains the cb2/vw1 selector fixes for package `__init__.py`.
- Added `test_cb2_repro_pair_targets_are_ordered_subset_of_legacy_selection` so the cb2 repro tool's smaller ordered pair remains an ordered subset of the legacy broad selection.

## Measurements and checks

Driver custody:
- ET4 original driver: 1,486 bytes, sha256 `5d2191a920708b61f814ca6a361fae8130135a0e31fc4719ce023cec87620854`.
- HB1 driver: 3,766 bytes, sha256 `0ad62e320ba5f391c0b501556a571857c3bc859de14fdbe108f0443642cb3e25`.

Static shell coverage:
- `/Volumes/VertigoDataTier/pact/*/*.sh`: 40 files scanned by the new helper, 0 warnings after the ET4/HB1 fixes.
- `tools/*.sh`: 17 files scanned by the new helper, 0 warnings after the `codex_companion_spawn.sh` waivers.

Executed checks:
- `.venv/bin/python -m pytest src/tac/tests/test_preflight_hook.py -q` -> 51 passed.
- `.venv/bin/python -m py_compile tools/preflight_hook.py src/tac/tests/test_preflight_hook.py tools/repro_cb2_pr130_lift_pose_ci_blind_order.py` -> pass.
- `.venv/bin/ruff check --isolated --force-exclude --select F821 --ignore-noqa tools/preflight_hook.py src/tac/tests/test_preflight_hook.py tools/repro_cb2_pr130_lift_pose_ci_blind_order.py` -> pass.
- `git diff --check -- tools/preflight_hook.py src/tac/tests/test_preflight_hook.py tools/codex_companion_spawn.sh` -> pass.
- `bash -n tools/codex_companion_spawn.sh /Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_n600_driver.sh /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/hpac_our_labels_driver.sh` -> pass.
- `.venv/bin/python tools/review_tracker.py scan` -> 160272 entities across 10220 files.
- Review pass 1 marked: `tools/preflight_hook.py` (34 entities), `src/tac/tests/test_preflight_hook.py` (52 entities), reviewer `fw1-pass1`.
- Review pass 2 marked: `tools/preflight_hook.py` (34 entities), `src/tac/tests/test_preflight_hook.py` (52 entities), reviewer `fw1-pass2`.

## Recall evidence

| scope/query | finding beyond charter seeds | plan change |
|---|---|---|
| `.omx/research/ddm_rr8_20260806/ROUND8_FINDINGS.md` | rr8-F2 showed the launcher wrapper captured child rc correctly; the broken layer was shell drivers that echoed stage rc and then returned success. rr8-F1/F3/F4 were separate owners. | Did not edit `launch_detached_process.py` rc semantics; fixed driver exits and added shell guard. |
| `/Volumes/VertigoDataTier/pact/*/*.sh` grep for `echo rc`, `rc=`, `exit 0`, `set -uo`, done receipts | Live one-level SSD scripts with the et4/hb1 class were ET4 original and HB1; ET4 repaired final already exited rc. | Patched ET4 original and HB1 only; recorded ET4 repair as already closed. |
| `tools/*.sh` scan | `tools/codex_companion_spawn.sh` writes done markers but intentionally uses `unknown_detached` for non-child detached watcher completion. | Added explicit same-line waivers instead of changing it to a fake child rc. |
| `ddm_cb2` receipt, `NEXT_IF_RESUMED.md`, and commits `7900930594`, `88dc45548f` | The #983 selector source fix was already present; the unpinned residual was the repro tool's ordered pair contract. | Added a test that pins `PAIR_TARGETS` as an ordered subset of `legacy_pose_token_targets()`. |
| `.omx/state/main_hot_state.md` | `ddm_fw1` owns rr8-F2 and #983; no scorer slot owned; own-vehicle pointer `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`. | No scorer/eval dispatch; final boundary keeps pointer unchanged. |
| `tools/list_canonical_equations.py --json` filtered for `ci`, `preflight`, `mlx`, `SIGBUS`, `driver`, `receipt`, `rc`, `#983`, `shell` | Many MLX/advisory and receipt equations, but no equation requiring update for this shell receipt or selector-test guard landing. | No canonical-equation edit. |

## Not measured

- No n600 scorer job.
- No `upstream/evaluate.py`.
- No contest-CPU or contest-CUDA authority.
- No new score, archive, or frontier row.

## Follow-ons

- rr8-F2 ET4/HB1 false-success class: FIRED.
- rr8-F2 warn-only guard: FIRED, with positive-control tests executed.
- #983 intra-ordering residual: FIRED by regression test.
- Additional scorer or exact-eval work: not owned by this charter and not queued from this arm.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed and unmoved.
