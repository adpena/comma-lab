# ddm_gov3 — Metal occupancy, progress-budget timeout, and SUM-over-RAM refusal

Status: COMPLETE for the bounded apparatus implementation and tests. This arm is `score_claim=false`, ran no scorer, launched no governed research process, created no bulky payload, and did not change the frontier.

# FORMALIZATION_PENDING:governor apparatus — the admission-table rows are seeded data, not a fitted law; the law lands when N≥3 windows exist

## RESULT

The launcher and admission path now charge the Metal footprint that Apple-Silicon process RSS misses, refuse a second Metal occupant until three distinct measured dual-Metal FIT windows exist, treat `history.jsonl.completed_steps` as the health signal when present, and hard-refuse a SUM-over-RAM projection more than 5 GiB above the adaptive ceiling. The 2026-09-05 replays are: 16:12 cl3 beside md3 → REFUSE; 22:30 md3 beside five CPU-only sj1 shards → ADMIT; 21:05 projected 125.2 GiB against 116.0 GiB → REFUSE.

## ITEM 3 — METAL OCCUPANCY

- Files and lines: `tools/measured_peaks.py:51-113,166-170,311-338,455-471` adds the >3 availability-delta/RSS classifier and persisted `metal_footprint_gib`; `tools/cell_admission.py:231-342,405-421,756-775,1316-1414,1516-1529` joins argv, manifest, and ledger evidence and makes the footprint bind declared live cost; `tools/launch_detached_process.py:480-535,1203-1288,1592-1596` writes the profile into the resource budget and manifest and charges it as candidate projected cost; `tools/costate_digest.py:348-389,407-410` names live occupants.
- Rule: `--metal`, `device mps`, `--device mps`, `--device=mps`, the legacy `run-config` cell shape, or a measured ledger availability-delta/RSS ratio >3 makes a governed job a Metal occupant. A Metal candidate needs at least two estimated free logical cores. If any Metal occupant is live, another is refused until the admission table has three distinct rows marked `measured_window=true`, `concurrent_metal_occupants>=2`, and `outcome=FIT`.
- Tests added, 13 item-facing assertions: `test_cl3_direct_trainer_argv_counts_as_an_occupant`, `test_run_config_cell_counts_as_an_occupant`, `test_cpu_pricer_does_not_count_as_an_occupant`, `test_ledger_ratio_above_three_counts_without_mps_argv`, `test_two_occupants_refuse_with_the_rule_chain`, `test_seed_table_has_profiles_but_no_dual_fit_proof`, `test_duplicate_fit_rows_do_not_fake_three_windows`, `test_cpu_candidate_does_not_inherit_live_metal_contention`, `test_launcher_derives_progress_far_cap_and_records_escape`, `test_1612_cl3_beside_md3_refuses`, `test_2230_md3_beside_five_cpu_shards_admits`, `test_lookup_backfills_metal_footprint_from_the_ratio`, and `test_names_a_non_run_config_metal_occupant`.
- Replay verdicts: `[macOS-CPU apparatus replay]` 16:12 is REFUSE because md3 is named as a live occupant and the table has 0/3 dual-fit proofs. `[macOS-CPU apparatus replay]` 22:30 is ADMIT because all five sj1 shards are CPU-only, 49.572 + 28.0 GiB stays below 116.0 GiB, and 11.5 estimated logical cores remain above the two-core feed floor.

### Admission-table seed rows

The immutable seed table is `.omx/research/ddm_gov3_metal_admission_table_seed_20260905.jsonl:1-3`; future measured windows append to the live consumer `.omx/state/metal_admission_table.jsonl`:

| row | occupant | Metal footprint / RSS | CPU fact | disposition |
|---|---|---:|---:|---|
| 1 | cl3 trainer | 34.75–38.622 / 1.5 GiB | — | PROFILE_ONLY; not a dual-fit proof |
| 2 | md3 Metal cell | 49.572 / 1.75 GiB | — | PROFILE_ONLY; not a dual-fit proof |
| 3 | sj1 shard | non-Metal / 5.7 GiB RSS | 1.0–1.3 cores; Metal requires 2.0 free | CPU_PROFILE_ONLY |

Seed denominator: 3 profile rows; dual-Metal FIT numerator: 0; dual-Metal FIT denominator required for admission: 3 distinct measured windows.

## ITEM 5 — PROGRESS-BUDGET TIMEOUT

- Files and lines: `tools/launch_detached_process.py:515-535,1225-1284` derives the heartbeat path from the authorized run-config, defaults the stall budget to `10 * (serial_walltime_estimate_s / total_steps) / 60`, and makes the far ceiling 3× the serial estimate; `tools/safe_run.py:109-171,334-356,530-600,683-720` reads the last valid `completed_steps`, resets the stall clock only on progress, preserves the old timeout when no heartbeat ever appears, and records all timeout state in the status receipt.
- Rule: once any valid heartbeat exists, the old wall cap is not a kill condition. No advance for `--progress-stall-minutes N` kills; the far ceiling remains absolute. If no heartbeat ever exists, `--timeout` behaves exactly as before.
- Tests added, 4 item-facing assertions: `test_stalled_heartbeat_kills`, `test_slow_advancing_heartbeat_survives_the_old_cap`, `test_no_heartbeat_preserves_the_old_timeout`, and the launcher derivation assertions in `test_launcher_derives_progress_far_cap_and_records_escape`.
- Measured test behavior: the stalled fixture exits 124 with `status=progress_stall`; the advancing fixture runs beyond its 0.15 s old cap and exits 0; the absent-heartbeat fixture exits 124 with `status=timeout`.

## ITEM 6 — SUM-OVER-RAM HARD REFUSAL

- Files and lines: `tools/safe_run.py:65,296-331,391-428,596-600` names the 5 GiB attribution error, applies the hard refusal independently of global advisory mode, rejects placeholder escapes, prints the rule chain, and records escape rationale/use; `tools/launch_detached_process.py:1285-1288,1361-1411` passes and records only a substantive `--admit-over-projection` rationale.
- Rule: for a governor REFUSE, `projected_system_used_gib - adaptive_ceiling_gib > 5.0` returns rc 5 even when global admission is advisory. An overage at or below 5.0 retains the pre-existing advisory behavior. Only `--admit-over-projection "<substantive rationale>"` escapes the hard boundary; the older general override cannot.
- Tests added, 6 item-facing assertions: `test_125_2_vs_116_is_a_hard_refusal`, `test_118_vs_116_remains_advisory`, `test_substantive_escape_admits_and_placeholder_refuses`, `test_integrated_gate_prints_hard_rule_chain_and_escape`, the escape-record assertions in `test_launcher_derives_progress_far_cap_and_records_escape`, and `test_2105_projection_refuses`.
- Replay verdict: `[macOS-CPU apparatus replay]` 125.2 - 116.0 = 9.2 GiB > 5.0 GiB, so the 21:05 launch is REFUSE even with global enforcement off.

## VERIFIED-AT-SOURCE NUMERIC PREMISES

- 116.0 GiB ceiling — verified-at-source: `tools/system_memory_governor.py:167`; mirrored by `tools/cell_admission.py:82`.
- 5.0 GiB attribution error — verified-at-source: `.omx/research/ddm_gov3_directive_cooperative_pause_and_stale_instance_20260905.md:88`; named in `tools/safe_run.py:65`.
- 18,000 s killed-cell cap — verified-at-source: `.omx/research/ddm_gov3_directive_cooperative_pause_and_stale_instance_20260905.md:77` and `experiments/ddm_md3_different_init_cell.py:319`.
- `system_availability_delta_gib` ledger field — verified-at-source: `tools/measured_peaks.py:168`; its prior two-instrument rule is documented in `.omx/research/ddm_gov2_control_plane_permanence_20260904.md:113-120`.
- Historical footprints and CPU demand — verified-at-source: `.omx/research/ddm_gov3_directive_cooperative_pause_and_stale_instance_20260905.md:45-73`.

## TEST AND REVIEW RECEIPT

- `.venv/bin/pytest -q --tb=short src/tac/tests/test_cell_admission.py src/tac/tests/test_ddm_gov3_governor.py src/tac/tests/test_measured_peaks.py` → 138 passed.
- `.venv/bin/pytest -q --tb=short src/tac/tests/test_costate_digest_live_cells.py src/tac/tests/test_safe_run_double_gate.py tools/tests/test_watched_launch_hardening.py` → 31 passed.
- Total bounded regression surface: 169 passed; 21 test functions added and 2 existing concurrency tests updated to the independent-occupancy semantics.
- Targeted Ruff checks passed for all changed implementation files and the new test; `git diff --check` passed.
- All 9 changed Python files received two genuine, registered-principal `review_tracker.py mark-file` passes: `ddm_gov3_registered_pass1` and `ddm_gov3_registered_pass2`.
- Implementation and tests landed as `29f76e36d` with post-edit serializer hashes, `[no-triality] [p0-ledger-ok]`, and no attribution trailer.

## PREFLIGHT / TWO-LANDING DISPOSITION

Inspected `src/tac/preflight.py` and `src/tac/confound_gates.py:5824-5894`. Existing launch/admission preflight gates inspected: 1 (`check_cell_launches_only_through_queue_driver`, Catalog #413). Existing gates covering Metal classification, progress timeout, or the 5 GiB hard-refusal semantics: 0. New gates added: 0. Live covered-surface violation count: 0. Each of the three fixes lands with its tests in this implementation landing; a fake second gate was not added because Catalog #413 governs routing through the queue driver, not these three decision semantics.

The directive extractor registered ITEMs 3, 5, and 6 under owner `ddm_gov3`; each lifecycle now serves as `completed`, `test_status=green`, with implementation commit `29f76e36d`. The handoff memo and those append-only ledger events are the evidence landing; unrelated concurrent ledger rows remain outside this arm's custody.

## RECALL EVIDENCE

Searched the full `.omx/research/` corpus with `metal footprint|metal occupant|progress.stall|completed_steps.*timeout|125.2|attribution error|sum.over.ram`; searched the canonical equation listing and `src/tac/canonical_equations/` with `metal|memory|admission|timeout|progress`; searched `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, and `.omx/state/canonical_task_status.jsonl` with `metal footprint|metal occupant|progress budget|SUM-over-RAM|cell admission`.

Beyond the charter seeds, the recall found `adaptive_ceiling_admission_control_v1` at `src/tac/canonical_equations/adaptive_ceiling_admission_control_20260703.py:13-18,82-99`, the gov2 measured-peak max-of-two-instruments finding at `.omx/research/ddm_gov2_control_plane_permanence_20260904.md:108-120`, and the safe-run stale-reservation reconciliation fix at `.omx/research/ddm_gb1_memory_blackbox_fix_verdict_20260815.md:119-135`. This changed the implementation in two ways: the 5 GiB tolerance was layered on the existing canonical SUM-over-RAM decision instead of inventing another admission equation, and the new Metal column was made to bind both live declared cost and candidate projection rather than exist as report-only metadata. No additional current progress-timeout or dual-Metal-fit law was found in the searched scopes.

## WHAT WAS NOT DONE

- No live dual-Metal experiment was run: MAIN owns dispatch and single-flight; the existing N=2 evidence remains unresolved. The table therefore contains zero dual-fit proofs.
- No canonical Metal-concurrency law was fitted: N=0 qualifying dual-fit windows, below the required N=3.
- No new preflight gate was invented: the only nearby gate does not cover these semantics, and Catalog quota discipline forbids a decorative duplicate.
- No SIGSTOP, renice, or compressor-delta actuator was reintroduced. Those paths are closed by the directive's measured negatives.
- No scorer, Modal dispatch, sj1 file/process, `upstream/`, or `submissions/semantic_joint_ctxmix/` surface was touched. No score or frontier movement is claimed.

## NEXT_IF_RESUMED

- QUEUED-WITH-A-FIRE-ORDER — owner: MAIN; consumer store: `.omx/state/metal_admission_table.jsonl`; fire trigger: schedule an operator-authorized dual-Metal measurement only when the SUM-over-RAM projection fits, at least two logical cores remain free for each Metal cell, no scorer/single-flight collision exists, and the run will append a distinct measured window with an explicit FIT or REFUSE outcome.
- QUEUED-WITH-A-FIRE-ORDER — owner: ddm_gov3 successor; consumer store: `src/tac/canonical_equations/` plus the canonical equation registry; fire trigger: the admission table reaches at least three distinct qualifying dual-Metal FIT windows, at which point fit the concurrency law and replace this memo's FORMALIZATION_PENDING line.

## LIVE-HYPOTHESES

- Three controlled paired-Metal windows may prove safe concurrency for smaller-footprint families because the seeded cl3 and md3 costs leave materially different system headroom; this is plausible but untested, and the current 0/3 table cannot admit it.
- Progress-stall budgets derived from serial seconds/step should preserve healthy CPU-starved Metal cells because advancing `completed_steps` directly distinguishes slow work from no work; the unit fixtures prove mechanism behavior, not a live long-run distribution.

## DEAD-ENDS

- SIGSTOP pause actuation stays retired: four measured pauses helped zero times and made the stopped job a swap victim.
- Renicing CPU arms stays retired as the timeout cure: the measured Metal-host rate did not change after +10 niceness.
- Compressor delta stays retired as Metal-footprint attribution: it was the wrong 45-second instrument; launch-scoped system-availability delta is the retained measure.
- The two existing N=2 contention windows do not prove two Metal occupants fit: ratios 1.117 and 0.964 straddle the baseline and their spread exceeds the effect.

sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]
