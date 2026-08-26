# DDM PF2X preflight-chain burn-down — 2026-08-26

## Verdict

The OSS-export mirror-helper meta-gate is strict-clean: **20 -> 0** live sites,
landed as commit `f022869197`. The charter named ten sites, but live re-derivation
and the r56 receipt both show twenty; r56 printed the first ten and then
`... and 10 more`. All twenty were the same copied scanner-scope fact and took
sanctioned cure (a): filter the path through `_is_oss_export_mirror_path` before
any gate-specific parsing. No waiver, gate deletion, or scope relaxation outside
`comma_lab_public_export/` was introduced.

The full chain is **RED / BLOCKED_ENVIRONMENT_CAPABILITY** at r57. The detached
runner reached `check_no_live_mcp_processes(strict=True)`, where this managed
sandbox refused execution of `ps -axo pid=,command=` with
`PermissionError: [Errno 1] Operation not permitted: 'ps'`. A mocked or skipped
process census is not a real full-preflight pass, and changing that safety gate is
outside this charter's mechanical-hygiene authority. The chain therefore stopped
after one cure round and one blocked rerun. No scorer, Modal call, archive mutation,
or score measurement ran; the frontier did not move.

## RECALL EVIDENCE

The full required in-repo surfaces were searched by content before adjudication:

- `.omx/research/`, arm receipts, docs, reports, and state for
  `comma_lab_public_export`, `OSS export mirror`, `mirror helper`,
  `check_preflight_scanners_use_oss_mirror_helper`, `preflight_full_r5*`,
  Check 126, no-MPS scope, #1302, and #1303;
- `tools/list_canonical_equations.py --json` for `preflight`, `mirror`, `scanner`,
  `OSS export`, `cleanup`, `modal call`, `Rudin`, and `DP1`;
- `CANONICAL_RESEARCH_INDEX*`, every `sub015_DAG_*` FEED surface, the harness
  bridge, canonical task-status ledger, lane registry, and live hot state;
- r55/r56 runner scripts and receipts, MG1's disposition-table memo, the no-MPS
  scope precedent, Check-126 cure, HD1/FC1X serializer custody, the live mirror
  helper/meta-gate source, and every affected gate's focused tests.

Beyond the charter seeds, recall changed the plan in two material ways:

1. The live population was twenty, not ten. The extra ten were not a new genus:
   six adjacent Rudin/Daubechies scanners, two later preflight scanners, the
   residual-coverage scanner, and the Codex-spawn scanner had the identical
   omission. They were folded into the same cure batch rather than left for an
   immediately predictable r58 repeat.
2. The mirror meta-gate documents a known static limitation: it sees only literal
   `rglob("*.py")`, `rglob("*.sh")`, and `rglob("*")` forms. That did not change
   this cure—all twenty live sites were visible literal forms—but it remains an
   honest boundary on the gate's future completeness.

No canonical equation directly governed this scanner-scope repair. The index,
DAG, bridge, lane registry, and pre-existing task ledger did not contain a PF2X
task row in the searched scope. PF2X registered
`ddm_pf2x_preflight_chain_burndown` from the charter through the canonical
task-status store; this is bounded absence in those named stores, not a global
nonexistence claim.

## Per-site dispositions

Every row uses cure **(a)**: filter the recursively discovered path through the
canonical helper before any gate-specific logic. The behavior boundary is the same
for all rows: canonical sources remain scanned; only paths whose components contain
`comma_lab_public_export` are excluded as staging duplicates.

| # | Scanner | Disposition |
|---:|---|---|
| 1 | `_check_154_manifestless_cleanup_identity` | (a), filter `path`; canonical manifestless-cleanup producers remain checked. |
| 2 | `_check_199_iter_candidate_files` | (a), filter `path`; canonical operator-authorize bypass candidates remain checked. |
| 3 | `_check_209_iter_target_files` | (a), filter `path`; canonical DP1 distillation callers remain checked. |
| 4 | `_check_210_iter_target_files` | (a), filter `path`; canonical DP1 provenance builders remain checked. |
| 5 | `_check_211_iter_target_files` | (a), filter `path`; canonical DP1 composition callers remain checked. |
| 6 | `check_modal_dispatches_register_call_id` | (a), filter `py`; canonical Modal spawn sites remain checked. |
| 7 | `check_slim_ranker_consumes_canonical_taylor_proxies` | (a), filter `path`; canonical SLIM violations remain checked. |
| 8 | `check_falling_rule_list_canonical_use` | (a), filter `path`; canonical falling-rule violations remain checked. |
| 9 | `check_rashomon_ensemble_continual_update_locked` | (a), filter `path`; canonical persistence violations remain checked. |
| 10 | `check_compressive_landscape_canonical_use` | (a), filter `path`; canonical dense-anchor violations remain checked. |
| 11 | `check_wavelet_multi_scale_ranker_contract` | (a), filter `path`; canonical multi-scale violations remain checked. |
| 12 | `check_gosdt_dispatcher_whiteboard_discipline` | (a), filter `path`; canonical auto-promotion violations remain checked. |
| 13 | `check_preflight_slim_risk_scorer_canonical_use` | (a), filter `path`; canonical preflight-SLIM violations remain checked. |
| 14 | `check_preflight_falling_rule_list_canonical_use` | (a), filter `path`; canonical preflight rule-order violations remain checked. |
| 15 | `check_preflight_rashomon_ensemble_continual_update_locked` | (a), filter `path`; canonical preflight persistence violations remain checked. |
| 16 | `check_preflight_compressive_landscape_canonical_use` | (a), filter `path`; canonical preflight dense-anchor violations remain checked. |
| 17 | `check_preflight_wavelet_multi_scale_contract` | (a), filter `path`; canonical preflight multi-scale violations remain checked. |
| 18 | `check_preflight_gosdt_dispatcher_whiteboard_discipline` | (a), filter `path`; canonical preflight auto-promotion violations remain checked. |
| 19 | `check_residual_override_has_coverage_proof` | (a), filter `entry`; canonical residual-coverage consumers remain checked. |
| 20 | `check_codex_exec_spawn_paths_are_reaper_immune` | (a), filter `py`; canonical Codex spawn paths remain checked. |

Population accounting: (a) 20, (b) 0, (c) 0; zero waivers. Per #821 this is one
copied-pattern mechanism fact with twenty live source sites.

## Verification receipts

- Strict meta-gate before cure: **20 violations**; every live function named above.
- Strict meta-gate after cure: **0 violations** and strict mode returns normally.
- The affected gates' existing focused pytest population collected 154 tests and
  completed green after the cure. Their canonical positive fixtures still fire;
  the fix did not blanket-disable the gate-specific signatures.
- `py_compile src/tac/preflight.py`: green.
- `ruff check src/tac/preflight.py`: green.
- `git diff --check`: green.
- Two genuine `review_tracker.py mark-file` passes completed for
  `src/tac/preflight.py`; no review override was used.
- Serializer intent-manifest mode committed exactly one file and 40 inserted lines:
  `f022869197`. It ignored FC1X's separately owned dirty hunks in the shared file.

## Chain rounds

| Round | State | Gate / blocker | Disposition | Receipt |
|---|---|---|---|---|
| r56 | RED | Mirror-helper meta-gate, 20 live sites although only ten were printed before the truncation line | Mechanical scanner-scope cure (a), committed `f022869197` | `.omx/tmp/preflight_full_r56_20260826/PREFLIGHT_RESULT.json` |
| r57 | RED / BLOCKED_ENVIRONMENT_CAPABILITY | `check_no_live_mcp_processes`: real `ps` census denied by managed sandbox | STOP per charter; no mock, skip, waiver, or safety-gate edit | `.omx/tmp/preflight_full_r57_20260826/PREFLIGHT_RESULT.json` |

r57 was launched through `tools/launch_detached_process.py` with unique done receipt
`pf2x_r57`. The result JSON SHA-256 is
`50ddd7a5437933a1232a3da3115d934037846f4ac64dc97660b5e66427b68676`;
the run log SHA-256 is
`a177038552325aea3bc06f93542dffc0760b92c090f16c7a6dd4b06ece4e5750`;
the launch manifest SHA-256 is
`b7eddbe1c8b9fe0f1855c22ab02971a889c9f405e17d195f0666a3583769a541`;
the detached done receipt SHA-256 is
`52ce10d03e1f2c6bf13f002f2c734b1fb10460ac027e0ae85c8df8093ae8b285`.
The done receipt's rc=0 means the wrapper completed and wrote its typed result; it
does not override the result JSON's RED verdict.

## Ledger and boundaries

Canonical task `ddm_pf2x_preflight_chain_burndown` was registered and advanced to
`in_progress` by actor/session `ddm_pf2x` / `ddm_pf2x_20260826`; the terminal
blocked row names the r57 capability blocker and commit `f022869197`. Consumer
store: `.omx/state/canonical_task_status.jsonl`.

This arm measured scanner counts, test outcomes, process-execution capability,
and receipt hashes only. It did not execute a scorer, inspect or mutate archive
bytes, invoke Modal, change equations, change launch configs, make a score claim,
or move the canonical frontier.

GESTALT-DELTA: the apparent ten-site residue was a display-truncated twenty-site
population of one copied scanner pattern; that class is now dry, and the next
chain boundary is not another code finding but the environment's inability to
perform the safety gate's real live-process census.

## NEXT_IF_RESUMED

- **BLOCKED-WITH-A-FIRE-ORDER** — owner: MAIN in an execution environment that permits a real process inventory; consumer store: `.omx/tmp/preflight_full_r57_20260826/`; fire trigger: `ps -axo pid=,command=` executes successfully without mocking or bypass, then launch r58 with the same detached runner pattern and continue only for charter-defined mechanical hygiene reds.

## LIVE-HYPOTHESES

- `tools/codex_arm_queue.py` may be the next non-mirror code red after the process-census blocker clears. This is plausible because its focused live-count test currently reports one `check_codex_exec_spawn_paths_are_reaper_immune` violation, but r57 never reached that gate, so it is not yet a full-chain conclusion.
- The remaining chain may still contain dark-window mechanical hygiene debt. This is plausible because r50-r56 repeatedly exposed stale scanner/fixture populations, but r57 stopped too early to test the charter's prediction of at least three additional cures.

## DEAD-ENDS

- Treating the charter's displayed ten rows as the full population is closed: both the r56 truncation line and live strict scan prove twenty.
- Blanket waiver or helper-gate relaxation is closed: all twenty sites accepted direct canonical filtering and the strict count reached zero.
- Whole-file serializer staging is closed for this landing: FC1X owned separate uncommitted hunks in the same file; intent-manifest mode produced the exact 40-line commit without absorption.
- Mocking `ps`, supplying an empty synthetic process list, or swallowing `PermissionError` to call r57 green is closed: each would remove the live-process safety evidence the gate claims to provide.
- Opportunistically fixing `tools/codex_arm_queue.py` is closed in this arm: r57 did not reach that gate, and spawn behavior is not automatically a mechanical-hygiene cure.

Own-vehicle frontier unchanged: **GB1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600]**.
