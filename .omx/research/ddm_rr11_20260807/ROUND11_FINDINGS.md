Verdict: NOT-CLEAN. Clean counter remains 0/3.

Axis: [apparatus / scorer-free]. Score claim: false. Pointer moved: false.
No archive, upstream evaluator run, n600 scorer job, contest-CPU row, or contest-CUDA row was produced.

## Findings

### RR11-F1 - CRITICAL - mem-probe mode rewrites and corrupts the canonical mx1d launch ticket

Verdict scope: APPARATUS: ticket integrity / fire-order safety.

The mx1d ticket makes its own mem-probe command write to the canonical ticket path:
`experiments/ddm_mx1_pr130_semantic_renderer.py:1519-1541` builds a mem-probe command with
`--launch-ticket-path` set to `ticket_path`, and `experiments/ddm_mx1_pr130_semantic_renderer.py:1844-1848`
unconditionally writes `result["launch_ticket"] = launch_ticket(...)` to `_ticket_path_for_args(args)`
for `mem-probe` as well as for ticket generation.

The current dirty mx1d ticket proves the clobber shape:

- `owned_run_root` changed from `.omx/research/ddm_mx1d_20260807/row1_v2_two_arm` to `.omx/research/ddm_mx1d_20260807/row1_v2_two_arm/launch_arm_cap/n32_metal/mem_probe`.
- `mem_probe_receipt_path` changed from `.omx/research/ddm_mx1d_20260807/row1_v2_two_arm/launch_arm_cap/n32_metal/mem_probe/mem_probe_receipt.json` to `.omx/research/ddm_mx1d_20260807/row1_v2_two_arm/launch_arm_cap/n32_metal/mem_probe/launch_arm_cap/n32_metal/mem_probe/mem_probe_receipt.json`.
- `argv_n32_arm_veh` no longer contains `tq1c_seg_cache.pt` and instead contains `gt_seg_cache.pt`, collapsing the ARM-VEH public-wire arm into another GT-token arm.

RR11 reproduction: `.omx/research/ddm_rr11_20260807/runtime_checks/ticket_clobber_probe/ticket_clobber_probe_summary.json`.
The controlled copy changed under a stubbed mem-probe path with `ticket_changed=true`; after rewrite, the nested
`mem_probe_receipt_path` appeared and ARM-VEH lost the tq1c input-cache discriminator.

Impact: the fire ticket is no longer an immutable order. A probe can mutate the order it is supposed to
support, including the arm identity and downstream receipt paths. That invalidates any later "passed"
fire guard that reads the mutated ticket as if it were the original two-arm intent.

Required cure before any mx1d/mx1e Metal fire:

1. Make ticket generation an explicit ticket-only mode or explicit `--emit-launch-ticket` action.
2. For worker modes (`mem-probe`, `mlx-train`, parity), refuse writes to an existing canonical ticket path unless a dedicated overwrite flag is present and the mode is ticket generation.
3. Remove canonical `--launch-ticket-path` from mem-probe argv or add a `--no-ticket-write` worker flag that is checked before `write_json(_ticket_path_for_args(args), ...)`.
4. Add a regression test that runs `mem-probe` with a copied ticket and asserts the ticket bytes do not change and `argv_n32_arm_veh` still points at `tq1c_seg_cache.pt`.

Follow-on disposition: P0-FIRE-BLOCKER for mx1d/mx1e governed Metal fires until ticket immutability is structural.

### CARRIED-HIGH - RR10-F1 review interlock remains queued, not structural

Verdict scope: APPARATUS: review/fire interlock.

RR10-F1 required a `review_interlock_receipt` before a Metal fire can outrun an active recursive review
charter or unresolved HIGH finding on the same live fire surface. RR11 searched the live source and research
state for `review_interlock`, `RR10-F1`, `active review`, and `live review`.

Found:

- `.omx/research/ddm_ah1_20260807/FOLLOWON_LEDGER.jsonl` queues `ah1.rr10.review_interlock`.
- `.omx/research/ddm_ah1_20260807/AUDIT_TABLE.md` says RR10-F1 remains queued and had no AH1 code touch.
- `.omx/research/ddm_mx1d_20260807/CHARTER_ADDENDUM.md` describes the intended field, but that file is untracked and not a structural gate.
- No live source implementation was found in `tools`, `src`, `experiments`, or `docs`.

Impact: even after RR11-F1 is fixed, the fire protocol can still race a live review charter unless the operator
manually notices. That is the same class RR10 found.

Follow-on disposition: carry `ah1.rr10.review_interlock` as unresolved HIGH and block orchestrated Metal fires
that name the same live surface until the interlock receipt exists and has a positive-control test.

## Bounded Clean Checks

### AH1 hardening

AH1 commit checked: `50aac0ee16f47d92bb597e1ab8658355deb990b4`.

Observed file set: exactly AH1 research artifacts plus five #254 trainer guards, the launch/safe-run/watch
tools, and their focused tests. No unrelated file absorption was found from `git show --name-status`.
The commit has no co-author or AI trailer. `git show --check` reports four markdown "new blank line at EOF"
warnings in AH1 docs; this is hygiene, not a behavioral blocker.

Focused verification:

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tools/tests/test_codex_arm_watch.py \
  src/tac/tests/test_no_silent_failure_launch_hardening.py \
  src/tac/tests/test_admission_coverage_gate.py \
  experiments/tests/test_ddm_mx1_memory_probe.py \
  tools/tests/test_mx1_fire_guard.py -q
```

Result: `51 passed in 9.08s`.

Runtime spot checks:

- `tools/safe_run.py --status-receipt` timeout check wrote `.omx/research/ddm_rr11_20260807/runtime_checks/safe_run_timeout_status_receipt.json` with schema `safe_run_status_receipt.v1`, status `timeout`, `peak_rss_observed=true`, populated `last_sample_ts`, and kill action `SIGTERM_then_SIGKILL_process_group`.
- `tools/launch_detached_process.py --verify-alive-secs 0.2` refused a child exiting with rc 7; the committed manifest records the argv and output directory, while the gitignored transient log/pid files are intentionally not committed.
- The #254 scanner now reports 140 residual unguarded heavy trainers after AH1's five insertions.

Bounded verdict: AH1's named H1/H2/H3/H4 mechanics are locally supported by tests and runtime spot checks.
This does not make the mx1d/mx1e fire chain clean because RR11-F1 and CARRIED-HIGH remain.

### RR10-F2 current-source status

RR10-F2 said a stale or forged passed fire-guard verdict could bypass the intended guard chain.
The current live worktree has additional in-process guard hardening in
`experiments/ddm_mx1_pr130_semantic_renderer.py:1703-1773`: GPU `mlx-train` now requires
`--fire-guard-verdict`, `--launch-ticket-path`, and `--fire-argv-key`, reruns
`tools.mx1_fire_guard.evaluate_guard()`, and compares the verdict path, ticket path, receipt path, and argv key.

Current-source runtime check:
`.omx/research/ddm_rr11_20260807/runtime_checks/forged_passed_current_source/summary.json`.
A minimal fake `passed` verdict with missing ticket exited 9 and did not write the probe output.

Bounded verdict: RR10-F2 appears fail-closed in the current dirty worktree. It is not counted as landed clean
until the owning arm serializes that source/test state. RR11 retains the older forged-verdict runtime files only
as superseded evidence; the refreshed current-source check is the controlling boundary.

### mx1d probe-refusal chain

The mx1d mem-probe receipt exists at
`.omx/research/ddm_mx1d_20260807/row1_v2_two_arm/launch_arm_cap/n32_metal/mem_probe/mem_probe_receipt.json`
and re-derives a refusal:

- schema `ddm_mx1_load_phase_peak_receipt.v1`
- status `failed`
- mode `mem-probe`
- blocker `MemoryLimitConfigurationError`
- sample_count `6`
- `metal_fire_clearance=false`

This is a valid refusal artifact for the attempted hard-memory-limit path. It is not a passed Metal
load-stage receipt and it is not a safe_run status receipt. A search of the mx1d research tree and the SSD
mx1d path found no safe_run status receipt for that probe run.

### MAIN soft-cap / mx1e status

The mx1e prompt stated a MAIN observation that MLX `set_memory_limit` behaved as a soft guideline.
RR11 did not find a landed mx1e research receipt, `.done` file, or mx1e commit in the checked scopes.
Observed only the charter-run mx1e log/stdin pair, with no mx1e done receipt.

Bounded verdict: do not promote the mx1e soft-cap observation as a landed receipt in RR11. Treat it as live /
not yet serialized unless the owning arm lands a receipt after this audit.

## RECALL EVIDENCE

| Scope | Evidence checked | Result | Effect on plan |
|---|---|---|---|
| Governing files | `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, rr11 prompt, common contract | Current own-vehicle pointer is `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer borrowed/unmoved. | Kept rr11 scorer-free and pointer-honest. |
| Prior rounds | `ROUND2_FINDINGS.md` through `ROUND10_FINDINGS.md` | RR9 mem-probe fire protocol, RR10-F1 review race, RR10-F2 stale verdict, RR10-F3 safe_run receipt boundary were the relevant predecessor chain. | Audited AH1 against the predecessor chain instead of re-opening unrelated rows. |
| AH1 artifacts | `.omx/research/ddm_ah1_20260807/AUDIT_TABLE.md`, `RECEIPT.md`, `FOLLOWON_LEDGER.jsonl`, `NEXT_IF_RESUMED.md` | AH1 landed H1/H2/H3/H4 hardening; RR10-F1 stayed queued. | Split AH1 clean checks from unresolved fire-order blockers. |
| mx1d artifacts | `.omx/research/ddm_mx1d_20260807/RECEIPT.md`, launch ticket, fire verdict, mem-probe receipt, dirty current ticket diff | Hard-cap refusal exists, but the ticket was mutated by the mem-probe path. | Promoted ticket immutability to RR11-F1. |
| Source surfaces | `experiments/ddm_mx1_pr130_semantic_renderer.py`, `tools/mx1_fire_guard.py`, `tools/safe_run.py`, `tools/launch_detached_process.py`, `tools/codex_arm_watch.py` | Found unconditional ticket rewrite in worker modes; current fire-guard entrypoint re-evaluates in-process. | Kept only the ticket corruption as a new active blocker; downgraded stale-verdict bypass to current-source boundary. |
| Canonical equations | `tools/list_canonical_equations.py --json` filtered for `rr9`, `mem_probe`, `fire_protocol`, `review_interlock`, `stage_rc`, `safe_run`, `mx1` | Relevant recall anchors: `ddm_rr9_mem_probe_fire_protocol_v1` and `ddm_rr8_stage_rc_success_contract_v1`; no review-interlock equation found. | Reused the existing mem-probe-before-fire contract and did not invent a scorer claim. |
| Targeted searches | `rg review_interlock/RR10-F1/live review`; `find`/`git log` for mx1e done/receipt/commit; memory registry search for rr11/ah1/mx1d/mx1e/safe_run terms | Review interlock exists only as queued/untracked prose; mx1e has no landed done/receipt/commit in checked scope; no direct memory hit. | Kept RR10-F1 carried and mx1e non-promoted. |
| Runtime probes | rr11 `runtime_checks/` JSON/manifest artifacts | safe_run receipt and dead-child liveness hardening worked; ticket clobber reproduced; fake passed verdict fails closed on current source. | Report has machine-readable local evidence without absolute temp paths. |

## Verification Commands And Results

1. Focused tests: `51 passed in 9.08s`.
2. Safe-run timeout receipt: status `timeout`, exit behavior observed through receipt, peak sampling field present.
3. Detached child liveness check: early child rc 7 returned rc 7; the committed manifest records launch context.
4. Ticket-clobber reproduction: `ticket_changed=true`; ARM-VEH tq1c cache discriminator lost.
5. Current-source fake passed verdict check: exit 9; no probe output written.
6. mx1d mem-probe receipt: status `failed`, blocker `MemoryLimitConfigurationError`, `metal_fire_clearance=false`.
7. AH1 commit file-set check: no unrelated absorption found; markdown EOF whitespace warnings only.

## Boundaries

- No Python files were edited by rr11.
- No staged index, protected file, stash, upstream file, remote, GPU job, n600 scorer slot, archive, or evaluator run was touched.
- The stale forged-verdict probe artifacts under `runtime_checks/` are superseded by the refreshed current-source check and are not the basis for an active RR11 finding.
- The mx1d launch ticket and mx1d mem-probe directory were dirty before rr11 and were not reverted.
- Gitignored transient `run.log` and `run.pid` files from the dead-child spot check are left uncommitted.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed and unmoved.
