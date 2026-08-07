# ddm_rr10 Round 10 Findings

Verdict: NOT-CLEAN. Counter remains 0/3.

Axis: `[apparatus / scorer-free]`. Score claim: false. Pointer moved: false.

Audit cutoff: 2026-08-07T12:40:00Z. mx1d was still live at cutoff: no
`.omx/tmp/codex_runs/ddm_mx1d.done`, no `.omx/research/ddm_mx1d_20260807/RECEIPT.md`,
and no passed Metal `mem_probe_receipt.json` existed in the inspected tree.

## Findings

### RR10-F1 - HIGH - MAIN outran a live review charter for the same Metal fire

The fire was not just missing a receipt. It was fired while rr9 was charter-bound to audit
that exact live fire surface. rr9 then found the missing-receipt violation while the host
was still up. The smallest durable fix is a review interlock, not another advisory sentence.

Evidence:

- `.omx/tmp/codex_runs/rr9_prompt.md:16-20` explicitly assigns rr9 to audit the live
  ARM-CAP safe_run fire, its ticket protocol, safe_run receipts, and the free-memory
  0-1 GiB load behavior.
- `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch_arm_cap_governed/launch_manifest.json:2-47`
  shows the fire command was `safe_run.py ... --mode mlx-train --device gpu`, not the
  ticket-required `mem-probe`.
- `.omx/research/ddm_rr9_20260807/ROUND9_FINDINGS.md:9-16` records RR9-F1 HIGH: ARM-CAP
  outran its required Metal mem-probe receipt.
- `.omx/research/ddm_rr9_20260807/ROUND9_FINDINGS.md:48-57` gives the quarantine cure:
  refuse fire until a passed Metal receipt exists or enough post-hoc measured peak telemetry
  is attached.
- `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/concurrent_metal_fires_without_composed_preflight_oomed_the_machine_20260806.md:58-63`
  records the review timing explicitly: rr9 filed HIGH while the machine was still up; the
  review protocol worked, but MAIN outran it.

Smallest binding form:

1. Add a `review_interlock_receipt` requirement to any ticketed Metal fire whose surface is
   named by a live recursive review charter, live review arm, or unresolved HIGH finding.
2. A fire may proceed only when the interlock receipt says either no active review scope
   matches, or every matching review has landed CLEAN, or every matching finding is already
   FIRED, FOLDED, or QUEUED-WITH-A-FIRE-ORDER that explicitly authorizes this fire.
3. If a matching review lands HIGH while a target process is live, the orchestrator must
   quarantine that process and all follow-on fires until the finding disposition is recorded.

Follow-on disposition: QUEUED-WITH-A-FIRE-ORDER as `rr10_f1_review_interlock_before_metal_fire`.

### RR10-F2 - HIGH - mx1d guard verdict is bypassable by a stale or forged passed JSON

mx1d lands the right structural pieces in the current worktree, but the final entrypoint
check is too weak. `tools/mx1_fire_guard.py` validates the ticket, receipt, host, samples,
memory limits, and fire config. The `mlx-train` entrypoint only checks that the supplied
JSON has the expected schema and `status == "passed"`. A stale passed verdict from another
argv, or a hand-written minimal passed JSON, would satisfy the last line of defense without
re-validating the receipt-required gate.

Evidence:

- `tools/mx1_fire_guard.py:248-334` performs the substantive guard: ticket key lookup,
  `mem_probe_receipt_required`, receipt existence/parse/schema/status, host validation,
  sample validation, hard-limit validation, and config match.
- `tools/mx1_fire_guard.py:373-388` writes a failed verdict atomically even when the guard
  refuses. The observed mx1d precheck did this correctly at
  `.omx/research/ddm_mx1d_20260807/row1_v2_two_arm/launch_arm_cap/n32_metal/fire_guard_verdict.json:1-15`,
  with `status: failed` and `reason_code: mem_probe_receipt_missing`.
- `experiments/ddm_mx1_pr130_semantic_renderer.py:1458-1481` only reads the verdict file and
  refuses when schema/status are not the expected passed pair. It does not verify
  `reason_code == "fire_guard_passed"`, `argv_key`, `ticket_path`, `receipt_path`,
  `fire_config`, `receipt_config`, host, or that the current command matches the verdict.
- `experiments/ddm_mx1_pr130_semantic_renderer.py:1520-1521` calls this weak check before MLX
  setup for GPU `mlx-train`.
- `experiments/tests/test_ddm_mx1_memory_probe.py:257-277` covers refusal without any guard
  verdict, but the inspected focused tests do not cover a forged/stale passed verdict.

Smallest cure:

1. Make the entrypoint consume enough context to re-run the guard, not merely trust a file.
   Minimal shape: require `--launch-ticket-path` and `--fire-argv-key` for GPU `mlx-train`,
   call `tools.mx1_fire_guard.evaluate_guard(...)` in-process, and require a passed verdict
   matching the supplied `--fire-guard-verdict` path before any MLX setup.
2. If re-running is undesirable, validate all guard fields against the current argv and parsed
   receipt: schema, status, reason_code, ticket path, argv key, receipt path, host, memory
   limit status, and fire/receipt config match. Add a focused test where a minimal fake
   `{"schema": "...", "status": "passed"}` is refused.
3. Treat existing mx1d ticket artifacts as not fire-clearance until this bypass is closed and
   the Metal mem-probe receipt passes.

Follow-on disposition: QUEUED-WITH-A-FIRE-ORDER as `rr10_f2_mx1d_revalidate_guard_at_entrypoint`.

### RR10-F3 - MEDIUM - Second-incident memory overclaims the receipt boundary

The incident memory is directionally correct, but its "all verified from receipts" framing is
too broad. The raw evidence proves a governed `mlx-train` launch, no required mem-probe receipt,
a short run log, and a dead-process daemon reconciliation. It does not prove a safe_run kill
attempt, a safe_run peak receipt, or a machine-readable free-memory trace. That distinction must
be preserved because the cure is receipt-gated machinery, not recollection.

Evidence:

- `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/concurrent_metal_fires_without_composed_preflight_oomed_the_machine_20260806.md:54-77`
  correctly records the second incident chain, but line 57 says the three new legs are all
  receipt-verified.
- `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch_arm_cap_governed/launch_manifest.json:2-47`
  proves the launched command and `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch_arm_cap_governed/launch_manifest.json:52-58`
  records the done-receipt path and pid.
- `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch_arm_cap_governed/run.log:1-4`
  contains only warnings, admission OK, and the grouped-backward banner. It contains no result,
  no mem-probe receipt, no peak receipt, and no safe_run kill line.
- `.omx/state/durable_daemons.json:14086-14094` records the safe_run daemon row as `status:
  stopped`, `stopped_reason: reconcile_dead_process`, with an empty log field. That is
  dead-process reconciliation, not an in-process kill receipt.
- `.omx/research/ddm_mx1b_20260806/MEM_DIAGNOSIS.md:15-17` and
  `.omx/research/ddm_mx1b_20260806/MEM_DIAGNOSIS.md:32-33` keep the torch/scorer load and
  MLX allocator boundary honest: upstream scorer eager load is confirmed and necessary, while
  MLX allocator/cache pressure remains MAIN-only until a passed Metal mem-probe receipt exists.
- `last reboot` / `who -b` bounded the reboot to August 7, 2026 07:20 local. Direct
  `sysctl kern.boottime` and `ps` checks were denied in this sandbox, so process liveness and
  kernel boottime were not independently machine-read here.

Smallest cure:

Amend the incident account or the next incident ledger with terminal-evidence fields:
`launch_manifest_sha`, `run_log_line_count`, required receipt path and absence/presence,
manifest done/result absence, durable daemon terminal status, boottime source, whether a
safe_run kill/peak receipt exists, CPU-side measured load contribution, and explicit
`Metal allocator peak: unmeasured unless mem_probe_receipt.status=passed`.

Follow-on disposition: QUEUED as `rr10_f3_incident_terminal_evidence_amendment`.

## Clean Audits

### mx1d hard cap and failed-probe receipt path

No additional finding beyond RR10-F2. The current mx1d worktree changed the default memory
budget to 35 percent of available memory, caps mem-probe default at min(24 GiB, default), and
uses a signature-aware `relaxed=False` call for GPU hard-cap setup:

- `experiments/ddm_mx1_pr130_semantic_renderer.py:627-656` derives the 35 percent default and
  the 24 GiB mem-probe cap.
- Local installed MLX introspection at cutoff reported `set_memory_limit` and `set_cache_limit`
  signatures as `(*args, **kwargs)`, so the current helper takes the
  `value_relaxed_false_uninspectable` hard-cap path rather than the older soft default.
- `experiments/ddm_mx1_pr130_semantic_renderer.py:666-744` tries `relaxed=False` first and
  records failure/refusal when the hard form is unavailable.
- `experiments/ddm_mx1_pr130_semantic_renderer.py:770-828` refuses GPU mode if the hard memory
  cap is not satisfied and `--allow-soft-mem-limit` is not set.
- `experiments/ddm_mx1_pr130_semantic_renderer.py:1211-1220` writes mem-probe receipts
  atomically on pass, fail, or block.
- `.omx/research/ddm_mx1d_20260807/launch_ticket_v3_fire_guarded.json:265-388` publishes the
  intended precheck -> probe -> gate -> fire order, and
  `.omx/research/ddm_mx1d_20260807/launch_ticket_v3_fire_guarded.json:428-439` declares the
  required receipt path plus 35 percent policy.
- Focused verification at cutoff: `.venv/bin/python -m pytest experiments/tests/test_ddm_mx1_memory_probe.py tools/tests/test_mx1_fire_guard.py -q` returned `13 passed`, then `14 passed` after mx1d's additional test landed.

Boundary: this is a code/ticket audit only. The observed guard verdict is failed because the
required Metal receipt is missing; therefore mx1d does not clear any Metal fire at cutoff.

### hb1 second collateral

No finding at cutoff; the FORM_DEVIATED label is owed at harvest and is already present in the
live hot-state wording. The caveat remains binding on any row claim:

- `.omx/research/ddm_hb1_20260806/RESUME_CAVEAT.md:3-12` says `.latest.pt` is weights-only,
  a post-epoch-30 crash continuation is `FORM_DEVIATED_RESUME`, and the driver must not be
  edited mid-run.
- `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/driver.log:38-42` shows epoch 48 before the
  second reboot and then `resume from latest` with epoch counter reset to 0.
- `.omx/state/main_hot_state.md:23-27` now states the resumed hb1 process is `FORM_DEVIATED`
  and describes the 60-more-CPU-epochs continuation.
- `.omx/research/ddm_hb1_20260806/RESUME_CAVEAT.md:14-17` still bars row adoption unless all
  stage rc/report parse gates pass.

Round answer: hb1 may continue as a live collateral recovery, but any byte-race row from this
post-reboot continuation must carry `FORM_DEVIATED_RESUME`, prior_epoch=48, post_resume_epoch
count, total-steps-trained, and the stage2/stage3/stage4 exact-byte rc/report evidence before
it is tabled or adopted.

## Recall Evidence

| source searched | query / command | found beyond charter seeds | changed plan |
|---|---|---|---|
| Memory registry | `rg "ddm_rr10|rr10|mx1d|review interlock|macOS-CPU advisory|lane" /Users/adpena/.codex/memories/MEMORY.md` plus focused line read | Recalled artifact-producing, lane/axis separation, and score custody boundaries; no direct rr10 precedent. | Kept this scorer-free, did not claim a score, and separated `[macOS-CPU advisory]` from contest axes. |
| Governing files | Read `PROGRAM.md`, `.omx/tmp/codex_runs/rr10_prompt.md`, `_common_contract.md`, and relevant `CLAUDE.md` / `AGENTS.md` sections | Protected-file and serializer constraints; no scorer/eval/archive dispatch. | Wrote one findings artifact only. |
| Live state | Read `.omx/state/main_hot_state.md` before and after post-reboot updates | Hot state now labels hb1 as FORM_DEVIATED and keeps METAL HOLD until mx1d + merge review + hardened capped probe. | Downgraded hb1 from finding to harvest gate. |
| rr9 evidence | Read rr9 prompt, rr9 findings, final message, and mx1c ticket/NEXT files | rr9 was actively assigned to audit the live fire and found HIGH before the host died. | Elevated the missing interlock to RR10-F1. |
| Incident receipts | Read launch manifest, run.log, daemon row, receipt/result existence checks, reboot evidence | Found no result/done/mem-probe receipt and no safe_run kill/peak receipt; daemon status was reconcile-dead-process. | Added RR10-F3 for overbroad receipt wording. |
| mx1d live artifacts | Read current source, tests, `tools/mx1_fire_guard.py`, ticket artifacts, and failed guard verdict | Hard-cap/failed-receipt surfaces are mostly clean, but entrypoint trusts schema/status only. | Added RR10-F2 and did not certify mx1d as fire-clearance. |
| hb1 receipts | Read caveat, driver log, latest checkpoint facts, launch manifest, done absence | Post-ep48 resume is real and row labeling is owed, but no row exists yet. | Recorded clean harvest-gate answer instead of a finding. |
| Canonical equations / corpus | Queried canonical-equation registry and targeted `rg` over `.omx/research`, state, docs, tools, and experiments | No score-relevant equation or landed mx1d receipt superseded the charter; several broad searches were noisy but confirmed no exact-score row. | Kept findings apparatus-scoped and cutoff-scoped. |

## Boundaries

- Did not run a scorer, archive build, `upstream/evaluate.py`, GPU/Metal training, or remote dispatch.
- Did not edit protected files from the common contract.
- Did not edit or commit mx1d source/test/tool changes; they were live unrelated work observed for audit only.
- Direct `sysctl kern.boottime` and `ps` were denied in this sandbox, so those checks are bounded absence, not global nonexistence.
- Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.

## Round-2 Answer

NOT-CLEAN. Clean counter remains 0/3. The immediate fire order is:

1. Close RR10-F2 before any mx1d-governed Metal fire: the entrypoint must re-run or fully validate the guard, not just trust a passed JSON.
2. Close RR10-F1 before the next orchestrated fire: no launcher may outrun an active review charter over the same live surface.
3. Record RR10-F3's terminal-evidence amendment so future postmortems distinguish receipt facts from live observation.
4. Preserve hb1 as `FORM_DEVIATED_RESUME` at harvest; do not promote the row without the caveat fields and exact-byte stage gates.

Own-vehicle frontier: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
