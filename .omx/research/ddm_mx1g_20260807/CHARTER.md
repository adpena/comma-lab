# ddm_mx1g — ticket generator derives projection FROM receipt + receipt-name uniqueness + resume argv

**Critical-path clause:** the ARM-VEH fire and the n120 dispatch both regenerate/reuse the v4
launch ticket. Today's ARM-CAP fire needed a HAND-CORRECTED outer wrapper because the ticket
generator LATCHES a pre-mx1f projection constant (66.268951 GiB) that the governor rightly
refuses. Every future fire inherits that refusal until this lands. This is the
cross-regime-constant-transfer genus (memory
`cross-regime-constant-transfer-genus-finishing-stage`): a constant derived under the
full-batch allocator regime survived into the chunked regime.

**Recall-first:**
- `experiments/ddm_mx1_pr130_semantic_renderer.py::launch_ticket` (~line 1445) + the v4 schema
  `ddm_mx1_row1_launch_ticket.v4_software_cap_fire_guarded` (~1574) + `_ticket_path_for_args`
  (~1686). RR11-F1 made ticket-writing probe-mode-only — PRESERVE that (regression test
  `experiments/tests/test_ddm_rr12_mx1_ticket_immutability.py` must stay green).
- The mem-probe receipt schema: written by `--mode mem-probe` runs beside the run dir
  (`mem_probe/` under the argv's run dir); `tools/mx1_fire_guard.py::_receipt_config` +
  `_validate_config_match` show which fields exist. The guard freshness window is ≤6h.
- safe_run receipts: `tools/safe_run.py` (rr12 added `--child-pidfile` + SIGTERM-safe writes).
- The live fire5 worked by firing the INNER trainer argv (post-last-`--`) under a corrected
  wrapper `--projected-gib 15 --rss-mb 45000` — the guard binds the unwrapped inner argv
  (`_unwrap_safe_run`), which is why this was legal. mx1g makes the hand-correction unnecessary.

## Deliverable 1 — projection derived from the receipt (the P0 of this arm)
In `launch_ticket()`: for each emitted argv, when a PASSED mem-probe receipt exists at that
argv's receipt path, derive the nested safe_run wrapper's `--projected-gib` and `--rss-mb` FROM
the receipt (measured peak + an explicitly derived margin — state the margin rule in a comment
and in the ticket json, e.g. peak×1.5 ceil, floor at a small constant; provenance: cite the
receipt path + sha in the ticket). When NO receipt exists, emit the wrapper with an explicit
`"REQUIRES_FRESH_MEM_PROBE"` sentinel value that safe_run/governor would refuse — fail-closed,
never the latched constant. DELETE the 66.268951 latch entirely.

## Deliverable 2 — receipt/done name uniqueness
fire5 reused fire4's done-receipt name, so `.done` showed a stale rc=9 while the process was
alive. Make the done/status receipt path attempt-unique (timestamp or pid suffix), OR refuse to
start when an existing receipt at the path belongs to a different (dead) pid — pick the smaller
correct fix, and cover the "stale receipt visible while live" case with a test.

## Deliverable 3 — resume argv as a first-class ticket key
The 12h timeout lands ~step 5,400 of 6,000; a resume leg is REQUIRED. The guard's
`_validate_config_match` uses a named key set that does NOT include `--resume-from`, so
appending it is legal — but implicit legality is fragile. Emit explicit
`argv_n32_arm_cap_resume` (etc.) keys: identical flags + `--resume-from <run_dir>/mlx.latest.npz`,
same guard binding. Document in the ticket json that resume requires a FRESH mem-probe receipt
(freshness ≤6h) and that the probe must run through the SAME chunked microbatch config.

## Seeded review row (hand to rr13, do not fix here unless trivial)
The guard's comparison key set omits `microbatch_pairs` — the flag that DETERMINES the
load-phase footprint post-mx1f. Note it in your findings; rr13 owns the adjudication.

## OPTIMAL FORM
Mechanism = real receipt parsing + real guard semantics (import the guard's own reader if
exposed; do not duplicate its parsing). Scope: this trainer's ticket only. No new frameworks.

## Discipline
Serializer + POST-EDIT shas; tags `[no-triality] [p0-ledger-ok]`; review_tracker ×2 per .py;
NO Claude/AI attribution or Co-Authored-By trailer — commits are the operator's alone.
Tests for all three deliverables; RR11-F1 + rr12 regressions stay green.
Findings: `.omx/research/ddm_mx1g_20260807/MX1G_FINDINGS.md`.
