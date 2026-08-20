# ddm_rr14 — round-14 recursive adversarial review: the post-fix state (counter 0/3)

**Critical-path clause:** round 13 was NOT-CLEAN (F1 microbatch guard hole found+fixed), so the
cycle continues at 0/3. Three commits landed AFTER round 13 read the corpus and are therefore
UNREVIEWED NEW CODE on the fire chain the ~21:00 resume + ARM-VEH + n120 fires all traverse:
- `d98cf49bfa` — the F1 fix itself (fire-guard microbatch_pairs comparison + positive control)
- `46bc66f241` + shared hunks in `d7f557bb7c` — mx1g ticket generator (receipt-derived
  projection, REQUIRES_FRESH_MEM_PROBE sentinel, argv_*_resume keys, attempt-unique receipts)
- `d7f557bb7c` — mx1h `--mode torch-verdict` (npz→torch mapping reuse, eval-only verdict,
  receipt with proxy↔authority comparison)
Fixes are unreviewed new code; round-finished ≠ clean-pass. A clean round here advances 0/3→1/3.

**Review corpus:** the three commits' full diffs (`git show <sha>`) + the regenerated ticket
`.omx/research/ddm_mx1g_20260807/launch_ticket_mx1g_from_regen2.json` + MX1H receipt
`.omx/research/ddm_mx1h_20260807/receipts/arm_cap_torch_verdict.json` + ROUND13_FINDINGS.md +
MX1G_FINDINGS.md + MX1H_FINDINGS.md + `tools/mx1_fire_guard.py` at HEAD.

**MAIN-observed rows to adjudicate (from the live dry run, 2026-08-07 ~16:10 local):**
- **R1 (verdict field ergonomics):** `evaluate_guard` verdicts carry `reason_code`, NOT
  `reason`. MAIN's dry-run consumer printed `v.get('reason')` → None and initially misread the
  failure. Sweep ALL consumers of the verdict dict (entrypoint in-process guard, any watcher/
  tooling) for `get('reason')` vs `reason_code`; consider adding a `reason` alias or renaming
  consistently — pick smallest-correct, add a schema test.
- **R2 (dry-run PASS half missing):** MAIN validated the resume argv fails closed for the RIGHT
  reason (mem_probe_receipt_missing at the resume-specific path) and that mem_probe_commands
  cover all 8 argv keys. NOT yet validated: that a PASSED resume-path receipt actually
  round-trips (probe → receipt schema → guard PASS on the resume argv). Build a $0 synthetic
  test: fabricate a receipt at the resume path from the real passed receipt's schema (host-
  fingerprint-correct) and assert evaluate_guard PASSES the resume argv — WITHOUT touching
  Metal. This is the one seam the 21:00 boundary still takes on faith.
- **R3 (projection derivation audit):** mx1g derives projected_gib=21 from the receipt. Verify
  the margin rule is stated, provenance-pinned (receipt sha in ticket), and conservative vs the
  receipt's measured peak; verify the n120 keys' fail-closed sentinel actually REFUSES through
  safe_run/governor (not just a string nobody reads).
- **R4 (torch-verdict parity custody):** mx1h reused the mlx-parity mapping. Verify the npz→
  torch mapping FAILS CLOSED on missing/extra tensors (charter demanded it — confirm the code
  does it, not just the findings doc), and that the receipt's proxy-comparison reads the
  history step MATCHING the checkpoint step (off-by-one between meta::step and the last
  history record would silently misalign the gap measurement).
- **R5 (absorption hygiene):** mx1h's commit absorbed mx1g-authored trainer hunks (#911 genus,
  disclosed by mx1g). Verify nothing was LOST or duplicated in the interleave: the trainer at
  HEAD contains exactly one coherent copy of both arms' intended changes.

**Protocol:** findings CRITICAL/Medium/Low → fix-or-route inline where small → per-round
assumption-challenge axis answered explicitly → ROUND14_FINDINGS.md with verdict CLEAN or
NOT-CLEAN and the counter state. Do NOT touch the live run dir; NO Metal; NO scorer slot
needed (R2's test is synthetic-receipt, schema-level).

**Discipline:** serializer + POST-EDIT `--expected-content-sha256` per file; tags
`[no-triality] [p0-ledger-ok]`; review_tracker ×2 per .py; NO Claude/AI attribution or
Co-Authored-By trailer — commits are the operator's alone.
Findings: `.omx/research/ddm_rr14_20260807/ROUND14_FINDINGS.md`.

## MAIN ADDENDUM (post-spawn adjudication, 2026-08-07 ~16:3x — grade findings against this)
Duplication sweep by MAIN AFTER spawn; the harvest will grade rr14's rows against these facts:
- **R1 REFUTED for production**: the entrypoint consumer uses `reason_code` consistently
  (trainer lines ~127/136/191/198/206). The misread was in MAIN's ad-hoc dry-run snippet only.
  Verifying this = legitimate review work; building an alias/rename = NOT needed unless a real
  external consumer reading `reason` is found.
- **R2 NARROWED**: `test_mx1_fire_guard_passes_matching_receipt` exists (tools/tests/
  test_mx1_fire_guard.py:159) but is pinned to base key `argv_n32_arm_cap`. The genuinely
  absent piece = ONE resume-key pass-variant (receipt at the resume-specific path). Small test,
  not a build program.
- **R4 HALF-DONE**: `test_torch_verdict_history_comparison_refuses_missing_checkpoint_step`
  (test_ddm_mx1_memory_probe.py:265) covers the step-alignment half. The npz→torch TENSOR
  mapping fail-closed (missing/extra tensors) shows NO test coverage — that half stands.
- R3 + R5 stand as chartered.
