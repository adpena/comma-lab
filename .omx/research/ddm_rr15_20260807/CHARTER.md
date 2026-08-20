# ddm_rr15 — round-15 recursive adversarial review (counter 0/3; convergence round)

**Critical-path clause:** rounds 13+14 were NOT-CLEAN (real findings, fixed inline). The cycle
seals on 3 consecutive clean rounds. Round 15 reviews ONLY the small post-round-14 diff — the
cheapest path to the first clean pass before the ~21:00 resume boundary and the ARM-VEH fire.

**Review corpus (the complete new-code set since round 14 read the tree):**
- `git show 29741ff843` (rr14's own fixes: guard `reason` alias + resume-key PASS test +
  MX1H npz-tensor fail-closed tests + history-step refusal test)
- `git show 7a168e546877995d2c9d4d7e3ef3819daa8c3f38` (sb3's lazy Metal-allocation gate in
  test_compact_renderer_mlx_spine_runner.py + xfail wiring)
- ROUND14_FINDINGS.md + SB3_FINDINGS.md

**Review axes (findings CRITICAL/Medium/Low; fix-or-route inline where small):**
1. The `reason` alias: does it shadow or desync from `reason_code` anywhere (one value updated,
   the other not)? Is the alias emitted at EVERY verdict construction site or just some?
2. The resume-key PASS test: does it fabricate the receipt at the TRUE resume-specific path the
   ticket emits (mem_probe_resume/), host-fingerprint-correct, or a simplified path that would
   pass even if the real wiring broke?
3. sb3's lazy gate: does the allocation probe itself leak Metal state or interfere when a REAL
   Metal training process owns the device (probe-allocation size, cleanup)? Could the gate
   xfail-mask a REAL future SIGBUS (the original #983 class) rather than the no-Metal class —
   is the xfail reason string distinguishable?
4. Anything in these diffs touching the live fire chain that changes behavior beyond tests.
5. Per-round assumption-challenge axis: answer explicitly.

**Verdict:** ROUND15_FINDINGS.md with CLEAN (counter → 1/3) or NOT-CLEAN (fix inline, 0/3).
Do NOT touch the live run dir; NO Metal; NO scorer slot.

**Discipline:** serializer + POST-EDIT `--expected-content-sha256` per file; tags
`[no-triality] [p0-ledger-ok]`; review_tracker ×2 per .py; NO Claude/AI attribution or
Co-Authored-By trailer — commits are the operator's alone.
