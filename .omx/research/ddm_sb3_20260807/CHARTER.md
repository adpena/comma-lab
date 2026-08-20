# ddm_sb3 — #983 CI-blind SIGBUS triage: test_compact_renderer_mlx_spine_runner

**Critical-path clause:** task #983 — a SIGBUS/failure in `test_compact_renderer_mlx_spine_runner`
via `mlx_score_aware/adapter.py:5516`, pre-existing on main, surfaced by the mx2 landing. A
SIGBUS in the test suite is a CI-blind hazard: it can kill a pytest worker mid-run and silently
truncate coverage (the vacuity==pass genus — a crashed worker reports fewer tests, not a
failure). Burn-window slot work; scorer-free; NO Metal training.

**Recall-first:**
- Task #983's filing: pre-existing on main, via `src/tac/mlx_score_aware/adapter.py:5516`.
- The #856 xfail pattern (hy1): MLX-env failures gated on `metal::load_device` were closed as
  NAMED xfails with an owner — if this SIGBUS is the same import-time-Metal class, the cure may
  be the same gate, not a code fix. Check hy1's receipts + the #856 xfail implementations first.
- The #851/#942 MLX-gated-test triage lineage (RED modules structurally repaired or xfail'd
  with named owners; no silent skips).
- CAUTION: a SIGBUS reproducer may crash the interpreter — run the reproduction in a SUBPROCESS
  (pytest in its own process, capture rc/signal) so your own session survives; record the exact
  signal + faulting frame (faulthandler / crash log) as the diagnosis receipt.

**Deliverables:**
1. REPRODUCE in isolation (subprocess): exact rc/signal, faulting stack via faulthandler or
   macOS crash report, MLX version + device state at fault. If NOT reproducible at HEAD,
   document the non-repro honestly (environment, attempts) and re-scope the task.
2. DIAGNOSE adapter.py:5516 — what is dereferenced/mapped there; Metal buffer lifetime?
   mmap'd npz? fp16 alignment? Name the mechanism with evidence, not a guess.
3. CURE smallest-correct: real fix if the mechanism is ours; #856-style named xfail/gate if
   it's the import-time-Metal environment class (cite the precedent); either way a regression
   test that runs SAFELY (subprocess-isolated if needed).
4. SWEEP the sibling surface: does the same mechanism exist at other adapter callsites?
   (bug classes have 6-7× spread). List them; fix or file.

**OPTIMAL FORM:** mechanism = the real test + real adapter on this host; scope = this test
module + named siblings. No proxy "it probably works" closures; a non-repro is a finding with
its evidence, never a silent pass.

**Discipline:** serializer + POST-EDIT `--expected-content-sha256` per file; tags
`[no-triality] [p0-ledger-ok]`; review_tracker ×2 per .py; NO Claude/AI attribution or
Co-Authored-By trailer — commits are the operator's alone. NO Metal training, no scorer slot,
no live-run-dir touches. Findings: `.omx/research/ddm_sb3_20260807/SB3_FINDINGS.md`.

## MAIN ADDENDUM (post-spawn adjudication, 2026-08-07 ~16:3x)
Duplication sweep by MAIN: NOT duplicative — no live owner arm, no xfail on the module (the
importorskip at :1917 is the missing-MLX env gate, unrelated), no prior SIGBUS triage memo.
PRIOR CONTEXT to consume: `.omx/research/codex_findings_compact_renderer_mlx_spine_runner_20260601T134313Z_codex.md`
(June 1 codex findings on this exact test file — predates the mx2-surfaced SIGBUS; read before
diagnosing so old known issues aren't re-discovered as new).
