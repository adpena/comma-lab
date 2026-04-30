# Shannon-Floor Paper Claim Hygiene - 2026-04-30 Codex

Scope: reviewed the claim-control and writeup surfaces requested for stale
score claims, CPU/local evidence leakage, and minimal safe patch candidates.
No full-paper rewrite was performed.

Reviewed:

- `AGENTS.md`
- `.omx/research/shannon_floor_claim_matrix_20260430_codex.md`
- `docs/paper/ara/PAPER.md`
- `docs/paper/ara/logic/claims.md`
- `docs/paper/ara/logic/experiments.md`
- `docs/paper/ara/logic/problem.md`
- `docs/paper/ara/logic/related_work.md`
- `docs/paper/ara/src/index.md`
- `docs/paper/ara/evidence/index.md`
- `docs/paper/ara/evidence/results_index.json`
- `docs/paper/ara/trace/dead_ends_to_revisit.md`
- `docs/paper/ara/trace/compilation_log.md`
- `reports/latest.md`
- `reports/writeup_working.md`

## Governing Rule Applied

Exact CUDA auth eval on exact archive bytes through the canonical
`archive.zip -> inflate.sh -> upstream/evaluate.py` path is the only score
truth. CPU, MPS, Modal-CPU, local proxy, byte-only, smoke, stale-log, and
uncustodied Modal rows cannot promote, rank, kill, retire a family, validate a
stack, or anchor paper claims.

## Highest-Risk Patch Landed

Patched `reports/latest.md`, because it is the highest-risk current-status
surface and was using Modal/CPU diagnostics to draw retirement/kill
conclusions.

Changes:

- Lane GP v3 is now explicitly tagged as `[Modal-T4-CPU diagnostic; invalid
  for score claims]`.
- Lane GP v3 no longer says the off-manifold hypothesis is disproved or that
  the polynomial path is dead.
- Lane UNIWARD v8 is now explicitly tagged as `[Modal-T4-CPU diagnostic;
  invalid for score claims]`.
- Lane UNIWARD v8 no longer says standalone is killed.
- The caveat now requires lane-local CUDA `contest_auth_eval.json`, archive
  SHA/bytes, component recomputation, and custody before Modal/local rows can
  be treated as more than advisory.

This is intentionally narrower than a full paper cleanup.

## Residual Risks Not Patched In This Pass

1. `reports/writeup_working.md` still lists Lane GP v3 as a "nearby
   catastrophic" `89.67 [Modal-T4-CPU]` result. It does not contain the same
   broad kill sentence as `reports/latest.md`, but it should be quarantined
   before any public writeup extraction.

2. `docs/paper/ara/evidence/index.md` lists Lane GP v3 and UNIWARD v8 as
   `contest_auth_eval.json` references backed by memory-only pointers. Until
   lane-local JSON, archive bytes/SHA, components, and custody are present,
   those rows should be renamed or marked diagnostic/non-promotable in the
   evidence layer.

3. `docs/paper/ara/evidence/results_index.json` contains multiple
   `[Modal-T4-CUDA]` rows with null archive bytes and null components. Those
   are useful harvest metadata, but not sufficient paper evidence unless
   extended with custody fields or demoted to diagnostic/advisory status.

4. `docs/paper/ara/logic/claims.md` C8 still uses the Modal T4 reproduction as
   support for the Lane G v3 historical mechanism. This is acceptable only as
   historical/mechanism context; PFP16 A++ remains the current frontier anchor.

5. `docs/paper/ara/src/index.md` has a historical Modal T4 reproduction
   command for Lane G v3 predecessor context. It should not be used as a
   current-frontier reproduction path; the PFP16 `contest_auth_eval.py
   --device cuda` command is the correct public command.

6. `reports/latest.md` labels the leaderboard section "Live leaderboard" while
   the data is a 2026-04-29 snapshot. The row wording says it is against that
   snapshot, but the section heading should be downgraded before publication
   unless the leaderboard is refreshed.

7. Several Ara trace/problem files use broad "dead", "kill criterion", or
   "dead-end" language. Some of that is taxonomy rather than score evidence,
   but public-facing prose should prefer "suspended", "retest criterion", or
   "measured implementation retired" unless exact evidence or proof supports a
   stronger claim.

## Minimal Next Patches

1. Mirror the `reports/latest.md` CPU/diagnostic quarantine wording into
   `reports/writeup_working.md` line 15.
2. Demote or rename the Lane GP v3 and UNIWARD v8 memory-only entries in
   `docs/paper/ara/evidence/index.md`.
3. Add evidence-grade/custody fields to `docs/paper/ara/evidence/results_index.json`
   or exclude Modal rows from paper evidence until complete custody exists.

## Changed Files

- `reports/latest.md`
- `.omx/research/shannon_floor_paper_claim_hygiene_20260430_codex.md`
