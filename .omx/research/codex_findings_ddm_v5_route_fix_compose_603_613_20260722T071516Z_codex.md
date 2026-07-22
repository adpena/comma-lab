---
task: 603
feeds_task: 613
review_round: 1
reviewer: codex
research_only: true
main_landing_review_required: true
---

# Codex round-one findings: DDM v5 route-fix composition

## Disposition

`PASS_AFTER_FIX`, scoped to source correctness, deterministic local receiver closure, and the
n64/n256 advisory receipts.  Promotion, d_seg, and contest score remain unreviewed and unauthorized.

## Finding R1-1: inherited value excluded from the maximization set

- Severity: high for the delegated premise.
- Original behavior: inherited S4 paint was measured but excluded from the five C1-derived
  candidates passed to `select_role_paint_values`.
- Consequence: Lane selected membership `0.466113811839` even though inherited paint measured
  `0.917071775808`; Undrivable similarly discarded a slightly better inherited value.
- Fix: include the self-detected inherited palette row as candidate six for every role, then apply
  the same own-class maximum and deterministic tie break.  The unit fixture now gives one role a
  candidate-six maximum and verifies all roles return their exact maximum.
- Verification: Ruff green; focused suite `48 passed`; fresh receipts bind source commit
  `b55caa1b25` and show Lane membership `0.582142857143` n64 / `0.633187993979` n256.

## Fresh-eyes checks

- No hardcoded role-to-class index table remains in the structured route path.
- Non-prefix target RGB, cached cells, Pose6 codes, and S4 event/component records share the same
  global-to-local pair mapping.
- Both composed archives parse/recompile byte-identically and report every byte in one ZIP home.
- Movable records are nonempty in both windows (global pair 456, 35 sites).
- Scorer weights are absent from archives; SSD source/cache paths were read-only.
- Every negative is formulation-scoped in the memo/register; the pointer is unchanged.

## Remaining review debt for MAIN

- Rerun the focused suite and bounded n64 measurement from the branch commits.
- Verify archive and receipt hashes, especially the candidate-six selector path.
- Treat Movable's low membership and n256's rate miss as formulation-scoped; do not promote them to
  family negatives.
- No canonical equation registry mutation is authorized until MAIN reviews the
  `FORMALIZATION_PENDING` draft.

