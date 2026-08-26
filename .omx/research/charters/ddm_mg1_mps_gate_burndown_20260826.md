# ddm_mg1_mps_gate_burndown — adjudicate the 21 residual no-mps-decision gate sites and flip test_repo_no_proxy_decision_violations_clean green

## MANDATE

After the 5fcf9c1c7f scope fix (`.omx/tmp/` + generated graph_memory exemptions + the
D42 within-instrument waiver), `tac.preflight.check_no_proxy_metric_drives_decision`
holds exactly 21 genuine violations across 15 files (#1303's measured list — re-run the
gate yourself for the live list, do not trust the task text). Each site needs an HONEST
per-site adjudication against the gate's contract, then the repo-clean test goes green
and the r52+ full-preflight chain unblocks at this gate.

1. For each site, READ the surrounding context and classify: (a) RULE-RESTATEMENT
   (negated verb — reword to hit the gate's `_NEGATED_VERB` form, or leave if already
   negated and the gate mis-parses: then the FIX IS THE GATE, not the text); (b)
   HISTORICAL/POST-MORTEM record — add the gate's recognized tags (`[WITHDRAWN]`,
   `POST-MORTEM`, `per CLAUDE.md`, …) where TRUE; (c) advisory measurement with a
   decision verb where a [contest-CUDA] artifact EXISTS — cite it within ±10 lines;
   (d) genuine advisory-derived decision language with NO CUDA backing — reword the
   verb honestly (downgrade the claim, never invent a CUDA citation); (e) irreducible
   same-line waiver `MPS-DECISION-WAIVED:<substantive rationale>` ONLY where the line
   is a within-instrument/bit-identical comparison (the D42 precedent) — placeholder
   rationales FORBIDDEN.
2. Verify: gate returns [] strict-clean; `pytest
   src/tac/tests/test_callsite_contracts_and_no_mps_decision.py` fully green.
3. Ledger rows (actor ddm_mg1) closing #1303; per-site disposition table in the memo.

## HARD CONSTRAINTS

- NEVER fabricate a [contest-CUDA] reference — cures (c) require a REAL artifact path
  the site's claim actually rests on; when in doubt use (d) reword-downgrade.
- Memo corrections APPEND-ONLY where the file is HISTORICAL_PROVENANCE class;
  reports/lane_maturity.md is GENERATED (regenerate via tools/lane_maturity.py report
  if the violation lives in generated text — fix the GENERATOR/registry note, not the
  output). `upstream/` READ-ONLY. NO Modal, NO scorer.
- Serializer commits w/ post-edit shas; `.py` = 2 genuine review passes (comment-only
  edits still count as .py touches); #1293 bundle path on git denial.

## PRIOR NEGATIVE SIGNAL

- #821: N violations of one template = ONE fact — if several sites are one copied
  phrase, fix the phrase class and say so (population honesty), not 21 heroics.
- The vacuous-control class (#1086): after the flip, run the POSITIVE control — inject
  a synthetic violation in a temp file under a scanned dir and prove the gate still
  fires; the gate must not have been neutered by the cures.

## OPTIMAL FORM

- Reference: the gate source (`src/tac/preflight.py` `_check_mps_decision_in_text` +
  `_MPS_DECISION_EXEMPT_TAGS`) + the 5fcf9c1c7f scope-fix commit + the D42 waiver as
  the worked example of an honest waiver rationale.
- SCOPE: the 21 live sites (re-measured). MECHANISM reductions FORBIDDEN: no blanket
  waivers, no exemption-list additions for individually-cureable files.
- **PRIOR-LAW PREDICTION (falsifiable):** most sites classify as (a)/(b) rule-
  restatements and historical anchors (the gate's own false-positive classes), ≤5 need
  (d) honest rewording, 0 need new exemption paths. FALSIFIER: >5 sites are genuine
  advisory-derived decisions — then the finding is a live measurement-discipline gap,
  reported per-site to MAIN, not silently reworded.

## DELIVERABLE

`.omx/research/ddm_mg1_mps_gate_burndown_20260826.md` — 21-row disposition table +
gate-clean + test-green + positive-control receipts + ledger rows closing #1303 +
GESTALT-DELTA line. Serializer commit (or bundle). End with the own-vehicle frontier
line.
