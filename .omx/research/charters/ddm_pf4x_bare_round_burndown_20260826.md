# ddm_pf4x_bare_round_burndown — cure the bare-.round()-in-eval-roundtrip gate's ~25 sites (r59 red), per-site, never weakening

## MANDATE

The r59 full-preflight chain (receipt
`.omx/tmp/preflight_full_r59_20260826/PREFLIGHT_RESULT.json`) is RED on
`check_no_bare_round_in_eval_roundtrip` (src/tac/preflight.py:20735): ~25 sites across
~20 files use bare `.round()` inside functions the scanner classifies as
eval-roundtrip-shaped (F.interpolate + "roundtrip" in name/docstring). Re-derive the
authoritative list live: run the check verbose (strict=False) and enumerate — the receipt
text is truncated.

1. PER-SITE adjudication (mg1 discipline, zero blanket moves), three typed classes:
   - (a) FORWARD-ONLY measurement/diagnostic/probe files (`probe_*`, `measure_*`,
     `diag_*`, `smoke_*`, parity runners): bare `.round()` is CORRECT there (no gradient
     consumer) — the gate's OWN sanctioned cure is a per-file entry in
     `_BARE_ROUND_READONLY_FILES` (preflight.py:20657, 4 existing exemplars) with a
     one-line rationale comment per entry. VERIFY forward-only by reading the file: no
     `.backward()`, no optimizer step, no loss consumed from the flagged path. A file
     that trains ANYWHERE does not take cure (a) — classify per function instead.
   - (b) DIFFERENTIABLE-NAMED flagged functions (e.g.
     `experiments/probe_hinerv_grid_vs_lever_dseg.py:123
     _roundtrip_to_eval_bhwc_differentiable`,
     `experiments/smoke_pose_film_cpu_disambiguator.py:95` same name): these claim
     differentiability while the scanner sees bare `.round()`. Adjudicate at source:
     if the `.round()` is on a detached/no-grad branch or a manual-STE idiom the
     scanner missed, record FALSE-POSITIVE with the exact line and use the smallest
     honest cure (same-line manual-STE conformance or readonly-file entry if the whole
     file is forward-only). If the gradient path is GENUINELY severed, that is a
     MEASUREMENT-INTEGRITY FINDING: fix with `Uint8STE.apply`
     (src/tac/quantization.py) AND name in the memo which banked results consumed the
     severed path (do NOT silently fix; do NOT re-run their measurements — report).
   - (c) TEST FIXTURES (`src/tac/torch_vehicle/tests/test_film_trunk_decoupling.py:131`,
     `test_kd_warm_start.py:458`): if the fixture replicates the roundtrip forward-only,
     cure with the manual-STE idiom or Uint8STE.apply ONLY if the test still passes
     bit-identically; else adjudicate as (a)-style with a scoped rationale.
2. NEVER WEAKEN THE GATE: no regex changes, no scan-scope reduction, no blanket
   exemptions. After all cures: POSITIVE CONTROL EXECUTED — a synthetic violating file
   (F.interpolate + roundtrip-named function + bare .round()) placed in-scope must FIRE
   the check; remove it; re-run the check strict → CLEAN. Record both directions.
3. Ledger rows via tools/canonical_task_status.py (actor ddm_pf4x); per-site disposition
   table in the memo. Do NOT relaunch the preflight chain from the arm (the codex
   sandbox denies `ps` — the r57 lesson); end with a typed fire-order for MAIN to
   launch r60.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal, NO scorer runs, NO archive mutation. `.py` = 2
  genuine review passes per edit batch; serializer commits w/ post-edit shas; on the
  #1293 git-objects denial retain the bundle (memo
  `.omx/research/ddm_hd1_apparatus_two_landings_20260826.md`).
- STOP AND REPORT as a typed blocker (do NOT improvise) on any site whose cure would
  change score-relevant semantics of a LIVE surface (a trainer on the launch path, a
  receiver, canonical equations) — historical probes are in scope, live trainers are
  not expected in this population; if one appears, blocker.
- Memo corrections APPEND-ONLY on HISTORICAL_PROVENANCE files.

## PRIOR NEGATIVE SIGNAL

- mg1's no-mps 21→0 burn-down (memo `.omx/research/ddm_mg1_mps_gate_burndown_20260826.md`,
  commit `7d7ddfc304`): the reference per-site form — zero waivers, positive control
  executed; #821: N sites of one copied pattern = ONE fact, fix the class, count
  honestly (the `_roundtrip` helper here is visibly copy-pasted across probes — expect
  few distinct patterns, many instances).
- The PCC2 cure (commit `1f1c3c92d9`): read the gate's OWN detection vocabulary before
  shaping a cure — the manual-STE regex at preflight.py:20654 is same-line only; a
  multi-line manual-STE would false-positive (candidate (b) mechanism).
- #842: these violations accrued in the dark (commit hook runs none of these gates) —
  stale populations, not fresh regressions; date the debt honestly.

## OPTIMAL FORM

- Family REFERENCE exemplars w/ provenance pins: mg1 disposition memo (`7d7ddfc304`) ·
  the gate's own `_BARE_ROUND_READONLY_FILES` entries (preflight.py:20657) · Uint8STE
  canonical (src/tac/quantization.py) · r59 receipt
  (`.omx/tmp/preflight_full_r59_20260826/PREFLIGHT_RESULT.json`).
- SCOPE reductions declared: this arm cures ONE gate (bare-round); subsequent chain reds
  are MAIN's loop. MECHANISM reductions FORBIDDEN: per-site reads (no filename-only
  classification), executed controls, no placeholder rationales.
- **PRIOR-LAW PREDICTION (falsifiable):** ≥80% of sites take cure (a)
  (forward-only measurement, readonly-file entries), the two `_differentiable`-named
  sites are the scanner's same-line-manual-STE miss (FALSE-POSITIVE) rather than real
  severed gradients, and zero live-surface blockers appear. FALSIFIER: a genuinely
  severed differentiable path in a banked measurement — then the memo leads with that
  measurement-integrity finding and names the affected receipts.

## DELIVERABLE

`.omx/research/ddm_pf4x_bare_round_burndown_20260826.md` — per-site disposition table
(site → class → cure → control) + executed positive/negative controls + check strict
CLEAN + ledger rows + GESTALT-DELTA line + typed r60 fire-order for MAIN. Serializer
commits (or bundles). End with the own-vehicle frontier line.
