# ddm_rw1 Receipt - True-Domain Rewire Smoke

Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE.
Score claim: false. Promotion eligible: false. Pointer moved: false.

## Answer First

q3x old-vs-rewired delta on the bounded n=1 smoke did not flip the old FOLDED verdict.

| path | n | solver cap | q3 net flips | retained seg fraction | outcome |
|---|---:|---:|---:|---:|---|
| old naive-round | 1 | 2 | 45 | 0.30612244898 | FOLDED |
| rewired DK1-CVP | 1 | 2 | -1 | -0.00680272109 | FOLDED |

Delta: DK1-CVP block-limited smoke was 46 flips worse on q3 net reduction and -0.31292517007 in retained fraction. The DK1 receipt realized 64/3117 snapped Q3 blocks with `cvp_tap_radius=0`; this is a bounded codepath smoke, not a full q3x cure or n600 regrade.

FD row-3 reopen smoke produced a positive realized accept-rate signal: 1/6 accepted proposals on pair 0 versus FD1's recorded 0/6 baseline. The accepted proposal changed realized SegNet flips from 197 to 196 with zero introduced flips at scorer site (208,216), target class 1, current class 0.

CA1 row-4 six-site disposition counts: 1 attached cap-stop receipt and remained cap-bound, 5 relabelled/held as CAP-BOUND-at-stop. No site was silently skipped.

## What Changed

- `experiments/ddm_q3x_q3_convergence_measurement.py` now defaults q3x projection realization to DK1 CVP (`--realizer dk1-cvp`) while preserving the old `--realizer naive-round` A/B path.
- q3x now runs a typed cap ladder (`--cap-ladder`, default `25,50,100`) and records CA1-compatible cap-stop receipts instead of silently relying on a 25-step cap.
- q3x exposes `--solver-form solve-within-null-basis` to route through the SW1 null-coordinate solve form; the default remains project-after for the direct old/new A/B.
- `src/tac/optimization/rw1_true_domain_instruments.py` carries cap receipts, full element-grade vectors, DK1 q3 realization receipts, and JSON utilities.
- `src/tac/optimization/fd_integer_near_margin_proposals.py` implements typed FD proposals generated directly on the uint8 camera lattice near realized argmax margins, with realized-argmax validation in the proposal loop.
- `tools/smoke_ddm_rw1_fd_integer_near_margin.py` runs the bounded FD accept-rate smoke and writes the durable receipt.

## Measured Artifacts

- q3x old: `.omx/research/ddm_rw1_20260806/q3x_old_naive_n1.json`
- q3x DK1-CVP: `.omx/research/ddm_rw1_20260806/q3x_dk1_cvp_n1_block64.json`
- FD smoke: `.omx/research/ddm_rw1_20260806/fd_integer_near_margin_smoke.json`
- CA1 disposition: `.omx/research/ddm_rw1_20260806/CA1_CLASS_B_DISPOSITIONS.md`
- Registry: `.omx/research/ddm_rw1_20260806/INSTRUMENT_REGISTRY.jsonl`

## Recall Evidence

Stores consulted before coding: `.omx/tmp/codex_runs/rw1_prompt.md`, `.omx/tmp/codex_runs/_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, `HANDOFF.md`, `SYSTEM_MAP.md`, `.omx/research/ddm_vo1_20260806/REOPEN_LEDGER.jsonl`, DK1/SW1/CA1 receipts, FD1/FD2 receipts, q3x source, SW1 source, SQ1 solver source, and the canonical equation registry via `tools/list_canonical_equations.py --json`.

Additional recall beyond the charter seeds:

- SB1 recorded q3x only as an n=2 smoke with `/tmp/ddm_q3x_smoke2.json`; no durable `.last.txt` receipt was available, so rw1 wrote durable JSON under `.omx/research/ddm_rw1_20260806/`.
- NB1/Q31 evidence was incomplete or cap-bound; rw1 therefore did not use the old q3x/Q31 rows as family-level kill evidence.
- DK1 established the lattice-native pose-null realizer as a pruned CVP/Babai kept-set optimizer, not a global integer optimum.
- FD2 established the zero-accept issue as a seg-realization gap at uint8, not a pose-null-only veto; rw1 therefore reopened with integer near-margin realized-argmax validation.
- CA1's classified inventory named exactly six Class-B cap-default live/reopen-risk sites; rw1 disposition covers all six.

## Boundaries

Not measured: no n600 scorer job, no exact contest eval, no full q3x DK1 CVP run over all 3117 snapped blocks, no FD campaign relaunch, no public-wire archive, and no frontier promotion.

Measured: fast unit tests, q3x old/new n=1 advisory smokes, and FD n=1/6-proposal advisory smoke.

The element-grade vectors intentionally remain PARTIALLY-CURED. q3x still has NAIVE-NAMED step rule/metric/tie-breaks and UNKNOWN seed; FD still has a NAIVE-NAMED tie-break and cache grade. The q3x solver selected cap 2 and was cap-bound/still descending in both A/B runs.

Own-vehicle frontier line: pointer unmoved; no exact archive row was produced, so this is apparatus and bounded advisory evidence only, not goal progress.
