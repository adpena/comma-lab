# lc2 exact-row adjudication + the CPU-axis verdict + the bar moved (PR135)

**Date:** 2026-08-10 · **Author:** MAIN · **Status:** ADJUDICATED (pointer updated)

## 1. The exact row — ADJUDICATED, POINTER MOVED

**S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, locked upstream venv, n600].**
Archive sha256 `f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45`.

Custody chain, each step verified by MAIN:

- Local bytes: `/Volumes/VertigoDataTier/pact/ddm_lc2_20260810/submission/archive.zip`
  hashes to `f154f0ab…`, 187,226 bytes. Matches both runs' `expected_archive_sha256`.
- Measured TWICE, bit-identical: calls `fc-01KZP3R7QWCTJQWGYHGBNJM4GQ` and
  `fc-01KZP50CVABM8P9VGXMF8TK1PS`. Both: n600, Tesla T4 (`gpu_t4_match=true`),
  `/opt/upstream-locked-venv/bin/python --upstream-uv-group cu128`, runtime tree
  `30a6fb66…` validated, `validation_errors=[]`, canonical path
  `archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda`.
- S recomputed from components (never the rounded `final_score`):
  `100*0.00029662 + sqrt(10*2.332e-05) + 25*187226/37545489 = 0.16959899569230852`.
- Correction of my own stale label: I had carried `[CUDA env-mismatch advisory]`
  over from the ai1 lineage. The run-2 command proves the locked venv was live.
  The label was wrong; the runs are clean.

**vs the PR130 bar (0.172141297491896447): Δ = −0.002542302.** First own-custody
sub-PR130 exact row. Pointer registration: adjudicated claims row on
`lane_ddm_lc2_paired_20260810_contest_cuda` (`posterior_update=accepted`) →
`tac.frontier_scan` → `canonical_frontier_pointer.json` CUDA anchor
0.2053300290 → **0.1695989957** (refreshed 2026-08-10T16:53:24Z).

Borrowed-substrate accounting (NO-FAKE #7): the vehicle is the PR130 base
(semantic-pose-HPAC CPR1) with OUR ANS/constriction token recode +
temporal_reversion, under the operator's 2026-08-06 off-the-shelf grant and the
2026-08-09 roadmap "reproduce PR130 → iterate on THAT base." This row is the
first down-step on that base, not an original-vehicle claim.

Submission remains blocked by #1008 (constriction dep closure). Evidence
update: the inflate self-install WORKED on both Modal axes
(`Installed 2 packages: brotli==1.2.0, constriction==0.5.0` via uv). Owed:
the bare-venv bootstrap smoke.

## 2. The CPU-axis verdict — BUDGET-INFEASIBLE, measured (settles #998's open leg)

Call `fc-01KZP70MKR3Z5B0XZG5BZK77GM` (contest_cpu, locked venv, 8-core/16GB
Modal container) returned rc=1: **inflate timeout at 1800 s**.

- Token decode: 1,777.6 s for 600/600 (~3 s/pair, sequential ANS decode).
- Render: 180.8 s. Total ≈ 1,958 s > the 30-minute contest budget.
- The 3,662,409,600-byte raw WAS fully written; the deadline had passed.
  Payload harvested to `experiments/results/ddm_lc2_exact_row_20260810/harvest_cpu/`
  BEFORE any scalar read (ALWAYS KEEP THE PAYLOAD).

Verdict: the lc2/PR130-lineage bytes are **CUDA-locked at current decode
speed** — the contest CPU runner is 4-core, strictly slower than the measured
8-core container. A CPU number bought with a raised timeout can never be a
score claim (budget exceeded), so no re-dispatch.
`verdict_scope=INSTANCE(lc2 decode speed)`. Reactivation: a decode that fits
1800 s on 4 cores (e.g., parallel per-stream ANS decode or a faster token
codec). This is #835's territory: decode wall-clock is a term we can move.

## 3. The bar moved: PR #135 at 0.162

The same refresh fetched the official leaderboard:
**PR #135 `semantic-pose-HPAC_CPR1_polished` rank 1 at 0.162** (PR130's author,
polished). Effective frontier: 0.172 → **0.162**.

Honest arithmetic: our new row beats the OLD bar by −0.002542 and sits
**+0.007599 ABOVE the NEW bar** (≈ 11,414 B rate-equivalent at 25/W). We
crossed the bar the day the bar moved. Consequence per the
zero-gravitational-pull law: re-derive the aim against PR135, not PR130.
PR135 intake fired (codex arm) per the public-frontier-watch default order.

## 4. Claims + records

- `lane_ddm_lc2_cpu_20260810_contest_cpu` → terminal
  `failed_inflate_timeout_budget_infeasible`.
- `lane_ddm_lc2_cpu_20260810_contest_cuda` → terminal
  `stopped_cancelled_by_main_kill_correction` — supersedes my FALSE
  `stale_superseded_never_spawned` row (the leg DID spawn,
  `fc-01KZP6X99F259EKV0XRBVG6963`, then my kill cancelled it). Full
  correction: `.omx/research/ddm_lc2_claim_correction_20260810.md`.
- `lane_ddm_lc2_paired_20260810_contest_cuda` → `completed_modal_auth_eval_adjudicated`
  (the pointer anchor row).

## 5. #381 Modal envelope — this boundary's spend (estimates, T4 $0.59/hr class)

| item | call | est. cost |
|---|---|---|
| CUDA run 1 (647 s T4 + image build) | fc-01KZP3R7… | ~$0.15 |
| CUDA run 2 (647 s T4) | fc-01KZP50C… | ~$0.11 |
| CUDA run 3 CANCELLED (my kill; build + partial) | fc-01KZP6X9… | ~$0.10–0.20 (waste, mine) |
| CPU run (1,965 s, 8-core/16GB) | fc-01KZP70M… | ~$0.55 |

Boundary total ≈ $0.9–1.0. The CPU spend bought a real verdict
(budget-infeasibility + #998), not a failure.
