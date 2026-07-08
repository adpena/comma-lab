---
doc_type: t5_crucible_recess_wave1_measurements
role: RECESS WAVE-1 FINISHER (respawn; predecessor credit-died mid-R3 — its partial AA-gate fix
  completed, verified against the trainer render_aa block, committed a0b82ba6c)
date: 2026-07-07
items: R3 (P11' AA memory gate AS AMENDED) · R1 (P5 LBND4-on-smoothed) · R6 (P7 n600 realized-parity)
axis: ALL numbers [macOS-CPU advisory] (MLX/CPU local; no contest axis touched). Pointer
  contest-CPU 0.19110 UNMOVED — this file is MEANS.
review_status: self-executed-pre-registered (bands + kills pre-registered in
  P5_second_redteam_verdict_20260707.md §4 BEFORE these executions; this doc written by the
  executor in the same session — flagged per the verdict-review-status discipline; a fresh-eyes
  pass may re-run every command below, all are deterministic + on-disk)
---

STORES CONSULTED: P5_second_redteam_verdict_20260707.md (full — R1/R3/R6 bands + kills + F4
amendment spec) · DRAFT_OPTIMAL_STACK_v2_20260707.md (full — ARM-PRIMARY config §1.1/1.2, P11
row §7, rate plan §5.1) · trainer `experiments/train_levelset_witness_realized_through_R_mlx.py`
render_aa block L2606-2760 (the reconciled Q3 arithmetic the gate model was verified against) ·
mod32cap run dir `levelset_n600_witness_mod32cap_20260706T115554Z/` (launch.sh = ARM-PRIMARY
basis; ckpts; levelset_train_result.json) · `tools/witness_memory_preflight.py` +
`tools/launch_witness_run.py` (predecessor diff, completed+committed) · CLAUDE.md non-negotiables
· docs/operating_manual_craft_handoff.md.

# RECESS WAVE-1 — R3 / R1 / R6 (measured; per-item command · numbers · verdict · unblocks)

## R0 (prerequisite) — the predecessor's AA-gate fix: COMPLETED + COMMITTED

The credit-died predecessor left `tools/witness_memory_preflight.py` (+80/−7),
`tools/launch_witness_run.py` (+15/−3), and 7 new tests UNCOMMITTED. Verified against the P5
F4 amendment spec AND the trainer's own reconciled Q3 arithmetic (render_aa block, "#224
Option-B DECISION"): the model adds (a) ONE shared fine curvelet tensor, (b) per-pair fine
dir-feats — 25.2 MB/pair @ ss=2/ndf2, matching the trainer's measured figure exactly; full
mode stores all P, batch mode the FIFO cap 8, the fail-closed default `refuse` is projected
at FULL cost (conservative), (c) the ss² fine-grid forward EXCESS (fwd 8 = base 2 + excess 6
@ ss=2 — matches the trainer's "fwd ~8"). It deliberately does NOT model the debunked naive
ss²×cf_mx_cache (~164 GiB). The launcher half implements the P5 OR-branch: the calibration
smoke (5 ep < the 25-ep verdict cadence, never fires a verdict) now ADDS the measured
chunked-verdict floor (6.0 GiB) to the smoke actual BEFORE the overrun verdict — conservative
(double-counts toward REFUSE, never toward false-SAFE).

- Verification: 33/33 tests pass (`pytest src/tac/tests/test_witness_memory_preflight.py`),
  ruff F821/F401 clean, review-gate two clean passes on all three files.
- Commit: **a0b82ba6c** (serializer, post-edit shas, [no-triality]).
- Pre-existing unrelated failure noted (NOT this diff): `test_launch_witness_run.py::
  test_build_launch_sh_structure` fails on HEAD — stale expectation `#!/bin/bash` vs the
  committed cross-platform `#!/usr/bin/env bash` launcher shebang. Confirmed by stash-test.
- Also in the working tree but NOT mine and NOT committed: `tools/dashboard_server.py`
  SANDBOX-tab work + 2 untracked docs (`modular_theory_deepmath_review_20260707.md`,
  `docs/sandbox_pontryagin_lie_deepmath_context.md`) — a different lane; left untouched.

## R3 — P11′ AA memory gate AS AMENDED, at v2's ARM-PRIMARY worst case

**Command** (config = mod32cap launch.sh + `--w-pose 1` + `--render-aa supersample
--aa-supersample 2 --aa-self-orient-fine-mode full --lane-render-band
--lane-band-start-epoch 350`; band/islands/pose flags carry no additional modeled memory term):

```
.venv/bin/python tools/witness_memory_preflight.py --launch-sh <arm_primary_worstcase.sh> \
    --total-ram-gib 128 --strict
```

**Measured numbers** [macOS-CPU advisory; deterministic model, MEASURED constants]:

| variant | breakdown (GiB) | projected peak | vs 0.70×128 = 89.6 | verdict |
|---|---|---:|---|---|
| ARM-PRIMARY worst case (ndf=4, fine-mode full/refuse) | fixed 15.0 + cf 47.13 + gt 3.41 + verdict 6.0 + **aa_fine 34.36** | **105.90** | EXCEEDS by 16.3 | **REFUSE (rc=3)** |
| same config, pre-amendment gate (AA-blind) | aa_fine 0.0 | 71.54 | under | false-SAFE — the F4 class, now extinct |
| fallback: fine-mode **batch** (FIFO 8) | aa_fine 6.61 | 78.15 | under (11.4 headroom) | SAFE — but trainer-measured ~29 s/ep fine-EDT thrash @ n600 ⇒ wall-clock NON-viable for the curriculum (trainer's own Wave-B numbers) |
| fallback: **ndf=2** full (the trainer's reconciled Q3 config) | aa_fine 20.30 (cf 43.2 @ in_feat 88) | 87.91 | under by **1.7 GiB** | SAFE — knife-edge; the assumed-margin ledger (10 GiB fallback) would flag it |

**Verdict vs the pre-registered R3 band ("peak RSS SAFE at REAL config incl. verdict spike"):**
**KILL-branch fires for AA-as-written** — the amended gate REFUSES the ARM-PRIMARY worst case
(ndf=4 + full). Per R3's own kill law: *"preflight REFUSE ⇒ AA → run-2 with measured cost
written."* The measured cost IS now written (this table): AA @ ndf=4/full needs 34.4 GiB it
does not have on a 128 GiB box at safe-frac 0.70 alongside the in_feat-96 cf cache.
Named alternatives with measured numbers: (i) ndf=2/full projects 87.9 SAFE but with only
1.7 GiB margin — BELOW the 10 GiB assumed spike/model-error margin, so honestly it is
margin-REFUSE too until the reconcile ledger measures a smaller p95; (ii) batch mode is
memory-SAFE but wall-clock-killed by the trainer's own measured ~29 s/ep EDT thrash;
(iii) `--render-aa ipe` (basis-level cone AA, ~0 memory/compute, self-orient-compatible,
fully wired) is the surviving AA form for run-1. Note also the trainer's Wave-D header:
supersample is train-only + decode-budget-disqualified + measured −49% witness-harm — v2's
AA-IN decision should be re-adjudicated against ipe at the recess close, not just on memory.

**s/ep half of R3:** NOT run here — the ≤1.5×107 s/ep throughput smoke requires a governed
5-ep trainer launch; with the memory half already REFUSING AA-as-written, the throughput
smoke on the REFUSED config is moot (it would fire the very config the gate refuses). The
launcher's calibration path now carries the verdict-delta amendment for whichever surviving
config (ipe or ndf=2-with-measured-margin) goes to GO.

**Unblocks:** the F4 AA-in decision (now DERIVED with measured cost, exactly what F4 demanded
either way) + §8's s/ep base re-projection is scoped to the surviving AA form.

## R1 — P5 LBND4-on-smoothed probe — PENDING (checkpoint 1 committed before start)

## R6 — P7 n600 realized-parity row — PENDING
