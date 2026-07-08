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

| variant | breakdown (GiB) | projected peak (GiB / MiB) | vs ceiling 89.6 GiB (91,750.4 MiB) | verdict |
|---|---|---:|---|---|
| ARM-PRIMARY worst case (ndf=4, fine-mode full/refuse) | fixed 15.0 + cf 47.13 + gt 3.41 + verdict 6.0 + **aa_fine 34.36** | **105.90 / 108,441.6 MiB** | EXCEEDS by 16.30 GiB (16,691.2 MiB) | **REFUSE (rc=3)** |
| same config, pre-amendment gate (AA-blind) | aa_fine 0.0 | 71.54 / 73,257.0 MiB | under by 18.06 GiB | false-SAFE — the F4 class, now extinct |
| fallback: fine-mode **batch** (FIFO 8) | aa_fine 6.61 | 78.15 / 80,025.6 MiB | under by 11.45 GiB | SAFE — but trainer-measured ~29 s/ep fine-EDT thrash @ n600 ⇒ wall-clock NON-viable for the curriculum (trainer's own Wave-B numbers) |
| fallback: **ndf=2** full (the trainer's reconciled Q3 config) | aa_fine 20.30 (cf 43.2 @ in_feat 88) | 87.91 / 90,019.8 MiB | under by **1.69 GiB (1,730.6 MiB)** | SAFE — knife-edge; the assumed-margin ledger (10 GiB fallback) would flag it |

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

## R1 — P5 LBND4-on-smoothed source ($0, ~23 s wall)

**Command:** scratchpad probe (reproducible: same build path as the measured artifact —
`build_lane_band_pairs_from_lstars(gt_n600 lstars)` → `temporal_smooth_pairs_lines(win)` →
`lane_band_res_rate_report`); artifact written to
`experiments/results/lane_band_res_coder_20260707/lbnd4_on_smoothed_r1_measured.json`.

**Measured numbers** [macOS-CPU advisory; brotli q11 counted bytes — the exact byte-close
5th-block quantity]:

| source | LBND2 (B) | LBND4 best (B) | scheme | rate term | decode-reencode identical |
|---|---:|---:|---|---:|---|
| raw (re-run) | **41,526** | 30,892 | varint | 0.02057 | true (prior artifact) |
| smoothed win=5 | 28,050 | 20,827 | varint | 0.01387 | **FALSE (all 3 schemes)** |
| smoothed win=9 | 26,260 | **18,832** | varint | 0.01254 | **true (all 3 schemes)** |

**Verdict vs pre-registered band (18-22 KB; kill ≥24,149 B):** **PASS.** Both smoothing
windows land INSIDE the predicted band (18,832 / 20,827 B exact) and under the kill by
5,317 B / 3,322 B. S-unit denominations (1 B = 6.6586e-7 S): the band-coder min is MEASURED
at 18,832 B (win9) = rate 0.012539 S — vs §5.1's P5 tail placeholder 18,000 B the tail was
832 B (5.54e-4 S) optimistic; vs the §5.1 central 30,892 B the smoothed option saves
**12,060 B = 0.008031 S = 4.51× the crossing margin (0.00178 S)**.

**Flag (one investigative sentence owed before shipping win5):** at win=5 the
decode→re-encode identity FAILS for all three schemes (quantized smoothed values likely
sitting on bin edges); win=9 round-trips bit-exact AND is smaller — win9 is the admissible
config; win5 is quarantined until the identity defect is explained.

**F16ii RESOLVED:** the LBND2 re-measure gives **41,526** (S4/S5's number); the S6/draft
figure 41,562 was the stale seat. The v2 §5.1 worst-joint tail should re-print with 41,526
(−36 B; direction favorable, arithmetic unchanged at the printed precision).

**Caveat inherited from the coder header:** smoothing is a LOSSY geometry change; whether the
smoothed band NETS lower S is the trained-with d_seg leg (P8/F8), out of R1's scope —
pre-registered that way.

**Unblocks:** band-coder min for the §0.2 crossing rate legs + F16ii closed.

## R6 — P7 n600 REALIZED-PARITY ROW on mod32cap ep650 (the apparatus-trust measurement)

**Commands (all foreground, chunked, resumable — the harness SIGURG-kills ~5-min calls, MEASURED
again this session: the single-call full run died rc=144; the chunked drivers below completed):**

1. Byte-close + packet (real tool, `--skip-parity --keep-packet`): `tools/levelset_byte_close_and_eval.py
   --ckpt-dir <mod32cap> --npz-name levelset_witness_ema_BEST.npz` → packet
   `experiments/results/levelset_packet_20260708T013253Z/` — **archive.zip = 83,427 B exact**
   (0.bin 84,126 B; rate_term 0.055551 S), decode tier `decode_cpu_16gb` (contest=True,
   bit_exact=True, 1-thread-BLAS inflate env).
2. Chunked REAL-path inflate: driver imports the PACKET'S OWN `inflate.py` and runs its
   `_init_worker`/`_render_pair` (op-for-op the shipped serial body; disjoint-offset writes;
   same spawn-Pool mechanism as the shipped main; fp64 `_FDT` default) over pair ranges
   4×150: walls 74.8 + 70.6 + 70.1 + 68.2 s = **283.7 s total ≈ 4.7 min** (band ≤20 min: PASS).
   State: `r6_inflate_state.jsonl`; raw 3,662,409,600 B (full shape OK).
3. Chunked n600 verdict: frozen CPU-torch `cpu_verdict_d_seg_batch`/`d_pose_batch` at
   **verdict-batch 32** over the .raw (3 calls: 0-96 / 96-352 / 352-600; per-pair rows appended
   resumable to `r6_verdict_pairs.jsonl`; 600/600 unique pairs; process RSS 10.9 GiB incl. the
   3.7 GiB GT cache — per-chunk verdict transient ~1-2 GiB < 8 GiB requirement).

**THE PARITY ROW** [macOS-CPU advisory; full precision per the operator pin]:

| side | d_seg (n600) | Δ vs decoded | Δ in S-units |
|---|---:|---:|---:|
| **decoded through the REAL inflate path** | **0.00361457** (0.0036145697699652775) | — | seg term 0.361457 |
| training-side, SAME load path (chain-A reconstructed-feats re-verdict) | 0.00351030 | **+0.00010427** | **+0.010427 S** |
| training-side, run-logged live (ep650 best) | 0.00336619 | +0.00024838 | +0.024838 S |

d_pose decoded = 124.3228 vs training-side ep650 123.9865 (pose-BLIND run, w_pose=0 — expected
magnitude; not a parity axis here). Per-pair d_seg: std 0.000612, max 0.005788 @ pair 518.
S_advisory on the decoded frames = 0.361457 + 35.259439 + 0.055551 = 35.676446 (pose-blind).

**Verdict vs pre-registered band (realized d_seg 0.0034±3e-4 → [0.0031, 0.0037]; kill
Δ > +5e-4 vs 0.0035103):** **PASS — decode integrity holds.** 0.0036146 is inside the band;
Δ = +1.0427e-4 < 5e-4 (the kill would have fired at ≥ 0.0040103).

**LOAD-BEARING FLAG (the operator pin's exact concern):** the measured Δ = **+1.0427e-4 d_seg
= +0.010427 S = 5.86× the crossing margin (0.00178 S)**. The apparatus is TRUSTED (no
fix-before-run defect) but NOT free: the decode/quantization leg currently eats ~5.9 crossing
margins. Decomposition of live→decoded (+2.4838e-4 = +0.024838 S total): reconstruction gap
(live→same-load-path) +1.4411e-4 = +0.014411 S (the chain-A +4.3% number, reproduced by
construction) + decode gap (same-load-path→decoded: int8-quantize + brotli roundtrip + fp64
inflate + chunked-verdict read-back) +1.0427e-4 = +0.010427 S (+3.0%). Every §0.2/§9.1 rung
that budgets "byte-closed realized +0..+1e-4" sits at the TOP edge of that prior at THIS
checkpoint — the +1e-4 allowance is consumed, not spare.

**Unblocks:** decode integrity for every later row (per-stage byte-closes, AA
byte-close-selection verdicts, twin comparisons) — all admissible; each should budget the
measured +1.04e-4 decode leg rather than assuming 0.

## Session totals (means/ends firewall)

R3 REFUSE-as-designed (gate now honest at the real config) · R1 PASS 18,832 B (win9,
bit-exact) · R6 PASS with the +0.010427 S decode-leg flag. All numbers [macOS-CPU advisory];
**pointer contest-CPU 0.19110 UNMOVED — everything here is MEANS** toward the §7 ROW.
