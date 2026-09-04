# ddm_md1 — PRE-REGISTRATION (committed BEFORE the numbers)

`[no-triality] [p0-ledger-ok]` · arm `ddm_md1` · charter
`.omx/research/charters/ddm_md1_micro_to_macro_dynamics_20260904.md` (commit `a5b58f1fa`) ·
axis `[macOS-CPU advisory; reconstructed from retained 16-step checkpoints; not contest authority]` ·
`score_claim = false` · $0, CPU only, 0 Metal / 0 Modal / 0 contest eval.

This file exists so the two predictions and their falsifiers are on the record with a commit
timestamp that precedes the measurement. The sweep that will answer them was launched at
2026-09-04T12:09:13Z; this file is committed while it runs.

## What is being measured

The exact per-site SegNet argmax at ~69 checkpoints along the QBR1 born-fairform trajectory
(dense every 16 steps through 0–512, every 64 to 2,048, every 256 to 5,000, plus the terminal
state), for BOTH the live weights and the EMA shadow, on all 32 pairs of the sealed no2
selection, through the trainer's own render → bicubic camera → uint8-STE → frozen CPU scorer
path. Two cells: the sealed cold control (seed 20260902) and the live ng1 warm-transition cell
(same seed, same data order, one lever — r10's AdamW moments carried).

Every site is then classified by what its error trajectory DID, under a falling rule whose whole
point is that it PARTITIONS: `CHURN` (more than 4 correct↔wrong flips) → `PERSISTENT` (wrong at
step 0 and wrong at ≥ 90% of swept checkpoints) → `NEW_PERSISTENT` (correct at 0, wrong at the
terminal checkpoint) → `TRANSIENT_BORN` (correct at 0, wrong in between, correct again at the
terminal) → `HEALED` (wrong at 0, not persistent, correct at the terminal) → `ALWAYS_CORRECT`.
`HEALED` is the fifth class a partition requires and the charter's four do not name; it is
reported explicitly, never folded into another class.

## PREDICTION 1 — reachability

**Prediction (charter, from gc1's capacity closure): `PERSISTENT` ≥ 60% of the TERMINAL d_seg,
and the persistent set is Lane- and edge-concentrated.**

**FALSIFIER: `PERSISTENT` < 40% of the terminal d_seg.** If it fires, optimization levers alone
could plausibly reach the accuracy target and the capacity closure is re-graded.

**Honest grading of the premise, stated before the answer.** The charter's ≥ 60% leans on
`ddm_gc1_generator_capacity_control_20260903.md`. gc1 is a *static capacity sweep* (four packet
points) on the GF1 four-stream analytic generator, scored by categorical Hamming mismatch against
the exact field — **not** a d_seg measurement, **not** the QBR1-born trained object, and **not**
a trajectory. What gc1 does supply is a capacity-resistance signal on the right class: Lane's
mismatches fall only 1.16× while Road falls 2.59× and Undrivable 3.03× as the packet grows
1.599×, so Lane's *share* of the residual rises from 319,147/1,334,939 = 23.9% to
275,034/725,965 = 37.9%. That is a real, measured, monotone signal for "Lane is the hard class",
and it is the strongest thing behind the 60% number — but it is not a measurement of persistence
in d_seg space on this object. The closer premises on the right object are sd1's 75.9–83.5% of
the excursion mass on Lane/Movable-winning edges and 58.83% of d_seg on the two Lane edges. The
60% threshold is therefore a PRIOR, not a transfer; I am recording that it is weakly supported
before I see the number.

## PREDICTION 2 — warm ⊂ cold

**Prediction: the warm cell's excursion sites are a SUBSET of the cold cell's (AdamW moments damp
the first steps, they do not redirect them).**

**FALSIFIER: more than 30% of the warm excursion sites are absent from the cold set.**

Excursion site := correct at step 0 AND wrong at that cell's own d_seg-peak checkpoint (the peak
is taken empirically from the swept series, not assumed at 2,000).

## PRIOR-LAW PREDICTION (the re-anchor discipline)

Under sd1's measured τ-schedule identity and its rare-class over-paint finding, the prior law
predicts, for the cold control:

1. Predicted/GT area for Lane and Movable rises from step 0, peaks at or near step 2,000, and
   partially recovers by 5,000 (sd1 measured 1.03339 → 1.09291 → 1.06288 for Lane and
   1.02591 → 1.05801 → 1.04261 for Movable at 0 / 2,000 / 5,000, DALI, n32 HT).
2. The exact `d_seg_hat` peaks at step 2,000 and ends above its start (recorded milestones:
   0.002518335978190104 → 0.0032170613606770835 → 0.002758916219075521).

If the 16-step reconstruction disagrees with either, the disagreement is the finding.

## What would make the whole measurement inadmissible

The burn ran on Metal; this reconstruction runs on CPU. The CPU-vs-retained-MPS argmax residual
is measured at every milestone that retained its argmax rather than assumed away. A step-0 probe
before launch measured **51 differing sites of 6,291,456 = 8.106e-6**, and
`d_seg_hat` 0.002519353230794271 (CPU) vs 0.002518335978190104 (retained MPS) = **+0.0404%
relative**. The trajectory classes are computed entirely within the CPU series, so they are not
contaminated by that gap; only the comparison to the recorded milestone carries it. If the
residual grows by more than an order of magnitude at any later milestone, the class tables are
reported with that caveat welded on.

## Calibration gate

The per-class contributions must sum to the total `d_seg_hat` at every checkpoint. The bridge is
carried in the exact INTEGER numerator `W(t) = Σ_p w_p · n_wrong(p,t)` (HT weights are integers
15/30 and site counts are integers), so the gate is `max |Σ_classes − total| == 0` in integers,
not a float tolerance. A non-zero residual invalidates the decomposition and the memo says so.

## Constraints honoured

$0 CPU only (4 threads, nice 10); the Metal belongs to the live warm cell; nothing is written
under any cell's `runs/`; `upstream/` and `submissions/` are read-only; no `/tmp`; payloads
(per-checkpoint 32-pair argmax + margin-band code, per-site class codes, trajectory codes) are
retained under `/Volumes/APDataStore/pact/ddm_md1_micro_macro/` with sha256s in the JSON.
