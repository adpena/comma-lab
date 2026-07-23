---
title: DDM v19c recursive correction saturation DAG feed
date_utc: 2026-07-23
lane_id: ddm_v19c_correction_saturation
research_only: true
execution_allowed: false
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
verdict: V19C_CORRECTION_SATURATION_ADMITTED_N600_ADVISORY
verdict_scope: "INSTANCE:V19C x SHA-bound v19b start x represented correction coordinates; no contest-axis, global-family, score, or promotion verdict"
pointer: "0.1910828242 [contest-CPU]"
pointer_moved: false
main_landing_review_required: true
---

# Purpose

DDM v19c measures the correction-line asymptote beyond v19b's fixed ten moves.
It recursively searches a deterministic 231-coordinate inventory, admits only
strict receiver-priced joint improvements, stops after 64 consecutive failed
proposals or 200,000 additional correction bytes, and then replays every DEV
admission sequentially on exact n600.

# Search surface

The typed inventory contains six interleaved actuator families:

1. inverse-solved signed row-band directions from every sealed v17 solve
   candidate;
2. Q8 all/template/sparse and pair-lifecycle regions over multiple signed
   quanta;
3. per-track worldsheet x/y events;
4. per-pair lifecycle worldsheet x/y events;
5. scorer-template swaps;
6. grammar-event template substitutions.

Pair-bearing coordinates are ranked within family by a measured Fisher-margin
error field. Cross-family admission is always the exact realized joint
objective. No predictive proxy or Fourier residual basis has verdict authority.

# Executable DAG

```text
SHA-bound v19b config + receipt
  |
  +-> reproduce exact v19b DEV archive and frozen scorer cells
  |
  +-> build 231-coordinate typed inventory
  |     inverse row-band directions
  |     Q8 regions/quanta
  |     track + pair-lifecycle worldsheet events
  |     template swaps + grammar substitutions
  |
  +-> deterministic interleave + recursive scale cycles
  |     -> compile current joint stack plus proposal
  |     -> exact receiver/uint8/Seg/Pose/archive measurement
  |     -> admit iff incremental joint Delta S < 0
  |     -> preserve candidate archive + atomic decision checkpoint
  |     -> stop at K=64 failures or +200,000 B
  |
  +-> exact n600 replay from sealed v19b n600 state
  |     -> replay DEV admissions in original order
  |     -> exact decoded-member/G1-raster pair-support classification
  |     -> exact NumPy camera identity before scorer-row reuse
  |     -> 16-pair preserved scorer checkpoints
  |     -> retain only strict incremental n600 Delta S < 0
  |     -> strict full-n600 replay of the final admitted archive
  |
  +-> saturation receipt
        -> per-admission (bytes,d_seg,d_pose,Delta S) curve
        -> role/residual c1 bucket attribution
        -> atom-order gauge
        -> artifact-hygiene and false-authority receipts
```

# Resumability and custody

- DEV restores every prior proposal decision, reconstructs the accepted state,
  and verifies the exact archive/scorer endpoint before continuing.
- n600 restores every prior decision, reconstructs only admitted proposals,
  and verifies the exact archive endpoint before continuing.
- Each n600 proposal also resumes from preserved 16-pair batch checkpoints.
- Pair support is re-derived from exact nested archive member deltas. G1
  changes are lifted, losslessly re-encoded, rasterized per changed pair, and
  compared to the current mask; unexpected carrier-member changes fall back
  to conservative full support.
- A support-bearing batch reuses its scorer row only after exact NumPy camera
  array identity. The final archive is then decoded by the strict published
  receiver and every n600 scorer row is re-derived into a second preserved
  batch checkpoint set.
- Candidate archives and stage receipts are durable; no operator evidence
  cites `/tmp`.
- Storage preflight is fail-closed, with a 116 GiB ceiling and SSD waterfall.
- No paid dispatch, remote execution, GPU run, live vehicle actuation, or
  contest-axis evaluation is authorized.

# Triality

- DSL:
  `.omx/research/configs/ddm_v19c_correction_saturation_20260723.json`
- DAG: this file plus immutable DEV/n600 stage checkpoints
- equations:
  `.omx/research/ddm_v19c_correction_saturation_canonical_equations_20260723.md`
- implementation:
  `tools/measure_ddm_v19c_correction_saturation.py`
- regression:
  `tools/tests/test_measure_ddm_v19c_correction_saturation.py`

# Measured stage exit

DEV stopped at the preregistered `K=64` consecutive-failure boundary after
1,002 proposals over 231 unique coordinates. It admitted 153 moves across all
six families. Exact sequential n600 replay retained 104, rejected 47 with
nonnegative measured joint delta, and classified two state-dependent
worldsheet proposals as `INFEASIBLE_N600_COMPILE`.

The strict final full-n600 replay matched the sequentially assembled endpoint:

- archive: 137,827 B,
  SHA `dc767b59c9e8671b6870e0f9f17a24cfe900dd0f2ae2a251825e41566b52e4c9`;
- `d_seg=0.024786978828`;
- `d_pose=163.061210029156`;
- `Delta S=-0.18073912464057892` versus sealed v19b;
- strict replay digest:
  `3efcc943769209a5393ee03ee59b80d4287b689d16f03dfcc29589f52ded6cc3`.

The complete 104-point admitted curve and every rejected/classified proposal
are in
`.omx/research/ddm_v19c_correction_saturation_20260723T063500Z/stage_checkpoints/02_n600_saturation_curve.json`.
The final receipt is
`.omx/research/ddm_v19c_correction_saturation_20260723T063500Z/ddm_v19c_correction_saturation_receipt.json`.

# MAIN landing review

MAIN must independently verify:

1. the authority prompt, v19b config, v19b receipt, and v19b starting
   archive/cell SHA bindings;
2. all 231 unique coordinates and the deterministic recursive proposal order;
3. inverse-solved, Q8-before-one-uint8, worldsheet feasibility, template-swap,
   and grammar-substitution semantics;
4. Fisher-margin use only as within-family ordering and strict realized
   `Delta S < 0` as cross-family authority;
5. the exact 64-failure/200,000-byte stopping condition;
6. DEV and n600 resume endpoint verification, decoded-support correctness,
   exact camera-identity reuse, and both proposal/final-replay n600 batches;
7. n600 curve, c1 bucket arithmetic, atom-order camera identity, artifact
   hygiene, false-authority labels, and pointer immobility.
