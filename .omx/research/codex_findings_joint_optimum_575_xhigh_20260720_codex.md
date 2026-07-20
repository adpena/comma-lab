# Codex findings — Task #575 C2 governed-fire attempt and recovered exact row

**Date:** 2026-07-20
**Lane:** `lane_joint_optimum_575_xhigh_20260720`
**Axis:** `[contest-CPU]`, Linux x86_64 CPU-torch, selection and exact eval in the same container
**Verdict:** `#575 N600 LAUNCH REFUSED RC=4; MEASURED PARTIAL PRESERVED; SEPARATE EXACT POINTER CANDIDATE; MAIN REVIEW REQUIRED`

## Required #575 outcome

The requested C2 banded-around-source n600 run **did not fire**. The exact ordered safety chain
reached the following terminal state:

1. Typed `IntegerPlaneEmitter` DSL compilation and consumption by the dedicated trainer parser:
   **PASS**, policy SHA
   `d3226b536f84be62fad5a4b66a5dab41fc76a1a18c6f13c0b89aa1c3fd7477ef`.
2. Dispatch single-flight after closing the completed harvest: **PASS**, zero active claims.
3. Strict system-aware memory preflight: intrinsic projection **SAFE** at 73.4 GiB versus the
   89.6 GiB 70% ceiling, but system admission **REFUSED rc=4** at 114.9 GiB projected total versus
   a 100.9 GiB adaptive ceiling (14.0 GiB excess). No bypass was used.
4. Governed launcher: **NOT ACTUATED**, as required by the rc=4 fail-closed gate.

Even after memory clears, the current C2 materializer is rc=6-blocked by construction: no real
n600 positive-anisotropic band manifest was found, #553 PDW2 remains target-only without a
scorer-free spatial/RGB pullback, and the receiver-bound curvelet/shearlet plus executable
Fisher/secant/QP EV carrier is absent. This is the exact current-instance blocker, not a verdict
against C2.

The already-settled measured partial remains intact and resumable: eight checkpoints across
`warmup`, `band_fit`, and `rate_polish`, ending at
`m1_c2_glue_real_control_20260719__ipe_stage002_rate_polish_ep000001_step000000000009.json`.
Its prefix-n6, zero-radius-control hard-oracle row is 94,352 bytes,
`d_seg=0.003132290323264897`, `d_pose=150.166140238444`; training moved `d_seg` by exactly zero.
It is explicitly non-n600, non-score, and non-promotion. The new machine refusal receipt is
`.omx/research/m1_575_governed_launch_refusal_20260720.json`.

## Separately recovered exact row

A completed but never harvested governed n600 inverse click-polish run was still intact on the
Modal Volume. Its exact candidate is **177,169 bytes**, SHA-256
`cb6cf0ba719a535bf8874b31675a4ec66a893423d320f1e4071a2012cd88a56f`. The stock
`upstream/evaluate.py` report binds those exact bytes to 600 samples on CPU and displays
`d_seg=0.00054530`, `d_pose=0.00002931`, and `S=0.19`. The same-container full-precision selection
row on those same exact bytes is:

| term | value | status |
|---|---:|---|
| `d_seg` | 0.0005453067355847452 | MEASURED, same-container exact scorer |
| `d_pose` | 0.00002930838566754801 | MEASURED, same-container exact scorer |
| archive bytes | 177,169 | MEASURED, exact ZIP |
| `100*d_seg` | 0.05453067355847452 | DERIVED |
| `sqrt(10*d_pose)` | 0.017119692073033325 | DERIVED |
| rate term | 0.11796956486570198 | DERIVED |
| total `S` | **0.18961993049720982** | DERIVED from full-precision measured components |

This is `-0.001462893694889944` below the session-start contest-CPU pointer
`0.19108282419209976`. It is therefore a real pointer candidate, not a proxy. This branch does
**not** mutate `reports/latest.md`; MAIN must re-check custody and own pointer ingestion.

The verdict is deliberately narrow. This row solves more of the compact **B-side** inverse
problem. It does not compose A's scorer-plane distortion with B's compact description and is not
the requested joint optimum. The remaining `0.07164993049720982` above the stated `0.11797`
floor is distortion debt, not rate debt.

## What the inverse solve actually did

The preserved run `fc-01KX6DZWCHNPQ6KN59V2MZ845J` searched all 600 pairs over 28 pair-local
latent coordinates, two exact-gated rounds, using deltas `+1,-1`. It never changed archive size.

| round | accepted events | `S` | delta |
|---:|---:|---:|---:|
| 0 | 489 | 0.1901723118238698 | -0.0009275123682299657 vs run incumbent |
| 1 | 343 | 0.18961993049720982 | -0.0005523813266599675 vs round 0 |

There were 832 accepted events, 828 net changed codes, and 489 pairs touched. Runtime was
12,789.5 s total: 75.1 s GT build, 12,550.8 s search, and 160.9 s exact eval. The output raw
custody assertion passed at exactly 3,662,409,600 bytes. The accepted-click ledger and final
candidate are preserved on the SSD evidence tier, so the stale call classification did not erase
the result.

No training or paid dispatch was launched in this arm. The latest strict preflight values and its
rc=4 refusal are recorded above and in the machine receipt.

## Recursive-fractal KKT reading

The exact row localizes the remaining optimization problem:

- **archive/global:** the rate term is already `0.11796956486570198`, effectively the named
  `0.11797` floor. The archive has 39,131 bytes of headroom below the honest 216,300-byte box and
  87,151 below the 264,320-byte box.
- **pair/constant:** exact coordinate descent found monotone slack across 489/600 pairs and 828
  net stored codes. This is a measured local KKT correction at the compact payload coordinates.
- **pixel/class/boundary/frame:** those coordinates were not made explicit by this solver. The
  candidate remains `0.00039334673558474517` worse in `d_seg` than the A-capstone reference, while
  its `d_pose` is `0.00007253161433245199` better. The unresolved debt is therefore the SegNet
  cell/boundary field and its compact receiver pullback.
- **epoch/training:** #575 is the operator-routed vehicle, but the current launch instance cannot
  legally begin until the memory governor admits it and the positive receiver-closed band custody
  exists. The prior zero-band control already shows that repeating the control training is not the
  requested experiment.

The C2 positive band must carry the A-capstone scorer-plane constraints into a compact receiver
description: target/class/boundary debt must pull back through the real resize/uint8 receiver to
stored coordinates, then be exact-gated after byte-close. The current positive M1 band cannot
supply that map: the real n600
38,077-candidate Fisher/secant/QP EV field, scorer-free PDW2 spatial/RGB pullback, and
receiver-bound curvelet/shearlet carrier remain absent. This is a **formulation-scoped blocker**,
not a death verdict on inverse joint optimization.

## Authority and provenance limits

The historical dispatch ledger did not record `mounted_code_git_head` or
`upstream_snapshot_sha256`. Those fields remain explicitly unknown; no hash is inferred. The
result nevertheless has stronger direct custody than the stale ledger suggested: exact archive
bytes/hash, Volume-resumable accepted-click ledger, same-container selection and stock evaluator,
full raw-byte assertion, runtime/microarchitecture, and durable SSD copies with hashes. MAIN must
decide whether the two legacy null provenance fields require a repeat exact eval before pointer
ingestion.

Borrowed-substrate accounting remains unchanged: the click-polish mechanism is a defensive-bank
method reference, while the payload/runtime is our PR110 lineage. This is not claimed as a new
representation or innovation.

## Triality and pointer delta

- **DSL:** the required C2 leg passed with typed policy SHA `d3226b536f84be62fad5a4b66a5dab41fc76a1a18c6f13c0b89aa1c3fd7477ef`
  and exact dedicated-trainer parser consumption. DSL is N/A only for the separate post-compile
  click-polish evidence edge. No invented flag or config surface was used.
- **Equation:** consume the existing
  `clickpolish_exact_gated_discrete_latent_ratchet_v1` plus the canonical contest score law. No new
  equation is minted from one recovered row.
- **DAG:** the companion feed records
  both the refused C2 fire edge and the separate
  `compact archive -> pair-local inverse solve -> byte-exact repack -> same-container exact eval -> MAIN pointer review`
  evidence edge.
- **Pointer:** candidate delta `-0.001462893694889944`; pointer unchanged in this branch; MAIN
  landing review required.

## STORES CONSULTED

`reports/latest.md`; `.omx/state/lane_registry.json`;
`.omx/state/subagent_progress.jsonl`; `.omx/state/modal_call_id_ledger.jsonl`;
`.omx/state/active_lane_dispatch_claims.md`;
`.omx/research/clickpolish_pr110_phase2_modal_runbook_20260710.md`;
`.omx/research/click_polish_399_20260711T2200Z.md`; the Task #575 authority file; and the harvested
SSD evidence directory.

Machine-readable custody and all exact hashes:
`.omx/research/m1_575_governed_launch_refusal_20260720.json` and
`.omx/research/joint_optimum_575_xhigh_exact_row_20260720.json`.

## Review closure

One read-only adversarial round was run with `gpt-5.6-sol` at `high` reasoning. Verdict:
**PASS — no actionable findings**. The reviewer independently checked the 09:21:49 correction,
the DSL/single-flight/memory/no-launch chain, eight resumable partial checkpoints, archive hashes
and score arithmetic, rounded-versus-full-precision separation, legacy provenance nulls, pointer
non-mutation, and the canonical `in_progress -> blocked` task transition. No recursive review was
run.
