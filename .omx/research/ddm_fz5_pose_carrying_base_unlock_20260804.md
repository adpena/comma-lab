---
title: "ddm_fz5 pose-carrying-base unlock: cr2_ep854 repair remains population-wide, partial F0PR folds"
unit: ddm_fz5
date_utc: 2026-08-04
axis: "[macOS-CPU frozen-scorer advisory] for component census; byte-closed skip-eval for candidate custody"
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
own_vehicle_frontier: "S = 0.7541459 @ 358,084 B [macOS-CPU advisory]"
tokens: [no-triality, p0-ledger-ok]
---

# ddm_fz5 - cr2_ep854 plus frame_0 repair

## Answer

fz5 did **not** beat the own-vehicle frontier. The byte-closed candidate is:

| row | bytes | d_seg | d_pose | S | axis |
|---|---:|---:|---:|---:|---|
| `cr2_ep854_f0pr_k6_partial` | 292,026 | 0.00394407 | 35.4509349783 | 19.4172738012 | `[macOS-CPU frozen-scorer advisory]` prediction, byte-closed skip-eval |
| live best baseline | 358,084 | 0.00431179 | 0.00071459 | 0.7541459 | `[macOS-CPU advisory]` |
| delta, fz5 minus live best | -66,058 B | -0.00036772 | +35.4502203883 | +18.6631279012 | same caution |

The repair did not fail as a receiver mechanism: the candidate ledger closes,
inflate returns rc=0, and every selected repaired pair survived runner-true
acceptance. It fails economically because the #827 pose damage is population-wide:
repairing the 27 available k6 pairs moved mean d_pose only
`37.877063 -> 35.450935`.

## Custody

Located #827 archive:

| artifact | bytes | sha256 |
|---|---:|---|
| legacy `v4d_composed_cr2_ep854_archive.zip` | 285,529 | `6edf45fa5052949da2b5ba32e6b12f227e2f49f024606295d194defc668c789d` |
| base transcode `ep854_v3warp_base_archive.zip` | 283,636 | `fd50925899b22c7cd09fd7353b40ae3bf372266d107d586aa864508c0bb44904` |
| fz5 ix2 control | 286,411 | `c3d542c569ea8830a0e471d476ff59bd5160002ee6340b79fdf1ac1ae2fc1c60` |
| fz5 partial repair candidate | 292,026 | `b2527b2294f8f817a369a465766677728e448ea2c31b95cf5b9e223e08344fbc` |

The ix2 control was built by `tools/cx1_build_ix2_container_archive.py`; its
receipt verifies token, renderer, selector, pose-warp, beta, dim0, and st-grid
identity against the legacy #827 archive. This re-containerization is necessary
because the F0PR receiver hook is in the ix2 `0.bin` path; the six-member v3warp
path would not consume an appended F0PR section.

Decoded raw custody:

| raw | bytes | sha256 |
|---|---:|---|
| existing `v4d_cr2_ep854` raw | 3,662,409,600 | `352c11cf7607410491b3d0822cc29748fa0ab80d84387c38df02fb6ef9375370` |
| fz5 repaired raw | 3,662,409,600 | `42f7a6c92eacbb4de60416a3c41e0a929b5623d457bc8b2b56ea0eab24331b36` |

Byte-close receipt:
`/Volumes/VertigoDataTier/pact/ddm_fz5_20260804/cr2_ep854_f0pr_k6_partial/fz5_byteclose_skip_eval_receipt.json`.
It records `closes=true`, `residual_bytes=0`,
`payload_reencodes_identically=true`, shipped inflate rc=0, one raw file,
`3,662,409,600` raw bytes, and `202.66 s`. No full n600 evaluate was run because
`sg4.done` was absent and sg4 owns the scorer slot.

## Census

fz5 reran the PoseNet-only damage census on the decoded #827 frames:
`/Volumes/VertigoDataTier/pact/ddm_fz5_20260804/fz5_cr2_ep854_damage_census.json`.

| fact | value |
|---|---:|
| n | 600 |
| mean d_pose | 37.8770625083 |
| evaluator control | 37.87713242 |
| median | 7.0807065964 |
| p90 | 146.1478302002 |
| census sha256 | `0e3a9ca38c602c7cb755d6c7ff040fecb50c002564ff62bd96d6f91796053a34` |

The control closes to 7e-5 absolute against the existing n600 evaluator report,
so GT decode + PoseNet component scoring are trusted for selection arithmetic.

## Recipe Correction

The charter said `k=4 int16, per the fz4 recipe`. Direct parse-back of fz4's
shipped `sub_final` F0PR section refutes the `k=4` label:

| fz4 `sub_final` section | value |
|---|---:|
| `k` | 6 |
| repaired pairs | 21 |
| raw section bytes | 4,312 |
| bytes per pair raw | 216 |

The hot-state `k=4` wording is a documentation error. No k4 coefficient store
exists under `/Volumes/VertigoDataTier/pact/ddm_fz1_20260804`; the saved stores
are k6/k8 for #827 and k6 for pu2. fz5 therefore used the verified fz4 carriage
recipe, k6 F0PR, rather than fabricating k4 coefficients or claiming the fz4
section was k4.

## Candidate

Command family used:

```bash
.venv/bin/python experiments/ddm_fz1_compose_rowB.py \
  --payload /Volumes/VertigoDataTier/pact/ddm_fz5_20260804/cr2_ep854_ix2_control/0.bin \
  --coefs-npz /Volumes/VertigoDataTier/pact/ddm_fz1_20260804/fz1_kladder_coefs.npz,/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/fz1_ext_a_coefs.npz,/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/fz1_ext_b_coefs.npz \
  --probe-json /Volumes/VertigoDataTier/pact/ddm_fz1_20260804/fz1_kladder.json,/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/fz1_ext_a.json,/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/fz1_ext_b.json \
  --mapped-census /Volumes/VertigoDataTier/pact/ddm_fz5_20260804/fz5_cr2_ep854_damage_census.json \
  --seg-census /Volumes/VertigoDataTier/pact/ddm_fz5_20260804/fz5_cr2_ep854_damage_census.json \
  --runtime /Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/runtime \
  --dest /Volumes/VertigoDataTier/pact/ddm_fz5_20260804/cr2_ep854_f0pr_k6_partial \
  --gt-mkv /Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/videos/0.mkv \
  --k 6 --threads 4 --d-seg 0.00394407 --live-best-s 0.7541459
```

Selected repaired pairs:
`0, 1, 2, 3, 10, 16, 125, 163, 219, 232, 239, 241, 327, 406, 420, 428, 441, 448, 460, 470, 475, 500, 511, 512, 535, 589, 592`.

F0PR section: 5,662 B, sha
`bb777d34da74985fae56620fbf79bd5c6e8861fbf4a666f7fec0f4c5634eb21b`,
`k=6`, 27 repaired pairs. Runner-true acceptance matched solve values on every
selected pair and `seg_f1_only=True` for all selected pairs.

Component arithmetic:

| component | value |
|---|---:|
| seg term | 0.3944070000 |
| pose term | 18.8284186745 |
| rate term | 0.1944481266 |
| S | 19.4172738012 |

## Scorer Slot

`.omx/tmp/codex_runs/sg4.done` was absent, so fz5 did not run a full n600
evaluate. The owed scorer-batch section was appended to
`.omx/research/scorer_batch_20260804.md`. It marks this row as folded for
frontier movement and includes the exact fz2 command only for an explicit
negative calibration after sg4.

## Follow-ons

- **FIRED**: #827 archive custody; ix2 control parse-back; fz5 n600 component
  census; partial F0PR composition; canonical byte ledger; foreground inflate.
- **FOLDED**: `cr2_ep854 + available partial k6 F0PR` as a frontier candidate.
  Verdict scope: INSTANCE. It repairs the stored coefficient subset only; it
  does not kill full-population frame_0 repair.
- **QUEUED-WITH-FIRE-ORDER**: all-600 repaired cr2_ep854 row. Fire only if it
  first produces a byte-closed predicted `S < 0.7541459`. The existing F0PR
  stream has a single `k` in the section header, while the fz1/fz4 estimate
  called for mixed k6/k8; a successor must either choose one global k or land a
  receiver-closed mixed-k section before banking the mixed-k estimate.

## Pointer Honesty

Contest pointer `0.1910828242 [contest-CPU]` is unmoved. Own-vehicle frontier
remains `S = 0.7541459 @ 358,084 B [macOS-CPU advisory]`. fz5 does not beat it.
