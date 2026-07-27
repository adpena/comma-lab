# Canonical equations note — DDM WF7 stream granularity

Date: 2026-07-25
Scope: exact seven-home lossless recoding instance; no contest-score or family
closure.

## 1. Physical-home conservation

For exact seeded state bytes `A` and its seven disjoint physical homes `H_i`:

`A = H_0 || H_1 || ... || H_6`

and

`|A| = sum_i |H_i| = 134211`.

The LP1 logical lane-seed delta is a lineage attribution, not a contiguous
wire home:

`270 = delta_manifest + |H_lane| + delta_CD`

`270 = 34 + 155 + 81`.

This equality must remain separate from the physical partition.

## 2. Lossless stream price

For home codec `C_i` with exact decoder `D_i`:

`D_i(C_i(H_i)) = H_i`.

The WF7 container `W` has a counted directory `K`:

`|W| = K + sum_i |C_i(H_i)|`.

Measured here:

`K = 21`,

`sum_i (|C_i(H_i)| - |H_i|) = -1797`,

so

`delta_bytes_WF7 = 21 - 1797 = -1776`.

The exact rate action is:

`delta_S_rate = 25 * delta_bytes / 37545489`

`delta_S_rate = -0.0011825655007449763`.

## 3. Deterministic receiver identity

If the full container receiver restores the exact state,

`decode_WF7(W) = A`,

then for any deterministic downstream receiver/scorer map `F`,

`F(decode_WF7(W)) = F(A)`.

Therefore:

`delta_d_seg = 0` and `delta_d_pose = 0`.

This identity does not itself establish that `W` is an E4 or contest packet.
That requires a separately materialized runtime binding.

## 4. Granularity nonadditivity

For alternative descriptions `W_g` of the same semantic pool at different
granularities `g`, compare their complete receiver-closed objective values:

`S_g = 100 d_seg,g + sqrt(10 d_pose,g) + 25 B_g / 37545489`.

Do not form:

`delta_S_total = delta_S_CC3 + delta_S_WF7 + delta_S_PF3`.

Those deltas are measured on different exact objects or competing
factorizations. They may falsify an empty-price premise, but they add only
after one same-object joint materialization proves composition.

## 5. Box admission

A row is a #613 member iff all are true:

`d_seg <= 0.00116`,

`d_pose <= 0.00161`,

`archive_bytes <= 200000`.

Lossless WF7 recoding can change only the third inequality. Since the current
C1, CC3, and E4 endpoints violate both distortion inequalities, the cheapest
measured box member is `NULL`, not zero.

MAIN landing review required.
