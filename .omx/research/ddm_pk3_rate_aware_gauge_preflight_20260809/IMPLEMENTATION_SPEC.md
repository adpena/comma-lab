# DDM PK3 receiver-aware rate-gauge preflight implementation spec

## Decision and authority

This arm is scorer-free. It launches no training, scorer, or Modal job and makes no
score claim. It decides only the PK2 reopening trigger on the frozen PR130 CPR1
instance:

- MET only when one parsed candidate saves at least 2,000 bytes in the actual full
  XZ+ZIP archive and its directly measured full-population receiver-product MSE is
  strictly below `2.5e-6`;
- otherwise NOT MET, closing the reopening on this pinned instance and declared
  search surface without claiming to refute arbitrary `GL(12)`, QAT, retraining, or
  another wire.

The metric is the shipped receiver surface
`C @ inflate.normalized_basis(B) / sqrt(12)`, not PK2's low-resolution raw `C @ B`
diagnostic. Every trigger-bearing MSE is a chunked direct product over all 600
coefficient rows and all `3 * 384 * 512` normalized-basis values per row. A
float64 Gram contraction of the same pinned fp32 normalized bases is only a search
measurement; it cannot carry the trigger without the direct same-row check.

## Frozen inputs and copy-out

- archive: `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`,
  191,052 bytes;
- intake commit: `e34f31bc4969042c0051ac81aa3c56884419a231`;
- PK2 reuse commit: `cfddfc503a76e72ec7654ac9e69ff2acdba439b1`;
- PK2 runner: `634ac0d899925c5c34df24bfc5efa68f99bd48bf74c2d7810e6d3152e55d9c7e`;
- `carrier_codec.py`: `d2f14402374b4e622b7f981d736389fb04f0ca0165180e4c75f3a32ffe996bed`;
- `inflate.py`: `335369c9b3b295707f1790feb0b5b7ae288338fae350056cc4bb03aaa18f0c9e`;
- `hpac_integer.py`: `6e6b4f4d0b293fb60cc1b751958756a4cd6c2ce7bcff68c6f03e20277856803f`;
- `hpac_integer_sparse.py`: `2240ee32c53fe949b560d316d349e0bbdccc0ceb78787307cd4d530623d42a0c`;
- `integer_model_io.py`: `6f91c91ed4785d203aa3570af362fbe9c6a64bb2249599f8554adb31174b80a5`.

The runner copies the archive and five-file runtime into
`/Volumes/VertigoDataTier/pact/ddm_pk3_20260809/receiver_v7/retained/inputs/`,
refuses a mismatched existing copy, records a manifest, and imports the receiver
only from that retained copy. It also copies its own exact source, binds that hash
into every candidate identity, and fails if the source changes during a run. The
intake is read-only.

The frozen raw CPR1 section is 23,054 bytes: 152 fixed bytes, 104,135 Huffman
bits (13,017 bytes), and 79,076 Rice bits (9,885 bytes). The charter's 23,384
bytes is a distinct measured full-archive leave-one-out marginal attributed to
pose. These are not competing measurements; the trigger uses actual candidate
full-archive deltas.

## Correct gauge law

For parsed raw basis `A`, coefficients `C`, and invertible mixing `H`, let `r(A)`
be the row RMS after the pinned bicubic upsample and mean removal. The prequantized
coarse target and coefficient target are

`A' = H diag(r(A)^-1) A`

`C' = C H^-1 diag(r(A'))`.

Then the normalized prequantized receiver product is unchanged. Independent real
int5/int12 projection makes non-monomial candidates lossy, so they are measured
after CPR1 encode/decode. Signs, permutations, positive basis-only radial scaling,
and a single constant integer shift across all channels and spatial positions of a
basis row are exact receiver gauges, subject to signed-int5 range and direct fp32
validation. Reciprocal coefficient scaling is forbidden because receiver basis
normalization already removes raw basis magnitude.

## Declared bank

The bank contains at most 452 unique quantized states, including control:

- every singleton sign, a 64-state cube over the six best singleton dimensions,
  and global sign anchors;
- legal coordinate DC shifts from `{-1,+1,minimum,maximum}` plus a 16-state cube
  over the four best unique dimensions;
- all 66 pairwise permutations;
- unit, nearest-power-of-two, and common-geometric-mean positive basis-scale
  canonicalizations, with coefficients unchanged;
- all 66 normalization-aware Givens supports at `+/-1/128`, then the best eight
  supports at `+/-{1/256,1/64,1/32}`;
- all 132 directed shear supports screened by a deterministic coefficient-delta
  correlation minus basis-Gram penalty, with only the top eight materialized at
  `+/-{1/512,1/256,1/128,1/64}`;
- coefficient-delta KLT and receiver-Gram-orthogonalized KLT homotopies at
  `{1/64,1/32,1/16,1/8}`.

Selection is stagewise minimum actual full-archive bytes subject to receiver MSE
below the strict bar. Screens are not candidates and do not enter the denominator.
Array hashes deduplicate all materialized states. Every materialized state retains
both `carrier.cpr1` and `archive.zip` before its row is reported.

## Controls and failure conditions

- identity must reproduce carrier 23,054 bytes, archive 191,052 bytes, SHA
  `0491...`, 104,135 Huffman bits, and 79,076 Rice bits;
- sign, permutation, legal global DC, and basis-only radial controls must preserve
  the actual normalized receiver product within pinned fp32 tolerance;
- raw-space Givens and reciprocal basis/coefficient scaling must fail actual
  receiver invariance even when raw `C @ B` is near-identical;
- permutation must preserve both inner bit totals;
- final basis and coefficient sentinels must change hashes and receiver MSE;
- Gram contraction must match a direct product control, and every verdict-bearing
  row receives the direct full-population check, and every materialized candidate
  records that direct MSE rather than an analytical estimate;
- trigger comparisons are strict and same-row: 1,999 bytes fails, MSE equal to the
  bar fails, and no byte/MSE splice is allowed;
- condition number above four, near-zero row RMS, code-range failure, parse-back
  mismatch, retained-payload drift, input drift, or more than 452 rows fails closed;
- selected carrier and archive are regenerated and must be byte-identical.

## Invalid preserved run

Before this correction, an unknown concurrent workspace actor launched the initial
untracked runner into `/Volumes/VertigoDataTier/pact/ddm_pk3_20260809/`. Those
payloads are preserved under the always-keep rule but are explicitly
`INVALID_FOR_TRIGGER_RAW_PRODUCT_V1`: that script used raw 24x32 `C @ B`, reciprocal
coefficient scaling, and uncorrected raw mixing. The first receiver-aware
`receiver_v2/` pass is preserved but excluded because its NumPy matrix-product
contraction emitted warnings. The warning-free `receiver_v3/` pass is also
preserved but superseded because its cache identity did not bind the PK3 runner
source and its generated prose conflated the raw CPR1 section with the distinct
archive marginal. `receiver_v4/` and `receiver_v6/` aborted on source drift;
`receiver_v5/` was superseded because only verdict-bearing rows had direct MSE.
The valid run uses isolated `receiver_v7/` identities and never reuses a prior
receipt.

## QAT disposition

If NOT MET, no ticket is emitted and the existing stable PK2 action is
`FOLDED` with reason `TRIGGER_FAILED`; no scorer or training action remains. If MET, PK3 seals a
rate-aware, exact-receiver, base-hash-pinned ticket with optimizer/scheduler/RNG/
order/cursor resume state, atomic periodic plus distinct stage checkpoints, SSD
retention, and a seeded stratified n120 first fire. PK3 still launches nothing and
does not mint a duplicate task.

## Final receiver-v7 hardening handoff

The implementation owner must leave `DEVELOPMENT_RUN_BLOCKED = True`; the primary
agent alone removes that one-line guard after review. That one-line removal freezes
the measured source hash, so the exact hash-bound verification receipt is prepared
after the removal and before measurement, with no later source or test edit. Use a
fresh `receiver_v7/` output and metric identity. Preserve and exclude
receiver-v6 as a source-drift/development run; do not delete or reuse any older
payload.

Before the seal:

- validate `--out-dir` as a proper descendant of only
  `/Volumes/VertigoDataTier/pact/ddm_pk3_20260809/`, explicitly rejecting the arm
  root, local paths, `/tmp`, and live `ddm_cx2`/`ddm_tm1` stores;
- treat `--resume-from` as deterministic candidate-cache replay, not cursor resume:
  it must stay inside the selected output, and a complete progress file must not
  bypass revalidation of copied inputs, all candidate carrier/archive bytes,
  parse-back arrays/scales, direct MSE, rows, repeat payloads, or final artifacts;
- on cache reuse, constrain receipt paths to the candidate directory, reconstruct
  the exact carrier and outer archive, compare retained bytes and hashes, reparse
  codes and float32 scales, and recompute the direct full-population receiver MSE;
- assert every materialized row has a non-null direct MSE over exactly
  `600 * 3 * 384 * 512 = 353,894,400` values before selection or verdict;
- retain an exact pytest log inside `receiver_v7/verification/`, validate it is
  inside the arm output and has a passing summary, and record its path, byte size,
  SHA-256, command, and passed count in the final receipt;
- run the source-drift guard before any QAT ticket or durable verdict write. A MET
  ticket lives under retained SSD custody and is path/size/SHA-bound. A NOT_MET
  result has `sealed_qat_ticket: null`, `future_action: null`, and folds the one
  existing `ddm_pk2_rate_aware_gauge_qat` action using queue lines 106--108 as
  provenance duplicates, never as three new tasks;
- record the exact bounded recall query commands, their preimplementation file
  counts, correct harness-bridge path, findings, and concrete plan changes;
- describe search as a deterministic stagewise greedy chain, give the actual
  per-stage denominator and formal `<=452` bound, and explicitly exclude the
  exhaustive Cartesian product of cross-stage operations;
- for NOT_MET, omit `## NEXT_IF_RESUMED` entirely. Render `## LIVE-HYPOTHESES`
  followed by `## DEAD-ENDS`, and identify every preserved invalid/superseded run.

Do not touch any file outside this PK3 implementation, its test, and this PK3
research directory. Do not launch the runner, a scorer, training, Modal, or a Git
operation. Acceptance before handback:

```sh
.venv/bin/python -m pytest -q experiments/tests/test_ddm_pk3_rate_aware_gauge_preflight.py
.venv/bin/python -m py_compile experiments/ddm_pk3_rate_aware_gauge_preflight.py experiments/tests/test_ddm_pk3_rate_aware_gauge_preflight.py
```
