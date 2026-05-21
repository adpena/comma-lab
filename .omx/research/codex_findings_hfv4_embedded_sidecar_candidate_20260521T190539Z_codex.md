# Codex Findings: HFV4 Embedded Delta Sidecar Candidate

- timestamp_utc: 2026-05-21T19:05:39Z
- lane: hfv4_embedded_pr101_hfv1_sidecar
- status: LANDED_SINGLE_MEMBER_BYTE_CLOSED_CANDIDATE_WITH_FULL_PARITY
- score_claim: false
- promotion_eligible: false
- exact_eval_executed: false

## What landed

New builder:

- `tools/build_hfv4_embedded_sidecar_candidate.py`

Updated proof tool:

- `tools/prove_hfv2_sparse_inflate_parity.py`

HFV4 takes the HFV3 repeated-row embedded sidecar and removes the dense default
row plus fixed-width active-pair indices. The active row remains archive-charged
as five float32 values; the sorted active-pair list is delta-uvarint encoded.
The omitted default row is transform-no-op at runtime because alpha is zero, but
the builder still proves exact dense-row reconstruction by reintroducing the
original dense HFV1 default row during its row-parity check.

## Candidate artifact

- Output directory: `experiments/results/hfv4_embedded_sidecar_candidate_20260521T190342Z`
- Archive: `experiments/results/hfv4_embedded_sidecar_candidate_20260521T190342Z/archive.zip`
- Submission runtime: `experiments/results/hfv4_embedded_sidecar_candidate_20260521T190342Z/submission_dir_hfv4_embedded`
- Manifest: `experiments/results/hfv4_embedded_sidecar_candidate_20260521T190342Z/hfv4_embedded_manifest.json`
- Paired dispatch plan: `experiments/results/hfv4_embedded_sidecar_candidate_20260521T190342Z/paired_dispatch_plan.json`

Hashes:

```text
38ab82d1e92af88c102d2ba0f64dbb4b6ff9c875b6340cb766cf77e63d8e9fd6  archive.zip
2bfc33e19b9620aa819c804eece4c536be8c2287df2d2acf6c750e48cb16b6e7  hfv4_embedded_manifest.json
5d0d2ab34ad71d095685afe41b46a4684ceb8e6eead1c95adc41340e08a5ea31  submission_dir_hfv4_embedded/archive_manifest.json
e8f07c2d2d3e2f86bc2e99b813e3911d0fd49bdd85ada03bf896caa9439e87fc  submission_dir_hfv4_embedded/inflate.py
f5b1f2e189a0869def65942d98e4e61aec08e59577a9296a041566f43062274d  paired_dispatch_plan.json
```

## Byte result

```text
dense HFV1 archive bytes        202649
HFV2 two-member archive bytes   179025
HFV3 embedded archive bytes     178607
HFV4 embedded archive bytes     178560
FEC6/PR110 baseline bytes       178517
```

HFV4 rate deltas:

```text
bytes saved vs dense HFV1       24089
bytes saved vs HFV3 embedded    47
bytes over FEC6/PR110 baseline  43
rate delta vs FEC6/PR110        0.0000286319349843
```

The required component gain to beat FEC6/PR110 on CPU-axis rate arithmetic is
now about `2.86e-05`, down from `5.99e-05` for HFV3 and `3.38e-04` for HFV2.

## HFV4 payload anatomy

```text
HFV4 payload bytes              43
header bytes                    25  (<4sBfffff)
delta-uvarint pair bytes        18
active sparse pairs             16
sidecar SHA-256                 24ecf817545d20f921bf758e982fbbc3298d144c65a3f77af2ead4ba43754b8c
```

ZIP anatomy:

```text
archive bytes                   178560
members                         1
member name                     x
member compression              stored
member compressed bytes         178460
member uncompressed bytes       178460
central directory bytes         47
```

## Full parity proof

- Output directory: `experiments/results/hfv4_embedded_inflate_parity_20260521T190406Z`
- JSON: `experiments/results/hfv4_embedded_inflate_parity_20260521T190406Z/sparse_foveation_inflate_parity.json`
- Markdown: `experiments/results/hfv4_embedded_inflate_parity_20260521T190406Z/sparse_foveation_inflate_parity.md`

Hashes:

```text
9e865d4ae03ced3303c740a429d28ad90d5f59c72bf0bcb3216456a69cbdf520  sparse_foveation_inflate_parity.json
7f5457a462ec38a7c7d1404a56ca8dd4294440a95b3d32d6977551bd7f19755a  sparse_foveation_inflate_parity.md
```

Proof result:

```text
pairs checked              600 / 600
frames checked             1200
dense_output_sha256        6d27ea355e0e1dbc66fbb9f1df4c31d28faa52e1afae58231ce0185511ae81d2
sparse_output_sha256       6d27ea355e0e1dbc66fbb9f1df4c31d28faa52e1afae58231ce0185511ae81d2
output_sha256_match        true
tensor_equal               true
max_abs_diff               0.0
mismatched_batches         0
sparse_sidecar_name        embedded_foveation_params.hfv4
```

This proves the embedded HFV4 candidate is decoder-output-identical to the
dense HFV1 hardpair candidate while charging only 43 additional archive bytes
over the current FEC6/PR110 packet.

## Dispatch plan

Plan-only paired Modal dispatch was materialized and did not execute Modal.

Plan result:

- archive SHA matched expected
- no existing promotable CPU anchor was found
- no existing promotable CUDA anchor was found
- pair group: `pair_hfv4_embedded_pr101_hfv1_sidecar_exact_eval_38ab82d1e92a`
- CPU/CUDA execute commands are recorded in `paired_dispatch_plan.json`

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tools/build_hfv4_embedded_sidecar_candidate.py \
  tools/prove_hfv2_sparse_inflate_parity.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_hfv4_embedded_sidecar_candidate.py \
  --output-dir experiments/results/hfv4_embedded_sidecar_candidate_20260521T190342Z

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/prove_hfv2_sparse_inflate_parity.py \
  --sparse-archive experiments/results/hfv4_embedded_sidecar_candidate_20260521T190342Z/archive.zip \
  --sparse-submission-dir experiments/results/hfv4_embedded_sidecar_candidate_20260521T190342Z/submission_dir_hfv4_embedded \
  --batch-pairs 8 \
  --output-dir experiments/results/hfv4_embedded_inflate_parity_20260521T190406Z

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py scan

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file \
  tools/build_hfv4_embedded_sidecar_candidate.py --status reviewed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file \
  tools/prove_hfv2_sparse_inflate_parity.py --status reviewed
```

Review tracker result:

- `tools/build_hfv4_embedded_sidecar_candidate.py`: 25 entities reviewed
- `tools/prove_hfv2_sparse_inflate_parity.py`: 17 entities reviewed

## Current blocker

I did not execute the paired Modal exact eval because `tools/claim_lane_dispatch.py
summary` still reports 13 active dispatch claims, including DP1 paired CPU/CUDA
auth-eval calls and NSCS06/Selfcomp Modal jobs. HFV4 is byte-closed,
full-parity proven, and plan-ready. Once the dispatch surface clears, HFV4
should supersede HFV3 as the exact-eval candidate because it cuts the remaining
rate hurdle from `5.99e-05` to `2.86e-05`.
