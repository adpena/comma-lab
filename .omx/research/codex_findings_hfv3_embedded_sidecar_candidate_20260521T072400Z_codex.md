# Codex Findings: HFV3 Embedded Sidecar Candidate

- timestamp_utc: 2026-05-21T07:24:00Z
- lane: hfv3_embedded_pr101_hfv1_sidecar
- status: LANDED_SINGLE_MEMBER_BYTE_CLOSED_CANDIDATE_WITH_FULL_PARITY
- score_claim: false
- promotion_eligible: false
- exact_eval_executed: false

## What landed

New builder:

- `tools/build_hfv3_embedded_sidecar_candidate.py`

Updated proof tool:

- `tools/prove_hfv2_sparse_inflate_parity.py`

HFV2 proved the dense HFV1 foveation table was pair-sparse. This pass found the
next rate reduction: all 16 active sparse pairs share the same non-default
five-float row. HFV3 encodes the default row, one repeated active row, and the
16 active pair indices, then embeds that 90-byte payload at the end of member
`x`. The archive stays single-member and removes the second ZIP member overhead.

## Candidate artifact

- Output directory: `experiments/results/hfv3_embedded_sidecar_candidate_20260521T072238Z`
- Archive: `experiments/results/hfv3_embedded_sidecar_candidate_20260521T072238Z/archive.zip`
- Submission runtime: `experiments/results/hfv3_embedded_sidecar_candidate_20260521T072238Z/submission_dir_hfv3_embedded`
- Manifest: `experiments/results/hfv3_embedded_sidecar_candidate_20260521T072238Z/hfv3_embedded_manifest.json`
- Paired dispatch plan: `experiments/results/hfv3_embedded_sidecar_candidate_20260521T072238Z/paired_dispatch_plan.json`

Hashes:

```text
06d0a3b41c353a5bafa5a6310a6d9418089e42e97396b9041d23d08aca337c64  archive.zip
6916406d23b30e6772248e466f65028e2f9c371996f8a94968cb9a276fbab694  hfv3_embedded_manifest.json
18fdc360f6ef50cf10ef81ce22cc0bf6b9de8b63b460d4e78d54356312a55243  submission_dir_hfv3_embedded/archive_manifest.json
0e230fb5bbe0118f0bd8ae891711a7909b8b37210506bde0298f7ca2be159e69  submission_dir_hfv3_embedded/inflate.py
9cd2ac4fa4987ba5a7d40379470fdabe4fd453cf097c90f0e899d17f34c2c903  paired_dispatch_plan.json
```

## Byte result

```text
dense HFV1 archive bytes        202649
HFV2 two-member archive bytes   179025
HFV3 embedded archive bytes     178607
FEC6/PR110 baseline bytes       178517
```

HFV3 rate deltas:

```text
bytes saved vs dense HFV1       24042
bytes saved vs HFV2 two-member  418
bytes over FEC6/PR110 baseline  90
rate delta vs FEC6/PR110        0.000059927305781
```

The required component gain to beat FEC6/PR110 on CPU-axis rate arithmetic is
now about `5.99e-05`, down from `0.0160685082567` for dense HFV1 and
`0.000338256348186` for HFV2.

## Full parity proof

- Output directory: `experiments/results/hfv3_embedded_inflate_parity_20260521T072248Z`
- JSON: `experiments/results/hfv3_embedded_inflate_parity_20260521T072248Z/sparse_foveation_inflate_parity.json`
- Markdown: `experiments/results/hfv3_embedded_inflate_parity_20260521T072248Z/sparse_foveation_inflate_parity.md`

Hashes:

```text
c4bf3f738abe348596ff9cda9896c15bcc77eebee37b44f63f3eb6ffa7d68b03  sparse_foveation_inflate_parity.json
1eb416cd70f16971b581e3f745cd789b01d13c07ba5650bcf97bc44eb2dc94c3  sparse_foveation_inflate_parity.md
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
```

This proves the embedded HFV3 candidate is decoder-output-identical to the
dense HFV1 hardpair candidate while charging only 90 additional archive bytes
over the current FEC6/PR110 packet.

## Dispatch plan

Plan-only paired Modal dispatch was materialized and did not execute Modal.

Plan result:

- archive SHA matched expected
- no existing promotable CPU anchor was found
- no existing promotable CUDA anchor was found
- CPU lane: `hfv3_embedded_pr101_hfv1_sidecar_exact_eval_contest_cpu`
- CUDA lane: `hfv3_embedded_pr101_hfv1_sidecar_exact_eval_contest_cuda`
- CPU runtime tree: `dd726970a4f5c5a530191c791fb47fd3d58059437437f83bc5e80afc36f227a4`
- CUDA runtime tree: `39fddbfe5b9b918a9ed9e6f94b351fc7e472bfa0d8940e8da3da857952bf3922`
- runtime content tree both axes: `a6ca37a995d32d13014e13119c7bc1d6cf767ec56c5ae8ab5624c5b45a85afe9`

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tools/build_hfv3_embedded_sidecar_candidate.py \
  tools/prove_hfv2_sparse_inflate_parity.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_hfv3_embedded_sidecar_candidate.py \
  --output-dir experiments/results/hfv3_embedded_sidecar_candidate_20260521T072238Z

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/prove_hfv2_sparse_inflate_parity.py \
  --sparse-archive experiments/results/hfv3_embedded_sidecar_candidate_20260521T072238Z/archive.zip \
  --sparse-submission-dir experiments/results/hfv3_embedded_sidecar_candidate_20260521T072238Z/submission_dir_hfv3_embedded \
  --batch-pairs 8 \
  --output-dir experiments/results/hfv3_embedded_inflate_parity_20260521T072248Z

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py scan

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file \
  tools/build_hfv3_embedded_sidecar_candidate.py --status reviewed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file \
  tools/prove_hfv2_sparse_inflate_parity.py --status reviewed
```

Review tracker result:

- `tools/build_hfv3_embedded_sidecar_candidate.py`: 21 entities reviewed; policy NORMAL, 0 violations
- `tools/prove_hfv2_sparse_inflate_parity.py`: 17 entities reviewed; policy NORMAL, 0 violations

## Current blocker

I did not execute the paired Modal exact eval because the Claude-owned DP1
paired claims remain active in `tools/claim_lane_dispatch.py summary`. The HFV3
packet is byte-closed, full-parity proven, and plan-ready. Once the dispatch
surface clears, HFV3 should supersede HFV2 as the exact-eval candidate.
