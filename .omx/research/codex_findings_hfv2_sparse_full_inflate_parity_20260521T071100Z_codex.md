# Codex Findings: HFV2 Sparse Full Inflate Parity

- timestamp_utc: 2026-05-21T07:11:00Z
- lane: hfv2_pair_sparse_pr101_hfv1_sidecar
- status: LANDED_FULL_INFLATE_PARITY_PROOF
- score_claim: false
- promotion_eligible: false
- exact_eval_executed: false

## What landed

New proof tool:

- `tools/prove_hfv2_sparse_inflate_parity.py`

The tool replays the PR101/FEC6 inflate path in memory and compares the dense
HFV1 and sparse HFV2 postprocessed frame tensors batch-by-batch before raw bytes
would be written. This avoids creating multi-GB raw output files while still
hashing the exact frame bytes that the runtime would write.

## Full parity artifact

- Output directory: `experiments/results/hfv2_sparse_inflate_parity_20260521T070854Z`
- JSON: `experiments/results/hfv2_sparse_inflate_parity_20260521T070854Z/hfv2_sparse_inflate_parity.json`
- Markdown: `experiments/results/hfv2_sparse_inflate_parity_20260521T070854Z/hfv2_sparse_inflate_parity.md`

Artifact hashes:

```text
07defb99b3166bd218e538bc0c0c3e33d7d2a8b2d9642184b03fb73b19dbb74e  hfv2_sparse_inflate_parity.json
603a0d8208ea8d49f5d77021e1e95ca180909f0617410ae25e839c039664c22d  hfv2_sparse_inflate_parity.md
```

## Result

Coverage:

- dense archive: `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/archive_seed_top16_component_hardpairs/archive.zip`
- sparse archive: `experiments/results/hfv2_sparse_sidecar_candidate_20260521T070412Z/archive.zip`
- sparse runtime: `experiments/results/hfv2_sparse_sidecar_candidate_20260521T070412Z/submission_dir_hfv2`
- device: CPU
- pairs checked: all 600
- frames checked: 1,200
- batch size: 8 pairs

Byte identity:

```text
x_payload_sha256        f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd
dense_output_sha256     6d27ea355e0e1dbc66fbb9f1df4c31d28faa52e1afae58231ce0185511ae81d2
sparse_output_sha256    6d27ea355e0e1dbc66fbb9f1df4c31d28faa52e1afae58231ce0185511ae81d2
output_sha256_match     true
tensor_equal            true
max_abs_diff            0.0
mismatched_batches      0
```

This proves the HFV2 sparse sidecar is a byte-rate reduction of the dense HFV1
candidate, not a decoder-output change.

## Rate implication

The sparse candidate remains:

- archive bytes: 179,025
- archive SHA-256: `488f2e53d81d6442d189b4f882508af0d4184010ca67558e83bfadf822138ee2`
- bytes saved vs dense HFV1 candidate: 23,624
- bytes over FEC6/PR110 baseline: 508
- rate hurdle vs FEC6/PR110 baseline: `0.000338256348186`

Any component gain larger than `0.000338256348186` over FEC6/PR110 clears the
CPU-axis rate hurdle. Exact eval is still required before any score claim.

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tools/prove_hfv2_sparse_inflate_parity.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/prove_hfv2_sparse_inflate_parity.py \
  --pair-indices 0,64,79,126,508,546,599 \
  --batch-pairs 2 \
  --output-dir experiments/results/hfv2_sparse_inflate_parity_smoke_20260521T0710Z

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/prove_hfv2_sparse_inflate_parity.py \
  --batch-pairs 8 \
  --output-dir experiments/results/hfv2_sparse_inflate_parity_20260521T070854Z \
  > /tmp/hfv2_sparse_full_parity.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py scan

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file \
  tools/prove_hfv2_sparse_inflate_parity.py --status reviewed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check \
  tools/prove_hfv2_sparse_inflate_parity.py
```

Review tracker result:

- `tools/prove_hfv2_sparse_inflate_parity.py`: 16 entities reviewed
- policy: NORMAL, 16 entities compliant, 0 violations

## Current blocker

Paired Modal exact eval is still not launched from this turn because the
Claude-owned DP1 paired claims remain active in `tools/claim_lane_dispatch.py
summary`. The HFV2 sparse packet is now both plan-ready and full-inflate-parity
proven; the remaining frontier-moving action is execution of the already
materialized paired dispatch command once the dispatch surface clears.
