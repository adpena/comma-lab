# Codex Findings: HFV2 Sparse Sidecar Candidate

- timestamp_utc: 2026-05-21T06:59:00Z
- lane: hfv2_pair_sparse_pr101_hfv1_sidecar
- status: LANDED_BYTE_CLOSED_RECODE_CANDIDATE
- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false

## What landed

New tool:

- `tools/build_hfv1_sparse_sidecar_candidate.py`

The prior rate-hurdle audit showed that the dense HFV1 PR101 adapter paid a
+24,132-byte archive penalty over the FEC6/PR110 baseline, requiring about
`0.0160685` non-rate component improvement just to tie the baseline CPU-axis
score. This pass recodes the dense `HFV1` foveation sidecar into an exact
pair-sparse `HFV2` sidecar and emits a generated runtime copy that can decode it.

The generated packet is still research-only. It has no exact-eval score and
must not be promoted without a fresh exact eval plus strict submission gate.

## Artifact

- Output directory: `experiments/results/hfv2_sparse_sidecar_candidate_20260521T065416Z`
- Output archive: `experiments/results/hfv2_sparse_sidecar_candidate_20260521T065416Z/archive.zip`
- Output runtime: `experiments/results/hfv2_sparse_sidecar_candidate_20260521T065416Z/submission_dir_hfv2`
- Output manifest: `experiments/results/hfv2_sparse_sidecar_candidate_20260521T065416Z/hfv2_sparse_manifest.json`
- Generated submission manifest: `experiments/results/hfv2_sparse_sidecar_candidate_20260521T065416Z/submission_dir_hfv2/archive_manifest.json`

SHA-256:

```text
488f2e53d81d6442d189b4f882508af0d4184010ca67558e83bfadf822138ee2  archive.zip
6883da27eab3a135ba9aeff4a79892908db2748f59d007313c58f54b7ee3156d  hfv2_sparse_manifest.json
33075301213db60d0fdf343a11718bae9ec55c2427cf2ea84814bf455970ee67  submission_dir_hfv2/archive_manifest.json
15ff03ce73cea643e544fd56a8f0253b9ff9df71315608c1647e971d826fab0c  submission_dir_hfv2/inflate.py
```

Byte counts:

```text
179025  archive.zip
2493    hfv2_sparse_manifest.json
2508    submission_dir_hfv2/archive_manifest.json
```

## Result

The dense HFV1 sidecar was exactly pair-sparse:

- dense source archive bytes: 202,649
- dense `foveation_params.bin` bytes: 24,016
- sparse output archive bytes: 179,025
- sparse `foveation_params.hfv2` bytes: 390
- sparse pairs: 16
- bytes saved vs dense HFV1 archive: 23,624
- bytes delta vs FEC6 baseline archive: +508
- rate delta vs dense HFV1 archive: `-0.0157302519086`
- rate delta vs FEC6 baseline archive: `+0.000338256348186`

ZIP member shape:

```text
foveation_params.hfv2  stored 390 bytes
x                      stored 178417 bytes
```

The candidate collapses the HFV1 rate hurdle from `0.0160685082567` to about
`0.000338256348186`. It therefore converts the seed hardpair HFV1 candidate
from "needs a very large component gain" into "needs only a small component
gain to overcome the sidecar rate term."

## Verification

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_hfv1_sparse_sidecar_candidate.py \
  --output-dir experiments/results/hfv2_sparse_sidecar_candidate_20260521T065416Z \
  > /tmp/hfv2_sparse_manifest.json

zipinfo -v experiments/results/hfv2_sparse_sidecar_candidate_20260521T065416Z/archive.zip

shasum -a 256 \
  experiments/results/hfv2_sparse_sidecar_candidate_20260521T065416Z/archive.zip \
  experiments/results/hfv2_sparse_sidecar_candidate_20260521T065416Z/hfv2_sparse_manifest.json \
  experiments/results/hfv2_sparse_sidecar_candidate_20260521T065416Z/submission_dir_hfv2/archive_manifest.json \
  experiments/results/hfv2_sparse_sidecar_candidate_20260521T065416Z/submission_dir_hfv2/inflate.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tools/audit_hfv1_pr101_rate_hurdle.py \
  tools/build_hfv1_sparse_sidecar_candidate.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py scan

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file \
  tools/build_hfv1_sparse_sidecar_candidate.py --status reviewed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check \
  tools/build_hfv1_sparse_sidecar_candidate.py
```

Review tracker result:

- `tools/build_hfv1_sparse_sidecar_candidate.py`: 21 entities reviewed
- policy: NORMAL, 21 entities compliant, 0 violations

Runtime parity smoke:

- generated runtime loads `HFV2_pair_sparse`
- generated runtime sees `pair_count=16`
- payload-level row reconstruction parity is exact
- runtime row lookup intentionally returns `None` for default/no-op rows, so a
  naive runtime row comparison differs on default frames
- synthetic transform parity on selected active/default frames was exact:
  `max_abs_diff=0.0`

## Current blocker

I did not dispatch exact eval from this turn because the Modal ledger is already
dirty with partner activity and active DP1 claims are present. The artifact is
byte-closed and mechanically ready for a controlled exact-eval launch once the
operator/partner dispatch surface is clean.

## Recommended next action

Launch a paired CPU/CUDA exact eval for the sparse HFV2 packet before any
identity/nonidentity dense HFV1 runs. If the component term improves by more
than `0.000338256348186`, this packet beats the FEC6/PR110 CPU-axis baseline on
rate arithmetic.
