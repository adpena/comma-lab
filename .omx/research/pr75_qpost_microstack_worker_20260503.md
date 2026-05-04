# PR75 QPost Micro-Stack Worker - 2026-05-03

Evidence grade: `byte_trace_planning_only_until_exact_cuda`.
Score claim: `false`.
Remote dispatch: `false`.

## Scope

Implemented a local-only C-089 PR75/QP1/P6 candidate builder that combines:

1. a decoded-stream-preserving P6 Brotli resweep of the current A++ frontier;
2. a charged PR65/Henosis bias-only qpost top-32 sidecar selected from
   positive public-trace pair opportunity.

No remote GPU job was claimed or dispatched. Exact score truth still requires
`archive.zip -> inflate.sh -> upstream/evaluate.py` through
`experiments/contest_auth_eval.py --device cuda` after a dispatch lane claim.

## Inputs

- Source C-089 frontier archive:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip`
- Source bytes: `276342`
- Source SHA-256:
  `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`
- Source exact score: `0.3154707273953505`
- PR65 qpost source archive:
  `experiments/results/top_submission_delta_reverse_engineering_20260503/sources/pr65_henosis_archive.zip`
- PR65 SHA-256:
  `b331cb4f6df9d8929db966b943b8c73624cdf3b6db71acbde361570852e59e68`
- Ranking traces:
  C-089 component trace minus PR65 compatibility component trace, filtered to
  PR65 bias-active positive-opportunity pairs.

## Code Landed

- `experiments/build_pr75_qpost_microstack_candidate.py`
- `src/tac/tests/test_build_pr75_qpost_microstack_candidate.py`

The builder fails closed on:

- source archive SHA drift;
- PR65 archive SHA drift;
- non-P6 or runtime-unparseable source payloads;
- missing `qpost.bin` runtime hook in `inflate.sh`;
- non-bias qpost streams, until post/region/motion risk gets a reviewed guard;
- qpost no-op selected subsets;
- insufficient positive-trace qpost-active pairs.

## Local Build

Command:

```text
.venv/bin/python experiments/build_pr75_qpost_microstack_candidate.py --output-dir experiments/results/pr75_qpost_microstack_worker_20260503
```

Artifacts:

- Summary:
  `experiments/results/pr75_qpost_microstack_worker_20260503/candidate_summary.json`
- Resweep manifest:
  `experiments/results/pr75_qpost_microstack_worker_20260503/c089_p6_lossless_resweep/manifest.json`
- Candidate manifest:
  `experiments/results/pr75_qpost_microstack_worker_20260503/c089_p6_resweep_pr65_qpost_bias_top032/manifest.json`

## Candidate

- Candidate id: `c089_p6_resweep_pr65_qpost_bias_top032`
- Archive:
  `experiments/results/pr75_qpost_microstack_worker_20260503/c089_p6_resweep_pr65_qpost_bias_top032/archive.zip`
- Bytes: `276542`
- SHA-256:
  `3816340f572d21df56fa0ca00e64f51ae8e7f6b7556353d69c7a1453e6d3051f`
- Delta vs C-089 source: `+200` bytes
- Formula rate delta vs source: `0.0001331717906244343`
- Public-trace opportunity bound: `0.0017422633563209306`
- Sub-0.314 break-even component gain after rate: `0.0016038991859749138`
- Trace-bound score if realized: `0.313861635829654`

This is not a score claim. It is a local exact-eval candidate because the
byte/trace bound clears break-even and the archive is byte-closed.

## Safety And Preflight

- Source P6 resweep is decoded-stream preserving.
- Resweep archive: `276333` bytes, SHA-256
  `3de0d1546c909404df2f9b40a9ab8218100be36650b6bfcb3132bac50400ec7f`.
- Resweep changed only Brotli encodings:
  - `masks.mkv`: `219472 -> 219465` bytes.
  - `seg_tile_actions.delta_varint`: `116 -> 115` bytes.
  - `optimized_poses.qp1`: `677 -> 676` bytes.
  - `renderer.bin`: unchanged `55965` bytes.
- Runtime parser confirms identical decoded members for masks, renderer,
  seg-tile actions, and QP1 pose.
- Final archive members are exactly `p` and `qpost.bin`.
- `qpost.bin` is `115` bytes; active stream is `bias` only.
- Selected qpost active atoms: `32` across `32` pairs.
- Non-selected qpost pairs default to identity.
- `randmulti` is omitted and marked unsupported for pair filtering.
- `inflate.sh` qpost runtime hook was present.

## Next Dispatch Draft

Run only after a successful lane claim; do not skip the claim row.

```text
ETA_UTC="$(date -u -v+45M +%Y-%m-%dT%H:%MZ)"
JOB_ID="exact_eval_pr75_qpost_microstack_bias032_t4_$(date -u +%Y%m%dT%H%MZ)"
tools/claim_lane_dispatch.py claim --lane-id pr75_qpost_microstack_bias032_c089p6 --platform lightning --instance-job-id "$JOB_ID" --agent codex:gpt-5.5 --predicted-eta-utc "$ETA_UTC" --status eval --notes "C089 P6 lossless resweep + PR65 bias top32 qpost; archive_sha256=3816340f572d21df56fa0ca00e64f51ae8e7f6b7556353d69c7a1453e6d3051f"
.venv/bin/python -u experiments/contest_auth_eval.py --archive /Users/adpena/Projects/pact/experiments/results/pr75_qpost_microstack_worker_20260503/c089_p6_resweep_pr65_qpost_bias_top032/archive.zip --inflate-sh submissions/robust_current/inflate.sh --upstream-dir upstream --device cuda --keep-work-dir --work-dir /Users/adpena/Projects/pact/experiments/results/pr75_qpost_microstack_worker_20260503/exact_eval_work/c089_p6_resweep_pr65_qpost_bias_top032
```

## Verification

```text
.venv/bin/python -m py_compile experiments/build_pr75_qpost_microstack_candidate.py src/tac/tests/test_build_pr75_qpost_microstack_candidate.py
.venv/bin/python -m pytest src/tac/tests/test_build_pr75_qpost_microstack_candidate.py -q
3 passed in 0.11s
git diff --check -- experiments/build_pr75_qpost_microstack_candidate.py src/tac/tests/test_build_pr75_qpost_microstack_candidate.py .omx/research/pr75_qpost_microstack_worker_20260503.md
```

Additional no-index whitespace check on new text artifacts passed.
