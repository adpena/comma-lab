# HNeRV Entropy Frontier Selection - 2026-05-07 Worker ENTROPY-FRONTIER

## Scope

Local entropy/frontier custody only. This pass did not dispatch, did not claim
score, and did not touch `submissions/pr103_pr106_final_runtime` or
`.omx/state` dispatch files.

## Selection Artifact

- JSON:
  `experiments/results/hnerv_entropy_frontier_selection_20260507_worker_entropy_frontier/selection.json`
- JSON SHA-256:
  `d98b507a75898b40f4d6e1ddf3ba1a01febf1aec47b0012a80cfe9e7d9bb8af3`
- Markdown:
  `experiments/results/hnerv_entropy_frontier_selection_20260507_worker_entropy_frontier/selection.md`
- Markdown SHA-256:
  `deb7ea62c43521ea180ca4b9ed00559547a92fce14a8ee4dc899ac1a8d69ba55`

## Active Excluded Candidate

The PR103-on-PR106 exact CUDA eval completed after the worker's first pass, so
the selector was hardened to treat the excluded PR103 archive as the active
rate-only byte floor.

- label: `active_pr103_pr106`
- archive bytes: `185578`
- archive SHA-256:
  `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`

## Selected Next Exact-Evaluable Artifact

None. HDM3 is static-packet-ready relative to its PR106x source archive, but it
is `186066` bytes, which is `488` bytes larger than the completed PR103-on-PR106
A++ rate floor (`185578` bytes). It is therefore recorded as dominated for
rate-only score-lowering and must not receive exact CUDA wall clock unless it
stacks with a scorer-changing axis or a new byte floor changes the comparison.

Adversarial-review hardening added after this selection: a static-ready manifest
whose candidate archive path is missing now records `candidate_archive_missing`
and cannot become `exact_evaluable_after_lane_claim`.

## Smaller Blocked Rows

Every ranked row is now blocked relative to the completed PR103-on-PR106 byte
floor:

- `pr101_split_brotli`: `185998` bytes. Blocked by
  `static_exact_eval_packet_not_ready`,
  `pr101_split_brotli_runtime_adapter_not_yet_integrated`, and
  `pr106_inflate_will_fail_on_pr101_decoder_format`, and
  `not_below_active_candidate_byte_floor:185578`.
- `pr101_schema`: `186044` bytes. Blocked by
  `static_exact_eval_packet_not_ready`,
  `pr101_schema_runtime_tree_parity_manifest_missing`,
  `pr101_schema_inflate_output_parity_missing`,
  `strict_pre_submission_compliance_json_missing`, and
  `not_below_active_candidate_byte_floor:185578`.
- `hdm3`: `186066` bytes. Blocked by
  `not_below_active_candidate_byte_floor:185578`.
- `pr106_q10`: `186088` bytes. Blocked by
  `static_exact_eval_packet_not_ready`, `requires_archive_manifest_preflight`,
  and `not_below_active_candidate_byte_floor:185578`.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest src/tac/tests/test_hnerv_entropy_frontier_selector.py -q
.venv/bin/python tools/select_hnerv_entropy_frontier_candidate.py \
  --active-candidate active_pr103_pr106=experiments/results/pr103_repack_pr106_standalone_20260507/final_runtime_packet_proof.json \
  --candidate pr101_schema=experiments/results/hnerv_pr101_schema_candidate_20260507_codex/manifest.with_tool_run.json \
  --candidate pr101_split_brotli=experiments/results/pr101_repack_pr106_20260507T152608Z_claude/manifest.json \
  --candidate hdm3=experiments/results/hnerv_hdm3_archive_candidate_20260507_codex/hdm3_archive_candidate_manifest.json \
  --candidate pr106_q10=experiments/results/hnerv_lowlevel_repack_pr106_q10_20260506_codex/manifest.json \
  --json-out experiments/results/hnerv_entropy_frontier_selection_20260507_worker_entropy_frontier/selection.json \
  --md-out experiments/results/hnerv_entropy_frontier_selection_20260507_worker_entropy_frontier/selection.md
shasum -a 256 \
  experiments/results/hnerv_entropy_frontier_selection_20260507_worker_entropy_frontier/selection.json \
  experiments/results/hnerv_entropy_frontier_selection_20260507_worker_entropy_frontier/selection.md
```
