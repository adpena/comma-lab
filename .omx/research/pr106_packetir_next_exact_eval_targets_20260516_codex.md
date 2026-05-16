# PR106 PacketIR next exact-eval target surface - 2026-05-16

## Status

Landed a non-promotional dispatch-target surface in
`src/tac/packet_compiler/pr106_candidate_matrix.py`.

The matrix now emits `next_exact_eval_targets` and
`next_exact_eval_target_count` alongside the existing PacketIR identity,
runtime-consumption, and exact-eval evidence join.

## Findings

- Candidate count: 16.
- Paired exact measured candidates: 0.
- Paired exact blocked candidates: 3.
- Runtime-consumed candidates needing both axes: 4.
- Single-axis candidates needing the missing pair axis: 9.
- Next exact-eval targets emitted: 17.

Each target records:

- candidate id, format id, archive path, archive SHA-256, and archive bytes;
- missing axis (`contest_cpu` or `contest_cuda`);
- provider target (`modal_linux_x86_64_cpu` or `modal_t4_cuda`);
- lane id, pair-group id, instance-job template, and output-dir template;
- runtime dir, inflate entry point, runtime source/content SHAs when available;
- axis blockers from stale/invalid/missing exact evidence;
- a Modal command template that uses provider-level detach and wrapper detach;
- `score_claim=false`, `promotion_eligible=false`, and
  `ready_for_exact_eval_dispatch=false`.

The command templates deliberately do not embed an
`--expected-runtime-tree-sha256` value. CPU and CUDA Modal entry points extract
uploaded runtimes under different remote roots, so the expected runtime-tree
hash must be computed axis-specifically immediately before dispatch.

## Hardening

Runtime dirs from older consumption artifacts are normalized to repo-relative
paths when they are inside the repository. This prevents operator-local
absolute home paths from leaking into generated dispatch templates.

The Markdown artifact now exposes the claim-before-dispatch status directly in
the next-target table so the human-facing artifact preserves the same guard as
the JSON.

Adversarial follow-up tightened the paired exact gate: candidates with both
CPU and CUDA exact artifacts are still `paired_exact_blocked` unless the same
runtime content tree is proven consumed by the runtime-consumption proof and
both exact axes. This currently blocks `format_0x0c_exact_radix`,
`format_0x0d_latent_score_table`, and `prefix_top_16_pr101grammar`; L5-v2 stack
selection sees zero runtime-bound paired candidates until that custody is
closed.

## Verification

```text
.venv/bin/ruff check tools/build_pr106_packetir_candidate_matrix.py \
  src/tac/packet_compiler/pr106_candidate_matrix.py \
  src/tac/tests/test_pr106_packetir_candidate_matrix.py
All checks passed.

.venv/bin/python -m pytest src/tac/tests/test_pr106_packetir_candidate_matrix.py -q
7 passed in 0.30s

.venv/bin/python tools/build_pr106_packetir_candidate_matrix.py
next_exact_eval_target_count = 17
status_counts = paired_exact_blocked:3, runtime_consumed_needs_paired_exact_eval:4,
  single_axis_exact_measured_needs_pair:9
```

Generated artifacts:

- `.omx/research/pr106_packetir_candidate_matrix_20260516_codex.json`
- `.omx/research/pr106_packetir_candidate_matrix_20260516_codex.md`

## Scope

No provider dispatch was launched. These are fail-fast targets only; every run
still requires `tools/claim_lane_dispatch.py`, paired CPU/CUDA custody, Modal
recovery, and adversarial result review before any status change.
