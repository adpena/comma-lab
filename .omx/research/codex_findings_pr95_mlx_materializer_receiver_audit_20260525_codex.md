<!-- SPDX-License-Identifier: MIT -->
---
schema: codex_findings_v1
topic: pr95_mlx_materializer_receiver_audit
created_at_utc: 2026-05-25T22:20:00Z
author: codex
lane_id: lane_codex_pr95_mlx_materializer_receiver_audit_20260525
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
dispatch_attempted: false
research_only: false
---

# PR95 MLX and Materializer Receiver Audit

## Scope

This memo preserves the harvested read-only findings from two completed
subagents after the MLX downstream-drift gate, Hinton surrogate smoke, and
Hinton queue-observer wire-in landed on `main`.

No files were edited by the subagents. Codex landed this memo to prevent the
audits from remaining only in transient chat/tool output.

## PR95 MLX Reproduction State

Implemented and real:

- Queueable PR95 MLX control profiles exist in
  `tools/build_pr95_mlx_optimizer_matrix_queue.py`.
- PR95/HNeRV MLX decoder, archive grammar, optimizer partition, source-video
  timing smoke, export, package, and runtime-consumption scaffolds exist.
- Local MLX training infrastructure updates trainable latents and writes
  checkpoints, but remains a local research-signal surface.

Still partial or fake relative to faithful PR95 reproduction:

- `full_pr95_source_video_runtime` is still a timing spine: one pair by
  default, synthetic fallback, and one-step cells.
- The stage matrix emits independent one-stage plans instead of a chained
  8-stage checkpoint/resume curriculum.
- The long-training objective is RGB/YUV-style reconstruction, not source-faithful
  PR95 scorer loss. SegNet/PoseNet loss remains blocked.
- Stage schedule, QAT, C1a/resume, and exact source hparams are not yet
  source-faithful.
- Runtime proof is byte-count consumption, not full-frame inflate parity.
- Long-training currently compares frame 0 against single-frame targets, so it
  is not pair-faithful.
- Packaging defaults to source-archive latents unless `--latents-from-pt` is
  used, so trained-latent loop closure is not the default path.

Highest-EV PR95 MLX next patches:

1. Add a fail-closed `pr95_faithful_reproduction_readiness` schema/helper that
   requires full source-video coverage, chained stages, source hparam/QAT/resume
   parity, scorer loss or strict surrogate, trained-latent export,
   full-frame inflate parity, and paired contest CPU/CUDA exact eval.
2. Convert `full_pr95_source_video_runtime` into a checkpointed DAG where stage
   N consumes stage N-1 and the manifest records one candidate lineage.
3. Replace or relabel the generic 3,000-epoch long-training schedule; either
   source-match the recovered PR95 curriculum or keep it as an abbreviated
   smoke only.
4. Wire real scorer loss or a strict surrogate contract and test gradient
   reachability through decoder outputs and latents.
5. Extend runtime proof to source-vs-candidate full-frame parity, including a
   regression where byte counts match but bytes differ.
6. Close trained-latent packaging with a checkpoint-to-archive-to-inflate
   parity test using `--latents-from-pt`.

## Materializer And Receiver-Proof State

Executable family-agnostic materializers:

- `archive_section_entropy_recode_v1`
- `packet_member_recompress_v1`
- `packet_member_merge_v1`
- `renderer_payload_dfl1_v1`
- `packet_member_zip_header_elide_v1`
- `tensor_factorize_v1`

Queue/default attack reality:

- Frontier final-rate bootstrap defaults only to
  `packet_member_zip_header_elide_v1` and `packet_member_recompress_v1`.
- `archive_section_entropy_recode_v1` and `tensor_factorize_v1` are optional.
- `packet_member_merge_v1` and `renderer_payload_dfl1_v1` are executable but
  silently omitted from the default/optional frontier target set.

Receiver-proof defects to fix:

1. `renderer_payload_dfl1_v1` can pass queue candidate postconditions without
   proving receiver/runtime consumption. Its queue postconditions should require
   receiver truth, runtime adapter readiness, full-frame inflate parity, and
   DFL1 inflate parity.
2. Queue sweep support is narrower than the standalone sweep CLI: merge and
   DFL1 are supported by the standalone sweep but absent from queue sweep
   adapters.
3. `tensor_factorize_v1` has declaration-only receiver proof; declared
   `cooperative_receiver_id` and `receiver_adapter_kind` should not be enough
   to set `receiver_contract_satisfied=true`.
4. Generic runtime proof verification is too permissive for semantic rewrites.
   Target-specific verifiers should require target-specific runtime artifacts.
5. Section entropy recode can produce byte-closed local candidates, but
   length-changing recodes remain exact-blocked until runtime offset/remap proof
   exists. Queue semantics should make that explicit.
6. Frontier bootstrap should either include merge/DFL1 with concrete
   missing-context blockers or explicitly surface their omission as a planned
   limitation.

Highest-EV final-rate next patches:

1. Add DFL1 fail-closed postcondition tests to
   `src/tac/tests/test_byte_shaving_campaign_queue.py`.
2. Add queue-level sweep tests for merge and DFL1 mirroring existing standalone
   sweep coverage.
3. Tighten tensor-factorize receiver verification so declaration-only metadata
   remains exact-blocked.
4. Split generic `verify_runtime_consumption_proof()` into target-specific
   proof contracts or add required runtime-adapter fields by materializer.
5. Add a section-recode test proving length-changing saved-byte recodes cannot
   clear receiver/exact readiness without runtime remap proof.
6. Update frontier bootstrap tests so executable omissions are visible and
   operator-routable instead of silent.

## Authority Boundary

All findings above are engineering readiness findings, not score claims. MLX
and local materializer outputs remain advisory/local until the relevant
fail-closed receiver, full-frame parity, and contest CPU/CUDA authority gates
are satisfied.
