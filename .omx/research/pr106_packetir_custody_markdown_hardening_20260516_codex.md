# PR106 PacketIR Custody Markdown Hardening - 2026-05-16

## Context

The PR106 PacketIR matrix JSON carried strong exact-eval custody, but the
Markdown summary was too compact for paper/review use. Paired rows showed
`valid` without surfacing the exact artifact paths, artifact hashes,
component distances, log paths, runtime-content SHA, or source artifact
promotion semantics.

## Fix

- Added `source_artifact_warnings` to matrix rows.
- Preserved source exact-eval artifacts with `score_claim=true` as visible
  warnings, even though the matrix itself remains non-promotional.
- Marked runtime-content SHA values derived from matching paired exact eval as:
  - `runtime_content_tree_sha256_derived_not_direct_manifested=true`
  - `runtime_content_tree_sha256_backfill_required=true`
  - `runtime_content_tree_sha256_backfill_path=<runtime consumption JSON>`
- Recomputed the current Modal-uploaded runtime content hash for each row's
  `runtime_dir` and fail-closed rows whose current runtime tree no longer
  matches the runtime-consumption artifact.
- Added archive-SHA custody to paired dispatch command templates:
  `--expected-archive-sha256 <sha256>`.
- Removed literal `<UTC>` run IDs from executable paired command templates; the
  dispatcher now generates the real timestamp when the command is run.
- Expanded the Markdown renderer with:
  - paired exact artifact path and SHA;
  - axis scores and SegNet/PoseNet component distances;
  - runtime-content SHA;
  - exact-eval log path;
  - source artifact score-claim flags;
  - runtime-content SHA derivation notes.

## Regenerated Artifacts

- `.omx/research/pr106_packetir_candidate_matrix_20260516_codex.json`
  - SHA-256: `cda1219fb880cc0513a5d0706af1b95fe74e7a8a52391588e054dba2c24ad93c`
- `.omx/research/pr106_packetir_candidate_matrix_20260516_codex.md`
  - SHA-256: `898f977ce73af842758a3242ec469488e794084a19a2077d11d8bdc3385a7b8a`

The L5-v2 pinned matrix SHA was updated to the regenerated JSON artifact.

## Scope

This is custody/provenance hardening only. It intentionally changes candidate
readiness: all 16 PacketIR rows now fail closed as
`runtime_consumption_blocked` until runtime-consumption proofs are regenerated
against the current `submissions/pr106_latent_sidecar_r2_pr101_grammar`
runtime. The matrix continues to emit `score_claim=false`,
`promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Current Blocker

The current Modal-uploaded runtime content hash is
`8790ec81e5153a8fe3cb250e82b522763ae82b052b48655556be94acb05d5d51`,
while paired exact/runtime-consumption rows carry older content hashes such as
`128604ad742deb46008fc312424801ac8a2e607c924266bdedaa763c059aaf72` and
`5d8e508cda5ea0a3264476ecf0bd526858baa61806627cba0517ec1fc7445de8`.
Next action is to regenerate direct runtime-consumption manifests for the
current runtime before re-exposing PacketIR exact-eval targets or L5-v2 stack
cell proposals.
