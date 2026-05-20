# PR110 Post-Merge Canonicalization Plan

## Current Gate

Checked GitHub on 2026-05-20T16:04:47Z:

- PR: `https://github.com/commaai/comma_video_compression_challenge/pull/110`
- State: open
- Merged: `false`
- `merged_at`: `null`
- Head SHA: `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`
- Reported merge commit SHA:
  `8fa62958b0d4dd8fdf59a1348795977475f7faaf`

Do not use the reported merge commit as canonical until `merged=true` and
`merged_at` is non-null.

## Decision

Yes: after PR110 merges, standardize all frontier work against the merged PR110
root. PR110 should become the frozen baseline for local development,
post-merge reproduction, byte-closed optimization, and exact-eval dispatch.

Until merge, use the current PR110 head only as a provisional integration target
and keep all artifacts marked `score_claim=false` and
`ready_for_exact_eval_dispatch=false`.

## Canonical Root After Merge

The post-merge baseline packet must record:

- PR number and URL.
- `merged_at`.
- merge commit SHA.
- head SHA.
- archive path, byte count, and SHA-256.
- ZIP member order, method, byte counts, CRCs, and SHA-256s.
- runtime tree file list and SHA-256s.
- report/README/manifest SHA-256s.
- exact eval axis labels: `[contest-CUDA]`, `[contest-CPU]`, and any local
  advisory axes kept separate.
- PR body snapshot SHA-256.

## Rerun Matrix

Rerun or rebuild these artifacts from the PR110 canonical root:

1. Public/live baseline custody:
   archive, runtime, report, README, manifest, PR body, and GitHub metadata.
2. Official `inflate.sh` raw-output reproduction:
   byte counts, raw SHA-256, runtime dependency closure, no scorer at inflate.
3. Component-response surfaces:
   SegNet frame-indexed sensitivity and PoseNet pair-indexed sensitivity kept
   separate.
4. Master-gradient anchors:
   merged-root archive bytes and member payload bytes, with axis-dominance
   metadata tagged non-authoritative unless exact eval proves it.
5. LFV1/HFV1 integration:
   rebase the local post-selector sidecar adapter onto the merged runtime root.
6. HFV1 sidecar search:
   identity no-op, small top-k hard-pair seed, sensitivity-weighted search,
   orthogonalized PoseNet/SegNet search, and stagewise freeze/unfreeze variants.
7. Archive grammar and packer:
   ZIP_STORED/member-order controls, sidecar byte overhead, and deterministic
   rebuild proof.
8. Dispatch readiness:
   lane claim, exact CUDA packet, exact CPU public-axis packet if needed, and
   terminal harvest path.

## Engineering Standard

The merged PR110 root should become a single importable/reproducible substrate:

- one canonical baseline manifest;
- one canonical runtime-copy builder;
- one official-inflate raw-control harness;
- one sidecar generator for HFV1/LFV1 candidates;
- one component-response runner or queue packet;
- one exact-eval dispatch packet generator;
- one ledger writer that records every artifact hash and blocker.

Do not keep optimizing against older PR101 or local pre-merge trees except as
forensic controls.

## Optimization Stack

The post-merge optimization staircase should run in this order:

1. Baseline freeze:
   reproduce merged PR110 exactly and lock hashes.
2. No-op controls:
   no sidecar, identity sidecar, and runtime-copy parity.
3. Local scorer-visible perturbation:
   prove only intended frames/pairs change.
4. Component smoke:
   measure component deltas before exact-eval spend.
5. Orthogonalized search:
   PCGrad/gradient projection between PoseNet pair loss and SegNet frame loss.
6. Freezing curriculum:
   freeze PR110 runtime, tune sidecar; then freeze sidecar, tune selector/atoms;
   only then allow joint search.
7. Byte allocator:
   decide sidecar bytes versus selector bytes versus archive grammar bytes.
8. Exact CUDA:
   dispatch only byte-closed candidates with identity controls and lane claim.

## Immediate Next Action

Wait for PR110 merge. Once merged, build:

`experiments/results/pr110_merged_canonical_baseline_<utc>_codex/`

with a full baseline manifest and rerun the PR101 LFV1/HFV1 integration controls
against the merged PR110 runtime, not the current provisional copy.
