# Roadmap Active Queue Refresh - 2026-05-09

<!-- generated_at: 2026-05-10T03:55:00Z -->
<!-- evidence_grade: roadmap_custody_refresh; no score claim; no dispatch -->

## Scope

Codex refreshed operator-facing routing after the A1 sidecar `528/600` local
chunk and sidecar dispatch-readiness hardening.

No remote job, GPU job, exact eval, or dispatch claim was launched.

## Current Active Queue

1. **AV discriminator harvest.** The only active dispatch claim remains
   `lane_avvideodataset_cuda_path_mechanism_discriminator` /
   `discriminator-sweep-20260509T110211Z`. Baseline CPU is harvested; the
   combined discriminator is not terminal. Do not relaunch a duplicate.

2. **A1 sidecar local completion.** Local resumable search is at `528/600`.
   The latest archive is `178,316 B`, SHA-256
   `b7c74bac342c8d8381a037c019ed446498632d464a133f05dc393d4804a6250b`.
   It remains non-dispatchable. Before exact eval, pairs `0..335` require
   `--recheck-unproven-pairs`, and the final packet requires exact
   `inflate.sh` signature smoke, runtime-output no-op proof, and structured
   dispatch/preflight records.

3. **Phase 1/T1 packet custody.** Keep local until the packet compiler emits a
   byte-different, runtime-consumed archive with strict runtime closure and
   no-op proof.

## Report Fix

`reports/latest.md` now marks the May 4 Omega-W-V3 launch-ready section as
historical / do-not-dispatch inline, not only in a later supersession block.
