# Handoff senior-review hardening (2026-05-11)

## Scope

This ledger records the senior-engineer / adversarial-math review pass over the
Downloads handoff:

`/Users/adpena/Downloads/pact_score_lowering_handoff_2026-05-11.md`

The goal was not to add strategy prose. The goal was to make the next
score-lowering tranche more executable, less stale-prone, and harder to
overclaim.

## Changes made

1. Scoped the current score frontier as the PR106/HNeRV-sidecar exact-CUDA
   workstream frontier, not automatically the global repository frontier.
2. Added a senior-review timestamp and made active-job rows explicitly
   timestamp-scoped.
3. Replaced stale-prone active-job prose with required refresh commands and
   claim-ledger timestamps.
4. Turned the radius-2 device-axis four-cell matrix into concrete planned lane
   IDs with dispatch-claim requirements.
5. Added a strict pre-submission compliance command template using the verified
   `scripts/pre_submission_compliance_check.py --help` surface. The template is
   intentionally blocked on a real R2 release surface path.
6. Added the latest measured preflight performance evidence: all 29 checks
   passed in 2.55s wall under the 30s hard budget, per
   `.omx/research/preflight_source_index_dispatch_attempted_fact_20260511_codex.md`.
7. Promoted non-HNeRV public mechanisms into concrete backlog items:
   PR97 H3/range masks, PR81/84/91/92 Quantizr-style mask+pose renderers, PR93
   flatpup/lowpass-luma/delta-varint pose, and ANR/HPAC token/context models.
8. Sharpened the native/Rust/Zig roadmap to stable byte transducers only:
   PR101/PR103 grammar parser, range/arithmetic coder, PR106 sidecar reducer
   bit-packer, and eventually preflight scan kernels after Python oracle
   stabilization.
9. Promoted the untracked device-axis matrix into `.omx/research/` and hardened
   its A1 wording from "PR-submission-ready" to "CPU-axis submission candidate
   for policy review".

## Adversarial conclusions

- The R2 exact-T4 score claim remains legitimate:
  `0.20664588545741508`, archive SHA-256
  `7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f`,
  bytes `186822`, Modal T4 CUDA, samples `600`.
- The claim is not yet paper/submission promotable because strict
  contest-final compliance has not been recorded on a concrete release surface.
- The device-axis behavior is packet-specific. A1 is CPU-favored; PR106 sidecar
  packets are CUDA-favored. No future roadmap item may use a global
  "CPU better" or "CUDA better" heuristic without paired packet evidence.
- The mathematical operating point makes pose-side improvements high-EV at the
  PR106 frontier: with pose around `3.2e-5`, the square-root pose term has
  larger marginal score weight than SegNet. This elevates RAFT/LAPose/foveation
  residual work relative to pure mask grammar for direct score lowering.
- The shortest path to another score decrease remains:
  harvest/classify yshift, complete R2 compliance/custody, run the R2 paired
  CPU/CUDA matrix, compress sidecar bytes with PR101/PR103 grammar, then launch
  structured PR106 residual/sidecar searches.

## Verification

Read-only verification performed during this pass:

- `git status --short --branch`
- `scripts/pre_submission_compliance_check.py --help`
- `tools/claim_lane_dispatch.py --help`
- `.omx/state/active_lane_dispatch_claims.md`
- R2 materialization files under
  `experiments/results/pr106_latent_sidecar_r2_from_kaggle_table_20260511_codex/`
- R2 exact Modal T4 artifacts under
  `experiments/results/modal_auth_eval/pr106_latent_sidecar_r2_20260511T160358Z/`

No remote dispatch was started by this review pass.

## Next implementation tranche

1. Materialize or identify the exact R2 release surface and run strict
   contest-final pre-submission compliance.
2. Refresh the Kaggle yshift and Modal T1 active jobs, then terminal-classify or
   harvest without duplicate dispatch.
3. Execute the R2 paired device-axis matrix with explicit claim rows.
4. Start packet-compiler conformance vectors for PR101/PR103 stream grammar.
5. Convert one non-HNeRV public mechanism into a PR106 residual or sidecar
   score-table producer with byte-closed materialization.
