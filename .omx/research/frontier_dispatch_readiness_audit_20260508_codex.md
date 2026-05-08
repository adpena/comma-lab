# Frontier Dispatch Readiness Audit - 2026-05-08

Scope: local-state-only dispatch readiness audit after the monolithic HNeRV
archive layout correction. No dispatch was attempted.

## Current Exact Floor

Active A++ HNeRV rate anchor:

- lane: `hnerv_pr103_pr106_ac_repack_exact_eval`
- job: `pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z`
- archive:
  `experiments/results/pr103_repack_pr106_standalone_20260507/exact_eval_static_release_surface/archive.zip`
- bytes: `185578`
- sha256:
  `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- strict formula score: `0.2089810755823297`
- report-reconstructed score: `0.20898105277982337`
- evidence: A++ contest T4, `600` samples, terminal dispatch claim present.

This floor makes PR106/PR106x rate-only packets above `185578` bytes
non-dispatch candidates unless they stack onto this packet or intentionally
change scorer-visible output with component-risk gates.

## Monolithic Archive Correction

The corrected layout manifest
`reports/frontier_monolithic_archive_layout_20260508.json` proves PR101 and
PR106 frontier archives are single-member ZIP packets with parser-proven
internal sections. ZIP-member-level mask/pose budgets are invalid on these
substrates. Future candidate accounting must target parser-proven sections
such as PR101 `decoder_blob` / `latent_blob` / `sidecar_blob` or PR106
`decoder_packed_brotli` / `latents_and_sidecar_brotli`.

## Dispatch State

Current shell env is not Lightning-ready. `env | sort | rg
'^(LIGHTNING_|VAST_|WANDB_|CUDA|AWS_|SSH_|PYTHONPATH|UV_)'` returned only
`SSH_AUTH_SOCK`; no `LIGHTNING_SSH_TARGET`, `LIGHTNING_REMOTE_PACT`,
`LIGHTNING_UPSTREAM_DIR`, `LIGHTNING_TEAMSPACE`, `LIGHTNING_STUDIO`, or
`LIGHTNING_SDK_USER` is present.

Active claim inside the 24h TTL:

- lane_id: `arch_shrink_x0.4_lightning`
- latest job: `arch-shrink-x0-4-lightning-20260508T010514Z`
- status: `active_dispatching`
- owner: `claude_lab`
- local evidence: source manifest only under
  `experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T010514Z/source_manifest.json`
- blocker for codex dispatch: active same-lane claim by another agent; do not
  dispatch or duplicate. Next safe action is monitor/harvest/terminal-close.

Recent completed exact-eval claim:

- lane_id: `hnerv_pr103_pr106_ac_repack_exact_eval`
- status: `completed_score_0_2089810528`
- closes the active PR103/PR106 eval reservation.

## Refreshed Readiness

Command run to `/tmp` only:

```text
.venv/bin/python tools/build_frontier_roadmap_status.py \
  --json-out /tmp/pact_frontier_dispatch_readiness_20260508.json \
  --md-out /tmp/pact_frontier_dispatch_readiness_20260508.md \
  --operator-approved-exact-cuda
```

Result:

- `ready_for_exact_eval_dispatch=false`
- `row_count=13`
- `dirty_path_count=69`
- `dirty_blocked_row_count=0`
- `ready_candidate_packet_count=0`
- `field_selection_ready_candidate_packet_count=0`
- `selected_candidate_packet=pr106_q10_151byte_brotli`
- `selected_candidate_decision=rate_only_candidate_above_active_pr103_pr106_floor`
- `effective_dispatch_candidate=none`

`tools/check_dispatch_cli_shell_hazards.py --strict` returned clean.

## Top Actionable Candidates

1. `categorical_qma9_clade_spade_openpilot`
   - exact claim lane ID when a byte-closed runtime candidate exists:
     `categorical_qma9_clade_spade_openpilot`
   - current status: local candidate artifacts exist, but decode/reencode and
     runtime parity are blocked.
   - blockers: PR91 HPM1 full 600-frame parity missing; phase-major stream
     fails after `15989` symbols; tile-major prefix fails at frame 0 group 12
     symbol 210; runtime consumer is fail-closed skeleton, not a decoder.
   - next non-GPU action: recover probability/range-state or context/order
     drift, then prove full decode/reencode parity before any claim.

2. `joint_admm_balle_arithmetic_stack`
   - exact claim lane ID when the runtime consumes charged bytes:
     `joint_admm_balle_arithmetic_stack`
   - current status: byte-closed `jcsp.bin` member exists, but submission
     runtime detects and refuses consumption.
   - blockers: Balle/hyperprior codecs not instantiated for non-fixture model
     streams; side information must be charged; no lane claim; no stacked exact
     CUDA eval; component wins are not composability proof.
   - next non-GPU action: wire `submissions/robust_current` to decode and
     consume `jcsp.bin`, then strict preflight before any claim.

3. `hnerv_per_tensor_context_entropy`
   - exact claim lane ID after a byte-positive runtime packet exists:
     `hnerv_per_tensor_context_entropy`
   - current status: shared-context HDC fixture landed, but remains
     byte-negative.
   - blockers: HDC1/HDC2 parity fixtures are raw-equal but still worse than
     source Brotli; no deterministic decoder runtime; no dispatchable archive.
   - next non-GPU action: cluster or codebook-share HDC2 context tables until a
     byte-positive archive plus deterministic runtime exists.

Packet-level near-misses:

- `pr106_q10_151byte_brotli`: lane ID `pr106_q10_151byte_brotli`; static
  preflight passed, but it is above the current `185578` byte floor, lacks an
  active claim, and this shell lacks Lightning env.
- `wr01_apply_pr106x_half`: lane ID `wr01_apply_pr106x_half`; local public
  replay preflight passed, but strict candidate preflight is refused, KKT proof
  is missing, and this shell lacks Lightning env.
- `pr106x_hdm3_decoder_recode_14byte`: candidate key
  `pr106x_hdm3_decoder_recode_14byte`; blocked by strict preflight not ready,
  missing dispatch identity for lane claim, and above-floor rate-only status.
