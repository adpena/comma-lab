# HPRC rate-collapse and compact substrate spine findings

Codex finding, 2026-06-01T04:25Z. Axis labels: all measurements here are
archive-byte / receiver-proof / local engineering signals. No contest CPU/CUDA
score authority is claimed.

## Live HPRC codec result

Source exact bridge:
`.omx/research/hprc_pair_scoped_bounded_runner_plan_20260601T015542Z_codex/hprc-threshold-abs-le-pairs-984a5d110c0f3a66/hprc_incremental_exact_gate_bridge.json`

Source archive:
`/Volumes/VertigoDataTier/pact/hprc_incremental_runner_execution_20260601Tstamp/hprc-threshold-abs-le-pairs-984a5d110c0f3a66/variants/residual_transform_threshold_abs_le_pairs_3_1_2_4_5_7_13_15_17_23_28_29_38_44_45_4_984a5d110c0f3a66/archive.zip`

Source bytes: 859,923. Source archive SHA-256:
`234ef386b71d4bb439517ecf7dab7623e737b0c54da66456fc4d189a8d6c0ec3`.

Lossless compact-section entropy wrapping over
`decoder_qw,latents_rc,selectors_rc,residual_rc,receiver_state` produced a
receiver-proven archive:

- archive bytes: 723,833
- contest rate term: 0.48197068361528067
- archive SHA-256: `fee867022745318e2a55685ba7bf483054c30e096dd70522fc1bc0e39d26e94c`
- report:
  `/Volumes/VertigoDataTier/pact/hprc_rate_collapse_exact_bridge_20260601T040914Z/hprc_rate_collapse_report.json`

Lossy residual-token collapse at `dz0_qd6` crossed the required byte scale and
also passed receiver proof:

- archive bytes: 211,917
- contest rate term: 0.14110683176879119
- archive SHA-256: `0bd4e31b9dbd785fbfa6c2c50f8d349de6d38a0eca7d506be8bdb117d6c52f9e`
- HPRC 0.bin SHA-256: `b1ff6532eec511858278264509c1f8376c90f48c4e8883b34faffd58d3dbae59`
- receiver proof:
  `/Volumes/VertigoDataTier/pact/hprc_rate_collapse_dz0_qd6_receiver_proof_20260601T041742Z/best_archive_export/receiver_proof/hprc_receiver_proof.json`
- exact-gate bridge:
  `/Volumes/VertigoDataTier/pact/hprc_rate_collapse_dz0_qd6_receiver_proof_20260601T041742Z/hprc_rate_collapse_exact_gate_bridge.json`

The exact-gate bridge is preclaim-ready and dispatchable after lane claim, but
score authority blockers remain:

- `contest_cpu_cuda_exact_eval_not_executed`
- `mlx_local_response_is_advisory_not_score_authority`

Important caveat: `dz0_qd6` changes 1,297,975 of 2,764,800 residual tokens
and records residual-token MSE 1.5104774236679077. It is a byte breakthrough,
not a score claim. Next authority step is full replay / exact CPU-CUDA.

## Engineering landed

The compact receiver now supports archive-contained entropy wrappers around
legacy semantic sections. Wrappers are decode-only, kind-checked, length-checked,
and SHA-checked before semantic parsing. Mutators unwrap before mutation and
rewrap only when the source section was wrapped.

The rate-collapse tool now accepts training results, exact HPRC bridges, or
source archives; rejects stale exact-bridge custody; emits false-authority
reports; can run lossless-only storage transcodes by default; and requires
`--target-rate-term` or `--enable-lossy-residual-collapse` before lossy residual
token candidates are materialized.

The HPRC queue builder now has a rate-collapse follow-up step before replay so
future compact-receiver rows do not leave residual payload mass as orphaned
manual work.

## Compact substrate research convergence

Three read-only research passes converged on the same implementation shape:
do not build a separate framework for every tiny-video representation. Use one
byte-audited packet spine with swappable sections:

- PR95/HNeRV-style compact receiver as the control arm.
- RNeRV/PACT-NeRV as trainable base renderers when their decoder plus latent
  bytes beat HPRC.
- Tree/Hi/SR/VQ-NeRV as section policies: temporal allocation, hierarchy,
  low-res plus super-resolution, and residual-token codebooks.
- C3/Cool-Chic-style latent-codebook codecs as a second-wave latent-grid
  competitor when model/table overhead is charged.
- SIREN/FINER/WIRE/BACON-style implicit bases as residual atoms or procedural
  bases, not assumed primary carriers until byte-value profiles justify them.
- Mamba/Dreamer/procedural driving priors only when every weight, state, and
  selector byte is archive-charged and receiver-consumed.

The byte math is the governor. At the contest denominator, 285,000 archive
bytes already contribute about 0.18977 rate score before any distortion. The
practical target for a frontier substrate is therefore PR95-scale or lower:
roughly 178-216 KB with enough distortion budget left for SegNet/PoseNet.

## Next action

Run the 211,917-byte receiver-proven `dz0_qd6` candidate through full local
replay and then paired contest CPU/CUDA exact evaluation. If distortion damage
is too high, keep the exact byte spine and search the residual-token
quantization schedule with scorer-value-per-byte acceptance, using PR95/HNeRV
and VQ/Pact-NeRV packets as the next compact controls.
