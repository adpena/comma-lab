# Codex Session Summary - NeRV Family-Optimal Fail-Closed Queue

UTC: 2026-06-06T00:54:36Z

## Scope

Implemented a family-specific NeRV campaign queue contract so HiNeRV and SNeRV
no longer share a generic launch surface. HiNeRV rows now expose the optimal
HiNeRV actuator set (decoder-weight waterfill, archive ladder, latent bitstream,
receiver-cache quality, scorer-input stabilization, direct-live SegNet escape,
Pose direct-live VJP, recon-pixel weights). SNeRV rows expose the SNeRV actuator
set (SNAR2 fixed binary header, LF/HF replacement queue, step-map packet recode,
official MFU/HFR/TUB source-forward payload, renderer nondegeneracy,
scorer-tethered LF/HF residuals, skip-high value-domain noncollapse, source-
faithful native MLX training).

## Landed Changes

- `src/tac/analysis/nerv_candidate_curriculum.py`
  - Added `nerv_family_optimal_strategy_contract.v1`.
  - Rejects foreign-family artifact evidence:
    - `hinerv_foreign_snerv_artifact_evidence_rejected`
    - `snerv_foreign_hinerv_artifact_evidence_rejected`
  - Threads the family strategy into HiNeRV and SNeRV curriculum blockers.

- `src/tac/analysis/nerv_long_training_campaign_plan.py`
  - Carries `family_optimal_strategy` on each campaign row and queue launch
    authority contract.
  - Treats HiNeRV prelaunch-critical missing actuators as launch blockers:
    recon-pixel weights, decoder waterfill, scorer-input profile/health,
    archive-in-loop byte oracle, partial-pair-only feedback, and non-
    representative distortion evidence.
  - Blocked rows retain command handoff metadata but have no launch steps.

- `src/tac/analysis/snerv_lf_hf_replacement_queue.py`
  - Prior slice remains active: proof-consumed SNeRV LF/HF rows block on bounded
    training binding rather than pretending runnable.

## SSD Artifacts

Primary artifact:

`/Volumes/VertigoDataTier/pact/experiments/results/nerv_campaign_plan_v30c_family_scoped_hinerv_smoke_feedback_20260605Tcodex/nerv_long_training_campaign_plan.json`

SHA-256:

`c07aade52994721446e97bba9a67acd0a69f0649ae219b2694e57c7f4c15d07e`

Companion artifacts:

- `experiment_queue.json`
  - SHA-256: `02ee9e1c6bfa0134055e5c805529ca458d72d0156522e7ae5493580f2ae1f35b`
- `snerv_lf_hf_replacement_queue.json`
  - SHA-256: `806a97c5f436ec802eece3f4e926f3818a031635b898ea6df7796e3513371b27`
- `snerv_lf_over_ceiling_reroute_queue.json`
  - SHA-256: `dbe4005012faaac92ed1af7d92e46839b2fe1d6ae9afb0c98e3a21c3270d876a`

Important counts:

- Campaign rows: 36
- Blocked rows: 36
- Launchable local rows: 0
- HiNeRV rows: 33, all disabled, strategy family `hi_nerv`
- SNeRV rows: 3, all disabled, strategy family `snerv`
- Blocked queue rows with launch steps: 0
- SNeRV LF/HF queue rows: 21, all blocked, runnable count 0
- Selected measured LF payload evidence: 879,605 bytes from
  `/Volumes/VertigoDataTier/pact/experiments/results/snerv_checkpoint_exports/snerv_epoch1299_ema_direct_packet_20260603T2155Z/snerv_checkpoint_archive_export.json`
- Candidate feedback sources: 81, including the harvested HiNeRV smoke feedback
  row from
  `/Volumes/VertigoDataTier/pact/codex_smokes/hinerv_lr_protected_auto_latent_16ep_onepair_20260606T013000Z/nerv_candidate_byte_feedback_row.json`
- Foreign-family feedback blocker leak count:
  - HiNeRV rows containing `snerv_*` blockers: 0
  - SNeRV rows containing `hi_nerv_*` or `hinerv_*` blockers: 0

## HiNeRV Smoke Harvest

Observed existing smoke:

`/Volumes/VertigoDataTier/pact/codex_smokes/hinerv_lr_protected_auto_latent_16ep_onepair_20260606T013000Z`

Status:

- `compact_renderer_mlx_spine_runner_report.json` exists.
- Telemetry rows: 16
- Last epoch: 15
- Last loss: 4,699,717.0
- Final checkpoint metadata count: 1
- Candidate feedback row SHA-256:
  `5d5fd8fee4f526877cf1f7c836888b029a9df43a08c0ed7a757e0de9ab59671b`

The smoke is false-authority partial-pair evidence. It remains useful launch
control signal only; it does not make any row launchable.

## Remaining True Blockers

HiNeRV:

- `requires_verified_joint_p18_p19_recon_pixel_weight_artifact`
- `hinerv_decoder_weight_waterfill_plan_missing`
- `hinerv_local_scorer_input_profile_missing`
- `hinerv_local_scorer_input_health_gate_failed`
- `hi_nerv_archive_in_loop_byte_oracle_missing`
- representative full600 or hard-pair distortion replay still required

SNeRV:

- Pre-long full-video feedback/receiver proof/prefilter/materialized best packet
  are missing.
- Renderer still blocked on direct-live SegNet argmax collapse and nondegenerate
  telemetry.
- LF-conditioned HF residual payload proof exists, but value-domain xray still
  blocks on unclipped outside uint8 and clipping-changing pixels.
- Joint codebook, temporal LF predictor, tiny-anchor SR, spectral allocator, and
  entropy hyperprior remain implementation gaps or receiver/byte-telemetry gaps.

## Validation

- `uv run ruff check src/tac/analysis/nerv_long_training_campaign_plan.py src/tac/tests/test_nerv_long_training_campaign_plan.py src/tac/analysis/nerv_candidate_curriculum.py src/tac/tests/test_nerv_candidate_curriculum.py src/tac/analysis/snerv_lf_hf_replacement_queue.py src/tac/tests/test_snerv_lf_hf_replacement_queue.py`
- `uv run pytest src/tac/tests/test_nerv_candidate_curriculum.py src/tac/tests/test_snerv_lf_hf_replacement_queue.py -q`
  - 52 passed
- `uv run pytest src/tac/tests/test_nerv_long_training_campaign_plan.py -q`
  - 121 passed

## Lane

Registered:

`lane_nerv_family_optimal_fail_closed_queue_20260605`

Marked:

`impl_complete`

## 2026-06-06 Continuation: Receiver-Closed HiNeRV Waterfill + All-Evidence Queue

Landed the HiNeRV waterfill source-normalization fix and rebuilt the normal
queue with all currently proven SNeRV evidence attached.  The prior HiNeRV
runner had produced a receiver-closed modelsize ladder
(`nerv_receiver_closed_modelsize_ladder.v1`), while the waterfill builder only
accepted `hinerv_archive_size_ladder.v1`; the runner was therefore failing its
post-export waterfill materializer even though the archive path and state NPZ
manifest were present.

Changed:

- `src/tac/analysis/hinerv_archive_ladder_waterfill.py`
  - Accepts receiver-closed modelsize ladder reports as waterfill sources.
  - Deterministically discovers the adjacent
    `hi_nerv_mlx_exported_state_npz_manifest.json` when the ladder row carries
    a measured archive path.
  - Preserves receiver-cache/scorer-basin blockers and adds
    `decoder_weight_waterfill_not_admissible_from_unfit_scorer_basin` instead
    of admitting allocator mutation from false-authority partial-pair evidence.

- `src/tac/tests/test_hinerv_archive_ladder_waterfill.py`
  - Covers receiver-closed ladder normalization and manifest discovery.

- `src/tac/tests/test_compact_renderer_mlx_spine_runner.py`
  - Covers the actual runner path that materializes waterfill from a
    receiver-closed ladder.

New HiNeRV waterfill artifact:

`/Volumes/VertigoDataTier/pact/experiments/results/hinerv_receiver_closed_waterfill_20260606Tcodex/hinerv_archive_ladder_waterfill.json`

SHA-256:

`0f0e1757af5b361aa53f913f822f6aed9aad4d1e5f6aaf0bdda4bd588c25e729`

Harvested one-candidate HiNeRV budget for the live smoke candidate:

`/Volumes/VertigoDataTier/pact/experiments/results/hinerv_receiver_closed_waterfill_20260606Tcodex/hinerv_modelsize_budget_selected_smoke_candidate.json`

SHA-256:

`6a9b213a3e20dc80974c7969028803c52af0ec451e167dd4051d4563ed7e353c`

Latest all-evidence queue:

`/Volumes/VertigoDataTier/pact/experiments/results/nerv_campaign_plan_v33_all_evidence_hinerv_waterfill_20260606Tcodex/nerv_long_training_campaign_plan.json`

SHA-256:

`3731943a29dda23ac0d864dc3c306611dae6b8abd8507421e85f1c70803f17b3`

Companion artifacts:

- `experiment_queue.json`
  - SHA-256: `f7062c58bd341eb61af46533d19d47fadc9e7eb827d21e8a849a6abe514c1eae`
- `snerv_lf_hf_replacement_queue.json`
  - SHA-256: `ffa2f9130acba6fabc01e4d5b3a5601fc1974a5fac7299f9de8a7223f99d46b1`
- `snerv_lf_over_ceiling_reroute_queue.json`
  - SHA-256: `688d6e222df8ff2a709814085e733989b02074cc4e958944320b9304990c9c79`

v33 counts:

- Campaign rows: 14
- Blocked rows: 14
- Launchable local rows: 0
- Experiment queue rows: 14, all `disabled`
- Launch steps: 0
- Candidate feedback sources consumed: 120
- HiNeRV rows: 11
- SNeRV rows: 3
- HiNeRV waterfill sources: 1
- HiNeRV waterfill attached rows: 11
- HiNeRV waterfill unattached sources: 0
- SNeRV measured LF payload reports consumed: 2
- SNeRV source-forward artifacts consumed: 1
- SNeRV LF/HF replacement rows: 21, all blocked, runnable count 0

SNeRV source-forward state in v33:

- Closed from missing/generic:
  - `snerv_official_mfu_hfr_tub_export_not_bound`
  - `snerv_official_mfu_hfr_tub_receiver_payload_not_bound`
  - `snerv_official_mfu_hfr_tub_frame_producing_export_missing`
  - `snerv_official_trained_checkpoint_state_dict_not_loaded`
  - `snerv_official_trained_checkpoint_hfr_weight_mapping_incomplete`
- Proven receiver-side facts:
  - frame-producing official payload replay: `true`
  - receiver runtime decode: `true`
  - receiver payload frame replay: `true`
  - receiver frame decode consumes `output2`: `true`
  - HFR trained checkpoint weight mapping: `true`
- Still blocked, correctly:
  - `snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority`
  - `snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing`
  - `snerv_official_mfu_native_receiver_activation_payload_not_upstream_weight_mapping`
  - `snerv_official_trained_checkpoint_state_dict_mapping_missing`
  - `snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_portable_temporal_encoder_weight_mapping_missing`
  - `snerv_official_tub_portable_output2_decoder_weight_mapping_missing`

SNeRV value-domain state in v33:

- LF-conditioned HF residual receiver payload remains implemented and decoded.
- Value-domain xray remains failed:
  - `snerv_receiver_decode_unclipped_outside_uint8_domain`
  - `snerv_receiver_decode_clipping_changes_pixels`
  - `snerv_official_skip_high_scalar_mean_receiver_range_unfit`
  - `snerv_lf_conditioned_hf_value_domain_noncollapse_proof_missing`

HiNeRV state in v33:

- Waterfill is attached to the live smoke candidate rows.
- Runner admission is still false with refusal reasons:
  - `decoder_weight_waterfill_full_video_coverage_missing`
  - `decoder_weight_waterfill_receiver_proof_not_ready`
  - `receiver_proof_not_satisfied`
  - `decoder_weight_waterfill_receiver_proof_path_missing`
  - `decoder_weight_waterfill_not_admissible_from_unfit_scorer_basin`
  - `full_video_coverage_missing`

Additional validation:

- `uv run ruff check src/tac/analysis/hinerv_archive_ladder_waterfill.py src/tac/tests/test_hinerv_archive_ladder_waterfill.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py`
  - passed
- `uv run pytest src/tac/tests/test_hinerv_archive_ladder_waterfill.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_runner_materializes_waterfill_from_trained_ladder src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_runner_materializes_waterfill_from_receiver_closed_ladder src/tac/tests/test_nerv_long_training_campaign_plan.py::test_build_long_training_campaign_plan_cli_selects_archive_ladder_waterfill_candidate src/tac/tests/test_nerv_long_training_campaign_plan.py::test_build_long_training_campaign_plan_cli_preserves_archive_ladder_waterfill_custody -q`
  - 13 passed
- `uv run pytest src/tac/tests/test_nerv_long_training_campaign_plan.py::test_archive_ladder_waterfill_reingest_preserves_replay_source_and_refuses_unfit_basin -q`
  - 1 passed
- `uv run python -m py_compile src/tac/analysis/hinerv_archive_ladder_waterfill.py tools/build_hinerv_archive_ladder_waterfill.py tools/run_compact_renderer_mlx_spine_runner.py src/tac/analysis/nerv_long_training_campaign_plan.py tools/build_nerv_long_training_campaign_plan.py`
  - passed
- `uv run python tools/lane_maturity.py validate`
  - 1682 lanes validated cleanly

Lane update:

- `lane_nerv_family_optimal_fail_closed_queue_20260605`
  - `real_archive_empirical` marked with the v33 all-evidence queue artifact.
  - Lane is now L2.
  - Contest CPU/CUDA, strict preflight, memory, and deploy-runbook gates remain
    unmarked.

## 2026-06-06 Continuation: SNeRV Source-Forward Authority Burn-Down

Ran the cheap local official TUB source fixture with a persisted value-state
artifact, then fed it through the MFU/HFR/TUB source-forward audit and rebuilt
the official TUB LF/HF decoder replacement authority gate.

New TUB source fixture artifact:

`/Volumes/VertigoDataTier/pact/experiments/results/snerv_official_tub_source_forward_value_state_v1_20260606Tcodex/snerv_official_tub_source_forward_replay.json`

SHA-256:

`cbaed9d53aee6db3fca42c4af741cc1ba1813e5a0a1dbfe32d79ebe2fb37ee44`

Persisted fixture state:

`/Volumes/VertigoDataTier/pact/experiments/results/snerv_official_tub_source_forward_value_state_v1_20260606Tcodex/official_tub_source_fixture_state_dict.npz`

SHA-256:

`ef266d22b452d3fd61de8cd719f12727613fe6408a2c7fa8e7e6fb9dea12cd49`

TUB fixture state:

- source-forward replay executed: `true`
- TUB temporal encoder/output2 source fixture replay passed: `true`
- full TUB source-forward parity proven in fixture scope: `true`
- trained checkpoint state-dict mapping verified: `true`
- value artifact ready: `true`
- source-forward replay authority in fixture scope: `true`
- preserved blocker: `snerv_official_pytorch_wavelets_runtime_dependency_missing`
- score claim: `false`

New MFU/HFR/TUB source-forward artifact:

`/Volumes/VertigoDataTier/pact/experiments/results/snerv_official_source_forward_value_state_v13_20260606Tcodex/snerv_official_mfu_hfr_tub_forward_parity.json`

SHA-256:

`6b69681737c05aad806a498cdf1610c961393349dbb71f660ad11b69c8c679d6`

Source-forward state:

- official MFU/HFR/TUB forward parity artifact passed: `true`
- trained checkpoint loaded: `true`
- trained checkpoint mapping verified: `true`
- full TUB source-forward parity proven: `true`
- source-forward replay authority: `true`
- score claim: `false`
- remaining artifact blockers:
  - `official_weight_tensor_mapping_not_loaded`
  - `full_official_mfu_forward_artifact_not_emitted`
  - `official_hfr_weight_tensor_mapping_not_loaded`
  - `full_official_hfr_forward_artifact_not_emitted`
  - `snerv_official_pytorch_wavelets_runtime_dependency_missing`

New official TUB LF/HF replacement authority gate:

`/Volumes/VertigoDataTier/pact/experiments/results/snerv_official_tub_lf_hf_authority_gate_value_state_v2_20260606Tcodex/snerv_official_tub_lf_hf_replacement_authority_gate.json`

SHA-256:

`22c2764d1f2cadb03d12cdf150c69bbce95de91c2aa1259a8d4194e056ce150b`

Gate state:

- official TUB LF/HF decoder replacement ready: `true`
- blocked gate rows: 0
- queue blockers: []
- score claim: `false`
- exact-eval dispatch ready: `false`

Latest queue after source-forward burn-down:

`/Volumes/VertigoDataTier/pact/experiments/results/nerv_campaign_plan_v34_source_forward_ready_hinerv_waterfill_20260606Tcodex/nerv_long_training_campaign_plan.json`

SHA-256:

`7db74b045c84928120444272bbb29960058ecdc3b6689b3e192d253ee2f03d93`

Companion artifacts:

- `experiment_queue.json`
  - SHA-256: `66de858d2b207d1ea55206b6fad8f49b76e0689a474c7ef44397b67c74f7a035`
- `snerv_lf_hf_replacement_queue.json`
  - SHA-256: `b00265bd1838d6e0a968d5f4a1a459dd46e8fb3ea773d8921abd9549a3addcc2`
- `snerv_lf_over_ceiling_reroute_queue.json`
  - SHA-256: `8f9648498f653018a7b73e3464906f050c51d0c689630bd47837509fbf7a81ad`

v34 counts:

- Campaign rows: 14
- Blocked rows: 14
- Launchable local rows: 0
- HiNeRV rows: 11
- SNeRV rows: 3
- Candidate feedback rows consumed: 120
- HiNeRV waterfill attached rows: 11
- SNeRV source-forward queue blockers: []
- Official replacement authority queue blockers: []
- LF/HF replacement queue rows: 21, all blocked, local executable commands: 0

The previously named SNeRV source-forward blockers are no longer blockers in
the v34 planner-consumed source-forward surface:

- `snerv_official_mfu_native_receiver_activation_payload_not_upstream_weight_mapping`
- `snerv_official_trained_checkpoint_state_dict_mapping_missing`
- `snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded`
- `snerv_official_tub_portable_temporal_encoder_weight_mapping_missing`
- `snerv_official_tub_portable_output2_decoder_weight_mapping_missing`
- `snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority`
- `snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing`

The replacement queue is still correctly non-launchable.  The remaining first
SNeRV LF/HF blockers are now renderer/value-domain blockers:

- official replacement rows:
  - `snerv_score_aware_long_training_direct_live_segnet_candidate_argmax_collapsed`
  - `snerv_renderer_nondegenerate_telemetry_contract_missing_or_failed`
- LF-conditioned HF residual rows:
  - `snerv_lf_conditioned_hf_value_domain_noncollapse_proof_missing`
  - `snerv_receiver_decode_unclipped_outside_uint8_domain`
  - `snerv_receiver_decode_clipping_changes_pixels`
  - `snerv_official_skip_high_scalar_mean_receiver_range_unfit`
  - `snerv_official_scalar_skip_high_no_range_safe_scalar_found`

Lane update:

- `lane_nerv_family_optimal_fail_closed_queue_20260605`
  - `real_archive_empirical` evidence pointer updated to the v34 queue.
  - Lane remains L2.
  - Contest CPU/CUDA gates remain unmarked.

Additional source-forward validation:

- `uv run pytest src/tac/tests/test_snerv_official_tub_source_forward_replay.py::test_snerv_official_tub_replay_cli_writes_value_state_npz src/tac/tests/test_snerv_official_tub_lf_hf_replacement_authority_gate.py -q`
  - 4 passed
- `uv run pytest src/tac/tests/test_hinerv_archive_ladder_waterfill.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_runner_materializes_waterfill_from_receiver_closed_ladder src/tac/tests/test_nerv_long_training_campaign_plan.py::test_archive_ladder_waterfill_reingest_preserves_replay_source_and_refuses_unfit_basin -q`
  - 11 passed
- `uv run python tools/lane_maturity.py validate`
  - 1682 lanes validated cleanly
- `git diff --check` on the owned code/memo/lane files
  - passed

Process state:

- No pytest, source audit, source replay, or campaign-plan build processes
  remain from this tranche.
- Existing unrelated live HiNeRV geometry tether smoke remains at PIDs
  82633/82643:
  `/Volumes/VertigoDataTier/pact/codex_smokes/hinerv_geometry_tether_8ep_pinned_argmax_w025_onepair_20260606T031000Z`

## 2026-06-06 Continuation: Spectral Allocator + LF Hyperprior Proofs Consumed

Added the last two LF/HF implementation proof lanes to the normal planner
surface. The new proof payloads are compact binary payloads, not JSON-header
payloads; the JSON reports are only the false-authority planner evidence.

Changed:

- `src/tac/substrates/snerv_inverse_steg_carrier/spectral_band_allocator.py`
  - Implements a score-tethered spectral band allocator over decoded receiver
    frames.
  - Stores a fixed binary header plus a uint16 `(pair, channel, 4-band)` budget
    table.
  - Proves NumPy receiver decode and section-native byte telemetry.
- `src/tac/substrates/snerv_inverse_steg_carrier/lf_latent_hyperprior.py`
  - Implements a fitted LF latent hyperprior over downsampled LF planes.
  - Stores compact mean/scale hyperprior sections plus zlib-compressed int16
    latent symbols behind a fixed binary header.
  - Proves NumPy replay of LF latents and records entropy/byte telemetry.
- `tools/build_snerv_spectral_band_allocator_payload_proof.py`
- `tools/build_snerv_lf_latent_hyperprior_payload_proof.py`
  - Runnable SSD-backed proof builders over a measured SNeRV packet.
- `src/tac/analysis/snerv_lf_hf_replacement_queue.py`
- `tools/build_snerv_lf_hf_replacement_queue.py`
- `src/tac/analysis/nerv_long_training_campaign_plan.py`
- `tools/build_nerv_long_training_campaign_plan.py`
  - Thread both proof families through the normal queue and campaign-plan
    surfaces.
  - Clear only their implementation/byte-telemetry blockers.
  - Replace cleared implementation blockers with runtime-binding blockers so
    no row becomes launchable from a receiver proof alone.

New proof artifacts:

- Spectral allocator proof:
  `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_lf_hf_replacements/snerv_lf_hf_replace_snerv_snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_skchannelm_6673eeba6ca2/spectral_band_allocator_payload_proof/snerv_score_tethered_spectral_band_allocator_receiver_proof.json`
  - SHA-256: `6fd27c3afca470ed9d7577e2bcd306cb3ecaf1fbd00276091d351fa1c0e417db`
  - Payload bytes: 74
  - Payload SHA-256:
    `97264e34cad863f214d27a2972ef13ec114b7862160e1fa68554a9a08155a8b0`
  - Closed blockers:
    `snerv_score_tethered_lf_hf_band_allocator_not_implemented`,
    `snerv_mfu_hfr_section_native_byte_telemetry_missing`
- LF latent hyperprior proof:
  `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_lf_hf_replacements/snerv_lf_hf_replace_snerv_snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_skchannelm_9b9e62ae49ff/lf_latent_hyperprior_payload_proof_scale_entropy/snerv_lf_latent_hyperprior_receiver_proof.json`
  - SHA-256: `f7d25e409375a324d5a6d42cae9dd10a8c8db38ff826893c661ac984dec27a5f`
  - Payload bytes: 386
  - Payload SHA-256:
    `8c3042bdc87f68026bdd458eb164f2861684ae1939de6470614c0cabb833f5cd`
  - Entropy model: `per_slice_laplace_scale_hyperprior`
  - Closed blockers:
    `snerv_lf_latent_hyperprior_not_implemented`,
    `snerv_lf_latent_hyperprior_numpy_decoder_missing`,
    `snerv_lf_latent_hyperprior_receiver_replay_missing`

Latest consumed queue:

`/Volumes/VertigoDataTier/pact/experiments/results/nerv_campaign_plan_v41_scale_entropy_hyperprior_consumed_20260606Tcodex/nerv_long_training_campaign_plan.json`

SHA-256:

`e4ca26c13f2ea399123c5865e5c81b42da49cde296319d179ce7667bfe91ba80`

Companion artifacts:

- `experiment_queue.json`
  - SHA-256: `f7ef255dc696ab7185cc4220f59d423c4a14840f8ebf292bc722eae72c8859f6`
- `snerv_lf_hf_replacement_queue.json`
  - SHA-256: `15b911b7a5578894c36dcf7ae60adfd1c8f94ba27793782089101c8bc697d3ce`

v41 state:

- Campaign rows: 14
- Blocked rows: 14
- Launchable local rows: 0
- SNeRV LF/HF queue rows: 21
- SNeRV LF/HF blocked rows: 21
- SNeRV LF/HF local executable rows: 0
- Score claim: false
- Ready for exact eval: false

The following previous implementation blockers are no longer present in the
v41 LF/HF queue:

- `snerv_score_tethered_lf_hf_band_allocator_not_implemented`
- `snerv_mfu_hfr_section_native_byte_telemetry_missing`
- `snerv_lf_latent_hyperprior_not_implemented`
- `snerv_lf_latent_hyperprior_numpy_decoder_missing`
- `snerv_lf_latent_hyperprior_receiver_replay_missing`

Remaining SNeRV LF/HF blockers after v41:

- `snerv_renderer_nondegenerate_smoke_missing`
- `snerv_renderer_nondegenerate_smoke_min16_pairs_missing`
- `snerv_renderer_nondegenerate_telemetry_contract_missing_or_failed`
- `snerv_score_aware_long_training_direct_live_segnet_candidate_argmax_collapsed`
- `snerv_lf_conditioned_hf_bounded_training_binding_missing`
- `snerv_joint_lf_hf_bounded_training_binding_missing`
- `snerv_temporal_lf_predictor_receiver_runtime_binding_missing`
- `snerv_lf_super_resolution_receiver_runtime_binding_missing`
- `snerv_score_tethered_lf_hf_band_allocator_runtime_binding_missing`
- `snerv_lf_latent_hyperprior_runtime_binding_missing`
- `snerv_lf_hf_replacement_queue_false_authority`

Validation for this continuation:

- `uv run pytest src/tac/substrates/snerv_inverse_steg_carrier/tests/test_spectral_band_allocator.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_lf_latent_hyperprior.py -q`
  - 4 passed
- `uv run pytest src/tac/tests/test_snerv_lf_hf_replacement_queue.py -q`
  - 33 passed
- `uv run pytest src/tac/tests/test_nerv_long_training_campaign_plan.py::test_long_training_campaign_plan_threads_source_forward_artifact_into_lf_hf_queue src/tac/tests/test_nerv_long_training_campaign_plan.py::test_build_long_training_campaign_plan_cli_writes_outputs -q`
  - 2 passed
- `uv run ruff check ...`
  - All touched files passed.

## 2026-06-06 Continuation: Bounded Training Binding Contract Gate

Closed the next false-readiness gap in the SNeRV LF/HF queue handoff.  Every
`snerv_lf_hf_replacement_candidate_row.v1` now carries an explicit
`snerv_lf_hf_bounded_training_binding_contract.v1` field.  The contract is
bound only when the row has a real runner actuator; otherwise it records the
family-specific bounded-training blocker even when earlier renderer/source
blockers mask it in the row-level blocker list.

Changed:

- `src/tac/analysis/snerv_lf_hf_replacement_queue.py`
  - Added `bounded_training_binding_contract` to all LF/HF replacement rows.
  - Official TUB LF/HF rows bind only to the existing bounded SNeRV smoke
    command when that command is actually emitted.
  - Residual, joint-codebook, temporal, tiny-anchor SR, spectral allocator,
    and hyperprior rows remain fail-closed until family-specific bounded
    training actuators exist.
- `tools/run_compact_renderer_mlx_spine_runner.py`
  - The planner-row launch gate now rejects SNeRV LF/HF queue rows whose
    bounded-training contract is missing, schema-mismatched, or unbound.
  - Runnable planner rows also require a concrete executable command; metadata
    alone can no longer make a row launchable.
  - Launch-contract blockers now bubble into the top-level queue guard.
- `src/tac/tests/test_snerv_lf_hf_replacement_queue.py`
- `src/tac/tests/test_compact_renderer_mlx_spine_runner.py`
  - Added producer and consumer regression coverage for the contract.

Latest consumed LF/HF queue:

`/Volumes/VertigoDataTier/pact/experiments/results/snerv_lf_hf_replacement_queue_v43_bounded_training_contract_20260606Tcodex/snerv_lf_hf_replacement_queue.json`

SHA-256:

`3a099da346cd5e9d24bde6e93357c1be06a72d27a4249a413ae04f4a424cc30d`

v43 state:

- SNeRV LF/HF queue rows: 21
- Bounded-training contract rows: 21
- Bound bounded-training actuators: 0
- SNeRV LF/HF blocked rows: 21
- SNeRV LF/HF local executable rows: 0
- Score claim: false
- Rank/kill eligible: false
- Ready for exact eval: false

Validation for this continuation:

- `uv run pytest src/tac/tests/test_snerv_lf_hf_replacement_queue.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_planner_row_launch_gate_accepts_snerv_lf_hf_replacement_queue_rows src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_planner_row_launch_gate_rejects_snerv_lf_hf_runnable_without_command src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_planner_row_launch_gate_rejects_snerv_lf_hf_missing_training_contract src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_planner_row_launch_gate_rejects_snerv_lf_hf_unbound_training_contract src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_planner_row_launch_gate_rejects_nonrunnable_queue_artifact -q`
  - 39 passed
- `uv run pytest src/tac/tests/test_snerv_lf_hf_runtime_binding.py src/tac/tests/test_snerv_lf_hf_replacement_queue.py src/tac/tests/test_nerv_long_training_campaign_plan.py::test_long_training_campaign_plan_threads_source_forward_artifact_into_lf_hf_queue src/tac/tests/test_nerv_long_training_campaign_plan.py::test_build_long_training_campaign_plan_cli_writes_outputs src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_planner_row_launch_gate_accepts_snerv_lf_hf_replacement_queue_rows src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_planner_row_launch_gate_rejects_snerv_lf_hf_runnable_without_command src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_planner_row_launch_gate_rejects_snerv_lf_hf_missing_training_contract src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_planner_row_launch_gate_rejects_snerv_lf_hf_unbound_training_contract -q`
  - 43 passed
- `uv run ruff check src/tac/analysis/snerv_lf_hf_replacement_queue.py tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_snerv_lf_hf_replacement_queue.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py`
  - All checks passed.
- `uv run python tools/lane_maturity.py validate`
  - 1684 lanes validated cleanly.
- `git diff --check -- src/tac/analysis/snerv_lf_hf_replacement_queue.py tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_snerv_lf_hf_replacement_queue.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py .omx/state/lane_registry.json .omx/state/lane_maturity_audit.log`
  - Passed with no output.

## 2026-06-06 Continuation: Renderer Nondegenerate Unblock Command

Turned the active renderer-collapse blocker into a queue-owned executable
unblock path without marking the blocked LF/HF candidate rows launchable.

Changed:

- `src/tac/analysis/snerv_lf_hf_replacement_queue.py`
  - Added `snerv_lf_hf_queue_unblock_launch_contract.v1`.
  - Official TUB LF/HF rows with only renderer/scorer-domain proof blockers now
    emit `unblock_command_argv` for a bounded 16-pair, 128-epoch SNeRV renderer
    nondegeneracy smoke.
  - The normal candidate `command_argv` remains empty while row blockers remain.
- `tools/run_compact_renderer_mlx_spine_runner.py`
  - Planner-row queue matching now distinguishes normal `launch` mode from
    blocked-row `unblock` mode.
  - The runner accepts an unblock command only when the separate unblock launch
    contract is schema-valid, runnable, command-present, and command-control
    matched.
- `src/tac/tests/test_snerv_lf_hf_replacement_queue.py`
- `src/tac/tests/test_compact_renderer_mlx_spine_runner.py`
  - Added producer and consumer regression coverage for blocked-row renderer
    unblock commands.

Latest consumed LF/HF queue:

`/Volumes/VertigoDataTier/pact/experiments/results/snerv_lf_hf_replacement_queue_v44_renderer_unblock_contract_20260606Tcodex/snerv_lf_hf_replacement_queue.json`

SHA-256:

`75f6eb04f721fec3dfb4dd90d6905e7d0bb097ba8ee6b72a7d87069b2d581603`

v44 state:

- SNeRV LF/HF queue rows: 21
- Blocked rows: 21
- Normal executable candidate rows: 0
- Renderer unblock command rows: 3
- `next_unblock_command_argv`: bounded SNeRV renderer nondegeneracy smoke
- Score claim: false
- Rank/kill eligible: false
- Ready for exact eval: false

Validation for this continuation:

- `uv run pytest src/tac/tests/test_snerv_lf_hf_runtime_binding.py src/tac/tests/test_snerv_lf_hf_replacement_queue.py src/tac/tests/test_nerv_long_training_campaign_plan.py::test_long_training_campaign_plan_threads_source_forward_artifact_into_lf_hf_queue src/tac/tests/test_nerv_long_training_campaign_plan.py::test_build_long_training_campaign_plan_cli_writes_outputs src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_planner_row_launch_gate_accepts_snerv_lf_hf_replacement_queue_rows src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_planner_row_launch_gate_rejects_snerv_lf_hf_runnable_without_command src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_planner_row_launch_gate_accepts_snerv_lf_hf_blocked_unblock_command src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_planner_row_launch_gate_rejects_snerv_lf_hf_unbound_unblock_command src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_planner_row_launch_gate_rejects_snerv_lf_hf_missing_training_contract src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_planner_row_launch_gate_rejects_snerv_lf_hf_unbound_training_contract -q`
  - 45 passed
- `uv run ruff check src/tac/analysis/snerv_lf_hf_replacement_queue.py tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_snerv_lf_hf_replacement_queue.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py`
  - All checks passed.
- `uv run python tools/lane_maturity.py validate`
  - 1685 lanes validated cleanly.
- `git diff --check -- src/tac/analysis/snerv_lf_hf_replacement_queue.py tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_snerv_lf_hf_replacement_queue.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py .omx/state/lane_registry.json .omx/state/lane_maturity_audit.log .omx/research/codex_session_summary_20260606T005436Z_nerv_family_optimal_fail_closed_queue_codex.md`
  - Passed with no output.
- Parsed v44 `next_unblock_command_argv` through
  `tools.run_compact_renderer_mlx_spine_runner._parse_args` and verified
  `_planner_row_launch_blockers(args) == []` with matched mode `unblock`.

## 2026-06-06 Continuation: LF/HF Runtime Binding Proof Consumed

Added the receiver-runtime binding handoff for SNeRV LF/HF payload families.
This is the bridge between a payload proof and a runner-consumable runtime
contract: the proof re-opens exact payload bytes, verifies size and SHA-256
against the payload proof, decodes through the family NumPy receiver module,
and records decoded shape/statistics. It remains false-authority.

Changed:

- `src/tac/analysis/snerv_lf_hf_runtime_binding.py`
- `tools/build_snerv_lf_hf_runtime_binding_proof.py`
  - New runtime-binding proof schema:
    `snerv_lf_hf_runtime_binding_proof.v1`.
  - Supports LF-conditioned HF residual, joint LF/HF codebook, temporal LF
    predictor, tiny-anchor LF SR, spectral band allocator, and LF latent
    hyperprior payload proofs.
- `src/tac/analysis/snerv_lf_hf_replacement_queue.py`
- `tools/build_snerv_lf_hf_replacement_queue.py`
- `src/tac/analysis/nerv_long_training_campaign_plan.py`
- `tools/build_nerv_long_training_campaign_plan.py`
  - Runtime-binding proof is now consumed by normal queue/campaign-plan
    surfaces.
  - Rows with payload proofs now emit a runtime-binding unblock command.
  - Runtime blockers clear only after the runtime proof is consumed, then rows
    remain blocked at bounded-training/renderer gates.

New runtime-binding artifact:

`/Volumes/VertigoDataTier/pact/experiments/results/snerv_lf_hf_runtime_binding_v1_20260606Tcodex/snerv_lf_hf_runtime_binding_proof.json`

SHA-256:

`057b9c729c98fec6c4f1f18e4a570cbbccd2e59ab7cb27e7f8db9fedfae8d19e`

Runtime rows: 6

Closed runtime blockers:

- `snerv_temporal_lf_predictor_receiver_runtime_binding_missing`
- `snerv_lf_super_resolution_receiver_runtime_binding_missing`
- `snerv_score_tethered_lf_hf_band_allocator_runtime_binding_missing`
- `snerv_lf_latent_hyperprior_runtime_binding_missing`

Latest consumed LF/HF queue:

`/Volumes/VertigoDataTier/pact/experiments/results/snerv_lf_hf_replacement_queue_v42b_runtime_binding_consumed_20260606Tcodex/snerv_lf_hf_replacement_queue.json`

SHA-256:

`9feddc2547387d3aac36a083d51e2c5ce7240b3416608f6b172f7746be6450bd`

v42b state:

- SNeRV LF/HF queue rows: 21
- SNeRV LF/HF blocked rows: 21
- SNeRV LF/HF local executable rows: 0
- Score claim: false
- Rank/kill eligible: false
- Ready for exact eval: false

Remaining blockers after v42b:

- `snerv_renderer_nondegenerate_smoke_missing`
- `snerv_renderer_nondegenerate_smoke_min16_pairs_missing`
- `snerv_renderer_nondegenerate_telemetry_contract_missing_or_failed`
- `snerv_score_aware_long_training_direct_live_segnet_candidate_argmax_collapsed`
- `snerv_lf_conditioned_hf_bounded_training_binding_missing`
- `snerv_joint_lf_hf_bounded_training_binding_missing`
- `snerv_temporal_lf_predictor_bounded_training_binding_missing`
- `snerv_lf_super_resolution_bounded_training_binding_missing`
- `snerv_score_tethered_lf_hf_band_allocator_bounded_training_binding_missing`
- `snerv_lf_latent_hyperprior_bounded_training_binding_missing`

Validation for this continuation:

- `uv run pytest src/tac/tests/test_snerv_lf_hf_runtime_binding.py src/tac/tests/test_snerv_lf_hf_replacement_queue.py src/tac/tests/test_nerv_long_training_campaign_plan.py::test_long_training_campaign_plan_threads_source_forward_artifact_into_lf_hf_queue src/tac/tests/test_nerv_long_training_campaign_plan.py::test_build_long_training_campaign_plan_cli_writes_outputs -q`
  - 39 passed
- `uv run ruff check ...`
  - All touched files passed.

## 2026-06-06 Continuation: Renderer-Unblock Queue Feedback Consumed

The v44 renderer-unblock command was rerun through the normal compact runner
path after binding the missing train-time scorer controls. Native long-training
telemetry reached 128 rows, final epoch 127, and proved the SNeRV scorer tethers
are no longer missing or inactive:

- `dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill = 0.0`
- `dual_ascent_missing_metric__snerv_segnet_last_frame_distill = 0.0`
- `dual_ascent_lambda__snerv_posenet_yuv6_pair_distill = 6.0`
- `dual_ascent_lambda__snerv_segnet_last_frame_distill = 6.0`

The run still failed closed on renderer nondegeneracy. The native terminal report
recorded direct-live SegNet candidate argmax collapse and target-class coverage
collapse, while the wrapper timed out during post-native finalization and wrote
an interrupted top-level report instead of a byte-closed export claim.

Native terminal report:

`/Volumes/VertigoDataTier/pact/experiments/results/snerv_lf_hf_replacement_queue_v44_renderer_unblock_contract_20260606Tcodex/snerv_lf_hf_replace_snerv_snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_skchannelm_95bef8d6f21f/bounded_smoke_retry2_deadcontrol_fixed/snerv_mlx_native_export/native_train_export/snerv_score_aware_long_training/snerv_score_aware_long_training.json`

Recovered top-level feedback row:

`/Volumes/VertigoDataTier/pact/experiments/results/snerv_lf_hf_replacement_queue_v44_renderer_unblock_contract_20260606Tcodex/snerv_lf_hf_replace_snerv_snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_skchannelm_95bef8d6f21f/bounded_smoke_retry2_deadcontrol_fixed/nerv_candidate_byte_feedback_row.json`

The feedback row is false-authority and explicitly keeps:

- `score_claim = false`
- `ready_for_exact_eval_dispatch = false`
- `snerv_scorer_domain_tether_passed = true`
- `snerv_scorer_input_distribution_guard_proof_passed = true`
- `snerv_renderer_nondegenerate_proof_passed = false`

Built v46 by replaying the full v42b evidence set plus the recovered retry2
feedback row.

Latest SNeRV LF/HF queue:

`/Volumes/VertigoDataTier/pact/experiments/results/snerv_lf_hf_replacement_queue_v46_renderer_unblock_feedback_consumed_20260606Tcodex/snerv_lf_hf_replacement_queue.json`

SHA-256:

`7aa1d102a5f01fdeb1d9f8a613cf0b3034b75a9c8c4b4afeea1e8caf156652a9`

v46 state:

- SNeRV LF/HF queue rows: 21
- SNeRV LF/HF blocked rows: 21
- SNeRV LF/HF normal executable rows: 0
- Renderer unblock command rows: 3
- Score claim: false
- Rank/kill eligible: false
- Ready for exact eval: false

v46 consumed evidence:

- LF measured payload reports: 2
- Candidate feedback rows: 9
- Runtime-binding proof rows: 6
- Source-forward artifact: `snerv_official_mfu_hfr_tub_forward_parity.v1`
- Official TUB authority gate: `snerv_official_tub_lf_hf_replacement_authority_gate.v1`
- Payload proofs for HF residual, joint codebook, temporal LF predictor,
  tiny-anchor LF SR, spectral allocator, and LF latent hyperprior

v46 remaining blockers are now concentrated where they should be: renderer
nondegeneracy/bounded-training binding. The row with recovered telemetry carries
`snerv_score_aware_long_training_direct_live_segnet_candidate_argmax_collapsed`
and `snerv_renderer_nondegenerate_telemetry_contract_missing_or_failed`; no
blocked long-training row is represented as normal-launchable.

## 2026-06-06 Continuation: HiNeRV Short-Scorer Readiness Control Closure

The broad queue/source-forward/HiNeRV validation pass exposed a real HiNeRV
dead-control gap: `short_scorer_readiness.py` consumed
`segnet_direct_live_target_min_ratio_floor_weight`, but
`HiNervTrainTimeControlConfig` did not own or forward that control. This made
readiness fail from an attribute error instead of a typed blocker or pass.

Closed by mirroring the existing target-mass-floor path:

- train-time control field
- nonnegative validation
- metadata emission
- `RendererBundle` forwarding
- train-time dual-ascent config forwarding
- CLI argument
- PR95 full-control class-escape aggregate

Also corrected the stale direct-full refusal test expectation: the default
HiNeRV SegNet objective is already `boundary_argmax_hinge`, so a direct full
refusal should not claim that objective is missing. The refusal remains blocked
by the canonical runner and PR95 control contracts.

Validation after the fix:

- `uv run pytest ...failing HiNeRV readiness/refusal subset... -q`
  - 12 passed
- `uv run pytest src/tac/tests/test_train_substrate_hi_nerv_mlx_local.py src/tac/substrates/hi_nerv/tests/test_short_scorer_readiness.py -q`
  - 94 passed
- `uv run pytest src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py src/tac/tests/test_snerv_lf_hf_replacement_queue.py src/tac/tests/test_snerv_lf_hf_runtime_binding.py src/tac/tests/test_snerv_official_source_forward_harness.py src/tac/tests/test_snerv_official_tub_lf_hf_replacement_authority_gate.py src/tac/tests/test_snerv_official_tub_source_forward_replay.py src/tac/tests/test_train_substrate_hi_nerv_mlx_local.py src/tac/substrates/hi_nerv/tests/test_short_scorer_readiness.py -q`
  - 528 passed
