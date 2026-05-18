# L5 v2 Z6 identity-predictor disambiguator

- schema: `z6_predictive_coding_vs_identity_disambiguator_v1`
- probe_id: `z6_predictive_coding_vs_identity_disambiguator`
- lane_id: `lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516`
- evidence_grade: `byte_closed_archive_pair_no_score`
- verdict: `pending_paired_exact_eval_json`
- paired_control_initialization: `shared_modules_seed_order_matched_v2`
- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- ready_for_paid_dispatch: `false`
- paradigm_claim_allowed: `false`

This report is a Z6-specific probe surface. It can route the next engineering action, but it is not contest score evidence.

## Source Archives

### full_film_predictor

- path: `experiments/results/z6_candidate4c_full600_zeroepoch_packet_codex_20260518T104201Z/0.bin`
- bytes: `213864`
- sha256: `0f448055c3147d7cae865e31a8d40ae441617e3aa20cd5c38214f14c26957778`
- zip_path: `experiments/results/z6_candidate4c_full600_zeroepoch_packet_codex_20260518T104201Z/archive.zip`
- zip_bytes: `211866`
- zip_sha256: `5b371490b4459b85e95e6173653fc1b9aa78010681862ec51111166a6c867c4b`
- zip_members: `['0.bin']`
- zip_member_matches_path_bytes: `True`
- contest_archive_bytes_basis: `211866`
- identity_predictor: `False`
- identity_predictor_disambiguator: `False`
- predictor_state_dict_key_count: `8`
- num_pairs: `600`
- ego_motion_dim: `8`
- predictor_architecture: `single_layer_film_75k`

### identity_predictor

- path: `experiments/results/z6_candidate4c_full600_zeroepoch_packet_codex_20260518T104201Z/0_identity_predictor_disambiguator.bin`
- bytes: `214183`
- sha256: `8c274c5a11b38a9f43e3ee54bd03f47743de8e88509705e84a7b6dfa2035c9b8`
- zip_path: `experiments/results/z6_candidate4c_full600_zeroepoch_packet_codex_20260518T104201Z/archive_identity_predictor_disambiguator.zip`
- zip_bytes: `212047`
- zip_sha256: `e6cd9bf67ca68bcdf93aa0e804435b75b813e420d5e3964b3a6cb6cee28e3589`
- zip_members: `['0.bin']`
- zip_member_matches_path_bytes: `True`
- contest_archive_bytes_basis: `212047`
- identity_predictor: `True`
- identity_predictor_disambiguator: `True`
- predictor_state_dict_key_count: `8`
- num_pairs: `600`
- ego_motion_dim: `8`
- predictor_architecture: `single_layer_film_75k`

## Paired Archive Checks
- encoder_state_dict_equal: `True`
- decoder_state_dict_equal: `True`
- predictor_state_dict_equal: `True`
- predictor_keysets_equal: `True`
- latent_init_equal: `True`
- residuals_equal: `True`
- ego_motion_equal: `True`

## Runtime Custody
- schema: `z6_inflate_runtime_closure_v1`
- runtime_root: `experiments/results/z6_candidate4c_full600_zeroepoch_packet_codex_20260518T104201Z/submission_dir`
- entrypoint: `experiments/results/z6_candidate4c_full600_zeroepoch_packet_codex_20260518T104201Z/submission_dir/inflate.sh`
- aggregate_sha256: `384938f3b6a14acb5944938c45c127ccc411f00983aff7589c7c9b98cfc56073`
- file_count: `10`
- total_bytes: `62261`
- archive_payloads_excluded_from_runtime_hash: `['0.bin', '0_identity_predictor_disambiguator.bin', 'archive.zip', 'archive_identity_predictor_disambiguator.zip']`
- python_cache_files_excluded_from_runtime_hash: `8`

## Inflate Output Comparison
- evidence_axis: `[local-inflate-output advisory]`
- evidence_grade: `local_inflate_output_comparison_no_score`
- runtime_output_changed: `True`
- same_output_file_set: `True`
- same_output_aggregate_sha256: `False`
- total_byte_differences: `33048720`
- inflate_sh_path: `experiments/results/z6_candidate4c_full600_zeroepoch_packet_codex_20260518T104201Z/submission_dir/inflate.sh`
- file_list_path: `upstream/public_test_video_names.txt`
- output_root: `experiments/results/z6_candidate4c_full600_zeroepoch_packet_codex_20260518T104201Z/z6_identity_inflate_output_comparison_runtime_source_custody_compact_20260518_codex`
- full_output_aggregate_sha256: `241f9cf0d6234a728a165173e0f352beb5254d358dacf0e6d7ff027b0f58c712`
- identity_output_aggregate_sha256: `5c0673169daabf7a90cddaa86b23b157019f96c63f68daa36eed786be368d94e`
- full_output_total_bytes: `3662409600`
- identity_output_total_bytes: `3662409600`

## Deltas
- identity_minus_full_archive_bytes: `319`
- identity_minus_full_parsed_member_bytes: `319`
- identity_minus_full_zip_bytes: `181`
- identity_minus_full_contest_archive_bytes_basis: `181`
- identity_minus_full_rate_term_basis: `0.00012052047051511301`

## Blockers
- `no_paired_exact_eval_json`
- `no_contest_cpu_cuda_pair`
- `not_score_authority`

## Reactivation Criteria
- provide both paired contest_auth_eval JSON files for the exact same ZIP sidecars
- keep axis labels adjacent to all score language
- treat full_minus_identity_score <= -decision_delta_s as a full-FiLM win
- do not promote, rank, or kill from this probe without operator review

## 2026-05-18 partial exact-eval update

The paired Modal exact-eval lifecycle was opened for the full-600 archive pair.
Both `[contest-CUDA]` calls recovered successfully; both `[contest-CPU]` calls
are still pending.

Recovered `[contest-CUDA]` artifacts:

| mode | score | seg | pose | bytes | runtime tree | inflated output aggregate |
|---|---:|---:|---:|---:|---|---|
| full FiLM predictor | `90.58142803863508` | `0.50482631` | `159.66197205` | `211866` | `09f53c44b021314639a94f48b4ceb143fe3b279ab88e5c0b2dc19ad27a3ae242` | `c3a5b9c582ff81ed43305bb959195d0cbb86a02503ec882135d2c11a4420152b` |
| identity predictor | `90.58427695093009` | `0.50482631` | `159.68377686` | `212047` | `09f53c44b021314639a94f48b4ceb143fe3b279ab88e5c0b2dc19ad27a3ae242` | `8a6cc8524f377a89732adae9f91851be268a26da3a6bf412219c55856936d879` |

`[contest-CUDA]` identity-minus-full:

- score: `0.0028489122950077217`
- SegNet: `0`
- PoseNet: `0.021804809999991903`
- archive bytes: `181`

Interpretation:

- This is an exact `[contest-CUDA]` anchor for the zero-epoch control packet,
  not a trained Candidate 4c result.
- Full FiLM is lower than identity on `[contest-CUDA]`, but the delta is below
  the configured `decision_delta_s=0.005`; the disambiguator therefore remains
  unresolved until `[contest-CPU]` lands and/or trained Candidate 4c archives
  are evaluated.
- Both scores are catastrophically above the frontier, so the zero-epoch packet
  is a measured bad configuration and not promotion material.
- Local inflate-output hashes remain advisory only: Modal `[contest-CUDA]`
  inflated-output aggregates differ from the local macOS advisory aggregates.

Pending `[contest-CPU]` calls:

- full: `fc-01KRXC4EZ3GY615KF1EJE33VZ2`
- identity: `fc-01KRXC4M333B38CRV7Q6HXNVN1`

Current blockers:

- `contest_cpu_pending`
- `contest_cuda_delta_below_decision_threshold`
- `zero_epoch_control_not_candidate_training_result`
- `not_promotion_authority`

## 2026-05-18 paired exact-eval completed

Both `[contest-CPU]` calls recovered after the partial CUDA update. The exact
eval pair is now complete across both axes.

Recovered `[contest-CPU]` artifacts:

| mode | score | seg | pose | bytes | runtime tree | inflated output aggregate |
|---|---:|---:|---:|---:|---|---|
| full FiLM predictor | `90.57816474855734` | `0.5048244` | `159.63742065` | `211866` | `f888831ce915d37c2a6a3069646a02291569a948a1c2108b1c9ff76b1e3ab57b` | `3aa8201a294d3d01aac5bb0cc719bacc3f55153e862c253714c72f2783293e2a` |
| identity predictor | `90.58102532784203` | `0.5048244` | `159.65931702` | `212047` | `f888831ce915d37c2a6a3069646a02291569a948a1c2108b1c9ff76b1e3ab57b` | `19f86fb4e9249fe122133872788490c43acb437d1dc8f81a5f7a6b3be4b07a57` |

`[contest-CPU]` identity-minus-full:

- score: `0.0028605792846860822`
- SegNet: `0`
- PoseNet: `0.021896370000007437`
- archive bytes: `181`

Final paired interpretation:

- Full FiLM is lower than identity on both `[contest-CUDA]` and
  `[contest-CPU]`.
- Both axis deltas are below `decision_delta_s=0.005`.
- The zero-epoch control is a measured bad configuration on both axes, with
  scores near `90.58`; this is not a trained Candidate 4c method result and
  must not be promoted.
- The useful signal is narrow but real: the predictive path is consumed by
  inflate and nudges PoseNet in the expected direction, but the untrained
  archive is nowhere near a frontier candidate.

Final blockers:

- `contest_cuda_delta_below_decision_threshold`
- `contest_cpu_delta_below_decision_threshold`
- `zero_epoch_control_not_candidate_training_result`
- `not_promotion_authority`
