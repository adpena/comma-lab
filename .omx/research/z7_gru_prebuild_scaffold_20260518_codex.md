# Z7-GRU prebuild scaffold

Date: 2026-05-18
Author: Codex

## Summary

`time_traveler_l5_z7_lstm_predictive_coding` no longer fails queue readiness
as an absent trainer/package. This pass added a narrow Catalog #240 prebuild
scaffold while preserving `research_only=true`, `dispatch_enabled=false`,
`score_claim=false`, and `promotion_eligible=false`.

## Artifacts

```text
src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/__init__.py
src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/architecture.py
experiments/train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py
src/tac/tests/test_z7_lstm_predictive_coding_scaffold.py
tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py
.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_lstm_predictive_coding_modal_t4_dispatch.yaml
scripts/remote_lane_substrate_time_traveler_l5_z7_lstm_predictive_coding.sh
src/tac/tests/test_time_traveler_l5_z7_remote_driver.py
```

## Contract

The scaffold exports `GruRecurrentPredictor` and
`Z7GruPredictiveCodingConfig`. The predictor matches the Z6-style signature:

```text
forward(z_prev: (B, latent_dim), ego_motion: (B, ego_motion_dim)) -> (B, latent_dim)
```

Implemented smoke checks:

- predictor instantiation;
- forward shape compatibility;
- identity-predictor zero-parameter control;
- gradient flow to latent and ego-motion inputs;
- stateful recurrent hidden-state behavior;
- false-authority smoke JSON emission.

## Blockers Preserved

The first trainer/package scaffold pass renamed the absent-implementation
blockers to precise prebuild blockers. The later full-main export smoke
superseded the `_full_main` hard blocker, and the score-aware smoke lift
superseded the proxy-only blocker:

```text
z7_score_aware_one_pair_smoke_not_contest_authority
z7_score_aware_packet_requires_paired_exact_eval
z7_dispatch_requires_wave_n_plus_1_council_after_z6_4c_outcome
z7_beta_ib_parameter_requires_c6_ibps_phase2_empirical_beta_anchor
z7_wave2_probe_requires_paired_exact_eval_json_from_probe_z7_temporal_coherence_vs_static_capacity_disambiguator
z7_same_archive_bytes_identity_control_exact_eval_required
```

The byte grammar was then advanced from package-only scaffold to a deterministic
Z7PCWM1 parser/replay scaffold:

```text
src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/archive.py
```

Current archive-surface facts:

- `Z7PCWM1_MAGIC=b"Z7GR"`.
- `pack_archive(...)` is deterministic.
- `parse_z7pcwm1_archive_bytes(...)` returns section offsets for header,
  encoder, decoder, predictor, latent init, residuals, ego-motion, and meta.
- `replay_latent_sequence(...)` consumes the parsed GRU predictor bytes; mutating
  the predictor state changes the replayed latent sequence.
- The metadata explicitly carries `score_claim=false`,
  `promotion_eligible=false`, and `ready_for_paid_dispatch=false`.

After the archive grammar scaffold and scorer-free inflate scaffold landed, the
remaining runtime/training blockers became more precise:

```text
z7_score_aware_one_pair_smoke_not_contest_authority
z7_score_aware_packet_requires_paired_exact_eval
```

The runtime scaffold lives at:

```text
src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/inflate.py
```

It replays parsed GRU latents, consumes the Z6-compatible decoder weight stream,
and writes contest-shaped raw RGB output. This is runtime-closure progress only,
not score or dispatch authority.

## Full-Main Export Smoke Lift

The Z7 trainer `_full_main` now emits a byte-closed prebuild packet instead of a
hard prebuild exception. New surfaces:

```text
src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/architecture.py
experiments/train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py
```

The model binds the GRU recurrent predictor to a Z6-compatible RGB decoder,
learned latent init, residual stream, and ego-motion buffer. The trainer decodes
real `upstream/videos/0.mkv` pairs, performs a proxy reconstruction/IB smoke,
then exports `0.bin`, deterministic `archive.zip`, and a scorer-free runtime
tree. This first artifact remains false-authority because it used the proxy loss
and no paired exact eval exists.

Local artifact produced on 2026-05-18:

```text
experiments/results/z7_gru_prebuild_export_codex_20260518T1226Z/
```

Evidence:

```text
archive_bin_bytes=5757
archive_bin_sha256=f7c1dd5936b9f530616fbf7e23a7a017367c01fa391dd8149fcadc498a3278c4
archive_zip_bytes=5178
archive_zip_sha256=429edc1657937d93d4116f094ac2ede66059c9a0f88acd330462784c5c4b560d
final_loss_proxy=0.3054906129837036 [trainer-proxy; not contest-CPU or contest-CUDA]
score_claim=false
promotion_eligible=false
ready_for_paid_dispatch=false
```

Runtime closure bug found and fixed: the first emitted runtime failed under its
default `python3` because the Z7 archive parser depended on external `brotli`.
`Z7PCWM1` now uses stdlib `zlib` for state-dict section compression. Rebuilt
runtime passed the contest-style signature:

```text
submission_runtime/inflate.sh <archive_dir> <output_dir> <file_list>
raw_bytes=12208032
raw_sha256=78c365731c2023b109e2fad08d205ffa1afffe6f8aada72c2734801720cccc61
```

The same run now emits the static-capacity control arm needed by the future
paired exact-eval disambiguator:

```text
static_control_archive_zip_bytes=5178
static_control_archive_zip_sha256=a74a4ce637e82d8340e5e854ec2a3f2b3296552c8cc7268bc367a921bc7f0c7a
same_archive_zip_bytes_as_recurrent=true
zip_comment_padding_bytes=1033
static_control_raw_bytes=12208032
static_control_raw_sha256=166ef65b0474ef05779eec5b68bc250526ed4501f934ad452a0380d296381079
runtime_output_changed_vs_recurrent=true
runtime_output_byte_differences_vs_recurrent=2150198
```

This closes the local missing-control-arm bug only. It does not satisfy the
future disambiguator: both recurrent and static-control packets still need
paired exact-eval JSON on the same contest axis before any method verdict.

The older `z7_dispatch_requires_z6_wave2_candidate_4c_paired_exact_eval_outcome`
blocker remains recorded as superseded by the closed Candidate 4c exact-eval
outcome in the readiness tooling.

## Score-Aware Smoke Lift

Z7 now has an opt-in compress-time scorer path:

```text
--loss-mode score_aware
src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/score_aware_loss.py
```

The loss routes reconstructed pairs through
`score_pair_components_dispatch(...)`, applies the canonical eval-roundtrip
inside training, and leaves the emitted runtime scorer-free. The packed
Z7PCWM1 metadata now records whether the packet was trained with
`score_aware_scorer_loss_used=true`; if so, it replaces the older
`score_aware_training_absent_prebuild` blocker with
`score_aware_trained_packet_not_auth_eval_validated`.

Local artifact produced on 2026-05-18:

```text
experiments/results/z7_gru_score_aware_smoke_codex_20260518T1255Z/
archive_bin_bytes=5797
archive_bin_sha256=3516118e0ed9660f84b3289c993bef2544959caa7c437d081bcf9c5a3de7d1f4
archive_zip_bytes=5155
archive_zip_sha256=a8a4dcc3aebb4d37317a0b1afa4bec129ef04f531584a3012dbef9616914a92b
loss_mode=score_aware
final_loss=39.47663497924805 [trainer-score-aware local CPU smoke; not contest-CPU or contest-CUDA]
seg_term=0.3863946497440338
pose_term=0.06964391469955444
rate_term=0.002638799138367176
score_claim=false
promotion_eligible=false
ready_for_paid_dispatch=false
```

Same-byte static control from the same run:

```text
static_control_archive_zip_bytes=5155
static_control_archive_zip_sha256=2294467c9813e9497576b958c573aa5766066254893e7a8fea86e8c9db3e02c3
same_archive_zip_bytes_as_recurrent=true
zip_comment_padding_bytes=1034
recurrent_raw_sha256=8d0194a147fe41e14b25e2e9b1a1cca952d3252c1e567363191457cda8b54974
static_control_raw_sha256=677916b82b6ee883b6254d3563bdccf7000b319aeaf249bdc5531e053e6c7641
runtime_output_changed_vs_recurrent=true
runtime_output_byte_differences_vs_recurrent=1011870
```

Timing smoke from the same artifact:

```text
timing_axis=[local-trainer-timing advisory]
device=cpu
epochs=1
num_pairs=1
seconds_per_epoch=1.3555798749439418
seconds_per_pair_epoch=1.3555798749439418
pairs_per_second_epoch=0.7376916834512268
decode_resize_seconds=0.09331295802257955
score_aware_scorer_load_seconds=0.5821676668711007
train_total_seconds=1.3555798749439418
export_packaging_seconds=0.002386040985584259
inflate_verify_seconds=0.12976029119454324
total_wall_seconds=2.1652982090599835
```

This is local CPU timing only. It is not provider/T4 spend authority, but it
converts the next Z7 step from an unbounded idea into a measured smoke gate:
the future T4 smoke must report the same timing schema before a full 600-pair
run or paired exact-eval handoff is authorized.

Bug found during this lift: the one-pair smoke produced NaNs because
`_ego_motion_from_pairs(...)` used unbiased `std()` with one sample. It now uses
population std (`unbiased=false`) so one-pair timing/smoke probes can run without
poisoning the recurrent decoder.

## Queue Consequence

The queue should now report:

```text
trainer_path=experiments/train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py
full_main_implemented=true
full_main_blocker=null
readiness_verdict=DEFER
top_blocker=z7_score_aware_one_pair_smoke_not_contest_authority
```

Refreshed queue artifact:

```text
.omx/state/asymptotic_pursuit/dispatch_queue_20260518T131951Z.json
```

This is not a score or dispatch claim. The next valid score-moving gate is a
Wave N+1-authorized full trainer plus a trained Z7PCWM1 byte-closed packet,
followed by same-archive-bytes paired exact-eval disambiguator evidence on the
proper contest axes.

## Z7 disambiguator probe custody wire-in

Bug class closed: the asymptotic-pursuit readiness helper understood the Z6
identity-disambiguator schema only. Z7's recipe already carried an explicit
`z7_wave2_probe_requires_paired_exact_eval_json_from_probe_z7_temporal_coherence_vs_static_capacity_disambiguator`
blocker, but the queue had no structured probe custody for that blocker. The
helper now accepts the Z7 temporal-vs-static schema as a distinct
fail-closed local disambiguator surface.

New probe artifact:

```text
.omx/research/probe_z7_temporal_coherence_vs_static_capacity_disambiguator_20260518_codex.json
schema=z7_temporal_coherence_vs_static_capacity_disambiguator_v1
verdict=pending_paired_exact_eval_json
score_claim=false
promotion_eligible=false
ready_for_paid_dispatch=false
decision_rule.same_archive_bytes_required=true
decision_rule.required_axis=contest_cuda
required_inputs=z7_recurrent_exact_eval_json, static_capacity_control_exact_eval_json
```

Queue consequence:

```text
local_identity_disambiguator_probe.path=/Users/adpena/Projects/pact/.omx/research/probe_z7_temporal_coherence_vs_static_capacity_disambiguator_20260518_codex.json
local_identity_disambiguator_probe.verdict=pending_paired_exact_eval_json
local_identity_disambiguator_probe.blockers=[]
dispatch_blocker_supersessions=z7_dispatch_requires_z6_wave2_candidate_4c_paired_exact_eval_outcome
ready_for_paid_dispatch=false
```

This does not remove the Z7 exact-eval blockers. It prevents signal loss by
making the future recurrent-vs-static control comparison machine-readable in
the queue before any Wave N+1 spend decision.

## Catalog #202 sentinel hygiene fix

Adversarial queue review found a launch-precondition parser bug: list-valued
`required_input_files_trainer` entries could be stringified into a literal
`['experiments/...', 'src/...']` missing path inside the Catalog #202 sentinel
snapshot. The sentinel collector in both `tools/operator_authorize.py` and
`tools/audit_catalog202_sentinel_cleanliness.py` now flattens list/tuple path
values and preserves `.omx/...` paths instead of normalizing them to
`omx/...`.

Z7 recipe adjustment:

```text
required_input_files_trainer=experiments/train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py
sentinel_files=tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py
operator-side research docs moved to dependencies, not Modal worker sentinels
```

Queue consequence after refresh:

```text
.omx/state/asymptotic_pursuit/dispatch_queue_20260518T131951Z.json
catalog202.current_sentinel_snapshot.effective_sentinel_file_count=6
malformed_list_missing_path_absent=true
outside_modal_mount_sentinel_files=[]
remaining_snapshot_blocker=scripts/remote_lane_substrate_time_traveler_l5_z7_lstm_predictive_coding.sh missing
```

The remaining missing remote driver is a real future launch blocker and was not
papered over. It is harmless while `research_only=true` and
`dispatch_enabled=false`, but it must be landed or the recipe changed before
any Z7 paid dispatch flip.

## Remote-driver sentinel blocker cleared

The missing remote driver is now landed:

```text
scripts/remote_lane_substrate_time_traveler_l5_z7_lstm_predictive_coding.sh
```

Contract:

- refuses to run without an active `lane_per_substrate_symposium_z7_lstm_predictive_coding_20260517`
  dispatch claim for the supplied instance/job id;
- defaults to the one-pair score-aware timing-smoke settings instead of full
  training;
- supports explicit `Z7_GRU_TRAINER_MODE=smoke|timing_smoke|full`;
- writes provenance JSON and quarantines stale preexisting stats before
  trainer launch;
- validates the emitted stats JSON before marking success;
- refuses completion when `score_claim`, `promotion_eligible`, or
  `ready_for_paid_dispatch` is truthy;
- terminalizes every claimed run as `completed_z7_gru_remote_driver_no_score_claim`
  or a precise `failed_z7_gru_*` status.

Focused coverage:

```text
src/tac/tests/test_time_traveler_l5_z7_remote_driver.py
```

Queue consequence after refresh:

```text
.omx/state/asymptotic_pursuit/dispatch_queue_20260518T132746Z.json
top_1_substrate=time_traveler_l5_z7_lstm_predictive_coding
readiness_verdict=DEFER
ready_for_paid_dispatch=false
paid_launch_command=null
catalog202.current_sentinel_snapshot.effective_sentinel_file_count=7
catalog202.current_sentinel_snapshot.missing_sentinel_files=[]
catalog202.current_sentinel_snapshot.outside_modal_mount_sentinel_files=[]
catalog202.current_sentinel_snapshot.snapshot_blockers=[]
catalog202.current_sentinel_snapshot_valid=true
```

This clears the Catalog #202 missing-file blocker only. Z7 remains
non-dispatchable for the binding evidence gates:

```text
CATALOG_315_COUNCIL_PROCEED_WITH_REVISIONS_NEEDS_ITERATION_TO_OPTIMAL_FORM
RECIPE_research_only=true_OPERATOR_NEEDS_TO_FLIP_AFTER_PHASE_2_COUNCIL
z7_score_aware_one_pair_smoke_not_contest_authority
z7_score_aware_packet_requires_paired_exact_eval
z7_dispatch_requires_wave_n_plus_1_council_after_z6_4c_outcome
z7_beta_ib_parameter_requires_c6_ibps_phase2_empirical_beta_anchor
z7_wave2_probe_requires_paired_exact_eval_json_from_probe_z7_temporal_coherence_vs_static_capacity_disambiguator
z7_same_archive_bytes_identity_control_exact_eval_required
```

No provider job was launched, no lane claim was opened, and this remains
`score_claim=false`, `promotion_eligible=false`,
`ready_for_paid_dispatch=false`.

## Exact-eval handoff doctor and literature-informed design hardening

The current same-byte recurrent/static-control Z7 packet is now classified by:

```text
tools/verify_z7_exact_eval_handoff.py
src/tac/tests/test_verify_z7_exact_eval_handoff.py
.omx/state/z7_exact_eval_handoff/z7_exact_eval_handoff_20260518T133855Z.json
```

Current verdict:

```text
ready_for_exact_eval_handoff=false
current_pair_count=1
required_pair_count=600
result_review_blockers=[z7_exact_handoff_current_packet_not_600_pairs]
same_archive_zip_bytes=true
runtime_output_changed_vs_recurrent=true
runtime_custody.aggregate_sha256=d7a3297e011dfc1d271b9e89997a3eda74271fa4237d577d4a7dd4f90e178d2f
```

The tool emits no executable Modal command for this one-pair packet, and it
suppresses even plan commands if false-authority fields become truthy.

Online literature check recorded in:

```text
.omx/research/z7_exact_eval_handoff_and_online_lit_review_20260518_codex.md
```

Key design consequence: DCVC-style contextual coding is a sharper next target
than plain predictive residual coding. The next optimized Z7 packet should
compare recurrent latent residual baseline against context-conditioned decoder
features and context-conditioned residual-symbol scales, under same archive
bytes and paired axes.

## Context-conditioned decoder branch

The first distinct Z7 contextual-coding branch is now implemented as an opt-in
runtime-consumed mode, not a cargo-cult label:

```text
--context-conditioning-mode none|latent_affine
--context-affine-strength <float>
```

`none` preserves the prior Z7-GRU baseline. `latent_affine` computes the
pre-residual recurrent prediction context, stores a tiny affine conditioner in
the existing Z7PCWM1 encoder section, and applies that conditioner at inflate
time before the Z6-compatible decoder consumes latents. Old archives default to
`none`; context-conditioned archives fail closed if the conditioner state is
missing.

Code and tests:

```text
src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/architecture.py
src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/archive.py
src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/inflate.py
experiments/train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py
scripts/remote_lane_substrate_time_traveler_l5_z7_lstm_predictive_coding.sh
src/tac/tests/test_z7_lstm_predictive_coding_scaffold.py
src/tac/tests/test_time_traveler_l5_z7_remote_driver.py
```

Local no-score artifact:

```text
experiments/results/z7_gru_context_affine_score_aware_smoke_codex_20260518T135435Z/
archive_bin_bytes=6315
archive_bin_sha256=350d5be5c78f71be617e2fd595cd4f346930911883dc26553d3720648a6b541e
archive_zip_bytes=5450
archive_zip_sha256=09bfa54ec5dea705b903d6086d71eb972142447ef0ec370f9d995bddf0a503ca
context_conditioner_params=84
loss_mode=score_aware
final_loss=48.95578384399414 [local-trainer-timing advisory; not contest-CPU or contest-CUDA]
inflate_verify_raw_sha256=e812787d006cf8e9ba53ae7e85889cabed511e09cde9113ecd5b316ba9cfb496
static_control_archive_zip_sha256=ead3938bdc3046d785f5500c68710ec8dab1386220cd490c67eb2648e9c82b62
same_archive_zip_bytes_as_recurrent=true
static_control_raw_sha256=c16057501f0393b782eeb9d60bd9a670c1f0f50fe98194e618ef5034b6167a3b
runtime_output_changed_vs_recurrent=true
runtime_output_byte_differences_vs_recurrent=1056252
score_claim=false
promotion_eligible=false
ready_for_paid_dispatch=false
```

Adversarial interpretation: this proves a byte-closed contextual decoder
mechanism exists and is consumed by runtime. It does not prove that contextual
coding lowers the contest score, because the packet is one pair, local CPU,
small-resolution, and not exact-evaled. The next independent-design step is to
compare `none`, `latent_affine`, and a future entropy-scale/residual-symbol
context mode under the same archive bytes, then promote only through paired
exact CPU/CUDA artifacts.
