---
review_kind: research_wave_intake_queue
review_id: research_wave_intake_queue_20260518_codex
review_date: "2026-05-18"
lane_id: lane_deep_research_wave_20260518
source_memo: .omx/research/comprehensive_research_wave_20260518.md
queue_artifact: .omx/state/asymptotic_pursuit/research_wave_intake_20260518T063424Z.json
readiness_artifact: .omx/state/asymptotic_pursuit/readiness_assessment_20260518T063400Z.json
dispatch_queue_artifact: .omx/state/asymptotic_pursuit/dispatch_queue_20260518T063400Z.json
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
---

# Research wave intake queue - 2026-05-18

## Purpose

The completed comprehensive research wave is now converted into a guarded,
machine-readable intake artifact:

- Artifact: `.omx/state/asymptotic_pursuit/research_wave_intake_20260518T063424Z.json`
- Readiness artifact:
  `.omx/state/asymptotic_pursuit/readiness_assessment_20260518T063400Z.json`
- Dispatch queue artifact:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T063400Z.json`
- DP1 L1 no-op/rate proof:
  `.omx/research/dp1_pr101_composition_noop_probe_20260518_codex.json`
- ATW V2-1 byte-closed side-info probe:
  `.omx/research/atw_v2_1_byte_closed_side_info_probe_20260518_codex.json`
  and
  `.omx/research/atw_v2_1_byte_closed_side_info_probe_20260518_codex.md`
- TT5L Lightning doctor output:
  `.omx/research/l5_v2_tt5l_lightning_required_doctor_20260517_codex.json`
- TT5L non-dry-run fail-closed gate:
  `.omx/research/l5_v2_tt5l_lightning_non_dry_run_gate_20260517_codex.json`
- Source memo SHA-256 recorded by artifact:
  `a9e16f905ee6459bae6c9940919599f7314ac8990780875a288d3d06ba457c44`
- Extracted rows: 5
- Current readiness joins: 4
- Evidence grade: `research_intake`
- Score authority: `score_claim=false`, `promotion_eligible=false`,
  `ready_for_paid_dispatch=false`

This is not a dispatch queue and not a ranking/promotion authority. It is an
operator-safe crosswalk from the top research-wave claims to the next
byte-closed gate each candidate needs before any score claim or provider spend.

## Intake crosswalk

| Rank | Intake substrate | Crosswalk substrate_id | Axis | Current readiness | Next gate | Blocking summary |
|---|---|---|---|---|---|---|
| 1 | TT5L V2 | `time_traveler_l5_autonomy` | `[contest-CPU]` | `DEFER` | `resolve_modal_billing_or_lightning_doctor_env_then_stage_manifest_and_claim` | Existing recipe has the C1/Z5/TT5L probe blocker, research-only gating, prediction-band custody blockers, Modal billing blocker, and now a concrete Lightning doctor failure: `ssh_auth`, `remote_supply_chain`, and `machine_inventory`. The byte-closed work unit and Lightning bundle exist, but non-dry-run submit is fail-closed at `0/10` ready cells. |
| 2 | Z7-as-GRU / Mamba | `time_traveler_l5_z7_lstm_predictive_coding` | `[contest-CPU]` | `DEFER` | `scale_z7_score_aware_smoke_to_ratified_packet_then_pair_exact_eval` | Z7-GRU now has predictor/substrate, Z7PCWM1 archive grammar, scorer-free runtime, a real-video proxy full-main export smoke, and a one-pair local score-aware scorer-loss smoke. It remains blocked on contest authority: Wave N+1 council, C6 beta anchor, same-bytes disambiguator, and paired exact-eval JSONs for the Z7 disambiguator. |
| 3 | ATW V2-1 | `atw_codec_v2` | `[contest-CPU]` | `DEFER` | `design_substrate_native_scorer_logit_sketch_or_trained_atw_residual_probe` | V2-1 side-info packets are byte-closed and under budget, but conditioning remains too weak: best `per_region_histogram` packet is 323 bytes with MI `0.047381530305` bits/symbol versus threshold `1.0`; other channels are weak or independent. Existing D4, PROCEED_WITH_REVISIONS, research-only, Dykstra, and variant-adjudication blockers still apply. |
| 4 | DP1 + PR101 composition | `dp1_pr101_composition` | `[contest-CPU]` | `DEFER` | `run_full_frame_parity_or_path2_lambda_prior_disambiguator_after_l1_noop_probe` | Current readiness joins the existing DP1 dual-stack lane. L1 no-op/rate proof landed; remaining blockers are `_full_main` NotImplementedError, research-only gating, variant A/B decision, PATH 2 lambda/prior-effect disambiguator, and pr101_lc_v2 integration premise verification. |
| 5 | lane_17_imp + Frankle LTH | `lane_17_imp` | `[contest-CPU]` | no current readiness row | `run_imp_cycle0_timing_smoke_after_operator_budget_and_claim` | Needs current readiness/recipe surface before budgeted provider work. |

## False-authority blockers carried by every row

- `research_wave_prediction_not_score_authority`
- `requires_byte_closed_archive_or_runtime_probe_before_score_claim`
- `requires_operator_session_directive_budget_and_lane_claim_before_provider_dispatch`
- `requires_paired_contest_cuda_cpu_harvest_before_promotion_or_ranking_claim`
- `prediction_axis_is_contest-CPU_not_contest_cuda`

## Operator-facing implication

The strongest immediate conversion paths are not score claims. They are
artifact-producing next gates:

1. TT5L V2: resolve Modal billing or provide Lightning `LIGHTNING_SSH_TARGET`,
   `LIGHTNING_TEAMSPACE`, and exactly one owner identity; then stage source
   manifests and create per-cell claims before any non-dry-run provider submit.
2. Z7-as-Mamba: recipe/readiness visibility now exists, but remains `DEFER`;
   the next artifact is the trainer/substrate package or the Z7 disambiguator
   design, after Z6 4c and C6 inputs are resolved.
3. ATW V2-1: the byte-closed side-info probe is now landed; it rules out the
   current histogram/argmax reducer family as dispatch-worthy and moves the
   next gate to a substrate-native scorer-logit sketch or a probe on trained
   ATW residuals.
4. DP1+PR101: L1 no-op detector is landed; next is either full-frame fec6-vs-
   composed parity as a control or PATH 2 lambda/prior-effect disambiguation.
5. lane_17_imp: refresh readiness around the old IMP evidence and only then
   run a claimed cycle-0 smoke.

No provider dispatch was launched. No lane claim was needed because this pass
only wrote local queue/intake artifacts and ledgers.

## Race-mode dispatch queue actuator surface

Because `.omx/state/RACE_MODE_ACTIVE.flag` exists, I revalidated the current
TOP-1 actuator path instead of adding another research-only review loop.

Fresh queue artifacts:

- readiness snapshot:
  `.omx/state/asymptotic_pursuit/readiness_assessment_20260518T063400Z.json`
- dispatch queue snapshot:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T063400Z.json`
- research-wave intake snapshot:
  `.omx/state/asymptotic_pursuit/research_wave_intake_20260518T063424Z.json`

Queue facts:

- TOP-1 substrate: `z6_v2_candidate_4c_scorer_logit`
- TOP-1 verdict: `READY`
- `ready_for_paid_dispatch_count`: `1`
- READY queue estimate: `$2.083`
- READY operator session budget floor: `$13.0`
- Candidate 4c blockers: `[]`
- local identity-disambiguator runtime-output-changed: `true`

The dispatch queue now emits both machine-actionable commands:

```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
  --operator-handle codex:z6_candidate4c_scorer_logit \
  --dry-run
```

and, only after explicit operator session budget authorization:

```bash
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=13.000 \
.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
  --operator-handle codex:z6_candidate4c_scorer_logit
```

Non-dry-run execution without those paired env vars exits before claim/provider
work with rc `9`. This keeps race-mode actuation ready without silently
inventing a spend directive.

## ATW V2-1 byte-closed side-info probe

This pass added
`tools/probe_atw_v2_1_byte_closed_side_info_channel.py` with tests in
`src/tac/tests/test_probe_atw_v2_1_byte_closed_side_info_channel.py`, then ran
the probe on the existing A1 latent/reducer evidence.

Probe artifact:
`.omx/research/atw_v2_1_byte_closed_side_info_probe_20260518_codex.json`

Packet/result facts:

| Reducer | Packet bytes | MI bits/symbol | Threshold | Verdict | Rate score cost |
|---|---:|---:|---:|---|---:|
| `per_pixel_histogram` | 204 | `0.022656927447` | `0.5` | `WEAK_CONDITIONING` | `0.000135835226` |
| `per_region_histogram` | 323 | `0.047381530305` | `1.0` | `WEAK_CONDITIONING` | `0.000215072442` |
| `per_pair_class_2_fraction` | 127 | `0.009692520351` | `0.2` | `INDEPENDENT` | `0.000084564087` |
| `per_frame_argmax` | 117 | `0.000000000000` | `0.2` | `INDEPENDENT` | `0.000077905498` |

Interpretation:

- Byte budget is not the blocker; every current side-info channel is <=2KB and
  round-trips through the `ATW21SI` packet decoder.
- Conditioning is the blocker; the best byte-closed channel recovers only
  `0.047381530305` bits/symbol and has a Wyner-Ziv ceiling fraction of
  `0.006731264914`.
- No ATW Modal A100 dispatch is justified from this evidence. The next
  artifact-producing gate is a substrate-native scorer-logit sketch or a
  trained-ATW-residual probe that preserves signal before reducer collapse.

## Verification

- `.venv/bin/python -m py_compile tools/research_wave_intake_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_research_wave_intake_queue.py -q`
- `.venv/bin/python tools/probe_atw_v2_1_byte_closed_side_info_channel.py`
- `.venv/bin/python -m py_compile tools/probe_atw_v2_1_byte_closed_side_info_channel.py`
- `.venv/bin/python -m pytest src/tac/tests/test_probe_atw_v2_1_byte_closed_side_info_channel.py -q`
- `.venv/bin/python -m py_compile tools/asymptotic_pursuit_candidate_readiness_assessment.py tools/research_wave_intake_queue.py`
- `.venv/bin/python -m py_compile tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py`
- `.venv/bin/python -m pytest src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py src/tac/tests/test_research_wave_intake_queue.py src/tac/tests/test_probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py -q`
- `.venv/bin/python tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py` (expected fail-closed plan mode)
- `.venv/bin/python tools/asymptotic_pursuit_candidate_readiness_assessment.py --write-artifact --json`
- `.venv/bin/python tools/asymptotic_pursuit_dispatch_queue.py --write-artifact --json`
- `.venv/bin/python tools/research_wave_intake_queue.py --write-artifact --json`
- `.venv/bin/python tools/validate_dispatch_required_inputs.py --trainer experiments/train_substrate_time_traveler_l5_z6.py --flag-value=--video-path=upstream/videos/0.mkv`
- `.venv/bin/python tools/run_modal_smoke_before_full.py --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch --operator-handle codex:z6_candidate4c_scorer_logit --dry-run`
- `env -u OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE -u OPERATOR_AUTHORIZE_SESSION_BUDGET_USD .venv/bin/python tools/run_modal_smoke_before_full.py --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch --operator-handle codex:z6_candidate4c_scorer_logit --smoke-only` (expected rc=9 fail-closed before claim/provider work)
- `.venv/bin/python -m pytest src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py -q`
- `.venv/bin/python tools/probe_dp1_pr101_composition_noop_detector.py --output .omx/research/dp1_pr101_composition_noop_probe_20260518_codex.json --json`
- `.venv/bin/python scripts/launch_lightning_batch_job.py doctor --json-out .omx/research/l5_v2_tt5l_lightning_required_doctor_20260517_codex.json --run-id l5_v2_tt5l_lightning_required_doctor_20260517 --strict --require-ssh --remote-supply-chain --require-remote-supply-chain --repo-dir /teamspace/studios/this_studio/pact --python-bin .venv/bin/python --require-machine-inventory --machine T4 --gpu-only` (expected fail-closed in this shell)
- `.venv/bin/python tools/check_l5_v2_tt5l_lightning_non_dry_run_gate.py`
- `.venv/bin/python tools/check_l5_v2_tt5l_lightning_non_dry_run_gate.py --strict-ready` (expected rc=1 fail-closed because doctor status is `FAIL`)
- `.venv/bin/python -m py_compile tools/asymptotic_pursuit_candidate_readiness_assessment.py tools/asymptotic_pursuit_dispatch_queue.py tools/research_wave_intake_queue.py tools/probe_dp1_pr101_composition_noop_detector.py`
- `.venv/bin/python -m pytest src/tac/tests/test_probe_dp1_pr101_composition_noop_detector.py src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py src/tac/tests/test_research_wave_intake_queue.py src/tac/tests/test_probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py -q`
- `jq empty .omx/research/atw_v2_1_byte_closed_side_info_probe_20260518_codex.json .omx/state/atw_v2_1_byte_closed_side_info_probe.json experiments/results/atw_v2_1_sideinfo_probe_20260518T062431Z/atw_v2_1_byte_closed_side_info_probe.json`
- `jq empty .omx/research/dp1_pr101_composition_noop_probe_20260518_codex.json .omx/research/l5_v2_tt5l_lightning_required_doctor_20260517_codex.json .omx/research/l5_v2_tt5l_lightning_non_dry_run_gate_20260517_codex.json .omx/state/asymptotic_pursuit/readiness_assessment_20260518T063400Z.json .omx/state/asymptotic_pursuit/dispatch_queue_20260518T063400Z.json .omx/state/asymptotic_pursuit/research_wave_intake_20260518T063424Z.json`
- `.venv/bin/python tools/claim_lane_dispatch.py summary`

## Prediction-band false-authority hardening added

This pass found a subtle planning-surface bug class: the readiness dataclass
uses legacy field names like `predicted_delta_s_band`, while recipe
`predicted_band` values are score-target bands. That is safe only when the
recipe labels the band kind, evidence axis, and validation status.

The readiness artifact now carries these explicit fields per candidate:

- `predicted_score_band`
- `predicted_band_kind`
- `predicted_band_axis`
- `predicted_band_validation_status`
- `predicted_band_metadata_blockers`

If a numeric predicted band is present but any required metadata is missing,
the candidate remains visible but its `ev_per_dollar` is forced to `0.0` and
the artifact records `PREDICTED_BAND_METADATA_BLOCKER:*`. This is a planning
false-authority guard, not a score verdict and not a lane kill.

Current concrete effects:

- Candidate 4c remains `READY`; its band is now labelled
  `predicted_score_band`, axis `contest-CUDA`,
  status `pending_post_training`.
- Z7 remains `DEFER`; its band is now labelled `predicted_score_band`,
  axis `contest-CPU`, status `research_prior_prebuild`.
- The older Z6 row now has zero EV pressure until its recipe/design surfaces
  backfill explicit band kind, axis, and validation status. Its existing DEFER
  blockers were already stronger; this only prevents stale unlabeled priors
  from influencing ordering.

## Z7 readiness surface added

This pass added
`.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_lstm_predictive_coding_modal_t4_dispatch.yaml`
as a non-dispatchable research-only recipe so Z7 is no longer invisible to
canonical readiness tooling.

Current Z7 queue facts:

- `substrate_id`: `time_traveler_l5_z7_lstm_predictive_coding`
- `readiness_verdict`: `DEFER`
- `latest_council_verdict`: `PROCEED_WITH_REVISIONS`
- `research_only`: `true`
- `dispatch_enabled`: `false`
- `horizon_class`: `asymptotic_pursuit`
- `predicted_score_band`: `[0.10, 0.13]` from recipe-pinned symposium prior,
  kind `predicted_score_band`, axis `[contest-CPU]`, status
  `research_prior_prebuild`
- `predicted_delta_s_band`: `[-0.025, -0.008]` from the research-wave TOP-5
  row, axis `[contest-CPU]`, research-only and not a score claim
- `top_2_substrate`: readiness/dispatch queue now surfaces Z7 as the Stage 2
  stacking candidate behind Candidate 4c

Z7 blockers carried forward at the initial recipe landing:

- `TRAINER_MISSING`
- `CATALOG_240_FULL_MAIN_BLOCKED:TRAINER_FILE_MISSING`
- `CATALOG_315_COUNCIL_PROCEED_WITH_REVISIONS_NEEDS_ITERATION_TO_OPTIMAL_FORM`
- `RECIPE_research_only=true_OPERATOR_NEEDS_TO_FLIP_AFTER_PHASE_2_COUNCIL`
- `z7_trainer_module_absent_verified_by_symposium_pv2`
- `z7_substrate_package_absent_verified_by_symposium_pv2`
- `z7_dispatch_requires_z6_wave2_candidate_4c_paired_exact_eval_outcome`
- `z7_dispatch_requires_wave_n_plus_1_council_after_z6_4c_outcome`
- `z7_beta_ib_parameter_requires_c6_ibps_phase2_empirical_beta_anchor`
- `z7_wave2_probe_requires_paired_exact_eval_json_from_probe_z7_temporal_coherence_vs_static_capacity_disambiguator`
- `z7_same_archive_bytes_identity_control_exact_eval_required` (supersedes
  the earlier missing-control-arm wording after the local same-byte control
  artifact landed)

## Z7 disambiguator surface added

This pass also added
`tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py` with
unit coverage in
`src/tac/tests/test_probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py`.

The tool is fail-closed:

- Plan mode emits `pending_paired_exact_eval_json`.
- Exact-eval comparison requires same axis, `contest_cuda`, same sample count,
  same archive bytes, and formula-recomputed scores.
- A Z7 recurrent win requires `static_minus_recurrent_score >= 0.005`.
- The output remains `score_claim=false`, `promotion_eligible=false`,
  `ready_for_paid_dispatch=false`.

This removes the "probe tool absent" blocker while preserving the then-current
real blockers: no Z7 trainer, no Z7 substrate package, no paired exact-eval
JSONs, and no Wave N+1 council authorization.

## DP1 L1 no-op proof and rate arithmetic correction

This pass added
`tools/probe_dp1_pr101_composition_noop_detector.py` with tests in
`src/tac/tests/test_probe_dp1_pr101_composition_noop_detector.py`, then ran it
on the existing packet
`experiments/results/dp1_plus_fec6_composition_20260517`.

Probe artifact:
`.omx/research/dp1_pr101_composition_noop_probe_20260518_codex.json`

Verified packet facts:

- Verdict: `l1_rate_only_noop_verified`
- Archive SHA-256:
  `507d2a000ecf5a220e9b1ab765f75e39015cfb7b2af00606be3cb0758b8eb855`
- Archive bytes: `204344`
- fec6 base archive bytes: `178517`
- fec6 base archive SHA-256:
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- DP1 prefix bytes: `25814`
- DP1 prefix SHA-256:
  `b11bab015fa5c0c60c4587713413511fca18e194cb3c1c36f06db3afae4ee0e9`
- DPCOMP header bytes: `13`
- Structural blockers: none
- False-authority blockers:
  `no_paired_contest_cuda_cpu_eval`, `frame_axis_effect_deferred_to_l2`,
  `not_score_authority`

The important correction: the rate-only cost is
`25 * (204344 - 178517) / 37545489 = +0.017197139182`, not `+0.0000172`.
The older PATH 1 DP1 docs/recipe had a 1000x arithmetic typo. With
`PACT_DP1_PRIOR_STRENGTH=0.0` and frame parity, the L1 packet is expected to
regress the fec6 CPU anchor from `0.19205` to `0.209247139182`.

Control-plane changes from that correction:

- `.omx/operator_authorize_recipes/dp1_plus_fec6_composition_modal_paired_dispatch.yaml`
  is now `research_only: true`, `dispatch_enabled: false`, and blocks paid
  control eval unless the operator explicitly wants the paired no-op anchor.
- `.omx/operator_authorize_recipes/substrate_pr101_with_dp1_prior_modal_cpu_smoke_dispatch.yaml`
  now distinguishes the landed L1 no-op proof from the still-missing PATH 2
  lambda/prior-effect disambiguator.
- `.omx/research/dp1_dual_stacking_design_20260517.md` and
  `.omx/research/council_per_substrate_symposium_dp1_deep_dive_20260517.md`
  carry supersession notes so older `+0.0000172` text is not used for operator
  action.

## TT5L provider gate concretized

The TT5L queue row previously said "campaign ledger plus timing smoke", but the
current worktree already has the material campaign pieces: byte-closed work
unit, first-anchor timing-smoke artifact, paired-axis plan, Lightning execution
preflight, execution bundle, route-unblock packet, doctor plan, and dry-run
verification.

This pass ran the actual required Lightning doctor command in the current shell
and wrote:
`.omx/research/l5_v2_tt5l_lightning_required_doctor_20260517_codex.json`.

Doctor result:

- `status`: `FAIL`
- local supply-chain check: `ok=true`
- failed checks: `ssh_auth`, `remote_supply_chain`, `machine_inventory`
- concrete missing environment/current-state causes:
  `--ssh-target` / `LIGHTNING_SSH_TARGET` absent,
  remote supply-chain scan not run because SSH was absent, and
  `--teamspace` / `LIGHTNING_TEAMSPACE` absent for machine inventory
- `score_claim=false`, no dispatch, no spend

The non-dry-run gate was then regenerated:
`.omx/research/l5_v2_tt5l_lightning_non_dry_run_gate_20260517_codex.json`.

Gate result:

- `ready_for_non_dry_run_submit=false`
- `ready_for_provider_dispatch=false`
- `ready_cells=0/10`
- blocker count: `166`
- top blockers include doctor failure, missing source manifests/stage receipts,
  command placeholders for `--studio`, `--teamspace`, and
  `--remote-preflight-ssh-target`, missing active Lightning claims, and missing
  identity mode
- `dispatch_attempted=false`, `provider_spend_attempted=false`

Current TT5L next gate is therefore not another local design memo. It is:

1. Resolve Modal workspace billing or use Lightning.
2. For Lightning, provide `LIGHTNING_SSH_TARGET`, `LIGHTNING_TEAMSPACE`, and
   exactly one of `LIGHTNING_SDK_USER` or `LIGHTNING_ORG`.
3. Re-run the doctor until `status=OK`.
4. Stage per-cell source manifests, create active lane claims, remove command
   placeholders, then submit paired CPU/CUDA side-info cells.

Until then, TT5L remains `DEFER`, with no score/rank/promotion authority.

## Race-Mode queue dirty-tree precondition refresh

The latest Race Mode queue snapshot is:
`.omx/state/asymptotic_pursuit/dispatch_queue_20260518T065153Z.json`.

The queue now carries a separate launchability field so the research-wave
intake does not overstate Candidate 4c's current execution state in the shared
dirty worktree:

- `ready_for_paid_dispatch_count=1`
- `top_ready_substrate=z6_v2_candidate_4c_scorer_logit`
- `immediately_runnable_paid_dispatch_count=0`
- `current_worktree_dirty_path_count=48`
- `ready_paid_rows_requiring_catalog202_dirty_tree_attestation_count=1`
- `top_ready_paid_launch_missing_preconditions=[
  "CATALOG_202_dirty_worktree_requires_paired_env_attestation_before_paid_dispatch"
  ]`

Interpretation: Candidate 4c remains the only logically paid-ready substrate,
but a paid provider launch in this live worktree still requires the external
Catalog #202 paired-env attestation that the Modal sentinel set is clean. No
provider job was launched and no lane claim was opened.

## Candidate 4c Catalog #202 sentinel audit

The dirty-tree precondition was audited with:

```bash
.venv/bin/python tools/audit_catalog202_sentinel_cleanliness.py \
  --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
  --json --write-artifact
```

Artifact:
`.omx/state/catalog202_sentinel_cleanliness/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T065035Z.json`

Verdict:

- `sentinel_set_clean_for_catalog202=false`
- `ready_for_catalog202_paired_env_attestation=false`
- blocker: `catalog202_sentinel_files_dirty_in_git`
- dirty sentinels:
  - `tools/probe_z6_predictive_coding_vs_identity_disambiguator.py`
  - `tools/run_modal_smoke_before_full.py`

This preserves the launch signal without overstating it: Candidate 4c is still
the top logical substrate, but current paid dispatch should not proceed through
the dirty-tree bypass until those sentinel edits are stabilized or the sentinel
contract is deliberately changed. No provider job was launched and no lane
claim was opened.

## Candidate 4c audit-backed launch prefix

Supersession note: the earlier `065035Z` audit proved the dirty-sentinel
problem. The follow-up authorization patch now supports an audit-backed dirty
sentinel snapshot instead of a bare boolean attestation. Fresh artifacts:

- Sentinel audit:
  `.omx/state/catalog202_sentinel_cleanliness/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T065922Z.json`
- Dispatch queue:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T070955Z.json`

Current launch interpretation:

- `ready_for_paid_dispatch_count=1`
- `immediately_runnable_paid_dispatch_count=0` because env vars are not set in
  this shell
- audit-backed dirty sentinel attestation is available:
  `ready_for_catalog202_audit_backed_dirty_sentinel_attestation=true`
- dirty sentinels are hash-pinned by
  `sentinel_set_sha256=9e94588c666d63cb187a766f71c6aaa75448efe35087a83e686e1800d04f1146`

The queue's `top_ready_audit_backed_paid_launch_command` now includes all
required prefixes:

```bash
OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 \
OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=catalog202_sentinel_audit:9e94588c666d63cb187a766f71c6aaa75448efe35087a83e686e1800d04f1146 \
OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON=.omx/state/catalog202_sentinel_cleanliness/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T065922Z.json \
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=13.000 \
.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
  --operator-handle codex:z6_v2_candidate_4c_scorer_logit
```

No provider job was launched and no lane claim was opened.

## Z7 exact-eval handoff doctor and online-review hardening

Codex added a no-spend handoff doctor for the current Z7 same-byte
recurrent/static packet:

```text
tools/verify_z7_exact_eval_handoff.py
src/tac/tests/test_verify_z7_exact_eval_handoff.py
.omx/state/z7_exact_eval_handoff/z7_exact_eval_handoff_20260518T133855Z.json
.omx/research/z7_exact_eval_handoff_and_online_lit_review_20260518_codex.md
```

Current handoff verdict:

```text
ready_for_exact_eval_handoff=false
current_pair_count=1
required_pair_count=600
result_review_blockers=[z7_exact_handoff_current_packet_not_600_pairs]
same_archive_zip_bytes=true
runtime_output_changed_vs_recurrent=true
score_claim=false
promotion_eligible=false
provider_dispatch_attempted=false
lane_claim_opened=false
```

This preserves the local mechanism signal while blocking accidental exact-eval
dispatch of the one-pair smoke. The next Z7 score-moving artifact is a ratified
600-pair recurrent/static packet, not another plan-only handoff.

Online review also tightened the design target:

- DCVC-style contextual coding suggests Z7 should condition decoder features and
  residual-symbol scales, not only predict a latent and ship an int8 residual.
- HNeRV/HiNeRV-style evidence argues decoder capacity should be a deliberate
  Z7-specific decision rather than automatic Z6-decoder reuse.
- Mamba/Mamba-2 evidence does not prove a speed/score win at 600 pairs x small
  latents; Z7-Mamba remains a measured-disambiguator branch.
- The Z7-Mamba runtime must not rely on external `brotli` or unproven
  `mamba_ssm` availability at inflate time; pure-PyTorch exported selective SSM
  recurrence is the promotion path unless dependency closure is proven.

## Candidate 4c paired exact-eval handoff repaired

The full-600 Candidate 4c handoff is now represented as a paired exact-eval
surface rather than four copyable single-axis Modal commands. The packet doctor
generates:

- `full_paired_contest_cpu_cuda`
- `identity_paired_contest_cpu_cuda`

Both commands route through `tools/dispatch_modal_paired_auth_eval.py`, use
archive SHA-256 guards, carry a paired group id, request
`--expected-runtime-tree-sha256 auto`, and keep `[contest-CUDA]` and
`[contest-CPU]` axes paired by construction.

Current no-spend packet:

- `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T105922Z.json`
- `exact_eval_handoff.ready_for_exact_eval_handoff=true`
- `exact_eval_handoff.latest_pair_count=600`
- `ready_for_operator_paid_execution=false`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`

This changes the next actionable Candidate 4c step from "choose a wrapper
command" to "open the paired exact-eval dispatch lifecycle deliberately." No
provider job was launched and no lane claim was opened.

## Candidate 4c paired exact-eval dispatch opened

The paired exact-eval lifecycle has now been opened for Candidate 4c. Four
detached Modal calls were accepted:

- full `[contest-CUDA]`: `fc-01KRXC3V6N13J3H9R5XZSXHPQ1`
- full `[contest-CPU]`: `fc-01KRXC4EZ3GY615KF1EJE33VZ2`
- identity `[contest-CUDA]`: `fc-01KRXC3WYZKE2ZEE04R7P714KE`
- identity `[contest-CPU]`: `fc-01KRXC4M333B38CRV7Q6HXNVN1`

Dispatch ledger:

- `.omx/research/z6_candidate4c_paired_modal_exact_eval_dispatch_20260518_codex.md`

Current claim summary after dispatch:

- `active=4`
- `stale_nonterminal=0`
- `score_claim=false`
- `promotion_eligible=false`

The next queue action is harvest/adjudication, not another launch. Do not
dispatch the same four lane ids while these calls are active.

CUDA harvest then landed for both Candidate 4c zero-epoch archives:

- full `[contest-CUDA]`: `90.58142803863508`
- identity `[contest-CUDA]`: `90.58427695093009`
- identity-minus-full: `0.0028489122950077217`
- active claims remaining: `2` (`[contest-CPU]` full and identity)

Queue consequence: do not promote or retire Candidate 4c from this. The
zero-epoch control packet is measured bad; the full-vs-identity delta is below
the disambiguator threshold; the trained Candidate 4c path remains the relevant
score-moving substrate after CPU harvest closes the paired result.

CPU harvest then landed:

- full `[contest-CPU]`: `90.57816474855734`
- identity `[contest-CPU]`: `90.58102532784203`
- identity-minus-full: `0.0028605792846860822`
- active claims after terminal rows: `0`

Final queue consequence: Candidate 4c zero-epoch exact eval is closed as a
bad-control anchor. It proves the predictive path is consumed and marginally
beats identity on both axes, but it does not meet the disambiguator delta and
does not represent trained Candidate 4c score potential.

## Candidate 4c full-600 handoff status

The Candidate 4c row remains TOP-1 by planning priority, but its status changed
from "exact handoff blocked by 2-pair archive pair" to "exact handoff ready,
paid training recipe still disabled".

Current artifacts:

- refreshed queue:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T104919Z.json`
- refreshed no-spend packet:
  `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T104931Z.json`
- full-600 packet ledger:
  `.omx/research/z6_candidate4c_full600_zeroepoch_handoff_20260518_codex.md`

Current packet status:

- `exact_eval_handoff.ready_for_exact_eval_handoff=true`
- `exact_eval_handoff.latest_pair_count=600`
- `local_identity_disambiguator_probe.runtime_output_changed=true`
- `ready_for_operator_paid_execution=false`
- `top_ready_substrate=null`
- `ready_for_paid_dispatch_count=0`

The next queue action is not the diagnostic Candidate 4c training recipe. The
next score-bearing action is paired exact eval of the full and identity ZIPs
through claimed `[contest-CUDA]` and `[contest-CPU]` lanes, using the generated
commands in the no-spend packet.

## Lane 17 Catalog #308 magnitude-criteria disambiguator

The no-spend disambiguator requested by the 2026-05-17 Lane 17 symposium is now
landed as an executable/importable tool:
`tools/probe_imp_magnitude_criteria_disambiguator.py`.

Fresh artifact:
`.omx/research/lane17_imp_magnitude_criteria_disambiguator_20260518_codex.json`.

The artifact is explicitly advisory:

- evidence axis: `[local-IMP-mask-byte-proxy advisory]`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_paid_dispatch=false`
- `rank_or_kill_eligible=false`
- `catalog308_disambiguator_landed=true`

Measured on the Lane G v3 FP4A anchor
`experiments/results/lane_g_v3_landed/iter_0/renderer.bin`:

| Criterion | Status | Sparsity | IMPS bytes | Savings vs anchor |
| --- | --- | ---: | ---: | ---: |
| L1 per-tensor canonical Frankle | local byte-mask proxy | 0.8931788840 | 154,552 | 47.9230% |
| Global L1 control | local byte-mask proxy | 0.8927248996 | 177,466 | 40.2020% |
| Hessian-trace OBD proxy | advisory proxy only | 0.8926716983 | 182,170 | 38.6170% |
| Score-gradient saliency Catalog #123 | blocked without sidecar | not measured | not measured | not measured |

Interpretation: per-tensor L1 is the current byte-proxy winner on this anchor,
but this does **not** prove scorer preservation, cycle-0 recovery, or
frontier movement. The Catalog #123 branch now refuses to emit a global-L1
surrogate measurement when the required score-gradient sidecar is absent; its
artifact row carries
`no_global_l1_surrogate_emitted_for_score_gradient_branch`.

The recipe blocker was narrowed from "build the disambiguator" to these
remaining dispatch blockers:

- `lane_17_imp_requires_catalog308_cycle0_empirical_regression_ratio_disambiguation`
- `lane_17_imp_requires_catalog123_score_gradient_saliency_sidecar_for_authority`
- train-distill/PCC3 wall-clock refresh
- score-aware distillation objective confirmation
- Quantizr composition plan before any frontier score claim
- operator budget, lane claim, and Vast instance before spend

No provider job was launched and no lane claim was opened.

## Queue refresh after Lane 17 blocker narrowing

After landing the Lane 17 disambiguator and narrowing the blocker set, the
machine-readable planning surfaces were regenerated:

- readiness assessment:
  `.omx/state/asymptotic_pursuit/readiness_assessment_20260518T075608Z.json`
- dispatch queue:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T075608Z.json`
- research-wave intake queue:
  `.omx/state/asymptotic_pursuit/research_wave_intake_20260518T075608Z.json`
- Candidate 4c no-spend launch packet:
  `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T075616Z.json`

Lane 17 remains `DEFER`, now for the explicit cycle-0 empirical
regression-ratio and Catalog #123 score-gradient sidecar gates. Candidate 4c
remains the top ready no-spend packet with `provider_dispatch_attempted=false`,
`lane_claim_opened=false`, `score_claim=false`, and `promotion_eligible=false`.

## Lane 17 IMP readiness-surface recovery

The comprehensive research wave ranked `lane_17_imp + Frankle LTH` in the
TOP-5, but the intake queue previously reported `no current readiness row`.
That was signal loss: the lane registry already has `lane_17_imp_10cycle` at
L2 with real archive-byte evidence, and the 2026-05-17 per-substrate symposium
withdrew the old stub-loop KILL while keeping binding revisions in force.

This pass made the signal actuator-visible without granting spend or score
authority:

- readiness assessment:
  `.omx/state/asymptotic_pursuit/readiness_assessment_20260518T074243Z.json`
- dispatch queue:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T074243Z.json`
- research-wave intake:
  `.omx/state/asymptotic_pursuit/research_wave_intake_20260518T074243Z.json`
- research-only recipe:
  `.omx/operator_authorize_recipes/lane_17_imp_cycle0_vastai_4090_timing_smoke_dispatch.yaml`

Current lane_17 state:

- `current_readiness_verdict=DEFER`
- `current_recipe_path=.omx/operator_authorize_recipes/lane_17_imp_cycle0_vastai_4090_timing_smoke_dispatch.yaml`
- `research_only=true`
- `dispatch_enabled=false`
- `ready_for_paid_dispatch=false`
- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `predicted_band_axis=contest-CPU`
- `predicted_score_band=[0.17705, 0.18705]` as a research-wave prior only
- `predecessor_probe_id=council_per_substrate_symposium_lane_17_imp_20260517`
- current blockers include Catalog #313, Catalog #315, the Catalog #308
  magnitude-criteria disambiguator, train-distill/PCC3 evidence refresh,
  score-aware distillation objective confirmation, Quantizr composition plan,
  and operator budget/lane-claim/Vast.ai instance approval before spend.

The dispatch queue now displays the correct no-spend Vast.ai dry-run surface
instead of a Modal smoke wrapper:

```bash
.venv/bin/python scripts/launch_lane_on_vastai.py --lane-script scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh --label lane_17_imp_cycle0_timing_smoke --predicted-band 0.17705 0.18705 --estimated-cost 2 --max-dph 0.5 --min-disk-gb 60 --dry-run
```

Interpretation: all five deep-research TOP-5 candidates now join to current
readiness state (`current_readiness_join_count=5`). Candidate 4c remains the
only `READY` row; lane_17_imp is preserved as a cheap frontier-pursuit
candidate with a launchable-looking dry-run path but no paid-dispatch
authority. No provider job was launched and no lane claim was opened.

## Candidate 4c packet refresh after Lane 17 queue recovery

Because the dispatch queue changed, the Candidate 4c no-spend launch packet
was regenerated against the current queue snapshot:

- queue:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T074243Z.json`
- no-spend launch packet:
  `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T074441Z.json`

Current packet state:

- `ready_for_operator_paid_execution=true`
- `queue_top_ready_substrate=z6_v2_candidate_4c_scorer_logit`
- `next_paid_command_ready=true`
- `next_paid_command_blockers=[]`
- `local_identity_disambiguator_probe_ready=true`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`

The queue now contains lane_17_imp, but Candidate 4c remains the only READY
paid row. No provider job was launched and no lane claim was opened.

## Candidate 4c paid-command audit binding gate

The no-spend launch packet now also verifies the copied operator-facing paid
command, not just the dry-run environment constructed by the packet doctor.
Fresh packet:
`.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T073219Z.json`.

The command must contain the current Catalog #202 audit path/hash, the session
directive and `$13.000` budget floor, the Candidate 4c recipe, the Candidate 4c
operator handle, and no `--dry-run`.

Current packet state:

- `ready_for_operator_paid_execution=true`
- `local_identity_disambiguator_probe_ready=true`
- `next_paid_command_ready=true`
- `next_paid_command_blockers=[]`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`

This closes the stale copied-command handoff risk while preserving the no-spend
status. No provider job was launched and no lane claim was opened.

## Candidate 4c no-spend launch packet

The latest no-spend handoff artifact is:
`.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T070956Z.json`.

It verifies the current Candidate 4c launch chain without opening a claim or
provider job:

- queue artifact:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T070955Z.json`
- sentinel audit artifact:
  `.omx/state/catalog202_sentinel_cleanliness/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T065922Z.json`
- `lane_claim_summary`: green, `active=0`
- required input validation: green
- smoke-before-full dry-run: green
- operator-authorize dry-run: green
- Catalog #202 audit-backed bypass probe: green

Packet verdict:

- `ready_for_operator_paid_execution=true`
- `checks_ok=true`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`

This keeps the research-wave queue honest: Candidate 4c is not a score claim,
but it is now a no-spend-verified launch packet awaiting intentional operator
paid execution. No provider job was launched and no lane claim was opened.

## Candidate 4c runtime-custodied local disambiguator proof

The Candidate 4c local identity-disambiguator artifact was regenerated with
runtime custody attached to the same full-vs-identity inflate-output proof:
`.omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.json`.

Current advisory-only facts:

- evidence axis: `[local-inflate-output advisory]`
- `runtime_output_changed=true`
- total raw-output byte differences: `22253`
- full raw-output aggregate SHA-256:
  `2b0e3345c5eb8f00beb71de62e0bdda60cc17933acbbe775ef451b2791aacf73`
- identity raw-output aggregate SHA-256:
  `856f15ea620ff704a3bebcdfef75f289e7af11d0d1e7b45f0fee7262505c6409`
- runtime closure aggregate SHA-256:
  `384938f3b6a14acb5944938c45c127ccc411f00983aff7589c7c9b98cfc56073`
- runtime closure file count: `10`
- archive payloads and Python cache files are excluded from the runtime hash
  and separately listed in the JSON.
- `score_claim=false`
- `promotion_eligible=false`

Interpretation: the identity-predictor switch is proven to affect local
inflate output under a hash-pinned source runtime, so the no-op concern is
closed for the local advisory mechanism proof. It is still not score authority;
the next authoritative gate remains paired contest-CUDA exact eval with matching
ZIP and runtime custody. No provider job was launched and no lane claim was
opened.

## Candidate 4c refreshed no-spend packet with runtime custody

The runtime-custodied local disambiguator proof now propagates through the
queue and launch packet:

- Catalog #202 sentinel audit:
  `.omx/state/catalog202_sentinel_cleanliness/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T072226Z.json`
- dispatch queue:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T072231Z.json`
- no-spend launch packet:
  `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T072242Z.json`

The first packet refresh failed closed on stale sentinel hash after the probe
edit, then the refreshed audit restored the no-spend packet to green:

- `ready_for_operator_paid_execution=true`
- `checks_ok=true`
- `active_lane_claims_clean=true`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`

The paid command now references the fresh audit hash
`a04c09b40cca80dd98c41968770ce7d2b19b672e2553d6a2bbb5c03b1a5aa387`. No
provider job was launched and no lane claim was opened.

## Candidate 4c launch-packet custody gate

The no-spend launch packet now fails closed if the queue-carried local
identity-disambiguator proof lacks runtime/output custody. Fresh packet:
`.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T072757Z.json`.

Required local-proof fields before readiness:

- `verdict=pending_paired_exact_eval_json`
- `runtime_output_changed=true`
- empty probe blockers
- runtime custody aggregate SHA-256
- full and identity raw-output aggregate SHA-256s
- positive raw-output byte-difference count
- output root

Current packet state:

- `ready_for_operator_paid_execution=true`
- `local_identity_disambiguator_probe_ready=true`
- `local_identity_disambiguator_probe_blockers=[]`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`

No provider job was launched and no lane claim was opened.

## Candidate 4c packet supersession after queue-immediate gate

Later adversarial review found that the no-spend packet was still trusting
`top_ready_substrate` plus an audit-backed command string instead of the queue's
new `immediately_runnable_paid_launch` contract. The older packet artifacts
above that say `ready_for_operator_paid_execution=true` are historical only and
must not be used as current launch authority.

Current latest packet:
`.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T091134Z.json`

Current verdict:

- `ready_for_operator_paid_execution=false`
- `queue_immediate_launch_ready=false`
- `next_paid_command_ready=true`
- `catalog202_audit_backed_bypass_probe_accepted=true`
- `checks_ok=true`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`

The packet is fail-closed because the live queue reports
`immediately_runnable_paid_dispatch_count=0` and non-empty Catalog #202
preconditions. This is a custody/dispatch correction only; Candidate 4c remains
the top ready candidate in the planning queue, but it is not immediately
launchable from the current shell until the Catalog #202 env/audit preconditions
are satisfied and a fresh packet verifies that state. No provider job was
launched and no lane claim was opened.

## Candidate 4c diagnostic-only handoff doctor repair

After Candidate 4c was split back to a diagnostic-only Modal training recipe,
the launch-packet doctor still modeled the lane as a paid-launch surface. That
was a false blocker shape: the stale Catalog #202 audit hash and missing paid
command were consequences of `dispatch_enabled=false`, not the real score-moving
handoff blocker.

Codex repaired `tools/verify_candidate4c_launch_packet.py` and reran the
no-spend packet:

- artifact:
  `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T103550Z.json`
- `current_mode=diagnostic_only_exact_eval_handoff_required`
- `checks_ok=true`
- `active_lane_claims_clean=true`
- `diagnostic_smoke_dry_run_ready=true`
- `catalog202_audit_backed_bypass_probe_accepted=true` because the bypass probe
  is now explicitly skipped while paid training is out of scope
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`
- `exact_eval_handoff.ready_for_exact_eval_handoff=false`
- blocker: `candidate4c_exact_handoff_latest_archive_pair_not_600_pairs`

The latest full/identity archive pair remains the 2-pair local artifact in
`.omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.json`.
It is byte-closed and proves runtime-output difference, but it is not score
authority. The doctor now emits only exact-eval command templates until a full
600-pair archive/runtime packet exists, so the pair-capped artifact cannot be
accidentally laundered into a Modal dispatch surface.

Queue consequence: Candidate 4c remains high-EV, but the next concrete artifact
is a harvested full 600-pair Candidate 4c archive pair, followed by four claimed
auth-eval axes: full/identity `[contest-CUDA]` and full/identity
`[contest-CPU]`. No provider spend, lane claim, score claim, or promotion claim
occurred in this repair.

## ATW V2-1 Faiss-PQ disambiguator completed

Codex converted TOP-5 #3 (`ATW V2-1`) from "planned V1/V2/V3
disambiguator" into a completed local diagnostic artifact:

- state artifact: `.omx/state/atw_v2_1_faiss_pq_disambiguator_probe.json`
- research JSON: `.omx/research/atw_v2_1_faiss_pq_disambiguator_probe_20260518_codex.json`
- research Markdown: `.omx/research/atw_v2_1_faiss_pq_disambiguator_probe_20260518_codex.md`
- local forensic bytes: `experiments/results/atw_v2_1_faiss_pq_probe_20260518T100524Z/`
- axis: `[diagnostic-CPU; ATW V2-1 Faiss-PQ side-info MI probe]`
- `score_claim=false`, `promotion_eligible=false`, `provider_spend_attempted=false`

Engineering/adversarial findings:

- `faiss-cpu` and Torch abort on this macOS host when Faiss training occurs in
  the same Python process as Torch. The probe now saves Torch/SegNet softmax
  arrays first and runs the Faiss worker in an isolated subprocess.
- The Faiss helper now pads 5-class SegNet softmax vectors to a PQ-compatible
  dimension and trims decoded tensors back to 5 classes.
- The Faiss helper now serializes trained quantizers only (`ntotal=0`); adding
  training vectors to the IVF index had produced false over-budget codebooks.

Measured post-fix consequence:

| Variant | Brotli archive bytes | MI bits/symbol | Guarded verdict |
|---|---:|---:|---|
| `v3_pool_shared` | 3,114 | 0.121512378237 | byte-closed but `WEAK_CONDITIONING`; no dispatch authority |
| `v2_sparse_top_k` | 7,941 | 2.457397664695 | `MEANINGFUL_CONDITIONING` only as high-cardinality plug-in MI upper bound (`unique_fraction=1.0`); no dispatch authority |
| `v1_dense` | 452,799 | 2.457397664695 | high-cardinality upper bound and over-budget; no dispatch authority |

Queue consequence: ATW V2-1 should remain `research_only=true` /
`dispatch_enabled=false`. The actionable next gate is no longer "run the
Faiss-PQ disambiguator"; it is
`pivot_to_scorer_logit_compression_or_trained_atw_residual_probe`, preserving
the ATW cooperative-receiver paradigm while rejecting this measured Faiss-PQ
channel configuration. No provider job was launched and no lane claim was
opened.

## Candidate 4c Modal diagnostic-only split

The Candidate 4c pre-dispatch blocker was converted into a corrected launch
architecture. The Modal training recipe is now diagnostic-only and no longer
advertises a normal contest-exact paid launch.

Artifacts:

- queue: `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T095039Z.json`
- packet:
  `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T095046Z.json`
- review:
  `.omx/state/candidate4c_launch_packet/candidate4c_codex_pre_dispatch_review_20260518T0952Z.json`

Current queue interpretation:

- TOP-1 substrate remains `z6_v2_candidate_4c_scorer_logit`
- TOP-1 verdict is `NEEDS_FIX`
- ready paid dispatch count is `0`
- no top ready paid command exists
- Candidate 4c blockers are:
  `RECIPE_dispatch_enabled=false` and
  `RECIPE_DISPATCH_BLOCKER:candidate4c_modal_training_recipe_is_diagnostic_only_exact_cuda_handoff_required`
- fresh Codex review is `approve`
- no provider dispatch, no lane claim, no score claim, no promotion claim

Corrected architecture:

- Pair-capped Modal run is `training_artifact_v1` only.
- `Z6_MAX_PAIRS=64` and `Z6_SKIP_AUTH_EVAL=1` prevent truncated diagnostic
  archives from entering auth-eval.
- Exact-CUDA authority is a later handoff from a full 600-pair archive/runtime
  packet through a canonical exact-eval provider path.

Verification:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_run_modal_smoke_before_full.py \
  src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py \
  src/tac/tests/test_verify_candidate4c_launch_packet.py \
  src/tac/tests/test_check_271_pre_dispatch_codex_review.py
# 197 passed in 29.68s
```

No provider job was launched and no lane claim was opened.

Verification:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_verify_candidate4c_launch_packet.py \
  src/tac/tests/test_check_271_pre_dispatch_codex_review.py \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py
# 125 passed in 16.79s

.venv/bin/python tools/check_dispatch_cli_shell_hazards.py --strict
# PASS
```

## Candidate 4c env-attested immediate launch packet

The Catalog #202 env/audit preconditions were then replayed explicitly with the
current sentinel audit tuple, producing an operator-attested no-spend handoff:

- env-attested queue:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T091517Z.json`
- env-attested launch packet:
  `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T091519Z.json`

Current env-attested verdict:

- `ready_for_paid_dispatch_count=1`
- `immediately_runnable_paid_dispatch_count=1`
- `ready_for_operator_paid_execution=true`
- `queue_immediate_launch_ready=true`
- `queue_immediate_launch_blockers=[]`
- `catalog202_audit_backed_bypass_probe_accepted=true`
- `active_lane_claims_clean=true`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`

Required env tuple:

```bash
OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1
OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=catalog202_sentinel_audit:a04c09b40cca80dd98c41968770ce7d2b19b672e2553d6a2bbb5c03b1a5aa387
OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON=.omx/state/catalog202_sentinel_cleanliness/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T072226Z.json
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=13.000
```

Interpretation: the unauthenticated-shell packet and the env-attested packet
now intentionally disagree. The former proves fail-closed behavior without
Catalog #202 authority; the latter is the current no-spend green handoff for a
future explicit claimed paid smoke/full launch. No provider job was launched
and no lane claim was opened.

Verification:

```bash
.venv/bin/python tools/verify_candidate4c_launch_packet.py --json \
  --queue-path .omx/state/asymptotic_pursuit/dispatch_queue_20260518T091517Z.json
# rc=0; ready_for_operator_paid_execution=true

.venv/bin/python -m pytest -q \
  src/tac/tests/test_verify_candidate4c_launch_packet.py \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py
# 69 passed in 12.33s

.venv/bin/python tools/check_dispatch_cli_shell_hazards.py --strict
# PASS
```

## Candidate 4c codex pre-dispatch blocker

The env-attested packet was then run through the mandatory Catalog #271 codex
pre-dispatch review. That review is currently blocking paid execution:

- review artifact:
  `.omx/state/candidate4c_launch_packet/candidate4c_codex_pre_dispatch_review_20260518T0920Z.json`
- latest no-spend packet:
  `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T092725Z.json`

Current verdict:

- `queue_immediate_launch_ready=true`
- `codex_pre_dispatch_review_ready=false`
- `codex_pre_dispatch_review_blockers=[
  candidate4c_codex_pre_dispatch_review_blocking_needs-attention]`
- `ready_for_operator_paid_execution=false`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`

Blocking finding:

```text
- [high] Contest-CUDA smoke contract is incompatible with the Modal training runtime's forced CPU auth-eval
```

Interpretation: Candidate 4c remains the top planning row, and its
Catalog #202/env-attested queue state is immediately runnable mechanically, but
the current Modal recipe should not be launched as contest-CUDA because the
Modal worker forces CPU advisory auth-eval while the smoke validator expects a
contest-CUDA claim. Next score-moving action is to either move the exact-CUDA
eval handoff to a runtime/provider path that does not force CPU auth-eval, or
split the current Modal run into an explicit diagnostic archive-producing
smoke plus a separate claimed exact-CUDA eval handoff. No provider job was
launched and no lane claim was opened.

## ATW V2-1 queue visibility refresh

The ATW V2-1 Faiss-IVF-PQ WIP is now visible in the canonical dispatch queue as
`atw_codec_v2_1_faiss_ivf_pq`, with recipe
`.omx/operator_authorize_recipes/substrate_atw_v2_1_modal_t4_smoke_dispatch.yaml`
and lane `lane_top5_3_atw_v2_1_faiss_ivf_pq_scaffold_design_20260518`.

Queue artifact:

```text
.omx/state/asymptotic_pursuit/dispatch_queue_20260518T112720Z.json
```

The row is correctly `DEFER`, not spend authority. Its first blockers are the
actual current blockers:

- `_full_main` still raises `NotImplementedError`.
- ATW V2-1 still has `PROCEED_WITH_REVISIONS` council state.
- Recipe remains `research_only=true`.
- The Faiss-PQ disambiguator found V3 pool-shared weak (`MI=0.121512378237`,
  `3114` bytes) and V2/V1 high-cardinality upper-bound positives, not dispatch
  authority.
- The stale pending Z6 4c cross-pollination blocker was replaced with the
  measured exact outcome:
  `z6_wave_2_4c_zeroepoch_exact_outcome_did_not_validate_scorer_logit_channel_delta_below_0_005`.

Dedicated ledger:

```text
.omx/research/atw_v2_1_queue_visibility_and_blocker_refresh_20260518_codex.md
```

## ATW V2-1 scorer-softmax sketch gate

Codex ran the next $0 ATW V2-1 gate named by the Faiss-PQ result: convert the
cached A1/SegNet softmax arrays into byte-closed scorer-derived sketches and
test whether any low-cardinality packet preserves enough latent information to
justify a new D4 probe.

Artifacts:

```text
tool: tools/probe_atw_v2_1_scorer_softmax_sketch.py
tests: src/tac/tests/test_probe_atw_v2_1_scorer_softmax_sketch.py
state: .omx/state/atw_v2_1_scorer_softmax_sketch_probe.json
research_json: .omx/research/atw_v2_1_scorer_softmax_sketch_probe_20260518_codex.json
research_md: .omx/research/atw_v2_1_scorer_softmax_sketch_probe_20260518_codex.md
local_packets: experiments/results/atw_v2_1_scorer_softmax_sketch_probe_20260518T113825Z/
```

Measured diagnostic result:

```text
[diagnostic-CPU; ATW V2-1 scorer-softmax sketch MI probe]
best_variant=region256_coarse_entropy_anchor_q4
best_mi=0.076162617811 bits/symbol
best_packet_bytes=378
best_high_cardinality_bias_guard=false
phase2_status=scorer_softmax_sketches_only_weak_or_biased_conditioning
recommended_next_gate=trained_atw_residual_probe_or_raw_scorer_logit_head_design
score_claim=false
promotion_eligible=false
provider_spend_attempted=false
```

Interpretation: the cached scorer-softmax family is byte-closed but too weak.
This supersedes `pivot_to_scorer_logit_compression_or_trained_atw_residual_probe`
with the narrower gate `trained_atw_residual_probe_or_raw_scorer_logit_head_design`.
The branch is not retired as a representation family; it needs a trained
residual channel or raw-logit capture, not another low-cardinality softmax
sketch.

Verification:

```bash
.venv/bin/python -m py_compile tools/probe_atw_v2_1_scorer_softmax_sketch.py
.venv/bin/python -m pytest src/tac/tests/test_probe_atw_v2_1_scorer_softmax_sketch.py -q
.venv/bin/python tools/probe_atw_v2_1_scorer_softmax_sketch.py
```

Queue refresh:

```text
.omx/state/asymptotic_pursuit/dispatch_queue_20260518T114053Z.json
```

The ATW V2-1 row remains `DEFER`; its blocking issues now include
`scorer_softmax_sketch_completed_20260518_all_byte_closed_but_best_mi_0_076162617811_weak`
and `selected_next_gate_is_trained_atw_residual_probe_or_raw_scorer_logit_head_design`.

## Candidate 4c diagnostic-DEFER queue fix

The dispatch queue was still classifying Candidate 4c as `NEEDS_FIX` solely
because the Modal training recipe had `dispatch_enabled=false`. That was a
false-actionability bug after the diagnostic-only split: the current recipe is
not supposed to become a contest-CUDA launch surface, and the zero-epoch
full/identity archive pair has already been paired-evaluated on both axes.

Patch:

```text
tools/asymptotic_pursuit_candidate_readiness_assessment.py
src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py
```

Dedicated ledger:

```text
.omx/research/candidate4c_diagnostic_defer_queue_fix_20260518_codex.md
```

Refreshed queue:

```text
.omx/state/asymptotic_pursuit/dispatch_queue_20260518T114641Z.json
```

Current queue consequence:

```text
candidate4c.readiness_verdict=DEFER
candidate4c.ready_for_paid_dispatch=false
ready_for_paid_dispatch_count=0
top_1_substrate=time_traveler_l5_z7_lstm_predictive_coding
top_1_readiness_verdict=DEFER
```

This keeps Candidate 4c visible as a high-EV trained-method branch without
inviting an unsafe `dispatch_enabled=true` edit on a diagnostic training recipe.
The reactivation path is a trained 600-pair archive/runtime packet followed by
a fresh claimed paired exact-eval handoff, not another zero-epoch handoff and
not a direct Modal training score claim.

## Z7-Mamba-2 queue visibility and predicted-band axis fix

The Z7-Mamba-2 scaffold already existed as WIP, but it was not represented in
the canonical asymptotic-pursuit queue. Direct assessment fell through to the
wrong `_modal_t4_` recipe basename, which hid the active recipe/trainer/ledger
surface and produced false `RECIPE_MISSING` / `LANE_REGISTRY_NOT_REGISTERED`
diagnostics.

Patch:

```text
tools/asymptotic_pursuit_candidate_readiness_assessment.py
src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py
```

Dedicated ledger:

```text
.omx/research/z7_mamba2_queue_visibility_and_axis_fix_20260518_codex.md
```

The same pass fixed a predicted-band evidence bug: when a recipe carries
explicit `predicted_band` plus `predicted_band_kind`, the readiness assessment
now prefers that labelled recipe metadata over loose design-memo regex
extraction. This prevents a `predicted_delta_s_band` such as
`[-0.025, -0.008]` from being mislabeled as a `predicted_score_band`.

Refreshed queue:

```text
.omx/state/asymptotic_pursuit/dispatch_queue_20260518T115056Z.json
```

Current Z7-Mamba-2 queue row:

```text
substrate_id=time_traveler_l5_z7_mamba2
readiness_verdict=DEFER
predicted_score_band=[0.167, 0.184]
predicted_band_axis=contest-CPU
horizon_class=frontier_pursuit
```

This is not spend authority. It preserves WIP signal and keeps the next real
gate explicit: implement the trainer `_full_main` / substrate package only
after the Z7-GRU sequencing dependency, C6 beta anchor, and Wave-N+1 council or
operator override are resolved.

## Z7-GRU prebuild scaffold landed

The top queue row for `time_traveler_l5_z7_lstm_predictive_coding` no longer
fails as an absent trainer/package. Codex added the narrow prebuild scaffold:

```text
src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/__init__.py
src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/architecture.py
experiments/train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py
src/tac/tests/test_z7_lstm_predictive_coding_scaffold.py
```

The scaffold exports a `GruRecurrentPredictor` with the Z6-compatible
`forward(z_prev, ego_motion) -> z_pred` signature, identity-predictor mode,
state reset, batch-first sequence unroll, parameter counting, and smoke stats.
It now also exports `Z7GruPredictiveCodingSubstrate`, binding the recurrent
predictor to a Z6-compatible RGB decoder plus latent/residual/ego-motion
streams. `_full_main` emits a byte-closed proxy packet instead of stopping at a
hard prebuild exception.

Recipe/probe blocker refresh after the trainer/package scaffold, superseded by
the later score-aware smoke lift:

```text
z7_score_aware_one_pair_smoke_not_contest_authority
z7_score_aware_packet_requires_paired_exact_eval
```

The archive surface was then advanced to a deterministic Z7PCWM1 parser/replay
scaffold:

```text
src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/archive.py
```

That scaffold packs/parses encoder, decoder, GRU predictor, latent init,
predictive residuals, ego-motion, and metadata sections. It also provides a
`replay_latent_sequence(...)` proof that the GRU predictor bytes are consumed:
mutating the predictor state changes the replayed latent sequence.

The follow-on scorer-free runtime scaffold lives at:

```text
src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/inflate.py
```

It replays parsed GRU latents, consumes the Z6-compatible decoder weight stream,
and writes contest-shaped raw RGB output. The full-main export smoke produced a
real-video proxy packet and exposed one runtime dependency bug: `brotli` was
not available under default `python3`. Z7PCWM1 now uses stdlib `zlib`, and the
rebuilt runtime passes the contest-style inflate signature.

Local artifact:

```text
experiments/results/z7_gru_prebuild_export_codex_20260518T1226Z/
archive_zip_bytes=5178
archive_zip_sha256=429edc1657937d93d4116f094ac2ede66059c9a0f88acd330462784c5c4b560d
inflate_cli_raw_bytes=12208032
inflate_cli_raw_sha256=78c365731c2023b109e2fad08d205ffa1afffe6f8aada72c2734801720cccc61
static_control_archive_zip_bytes=5178
static_control_archive_zip_sha256=a74a4ce637e82d8340e5e854ec2a3f2b3296552c8cc7268bc367a921bc7f0c7a
same_archive_zip_bytes_as_recurrent=true
static_control_raw_sha256=166ef65b0474ef05779eec5b68bc250526ed4501f934ad452a0380d296381079
runtime_output_changed_vs_recurrent=true
runtime_output_byte_differences_vs_recurrent=2150198
score_claim=false
promotion_eligible=false
```

The static-control artifact closes only the local control-arm construction
bug. It does not make the Z7 method verdictable: recurrent and static-control
packets still need paired exact-eval JSON on the same contest axis.

Refreshed readiness consequence:

```text
substrate_id=time_traveler_l5_z7_lstm_predictive_coding
readiness_verdict=DEFER
trainer_path=experiments/train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py
full_main_implemented=true
full_main_blocker=null
top_blocker=z7_score_aware_one_pair_smoke_not_contest_authority
predicted_score_band=[0.10, 0.13]
predicted_band_axis=contest-CPU
score_claim=false
promotion_eligible=false
```

Refreshed queue:

```text
.omx/state/asymptotic_pursuit/dispatch_queue_20260518T130645Z.json
```

This is not spend authority. It turns the top row from "missing implementation"
and "proxy-only trainer" into a concrete build gate: scale the score-aware
one-pair smoke to a ratified packet, then validate recurrent-vs-static-control
under same-archive-byte paired exact-eval inputs only after Wave N+1 council and
C6 beta anchor are ready.

## Z7-GRU score-aware one-pair smoke landed

The Z7 trainer now accepts an explicit `--loss-mode score_aware` path that loads
frozen differentiable scorers at compress time only, routes reconstructed pairs
through `score_pair_components_dispatch(...)`, and still emits a scorer-free
runtime tree. The default remains `proxy` for cheap local export smoke.

Local artifact:

```text
experiments/results/z7_gru_score_aware_smoke_codex_20260518T1255Z/
archive_zip_bytes=5155
archive_zip_sha256=a8a4dcc3aebb4d37317a0b1afa4bec129ef04f531584a3012dbef9616914a92b
loss_mode=score_aware
final_loss=39.47663497924805 [trainer-score-aware local CPU smoke; not contest-CPU or contest-CUDA]
seg_term=0.3863946497440338
pose_term=0.06964391469955444
score_claim=false
promotion_eligible=false
```

Same-byte static control:

```text
static_control_archive_zip_bytes=5155
static_control_archive_zip_sha256=2294467c9813e9497576b958c573aa5766066254893e7a8fea86e8c9db3e02c3
same_archive_zip_bytes_as_recurrent=true
runtime_output_changed_vs_recurrent=true
runtime_output_byte_differences_vs_recurrent=1011870
```

Measured local timing from the same artifact:

```text
timing_axis=[local-trainer-timing advisory]
seconds_per_epoch=1.3555798749439418
seconds_per_pair_epoch=1.3555798749439418
pairs_per_second_epoch=0.7376916834512268
score_aware_scorer_load_seconds=0.5821676668711007
inflate_verify_seconds=0.12976029119454324
```

The lift also fixed a one-pair NaN bug in `_ego_motion_from_pairs(...)` by using
population std (`unbiased=false`). This matters because timing/smoke probes often
start at `--max-pairs 1`.

## Z7 probe-custody queue refresh

The Z7 temporal-vs-static disambiguator now has a committed fail-closed JSON
contract and is wired into the Z7 recipe/readiness parser:

```text
.omx/research/probe_z7_temporal_coherence_vs_static_capacity_disambiguator_20260518_codex.json
schema=z7_temporal_coherence_vs_static_capacity_disambiguator_v1
verdict=pending_paired_exact_eval_json
score_claim=false
promotion_eligible=false
ready_for_paid_dispatch=false
same_archive_bytes_required=true
required_axis=contest_cuda
```

Refreshed queue:

```text
.omx/state/asymptotic_pursuit/dispatch_queue_20260518T131951Z.json
top_1_substrate=time_traveler_l5_z7_lstm_predictive_coding
top_1_readiness_verdict=DEFER
local_identity_disambiguator_probe.verdict=pending_paired_exact_eval_json
local_identity_disambiguator_probe.blockers=[]
ready_for_paid_dispatch=false
ready_for_paid_dispatch_count=0
```

Adversarial note: this is a custody/readiness hardening artifact, not a score
claim. It keeps the Z7 blocker machine-readable so a future exact-eval JSON
pair must satisfy same-axis, same-sample-count, same-archive-byte comparison
before the queue can treat recurrent temporal coherence as evidence.

Catalog #202 hygiene addendum: the same pass found and fixed a sentinel parser
bug where list-valued recipe paths could appear as literal list strings in the
Modal sentinel snapshot. The Z7 recipe now keeps only mounted worker files in
`sentinel_files` and moves `.omx` control-plane docs to `dependencies`.
The refreshed queue has no malformed list sentinel and no outside-mount
sentinel for Z7; the only remaining snapshot blocker is the intentionally
unlanded future remote driver
`scripts/remote_lane_substrate_time_traveler_l5_z7_lstm_predictive_coding.sh`.

## Z7 remote-driver actuator landed

The Z7 remote driver is now present and tested:

```text
scripts/remote_lane_substrate_time_traveler_l5_z7_lstm_predictive_coding.sh
src/tac/tests/test_time_traveler_l5_z7_remote_driver.py
```

It is a claim-verified no-score actuator for the existing score-aware timing
smoke. It refuses missing lane claims, writes provenance, rejects stale/missing
stats, and terminalizes claimed jobs as either
`completed_z7_gru_remote_driver_no_score_claim` or a precise
`failed_z7_gru_*` status. Completion is blocked if the stats JSON asserts
`score_claim`, `promotion_eligible`, or `ready_for_paid_dispatch`.

Fresh dispatch queue:

```text
.omx/state/asymptotic_pursuit/dispatch_queue_20260518T132746Z.json
top_1_substrate=time_traveler_l5_z7_lstm_predictive_coding
top_1_readiness_verdict=DEFER
ready_for_paid_dispatch_count=0
catalog202.current_sentinel_snapshot.missing_sentinel_files=[]
catalog202.current_sentinel_snapshot.snapshot_blockers=[]
catalog202.current_sentinel_snapshot_valid=true
```

Interpretation: the old Z7 Catalog #202 missing-remote-driver blocker is
cleared. This does not authorize spend or promotion. Z7 remains blocked by
research-only dispatch state, `PROCEED_WITH_REVISIONS`, the one-pair smoke
being non-authoritative, Wave N+1 council, C6 beta-anchor evidence, and paired
same-byte recurrent-vs-static exact-eval JSONs on the proper contest axis.

No provider job was launched and no lane claim was opened.
