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
| 2 | Z7-as-GRU / Mamba | `time_traveler_l5_z7_lstm_predictive_coding` | `[contest-CPU]` | `DEFER` | `land_z7_mamba_design_memo_recipe_and_timing_smoke` | New research-only recipe/readiness surface exists; blocks on missing trainer, missing substrate package, Z6 4c paired exact-eval outcome, Wave N+1 council, C6 beta anchor, and paired exact-eval JSONs for the Z7 disambiguator. |
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

Z7 blockers carried forward:

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
- `z7_requires_same_archive_bytes_identity_disambiguator_before_full_dispatch`

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

This removes the "probe tool absent" blocker while preserving the real blockers:
no Z7 trainer, no Z7 substrate package, no paired exact-eval JSONs, and no
Wave N+1 council authorization.

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
