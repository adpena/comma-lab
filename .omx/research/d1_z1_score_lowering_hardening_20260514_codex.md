# D1 + Z1 Score-Lowering Hardening - 2026-05-14

## Scope

This ledger records code-backed hardening from the 2026-05-14 score-lowering
pass. It is not a strategy memo; it captures dispatch-affecting fixes and exact
failure classifications.

## D1 SegNet Margin Polytope

Verdict: `blocked_l1_noop_overlay`.

The D1 L1 packet is now classified as a custody scaffold, not a score-lowering
candidate. Its runtime verifies `d1_polytope.bin` and the frozen A1 base hash,
then delegates to A1 without applying the polytope overlay. Therefore exact eval
would measure A1 output plus D1 sidecar rate overhead, not the D1 hypothesis.

Hardening landed:

- `experiments/train_substrate_d1_segnet_margin_polytope.py` skips auth eval
  unless readiness reports `ready_for_exact_eval_dispatch=true`.
- The generated runtime includes `d1_verify.py`, a stdlib-only sidecar/base
  custody guard called by `inflate.sh`.
- `src/tac/substrates/d1_segnet_margin_polytope/archive.py` emits
  `runtime_overlay_consumed=false`, null current predicted score bands, and
  L2-only projected bands.
- `.omx/operator_authorize_recipes/substrate_d1_segnet_margin_polytope_modal_t4_dispatch.yaml`
  now has `dispatch_enabled=false` with blockers:
  `d1_l1_runtime_overlay_not_consumed`, `l2_overlay_adapter_missing`,
  `current_l1_packet_base_renderer_plus_rate_only`, and
  `do_not_dispatch_until_ready_for_exact_eval_dispatch_true`.

Remote result classification:

- Latest D1 smoke produced archive SHA
  `3e4a59e849d17895f3ec38562d73f66235a03cce49b98f5afe03fd254e62c7db`
  and passed inflate strict validation, then failed inside DALI/NVML with
  `nvml error (999)`.
- Because the L1 runtime did not consume the overlay, this is not a model
  result and not a score result. It is `blocked_l1_noop_overlay` plus
  infrastructure failure, not a D1 negative.

Reactivation criterion:

Build the L2 fused D1+A1 runtime that applies the polytope overlay during
inflate, flips `runtime_overlay_consumed=true`, and passes byte-closed exact
eval custody.

## Z1 Scorer-Conditional MDL

Verdict: `parser_conditioned_headroom_small`.

The Z1 ablation now slices PR106 sidecar section offsets against the parser
input payload, not the outer sidecar wrapper. It also validates eval JSON
custody against archive bytes/SHA and infers legacy Lightning exact-eval paths
as `contest_cuda`.

Clean artifact:

- `experiments/results/z1_scorer_conditional_mdl_20260514_codex_v2/scorer_conditional_mdl_ablation.json`

Result:

- A1 axis: `cpu_advisory`, custody matched.
- PR106x axis: `contest_cuda`, custody matched.
- HDM8 axis: `contest_cuda`, custody matched.
- Parser-role conditioned gain: 152 bytes.
- Scorer-feature proxy gain: 212 bytes.

Routing implication:

Generic parser/ZIP/Brotli byte shaving is near saturation for these packets.
Next score-lowering work should target decoded semantic/scorer-bound transforms
or runtime-consumed overlays rather than more archive-header micro-optimization.

## D4 Wyner-Ziv Frame-0

Verdict: stale-source OOM for the first failed smoke; current local code is
mini-batched.

The harvested D4 failure at
`substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch_20260514T131025Z__smoke__100ep`
OOMed in a stale remote full-batch call to `reconstruct_pair(gt_f1_all)`.
Current local source reconstructs with `pair_indices` inside the mini-batch
loop and the later T4 retry metadata matches current trainer/architecture SHAs.

Verification:

- `python -m pytest src/tac/substrates/d4_wyner_ziv_frame_0/tests/test_d4_substrate.py -q`
  passed 65 tests, including the mini-batch/OOM guard tests.

## HDM8 Postfilter Recover

Verdict: recover-sweep robustness fix.

The HDM8 postfilter recovery loop encountered a dry-run directory without a
Modal spawn manifest. The recover CLI now recognizes dry-run request artifacts
and writes `dry_run_no_remote_call` instead of raising `FileNotFoundError`.
This prevents batch harvest loops from stopping before checking later live
calls.

Verification:

- `python -m pytest src/tac/tests/test_modal_hdm8_postfilter_sweep.py -q`
  passed 8 tests.
- Live HDM8 calls remained `pending` as of the latest recover pass; no score
  claim was created.

## PR101 Nonlocal Sweep

Verdict: no dispatch under the operator's `<0.192` continuation threshold.

The PR101/HNeRV nonlocal sweep produced byte-closed packets, but the best new
local macOS CPU advisory packet regressed relative to PR101 local CPU baseline.

Evidence:

- PR101 local CPU advisory baseline: `0.1928610127024255`.
- `bias_refine_cmaes_0047` local CPU advisory: `0.19313928561413787`.
- Prior PR101 exact CUDA negatives remain around `0.2265`.
- Best current HNeRV-family CUDA anchor in this set remains HDM8 at
  `0.20636166502462222` `[contest-CUDA]`.

Routing implication:

Do not dispatch the PR101 bias-refine packet family. Reopen only for a
CUDA-derived or non-bias mechanism with charged bytes, runtime-consumption
proof, exact-readiness, and a fresh lane claim.

## Frame Exploit Selector

Verdict: promising mechanism family, but current FES1 is a CUDA transfer
failure and must not be ranked from CPU/MPS positives.

New artifacts:

- `experiments/results/frame_exploit_selector_packet_20260514_codex/archive.zip`
  bytes `187209`, SHA-256
  `30ace55bcaef0d86e4164bd75fd7e3e04682f97eb9c8e9e059655cb457793e2e`.
- `experiments/results/frame_exploit_cuda_transfer_audit_20260514_agent_codex/audit.json`.

Measured axis split:

- FES1 versus PR106 source on CPU: score delta `-0.019201533209164928`,
  mostly PoseNet improvement.
- FES1 versus PR106 source on CUDA: score delta `+0.019480460642093023`,
  PoseNet regression.

Routing implication:

CPU/MPS frame-exploit positives are proposal priors only. The highest-EV next
CUDA control was the byte-closed all-none FES1 runtime-control archive at
`experiments/results/frame_exploit_cuda_transfer_audit_20260514_agent_codex/fes1_all_none_control/archive.zip`.
It has already landed on `[contest-CUDA]`:

- All-none control SHA-256:
  `72a9516608e9f8770211f763d1e5689d6ab31e5594e3ff0b618828dd83193a8b`.
- All-none control score: `0.20690367421246367` `[contest-CUDA]`.
- PR106 source score: `0.20664588545741508` `[contest-CUDA]`.
- Difference is the charged selector rate penalty; PoseNet/SegNet components
  match PR106 source (`avg_posenet_dist=3.236e-05`,
  `avg_segnet_dist=0.0006426`).

Conclusion: the FES wrapper/trailer is CUDA-safe. The learned perturbations,
not the runtime wrapper, caused the CUDA PoseNet regression. FES2 should rank
from CUDA rows only, with the all-none control as the baseline.

Verification:

- `python -m pytest -q src/tac/tests/test_frame_exploit_selector_packet.py src/tac/tests/test_frame_exploit_cuda_transfer_audit.py src/tac/tests/test_frame_exploit_segnet_posenet_sweep.py`
  passed 11 tests.
- `experiments/results/modal_auth_eval/frame_exploit_fes1_all_none_control_cuda_20260514T133300Z/contest_auth_eval.json`
  is the exact CUDA control artifact.

## Verification

Focused gates run locally:

```text
python -m pytest src/tac/substrates/d1_segnet_margin_polytope/tests/test_d1_substrate.py src/tac/tests/test_scorer_conditional_mdl_cli.py src/tac/tests/test_hnerv_packet_sections.py -q
84 passed

python -m pytest src/tac/substrates/d4_wyner_ziv_frame_0/tests/test_d4_substrate.py -q
65 passed

python -m pytest src/tac/tests/test_modal_hdm8_postfilter_sweep.py -q
8 passed

python -m pytest -q src/tac/tests/test_frame_exploit_selector_packet.py src/tac/tests/test_frame_exploit_cuda_transfer_audit.py src/tac/tests/test_frame_exploit_segnet_posenet_sweep.py
11 passed

git diff --check -- <touched files>
passed
```

## PacketIR Current-Frontier Reference Fix

Finding after the first closure refresh: `tools/build_pr106_r2_packetir_exact_closure.py`
was still using the older HLM1 exact CUDA artifact as its default
`current_best_cuda_eval`, while the live exact CUDA frontier had moved to
HDM8:

- current exact CUDA reference:
  `experiments/results/modal_auth_eval/hnerv_hdm8_fixed_lengths_modal_t4_retry2_20260514T095000Z/contest_auth_eval.json`
- score: `0.20636166502462222` `[contest-CUDA]`
- archive bytes: `186395`

Fix landed:

- `src/tac/hnerv_frontier_defaults.py` now exposes
  `ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_EVAL`.
- `tools/build_pr106_r2_packetir_exact_closure.py` consumes that canonical
  path instead of hard-coding the stale HLM1 eval path.
- `src/tac/tests/test_packetir_exact_closure.py` now asserts the closure tool
  uses the active HDM8 reference.

Regenerated closure:

- `experiments/results/pr106_r2_packetir_exact_closure_20260513_codex/closure.json`
- classification: `exact_measured_not_current_frontier`
- blockers: none
- `delta_vs_source_cuda`: `-0.00010065943776227382`
- `delta_vs_current_best_cuda`: `+0.00015581099503059193`

This preserves the positive 151-byte PacketIR rate-only measurement while
preventing stale-frontier routing or accidental duplicate dispatch.

Additional gates:

```text
python -m pytest -q src/tac/tests/test_packetir_exact_closure.py
11 passed

python -m pytest -q src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_operator_briefing.py src/tac/tests/test_all_lanes_pr106_sidecar_runtime_gate.py
45 passed
```

## Recursive Evidence-Surface Hardening

Fresh adversarial review found three authority-leak surfaces that could have
turned useful signal into a wrong routing decision:

1. `auth_eval_roundtrip_results.json` rows could preserve recovered
   `[contest-CUDA]` / `[contest-CPU]` scores while leaking row-level
   `score_claim=true`.
2. Z1 eval JSON custody treated missing archive identity as a match.
3. PacketIR CPU-axis diagnostic rows carried empty promotion/rank blockers.

Fixes landed:

- `src/tac/auth_eval_roundtrip_matrix.py` now makes every recovered row
  `score_claim=false`, `rank_or_kill_eligible=false`, and adds
  `contest_axis_anchor`, `score_claim_possible_after_result_review`, and
  explicit `result_review_blockers`.
- `src/tac/analysis/scorer_conditional_mdl.py` now reports custody strength:
  `archive_sha256_and_bytes` for strong matches and
  `missing_archive_identity` plus blockers when an eval JSON lacks archive
  bytes/SHA.
- `src/tac/packetir_exact_closure.py` now marks non-CUDA axes with
  `not_contest_cuda_axis`, `cpu_axis_not_rank_or_kill_authority`, and
  `requires_cuda_cpu_policy_review` in both promotion and rank/kill blocker
  arrays.

Permanent self-protection:

- Catalog #221 claimed.
- `src/tac/preflight.py` now includes
  `check_auth_eval_result_artifacts_fail_closed_for_score_claims(strict=True)`.
- `CLAUDE.md` catalog row #221 documents the bug class and strictness.
- `src/tac/tests/test_preflight_auth_eval_result_artifact_guards.py` covers
  row-level score-claim leakage, missing blockers, CPU-axis diagnostic
  blocker omission, strict raises, live-repo cleanliness, and `preflight_all`
  strict wiring.

Regenerated artifacts:

- `experiments/results/frame_exploit_selector_packet_20260514T125909Z_codex_v4/auth_eval_roundtrip_results.json`
  still preserves all 4 recovered rows and 3 pending rows, but now every row is
  non-promotional.
- `experiments/results/pr106_r2_packetir_exact_closure_20260513_codex/closure.json`
  now carries CPU diagnostic blockers while preserving the PacketIR
  classification `exact_measured_not_current_frontier`.
- `experiments/results/z1_scorer_conditional_mdl_20260514_codex_v2/scorer_conditional_mdl_ablation.json`
  was regenerated with strong custody for A1, PR106x, and HDM8.

Verification:

```text
python -m pytest -q src/tac/tests/test_preflight_auth_eval_result_artifact_guards.py src/tac/tests/test_auth_eval_roundtrip_matrix.py src/tac/tests/test_scorer_conditional_mdl_cli.py src/tac/tests/test_packetir_exact_closure.py src/tac/tests/test_build_pr106_sidecar_rank_elided_candidate.py src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py
53 passed

python -m pytest -q src/tac/tests/test_preflight_deep_hardening_pass_3.py src/tac/tests/test_preflight_auth_eval_result_artifact_guards.py
19 passed

check_auth_eval_result_artifacts_fail_closed_for_score_claims(strict=False)
0 violations across 1 roundtrip result file and 2 PacketIR closures

check_strict_preflight_callsites_have_claude_md_catalog_row(strict=False)
0 violations across 174 strict callsites
```

## PR106/R2 Rank-Elided Sidecar Prototype

The PR101 sidecar grammar has a fixed one-byte Huffman length-rank blob and a
metadata byte recording that width. For the PR106 schema, the delta alphabet has
four symbols and the Kraft-tight length vector is uniform two-bit, so a
successor runtime can infer the length rank instead of storing it.

Subagent build landed:

- `tools/build_pr106_sidecar_rank_elided_candidate.py`
- `src/tac/tests/test_build_pr106_sidecar_rank_elided_candidate.py`

Generated local artifact:

- candidate archive:
  `/tmp/pact_pr106_hdm7_rank_elided_format04_20260514/pr106_sidecar_rank_elided_format04_candidate.zip`
- candidate bytes: `186401`
- source bytes: `186405`
- delta: `-4` bytes

Classification:

`research_only=true`, `score_claim=false`, `ready_for_exact_eval_dispatch=false`.
The artifact is byte-closed as a prototype, but not byte-closed for the existing
runtime until a format-`0x04` decoder, runtime-consumption proof, and
same-runtime full-frame parity exist. It is a useful PacketIR successor
prototype, not a dispatchable score candidate.

Evidence-axis note:

All local results here are code/tests/custody classifications. No new
leaderboard score claim is made by this ledger.

## C6 Modal Smoke Harvest + Post-Train Crash Hardening

No-signal-loss harvest:

- `tools/harvest_modal_calls.py --execute --repo-root . --get-timeout-seconds 5`
  harvested two terminal C6 smoke failures and recorded terminal lane-claim
  rows for them.
- The second C6 Modal smoke
  `substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260514T142426Z__smoke__100ep`
  failed after scorer construction with `TypeError: new(): invalid data type
  'str'` at `src/tac/losses/core.py:740`.
- Root cause: `experiments/train_substrate_c6_e4_mdl_ibps.py` reversed the
  canonical scorer-loader return order. The API returns `(posenet, segnet)`;
  the trainer had assigned `(seg_scorer, pose_scorer)`.

Permanent self-protection:

- Catalog #222 claimed.
- `experiments/train_substrate_c6_e4_mdl_ibps.py` and the C6 real-scorer test
  now assign `pose_scorer, seg_scorer = load_differentiable_scorers(...)`.
- `src/tac/preflight.py` now includes
  `check_scorer_loader_assignment_order(strict=True)` to reject reversed
  canonical scorer-loader assignments before GPU spend.
- `src/tac/tests/test_preflight_scorer_loader_assignment_order.py` covers
  reversed differentiable/default loader assignments, clean order, waiver
  semantics, strict raise, live-repo cleanliness, and `preflight_all` wiring.

The first local CPU one-pair C6 rerun then exposed a second post-training
artifact bug:

- The trainer reached `epoch=0 loss=95.522728`, saved EMA, and emitted the
  runtime tree, then crashed at stats emission because
  `_canon_detect_hardware_substrate()` was called without the required
  `substrate_tag=` keyword.

Permanent self-protection:

- `experiments/train_substrate_c6_e4_mdl_ibps.py` now calls
  `_canon_detect_hardware_substrate(axis=..., substrate_tag=SUBSTRATE_TAG,
  env_var_candidates=("C6_E4_MDL_IBPS_GPU", "MODAL_GPU"))`.
- The existing Catalog #190 guard was extended to reject canonical detector
  calls under `experiments/train_substrate_*.py` that omit `substrate_tag=`.
- The stricter guard immediately found the same missing-tag pattern in
  `experiments/train_substrate_a1_plus_lapose.py`; that trainer now passes
  `substrate_tag="a1_plus_lapose"` and its GPU env ladder.

Verification:

```text
python -m pytest -q src/tac/tests/test_check_190_substrate_trainer_hardware_substrate.py
18 passed

check_substrate_trainer_does_not_hardcode_hardware_substrate(strict=False)
0 violations across 27 substrate trainers

python -m pytest -q src/tac/tests/test_check_190_substrate_trainer_hardware_substrate.py src/tac/tests/test_preflight_scorer_loader_assignment_order.py src/tac/substrates/c6_e4_mdl_ibps/tests/test_c6_substrate.py src/tac/tests/test_preflight_auth_eval_result_artifact_guards.py
72 passed

python experiments/train_substrate_c6_e4_mdl_ibps.py --output-dir experiments/results/c6_scorer_order_fix_local_cpu_20260514_codex --epochs 1 --batch-size 1 --max-pairs 1 --val-every-epochs 1 --val-pair-count 1 --full-cpu --advisory-cpu-explicitly-waived --skip-auth-eval --skip-archive-build --device cpu
exited 0; wrote archive.zip and stats.json
```

Local advisory artifact:

- `experiments/results/c6_scorer_order_fix_local_cpu_20260514_codex/stats.json`
- `archive_bytes=205295`
- `hardware_substrate=linux_x86_64_modal_cpu`
- `auth_eval_evidence_grade=skipped`

This is not a score claim. It is a trainer/artifact-path proof that C6 no
longer dies before terminal stats on the local CPU one-pair path.

## C6 Auth-Eval CLI + False Completion Hardening

Adversarial review then found a third C6-critical issue:

- The inline C6 auth-eval command used stale `contest_auth_eval.py` flags:
  `--archive-zip` and `--output-json`, with no emitted runtime
  `--inflate-sh`.
- `contest_auth_eval.py` exits argparse rc=2 on those flags, but the trainer
  caught the failure, left `auth_eval_score=None`, wrote `stats.json`, printed
  `DONE`, and returned `0`.
- The remote driver treated trainer rc=0 as a completed lane and wrote
  `LANE_C6_MDL_IBPS_DONE [contest-CUDA]` even though no valid score existed.

Observed pre-fix remote artifact:

- Label:
  `substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260514T144949Z__smoke__5ep`
- Modal call id: `fc-01KRKFEQARQX6BC4EDPNHN1E74`
- Terminal classification:
  `completed_modal_training_recovered_no_score_claim`
- `trainer.log` shows the stale command and rc=2:
  `contest_auth_eval.py --archive-zip ... --output-json ...`
- `stats.json` has `auth_eval_score=null`, `auth_eval_evidence_grade=skipped`,
  and no score-claim fields because it was mounted before the fix.
- The stale completion marker is explicitly pre-fix and non-authoritative.

Permanent self-protection:

- Catalog #223 claimed.
- C6 now routes inline auth eval through
  `tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call(...)`.
  That canonical path uses current CLI flags `--archive`, `--inflate-sh`,
  `--json-out`, refuses non-CUDA inline claims, and raises on nonzero rc or
  invalid claim JSON.
- C6 full stats now fail closed with:
  `score_claim=false`, `score_claim_valid=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`,
  `auth_eval_score_claim_valid`, `auth_eval_score_axis`,
  `auth_eval_exact_cuda_complete`, and `result_review_blockers`.
- `scripts/remote_lane_substrate_c6_e4_mdl_ibps.sh` now validates
  `stats.json` before writing `LANE_C6_MDL_IBPS_DONE [contest-CUDA]`; it
  requires `auth_eval_score_claim_valid=true`,
  `auth_eval_score_axis=contest_cuda`, and
  `auth_eval_exact_cuda_complete=true`.
- The C6 remote script now routes Modal-runtime output from `/tmp`/workspace
  paths to `/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output`, matching the
  PR95++ durable-output fix and avoiding `contest_auth_eval.py` temp-evidence
  refusal for score-grade JSON.
- `src/tac/preflight.py` now includes
  `check_substrate_auth_eval_invocations_use_current_cli(strict=True)`.
  It rejects stale `--archive-zip` / `--output-json` substrate-trainer flags
  and direct auth-eval calls missing `--archive`, `--inflate-sh`, or
  `--json-out` unless the trainer routes through the canonical gate.
- `src/tac/substrates/_shared/trainer_skeleton.py` now labels local macOS CPU
  as `macos_arm64` instead of `linux_x86_64_modal_cpu`.

Verification:

```text
python -m pytest -q src/tac/tests/test_preflight_substrate_auth_eval_cli_contract.py src/tac/tests/test_trainer_skeleton.py src/tac/tests/test_check_190_substrate_trainer_hardware_substrate.py src/tac/tests/test_preflight_scorer_loader_assignment_order.py src/tac/tests/test_preflight_auth_eval_result_artifact_guards.py src/tac/substrates/c6_e4_mdl_ibps/tests/test_c6_substrate.py
122 passed

py_compile experiments/train_substrate_c6_e4_mdl_ibps.py src/tac/substrates/_shared/trainer_skeleton.py src/tac/preflight.py
passed

bash -n scripts/remote_lane_substrate_c6_e4_mdl_ibps.sh
passed

check_substrate_auth_eval_invocations_use_current_cli(strict=False)
0 violations across 26 substrate trainers

check_strict_preflight_callsites_have_claude_md_catalog_row(strict=False)
0 violations across 176 strict callsites

python experiments/train_substrate_c6_e4_mdl_ibps.py --output-dir experiments/results/c6_auth_eval_guard_local_cpu_20260514_codex --epochs 1 --batch-size 1 --max-pairs 1 --val-every-epochs 1 --val-pair-count 1 --full-cpu --advisory-cpu-explicitly-waived --skip-auth-eval --skip-archive-build --device cpu
exited 0; wrote archive.zip and fail-closed stats.json
```

Local advisory proof:

- `experiments/results/c6_auth_eval_guard_local_cpu_20260514_codex/stats.json`
- `hardware_substrate=macos_arm64`
- `score_claim=false`
- `score_claim_valid=false`
- `auth_eval_score_claim_valid=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- blockers:
  `trainer_stats_not_authoritative_score_claim_surface`,
  `promotion_requires_separate_result_review`,
  `skip_auth_eval_explicitly_set`,
  `contest_cuda_auth_eval_not_validated`

Patched remote verification dispatch:

- `substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260514T150207Z__smoke__1ep`
- Modal call id: `fc-01KRKG566Z2F48CVCGF8JFA0S1`
- Purpose: verify patched source mount + durable output + fail-closed
  auth-eval/remote completion contract.
- Status at ledger update: spawned/running; no score claim. Harvest command:
  `.venv/bin/python tools/harvest_modal_calls.py --execute --repo-root . --get-timeout-seconds 2`
  reported `not_ready`.

Follow-up harvest:

- `tools/harvest_modal_calls.py --execute --repo-root . --get-timeout-seconds 15`
  recovered
  `substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260514T150207Z__smoke__1ep`.
- Result: `rc=1`, elapsed `363.06s`, `timed_out=false`,
  `score_claim=false`, `promotion_eligible=false`.
- Inflate succeeded under the patched durable path:
  `archive.zip` bytes `222306`, emitted one `.raw` file of
  `3,662,409,600` bytes, strict validation passed, inflate elapsed `173.2s`.
- Upstream CUDA `evaluate.py` failed after inflate with DALI/NVML:
  `nvml error (999): A nvml internal driver error occurred`.
- Classification: `provider CUDA+DALI runtime failure after successful
  inflate`, not a C6 model negative and not a score claim.
- The patched fail-closed behavior worked: no false
  `LANE_C6_MDL_IBPS_DONE [contest-CUDA]` marker was written; terminal claims
  are `failed_c6_mdl_ibps_claim_verification_rc_1` and
  `failed_modal_training_rc_1`.

Permanent self-protection follow-up:

- Catalog #224 claimed.
- `experiments/modal_train_lane.py` now threads shared Modal runtime constants
  `DALI_DISABLE_NVML_VALUE` and `PYTORCH_CUDA_ALLOC_CONF_VALUE` into
  `image.env(...)`, generated `env.sh`, and the subprocess environment.
- Catalog #203's strict check now also rejects a Modal training dispatcher that
  omits required DALI/NVML and CUDA allocator env literals. This protects D1,
  C6, and any other generic Modal training lane that runs inline CUDA
  auth-eval after successful inflate.

## HDM8 postfilter active-claim harvest + D4 claim cleanup

Active score-lowering harvest attempt:

- `hdm8_local_first_cuda_confirm_20260514T115726Z`
  (`fc-01KRK5KKSQYCCMAG95CN4FCE0K`) recovered as `pending`.
- `hdm8_modal_t4_policy_palette_v1_20260514T113221Z`
  (`fc-01KRK453W4GT5A99XFTMQ3KXMF`) recovered as `pending`.
- `hdm8_multiplicative_cuda_probe_20260514T121530Z`
  (`fc-01KRK6M42CB2VGZCZASGHBSV72`) recovered as `pending`.
- `hdm8_tile_chroma_cuda_probe_20260514T123320Z`
  (`fc-01KRK7KYQVJH4BCPBBDF93BHRR`) recovered as `pending`.

Coordination hardening:

- `.omx/state/active_lane_dispatch_claims.md` incorrectly counted two
  already-timed-out D4 rows as active because their statuses started with
  `timed_out_...` rather than the terminal vocabulary.
- Appended terminal rows:
  `failed_modal_training_timeout` for
  `fc-01KRKA5DA13RH1CP5BQNDVAM3C` and
  `fc-01KRKB7GFKQE8Y1JNKRYBWS3RJ`.
- `tools/claim_lane_dispatch.py summary` now reports `active=5`, down from
  `active=7`: four HDM8 pending postfilter probes plus the patched C6 smoke.

Permanent self-protection follow-up:

- Catalog #225 claimed.
- `tools/claim_lane_dispatch.py` and `tac.deploy.claims` now treat
  `timed_out*` statuses as terminal, alongside the existing `failed_*`
  timeout vocabulary.
- This prevents terminal timeout rows such as `timed_out_3600s_t4_smoke` from
  being counted as active, which was causing stale-lane churn and false
  same-lane blocking.

## C6 patched retry harvest + no-signal-loss recovery hardening

Patched retry harvest:

- `tools/harvest_modal_calls.py --execute --repo-root . --get-timeout-seconds 15`
  recovered
  `substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260514T151415Z__smoke__1ep`
  (`fc-01KRKGVFRWR0SXX1V067DSZJPT`).
- Provider/trainer status: `rc=0`, elapsed `440.48s`, timed out `false`.
- Inline auth eval status: valid byte-closed `[contest-CUDA]` artifact present:
  `contest_auth_eval_cuda.json`, score `90.78433465890384`, archive
  `223,289` bytes, archive SHA prefix `a8f2492d6e44`.
- Component contributions: pose `40.15368121604793`, SegNet `50.482631`,
  rate `0.14802244285591806`.
- Classification: legitimate exact-CUDA negative for the measured C6 smoke
  configuration; severe scorer collapse, not a candidate and not a full-lane
  retirement.
- The smoke-before-full plausible-band guard correctly refuses to escalate this
  to full dispatch because the contest-CUDA score is far outside `[0, 10]`.

Signal-preservation hardening:

- `tac.deploy.modal.training_claims` now detects recovered inline
  `contest_auth_eval_cuda.json` artifacts and records a non-promotional
  `recovered_auth_eval` block in the terminal claim.
- `tools/harvest_modal_calls.py` terminal evidence now preserves that recovered
  auth-eval signal while keeping `score_claim=false`,
  `promotion_eligible=false`, and `rank_or_kill_eligible=false` for the
  infrastructure row.
- A superseding terminal claim row was appended:
  `completed_modal_training_recovered_with_contest_cuda_auth_eval`.
- This prevents Modal training recovery from collapsing three distinct states
  into one bucket: no score artifact, failed auth eval, and recovered exact
  CUDA auth eval.

Latest HDM8 score-lowering probe harvest:

- The four active HDM8 postfilter probes were recovered individually rather
  than chained. All remain `pending` as of `2026-05-14T15:26:21Z`:
  `hdm8_local_first_cuda_confirm_20260514T115726Z`,
  `hdm8_modal_t4_policy_palette_v1_20260514T113221Z`,
  `hdm8_multiplicative_cuda_probe_20260514T121530Z`,
  and `hdm8_tile_chroma_cuda_probe_20260514T123320Z`.

Late churn observation:

- A separate agent spawned
  `substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260514T152845Z__smoke__100ep`
  (`fc-01KRKHNZSZF4JPHJ20WE35A68C`) after the 1-epoch C6 exact-CUDA negative
  had already landed.
- `tools/harvest_modal_calls.py --execute --repo-root . --get-timeout-seconds 2`
  reports it as `not_ready`; no score claim exists yet.
- This should be harvested for custody, but should not trigger another full
  dispatch unless its own inline auth-eval score is in the plausible band and
  the out-of-band C6 scorer-collapse risk is explicitly cleared.
