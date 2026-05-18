# Z6 Candidate 4c full-600 zero-epoch handoff packet

review_id: z6_candidate4c_full600_zeroepoch_handoff_20260518_codex
lane_id: lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518
generated_utc: 2026-05-18T10:49:31Z
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
provider_dispatch_attempted: false

## Result

The previous Candidate 4c exact-eval handoff blocker was
`candidate4c_exact_handoff_latest_archive_pair_not_600_pairs`. I materialized a
local CPU, no-provider, no-auth-eval full-600 archive pair to determine whether
that blocker was just missing custody or a full-path runtime failure.

The blocker is now cleared for handoff planning:

- latest launch packet:
  `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T104931Z.json`
- refreshed queue:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T104919Z.json`
- refreshed disambiguator:
  `.omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.json`
- local archive packet:
  `experiments/results/z6_candidate4c_full600_zeroepoch_packet_codex_20260518T104201Z/`

Key fields from the no-spend launch packet:

- `exact_eval_handoff.ready_for_exact_eval_handoff=true`
- `exact_eval_handoff.latest_pair_count=600`
- `ready_for_operator_paid_execution=false`
- `result_review_blockers=[
  candidate4c_recipe_dispatch_disabled_exact_eval_handoff_required,
  candidate4c_paid_training_launch_not_in_scope_recipe_dispatch_disabled]`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`

Interpretation: Candidate 4c now has a byte-closed full/identity 600-pair
archive pair suitable for a future claimed exact-eval handoff. It is not a
score claim, not a promotion claim, and not evidence that the zero-epoch
configuration is competitive.

## Local commands

The archive-pair materialization was run under a local CPU lane claim and then
closed terminally:

```bash
.venv/bin/python experiments/train_substrate_time_traveler_l5_z6.py \
  --video-path upstream/videos/0.mkv \
  --output-dir experiments/results/z6_candidate4c_full600_zeroepoch_packet_codex_20260518T104201Z \
  --epochs 0 \
  --batch-size 1 \
  --device cpu \
  --full-cpu \
  --advisory-cpu-explicitly-waived \
  --skip-auth-eval \
  --ego-source scorer_logit \
  --predictor-architecture single_layer_film_75k \
  --predictor-hidden-dim 72 \
  --predictor-film-mlp-hidden-dim 32 \
  --predictor-param-count-target 120000 \
  --predictor-ego-motion-dim 8 \
  --emit-identity-predictor-disambiguator-archive
```

The full-vs-identity local inflate-output comparison was also run under a
local CPU lane claim and then closed terminally:

```bash
.venv/bin/python tools/probe_z6_predictive_coding_vs_identity_disambiguator.py \
  --run-dir experiments/results/z6_candidate4c_full600_zeroepoch_packet_codex_20260518T104201Z \
  --inflate-sh experiments/results/z6_candidate4c_full600_zeroepoch_packet_codex_20260518T104201Z/submission_dir/inflate.sh \
  --file-list upstream/public_test_video_names.txt \
  --inflate-output-root experiments/results/z6_candidate4c_full600_zeroepoch_packet_codex_20260518T104201Z/z6_identity_inflate_output_comparison_runtime_source_custody_compact_20260518_codex \
  --output-json .omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.json \
  --output-md .omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.md
```

## Custody

Full-FiLM archive row:

- parsed archive bytes: `213864`
- parsed archive sha256:
  `0f448055c3147d7cae865e31a8d40ae441617e3aa20cd5c38214f14c26957778`
- ZIP bytes: `211866`
- ZIP sha256:
  `5b371490b4459b85e95e6173653fc1b9aa78010681862ec51111166a6c867c4b`
- parsed pair count: `600`

Identity-predictor archive row:

- parsed archive bytes: `214183`
- parsed archive sha256:
  `8c274c5a11b38a9f43e3ee54bd03f47743de8e88509705e84a7b6dfa2035c9b8`
- ZIP bytes: `212047`
- ZIP sha256:
  `e6cd9bf67ca68bcdf93aa0e804435b75b813e420d5e3964b3a6cb6cee28e3589`
- parsed pair count: `600`

Paired archive structural checks:

- encoder state dict equal: `true`
- decoder state dict equal: `true`
- predictor state dict equal: `true`
- latent init equal: `true`
- residuals equal: `true`
- ego-motion equal: `true`
- identity-vs-full ZIP byte delta: `181`
- identity-vs-full rate-term basis: `0.00012052047051511301`

Local inflate-output comparison:

- evidence axis: `[local-inflate-output advisory]`
- evidence grade: `byte_closed_archive_pair_no_score`
- runtime output changed: `true`
- total byte differences: `33048720`
- full output bytes: `3662409600`
- identity output bytes: `3662409600`
- full output aggregate sha256:
  `241f9cf0d6234a728a165173e0f352beb5254d358dacf0e6d7ff027b0f58c712`
- identity output aggregate sha256:
  `5c0673169daabf7a90cddaa86b23b157019f96c63f68daa36eed786be368d94e`
- runtime custody aggregate sha256:
  `384938f3b6a14acb5944938c45c127ccc411f00983aff7589c7c9b98cfc56073`

The local inflate comparison generated large rebuildable raw outputs under
`experiments/results/.../z6_identity_inflate_output_comparison_runtime_source_custody_compact_20260518_codex/`.
They remain in the ignored experiment artifact area; the committed ledger
should carry hashes and custody, not the raw payload bytes.

## Timing

Stage log from `provenance.json`:

- `2026-05-18T10:42:18Z`: scorers loaded
- `2026-05-18T10:42:25Z`: 600 pairs decoded
- `2026-05-18T10:45:41Z`: scorer-logit ego-motion derived
- `2026-05-18T10:45:41Z`: archive pair emitted

The zero-epoch materialization therefore spent about 7 seconds on pair decode
and about 196 seconds on CPU scorer-logit side-info derivation. Archive
emission itself was effectively immediate.

## Next gate

Do not launch the existing Candidate 4c Modal training recipe as a
contest-CUDA claim. It remains intentionally diagnostic-only:

- `recipe_status.current_mode=diagnostic_only_exact_eval_handoff_required`
- `dispatch_enabled=false`
- `smoke_only=true`
- `target_modes` omit `contest_exact_eval`

The next score-bearing action is a fresh claimed paired exact-eval handoff
using the no-spend packet's generated commands, preserving separate
`[contest-CUDA]` and `[contest-CPU]` axes for full and identity archives.

## 2026-05-18 paired-dispatch command repair

Adversarial review found one launch-surface bug after the full-600 archive pair
was materialized: `tools/verify_candidate4c_launch_packet.py` generated four
direct single-axis Modal wrapper commands for the post-harvest eval handoff
(`experiments/modal_auth_eval.py` plus `experiments/modal_auth_eval_cpu.py` for
full and identity). That was custody-weak even though no provider job was
launched, because an operator could copy only one axis and lose the intended
CPU/CUDA pairing.

The packet doctor now emits only canonical paired-dispatch commands through
`tools/dispatch_modal_paired_auth_eval.py`:

- `full_paired_contest_cpu_cuda`
- `identity_paired_contest_cpu_cuda`

For each archive mode the no-spend packet records a plan command without
`--execute` and an execution command with `--execute`. Both commands use
`--expected-runtime-tree-sha256 auto` and
`--skip-axis-if-promotable-anchor-exists`, with lane bases:

- `lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518_full`
- `lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518_identity`

Latest packet:

- `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T105922Z.json`
- `exact_eval_handoff.ready_for_exact_eval_handoff=true`
- `exact_eval_handoff.latest_pair_count=600`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_operator_paid_execution=false`

The final `ready_for_operator_paid_execution=false` remains intentional: the
Modal training recipe is still diagnostic-only and disabled for paid training.
The only ready surface in this packet is the byte-closed post-harvest paired
exact-eval handoff.

Verification:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_verify_candidate4c_launch_packet.py
# 11 passed in 0.19s

.venv/bin/python -m py_compile \
  tools/verify_candidate4c_launch_packet.py \
  src/tac/tests/test_verify_candidate4c_launch_packet.py
# PASS

.venv/bin/python tools/verify_candidate4c_launch_packet.py --json --write-artifact \
  --queue-path .omx/state/asymptotic_pursuit/dispatch_queue_20260518T104919Z.json
# rc=1; wrote candidate4c_no_spend_launch_packet_20260518T105922Z.json
# exact-eval handoff ready, paid training still refused by design
```

## 2026-05-18 paired Modal exact-eval dispatch

Race Mode was active and no live lane claims conflicted, so the repaired
handoff was actuated through the canonical paired dispatcher. Four detached
Modal auth-eval calls were accepted:

- full `[contest-CUDA]`: `fc-01KRXC3V6N13J3H9R5XZSXHPQ1`
- full `[contest-CPU]`: `fc-01KRXC4EZ3GY615KF1EJE33VZ2`
- identity `[contest-CUDA]`: `fc-01KRXC3WYZKE2ZEE04R7P714KE`
- identity `[contest-CPU]`: `fc-01KRXC4M333B38CRV7Q6HXNVN1`

Dispatch ledger:

- `.omx/research/z6_candidate4c_paired_modal_exact_eval_dispatch_20260518_codex.md`

Current status:

- active claims: `4`
- stale nonterminal claims: `0`
- initial recovery: `status=pending` for all four calls at
  `2026-05-18T11:05:20Z`
- `score_claim=false`
- `promotion_eligible=false`

Do not re-dispatch these lane ids while the active Modal calls are pending.
Next action is harvest through `tools/recover_modal_auth_eval.py`, then
adjudicate full vs identity separately on `[contest-CUDA]` and `[contest-CPU]`.

## 2026-05-18 CUDA exact-eval result

Both `[contest-CUDA]` calls recovered:

- full: `90.58142803863508`
- identity: `90.58427695093009`
- identity-minus-full score: `0.0028489122950077217`
- identity-minus-full bytes: `181`
- identity-minus-full PoseNet: `0.021804809999991903`
- SegNet delta: `0`

This is a measured zero-epoch configuration failure, not a Candidate 4c method
kill. Full FiLM is lower than identity on `[contest-CUDA]`, but the score delta
is below `decision_delta_s=0.005`, and both scores are far above the current
frontier. The zero-epoch packet must therefore remain a control/anchor only.

The two `[contest-CPU]` calls were still pending at `2026-05-18T11:09:16Z`.

## 2026-05-18 CPU exact-eval result and paired closure

Both `[contest-CPU]` calls recovered:

- full: `90.57816474855734`
- identity: `90.58102532784203`
- identity-minus-full score: `0.0028605792846860822`
- identity-minus-full bytes: `181`
- identity-minus-full PoseNet: `0.021896370000007437`
- SegNet delta: `0`

Final paired exact-eval classification:

- `[contest-CUDA]` and `[contest-CPU]` agree that full FiLM is lower than the
  identity predictor for this zero-epoch packet.
- The margin is below the `0.005` disambiguator threshold on both axes.
- The zero-epoch packet is not a trained Candidate 4c result and scores around
  `90.58`, so it is useful only as a negative control / consumed-by-inflate
  anchor.
- Active lane claims are closed: `active=0`, `stale_nonterminal=0`.
