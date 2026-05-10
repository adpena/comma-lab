# PR103 global-combo mid32 + latent-hi -16B clean-runtime CUDA result (2026-05-10)

## Verdict

The PR103 histogram transform stack has a second source-shell-contract Modal T4
CUDA positive.  The new packet preserves the source PR103 `inflate.sh`
contract, changes only consumed grammar lengths and payload bytes inside
`inflate.py` / `archive.zip`, and adds the latent-hi histogram byte contraction
on top of the previous global-combo transform:

- classification: `exact_source_shell_contract_cuda_positive_rate_only`
- evidence grade: `[contest-CUDA]`, Modal Tesla T4
- score movement versus source-shell-contract PR103 source: `-0.00001075`
- byte movement: `178223 -> 178207` (`-16` charged archive bytes)
- component movement: no measured movement at the reported component precision

This is a real score-lowering observation for the measured PR103 packet pair.
It is not yet a submission/promotion decision because the raw auth-eval result
still records the normal promotion blockers: adjudication, CPU counterpart, and
strict pre-submission compliance.

## Custody

Candidate packet:

- archive:
  `.omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_mid32_plus_latent_hi_candidate/packet/archive.zip`
- archive bytes: `178207`
- archive SHA-256:
  `8460014d70855ce9226285f80513d6d743ed23723870a6a38b009cfca40f423e`
- packet manifest:
  `.omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_mid32_plus_latent_hi_candidate/packet_manifest.json`
- packet-build runtime tree SHA-256, pre-dispatch support artifact:
  `d0fbc15d09906d241b468be30635057680f9fa94776c52e83bef41dbbdd1d600`
- adapter runtime tree SHA-256, pre-packet support artifact:
  `e5983ddfaebb050528a4c0a842eba6feb41dd41d187ad6c25c3e93f5be054542`
- clean runtime shell contract:
  `python "$HERE/inflate.py" "$SRC" "$DST"`
- full in-process render parity output SHA-256:
  `074f834f14ba4611f9358bb0a3f8e729bb43e4ea673be23e2acf85e7448dd1e5`

The authoritative evaluated runtime tree is the runtime manifest embedded in
the exact CUDA result JSON, not either pre-dispatch support manifest:

- evaluated candidate runtime tree SHA-256:
  `59c6a80f62b6bd8d7fab1b7252898b4dc19fa8736a91e2b7ecac6f8bb2e23ee2`
- evaluated source runtime tree SHA-256:
  `ea25380a6eee64b7f57a30a5e9c745fa6bd8867c728f92a293945cfd6dce5d42`
- interpretation: this is a source-shell-contract / same-evaluator CUDA pair,
  not an identical-runtime-tree pair.  The `inflate.py` hashes differ because
  the candidate intentionally changes consumed grammar constants
  (`HIST_LEN 895 -> 880`, `HI_HIST_LEN 15 -> 14`).

Candidate exact CUDA artifact:

- path:
  `experiments/results/modal_auth_eval/pr103_global_combo_mid32_latent_hi_16b_clean_runtime_exact_cuda_modal_20260510T2346Z/contest_auth_eval.json`
- pact commit: `e8e83f4303e467cb4211244cdf5f529525bf0eaa`
- GPU: `Tesla T4`
- canonical score: `0.22776742708207615`
- reported display score: `0.23`
- SegNet distance: `0.00067635`
- PoseNet distance: `0.00017199`
- archive bytes: `178207`
- evaluated runtime tree SHA-256:
  `59c6a80f62b6bd8d7fab1b7252898b4dc19fa8736a91e2b7ecac6f8bb2e23ee2`
- score contribution fields:
  - SegNet: `0.067635`
  - PoseNet: `0.04147167708207615`
  - rate: `0.11866075000000001`
- promotion blockers:
  `raw_auth_eval_does_not_verify_submission_policy_gates`,
  `cpu_leaderboard_reproduction_not_adjudicated`,
  `pre_submission_compliance_check_not_recorded`

Source-shell-contract PR103 source baseline:

- path:
  `experiments/results/modal_auth_eval/pr103_source_same_runtime_cuda_baseline_modal_20260510T2300Z/contest_auth_eval.json`
- pact commit: `6f4269b2b06644b35ded4380ffba6bfb367ada4d`
- canonical score: `0.22777817708207615`
- SegNet distance: `0.00067635`
- PoseNet distance: `0.00017199`
- archive bytes: `178223`
- archive SHA-256:
  `31881b2d23d027e6619f2d8df2fe35d4d207d08882ec673d6c1b7ff119f18c30`
- evaluated runtime tree SHA-256:
  `ea25380a6eee64b7f57a30a5e9c745fa6bd8867c728f92a293945cfd6dce5d42`
- score contribution fields:
  - SegNet: `0.067635`
  - PoseNet: `0.04147167708207615`
  - rate: `0.11867150000000001`

## Formula check

Using the contest rate term `25 * archive_bytes / 37,545,489`, a pure `-16`
byte contraction predicts approximately `-0.0000106537` score movement before
component serialization precision.  The measured Modal T4 pair reports
`0.22776742708207615 - 0.22777817708207615 = -0.00001075`, with the same
reported SegNet and PoseNet components and an exact recorded rate-contribution
movement of `0.11866075000000001 - 0.11867150000000001 = -0.00001075`.

## Adversarial classification

- Not a no-op: `HIST_LEN` changed `895 -> 880`, `HI_HIST_LEN` changed
  `15 -> 14`, the charged archive SHA changed, and runtime consumption probes
  parse the new lengths.
- Not a scorer or CUDA-axis extrapolation: both source and candidate went
  through Modal T4 `experiments/contest_auth_eval.py --device cuda`.
- Not a convenience-shell artifact: the candidate runtime was regenerated after
  removing the temporary `${PYTHON:-python}` shell rewrite and preserving the
  source PR103 shell contract.
- Still missing shell-level source-vs-candidate inflate-output parity as a
  local proof.  The source-shell-contract CUDA pair is the current score
  evidence; the in-process render parity proof remains necessary support, not a
  shell-parity substitute.
- Packet readiness manifests are intentionally pre-dispatch build artifacts and
  still say `dispatch_attempted=false` / `exact_cuda_auth_eval_missing`.  Do
  not read those fields as post-dispatch state.  The post-dispatch evidence is
  the exact CUDA result JSON plus the post-dispatch manifest added with this
  ledger.

## Next score-lowering work

1. Run strict pre-submission compliance on the candidate packet if it is to be
   promoted beyond internal frontier evidence.
2. Extend the grammar-aware histogram search to the next consumed sections only
   where decoder-state parity or full rendered-output parity can be proven.
3. Do not spend CUDA on raw stream deletion or non-grammar-preserving bit
   edits; PR103 hidden-gem deletion already showed that path can collapse under
   exact CUDA.
4. Use the current `-16B` clean-runtime packet as the PR103 rate-only anchor for
   further arithmetic/packing work unless a later exact source-shell-contract
   CUDA pair beats it.
