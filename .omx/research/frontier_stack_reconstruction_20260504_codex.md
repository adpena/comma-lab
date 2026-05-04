# Frontier Stack Reconstruction Pass - 2026-05-04

Scope: local-only contest-faithful deconstruction/reconstruction of PR85,
PR90, PR91, STBM/QRGB, line-search, renderer, and pose-stack artifacts. No
remote/GPU dispatch was performed. No score claim is made from public PR text,
CPU smoke, local anatomy, or planning output.

Primary artifact:

- profiler:
  `experiments/plan_frontier_stack_reconstruction.py`
- plan JSON:
  `experiments/results/frontier_stack_reconstruction_20260504_codex/frontier_stack_reconstruction_plan.json`
- plan Markdown:
  `experiments/results/frontier_stack_reconstruction_20260504_codex/frontier_stack_reconstruction_plan.md`
- stable digest:
  `012989f61f1aadfb5cd099dc96812a3835781e9b07d50c46c28abeae7d03ccf2`
- scanned:
  `31` archives, `335` exact-eval JSONs, `28` auxiliary JSON artifacts

## Exact Anchor

Current local exact CUDA anchor remains PR85:

- artifact:
  `experiments/results/lightning_batch/exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z/contest_auth_eval.adjudicated.json`
- archive bytes: `236328`
- archive SHA-256:
  `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- score: `0.25806611029397786`
- SegNet: `0.00057185`
- PoseNet: `0.0001894`
- samples: `600`
- device: CUDA, Tesla T4
- evidence: exact CUDA full-sample artifact

Public PR90 and PR91 report text is preserved only as external signal:

- PR90 external recomputed score from PR text:
  `0.278872180165691`, archive bytes `218080`
- PR91 external recomputed score from PR text:
  `0.248794804904161`, archive bytes `222404`
- both remain non-promotable until canonical local exact CUDA replay succeeds

## Archive And Runtime Contracts

Profiler byte accounting found strict single-member ZIP custody for the main
frontier artifacts:

- PR85:
  `archive.zip` bytes `236328`, member `x` bytes `236228`, ZIP overhead `100`,
  runtime contract `pr85_v5_micro_single_member_x_qma9_qh0_qp1_sidechannels`
- PR90:
  `archive.zip` bytes `218080`, member `p` bytes `217980`, ZIP overhead `100`,
  runtime contract `pr90_qrepro_single_member_p_fixed_offsets_stbm_qfq4_pose_qrgb`
- PR91:
  `archive.zip` bytes `222404`, member `x` bytes `222304`, ZIP overhead `100`,
  runtime contract `pr91_hpm1_pr85_v5_single_member_x_hpac_mask`
- PR85 STBM1BR:
  `archive.zip` bytes `229756`, member `x` bytes `229656`, ZIP overhead `100`,
  runtime contract `pr85_v5_with_stbm1br_mask_recode`
- PR85 STBM1BR+QRGB stack:
  `archive.zip` bytes `230038`, member `x` bytes `229938`, ZIP overhead `100`,
  runtime contract `pr85_stbm1br_plus_qrgb_archive_candidate`

Runtime rule: every candidate still needs the canonical
`archive.zip -> inflate.sh -> upstream/evaluate.py` CUDA path before score
promotion. Runtime tree hash and archive SHA must be preserved for every
comparison because fixed-runtime bridge candidates are runtime-custody
comparisons, not pure archive comparisons.

## Ranked Opportunities

1. `recover_pr91_hpm1_mask_contract_on_pr85_runtime`
   - Expected rate impact if components stay PR85-identical:
     `-13924` bytes, `-0.009271420063` score.
   - Component expectation: PR91 public text reports PR85-identical SegNet and
     PoseNet, but our local replay currently fails before score with HPM1
     entropy-model decode assertion.
   - Gate: full local HPM1 decode of `600x384x512` tokens, byte-exact reencode
     or reviewed source-contract explanation, runtime parity smoke, dispatch
     claim, exact CUDA auth eval.
   - Adversarial verdict: highest EV, but derivative HPM1+anything archives are
     invalid until base HPM1 replay is fixed.

2. `exact_eval_pr85_stbm1br_lossless_mask_recode`
   - Expected rate impact if components stay invariant:
     `-6572` bytes, `-0.00437602504` score.
   - Component expectation: local builder reports decoded render-order parity
     and `diff_pixels=0`; exact CUDA still required.
   - Gate: review `stbm1br_preflight.json` and manifest SHA, dispatch claim,
     standalone exact CUDA auth eval, component adjudication against PR85.
   - Adversarial verdict: cleanest byte-only PR85 candidate. It can lower the
     exact local floor if positive, but it does not by itself beat external
     PR91.

3. `pr90_topband_geometry_mask_prior_for_pr85`
   - Expected rate target from anatomy:
     PR90 mask body `152431` bytes vs PR85 QMA9 mask `159011`, `-6580` bytes,
     `-0.004381351912` score if component-neutral.
   - Component expectation: PR90 public report is worse than PR85, so PR90 is a
     prior, not a transplant.
   - Gate: derive a PR85-token geometry policy, prove full-stream decoded-token
     parity or explicit charged residual semantics, add fail-closed runtime
     magic, prove local archive byte win before any dispatch.
   - Adversarial verdict: implement as PR85 geometry profiler/builder only;
     avoid wholesale PR90 runtime transplant.

4. `pr91_hpm1_qrgb_atoms_after_hpm1_replay`
   - Expected rate impact: none today; PR91 QRGB candidates add bytes and need
     component wins.
   - Component expectation: local archives exist, but the PR91 base is invalid
     under our canonical replay.
   - Gate: PR91 HPM1 base exact replay fixed, PR91-specific component response
     or exact eval per atom, dispatch claim.
   - Adversarial verdict: keep as post-HPM1 probes only.

5. `qfq4_model_payload_serializer_probe`
   - Expected rate ceiling:
     PR90 model body `56385` bytes vs PR85 model `57074`, `-689` bytes,
     `-0.000458776819` score if tensor-equivalent.
   - Component expectation: static anatomy only; no tensor equivalence proof.
   - Gate: tensor-equivalent PR85 model serialization proof, byte-positive
     deterministic archive, runtime output parity, exact CUDA auth eval.
   - Adversarial verdict: low priority until HPM1/STBM mask lanes are resolved.

## Exact Negative Guardrails

- `pr85_qrgb_pair_atoms_negative_guardrail`:
  best exact QRGB T4 artifact found by the profiler is still worse than PR85:
  score delta `+0.000007047728`, archive delta `+7` bytes. Treat measured QRGB
  configs as narrow negatives and keep the compiler as negative-signal
  profiling infrastructure.
- `randmulti_deletion_waterfill_negative_guardrail`:
  exact T4 `randmulti_top001` saved `14592` bytes but worsened score by
  `+0.022672355574`; protected randmulti deletion remains a component cliff.
- `line_search_pose_stack_negative_guardrail`:
  best scanned line-search exact artifact is H100 diagnostic/full CUDA but
  worse than PR85 by `+0.057071036788`; reuse only as hard-pair profile signal.
- `pr85_stbm1br_plus_qrgb_randmulti_pair_0192_stack`:
  local archive bytes `230038`, apparent rate win `-6290` bytes, but dispatch
  remains blocked by standalone-positive gates and measured QRGB negatives.

## Next Gates

1. Fix or explain PR91 HPM1 local decode/reencode parity. This is the only
   route that plausibly reaches below the external PR91 reported score when
   later stacked with a separate byte or component win.
2. If a remote exact eval is desired for STBM1BR, first claim a lane with
   `tools/claim_lane_dispatch.py claim ...`, then run standalone exact CUDA
   only for the exact STBM archive SHA.
3. Build the PR90-derived PR85-token geometry profiler/builder only after
   explicitly excluding already-negative generic QMA9 row/alternate screens.
4. Do not dispatch QRGB, randmulti deletion, line-search, or STBM+QRGB stack
   candidates without new evidence that clears the exact-negative guardrails.

## Tests

- `.venv/bin/python -m py_compile experiments/plan_frontier_stack_reconstruction.py`
- `.venv/bin/python -m pytest src/tac/tests/test_plan_frontier_stack_reconstruction.py`

