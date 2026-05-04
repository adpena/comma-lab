# PR91 HPM1 Residual Lowering And Submission Gate

Recorded: 2026-05-04T07:03:20Z

This is a progress ledger, not a score claim.

## Current A++ Anchor

- Exact T4 archive: `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/archive.zip`
- Score: `0.25369011029397787`
- Bytes: `229756`
- SHA-256: `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`
- Runtime tree SHA-256: `d195f4ecd0743cfd146efafee6729e96ee5428bfb28bbd0ca87cbad055494440`
- Pre-submission compliance: `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/pre_submission_compliance_check.json`, status `passed`.

## PR91 Source Anatomy

PR91 public source intake contains:

- `inflate.sh`
- `inflate.py`
- `pr86_hpac.py`
- `range_mask_codec.cpp`

`range_mask_codec.cpp` is live only for `QMA6`, `QMA7`, `QMA8`, and `QMA9`
mask payloads through `inflate.py::load_range_mask`. PR91's `HPM1` branch does
not use this C++ decoder; it writes the embedded HPAC model to a temporary
`.pt.ppmd` file and calls the Python/constriction PR86 HPAC decoder.

The public PR91 HPM1 archive is byte-attractive but fail-closed in our canonical
T4/L40S exact path with `hpac_entropy_decode_contract_mismatch`. It is external
motivation and a parity target, not contest-faithful score evidence for us.

## Corrected Residual Signal

The PR85 token-source file is storage-order `N,W,H`, not render-order `N,H,W`.
The replay CLI now exposes `--raw-token-layout qma9_storage_wh_to_render_hw`
and normalizes into render order before HPM1 prototype encoding.

Corrected local-only residual prototype artifacts:

- 1 frame:
  - JSON: `experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr85_tokens_hpm1_residual_render_order_local_reencode_1frame_20260504_codex.json`
  - HPAC token bytes: `404`
  - HPM1 segment bytes including HPAC model: `28695`
  - Elapsed: `60.06s`
- 2 frames:
  - JSON: `experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr85_tokens_hpm1_residual_render_order_local_reencode_2frame_20260504_codex.json`
  - HPAC token bytes: `11912`
  - HPM1 segment bytes including HPAC model: `40203`
  - Elapsed: `122.829s`

Interpretation: the corrected residual/previous-context contract is real, but
the current PR86 HPAC model is not yet a full-video byte win by extrapolation.
Do not dispatch exact eval from this prototype. Next work is prefix profiling,
native lowering, and model adaptation/re-fitting if the prefix curve supports
it.

## Native Lowering

Worker `019df1c1-1e5a-75e1-8f06-540430d57bc2` added
`runtime-rs/crates/hpac-codec` with pinned `constriction = "=0.4.2"`.

The crate proves a Python constriction 0.4.2 f64 `Categorical(...,
perfect=False)` fixture exactly:

- symbols `[0, 4, 1]`
- queue word `[0x43958018]`

It also includes fixed-CDF/adaptive 5-symbol roundtrip helpers and little-endian
`uint32` token byte helpers. This is a native entropy primitive only; PyTorch
HPACMini probability-row generation and HPM1 parsing remain Python-owned.

Verification:

- `cd runtime-rs && cargo test -p hpac-codec` -> 5 passed.
- Python focused tests for PR provenance, pre-submission gate, residual contract
  and raw-token layout -> 10 passed.

## Pre-Submission Gate

Added `scripts/pre_submission_compliance_check.py`.

The gate is provider-agnostic and checks:

- required submission files;
- executable `inflate.sh`;
- ZIP validity, duplicate members, zip-slip/resource-fork names, and local
  header versus central directory name equality;
- exact archive SHA/bytes expectations;
- optional auth-eval JSON custody, CUDA/T4/sample/runtime-tree expectations,
  and exact score recomputation from contribution fields when available;
- optional public hygiene scan through `tac.preflight.check_public_release_hygiene`;
- optional public PR provenance via `--source-prs`.

`.gitignore` now excludes generated `submissions/*/archive.zip`,
`submissions/*/submission.zip`, `submissions/*/inflated/`,
`submissions/*/__pycache__/`, `submission_packets/`, and
`reports/public/submission_packets/`.

## Next Engineering Decision

The fastest contest-faithful path remains:

1. Lower residual-HPM1 prefix profiling so 8/16/32/64-frame curves are minutes,
   not hours.
2. If the curve is sub-PR91 trajectory, build full deterministic HPM1 residual
   candidate and runtime decode path, then run runtime parity before any exact
   eval claim.
3. If the curve is too large, shift HPM1 work to model adaptation/fine-tuning
   or abandon it as a measured implementation while preserving PR91 as an
   external byte target.
4. In parallel, keep the A++ PR85+STBM packet clean through
   `pre_submission_compliance_check.py` and docs/report release hygiene.

## Residual Sufficient-Program Profiling Addendum

Added a local planning profiler for the smaller/preciser representation
question:

- Tool:
  `experiments/profile_pr85_residual_sufficient_program.py`
- Tests:
  `src/tac/tests/test_profile_pr85_residual_sufficient_program.py`
- Real artifact:
  `experiments/results/pr85_residual_sufficient_program_20260504_codex/pr85_residual_sufficient_program_profile.json`
- Matrix integration:
  `experiments/results/pr85_full_stack_opportunity_matrix_20260504_codex_residual_density/pr85_full_stack_opportunity_matrix.json`

The profiler loads the PR85 QMA9 token source in storage order `N,W,H`,
normalizes to render order `N,H,W`, and scores deterministic predictors as
charged sufficient-statistic programs: absolute, previous-frame, left, up, and
previous-frame with left-border preservation. It records residual symbol
entropy, sparse zero-map entropy, row-span atoms, top changed frames, SHA-256
custody, and rate-only lower bounds. It does not build an archive, dispatch
GPU work, or claim score.

Real PR85 result:

- charged QMA9 mask bytes: `159011`
- best deterministic predictor: `left_zero_border`
- zero fraction: `0.9955987633599175`
- nonzero fraction: `0.0044012366400824655`
- best naive sufficient-program lower bound: `721370.0252622243` bytes
- estimated bytes saved versus QMA9: `-562359.0252622243`

Interpretation: the smaller representation is not a naive residual bitmap or
row-span map. The event-location cost dominates despite very high predictor
agreement. The useful signal is the density field: it should drive HPAC/native
learned mask coder training, active-subspace atom selection, and curriculum
weighting, not exact-eval dispatch by itself.

The full-stack matrix now ranks this as
`qma9_residual_sufficient_program_density`: blocked as a direct coder, but
high-stackability as a training/profile field. This keeps the search aggressive
without rerunning a locally refuted direct residual encoding.

Additional HPAC native hardening landed in
`runtime-rs/crates/hpac-codec/src/lib.rs`: a 32-symbol adaptive f64 fixture
generated from the Python constriction queue contract with exact queue words
`[0x197403aa, 0x40f050c1, 0xe60e4cac]`.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_profile_pr85_residual_sufficient_program.py -q`
  -> `3 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_plan_pr85_full_stack_opportunity_matrix.py src/tac/tests/test_profile_pr85_residual_sufficient_program.py -q`
  -> `7 passed`
- `cd runtime-rs && cargo test -p hpac-codec` -> `6 passed`
- `git diff --check` on touched files -> passed

## HPM1 Prefix Trajectory Smoke Addendum

The PR85-to-HPM1 residual prefix trajectory profiler ran on real local PR85
QMA9 tokens for a bounded `1,2` frame prefix. This remains planning-only:
`score_claim=false`, `dispatch_unlocked=false`, no scorer, no GPU dispatch.

Artifacts:

- JSON:
  `experiments/results/pr85_hpm1_residual_prefix_trajectory_20260504_codex/profile_1_2.json`
- Markdown:
  `experiments/results/pr85_hpm1_residual_prefix_trajectory_20260504_codex/profile_1_2.md`

Observed prefix slope:

| frames | raw token bytes | candidate HPM1 segment bytes | marginal segment bytes/frame | elapsed seconds |
| ---: | ---: | ---: | ---: | ---: |
| `1` | `196608` | `28695` | n/a | `59.97` |
| `2` | `393216` | `40203` | `11508.0` | `122.205` |

Interpretation:

- The HPM1 contract remains a high-value mask-coding signal, but the Python
  prefix path is too slow for broad search. It is a calibration profiler and
  parity harness, not the production search loop.
- Native lowering should target residual-symbol generation and HPAC probability
  queue parity first. Any future all-600 search should run through Rust or a
  vectorized native bridge before it is allowed to consume serious wall clock.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_profile_pr85_hpm1_residual_prefix_trajectory.py -q`
  -> `3 passed`
- `.venv/bin/python experiments/profile_pr85_hpm1_residual_prefix_trajectory.py --frame-counts 1,2 --json-out experiments/results/pr85_hpm1_residual_prefix_trajectory_20260504_codex/profile_1_2.json --md-out experiments/results/pr85_hpm1_residual_prefix_trajectory_20260504_codex/profile_1_2.md`
  -> wrote planning-only JSON/Markdown
