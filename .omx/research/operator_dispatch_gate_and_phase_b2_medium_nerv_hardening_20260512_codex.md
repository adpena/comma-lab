# Operator Dispatch Gate + Phase B-2 Medium-NeRV Hardening - 2026-05-12

## Scope

Lane: `lane_operator_dispatch_gate_smoke_before_full_hardening_20260512_codex`

This landing hardens the operator-dispatch boundary before any additional
Modal/Vast/Lightning score-lowering spend:

- `tools/operator_authorize.py` now refuses recipes with
  `dispatch_enabled=false` or `pre_promotion_blockers` before lane claim,
  confirmation, provider setup, or trainer-required-input validation.
- `tools/validate_dispatch_required_inputs.py` accepts explicit
  `--flag-value FLAG=PATH` values from recipes, so local validation uses the
  recipe's local `required_input_files.default_path` instead of accidentally
  reading remote `/workspace/...` env defaults or non-literal trainer manifest
  expressions.
- Catalog #167 smoke-before-full enforcement remains clean at 0 findings.
- HIGH-target SIREN and Cool-Chic now have discoverable smoke-before-full
  operator wrappers.
- MEDIUM-target HNeRV-family variants (`tc_nerv`, `block_nerv`, `ff_nerv`,
  `ds_nerv`, `hi_nerv`) are build/smoke-ready but fail closed until a
  HIGH-target substrate anchor lands or the operator explicitly reroutes them.
- MEDIUM/experimental grayscale-LUT is implemented, trainer-backed, and
  smoke-wrapper backed, but also fails closed behind the same canary/cost-band
  calibration gate.
- The GHA CPU dispatcher now passes the public release asset
  `browser_download_url` into `eval.yml`, not the GitHub API asset `.url`.
- The Modal harvest pass closed the duplicate `remote_lane_substrate_sane_hnerv`
  inner claims as `failed_modal_training_rc_1`. This is a trainer/scorer
  preprocessing failure class, not an empirical substrate score result.
- CNeRV's remote driver landed from partner work and is syntactically valid,
  but the recipe remains fail-closed because the trainer still lacks the
  Catalog #151 TIER manifest and refuses non-smoke training.

No GPU dispatch was launched. No score claim is made.

## Validation

Commands run from `/Users/adpena/Projects/pact`:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m ruff check \
  tools/operator_authorize.py tools/validate_dispatch_required_inputs.py \
  tools/run_modal_smoke_before_full.py tools/claim_catalog_number.py \
  experiments/train_substrate_{block_nerv,ds_nerv,ff_nerv,hi_nerv,tc_nerv}.py \
  src/tac/tests/test_operator_authorize_dispatch_gates.py \
  src/tac/tests/test_check_152_operator_wrapper_validates_required_inputs.py \
  src/tac/tests/test_check_162_operator_authorize_canonical_use.py \
  src/tac/tests/test_check_167_smoke_before_full_pattern.py
```

Result: `All checks passed`.

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_operator_authorize_dispatch_gates.py \
  src/tac/tests/test_check_152_operator_wrapper_validates_required_inputs.py \
  src/tac/tests/test_check_162_operator_authorize_canonical_use.py \
  src/tac/tests/test_check_167_smoke_before_full_pattern.py \
  src/tac/tests/test_claim_catalog_number_atomic.py -q
```

Result: `84 passed in 6.27s`.

Additional focused checks after the NVIDIA/Fields-medal review returned:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_dispatch_cpu_eval_via_github_actions.py \
  src/tac/tests/test_operator_authorize_dispatch_gates.py -q
```

Result: `29 passed in 0.24s`.

```bash
.venv/bin/python tools/operator_authorize.py \
  --recipe substrate_grayscale_lut_modal_a100_dispatch --dry-run
```

Result: refused before validation/claim/dispatch with
`dispatch_enabled=false` and blockers
`high_target_substrate_anchor_required`,
`modal_cost_band_success_anchor_required`, and `modal_smoke_not_yet_green`.

```bash
.venv/bin/python tools/operator_authorize.py \
  --recipe scpp_stage1_modal_a100_dispatch --dry-run
```

Result: refused before validation/claim/dispatch because the recipe still
declares pre-promotion blockers and does not yet represent a byte-closed exact
eval packet.

```bash
.venv/bin/python tools/harvest_modal_calls.py
```

Result: sane_hnerv duplicate inner Modal claims
`substrate_sane_hnerv_modal_a100_dispatch_20260512T202035Z` and
`substrate_sane_hnerv_modal_a100_dispatch_20260512T202633Z` harvested/closed
as `rc=1`. No exact score artifact was produced.

```bash
bash -n scripts/remote_lane_substrate_cnerv.sh
.venv/bin/python tools/operator_authorize.py \
  --recipe substrate_cnerv_modal_a100_dispatch --dry-run
```

Result: shell syntax clean; recipe refused before validation/claim/dispatch
with blockers `tier_manifest_required_for_catalog_151_compliance` and
`trainer_scaffold_only_non_smoke_systemexit_gate`.

```bash
env GITHUB_ACTIONS=true PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 \
  PACT_PREFLIGHT_PARALLEL_WORKERS=8 PYTHONPATH=src:upstream:$PWD \
  time .venv/bin/python -m tac.preflight
```

Result: `PREFLIGHT PASSED` in `10.15 real`, under the 30 second operator
wall-clock budget.

Final focused gate after CNeRV registration and review-fix cleanup:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_operator_authorize_dispatch_gates.py \
  src/tac/tests/test_check_152_operator_wrapper_validates_required_inputs.py \
  src/tac/tests/test_check_162_operator_authorize_canonical_use.py \
  src/tac/tests/test_check_167_smoke_before_full_pattern.py \
  src/tac/tests/test_check_168_ast_walker_handles_assign_and_annassign.py \
  src/tac/tests/test_dispatch_cpu_eval_via_github_actions.py \
  src/tac/tests/test_qma9_range_mask_byte_search.py \
  src/tac/tests/test_claim_catalog_number_atomic.py -q
```

Result: `135 passed in 13.77s`.

```bash
env GITHUB_ACTIONS=true PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 \
  PACT_PREFLIGHT_PARALLEL_WORKERS=8 PYTHONPATH=src:upstream:$PWD \
  /usr/bin/time -p .venv/bin/python -m tac.preflight
```

Result: `PREFLIGHT PASSED`, `real 10.46`; final rerun after Catalog #168
test cleanup: `real 9.89`.

Known broad-test residual from this pass: adding a broader PR82/Henosis test
subset surfaced historical fixture/rehydration blockers unrelated to this
dispatch-gate landing:

- `test_actual_pr82_intake_profile_pins_static_contract`: missing
  `experiments/results/public_pr82_henosis_frontier_intake_20260503_codex/replay_submission/inflate.py`.
- two `test_build_pr84_pr82_henosis_stack_candidate` synthetic cases:
  `tac.henosis_pr82_transfer.parse_replay_contract` still raises the
  documented `rehydration incomplete` error.

These are not score claims and do not authorize/retire PR82/Henosis work; they
remain implementation blockers for that lane.

Local CPU smoke checks also passed for `sane_hnerv`, `balle_renderer`,
`siren`, `cool_chic`, `vq_vae`, `self_compress_nn`,
`hybrid_renderer_residual`, `block_nerv`, `tc_nerv`, `ds_nerv`, `ff_nerv`,
and `hi_nerv`. These were `--smoke` checks only and are not score evidence.

## Current Score-Lowering Queue

Highest priority remains byte-closed exact-eval work:

1. Harvest/close any active Modal calls with canonical recovery before new
   spend.
2. Trigger/harvest `lane_g_v3` [contest-CPU] closure if workflow capacity is
   available.
3. Use smoke-before-full for the next provider dispatch. First dispatches
   should target HIGH-substrate candidates with plausible sub-PR101 upside:
   SIREN, Cool-Chic, VQ-VAE, self-compress, Ballé, then hybrid only after its
   base anchors land.
4. Keep MEDIUM HNeRV variants and grayscale-LUT deferred until the HIGH-target
   anchor queue provides empirical routing signal.
5. Before relaunching sane_hnerv, fix the scorer preprocessing path that
   previously passed 5D RGB directly to SegNet and semantically bypassed
   PoseNet preprocessing.

## Evidence Discipline

- `[contest-CPU]`, `[contest-CUDA]`, `[macOS-CPU advisory]`, and smoke/proxy
  remain separate axes.
- Disabled recipes are explicit routing state, not method negatives.
- Smoke success only proves local/trainer path health; it does not promote a
  lane or authorize a score claim.
- Any future full dispatch must use lane claim + provider detach + canonical
  harvest and must append terminal claim status.

## 6-Hook Wire-In

This is dispatch-infrastructure hardening, not a new empirical substrate
anchor:

1. Sensitivity-map: N/A - no new saliency data.
2. Pareto constraint: N/A - provider gating only.
3. Bit-allocator: N/A - no archive allocation change.
4. Cathedral autopilot dispatch: wired through recipe-level fail-closed
   dispatchability and smoke-before-full wrappers.
5. Continual-learning posterior: N/A until a real empirical anchor is
   harvested.
6. Probe-disambiguator: N/A - single correctness interpretation.

## Follow-up: lane_g_v3 GHA CPU Retry3

- Dispatch claim:
  `lane_g_v3_gha_cpu_eval_l3_promotion_20260512_retry3`.
- GHA run: `25771036919`.
- Classification: `failed_gha_cpu_eval_toolchain_pre_score`. This is an
  infrastructure/runtime bootstrap failure, not a score result and not a lane
  or model negative.
- Root cause: the GHA workflow still hard-required the robust-current ffmpeg
  color-contract gate during setup, even for `PYTHON_INFLATE=renderer`
  archives that do not use the ffmpeg decode branch. BtbN `latest` also no
  longer satisfied the historic `in_primaries` scale-option assumption, so the
  setup step failed before archive custody and auth eval began.
- Fix landed locally for the next retry: `scripts/ensure_parity_ffmpeg.sh`
  supports non-strict bootstrap for renderer-only workflows, and
  `submissions/robust_current/inflate.sh` now applies the ffmpeg color-contract
  gate only to runtime modes that can actually reach the ffmpeg decode path.
  Renderer modes still require `uv` and fail closed later if renderer
  dependencies are missing.

## Follow-up: lane_g_v3 GHA CPU Retry4

- GHA run: `25771219590`.
- Classification: `failed_gha_cpu_eval_toolchain_stdout_contamination`. This
  is also a pre-score infrastructure failure.
- Root cause: the reusable `ensure_parity_ffmpeg.sh` printed curl progress from
  the BtbN download to stdout while the workflow captured stdout as
  `SELECTED_FFMPEG`. The selected binary path became a multi-line progress log
  plus the real path, so the workflow failed with `File name too long`.
- Fix: BtbN download progress now goes to stderr; stdout is reserved for the
  single selected ffmpeg path. The focused regression test asserts this stdout
  purity rule.

## Follow-up: lane_g_v3 GHA CPU Retry5

- GHA run: `25771307665`.
- Classification: `failed_gha_cpu_eval_missing_upstream_checkout`. This is a
  pre-score infrastructure failure.
- Progress: retry5 passed parity ffmpeg bootstrap, CPU torch install, and
  committed archive custody before failing.
- Root cause: `upstream/` is intentionally gitignored in this repo, but the
  workflow still assumed `upstream/evaluate.py` was present after checkout.
- Fix: the workflow now clones the official comma challenge repository and
  checks out pinned upstream commit
  `11ad728f563d8970929e8947a1cf6124ee6303e4` before verifying scorer inputs.
