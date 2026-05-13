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

## Follow-up: provider-gate repair + lane_g_v3 Retry6 (2026-05-13)

- Commit `46fd39f8` keeps `tools/operator_authorize.py --dry-run` plan-only:
  it now prints the cost/refusal banner and returns before required-input
  validation, confirmation, lane claim, or provider dispatch. This prevents
  missing local optional artifacts from masquerading as a dispatch outcome.
- The same commit moves required-input validation after explicit operator
  confirmation but still before native provider preflight, lane claim creation,
  or any GPU-metered work.
- Focused verification:
  - `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_operator_authorize_canonical_tool.py src/tac/tests/test_operator_authorize_scripts.py src/tac/tests/test_operator_authorize_dispatch_gates.py -q`
    -> `118 passed`.
  - `.venv/bin/ruff check tools/operator_authorize.py src/tac/tests/test_operator_authorize_scripts.py`
    -> pass.
  - `GITHUB_ACTIONS=true PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 PACT_PREFLIGHT_PARALLEL_WORKERS=8 PYTHONPATH=src:upstream:$PWD /usr/bin/time -p .venv/bin/python -m tac.preflight`
    -> `PREFLIGHT PASSED`, `real 9.53`.
- Modal live-state check:
  - `.venv/bin/modal app list --json` showed no running Modal tasks
    (`Tasks: "0"` on listed apps).
  - `.venv/bin/python tools/claim_lane_dispatch.py summary --ttl-hours 24`
    showed no active Modal claim. The only active claim is the lane_g_v3 GHA
    retry6 CPU eval.
- Retry6 dispatch:
  - lane id:
    `lane_g_v3_gha_cpu_eval_l3_promotion_20260512_retry6`
  - GHA run: `25771550637`
  - URL:
    `https://github.com/adpena/comma-lab/actions/runs/25771550637`
  - Head SHA: `8c9a5e7f1c94bc7d5a2c45b03ce77d0e48db0bf2`
  - Current status at 2026-05-13T01:13Z: still inside
    `Run [contest-CPU] auth eval (canonical contest_auth_eval.py)`.
  - Setup progress: pinned-upstream scorer checkout and upstream/input
    verification both passed, so retry5's missing-upstream blocker is cleared.
  - Score status: `score_claim=false`; no score or promotion claim until the
    uploaded artifact is harvested and component fields are recomputed.

## Follow-up: A1 / PR106 CPU-CUDA axis refresh (2026-05-13)

The canonical non-promoting analyzer was run on A1 and PR106 exact-pair rows:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/analyze_cpu_cuda_eval_drift.py \
  --exact-pair CPU_JSON CUDA_JSON \
  --json-out .omx/research/artifacts/cpu_cuda_pair_refresh_20260513/<name>_analysis.json \
  --markdown-out .omx/research/artifacts/cpu_cuda_pair_refresh_20260513/<name>_analysis.md
```

Artifacts:

- `.omx/research/artifacts/cpu_cuda_pair_refresh_20260513/a1_analysis.json`
- `.omx/research/artifacts/cpu_cuda_pair_refresh_20260513/a1_analysis.md`
- `.omx/research/artifacts/cpu_cuda_pair_refresh_20260513/pr106_r2_analysis.json`
- `.omx/research/artifacts/cpu_cuda_pair_refresh_20260513/pr106_r2_analysis.md`
- `.omx/research/artifacts/cpu_cuda_pair_refresh_20260513/pr106_r2_pr101_grammar_analysis.json`
- `.omx/research/artifacts/cpu_cuda_pair_refresh_20260513/pr106_r2_pr101_grammar_analysis.md`

Adversarial verdict:

- A1 same-archive CPU/CUDA scores remain individually useful axis evidence, but
  the pair is **not** valid for full mechanism attribution because the CPU JSON
  lacks `runtime_tree_sha256` and both axes lack raw-output aggregate hashes.
  The analyzer classifies it as `custody_incomplete`, with
  `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`.
- PR106 R2 and PR106 R2 + PR101 grammar are same-archive pairs, but the runtime
  tree hashes and inflated raw-output aggregate hashes differ across CPU/CUDA.
  The analyzer classifies both as
  `different_raw_outputs_runtime_or_inflate_drift`, not scorer-only drift.
- This reinforces the current apples-to-apples rule: do not convert CPU to CUDA
  or CUDA to CPU, and do not generalize "CPU better" or "CUDA better" from one
  archive family. A1 is CPU-better on its recorded axis; PR106 R2 is
  CUDA-better on its recorded axis; the mechanism evidence still depends on
  runtime/raw-output custody.

## Follow-up: lane_g_v3 Retry6 terminal classification + CPU torch fix (2026-05-13)

- Retry6 GHA run `25771550637` reached real inflate and generated
  `1120/1200` frames, then failed with:
  `OSError: [Errno 28] No space left on device` while writing `frame_out` to
  the raw output file.
- Terminal claim row appended:
  `failed_gha_cpu_eval_runner_disk_full`.
- Artifact harvest:
  `experiments/results/gha_cpu_eval/lane_g_v3_retry6_25771550637/contest_cpu_eval-lane_g_v3-25771550637/eval_work/provenance.json`.
  No `contest_auth_eval.json` was emitted, so this is **not** a score result.
- Root cause:
  `.github/workflows/contest_cpu_eval.yml` installed CPU torch in the outer
  repo venv, but `submissions/robust_current/inflate.sh` creates its own
  `uv run` environment. That inner environment saw `INFLATE_TORCH_SPEC=torch==2.5.1`
  and resolved PyPI CUDA wheels, consuming several extra GB before the
  3.66GB raw file was complete.
- Fix:
  the workflow now sets `INFLATE_TORCH_SPEC=torch==2.5.1+cpu` plus
  `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu`, so both the outer
  workflow venv and the inner inflate `uv run` env resolve the CPU wheel.
- Guard:
  `src/tac/tests/test_contest_cpu_eval_workflow_ffmpeg_bootstrap.py` now
  asserts the workflow pins CPU torch for both the runner and inflate env and
  does not contain a CUDA PyTorch wheel index.
- Tooling correction:
  `tools/analyze_cpu_cuda_eval_drift.py` now reports
  `valid_individual_axis_scores` and `valid_same_archive_axis_score_pair`
  separately from stricter same-runtime/mechanism readiness. This preserves
  A1/PR106 axis evidence without weakening the mechanism/promotion block.
- Verification:
  - `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_contest_cpu_eval_workflow_ffmpeg_bootstrap.py src/tac/tests/test_analyze_cpu_cuda_eval_drift.py -q`
    -> `10 passed`.
  - `.venv/bin/ruff check tools/analyze_cpu_cuda_eval_drift.py src/tac/tests/test_analyze_cpu_cuda_eval_drift.py src/tac/tests/test_contest_cpu_eval_workflow_ffmpeg_bootstrap.py`
    -> pass.

Next exact action: retry lane_g_v3 GHA CPU eval from the CPU-torch-fixed
workflow commit. This should spend the same free GHA runner minutes and remove
the CUDA-wheel disk pressure from the prior attempt.

## Follow-up: lane_g_v3 Retry7 launched + PacketIR proof hardening (2026-05-13)

- Retry7 GHA run:
  `https://github.com/adpena/comma-lab/actions/runs/25772267506`
- Dispatch claim:
  `lane_g_v3_gha_cpu_eval_l3_promotion_20260512_retry7` /
  `gha_run_25772267506`.
- HEAD:
  `7b8f7db0e9620b45ce9ab36f345f377a46570593`.
- Status at launch review:
  setup passed through the inner `uv` environment and reached
  `Run [contest-CPU] auth eval (canonical contest_auth_eval.py)`. That
  confirms the CPU torch wheel fix cleared the prior disk-pressure setup
  failure. No score claim exists until the uploaded artifact is harvested.

Parallel hardening landed while the auth eval runs:

- `tac.packet_compiler.deterministic_compiler.compile_packet(..., mode="optimize")`
  now refuses bare boolean `runtime_consumption_proof=True`. Optimize mode
  requires a typed proof mapping or JSON path tied to candidate archive SHA,
  runtime content SHA, and consumed byte/section evidence.
- Oracle inspection failures now become explicit
  `packet_oracle_inspect_failed:<reason>` blockers instead of silently
  producing an empty-manifest runtime hash.
- PR106 PacketIR archive helpers now auto-detect both known single-member
  packet names, `0.bin` and `x`, while still preserving explicit
  `expected_member_name` fail-closed behavior. This keeps public HNeRV
  replay/repack artifacts on the same canonical parser surface instead of
  creating `0.bin`-only one-offs.
- Verification:
  - `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py src/tac/tests/test_deterministic_compiler.py -q`
    -> `49 passed`.
  - `.venv/bin/ruff check src/tac/packet_compiler/deterministic_compiler.py src/tac/packet_compiler/pr106_sidecar_packet.py src/tac/packet_compiler/__init__.py src/tac/tests/test_deterministic_compiler.py src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py tools/build_deterministic_packet.py`
    -> pass.
  - `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_materialize_residual_pr106_sidecars.py -q`
    -> `48 passed`.
  - `GITHUB_ACTIONS=true PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 PACT_PREFLIGHT_PARALLEL_WORKERS=8 PYTHONPATH=src:upstream:$PWD /usr/bin/time -p .venv/bin/python -m tac.preflight`
    -> `PREFLIGHT PASSED`, `real 9.53`.

Score-lowering implication:

- PacketIR/sidecar byte transforms can still be pursued aggressively, but the
  compiler no longer lets a candidate cross from byte-changing planning into
  exact-eval staging on a boolean assertion. The shortest safe next PacketIR
  path remains: identity parse/re-emit -> typed runtime-consumption proof ->
  same-runtime/full-frame parity or exact auth eval -> score claim only from
  harvested `[contest-CUDA]` / `[contest-CPU]` artifacts.

## Follow-up: PR106 runtime proof member-name parity (2026-05-13)

The PacketIR parser accepted both known public HNeRV single-member archive
names (`0.bin` and `x`) after commit `4e960311`, but the runtime-consumption
proof helpers still defaulted to `expected_member_name="0.bin"`. That was a
harness bias: it preserved safety for PR106-style packets but forced public
`x`-member artifacts onto one-off command flags before the same-runtime proof
could run.

Code hardening:

- `prove_pr106_sidecar_runtime_decode_consumption(...)` and
  `prove_pr106_same_runtime_full_frame_parity(...)` now default to the same
  member-name autodetection as the canonical PacketIR archive reader.
- Explicit `expected_member_name="0.bin"` / `"x"` still fails closed if the
  archive member does not match.
- Runtime proof manifests now record the consumed member name:
  `archive_member_name` for decode-consumption and `member_name` on both
  source/candidate archive records for same-runtime streaming parity.
- CLI defaults for `tools/prove_pr106_sidecar_runtime_consumption.py` and
  `tools/prove_pr106_same_runtime_frame_parity.py` now auto-detect `0.bin` /
  `x`; operators can still pass `--member-name` for strict matching.

Verification:

- `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py -q`
  -> `8 passed`.
- `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py -q`
  -> `21 passed`.
- `.venv/bin/ruff check src/tac/packet_compiler/pr106_runtime_consumption.py src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py tools/prove_pr106_same_runtime_frame_parity.py tools/prove_pr106_sidecar_runtime_consumption.py`
  -> pass.
- CLI proof smoke:
  `tools/prove_pr106_sidecar_runtime_consumption.py --archive submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip --runtime-dir submissions/pr106_latent_sidecar_r2_pr101_grammar`
  emitted `schema=pr106_sidecar_runtime_decode_consumption_proof_v1`,
  `runtime_sidecar_decode_consumption_claim=true`,
  `runtime_sidecar_apply_consumption_claim=true`, and `score_claim=false`.
  Ignored raw artifact:
  `experiments/results/pr106_r2_pr101_runtime_decode_consumption_default_member_autodetect.json`.

Score-lowering implication:

- Public-HNeRV `x` archives and internal PR106 `0.bin` archives now share the
  same runtime-consumption proof surface. This reduces harness mismatch risk
  before PR106/PR101/PR103 PacketIR transforms enter exact-eval queues, without
  promoting parser-consumption evidence into scorer or contest-axis evidence.

## Follow-up: sane_hnerv Modal failure classification + local test-contract fix (2026-05-13)

Recovered Modal canary artifacts show the latest sane_hnerv A100 attempts are
terminal failures, not active jobs:

- `substrate_sane_hnerv_modal_a100_dispatch_20260512T202035Z`: rc=1 after
  12.87s, terminal claim `failed_modal_training_rc_1`.
- `substrate_sane_hnerv_modal_a100_dispatch_20260512T202633Z`: rc=1 after
  72.03s, terminal claim `failed_modal_training_rc_1`.

The first failure was a stale bootstrap harness path that sourced
`remote_archive_only_eval.sh` without the source-only sentinel and failed on a
missing `/workspace/pact/iter_0/archive.zip`. The second reached training and
failed in the score-aware loss path with a SegNet 5D input:
`[32, 1, 3, 384, 512]`.

Current source status:

- Production `sane_hnerv` loss already delegates to
  `tac.substrates.score_aware_common.score_pair_components(...)`, which stages
  `(B,T=2,C,H,W)` pairs and calls each scorer's `preprocess_input` before
  `forward`.
- The local smoke test had drifted from that contract: its dummy PoseNet
  returned a bare tensor, while upstream PoseNet returns `{"pose": tensor}`.
  The test now mirrors the upstream dict contract so green tests exercise the
  same shape/protocol path that Modal needs.

Verification:

- `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/sane_hnerv/tests/test_score_aware_loss_real_scorer_forward.py src/tac/substrates/sane_hnerv/tests/test_train_substrate_sane_hnerv_full_main.py::test_score_aware_loss_runs_on_dummy_scorers -q`
  -> `7 passed`.
- `.venv/bin/ruff check src/tac/substrates/sane_hnerv/tests/test_train_substrate_sane_hnerv_full_main.py`
  -> pass.

Score-lowering implication:

- The next sane_hnerv canary must be launched from current `main`, after local
  preflight, with the source-only bootstrap sentinel and the canonical scorer
  contract. The two recovered Modal failures remain infrastructure/harness
  failures with `score_claim=false`; they do not falsify sane_hnerv as a
  substrate.

## Follow-up: lane_g_v3 GHA [contest-CPU] closure harvested (2026-05-13)

Run:

- GitHub Actions run: `https://github.com/adpena/comma-lab/actions/runs/25772267506`
- Workflow/job: `contest_cpu_eval` / `contest_cpu_eval`
- Axis: `[contest-CPU GHA Linux x86_64]`
- Artifact path: `experiments/results/gha_cpu_eval/lane_g_v3_retry7_25772267506/contest_cpu_eval-lane_g_v3-25772267506/contest_auth_eval.json`
- Evaluated commit: `7b8f7db0e9620b45ce9ab36f345f377a46570593`
- Archive SHA-256: `9b20bdfca246d8e32cc19da966c84cdae7e34f6b247161d107ec43cb9ef6870b`
- Archive bytes: `694074`
- Samples: `600`
- `avg_segnet_dist`: `0.00400702`
- `avg_posenet_dist`: `0.0030529`
- Recomputed formula:
  - seg term `100 * seg = 0.400702`
  - pose term `sqrt(10 * pose) = 0.17472549899771356`
  - rate term `25 * bytes / 37_545_489 = 0.46215538702931797`
  - total `1.0375828860270315`
- JSON canonical score: `1.0375828860270313`
- JSON final score: `1.04`
- `promotion_eligible=false`

Dispatch claim:

- Closed with terminal status
  `completed_gha_cpu_eval_score_1_037582886` for
  `lane_g_v3_gha_cpu_eval_l3_promotion_20260512_retry7` /
  `gha_run_25772267506`.
- `tools/claim_lane_dispatch.py summary --ttl-hours 24` reports
  `active=0`; remaining nonterminal rows are stale staged Phase 2/3 rows, not
  live provider work.

Classification:

- Legitimate GHA `[contest-CPU]` exact-eval closure for this archive/runtime.
- It is not score-lowering relative to the current HNeRV/PR106 frontier and is
  not promotion-eligible.
- The result is still valuable as an apples-to-apples closure of the previously
  open `lane_g_v3` CPU claim and removes a phantom active dispatch from the
  queue.

## Follow-up: substrate trainer pose-default bug class fixed + strict gate (2026-05-13)

Bug class:

- The score-aware substrate loss classes had been moved to the contest pose
  coefficient (`sqrt(10) * sqrt(d_pose)`), but trainer argparse defaults still
  overrode that with the older operating-point pair:
  `--gamma-pose=1.0` and `--pose-weight-scale=2.71`.
- That made first-anchor training objectives non-apples-to-apples before GPU
  spend and kept the PR106-specific pose multiplier as default rather than as
  an explicit sweep knob.

Fix:

- Updated 14 `experiments/train_substrate_*.py` trainer defaults to:
  `--gamma-pose=math.sqrt(10.0)` and `--pose-weight-scale=1.0`.
- Kept `--pose-weight-scale` as an explicit optional multiplier for future
  operating-point sweeps.
- Added strict preflight check
  `check_substrate_trainer_pose_defaults_match_contest_formula(...)` in
  `src/tac/preflight.py` and wired it into `preflight_all()`.
- Added regression tests in
  `src/tac/tests/test_substrate_trainer_pose_defaults_preflight.py`.

Verification:

- `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_substrate_trainer_pose_defaults_preflight.py src/tac/substrates/sane_hnerv/tests/test_train_substrate_sane_hnerv_full_main.py::test_argparse_smoke_mode_required_flags_minimal src/tac/substrates/balle_renderer/tests/test_train_substrate_balle_renderer_full_main.py::test_argparse_smoke_mode_required_flags_minimal -q`
  -> `5 passed`.
- Direct strict preflight check:
  `check_substrate_trainer_pose_defaults_match_contest_formula(strict=True)`
  -> `[]` with `14 trainer(s) scanned`.

## Follow-up: T1 PR95 parity-profile dispatch gate repaired (2026-05-13)

Bug class:

- The canonical PR95 trainer-parity profile exists at
  `.omx/research/pr95_hnerv_muon_trainer_parity_profile_20260510.json`, and
  `experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py`
  already uses that path as `DEFAULT_PR95_PARITY_PROFILE`.
- The operator-authorize recipe for T1 still pointed wrapper-side required
  input validation at a nonexistent
  `docs/pr95_parity_profile_council_20260512.json`.
- The TIER-1 metadata generator command also used a stale `--output` flag for
  `experiments/profile_pr95_hnerv_muon_intake.py`; the profiler's actual CLI
  uses `--json-out`.

Fix:

- Updated
  `.omx/operator_authorize_recipes/phase1_t1_balle_cheap_config_dispatch.yaml`
  so `--pr95-parity-profile` validates the tracked `.omx/research` profile.
- Updated the trainer's `generator_command` metadata to use
  `--json-out .omx/research/pr95_hnerv_muon_trainer_parity_profile_20260510.json`.
- Added a regression test tying the operator recipe default, trainer default,
  tracked profile existence, and generator command together.
- Removed unused imports from the T1 trainer while touching the file.

Verification:

- `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_phase1_trainer_auth_eval_contract.py::test_phase1_trainer_default_pr95_parity_profile_is_tracked_and_loadable src/tac/tests/test_phase1_trainer_auth_eval_contract.py::test_phase1_operator_recipe_pr95_profile_default_matches_trainer_contract -q`
  -> `2 passed`.
- `PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/validate_dispatch_required_inputs.py --trainer experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py --flag-value=--pr95-parity-profile=.omx/research/pr95_hnerv_muon_trainer_parity_profile_20260510.json`
  -> validated 1 required input.
- `PYTHONPATH=src:upstream:$PWD .venv/bin/python experiments/profile_pr95_hnerv_muon_intake.py --no-write`
  -> emitted `pr95_hnerv_muon_static_intake_profile_v1` with
  `trainer_parity_contract.schema=pr95_hnerv_muon_t1_trainer_parity_v1` and
  `local_trainer_parity_preflight_passed=true`.
- `PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/operator_authorize.py --recipe phase1_t1_balle_cheap_config_dispatch --dry-run`
  -> plan-only banner succeeded with no claim/dispatch.

Score-lowering implication:

- The previous T1 Modal failure classified as `pr95_parity_profile_missing`
  was a stale dispatch-contract bug, not a T1/Ballé method result.
- T1 still should not launch as a full 3000-epoch Modal T4 job without a
  reduced-epoch or cheaper-provider route, but the local required-input gate
  now points at the real PR95 parity evidence.

## Follow-up: PR106/R2 GHA CPU eval archive-custody failure fixed (2026-05-13)

Bug class:

- GHA run `25773515494` for
  `lane_pr106_latent_sidecar_r2_pr101_grammar` failed before scoring at the
  workflow's "Verify committed archive exists" step.
- The canonical packet existed locally at
  `submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip`, but the
  repo-level scratch-artifact ignore rule `submissions/*/archive.zip` kept it
  out of the dispatched commit.
- Classification: `archive_custody_missing_on_dispatched_sha`. This is an
  infrastructure/custody failure, not a method result and not a score claim.

Fix:

- Added a narrow `.gitignore` allowlist for the PR106/R2 PR101-grammar archive.
- Tracked the 186,780-byte archive whose SHA-256 matches
  `archive_manifest.json`:
  `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`.
- Closed the failed GHA dispatch claim with terminal status before relaunch.

Score-lowering implication:

- The next GHA CPU run is the first apples-to-apples public CPU-axis closure
  attempt for this exact PR106/R2 PR101-grammar archive/runtime packet.
- No CPU/CUDA conversion is made here; CUDA remains the existing Modal T4 score
  in the submission artifact, and GHA CPU must be measured directly.

## Follow-up: auth-eval inflate interpreter leak fixed (2026-05-13)

Bug class:

- GHA run `25773673132` for
  `lane_pr106_latent_sidecar_r2_pr101_grammar` passed archive custody and
  failed inside `inflate.sh` before scoring.
- Root cause: `experiments/contest_auth_eval.py` defaulted `PYTHON` to the
  evaluator interpreter, but this packet's `inflate.sh` uses
  `PYTHON_BIN="${PYTHON_BIN:-python}"`. The child runtime therefore executed
  outside the `.venv` dependency closure and missed the hard package
  dependency `brotli`.
- Classification: `runtime_dependency_closure_env_leak`. This is an
  infrastructure/runtime-contract failure, not a method result and not a score
  claim.

Fix:

- Updated canonical auth eval `_run_inflate(...)` to set both `PYTHON` and
  `PYTHON_BIN` to `sys.executable` by default.
- Extended the regression test so future exact-eval launches cannot silently
  run shell-configured inflaters outside the evaluator interpreter.

Score-lowering implication:

- This fix benefits PR106/R2 and any public HNeRV/A1-style packet that uses a
  lane-local `PYTHON_BIN` shell variable.
- The relaunch remains an exact [contest-CPU GHA Linux x86_64] measurement of
  the same committed archive/runtime packet; no CPU/CUDA conversion is made.

Preflight/DX evidence:

- `python -m tac.preflight --scope dev --timings-json reports/preflight_dev_timing_20260513.json`
  passed and wrote a profile artifact.
- The dev-scope hot path finished in about 6.7s wall-clock; slowest checks were
  shared-state writer scan (~5.0s), codebase drift (~4.3s), and MPS fallback
  scan (~3.9s), all parallelized under the 30s DX crash gate.
- A direct all-scope `preflight_all()` intentionally hit the 30s crash gate
  before release-scope checks completed. Classification: release/custody sweep
  requires explicit slow-release override; dev preflight remains the normal
  fast gate for iteration.

## Follow-up: substrate provider fan-out metadata gates backfilled (2026-05-13)

Bug class:

- Release-scope preflight surfaced three provider-safety gaps across Modal
  substrate recipes: missing `min_vram_gb`, missing `video_input_strategy`, and
  missing `canary_status`.
- It also surfaced the mixed-precision speed gap: 14 substrate trainers do not
  yet implement the canonical T1 `--enable-autocast-fp16` pattern.

Fix:

- Added top-level `min_vram_gb: 40`,
  `video_input_strategy: per_dispatch_local_copy`, and canary-order metadata
  to all 22 `substrate_*_modal_a100_dispatch.yaml` recipes.
- Marked `sane_hnerv` and `balle_renderer` as canaries; all other substrate
  Modal A100 recipes are `post_canary_dependent` on `sane_hnerv`.
- Added explicit file-level FP16 autocast waivers to the 14 substrate trainers
  that do not yet implement mixed precision. This preserves truth: the speed
  gap is acknowledged rather than falsely claimed as implemented.

Verification:

- `check_substrate_recipes_declare_min_vram_gb_floor(strict=True)` -> 0
  violations across 22 recipes.
- `check_substrate_recipes_declare_video_input_strategy(strict=True)` -> 0
  violations across 22 recipes.
- `check_substrate_dispatch_honors_canary_first_ordering(strict=True)` -> 0
  violations across 22 recipes.
- `check_substrate_trainers_declare_autocast_fp16_support(strict=True)` -> 0
  violations across 14 trainers.

Score-lowering implication:

- Future substrate dispatches now carry enough provider metadata to refuse
  under-VRAM instances, avoid shared-video contention, and preserve canary-first
  ordering after the sane_hnerv failure history.
- The FP16 waiver list is a precise backport queue: replacing each waiver with
  real canonical autocast support is expected to improve training wall-clock
  without creating false score authority.

## Result: PR106/R2 PR101-grammar GHA CPU closure landed (2026-05-13)

Artifact:

- GHA run: `25773960214`
- Workflow/job: `contest_cpu_eval.yml` / `contest_cpu_eval`
- Head SHA evaluated: `a706f04f601d5f49fdbeaa7a4a5292ba3cc14baa`
- Artifact JSON:
  `experiments/results/gha_cpu_eval/pr106_r2_pr101_grammar_25773960214/contest_cpu_eval-lane_pr106_latent_sidecar_r2_pr101_grammar-25773960214/contest_auth_eval.json`
- Artifact JSON SHA-256:
  `d1b61f37beb5e3dc81c7d1e64fbc6ee4f1a699ac198376f78acd08575debee43`

Custody:

- Archive:
  `submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip`
- Archive bytes: `186780`
- Archive SHA-256:
  `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`
- Runtime tree SHA-256:
  `bc9624c900cd2a171e6a1cab8236fe9068cc3c329da3bebf97c7a240f42fabf1`
- Inflated output aggregate SHA-256:
  `9249df4f58f4824fe40dfc415df7c1ea9ca845a48857999d434da424ec9c4150`

Score:

- Axis: `[contest-CPU GHA Linux x86_64]`
- `avg_segnet_dist`: `0.00063197`
- `avg_posenet_dist`: `0.00016402`
- Rate term input: `186780 / 37545489`
- Recomputed score:
  `100*0.00063197 + sqrt(10*0.00016402) + 25*186780/37545489 =
  0.22806551797550428`
- Display-rounded score: `0.23`
- `score_claim=false`, `promotion_eligible=false`,
  `cpu_leaderboard_reproduction_eligible=true`.

Classification:

- Legitimate GHA CPU-axis closure for the exact committed archive/runtime
  packet.
- Not CUDA, not submission-promotion evidence, and not a CPU->CUDA conversion.
- The result matches the prior Modal CPU advisory value within ~`8.9e-7`, which
  strongly suggests the earlier Modal CPU signal was not a harness artifact,
  but only this GHA run carries the public CPU-axis tag.
