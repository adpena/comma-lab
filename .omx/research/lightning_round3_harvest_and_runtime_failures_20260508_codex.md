# Lightning round-3 harvest and runtime failure ledger - 2026-05-08

Evidence standard reminder: none of the rows below is a score claim unless a
`contest_auth_eval.json` with numeric score, archive bytes, CUDA device, and
matching archive SHA is harvested. CPU/proxy/MPS/runtime-failure evidence does
not promote, rank, kill, or falsify a method family.

## Private provider hints

Operator-provided Lightning hints were saved to
`.omx/state/lightning_provider_hints.json`, not to AGENTS/CLAUDE/docs/paper or
public release surfaces:

- user: `adpena`
- primary teamspace: `comma-lab`
- primary studio: `lossy-compression-challenge`
- operator-mentioned possible alias: `lossy-video-compression-challenge`
- last-seen SSH target: `s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai`

## Harvested failed jobs

### lossy-coarsening-cuda-20260508T020152Z

Status: `failed_no_auth_eval_json`

Remote artifact source:
`/teamspace/jobs/lossy-coarsening-cuda-20260508t020152z/artifacts/pact/experiments/results/lightning_batch/lossy-coarsening-cuda-20260508T020152Z/`

Local artifact mirror:
`experiments/results/lightning_batch/lossy-coarsening-cuda-20260508T020152Z/`

Failure class: inflate-runtime dependency bug before scoring.

Observed failure:
`ModuleNotFoundError: No module named 'brotli'` in the forked
`submission_dir/inflate.py` before `upstream/evaluate.py` produced any score.

Interpretation: not evidence against `lossy_coarsening_analytical`; not an
exact score; not a family kill. The stale packet used plain Python for an
inflate path that imports brotli. Current workspace code has the fix: generated
`inflate.sh` runs `uv run --with brotli==1.2.0 --with torch==... --with
numpy==...`.

Follow-up already queued by the existing dispatch ledger:
`lossy-coarsening-cuda-20260508T024250Z`, status observed as `running` at
2026-05-08T02:53:25Z.

### arch-shrink-x0-4-lightning-20260508T020205Z

Status: `failed_no_auth_eval_json`

Remote artifact source:
`/teamspace/jobs/arch-shrink-x0-4-lightning-20260508t020205z/artifacts/pact/experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T020205Z/`

Local artifact mirror:
`experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T020205Z/`

Failure class: training/runtime contract bug before scoring.

Observed failure:
`RuntimeError: Q-FAITHFUL forward requires an explicit deployed pose tensor`.

Interpretation: not evidence against arch shrink, Q-FAITHFUL, Quantizr-style
training, or the 88K architecture target. The guard correctly prevented a
zero-pose fallback that would train a different FiLM contract than inflate/eval.
Current workspace code has the fix: the train path builds `forward_kwargs`,
preserves `ego_flow`, and passes `pose=qfaithful_pose` into the model call.

Follow-up already queued by the existing dispatch ledger:
`arch-shrink-x0-4-lightning-20260508T024304Z`, status observed as `running` at
2026-05-08T02:53:25Z.

### lossy-coarsening-cuda-20260508T024250Z

Status: `failed_dependency_runtime_missing_module_before_score`

Remote artifact source:
`/teamspace/jobs/lossy-coarsening-cuda-20260508t024250z/artifacts/pact/experiments/results/lightning_batch/lossy-coarsening-cuda-20260508T024250Z/`

Local artifact mirror:
`experiments/results/lightning_batch/lossy-coarsening-cuda-20260508T024250Z/`

Failure class: evaluator dependency environment clobber before scoring.

Observed failure:
the brotli runtime fix worked and inflate produced `0.raw`, but then
`upstream/evaluate.py` failed with `ModuleNotFoundError: No module named
'timm'`.

Interpretation: not evidence against `lossy_coarsening_analytical`; not an
exact score; not a family kill. The forked `inflate.sh` used `uv run --with`
from inside the repo tree without `--no-project`, allowing uv to discover and
sync/remove the shared project `.venv`. That made the subsequent evaluator
process lose its `timm` dependency. Current workspace code has the fix: the
generated inflate command now uses `uv run --no-project --with ...` so
inflate-time dependencies are isolated from the evaluator environment.

## Relaunched active jobs after hardening

### lossy-coarsening-cuda-20260508T0312-noproject

Status: `completed_score_0.351719` after strict harvest at
2026-05-08T03:30:36Z.

Purpose: exact CUDA auth eval of the same `lossy_coarsening_analytical`
candidate after the `uv run --no-project` hardening.

Local build artifact:
`experiments/results/lossy_coarsening_20260508T030750Z/`

Candidate archive:
`experiments/results/lossy_coarsening_20260508T030750Z/archive.zip`

Build facts:

- archive bytes: 156,404
- archive SHA-256 prefix: `ab8a8a13c70b3d3b`
- decoder section bytes: 140,310
- brotli payload bytes: 140,222
- CPU roundtrip rel_err vs quantized fp32: 0.034811
- max per-tensor rel_err: 0.049733
- generated `submission_dir/inflate.sh` contains `uv run --no-project`

Strict exact CUDA result:

- score: 0.351718793322788
- archive bytes: 156,404
- archive SHA-256:
  `ab8a8a13c70b3d3bbf2ce3d8a81a77691b776a6e0fb1cbe9ce504dc3f59c1b28`
- PoseNet distortion: 0.00037762
- SegNet distortion: 0.00186125
- rate: 0.00416572
- samples: 600
- device: CUDA
- T4 match: true
- runtime tree SHA-256:
  `d55ed9a31ab76a2498fdce98ddb5852544c504ad166fd68df1323f341ca4b3e7`
- auth-eval artifact:
  `experiments/results/lightning_batch/lossy-coarsening-cuda-20260508T0312-noproject/auth_eval_work/contest_auth_eval.json`
- evidence row:
  `reports/cathedral_autopilot_evidence.jsonl` line containing
  `job_name=lossy-coarsening-cuda-20260508T0312-noproject`

Interpretation: exact negative for this measured implementation/config. The
candidate saved bytes but increased SegNet/PoseNet enough to lose badly against
the current exact HNeRV floor. This is not a family kill for all coarsening; it
is a guardrail requiring future coarsening to be score-aware/selective and
component-safe before dispatch.

### arch-shrink-x0-4-lightning-20260508T024304Z

Status: `active_dispatching` / Lightning status `running` at
2026-05-08T03:30:55Z.

Interpretation: still in flight. No local result is promotable or rankable
until the exact CUDA auth-eval artifact lands and passes custody checks.

## Local hardening verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_cathedral_autopilot_proxy_guards.py src/tac/tests/test_cathedral_autopilot.py src/tac/tests/test_preflight_implementation_model_match.py src/tac/tests/test_pr101_a1_cpu_anchor_tools.py src/tac/tests/test_pr101_lossy_proxy_guardrails.py src/tac/tests/test_pr101_lossy_int4_qat_dispatch_contract.py src/tac/tests/test_lossy_coarsening_lightning_tools.py src/tac/tests/test_pr106_archive_substitution_surgery.py`
  - result after adding the two round-3 regression assertions: 67 passed
- Regression assertions added after harvest:
  - lossy generated `inflate.sh` must declare brotli via `uv run --with`
  - lossy generated `inflate.sh` must use `uv run --no-project` so
    inflate-time dependency resolution cannot mutate the evaluator venv
  - Q-FAITHFUL training path must pass `pose` through `forward_kwargs`
- Additional verification after `lossy-coarsening-cuda-20260508T024250Z`
  harvest:
  - `.venv/bin/python -m pytest -q src/tac/tests/test_lossy_coarsening_lightning_tools.py src/tac/tests/test_omega_opt_linear_stack_packet.py src/tac/tests/test_omega_opt_claims.py src/tac/tests/test_omega_opt_anchor_discipline_tool.py src/tac/tests/test_codec_stack_planner.py src/tac/tests/test_preflight_implementation_model_match.py`
    - result: 51 passed
  - `.venv/bin/python -m ruff check experiments/lossy_coarsening_lightning_cuda_test.py experiments/lossy_coarsening_lightning_harvest.py experiments/arch_shrink_x0.4_lightning_harvest.py src/tac/deploy/lightning/round3_harvest.py src/tac/omega_opt_linear_stack_packet.py tools/check_omega_opt_linear_stack_packet.py src/tac/tests/test_lossy_coarsening_lightning_tools.py src/tac/tests/test_omega_opt_linear_stack_packet.py`
    - result: all checks passed
- Adversarial review fixes after Kepler findings:
  - `src/tac/deploy/lightning/round3_harvest.py` now rejects loose numeric
    auth-eval JSON. `[contest-CUDA]` evidence requires finite recomputed
    score components, 600 samples, CUDA/T4 provenance, archive byte/SHA
    closure, and runtime-tree SHA custody.
  - both round-3 harvesters call the strict shared validator before appending
    evidence rows or closing a claim as `completed_score_*`.
  - remote `RESULT_JSON` scraping now reads the structured
    `{auth_eval_dir}/contest_auth_eval.json` artifact instead of parsing a
    nested JSON object from logs.
  - `experiments/arch_shrink_x0.4_lightning_full.py` now exports
    `UV_PROJECT_ENVIRONMENT="$WORKSPACE/{auth_eval_dir}/uv_project_env"`
    before `contest_auth_eval.py`, so `submissions/robust_current/inflate.sh`
    cannot mutate the evaluator `.venv`.
  - Omega-OPT linear-stack packet promotion now requires local archive
    bytes/SHA match, parsed strict CUDA auth-eval JSON, existing runtime and
    inflate artifacts, existing 1:1 anchor artifact, and per-layer runtime
    input/output SHA proof. Dummy path strings no longer unlock promotion.
  - materialized packet:
    `reports/omega_opt_linear_stack_packet_20260508.json`.
  - verification:
    `.venv/bin/python -m pytest -q src/tac/tests/test_omega_opt_linear_stack_packet.py src/tac/tests/test_omega_opt_claims.py src/tac/tests/test_omega_opt_anchor_discipline_tool.py src/tac/tests/test_codec_stack_planner.py src/tac/tests/test_preflight_implementation_model_match.py src/tac/tests/test_lossy_coarsening_lightning_tools.py`
    - result: 55 passed
  - verification:
    `.venv/bin/python tools/check_omega_opt_linear_stack_packet.py --strict --format json reports/omega_opt_linear_stack_packet_20260508.json`
    - result: zero findings, promotion flags false, blockers enumerate exact
      missing anchors.
- `.venv/bin/python -m ruff check src/tac/tests/test_lossy_coarsening_lightning_tools.py`
  - result: all checks passed
- `git diff --check -- src/tac/tests/test_lossy_coarsening_lightning_tools.py .omx/research/lightning_round3_harvest_and_runtime_failures_20260508_codex.md .omx/state/lightning_provider_hints.json`
  - result: clean
- 2026-05-08T03:30Z strict-harvest follow-up:
  - `src/tac/deploy/lightning/round3_harvest.py` now requires the canonical
    `score_recomputed_from_components`, `archive_size_bytes`, and
    `canonical_score_source=score_recomputed_from_components` fields in strict
    mode. Alias fields remain display helpers only.
  - the strict score recompute check now accepts the auth-eval JSON's explicit
    score contribution fields when present, avoiding false custody failures
    from rounded component fields while still bounding every contribution.
  - both harvesters close a terminal claim as
    `failed_invalid_auth_eval_custody` if a present auth-eval JSON fails strict
    validation before evidence emission.
  - terminal claim append failures now raise loudly; the active-jobs state is
    not marked terminal after a failed cross-agent claim append.
  - arch-shrink harvest now requires local `archive.zip` custody and SHA/byte
    match before `[contest-CUDA]` evidence emission.
  - `experiments/contest_auth_eval.py` sets an eval-local
    `UV_PROJECT_ENVIRONMENT` for inflate and no longer recovers score-grade
    measurements from stdout when `report.txt` is missing.
  - `.venv/bin/python -m pytest -q src/tac/tests/test_lossy_coarsening_lightning_tools.py src/tac/tests/test_preflight_implementation_model_match.py`
    - result: 39 passed
  - `.venv/bin/python -m ruff check src/tac/deploy/lightning/round3_harvest.py experiments/lossy_coarsening_lightning_harvest.py experiments/arch_shrink_x0.4_lightning_harvest.py experiments/contest_auth_eval.py src/tac/tests/test_lossy_coarsening_lightning_tools.py`
    - result: all checks passed

## Next tranche

1. Poll and harvest `arch-shrink-x0-4-lightning-20260508T024304Z`.
2. If it fails pre-score again, preserve remote artifacts, classify the
   failure narrowly, add the exact regression test, and relaunch only after the
   dispatch ledger contains a terminal row for the failed job.
3. If it produces `contest_auth_eval.json`, recompute the score from
   components, verify archive bytes/SHA/runtime-tree custody, and only then
   emit a promoted evidence row.
4. Build the monolithic HNeRV packet composer/export path next. The current
   frontier packets are parser-section monoliths, not independent ZIP member
   budgets; stack and ADMM work should mutate parser-proven sections with
   old/new SHA and charged-byte proof before exact dispatch.
