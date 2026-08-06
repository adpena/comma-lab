# ddm_us2 Next If Resumed

Mode constraint: still `$0` unless explicitly escalated. Do not run a scorer,
do not dispatch, do not mutate `upstream/`, and do not touch live run dirs.

## Folded

1. Dynamic denominator hazard is folded into existing guards.
   - Existing consumers: `src/tac.contest_score.verify_upstream_videos_clean`
     and `experiments/contest_auth_eval._validate_uncompressed_dir`.
   - Resume action: none unless a future rate path bypasses those guards.

2. Partial raw / short iterator hazard is folded into existing guards.
   - Existing consumers: `experiments/contest_auth_eval` strict raw byte-count
     validation and `n_samples == 600` report parser check.
   - Resume action: none unless a row is produced outside `contest_auth_eval`.

3. Shared `D` / blind geometry exploit is folded into #401/m86/bp2/sg2.
   - Three-way rule: generic operator properties are free, but only
     `COUNTED_PAYLOAD_RATE_CREDIT` can be consumed as rate.
   - Resume action: when pricing a candidate, require actual counted payload
     bytes removed; do not price nullity/blind area alone.

4. Contest venv free-import surface is folded into UA2/#214/e4.
   - Three-way rule: libraries/tools are free, extra declared deps are an
     economic time-budget trade, and large artifacts are counted.
   - Resume action: consume the UA2 residual-budget table before arguing about
     dependency/runtime feasibility.

## Queued With Fire Order

1. `us2_auth_eval_env_purity_patch`
   - Patch target: `experiments/contest_auth_eval.py`.
   - Problem: `_run_upstream_evaluate` calls `sys.executable`, so root-venv
     invocations run upstream source with root-lab dependency versions.
   - Fire order:
     1. Add an optional `--upstream-python` or authority-python resolver that
        prefers `upstream/.venv/bin/python` when present.
     2. Record evaluator Python path and package versions in the JSON result.
     3. Label root-venv runs advisory unless exact-lock parity is proven.
     4. Add unit tests around command construction and result metadata using a
        fake upstream python path; no scorer run required.

2. `us2_report_precision_label_patch`
   - Patch target: `experiments/contest_auth_eval.py::_parse_report`.
   - Problem: report components are 8-decimal strings and final score is
     2-decimal; `canonical_score` is recomputed from rounded report values.
   - Fire order:
     1. Rename or supplement fields to state
        `score_recomputed_from_report_8dp_components`.
     2. Add a conservative `score_precision_bound_S` derived from 8-decimal
        component rounding and exact archive bytes.
     3. Keep final-score cross-check tolerance, but stop implying the JSON
        reconstructs upstream's unrounded internal `.item()` values.
     4. Add synthetic-report parser tests; no scorer run required.

3. `us2_runtime_forbidden_artifact_scan`
   - Patch targets: `experiments/contest_auth_eval.py` runtime dependency scan
     and/or `src/tac/submission_packet/linter.py`.
   - Problem: `check_no_scorer_load_at_inflate` scans `submissions/*/inflate*`
     patterns, but actual `contest_auth_eval.py --inflate-sh` can point at a
     different runtime root.
   - Fire order:
     1. Scan the actual `inflate_sh` and declared runtime dependency roots.
     2. Refuse decode-time references to `upstream/models/*.safetensors`,
        `models/posenet.safetensors`, `models/segnet.safetensors`,
        `upstream/videos/0.mkv`, or `videos/0.mkv`.
     3. Permit compression-time source/scorer use only in encoder tools, never
        in inflate/runtime paths.
     4. Add fixture tests with both forbidden and allowed paths; no scorer run.

4. `us2_raw_size_comment_fix`
   - Patch target: `experiments/contest_auth_eval.py` comments near the raw
     byte-count guard.
   - Problem: comment names `3,663,237,120`; correct
     `1164*874*1200*3` is `3,662,409,600`. Code arithmetic is already correct.
   - Fire order:
     1. Comment-only correction.
     2. Add or reuse a unit assertion for `1164*874*1200*3`.
     3. Serializer commit with the exact post-edit sha.

5. `us2_ci_duration_measure`
   - Patch target: none; measurement only, operator-approved dispatch.
   - Problem: UA2 still carries estimated CI setup durations.
   - Fire order:
     1. On a fork, fire the existing `eval.yml` with `submission_name=baseline`
        on `ubuntu-latest`.
     2. Harvest per-step durations from the Actions log.
     3. Record as `$0`/GitHub-free if no paid runner is used.
     4. Do not infer CUDA/T4 cache behavior from CPU-axis data.

## Resume First Commands

Use these read-only checks before any edit:

```bash
git status --porcelain=v1
git -C upstream status --porcelain=v1
shasum -a 256 .omx/research/ddm_us2_20260805/RECEIPT.md .omx/research/ddm_us2_20260805/NEXT_IF_RESUMED.md
```

If editing code, re-read `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, the common
contract, and the target code section, then use the serializer with post-edit
`--expected-content-sha256`.
