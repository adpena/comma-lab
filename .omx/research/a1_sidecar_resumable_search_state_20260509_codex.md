# A1 sidecar resumable search-state hardening (2026-05-09)

<!-- generated_at: 2026-05-09T23:11:00Z -->
<!-- research_only=true -->

## Scope

Local-only advance for `lane_a1_per_pair_latent_sidecar_resampled`.
No GHA, remote, GPU, or exact-eval dispatch was launched.

Owned code surfaces:

- `tools/build_a1_per_pair_latent_correction_sidecar.py`
- `src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py`

## Change

The A1 sidecar builder now writes deterministic `sidecar_choice_state.json`
state with per-pair `dims`, `delta_idx`, `searched_mask`, coverage, archive
custody, old-sidecar SHA-256, search signal, search device, and encode format.

New local workflow:

```bash
.venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 600 \
  --resume-search-state \
  --max-search-seconds <seconds> \
  --runtime-smoke \
  --output-dir experiments/results/<a1-sidecar-full-run>
```

If the wall-clock guard stops the run, the packet is still materialized for
inspection but remains fail-closed until all 600 pairs are searched.

## Custody Guard

`ready_for_exact_eval_dispatch=true` now additionally requires:

- a manifest-bound sidecar choice-state record;
- choice-state SHA-256 matching the materialized state file;
- choice-state context matching the exact old archive, old sidecar, search
  signal, search device, and encode format;
- full 600-pair coverage for non-smoke readiness;
- re-encoded sidecar bytes from choice state matching the archive-consumed
  sidecar SHA-256 and byte count.
- malformed choice-state payloads fail closed as manifest blockers instead of
  relying on implicit array coercion.

This closes the gap where a future full manifest could claim full search
coverage without auditable per-pair choices.

## Local Dry Run

Artifact:
`experiments/results/a1_sidecar_resumable_codex_20260509T_local`

First command:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 1 \
  --max-search-seconds 1 \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- searched pair `0`;
- elapsed `17.30s` real;
- archive SHA-256
  `d979632acb2602202d1e65d1d13c73276bc97c9c5fc889f3713ef1842e8fb89e`;
- archive bytes `178316`.

Second command:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 2 \
  --resume-search-state \
  --max-search-seconds 1 \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- resumed `sidecar_choice_state.json`;
- skipped already-completed pair `0`;
- searched pair `1`;
- elapsed `17.24s` real;
- archive SHA-256
  `28b16af4af8aa7758929ff90d17c3d4aa471e5f7a91a456713cf33da580ceb3a`;
- archive bytes `178316`;
- choice-state SHA-256
  `14ed4c03c37bf0e12ca0c79b41dfdd8de24d4b36e80142bb541da0640979589f`;
- `runtime_smoke_checked=true`;
- `n_pairs_searched=2`;
- `n_pairs_completed_this_run=1`;
- `n_pairs_skipped_already_completed=1`;
- `full_non_smoke_search=false`;
- `ready_for_exact_eval_dispatch=false`;
- remaining blocker includes `non_full_sidecar_search_not_exact_eval_ready`.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py -q
```

Result: `25 passed in 0.52s`.

Codex integration hardening added one malformed-state regression test:
`26 passed in 0.52s`.

```bash
.venv/bin/python -m py_compile \
  tools/build_a1_per_pair_latent_correction_sidecar.py
```

Result: passed.

## Remaining Blockers

- Full 600-pair CPU proxy search remains slow: the scalar path is still about
  `17s` per pair on this local run.
- Current local artifact has only `2/600` searched pairs and is not a score
  claim.
- Exact-eval readiness still requires full coverage, runtime smoke on the final
  packet, exact-eval dispatcher preflight, fresh lane claim, and terminal claim
  closure before any GHA/remote/GPU/eval work.

## Codex follow-up local chunk

<!-- generated_at: 2026-05-09T23:35:00Z -->
<!-- evidence_grade: local_cpu_proxy_partial; no score claim; no remote dispatch -->

After commit `a94650f5`, Codex ran one bounded local resume chunk to advance the
actual ignored candidate artifact without launching remote/GPU/eval work:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 5 \
  --resume-search-state \
  --max-search-seconds 45 \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- resumed `sidecar_choice_state.json`;
- skipped already-completed pairs `0` and `1`;
- searched pairs `2`, `3`, and `4`;
- elapsed `real 48.65`, `user 114.15`, `sys 2.88`;
- choice-state SHA-256
  `b4fc7c53a88189c90104c2458cae31bdbe81d170101359920453050079babcc7`;
- archive SHA-256
  `e140ad653d865d55b33e88e849c05a885ad02a5af98a79a3c99d3a0cd16eeda4`;
- archive bytes `178316`;
- `runtime_smoke_checked=true`;
- `n_pairs_searched=5`;
- `n_pairs_completed_this_run=3`;
- `n_pairs_skipped_already_completed=2`;
- `full_non_smoke_search=false`;
- `ready_for_exact_eval_dispatch=false`;
- blocker remains `non_full_sidecar_search_not_exact_eval_ready`.

The raw artifact directory remains ignored:
`experiments/results/a1_sidecar_resumable_codex_20260509T_local/`.

## Codex follow-up local chunk 2

<!-- generated_at: 2026-05-09T23:55:00Z -->
<!-- evidence_grade: local_cpu_proxy_partial; no score claim; no remote dispatch -->

Codex resumed the same ignored local artifact for another bounded chunk:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 8 \
  --resume-search-state \
  --max-search-seconds 60 \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- skipped already-completed pairs `0` through `4`;
- searched pairs `5`, `6`, and `7`;
- elapsed `real 47.54`, `user 112.27`, `sys 2.89`;
- choice-state SHA-256
  `54e87167376b1be3b7a7c9d904e5cebeb0966f9a6f9026f90c8c1e1a9a5ecd74`;
- archive SHA-256
  `d051054806799ddd6b0715f53c7ef4d147c72736972b75a9e8c7268ebe9ebdd8`;
- archive bytes `178316`;
- `runtime_smoke_checked=true`;
- `n_pairs_searched=8`;
- `n_pairs_completed_this_run=3`;
- `n_pairs_skipped_already_completed=5`;
- `full_non_smoke_search=false`;
- `ready_for_exact_eval_dispatch=false`;
- blocker remains `non_full_sidecar_search_not_exact_eval_ready`.

## Codex follow-up local chunk 3

<!-- generated_at: 2026-05-10T00:10:00Z -->
<!-- evidence_grade: local_cpu_proxy_partial; no score claim; no remote dispatch -->

Codex resumed the same ignored local artifact for another bounded chunk:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 12 \
  --resume-search-state \
  --max-search-seconds 75 \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- skipped already-completed pairs `0` through `7`;
- searched pairs `8`, `9`, `10`, and `11`;
- elapsed `real 78.62`, `user 166.41`, `sys 4.86`;
- choice-state SHA-256
  `6031ae1ae668c630815c46f28120dd20325bf502f7f59032ef572f57ada4a814`;
- archive SHA-256
  `8480dc4cc36bd3a50f60e76de76040f1ae132d5993751706a7c184952c96d061`;
- archive bytes `178316`;
- `runtime_smoke_checked=true`;
- `n_pairs_searched=12`;
- `n_pairs_completed_this_run=4`;
- `n_pairs_skipped_already_completed=8`;
- `full_non_smoke_search=false`;
- `ready_for_exact_eval_dispatch=false`;
- blocker remains `non_full_sidecar_search_not_exact_eval_ready`.

## Solver Wire-In

- Sensitivity-map contribution: N/A - no authoritative empirical anchor.
- Pareto constraint: N/A - local partial search-state custody only.
- Bit-allocator hook: N/A - no exact component response landed.
- Cathedral autopilot dispatch hook: blocked by
  `ready_for_exact_eval_dispatch=false`.
- Continual-learning posterior update: not run; no authoritative tag.
- Probe-disambiguator: N/A - no competing interpretation; state coverage and
  archive-consumed sidecar bytes either match or fail closed.

## Codex follow-up local chunk 4

<!-- generated_at: 2026-05-10T00:32:00Z -->
<!-- evidence_grade: local_cpu_proxy_partial; no score claim; no remote dispatch -->

After landing the fail-closed candidate-batch profiler in commit `f85c5b7a`,
Codex resumed the same ignored local artifact for another bounded chunk:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 16 \
  --resume-search-state \
  --max-search-seconds 120 \
  --profile-candidate-batches 1 4 \
  --auto-candidate-batch-size \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- skipped already-completed pairs `0` through `11`;
- searched pairs `12`, `13`, `14`, and `15`;
- elapsed `real 83.20`, `user 243.91`, `sys 5.15`;
- profiler selected `candidate_batch_size=4`;
- scalar-reference profile for first searched pair:
  - batch `1`: `elapsed_seconds=16.860764542012475`,
    `best_dim=17`, `best_delta_idx=0`, `best_mse=7160.3056640625`,
    `semantic_match_scalar_reference=true`;
  - batch `4`: `elapsed_seconds=12.80810704198666`,
    `best_dim=17`, `best_delta_idx=0`, `best_mse=7160.3056640625`,
    `semantic_match_scalar_reference=true`;
- choice-state SHA-256
  `770261964435c5e8e5511f353772365555c21b45befb7a9a03c73c0311e2c741`;
- archive SHA-256
  `1581050bf759b26c608962189094eae36a86c1ee53943593cc6a6ec14721d46b`;
- archive bytes `178316`;
- `runtime_smoke_checked=true`;
- `n_pairs_searched=16`;
- `n_pairs_completed_this_run=4`;
- `n_pairs_skipped_already_completed=12`;
- `full_non_smoke_search=false`;
- `ready_for_exact_eval_dispatch=false`.

Dispatch blockers in the manifest remain:

- claim lane before any GHA/remote eval dispatch;
- run exact-eval dispatcher preflight against `submission_dir`;
- record runtime tree SHA and terminal dispatch claim row;
- `non_full_sidecar_search_not_exact_eval_ready`.

## Codex follow-up local chunk 5

<!-- generated_at: 2026-05-10T01:05:00Z -->
<!-- evidence_grade: local_cpu_proxy_partial; no score claim; no remote dispatch -->

Codex resumed the same ignored local artifact for another bounded chunk:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 20 \
  --resume-search-state \
  --max-search-seconds 120 \
  --profile-candidate-batches 1 4 \
  --auto-candidate-batch-size \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- skipped already-completed pairs `0` through `15`;
- searched pairs `16`, `17`, `18`, and `19`;
- elapsed `real 92.68`, `user 225.48`, `sys 6.14`;
- profiler selected `candidate_batch_size=1`;
- fail-closed reason: `non_scalar_batches_semantic_mismatch`;
- scalar-reference profile for first searched pair:
  - batch `1`: `elapsed_seconds=15.847406500019133`,
    `best_dim=18`, `best_delta_idx=0`, `best_mse=7178.86474609375`,
    `semantic_match_scalar_reference=true`;
  - batch `4`: `elapsed_seconds=12.309741540928371`,
    `best_dim=18`, `best_delta_idx=0`, `best_mse=7178.8642578125`,
    `semantic_match_scalar_reference=false`;
- choice-state SHA-256
  `06b6c68cc2b24bc5cfa3390553a783d948c36d200cb057774f168fb7d9826977`;
- archive SHA-256
  `4de28160c6d253c68e29353d32c86185164a0c15d085d77e37027385988882ec`;
- archive bytes `178316`;
- `runtime_smoke_checked=true`;
- `n_pairs_searched=20`;
- `n_pairs_completed_this_run=4`;
- `n_pairs_skipped_already_completed=16`;
- `full_non_smoke_search=false`;
- `ready_for_exact_eval_dispatch=false`.

The batch-profile mismatch is a positive guard result, not a method negative:
the profiler refused the faster batch because `best_mse` differed from the
scalar reference by `0.00048828125`, preserving scalar custody for this chunk.

## Codex follow-up local chunk 6

<!-- generated_at: 2026-05-10T00:38:04Z -->
<!-- evidence_grade: local_cpu_proxy_partial; no score claim; no remote dispatch -->

Codex resumed the same ignored local artifact for another bounded local-only
chunk:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 24 \
  --resume-search-state \
  --max-search-seconds 90 \
  --profile-candidate-batches 1 4 \
  --auto-candidate-batch-size \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- skipped already-completed pairs `0` through `19`;
- searched pairs `20`, `21`, `22`, and `23`;
- elapsed `real 75.19`, `user 223.13`, `sys 4.97`;
- profiler selected `candidate_batch_size=4`;
- scalar-reference profile for first searched pair:
  - batch `1`: `elapsed_seconds=15.244385792058893`,
    `best_dim=16`, `best_delta_idx=0`, `best_mse=7130.49560546875`,
    `semantic_match_scalar_reference=true`;
  - batch `4`: `elapsed_seconds=11.340919040958397`,
    `best_dim=16`, `best_delta_idx=0`, `best_mse=7130.49560546875`,
    `semantic_match_scalar_reference=true`;
- choice-state SHA-256
  `b2c10b3cb95d2090a692604276e1472c3623b7d36351e4bb7cd2e986d588e772`;
- archive SHA-256
  `b3ab792433fd477ac9475b4f7dd3940d089ef41c549b2ad27f414869054aa9e2`;
- archive bytes `178316`;
- runtime tree SHA-256
  `3497c774d94fe202563bccba2af4a5f90925cb8d9b2e982cf4428d0efbea0190`;
- `runtime_smoke_checked=true`;
- `n_pairs_searched=24`;
- `n_pairs_completed_this_run=4`;
- `n_pairs_skipped_already_completed=20`;
- `full_non_smoke_search=false`;
- `ready_for_exact_eval_dispatch=false`.

Dispatch blockers remain:

- claim lane before any GHA/remote eval dispatch;
- run exact-eval dispatcher preflight against `submission_dir`;
- record runtime tree SHA and terminal dispatch claim row;
- `non_full_sidecar_search_not_exact_eval_ready`.

The search remains scalar-custody preserving: batch `4` was used only after
matching the scalar reference exactly on the profiled pair for this chunk.

## Codex follow-up local chunk 7

<!-- generated_at: 2026-05-10T00:58:00Z -->
<!-- evidence_grade: local_cpu_proxy_partial; no score claim; no remote dispatch -->

After a read-only custody review found that batch acceleration is not globally
exactness-preserving, Codex resumed the same ignored local artifact using the
scalar candidate path:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 48 \
  --resume-search-state \
  --max-search-seconds 900 \
  --candidate-batch-size 1 \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- skipped already-completed pairs `0` through `23`;
- searched pairs `24` through `47`;
- elapsed `real 389.31`, `user 930.65`, `sys 21.90`;
- `candidate_batch_size=1`;
- no candidate-batch profiler was used;
- choice-state SHA-256
  `1a56e3527575740e1678d8044105bd43aade4b98f2688672eb7a3d64f92e9079`;
- archive SHA-256
  `f19c42e229f6b2b855c9cd124f4f19a5ffd8360b6ce853ec192f9faf8b85fc1e`;
- archive bytes `178316`;
- runtime tree SHA-256
  `3497c774d94fe202563bccba2af4a5f90925cb8d9b2e982cf4428d0efbea0190`;
- `runtime_smoke_checked=true`;
- `n_pairs_searched=48`;
- `n_pairs_completed_this_run=24`;
- `n_pairs_skipped_already_completed=24`;
- `full_non_smoke_search=false`;
- `ready_for_exact_eval_dispatch=false`.

Dispatch blockers remain unchanged:

- claim lane before any GHA/remote eval dispatch;
- run exact-eval dispatcher preflight against `submission_dir`;
- record runtime tree SHA and terminal dispatch claim row;
- `non_full_sidecar_search_not_exact_eval_ready`.

The next custody-preserving continuation should keep
`--candidate-batch-size 1` unless a future disambiguator proves a faster batch
mode matches scalar choices across more than one lucky profiled pair.

## Codex follow-up local chunk 8

<!-- generated_at: 2026-05-10T01:18:00Z -->
<!-- evidence_grade: local_cpu_proxy_partial; no score claim; no remote dispatch -->

Codex resumed the same ignored local artifact using the scalar candidate path:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 96 \
  --resume-search-state \
  --max-search-seconds 900 \
  --candidate-batch-size 1 \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- skipped already-completed pairs `0` through `47`;
- searched pairs `48` through `95`;
- elapsed `real 774.86`, `user 1830.16`, `sys 45.21`;
- `candidate_batch_size=1`;
- choice-state SHA-256
  `385ba5b1280c24c2908f02bde48266994806c82ccfca3d92297e28decdcb7f9b`;
- archive SHA-256
  `f97373eec890d42a7cdc6d109de2c6d906683c703e30d2425e29bcebec8196e6`;
- archive bytes `178316`;
- runtime tree SHA-256
  `3497c774d94fe202563bccba2af4a5f90925cb8d9b2e982cf4428d0efbea0190`;
- `runtime_smoke_checked=true`;
- `n_pairs_searched=96`;
- `n_pairs_completed_this_run=48`;
- `n_pairs_skipped_already_completed=48`;
- `full_non_smoke_search=false`;
- `ready_for_exact_eval_dispatch=false`.

Dispatch blockers remain unchanged:

- claim lane before any GHA/remote eval dispatch;
- run exact-eval dispatcher preflight against `submission_dir`;
- record runtime tree SHA and terminal dispatch claim row;
- `non_full_sidecar_search_not_exact_eval_ready`.

Custody note from read-only review: during active builder runs,
`sidecar_choice_state.json` can be ahead of the manifest/archive. Treat only the
post-exit manifest-bound state above as finalized.

## Codex follow-up local chunk 9

<!-- generated_at: 2026-05-10T02:25:00Z -->
<!-- evidence_grade: local_cpu_proxy_partial; no score claim; no remote dispatch -->

Codex resumed the same ignored local artifact using the scalar candidate path:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 144 \
  --resume-search-state \
  --max-search-seconds 900 \
  --candidate-batch-size 1 \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- skipped already-completed pairs `0` through `95`;
- searched pairs `96` through `143`;
- elapsed `real 804.27`, `user 1862.40`, `sys 40.40`;
- `candidate_batch_size=1`;
- choice-state SHA-256
  `b0e309862d902934b25d82723314302d92efd2b933bea4f469f0dc383fe7a97c`;
- manifest SHA-256
  `91e778214c365088bdc221f50eb4a08c7a65a1084598c463ffc84b208a919b71`;
- archive SHA-256
  `666aaca900cf2537d35294b315956db272d940a38e1fc392ca67e3fce0fbbb48`;
- archive bytes `178316`;
- runtime tree SHA-256
  `3497c774d94fe202563bccba2af4a5f90925cb8d9b2e982cf4428d0efbea0190`;
- `runtime_smoke_checked=true`;
- `n_pairs_searched=144`;
- `n_pairs_completed_this_run=48`;
- `n_pairs_skipped_already_completed=96`;
- `full_non_smoke_search=false`;
- `ready_for_exact_eval_dispatch=false`.

Dispatch blockers remain unchanged:

- claim lane before any GHA/remote eval dispatch;
- run exact-eval dispatcher preflight against `submission_dir`;
- record runtime tree SHA and terminal dispatch claim row;
- `non_full_sidecar_search_not_exact_eval_ready`.

Current completion: `144/600` pairs. Continue scalar chunks until `600/600`;
do not dispatch partial sidecars.

## Codex follow-up local chunk 10

<!-- generated_at: 2026-05-10T02:50:00Z -->
<!-- evidence_grade: local_cpu_proxy_partial; no score claim; no remote dispatch -->

Codex resumed the same ignored local artifact using the scalar candidate path:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 192 \
  --resume-search-state \
  --max-search-seconds 900 \
  --candidate-batch-size 1 \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- skipped already-completed pairs `0` through `143`;
- searched pairs `144` through `191`;
- elapsed `real 741.46`, `user 1809.51`, `sys 39.02`;
- `candidate_batch_size=1`;
- choice-state SHA-256
  `e8b73c1465a35b320dcd3742ee0344c9ec4053f9257cd82ba63b0244003872e8`;
- manifest SHA-256
  `141a13d55bf02f4c0d3b4808e4af454ecc9c45cfb3371662ecddbd2b9c5a0be8`;
- archive SHA-256
  `aef706d7fea549676c6187d4c173b10c6147eaf81919e8289e602b07c07a1989`;
- archive bytes `178316`;
- runtime tree SHA-256
  `3497c774d94fe202563bccba2af4a5f90925cb8d9b2e982cf4428d0efbea0190`;
- `runtime_smoke_checked=true`;
- `n_pairs_searched=192`;
- `n_pairs_completed_this_run=48`;
- `n_pairs_skipped_already_completed=144`;
- `full_non_smoke_search=false`;
- `ready_for_exact_eval_dispatch=false`.

Dispatch blockers remain unchanged:

- claim lane before any GHA/remote eval dispatch;
- run exact-eval dispatcher preflight against `submission_dir`;
- record runtime tree SHA and terminal dispatch claim row;
- `non_full_sidecar_search_not_exact_eval_ready`.

Current completion: `192/600` pairs. Continue scalar chunks until `600/600`;
do not dispatch partial sidecars.
