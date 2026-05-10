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

## Codex follow-up local chunk 11

<!-- generated_at: 2026-05-10T03:58:00Z -->
<!-- evidence_grade: local_cpu_proxy_partial; no score claim; no remote dispatch -->

Codex resumed the same ignored local artifact using the scalar candidate path:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 240 \
  --resume-search-state \
  --max-search-seconds 900 \
  --candidate-batch-size 1 \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- skipped already-completed pairs `0` through `191`;
- searched pairs `192` through `239`;
- elapsed `real 757.16`, `user 1799.26`, `sys 44.23`;
- `candidate_batch_size=1`;
- sidecar encoded bytes `661` (old preserved sidecar was `607`);
- choice-state SHA-256
  `3639a2fd20bc86b1ec02072342c95d5e513f9dcdf3d5f503d4fee50116761320`;
- manifest SHA-256
  `ae124f125f9565716f487059b7c145e5e860dd4674d4bb08ecaac97e9c9f500f`;
- archive SHA-256
  `077d8148302f5ace206402f1e011e5559bd4bf87b5d40c13db2d2a6744da7901`;
- archive bytes `178316`;
- runtime tree SHA-256
  `3497c774d94fe202563bccba2af4a5f90925cb8d9b2e982cf4428d0efbea0190`;
- `runtime_smoke_checked=true`;
- `n_pairs_searched=240`;
- `n_pairs_completed_this_run=48`;
- `n_pairs_skipped_already_completed=192`;
- `full_non_smoke_search=false`;
- `ready_for_exact_eval_dispatch=false`.

Dispatch blockers remain unchanged:

- claim lane before any GHA/remote eval dispatch;
- run exact-eval dispatcher preflight against `submission_dir`;
- record runtime tree SHA and terminal dispatch claim row;
- `non_full_sidecar_search_not_exact_eval_ready`.

Current completion: `240/600` pairs. Continue scalar chunks until `600/600`;
do not dispatch partial sidecars. MPS remains advisory-only for sweep/ranking
work and is not auth-eval or promotion evidence.

## Codex follow-up local chunk 12

<!-- generated_at: 2026-05-10T04:15:00Z -->
<!-- evidence_grade: local_cpu_proxy_partial; no score claim; no remote dispatch -->

Codex resumed the same ignored local artifact using the scalar candidate path:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 288 \
  --resume-search-state \
  --max-search-seconds 900 \
  --candidate-batch-size 1 \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- skipped already-completed pairs `0` through `239`;
- searched pairs `240` through `287`;
- elapsed `real 785.52`, `user 1848.40`, `sys 41.02`;
- `candidate_batch_size=1`;
- sidecar encoded bytes `661` (old preserved sidecar was `607`);
- choice-state SHA-256
  `233204fd47df5897526dd142b50b0839e5e79f6faf891a26121583f3e939f1c5`;
- manifest SHA-256
  `a6cddf9f47f313fa3d61b6b863017635e023943ff98e40b2263c19da346f86a3`;
- archive SHA-256
  `e7a4a362d9260de7ca9ee8f0db90102f8bfe3f2d4453c15cf8a55ffd1c70bcdb`;
- archive bytes `178316`;
- runtime tree SHA-256
  `3497c774d94fe202563bccba2af4a5f90925cb8d9b2e982cf4428d0efbea0190`;
- `runtime_smoke_checked=true`;
- `n_pairs_searched=288`;
- `n_pairs_completed_this_run=48`;
- `n_pairs_skipped_already_completed=240`;
- `full_non_smoke_search=false`;
- `ready_for_exact_eval_dispatch=false`.

Dispatch blockers remain unchanged:

- claim lane before any GHA/remote eval dispatch;
- run exact-eval dispatcher preflight against `submission_dir`;
- record runtime tree SHA and terminal dispatch claim row;
- `non_full_sidecar_search_not_exact_eval_ready`.

Current completion: `288/600` pairs. Continue scalar chunks until `600/600`;
do not dispatch partial sidecars.

## Codex follow-up local chunk 13

<!-- generated_at: 2026-05-10T02:26:42Z -->
<!-- evidence_grade: local_cpu_proxy_partial; no score claim; no remote dispatch -->

Codex resumed the same ignored local artifact using the scalar candidate path:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 336 \
  --resume-search-state \
  --max-search-seconds 900 \
  --candidate-batch-size 1 \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- skipped already-completed pairs `0` through `287`;
- searched pairs `288` through `335`;
- elapsed `real 743.54`, `user 1804.03`, `sys 41.90`;
- `candidate_batch_size=1`;
- sidecar encoded bytes `661` (old preserved sidecar was `607`);
- choice-state SHA-256
  `873b252c2c445928632ec870f741991ffce7b05f5022b3f46e4471fbb596ede3`;
- manifest SHA-256
  `8349d20c90f0ebc3c700d0d012562c888013447be3bb868451feac0e3e893a5c`;
- archive SHA-256
  `5e8d6d64f926c02da94cf1a7ba4f67283db8c2d74faf7140e8f2277555c8d989`;
- archive bytes `178316`;
- runtime tree SHA-256
  `3497c774d94fe202563bccba2af4a5f90925cb8d9b2e982cf4428d0efbea0190`;
- `runtime_smoke_checked=true`;
- `n_pairs_searched=336`;
- `n_pairs_completed_this_run=48`;
- `n_pairs_skipped_already_completed=288`;
- `full_non_smoke_search=false`;
- `ready_for_exact_eval_dispatch=false`.

Dispatch blockers remain unchanged:

- claim lane before any GHA/remote eval dispatch;
- run exact-eval dispatcher preflight against `submission_dir`;
- record runtime tree SHA and terminal dispatch claim row;
- `non_full_sidecar_search_not_exact_eval_ready`.

Adversarial review during this chunk found that pre-patch chunk history lacks
machine-readable per-pair scalar-equivalence provenance. The builder now emits
per-pair search records and a single-writer output-dir lock; future exact-eval
readiness must fail closed unless every completed pair has scalar-equivalent
provenance. This chunk is therefore preserved as local proxy progress only.

Current completion: `336/600` pairs. Continue scalar chunks until `600/600`,
then either recheck legacy-unproven pairs with the patched builder or preserve
the candidate as non-dispatchable proxy evidence.

## Codex follow-up local chunk 14

<!-- generated_at: 2026-05-10T02:44:46Z -->
<!-- evidence_grade: local_cpu_proxy_partial; no score claim; no remote dispatch -->

Codex resumed the same ignored local artifact using the patched scalar
candidate path with single-writer lock and per-pair provenance records:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 384 \
  --resume-search-state \
  --max-search-seconds 900 \
  --candidate-batch-size 1 \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- skipped already-completed pairs `0` through `335`;
- searched pairs `336` through `383`;
- elapsed `real 777.74`, `user 1842.30`, `sys 41.17`;
- `candidate_batch_size=1`;
- sidecar encoded bytes `661` (old preserved sidecar was `607`);
- choice-state SHA-256
  `0b7d416c723ac49d0c4f210445861a0ae5df1068f7c67d2827b613b053895b83`;
- manifest SHA-256
  `3a7ee2f406bbe01c9088ad92a0ed564bc73739a34ee081614d47d7762dee10ba`;
- archive SHA-256
  `226ac875dcd784773a03ce879fd361624bca1d8c5e6f952f036fb4e262139f10`;
- archive bytes `178316`;
- runtime tree SHA-256
  `3497c774d94fe202563bccba2af4a5f90925cb8d9b2e982cf4428d0efbea0190`;
- `runtime_smoke_checked=true`;
- `n_pairs_searched=384`;
- `n_pairs_completed_this_run=48`;
- `n_pairs_skipped_already_completed=336`;
- `full_non_smoke_search=false`;
- `ready_for_exact_eval_dispatch=false`;
- sidecar choice-state provenance: `48` machine-recorded scalar-equivalent
  pairs, all from this chunk.

Dispatch blockers now include the new provenance guard:

- claim lane before any GHA/remote eval dispatch;
- run exact-eval dispatcher preflight against `submission_dir`;
- record runtime tree SHA and terminal dispatch claim row;
- `sidecar_pair_search_records_missing_for_completed_pairs:336`;
- `non_full_sidecar_search_not_exact_eval_ready`.

Patched guard check on the terminal manifest returned only:

- `sidecar_pair_search_records_missing_for_completed_pairs:336`.

Current completion: `384/600` pairs. Continue scalar chunks for coverage, but
do not dispatch until legacy searched pairs have per-pair scalar-equivalence
records or are rechecked with `--recheck-unproven-pairs`.

## Codex follow-up local chunk 15

<!-- generated_at: 2026-05-10T03:02:40Z -->
<!-- evidence_grade: local_cpu_proxy_partial; no score claim; no remote dispatch -->

Codex resumed the same ignored local artifact:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 432 \
  --resume-search-state \
  --max-search-seconds 900 \
  --candidate-batch-size 1 \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- skipped already-completed pairs `0` through `383`;
- searched pairs `384` through `431`;
- elapsed `real 830.86`, `user 1891.49`, `sys 42.30`;
- `candidate_batch_size=1`;
- sidecar encoded bytes `661` (old preserved sidecar was `607`);
- choice-state SHA-256
  `44eea97d7ff1533bf4dbb86e464310d19c06f1f1eeee425ec8e584ee444d81e6`;
- manifest SHA-256
  `02006ca0733d5fd3537eacf1a02c0c332897672cbfabcf88bd6a4fa9de9c320f`;
- archive SHA-256
  `9a2df4d31377e95283b9531f9eea2164575e8d52355ac99442f5e75b643a6507`;
- archive bytes `178316`;
- runtime tree SHA-256
  `3497c774d94fe202563bccba2af4a5f90925cb8d9b2e982cf4428d0efbea0190`;
- `runtime_smoke_checked=true`;
- `n_pairs_searched=432`;
- `n_pairs_completed_this_run=48`;
- `n_pairs_skipped_already_completed=384`;
- `full_non_smoke_search=false`;
- `ready_for_exact_eval_dispatch=false`;
- sidecar choice-state provenance: `96` machine-recorded scalar-equivalent
  pairs, all from post-patch chunks.

Dispatch blockers:

- claim lane before any GHA/remote eval dispatch;
- run exact-eval dispatcher preflight against `submission_dir`;
- record runtime tree SHA and terminal dispatch claim row;
- `sidecar_pair_search_records_missing_for_completed_pairs:336`;
- `non_full_sidecar_search_not_exact_eval_ready`.

Patched sidecar choice-state guard on the terminal manifest returned only:

- `sidecar_pair_search_records_missing_for_completed_pairs:336`.

Current completion: `432/600` pairs. Continue scalar chunks for coverage, then
run `--recheck-unproven-pairs` to replace legacy `0..335` provenance before
any exact-eval dispatch.

## Codex follow-up local chunk 16

<!-- generated_at: 2026-05-10T03:20:28Z -->
<!-- evidence_grade: local_cpu_proxy_partial; no score claim; no remote dispatch -->

Codex resumed the same ignored local artifact:

```bash
/usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --n-pairs 480 \
  --resume-search-state \
  --max-search-seconds 900 \
  --candidate-batch-size 1 \
  --runtime-smoke \
  --runtime-smoke-pairs 1 \
  --output-dir experiments/results/a1_sidecar_resumable_codex_20260509T_local
```

Observed:

- skipped already-completed pairs `0` through `431`;
- searched pairs `432` through `479`;
- elapsed `real 799.43`, `user 1861.44`, `sys 42.43`;
- `candidate_batch_size=1`;
- sidecar encoded bytes `661` (old preserved sidecar was `607`);
- choice-state SHA-256
  `9a5e854bbf111f93ec4d0b7fa43704c6928a3b7440f6259ebf22a4ef836b85b2`;
- manifest SHA-256
  `58c4efce30803027e51c935746dbe767bff4e8b1e52ac29702ae47ec75dc859d`;
- archive SHA-256
  `74ca606651befb7a1d0d5d0896783b5ec546c33d50583b15a13e49e67fcdb1de`;
- archive bytes `178316`;
- runtime tree SHA-256
  `3497c774d94fe202563bccba2af4a5f90925cb8d9b2e982cf4428d0efbea0190`;
- `runtime_smoke_checked=true`;
- `n_pairs_searched=480`;
- `n_pairs_completed_this_run=48`;
- `n_pairs_skipped_already_completed=432`;
- `full_non_smoke_search=false`;
- `ready_for_exact_eval_dispatch=false`;
- sidecar choice-state provenance: `144` machine-recorded scalar-equivalent
  pairs, all from post-patch chunks.

Dispatch blockers:

- claim lane before any GHA/remote eval dispatch;
- run exact-eval dispatcher preflight against `submission_dir`;
- record runtime tree SHA and terminal dispatch claim row;
- `sidecar_pair_search_records_missing_for_completed_pairs:336`;
- `non_full_sidecar_search_not_exact_eval_ready`.

Patched sidecar choice-state guard on the terminal manifest returned only:

- `sidecar_pair_search_records_missing_for_completed_pairs:336`.

Current completion: `480/600` pairs. Continue scalar chunks to `600/600`, then
run `--recheck-unproven-pairs` to replace legacy `0..335` provenance before
any exact-eval dispatch.
