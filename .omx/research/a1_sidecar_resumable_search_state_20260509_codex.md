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
