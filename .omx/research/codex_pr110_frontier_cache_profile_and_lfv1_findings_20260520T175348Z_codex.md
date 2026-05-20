# Codex PR110 Frontier Cache/Profile + LFV1 Findings

Generated: 2026-05-20T17:53:48Z
Author: Codex
Scope: PR110-safe local tooling and advisory frontier filters. No live PR110
runtime/body/archive files were edited.

## Verdict

PR110 remains open and unmerged as of `gh pr view 110` at this timestamp:
head `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`, one GitHub Actions
acknowledgement comment, no reviews, no maintainer eval result yet.

The PR110 release archive is not missing from the submission path. The required
archive is supplied as the PR release asset rather than as a committed tree
file, which matches the upstream submission pattern. No PR110-blocking cleanup
was found in this pass.

## Adversarial Review Inputs Incorporated

Four xhigh adversarial workers reviewed PR110 custody, runtime fidelity,
oracle/cache tooling, and math/signal selection. Their actionable findings were
incorporated into local tooling:

- stale sparse-cache authority: fixed by broader cache keys and an explicit
  baseline raw SHA path;
- loose-file inflate false proof: fixed by extracting candidate archives and
  making official control inflate consume `archive_extracted_data_dir`;
- LFV1 duplicate/out-of-range rows: runtime and builder now fail closed;
- HFV1/LFV1 precedence ambiguity: runtime and sparse probes reject both
  sidecars present;
- no-op and locality failures: no-op candidate raw outputs are skipped by
  default, and locality now allows subset-selected changes while rejecting
  out-of-selection changes;
- stale advisory cache and contest-axis wording: advisory labels reject
  `[contest-*]` and cache keys include raw/archive/evaluator/runtime inputs;
- arbitrary grid prefix and mixed-unit ranking: planner now stratifies grid
  coverage, and ranking uses typed component masses where available.

## Tooling Landed

New or updated local tools/modules:

- `tools/build_hfv1_sidecar_candidate.py`
- `tools/probe_lfv1_sparse_visibility.py`
- `tools/run_lfv1_sparse_visibility_batch.py`
- `tools/run_lfv1_sparse_visibility_cached_batch.py`
- `tools/run_contest_oracle_batch.py`
- `tools/run_raw_advisory_eval.py`
- `tools/summarize_lfv1_visibility_calibration.py`
- `src/tac/atom/contest_granularity.py`
- `src/tac/optimization/contest_oracle_search.py`
- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/runtime_hfv1/inflate.py`

Focused validation passed:

```text
.venv/bin/python -m py_compile tools/probe_lfv1_sparse_visibility.py tools/run_lfv1_sparse_visibility_batch.py tools/run_lfv1_sparse_visibility_cached_batch.py tools/run_contest_oracle_batch.py tools/run_raw_advisory_eval.py tools/build_hfv1_sidecar_candidate.py tools/plan_contest_oracle_search.py tools/summarize_lfv1_visibility_calibration.py src/tac/optimization/contest_oracle_search.py src/tac/atom/contest_granularity.py experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/runtime_hfv1/inflate.py
.venv/bin/python -m pytest -q src/tac/atom/tests/test_contest_granularity.py src/tac/tests/test_contest_oracle_search.py
5 passed in 0.17s
```

## Profile And Lowering Result

Single sparse-probe cProfile artifact:

`experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_batch/lfv1v2_sparse_visibility_hardened_smoke/sparse_probe_profile.prof`

Observed profile summary:

- total runtime: 1.688s for one sparse probe;
- `_decode_pairs`: about 1.077s;
- `torch.conv2d`: about 0.728s;
- import/module startup: about 0.670s;
- bicubic upsample: about 0.100s;
- archive parse: about 0.084s.

Lowering decision: first-order optimization is in-process batching and chunk
cache reuse, not Rust/Zig/Metal yet. The hot path is dominated by PyTorch model
decode plus repeated interpreter/runtime startup. SIMD/native lowering is still
appropriate for later raw diff/hash/ZIP grammar scans if profiling shows Python
loops dominating those paths.

Implemented cached sparse batch:

`tools/run_lfv1_sparse_visibility_cached_batch.py`

Cached batch artifact:

`experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_batch/lfv1v2_typed_component_cached_sparse_visibility_20260520_codex/sparse_batch_manifest.json`

Result:

- candidates: 64
- decoded chunks: `[48]`
- visible candidates: 19
- elapsed: 3.2132406669988995s
- candidate-level visibility diff versus prior subprocess batch: exact match
- prior subprocess batch wall from manifest timestamps: 21.508406s
- speedup: about 6.7x wall-clock on this 64-candidate filter

## LFV1 Empirical Findings

Baseline no-sidecar advisory:

- axis: `[macOS-CPU advisory]`
- canonical score: `0.19206142414659494`
- PoseNet: `0.00002943`
- SegNet: `0.00056039`
- rate: `0.00475469`
- archive bytes: `178517`
- `score_claim=false`, `promotion_eligible=false`

Old mixed-scalar first visible candidate:

- artifact: `lfv1v2_first_visible_alpha2e-5_k1_eval`
- candidate: `lfv1v2_k01_a0p00002_r0p45_p0p8_oy0p38_9b8548d951cd`
- pair: 43
- alpha: `0.00002`
- advisory score: `0.19216592414659495`
- PoseNet: `0.00002943`
- SegNet: `0.00056039`
- rate: `0.00475887`
- archive bytes: `178674`
- interpretation: visible at uint8 level, but component-neutral and rate-negative.

Typed-component first visible candidate:

- artifact: `lfv1v2_typed_first_visible_eval_subsetok_20260520_codex`
- candidate: `lfv1v2_k01_a0p00001_r0p95_p2_oy0p45_a97810186557`
- pair: 61
- alpha: `0.00001`
- radius scale: `0.95`
- power: `2.0`
- origin_y fraction: `0.45`
- full raw changed frames: `[122]`, within selected pair `[122, 123]`
- advisory score: `0.19216592414659495`
- PoseNet: `0.00002943`
- SegNet: `0.00056039`
- rate: `0.00475887`
- archive bytes: `178674`
- interpretation: visible at uint8 level, but component-neutral and rate-negative.

Do not mark LFV1 killed. The current evidence says visibility-only LFV1 arms
tested here are not moving PoseNet/SegNet and only add rate. Further LFV1 spend
needs a component-moving hypothesis or a cheap component-response gate before
full advisory scorer evaluation.

## Next Frontier Action

Use the typed/Pareto atom queue and cached sparse filter only as a fast rejector.
Do not burn evaluator time on more generic LFV1 visibility candidates unless
they are tied to a scorer-aware mechanism.

Recommended next artifact path:

1. Build a scorer-response micro-batch harness that evaluates only candidates
   passing sparse visibility and component-moving priors.
2. Add component-response features back into the oracle planner so candidate
   selection explicitly separates PoseNet-sensitive, SegNet-sensitive, byte-rate,
   pixel-visible, and master-gradient signals.
3. Move the next frontier attempt away from raw foveation visibility and toward
   scorer-aware selector/byte/codec transforms or another representation family
   with a credible component movement path.
4. Re-profile after that harness exists; only lower to Rust/Zig/Metal/MLX/MPS
   where measured hot paths are Python/native-loop bound rather than PyTorch
   model decode bound.

All candidate scores above are `[macOS-CPU advisory]`, not contest score claims
and not promotion/ranking evidence.
