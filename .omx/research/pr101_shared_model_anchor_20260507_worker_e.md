# PR101 Shared-Model PMF Anchor - Worker E - 2026-05-07

## Scope

Worker E pursued the shared-model entropy path for PR101 quantized decoder
weights without editing existing PR101 entropy tools or shared allocator/state
surfaces. This tranche added a CPU-only planning probe:

- `tools/pr101_shared_parametric_pmf_probe.py`
- `src/tac/tests/test_pr101_shared_parametric_pmf_probe.py`
- artifact:
  `experiments/results/pr101_shared_model_anchor_20260507_worker_e/pr101_shared_parametric_pmf_probe.json`

No dispatch was attempted. No archive substitution or runtime decoder was
created.

## Input Custody

- Autodiscovered state dict:
  `experiments/results/cma_pr101_real_substrate_cmaes_20260507T223229Z/pr101_decoder_state_dict.pt`
- Input SHA-256:
  `b863362aaba1b9cae9b944f5e5b1a43a53ca824b7899ed7b80a2e2146d66f053`
- Tensors: `28`
- Quantized symbols: `228958`

## Model Families Tested

The probe charged model bytes for every estimate and kept all evidence
planning-only:

- shared spike/Laplace/Gaussian canonical grid over identity symbols;
- shared spike/Laplace/Gaussian canonical grid over fixed modulo-255 deltas;
- shared empirical PMF plus per-tensor temperature/zero-spike assignments;
- shared canonical PMF clusters with brotli-compressed fp16 shared tables plus
  per-tensor assignments.

All rows set `score_claim=false`, `ready_for_exact_eval_dispatch=false`, and
include blockers for no actual range/ANS bitstream, no runtime model
serializer/decoder, no archive substitution, and missing exact CUDA auth eval.

## Byte Results

Reference anchors:

- Brotli+Optuna PR101 archive: `178144` bytes
- per-tensor AAC archive: `178181` bytes
- IID per-tensor floor archive: `175916` bytes

Best shared result:

- model: `shared_canonical_pmf_clusters_identity_k12`
- payload estimate: `160255` bytes
- charged model parameter bytes: `2697`
- archive estimate: `179046` bytes
- delta vs `178144`: `+902`
- delta vs `178181`: `+865`
- delta vs `175916`: `+3130`
- cluster init: `largest_tensors_first`
- cluster sizes: `[14, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2]`

Selected negative controls:

- `shared_parametric_spike_laplace_gaussian_identity`: `206561` archive bytes,
  `+28417` vs `178144`
- `shared_parametric_spike_laplace_gaussian_delta_mod255`: `219209` archive
  bytes, `+41065` vs `178144`
- `shared_empirical_temperature_spike_identity`: `202100` archive bytes,
  `+23956` vs `178144`

The clustered shared-PMF family does beat this probe's brotli-fp16 per-tensor
PMF-table estimate (`179389`) by `343` bytes, but that is not enough to beat
the actual PR101 Brotli+Optuna archive or per-tensor AAC anchor.

## Verdict

Planning-negative, narrow scope: the strongest CPU-first shared-model estimate
in this tranche loses after charged model bytes. This does not kill the
shared-model family because no real range/ANS bitstream, runtime decoder, or
archive substitution was built. The low-parameter parametric families are not
close; the only competitive model transmits 12 shared fp16 PMF tables, which is
a table-sharing variant rather than the desired tiny hyperprior. Do not dispatch
this lane without a real range/ANS bitstream, runtime decoder, archive
substitution proof, and exact CUDA auth eval.

## Verification

Commands run:

```bash
ruff check tools/pr101_shared_parametric_pmf_probe.py src/tac/tests/test_pr101_shared_parametric_pmf_probe.py
UV_CACHE_DIR=/tmp/uv-codex-pr101-shared uv run --extra dev python -m pytest src/tac/tests/test_pr101_shared_parametric_pmf_probe.py -q
.venv/bin/python tools/pr101_shared_parametric_pmf_probe.py --output experiments/results/pr101_shared_model_anchor_20260507_worker_e/pr101_shared_parametric_pmf_probe.json
uv run ruff check tools/pr101_shared_parametric_pmf_probe.py src/tac/tests/test_pr101_shared_parametric_pmf_probe.py tools/cathedral_autopilot.py src/tac/tests/test_cathedral_autopilot.py
uv run --with pytest python -m pytest src/tac/tests/test_pr101_shared_parametric_pmf_probe.py src/tac/tests/test_cathedral_autopilot.py -q
uv run python tools/pr101_shared_parametric_pmf_probe.py --state-dict-path experiments/results/cma_pr101_real_substrate_cmaes_20260507T223229Z/pr101_decoder_state_dict.pt --cluster-k-values 1,2,3,4,5,6,7,8,9,10,11,12 --scale-grid-size 8 --output experiments/results/pr101_shared_model_anchor_20260507_worker_e/pr101_shared_parametric_pmf_probe.json --output-evidence experiments/results/pr101_shared_model_anchor_20260507_worker_e/cathedral_autopilot_evidence.jsonl
uv run python tools/cathedral_autopilot.py evidence-update --prior-evidence experiments/results/pr101_shared_model_anchor_20260507_worker_e/cathedral_autopilot_evidence.jsonl --output experiments/results/pr101_shared_model_anchor_20260507_worker_e/cathedral_autopilot_catalog_after_shared_model.json
```

Results:

- ruff: `All checks passed!`
- focused pytest: `4 passed in 0.54s`; post-hardened suite: `21 passed in
  0.68s`
- probe: wrote the JSON artifact above; best archive estimate `179046` bytes.
- post-hardened manifest uses repo-relative input paths and emits a
  non-promotable `shared_canonical_pmf_clusters` evidence row for
  `cathedral_autopilot`.
