# Shared Model PMF Worker S - 2026-05-07

## Scope

Worker S hardened the shared-model PMF / neural-hyperprior idea into a
deterministic CPU research artifact. This is not score evidence and is not a
dispatchable contest candidate.

Evidence semantics: `cpu_shared_model_pmf_exact_roundtrip_research_artifact`
with `score_claim=false`, `promotion_eligible=false`,
`rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Implementation Artifact

- Reusable module: `src/tac/shared_pmf_model.py`
- Probe CLI: `tools/pr101_shared_model_pmf_probe.py`
- Focused tests: `src/tac/tests/test_shared_pmf_model.py`
- Probe JSON written outside git state for this run:
  `/tmp/pr101_shared_model_pmf_probe_20260507_worker_s.json`

The module fits deterministic shared static PMF tables, serializes the model
as `brotli(serialized PMF tables + assignments + tensor lengths)`, range-codes
the quantized symbols with `tac.lossless.range_coder`, and verifies exact model
and payload reconstruction.

## Run

Command:

```bash
.venv/bin/python tools/pr101_shared_model_pmf_probe.py \
  --output /tmp/pr101_shared_model_pmf_probe_20260507_worker_s.json
```

Input state dict:

```text
experiments/results/cma_pr101_real_substrate_cmaes_20260507T223229Z/pr101_decoder_state_dict.pt
```

Source state-dict SHA-256:

```text
b863362aaba1b9cae9b944f5e5b1a43a53ca824b7899ed7b80a2e2146d66f053
```

Deterministic seed: `20260507`

Tensor/symbol count: `28` tensors, `228,958` symbols.

## Result

Best exact-roundtripped shared model:

- `K=12` shared PMFs
- Payload estimate: `160,505` bytes
- Exact encoded payload: `160,529` bytes
- Range stream: `160,505` bytes
- Payload header: `24` bytes
- Model bytes: `2,314` bytes
- Raw model bytes: `6,296` bytes
- Raw frequency table bytes: `6,120` bytes
- Raw assignment bytes: `28` bytes
- Raw tensor length bytes: `112` bytes
- Archive overhead: `16,094` bytes
- Archive-byte estimate: `178,937` bytes

Roundtrip:

- Model serialization roundtrip: `true`
- Brotli-compressed model roundtrip: `true`
- Range payload roundtrip: `true`
- Exact reconstruction: `true`
- Source/reconstructed symbol SHA-256:
  `684474c743b17957ae3f830539a29c7fa5ac939289f77cb931c2c308d6b917ae`

Model/payload hashes:

- Model raw SHA-256:
  `04473e778cbac4310b84880fa736fe8d1a05043ad8c6d8a54bb06bcb61ff0877`
- Model brotli SHA-256:
  `fdce1a10895df66dacee092f10db1a8978c6aa6b243c36e7d78b4ad68539f073`
- Payload SHA-256:
  `297308096d4642005d7e4d4ad7852489d813816ce9aadb814bedeb13cd830b46`

## Artifact Disposition

This exact CPU artifact is negative after charged bytes:

- Delta vs Brotli/Optuna reference `178,144`: `+793` bytes
- Delta vs per-tensor AAC reference `178,181`: `+756` bytes
- Delta vs IID per-tensor floor reference `175,916`: `+3,021` bytes
- Delta vs naive shared pooled range reference `203,196`: `-24,259` bytes

Disposition:

```text
negative_loses_to_brotli_and_per_tensor_aac_after_model_and_payload_bytes
```

Do not promote, rank, kill the broader family, or dispatch from this artifact.
The useful signal is narrower: deterministic clustered shared PMFs beat the
single pooled shared-PMF baseline by a large margin but still fail the current
Brotli/AAC byte bar once model overhead and an exact range payload are charged.

This is not a family falsification. Learned hyperpriors, multi-pass learned
models, HStack/VStack composition, and runtime-integrated model-as-code
variants remain active research lanes until they receive their own charged
artifacts and review.

## Dispatch Blockers

- CPU research artifact only
- No PR101 archive substitution performed
- No inflate runtime decoder wired
- No runtime-tree manifest or submission packet
- Missing exact CUDA auth eval
- Score claim false
- Promotion requires charged archive bytes and CUDA auth eval

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_shared_pmf_model.py
.venv/bin/python -m ruff check \
  src/tac/shared_pmf_model.py \
  tools/pr101_shared_model_pmf_probe.py \
  src/tac/tests/test_shared_pmf_model.py
```

Both focused checks passed.
