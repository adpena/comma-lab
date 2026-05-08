# PR101 tiny-NN PMF smoke - Worker A - 2026-05-07

## Scope

Worker A owned only the tiny-NN PMF/hyperprior smoke lane. The question was
whether a small learned predictor can replace transmitted PMF/context tables
for PR101 quantized decoder symbols after charging model parameters.

No archive was changed. No score claim is made. No exact-eval dispatch is
ready.

## Artifacts

- `tools/pr101_tiny_nn_predict_pmf.py`
- `src/tac/tests/test_pr101_tiny_nn_predict_pmf.py`
- `reports/pr101_tiny_nn_pmf_smoke.json`

Input:

`experiments/results/cma_pr101_real_substrate_20260507T222605Z/pr101_decoder_state_dict.pt`

Input SHA-256:

`b863362aaba1b9cae9b944f5e5b1a43a53ca824b7899ed7b80a2e2146d66f053`

## Method

The tool consumes a PR101 decoder `state_dict`, quantizes with
`tac.pr101_split_brotli_codec.FIXED_STATE_SCHEMA` and `_quantize_tensor`, and
fits deterministic CPU-only low-rank softmax predictors over symbols in
`[0, 254]`.

Model families tested in the final report:

| Variant | Conditioning |
|---|---|
| `tensor_only` | tensor id plus fixed tensor features |
| `tensor_prev_symbol` | tensor id, fixed tensor features, and previous symbol |

Final config:

- rank: 7
- epochs: 500
- learning rate: 0.02
- seed: 17
- torch threads: 1
- parameter charge: Brotli-compressed symmetric-int8 model estimate

The model is fit on the same symbols being coded. This is valid only as a
minimum-description-length planning estimate because the model parameters are
charged; it is not a generalization claim.

## Result

Reference anchors:

| Anchor | Archive bytes |
|---|---:|
| Brotli+Optuna PR101 reference | 178,144 |
| IID per-tensor floor | 175,916 |
| Per-tensor AAC estimate | 178,181 |
| Naive Markov-1 AAC round trip | 199,238 |

Final measured estimates:

| Variant | Symbol payload | Model bytes | Estimated archive | Delta vs 178,144 | Delta vs 175,916 | Delta vs 178,181 |
|---|---:|---:|---:|---:|---:|---:|
| `tensor_only` | 160,765 | 1,920 | 178,779 | +635 | +2,863 | +598 |
| `tensor_prev_symbol` | 160,913 | 2,627 | 179,634 | +1,490 | +3,718 | +1,453 |

Disposition: negative smoke. The best tiny-NN predictor still loses after
charging model parameters. The previous-symbol neural context did not recover
the oracle Markov-1 headroom; it increased model cost and did not improve
payload enough.

An opt-in decoder-known position feature was also tried during the smoke. Best
temporary position-aware run was rank 7 / 500 epochs at 178,825 bytes, still
negative and slightly worse than the no-position best. The tool keeps position
features behind `--include-position-features` for reproducible follow-up, but
the final report leaves them off.

## Blockers

The manifest explicitly keeps:

- `score_claim=false`
- `score_affecting_payload_changed=false`
- `charged_bits_changed=false`
- `ready_for_exact_eval_dispatch=false`

Dispatch blockers:

- planning probe only
- no actual range or ANS bitstream
- no runtime model serializer or decoder
- model parameter quantization is estimated, not packet verified
- no archive substitution performed
- missing exact CUDA auth eval

## Next implication

Do not dispatch this tiny-NN PMF lane as-is. The best result is only 635 bytes
behind the Brotli+Optuna reference, but the deployable route needs a real packet
compiler and a predictor that either reaches the IID floor with less than about
1.6 KB charged model cost or exploits context substantially better than this
low-rank model. The higher-EV continuation is a deterministic context coder or
ANS/range packet harness with golden vectors, not another unconstrained PMF
table compression pass.

## Verification

- `uv run ruff check tools/pr101_tiny_nn_predict_pmf.py src/tac/tests/test_pr101_tiny_nn_predict_pmf.py`
- `uv run --with pytest python -m pytest src/tac/tests/test_pr101_tiny_nn_predict_pmf.py -q`
- `.venv/bin/python tools/pr101_tiny_nn_predict_pmf.py --state-dict-path experiments/results/cma_pr101_real_substrate_20260507T222605Z/pr101_decoder_state_dict.pt --output reports/pr101_tiny_nn_pmf_smoke.json --rank 7 --epochs 500 --learning-rate 0.02 --variants tensor_only,tensor_prev_symbol`
- `.venv/bin/python -m json.tool reports/pr101_tiny_nn_pmf_smoke.json`

Focused tests: 4 passed, with the existing pytest-config warning about unknown
`timeout`.
