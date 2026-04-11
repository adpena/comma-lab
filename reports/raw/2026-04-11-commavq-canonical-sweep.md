# commavq Canonical Sweep

## authoritative surfaces

- canonical repo: `workspace/upstream/commavq`
- canonical VQ bridge:
  - `workspace/upstream/commavq/utils/vqvae.py`
  - `workspace/upstream/commavq/notebooks/decode.ipynb`
  - `workspace/upstream/commavq/notebooks/encode.ipynb`
- canonical GPT surface:
  - `workspace/upstream/commavq/utils/gpt.py`
  - `workspace/upstream/commavq/notebooks/gpt.ipynb`
  - `workspace/upstream/commavq/nanogpt/prepare.py`
- canonical lossless token contract:
  - `workspace/upstream/commavq/compression/compress.py`
  - `workspace/upstream/commavq/compression/decompress.py`
  - `workspace/upstream/commavq/compression/evaluate.py`

## official model assets

- Hugging Face model repo: `commaai/commavq-gpt2m`
- available shipped artifacts:
  - `decoder_pytorch_model.bin`
  - `encoder_pytorch_model.bin`
  - `pytorch_model.bin`
  - `decoder.onnx`
  - `encoder.onnx`
  - `gpt2m.onnx`
  - `gpt2m_share_buffer.onnx`
  - `temporal_decoder.onnx`

## what is usable immediately

- token->RGB decode via the canonical decoder notebook/class path
- GPT next-token prediction via canonical `utils/gpt.py`
- exact token archive reshape/transpose contract via canonical compression scripts

## what is still missing even in canonical source

- no off-the-shelf semantic extractor
- no documented ONNX runtime wrappers
- no off-the-shelf GPT arithmetic coder
- no canonical token->RGB batch bridge wrapper

## local result

- local off-the-shelf bridge now works through `tac lossless token-rgb-sample`
- sample artifact metadata:
  - `reports/raw/2026-04-11-commavq-token-rgb-bridge/example_tokens_rgb_16.json`
- decoded RGB cache shape:
  - `(16, 128, 256, 3)`
- runtime:
  - canonical decoder weights from `commaai/commavq-gpt2m`
  - device request `mps`
  - resolved dtype `float32`
