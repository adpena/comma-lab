# commavq ONNX Inspect

- source repo: `commaai/commavq-gpt2m`
- file: `decoder.onnx`
- cached path: `/Users/adpena/.cache/huggingface/hub/models--commaai--commavq-gpt2m/snapshots/12f0a5e31c22b492dc391aa348cbe422139e3087/decoder.onnx`
- input:
  - `encoding_indices`
  - dtype enum `7` (`int64`)
  - shape `(b, 8, 16)`
- output:
  - `big_decoded_img`
  - dtype enum `1` (`float32`)
  - shape `(b, 3, 128, 256)`
- local available providers:
  - `CoreMLExecutionProvider`
  - `CPUExecutionProvider`
  - `AzureExecutionProvider`

Implication: the canonical decoder already has an off-the-shelf inference artifact that matches the token cube layout directly and can be accelerated locally through ONNX Runtime without inventing a new model format.
