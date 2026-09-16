# Compact token video decoder

The 179,286-byte archive contains a learned probability model, a compressed field of five-class tokens, a small token renderer, and a low-rank pose carrier. All learned weights, coefficients, and per-frame choices are inside the archive.

Decode first reconstructs the model weights. A learned prior and online context statistics then arithmetic-decode the token field in causal groups. The token renderer produces the second RGB frame of each pair; the pose carrier and stored pixel transformations produce the first. The output is 1,200 raw RGB frames at 1164 × 874 pixels.

Dependencies: Python 3.11 or newer, NumPy, PyTorch, and Brotli. NumPy and PyTorch are in the evaluator environment. If Brotli is absent, decoding exits with an installation instruction.

`range_decoder.c` is an optional C11 arithmetic-decoding loop compiled with `cc`; without a working compiler, Python decodes identically.

Run from the submission directory inside the contest repository:

```sh
python3 -m zipfile -e archive.zip extracted
sh inflate.sh extracted inflated ../../public_test_video_names.txt
```

The file list must contain `0.mkv`. The decoder writes `inflated/0.raw`.
CUDA is required by default. Add `--device cpu` to select CPU explicitly; CPU and CUDA output can differ.
The original receiver decoded this archive in 1,186 seconds on a T4. This rewritten receiver still requires its own T4 timing and evaluation.

Archive SHA-256: `aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957`

Archive size: 179,286 bytes.
