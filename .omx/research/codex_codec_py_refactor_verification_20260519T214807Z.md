# codec.py Refactor Verification - Codex

Timestamp UTC: 2026-05-19T21:48:07Z

Directive source: `.omx/research/codex_routing_directive_codec_py_refactor_with_byte_identity_verification_20260519T211500Z.md`

## Scope

Refactor target:

`experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/codec.py`

Frozen archive:

`experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip`

Archive SHA-256:

`6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`

No archive bytes were changed. No score, promotion, rank, or exact-eval claim is made by this refactor.

## Refactor Result

Runtime-code commits:

- `11500bbfe` - `codec: extract FEC6 sidecar helpers byte-identically`
- `c5a01413e` - `codec: document compact FEC6 parse helpers`

Changed files:

- `.gitignore`
- `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/codec.py`
- `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/codec_sidecar.py`

Line-count outcome:

- `codec.py`: 508 physical lines before refactor, 181 after refactor/docstring polish.
- `codec_sidecar.py`: 349 physical lines.
- Public runtime entrypoint preserved: `parse_archive(archive_bytes)`.

Helper split:

- `codec.py` keeps compact archive parsing, Brotli/LZMA decoding, tensor reconstruction, and the public `parse_archive` API.
- `codec_sidecar.py` owns latent sidecar/Huffman/rank decoding helpers and exposes `apply_latent_sidecar(latents, data)`.

The sidecar constants are local to `codec_sidecar.py` to avoid circular import and keep the submission runtime import path simple under `inflate.py`'s `src` path injection.

## Byte-Identity Gate

Baseline command shape:

```bash
PACT_PYTHON_BIN=$PWD/.venv/bin/python \
  bash "$SUBDIR/inflate.sh" "$DATA" "$OUT" upstream/public_test_video_names.txt
```

CPU thread environment:

- `OMP_NUM_THREADS=$(sysctl -n hw.ncpu)`
- `MKL_NUM_THREADS=$(sysctl -n hw.ncpu)`

Inflated output hash, baseline:

```text
d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c  ./0.raw
```

Inflated output hash, post-refactor/docstring:

```text
d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c  ./0.raw
```

Normalized relative diff verdict:

```text
0 /tmp/codec_refactor_20260519T213400Z/baseline_repeat_diff.relative.txt
0 /tmp/codec_refactor_20260519T213400Z/post_refactor/diff_verdict.relative.txt
0 /tmp/codec_refactor_20260519T213400Z/post_docstring/diff_verdict.relative.txt
```

Interpretation: PASS. The current refactored runtime inflates the frozen archive to byte-identical output relative to the baseline runtime.

## Timings

Local CPU inflate timings on this machine:

- Baseline run 1: `real 37.66`, `user 102.47`, `sys 6.83`
- Baseline run 2: `real 40.20`, `user 104.95`, `sys 7.49`
- Post-refactor run: `real 39.46`, `user 103.30`, `sys 7.25`
- Post-docstring run: `real 38.92`, `user 103.28`, `sys 7.42`

The refactor is performance-neutral within local run variance.

## Harness Correction

The directive's literal diff sketch is unsafe if hash manifests include absolute scratch paths. Absolute-path manifests produce a non-empty diff even when file bytes are identical because the path fields differ.

The byte-identity gate used here hashes from inside each inflate output directory:

```bash
(cd "$OUT" && find . -type f | sort | xargs shasum -a 256 > ../output_shas.relative.txt)
```

This keeps the manifest paths stable (`./0.raw`) and makes the diff verdict reflect output bytes rather than scratch-directory names.

## Follow-Up

This closes the internal readability/refactor part of the codec directive. The next frontier-moving use of this surface is not another cosmetic split; it is a byte-closed payload experiment or packer/deconstruction lane that uses the now-readable sidecar/Huffman helpers while preserving the same normalized byte-identity discipline.
