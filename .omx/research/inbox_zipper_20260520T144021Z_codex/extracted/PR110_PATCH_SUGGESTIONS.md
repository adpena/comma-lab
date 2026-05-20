# Patch Suggestions for PR #110

The following snippets illustrate how the PR body could be tightened.  They are
suggested in patch‑ready form, assuming the PR description is written in
Markdown.  Line numbers are illustrative; maintainers should apply the
replacements where the matching text occurs.

## Replace the opening paragraph

**Current:**

> hnerv_fec6_fixed_huffman_k16#110
>
> Hosted at: `https://github.com/adpena/comma_video_compression_challenge/releases …`
> ...

**Suggested replacement:**

```markdown
### Submission summary

This submission extends the PR #101 HNeRV packet with a 31‑mode frame‑entropy
coding (FEC6) selector.  The selector chooses one of 31 deterministic frame
transforms per frame; 16 of these modes are active in the archive.  A fixed
Huffman code of order 16 encodes the selector indices.  The new sidecar is
appended outside PR #101’s Brotli‑compressed payload and adds 259 bytes to the
archive.  The result lowers the CPU score to 0.192051 and the CUDA score to
0.226210 (using the upstream evaluator on commit `<commit>`).  See below for
exact archive and evaluation details.
```

This opening frames the submission concisely without diving into the broader
research narrative.

## Condense the archive and runtime facts into a bullet list

**Current:** long paragraphs describing the archive, runtime tree and
dependencies.

**Suggested replacement:**

```markdown
### Evidence

* **Archive:** `archive.zip` hosted at [GitHub Releases](<release‑url>),
  SHA‑256 `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`,
  178 517 bytes.  Contains one member `x` (178 417 bytes) stored with
  `compression_type=0` (ZIP_STORED).
* **Runtime:** `inflate.sh`, `inflate.py`, `src/codec.py`,
  `src/codec_sidecar.py`, `src/frame_selector.py`, `src/model.py`.  All staged
  under `submissions/hnerv_fec6_fixed_huffman_k16/`.
* **Dependencies:** Python 3.x; `torch`, `numpy`, `brotli`; no other repos.
* **Evaluation (CPU):** `upstream/evaluate.py --device cpu` on Linux x86_64 with
  `num_threads=2` produced `0.192051` total score.
* **Evaluation (CUDA):** same archive evaluated on NVIDIA T4 produced `0.226210`.
* **Training:** one‑shot per video on an A100; training scripts live in
  `comma‑lab` and are not part of this submission.
```

## Remove internal paths and revision noise

Delete any references to local working directories (e.g., `.omx/research/…`) and
replace them with public repository commit hashes only.  Example:

```markdown
*Source pinned to commit `b392343d758aba0d3595dd18609f9ca8a8af3e1b` in
[`comma‑lab`](https://github.com/adpena/comma‑lab) for reproducibility.*
```

## Soften the saturation claim

**Current:**

> post‑hoc general‑purpose recompression of the actual contest `archive.zip`
> member bytes has no deliverable rate‑term gain…

**Suggested replacement:**

> In our experiments on this archive and the related HNeRV variants, running
> generic compressors on the exported `archive.zip` bytes did not materially
> reduce the score.  Consequently, further improvements will likely require
> changing the emitted representation, training priors, or entropy model rather
> than re‑compressing the same bytes.

This phrasing clearly indicates that the observation is local to the tested
archives and avoids implying a general impossibility.

## Move research context to a separate document

At the end of the PR description, add a short paragraph:

```markdown
For a detailed discussion of potential future directions (pretraining, new
representations, quantization, priors, etc.), please see the source map in
[`comma‑lab/docs/full_stack_source_map.md`](https://github.com/adpena/comma‑lab/blob/main/docs/full_stack_source_map.md).
These materials are not part of this runtime.
```
