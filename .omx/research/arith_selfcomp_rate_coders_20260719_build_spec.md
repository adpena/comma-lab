# Arithmetic and self-compression rate coders: implementation spec

Date: 2026-07-19
Lane: `lane_arith_selfcomp_rate_coders_20260719`
Authority: delegated local build/measure only; no launch, paid dispatch, score,
promotion, or pointer authority

## Objective

Land deterministic, parse-back-counted codec primitives and a resumable local
measurement tool for the operator's arithmetic-coding/self-compression rung.
The tool must rederive the existing n24 seg-secant payloads from their frozen
cache/custody and immutable stages, compare new lossless coders with the already
measured Brotli-Q11/zstd-19 bytes, measure the existing PDW1 bytes and the
explicitly DERIVED PDW2 construction layout, and measure int8 plus block-FP
coding on the read-only mod32cap EMA donor when that donor is supplied.

## Constraints

- Stay in this isolated worktree. Never modify `main`, `upstream`, or any other
  worktree.
- Sacred read-only tree:
  `experiments/results/levelset_n600_witness_20260717T113932Z/`.
- Bulk evidence belongs under
  `/Volumes/VertigoDataTier/pact/evidence/arith_selfcomp_20260719/`; do not write
  durable evidence to `/tmp`.
- Deterministic, seeded, exact parse-back; hashes and exact inputs/argv in the
  receipt. No scorer, launch, archive, or contest-axis claims.
- `constriction` may be used on the encode side if present, but must remain an
  optional measurement dependency. Do not add it to the production/inflate
  dependency set. Any adoption path must decode with repository-owned pure
  Python/NumPy code and report its decoder/source overhead separately.
- Preserve labels: the 338-byte PDW1 packet is MEASURED; the 138-byte PDW2
  margin-preserving construction is DERIVED until an actual strict PDW2 encoder
  exists. Do not call a synthesized or sliced 138-byte fixture PDW2.
- The 52.6 KB / 6.54 bpp #496 row is a prior measured comparison, not a new
  measurement. Current donor byte-close receipt records 61,842 base-weight and
  20,355 code-weight int8+Brotli bytes; keep section identity explicit.
- The existing block-FP implementation is ternary shared-exponent and lossy.
  Never call its historical ~1.017 b/parameter class a measured donor result.
- Do not change the pointer `0.1910828242 [contest-CPU]`.

## Owned files

- New `src/tac/optimization/arith_selfcomp_rate_coders.py`
- New `src/tac/tests/test_arith_selfcomp_rate_coders.py`
- New `tools/measure_arith_selfcomp_rate_coders.py`
- This build spec only. The parent session owns the final receipt and
  `.omx/research/arith_selfcomp_rate_coders_20260719_codex.md` after review.

Do not edit the existing secant tool/module, block-FP module, power-diagram
module, shared ledgers, `CLAUDE.md`, `AGENTS.md`, or reports.

## Required codec surfaces

1. A lossless codec ladder accepting bytes and, for residual context, signed
   integer arrays:
   - raw LZMA with explicit deterministic format/preset;
   - spatial-neighbor sign/magnitude context arithmetic coding, with a strict
     repository-owned decoder and exact decoded shape/dtype/value recovery;
   - zigzag + RLE + range/arithmetic floor check with strict parse-back;
   - Brotli-Q11 and zstd-19 comparators without changing their established
     semantics.
2. Count total framed bytes, not an entropy ideal. Headers/models/termination
   needed by a decoder must be included. Return SHA-256, parse-back status,
   implementation/dependency identity, and measured encoder/decoder overhead.
3. Context-versus-iid surfaces for int8 tensors must use a fully decodable
   stream and include per-tensor framing/model costs. Do not report ideal
   Shannon bits as coded bytes.
4. Block-FP surface must reuse the repository's shared-exponent ternary algebra
   or reproduce it byte-for-byte, report qint/exponent bytes separately, and
   make distortion explicit. It is a substitute quantizer; any sensitivity
   allocator composition remains unmeasured unless an actual allocation is
   applied.

## Measurement tool

- Provide a fail-closed CLI with explicit paths for composed secant receipt(s),
  cache, donor checkpoint, PDW1 receipt, output receipt, stage directory/state,
  and `--resume` where work is multi-stage.
- Rebuild each of the 24 pairs x reference plus nine operating points using the
  frozen cache, VJP margin custody, generated predictor, and existing
  transforms. Verify rebuilt hashes against immutable stage documents before
  counting new codecs. No frozen SegNet/PoseNet rerun is needed for the coder
  split; consume the existing hard-oracle d_seg/d_pose rows.
- Aggregate bytes/pair per operating point for every coder, best/Brotli ratio,
  adjacent secants, and rerun `solve_measured_waterfill` using each new measured
  byte column. State whether a discrete decision boundary changes.
- Extract exact PDW1 bytes from the custodied receipt and measure every lossless
  coder. Encode the 138-byte PDW2 construction only if a real strict encoder is
  present; otherwise retain a DERIVED accounting row and explicitly block an
  entropy-coded byte claim.
- If the donor checkpoint path is present, load it read-only, identify the exact
  EMA weight sections, measure deterministic int8 streams with all coders and
  block-FP at a bounded set of block sizes/thresholds. Produce a decode-side
  reconstruction checkpoint or arrays sufficient for a later byte-close scorer
  pass. If a real n>=24 byte-close scorer run cannot be done under the supplied
  environment, emit a precise blocker; do not substitute weight MSE for matched
  d_seg.
- Before large output, perform a storage preflight. Use atomic/write-once stage
  receipts and preserve all stages. No cleanup of source custody.

## Acceptance

- Tests cover deterministic repeatability, malformed/truncated/trailing input,
  signed extremes/zeros, exact residual and int8 parse-back, framed-byte
  accounting, context/iid distinction, block-FP reconstruction metadata, and
  fail-closed authority labels.
- Targeted pytest passes with warnings as errors; Ruff and `git diff --check`
  pass on owned files.
- No new mandatory runtime dependency; no modification outside owned files.
- Implementation does not commit. The parent session reviews, measures, writes
  the final memo/receipt, and commits.
