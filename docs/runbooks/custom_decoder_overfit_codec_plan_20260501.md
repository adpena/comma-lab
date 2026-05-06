# Custom Decoder Overfit Codec Plan

Updated: 2026-05-01

## Purpose

Design a contest-compliant custom decoder path for overfit mask and latent
payloads. The intended use case is offline search with exact knowledge of the
contest data and hardware, followed by deterministic inflate from charged bytes
only. This runbook is a design and bounded probe plan, not score evidence.

Any score claim still requires the canonical path:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

Prefer `experiments/contest_auth_eval.py --device cuda` and use structured JSON
artifacts as the authority. CPU, MPS, byte-only, and synthetic probe results are
empirical design evidence only.

## Non-Negotiable Constraints

- All score-affecting payload bytes must be inside `archive.zip` or fixed
  contest code. No external sidecars, environment files, cache reads, or host
  paths may affect reconstruction.
- Do not modify upstream scorer files or rely on scorer internals outside the
  published eval path.
- Offline compression may be arbitrarily expensive. Inflate must be
  deterministic, bounded, reproducible, and within contest runtime budget.
- Every decoder output used for scoring must be produced from archive members
  plus fixed code. Learned tables, scripts, grammars, bytecode, indexes, and
  repair streams are side information and must be charged if data-specific.
- Archive construction must be deterministic: fixed member order, timestamps,
  permissions, compression settings, manifest records, member hashes, and
  zip-slip-safe paths.
- Promotion requires CUDA auth eval on exact archive bytes, archive SHA-256,
  component distances, sample count, recomputed score, command, environment,
  manifest, provenance, and logs.

## Candidate Decoder Families

| Family | Best Use | Byte Model | Decoder Risk | Status |
| --- | --- | --- | --- | --- |
| Bitpacked RLE masks | Large flat class-id masks with long row/temporal runs | Header + varint run lengths + bitpacked class values; optional zlib/Brotli member compression | Low in Python/C; easy to bounds-check and audit | Prototype first |
| Range-coded or ANS masks | Residuals with stable class/context probabilities | Static or per-block probability tables charged in archive; entropy stream charged | Medium; arithmetic edge cases and model/table custody need tight tests | Second after RLE baseline |
| Connected-component scripts | Masks dominated by contiguous semantic blobs | Per-frame components, boxes, polygons, fill class, holes, repair pixels | Medium; geometry rasterization must be bit-exact across platforms | Useful for Alpha masks |
| Temporal grammar | Slowly evolving objects, lanes, road regions, or latent tensors | Keyframes + operations: translate, dilate, erode, split, merge, patch | Medium-high; grammar search can overfit and decoder must reject invalid programs | Strong offline lane |
| Small VM/bytecode decoder | Mixed mask/latent structures where no fixed grammar wins | Fixed VM opcodes in code; bytecode/data charged in archive | High; must prove termination, bounds, determinism, and no hidden host access | Later only |
| Rust/Zig/C static binary | Need faster deterministic inflate than Python | Generic decoder binary as fixed contest code, or charged if shipped in archive; data-specific tables charged | Medium-high; toolchain/repro/custody burden | Feasible after Python proof |
| Assembly/SIMD kernels | Hot loops after format is frozen | No new side info; code-only speed path | High portability and CPU-feature risk; fallback required | Avoid until needed |
| Python-only fallback | Auditability and fastest iteration | More bytes may be needed for simpler format; slower inflate | Low; contest runtime budget must be measured | Keep as reference |

## Recommended Architecture

Start with a simple, auditable `CMCP_RLE1` mask payload:

```text
magic/version/shape/classes/run_count
varuint run lengths in row-major frame order
bitpacked run class values
crc32 over header+payload
```

This is not expected to beat a mature video codec on every candidate, but it is
the right reference decoder because it is deterministic, closed-form, and easy
to port to C/Rust/Zig. It also gives a clean baseline for more complex offline
search:

- Replace raw run lengths with delta-coded, block-local, or range-coded lengths.
- Add context-coded class values using previous row/frame predictors.
- Add repair streams for lossy geometric primitives.
- Split into independently decoded tiles to bound memory and permit early
  validation.

For latent payloads, keep the same discipline: fixed header, explicit tensor
shape/dtype, charged quantization tables, deterministic decode, and a hashable
manifest. Do not hide latent priors in code if they are fitted to the contest
data.

## Static Binary Feasibility

A Rust/Zig/C decoder is contest-feasible only if the boundary is clean:

- Generic decoder code can be fixed contest code when the contest submission
  format permits it. Any data-specific tables, grammar programs, selected
  constants, or learned parameters must remain charged archive bytes.
- A static binary shipped inside `archive.zip` is charged by definition. This
  can still win if it replaces much larger payload bytes, but the binary size
  must be part of the score formula.
- The build must be reproducible enough for custody: source SHA, compiler
  version, target triple, flags, binary SHA, size, and smoke output.
- The binary must not read outside the extracted archive/work directory, must
  reject unsafe paths, and must have fixed memory bounds.
- Python reference decode remains the oracle until the native decoder passes
  bit-exact cross-platform tests against the same payloads.

Assembly and explicit SIMD should wait. Plain C/Rust/Zig loops are usually fast
enough for class-id masks, and CPU-feature-specific code creates avoidable
reproducibility and fallback risks.

## Implementation Order

1. Land a non-promotable pure-Python probe that round-trips deterministic
   synthetic masks, records byte counts, hashes, and compliance warnings.
2. Add real decoded-mask ingestion only after the source archive and decoded
   mask contract are already in custody. The probe must validate source SHA,
   shape, and member hashes before computing byte estimates.
3. Evaluate independent payload families offline: RLE, RLE+range-coded lengths,
   component scripts, temporal grammar, and lossy primitive plus repair stream.
   Treat every output as empirical until an archive is built and CUDA-evaluated.
4. Freeze one decoder format and write a Python inflate reference with
   fail-closed validation: magic, version, dimensions, counts, CRC/hash, class
   range, total pixels, and no trailing bytes.
5. If Python inflate is too slow, port only the frozen decoder to Rust/Zig/C.
   Keep the Python reference and add bit-exact fixtures for the native binary.
6. Integrate into a candidate archive in a separate scoped change: deterministic
   archive build, manifest, payload closure check, local smoke, then CUDA auth
   eval.
7. Stack only after the standalone custom-decoder archive has exact evidence;
   stacked archives need their own exact eval.

## Custody And Eval Plan

Every candidate archive must preserve:

- `archive.zip` bytes, size, SHA-256, deterministic member inventory, and
  zip-slip/hidden-sidecar validation.
- Decoder source or binary identity, including source SHA and binary SHA when
  native code is used.
- Payload manifest with schema, version, shape, dtype/classes, stream byte
  counts, hashes, selected model/table hashes, and exclusion reasons.
- Inflate command, exact environment, hardware, logs, stdout/stderr, and
  runtime budget proof.
- `contest_auth_eval.json` from CUDA auth eval, plus recomputed score from
  `seg_dist`, `pose_dist`, and exact archive bytes.

A custom decoder result can be promoted only when the scored artifact proves
payload closure through `archive.zip -> inflate.sh -> upstream/evaluate.py`.
No local renderer shortcut, extracted workdir sidecar, or stale decoded mask
file may participate in the score path.

## Current Prototype Boundary

`experiments/custom_mask_codec_probe.py` is allowed to exist only as a local
diagnostic scaffold. It must emit:

- `score_claim=false`
- `promotion_eligible=false`
- `evidence_grade=empirical`
- `local_probe_only=true`
- `scorer_network_loaded=false`
- the canonical CUDA auth eval path required for any future score claim

The probe may demonstrate deterministic bitpacked RLE round-trip behavior and
byte accounting on synthetic masks. It must not build a contest archive, modify
inflate, call scorer networks, or imply promotion eligibility.
