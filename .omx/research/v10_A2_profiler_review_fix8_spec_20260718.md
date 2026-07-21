# v10 A2 profiler review FIX8 specification — 2026-07-18

`research_only=false`

Authority is the delegated A2 prompt plus the three independent final-byte
reviews recorded as round 3 `NOT_CLEAN` for
`working-tree-sha256:2dc2421d285522154bcf0bd26ff2b74f10df0b8ccc7d9625d83dcb3f129397f0`.
No measurement or source-freeze commit may proceed until all requirements
below land, cheap tests pass, and three fresh reviews inspect the same bytes.

## Receiver and scientific closure

1. The receiver MUST define the exact scorer-effective support union `S` for
   the canonical 874x1164 -> 384x512 resize. It MUST reject overlap, missing
   support values, excess values, or coverage unequal to `S`.
2. Bytes outside `S` MUST use a declared deterministic fill. They are not
   source-byte reconstruction and are not charged as source residuals.
3. A source-seeded node-cap-1 control MUST prove equality to the source on
   `S`, exact equality of all integer resize numerators, receiver parse-back,
   and frozen-source/candidate SegNet argmax equality. It MUST NOT require or
   claim full source-byte identity.
4. Scored rows MUST cross-bind the cache's SegNet weights, upstream modules,
   frame utilities, TAC scorer, extractor, cache module, and runtime contract
   to the current execution. Every scored source frame MUST be freshly
   re-forwarded; cached labels must equal that current frozen-source argmax
   before candidate mismatch counts are admitted.
5. Final validation MUST reconstruct the entire RD row, selected/total counts,
   scorer and rate frame scopes, axis, d_seg, claims, positive control,
   derivation, and authority from immutable identity plus semantically replayed
   stage receipts. Persisted result prose is never trusted.
6. zlib/Brotli rows are deterministic encoder byte counts. Their emitted
   chunks MUST be decompressed in-process to the identical raw stream before a
   codec-parse-back field is true. This is not a global compressed-stream
   minimum claim.

## Admission, storage, and cache closure

1. The public local-test flag MUST never bypass governed admission. Local
   output is permitted only under an actual pytest process and a tiny prefix;
   it can never complete n600 or emit `FULL_N600`.
2. Both entrypoints MUST require a direct `tools/safe_run.py` parent, reject
   `--skip-admission-gate` and admission overrides, bind the parent command to
   the child's exact argv, and require outer RSS/time values equal to the
   child's requested caps. A raw environment marker is insufficient.
3. The executed safe-run/admission/helper source bytes MUST be in identity
   custody. Resource fields must distinguish parent-command attestation from a
   completed safe-run status receipt.
4. Extractor argv MUST reparse exactly to the effective namespace. Resume is
   removed only from the canonical fresh rebuild argv.
5. Cache roots and custody files MUST reject symlinks, hardlinks, and
   non-regular/non-directory paths before following them, including writable
   resume array files.
6. Creation recovery MUST enumerate the exact certified staging entry set.
   Unknown bytes cause a blocker and are preserved. Recursive deletion is
   allowed only after every staged entry is proved part of the certified
   rebuildable pre-final layout.
7. The SSD waterfall MUST select the first existing approved root. A lower
   tier while a higher tier exists is refused unless a future explicit typed
   operator override is added; FIX8 adds no override.

## Label and scope closure

1. `n*H(empirical order-0 PMF)` is an order-0 IID plug-in ideal-length
   estimate, not a universal lower bound. Rename every schema label and test.
2. Cardinality lower/upper bounds are for the exact affine source-resize fiber
   only. They are not upper bounds on the broader SegNet argmax winner-cell
   preimage or Kolmogorov complexity.
3. Per-block exhaustive selector exactness remains distinct from global
   zlib/Brotli stream optimality. Factor 2 and factor 6 remain `PARTIAL`; factor
   10 remains `MISSING`; score/promotion authority remains false.

## Required regressions

- canonical production geometry support complement and deterministic-fill
  receiver closure;
- node-cap-1 source seed equality on `S` plus exact resize numerators;
- stale/foreign scorer-cache rejection and current-source re-forward mismatch;
- full final-RD-row tamper rejection;
- zlib and Brotli decompression equality;
- raw marker, safe-run skip, mismatched outer caps, and local full-n600 refusal;
- extractor exact-argv substitution refusal;
- cache symlink/hardlink refusal;
- unknown staging sentinel preservation;
- Vertigo-first waterfall refusal;
- repeated-symbol stream showing the order-0 estimate can exceed actual
  context-compressed bytes without being called a lower bound.

Pointer remains unchanged. Sacred result bytes remain read-only. No Fourier,
GPU, paid dispatch, score promotion, pose-bank claim, or factor-10 claim.
