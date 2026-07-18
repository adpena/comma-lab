# v10 power-diagram blocked-prefix harvest specification (2026-07-18)

## Trigger and immutable evidence

- The governed n600 run stopped fail-closed at canonical frame 195 after
  frames 0 through 194 were durably committed.
- The preserved historical measurement-source SHA-256 is
  `be094a1540a94bf51aa98706b6d4515eec150bb569380f69b308ed66556cd7c9`.
- The blocked extraction checkpoint SHA-256 is
  `58656d231af5c63b12b3594d8eeeeccf0b2d0f25c09154ef3ef6da759e1fce4b`.
- The preallocated quotient cache SHA-256 is
  `59e96781aa1bac153bc8bb277cecdbd4b4e98fdfd41f50aa2294537b90390944`.
- The checkpoint records `next_canonical_frame=195`, exactly
  `38,338,560 = 195*384*512` committed samples, one frozen-head transformed
  power mismatch at frame 195, and zero CPU-Torch-forward mismatches.

The exact original source bytes and blocked SSD artifacts are immutable
evidence. The source is stored only inside a deterministic non-source gzip
container with a v2 manifest; validation decompresses it in memory only. The
live historical tool path is a fail-closed tombstone because its cleanup
certificate was unsafe. Do not materialize or execute the source, resume the
blocked checkpoint, truncate the cache, or delete/move either SSD file.

## Narrow diagnostic authority

Add a separate deterministic harvester plus focused tests. It must consume the
preserved blocked checkpoint/cache and emit one durable JSON receipt labeled
`ADVISORY_POSTHOC_PREFIX_0_194_OF_600`. It may fit and rate a power target on
the committed prefix only. It must not call that prefix n600, through-R, an RGB
receiver, an equivalent-rate comparison, factor-6 completion, or closure of
the 4.5 percent score gap.

The harvester must:

1. Accept explicit absolute checkpoint, feature-cache, GT-cache, upstream-root,
   historical-container, historical-manifest, current-tombstone, and output
   paths; restrict output to a new `.json` beneath the existing resolved
   `REPO_ROOT/.omx/research` tree and refuse overwrite, parent creation, or
   source/main/SSD/transient/symlink-escape custody.
2. Verify the exact checkpoint/cache/container and in-memory decompressed source
   hashes above, manifest lineage,
   the reviewed tombstone hash, the scratch marker,
   the checkpoint schema and immutable identity, all pinned source/model/cache
   hashes, blocked status/reason, geometry, canonical prefix length, sample
   counts, and float32 memmap byte geometry before fitting.
3. Reuse the checkpoint's float64 sufficient statistics and committed
   adjacency; fit only frames 0..194; scan exactly those 38,338,560 cached
   feature/label pairs to measure fitted feature-pullback mismatch.
4. Strictly encode/decode/re-encode the fitted `PDW1`. Report measured raw and
   Brotli-Q11 bytes plus the derived optimistic rounded-up ideal order-0 entropy
   estimate under free-PMF/no-overhead assumptions. Never call it a realizable
   ceiling or a lower bound on a legal archive. Compare against
   228,764 B, 235,974 B, and 225,272 B only as non-equivalent target-payload vs
   full-realization references.
5. Preserve the positive-control exposure separately: frames 0..195 were
   observed, frame 195 was excluded from the post-hoc fit because it triggered
   the stop, power mismatch count one, CPU-Torch-forward mismatch count zero.
6. Record both artifact hashes/bytes and state that the cache is preallocated
   for 600 frames while only prefix frames 0..194 are committed. Never cleanup.
7. Emit the narrow instance/formulation verdict
   `FROZEN_HEAD_FLOAT32_POWER_TARGET_POSITIVE_CONTROL_BLOCKED_AT_FRAME_195`;
   family and paradigm remain open; receiver arithmetic is unspecified;
   equation registration and pointer movement remain unauthorized.
8. Atomically write the receipt and validate fail-closed authority fields.

The numerical-tie pixel and arithmetic alternatives must be backed by the
separate governed one-frame reproducer. It reads only frame 195 and one target
pixel, records exact inner argv plus wrapper custody, and proves all input hashes
and metadata unchanged. No epsilon, ULP nudge, tie override, or implicit
receiver arithmetic is allowed.

## Acceptance

- Small fixtures prove exact prefix denominator/order, blocked-state custody,
  dense-vs-streaming fit parity, strict PDW1 parse-back, non-equivalent rate
  labels, container/tombstone lineage, output-boundary/overwrite refusal, and refusal on
  any hash/state/geometry drift or live-tool masquerade.
- Focused tests, Ruff, py_compile, and diff checks are clean.
- The real harvester runs read-only over the preserved SSD artifacts and writes
  only the durable receipt.
- The governed one-frame diagnostic reproduces generic-f64 class 1, native-f32
  exact tie/first-max class 0, and CPU-Torch class 0 at `(195,214,112)` without
  altering the checkpoint, cache, GT, or any other verified input; every input
  is re-hashed after inference.
