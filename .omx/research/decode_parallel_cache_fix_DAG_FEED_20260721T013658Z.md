# DAG FEED — `FEED-592-DECODE-PARALLEL-SHARED-CACHE-20260721`

Authority: BUILD + `[macOS-CPU local timing]`; pointer UNMOVED; no launch,
score, promotion, or live-run mutation. MAIN landing review is a hard successor.

```text
counted base archive SHA-256 + exact decoder SHA-256
                      |
                      v
cache key = archive_sha[:16] + "_" + decoder_sha[:16]
                      |
          +-----------+-----------+
          |                       |
          v                       v
     canonical hit          locked cache miss
  receipt canonical?       stale partials absent?
  identities exact?        archive grammar exact?
  .npy SHA/bytes exact?     decoder env resolved?
  dtype/shape exact?        parallel decode n600
          |                       |
          |                       v
          |              exact batched factor-2 A
          |                       |
          |              fsync + atomic renames
          |                       |
          +-----------+-----------+
                      |
                      v
       read-only memmap of shared scorer planes
                      |
          +-----------+-----------+
          |                       |
          v                       v
   current M1 trainer       later run/arm consumers
     no file copy             no repeated decode
```

## Measured edge

- n16 exact M1 decoder: 111.62 s serial -> 21.17 s at 16 workers, 5.273x;
  raw SHA equal `6b550f16719b9379ec3237decd653f5e6ebafe5dd23d7c7e6f57b7c5d8ec5f2b`.
- n16 exact scorer projection: 0.772911 s serial -> 0.321709 s batched,
  2.403x; scorer SHA equal
  `5f4a6d1aa68754d98b94f5a8f599ecff7a43db49be44069bd2f44a108828c495`.
- Cache hit cost still includes a full `.npy` SHA verification by design; cache-hit
  latency was not separately benchmarked and is not claimed.

## Consumer wiring

- Sensitivity, Pareto, and bit allocation are unchanged because output bytes are
  proven identical and no score observation was produced.
- Cathedral/autopilot receives a preprocessing-availability edge only after full
  cache custody validation; any stale/noncanonical state refuses.
- Continual-learning signal is the reusable decomposed wall-time receipt, not a
  score posterior update.
- Probe disambiguation selected bounded NumPy batching over serial and process-pool
  projection from measured exact-byte and complexity criteria. Decoder workers
  remain explicitly overridable for the serial debug/proof arm.

Triality: DSL = existing env override/no invented flag; DAG = this feed;
equations = unchanged exact factor-2 numerator map. Pointer delta = none.
