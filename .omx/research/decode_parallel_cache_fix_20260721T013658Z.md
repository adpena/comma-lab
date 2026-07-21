# Decode parallelism and shared base-plane cache — Task #592

Authority: `[macOS-CPU local timing]`, BUILD + local verification only. No score,
launch, promotion, pointer mutation, or inference onto contest CPU/CUDA. MAIN must
review this branch and merge only at the post-M1 quiet boundary.

## Outcome

The C2 base materializer now honors an explicit `INFLATE_WORKERS`; otherwise it
requests `max(1, cpu_count - 2)`. The real M1 `yhat_rd_ladder` decoder was not
covered by C1's receiver identity receipt, so it was independently re-proved on
an n16 prefix before the default changed.

| component | serial | selected implementation | byte identity | local speedup |
|---|---:|---:|---|---:|
| exact base decode | 111.62 s, 1 worker | 21.17 s, 16 workers | raw SHA `6b550f16719b9379ec3237decd653f5e6ebafe5dd23d7c7e6f57b7c5d8ec5f2b` | 5.273x |
| factor-2 scorer projection | 0.772911 s, 32 serial plane calls | 0.321709 s, NumPy batches of 4 planes | scorer-byte SHA `5f4a6d1aa68754d98b94f5a8f599ecff7a43db49be44069bd2f44a108828c495` | 2.403x |
| decomposed materialization compute | 112.392911 s | 21.491709 s | component proofs above | 5.230x DERIVED |

The projection comparison measured batch sizes 1/2/4 at 0.563435/0.434462/
0.321709 s; all were byte-identical to serial. Batch 4 is the simplest measured
winner and bounds its transient integer workspace.

## Cache and custody contract

The shared entry is
`/Volumes/VertigoDataTier/pact/cache/base_scorer_planes/<archive_sha16>_<decoder_sha16>/`.
It contains `base_scorer_planes.npy`, a canonical
`c2_base_plane_materialization.v1` receipt, and a small lock file. Every hit:

1. parses and canonical-reencodes the receipt;
2. checks archive SHA, decoder SHA, cache key, exact path, geometry, dtype, and bytes;
3. rehashes the complete `.npy` before returning a read-only memmap of the shared
   entry (no per-run copy).

Miss population is single-flight under `flock`. Scorer bytes and receipt are each
written to `.partial`, fsynced, then atomically renamed. Incomplete, stale,
noncanonical, or SHA-mismatched entries refuse; the pre-existing per-run packet,
raw, and scorer-partial stale-scratch refusal remains on the miss path. The receipt
certifies packet/raw scratch as deterministically rebuildable before success-only
cleanup.

## Evidence and tests

- Exact M1 input: archive SHA `149fefd097c1fa85c4afb6cb2d8ab20311035d7ba8063f1e72137b843a9b89f3`;
  decoder SHA `4b54d512565f7275c53f697a931dd087222a36a69495b6e536a6b65dede36224`.
- C1 receipts consumed but not over-generalized: serial
  `acb1f6c3d4991b57591b21f619294e285f4acc3dd935ef22ff47420d9f544672`,
  parallel-1 `3a4eed82f43711ab9105f73e422f8a25b613548d05c3c6385d4afd8a1a586822`,
  parallel-2 `d8ae7dd0a2c15b223e77f4c3d6b0566894b25718e779cbc5b19660975af84d21`.
  Those receipts share raw SHA `31d77be9ab9f00e9f814542368396a35ffa119a32571e701636d4747540e255b`
  but bind production receiver SHA
  `6e12ba86ad88dfb178535009f6a1f1c13f799e81ca62e0b5a384ee6b9563f737`,
  not the M1 decoder above.
- Canonical measurement receipt:
  `.omx/research/decode_parallel_cache_fix_20260721T013658Z.json`.
- Regression surface covers worker env/default resolution, shared miss then hit
  without a second decoder call, stale canonical receipt refusal, serial-versus-
  parallel tiny materialization equality, and batched-versus-serial exact factor-2
  projection equality.

## Triality and pointer delta

- DSL: no new flag. Existing environment override remains an operational decoder
  control; the default changes only when it is absent.
- DAG: `decode_parallel_cache_fix_DAG_FEED_20260721T013658Z.md` wires archive and
  decoder content identities to one cache node and all materializer consumers.
- Equations: exact separable factor-2 integer numerator operator is unchanged;
  batching changes evaluation order only across independent planes, not within a
  numerator reduction. Serial equality is measured above.
- Pointer delta: none. This landing changes preprocessing wall time and custody,
  not witness bytes, scorer output, or frontier authority.

`verdict_scope`: n16 exact M1 decoder and projection identity plus local timing;
shared-cache code/test correctness. No full-n600 timing, live-run, contest-axis,
score, efficacy, or promotion verdict.
