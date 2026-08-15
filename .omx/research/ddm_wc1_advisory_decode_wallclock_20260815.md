# DDM WC1 advisory decode wall-clock

Date: 2026-08-15  
Disposition: **PREFIX IDENTITY PROVEN; FULL N600 QUEUED BEHIND A HOST MEMORY-PRESSURE BLOCKER**  
Axis: `[M5-CPU scorer-free advisory decode]`

## Finding

The default-off advisory fast path reproduces the retained hv1 bytes on real n4 and n32
prefixes with four independent 4-thread render workers and the F26R native token decoder. The
n32 run retained and matched 6,291,456 token bytes and 195,328,512 raw bytes exactly. The full
n600 timing, scalar twin, and micro-edit cache parity are **NOT MEASURED**: the required watched
detached n600 process tree was placed under reversible `SIGSTOP` by the canonical memory governor
four seconds after launch, before any token, render, or raw payload existed. At the blocker
snapshot, available memory was 40.031 GiB against the active 64 GiB safety floor. The canonical
wrapper terminated the still-paused tree at 1,200.706 s with `rc=124`; this terminal timeout does
not convert the pre-compute pressure block into a decoder verdict.

No admission receipt exists. The shipping packet, PQ1 generation, upstream tree, eval mirror, and
live MP2 run directories were not modified. MP2 therefore retains its slow path; it can enable WC1
only after the missing full identity and cache gates produce a complete, source-hash-bound
`ADMISSION_GATE.json`.

## Measured table

All after rows are prefix-scope measurements and are not n600 timing claims.

| Stage | Starting n600 receipt | n4 prefix | n32 prefix | Full n600 after |
|---|---:|---:|---:|---:|
| archive setup | 2.6526 s | 2.3825 s | 2.3526 s | **NOT MEASURED** |
| token stage | 642.1387 s | 1.3272 s | 8.0948 s | **NOT MEASURED** |
| neural render, retained copy included | 1,179.5858 s | 3.7770 s | 18.9170 s | **NOT MEASURED** |
| selector and final I/O | 79.6775 s | 0.3014 s | 2.4798 s | **NOT MEASURED** |
| internal total | 1,905.4434 s | 7.8053 s | 31.9175 s | **NOT MEASURED** |
| subprocess wall | 1,907.2 s external baseline | 8.4980 s | 32.6311 s | **NOT MEASURED** |

The n32 worker instrument was four processes, four torch threads per process, interop 1, spawned
processes, canonical disjoint pair ranges, and exact 874x1164 RGB output. The maximum measured
worker RSS was 1,498,185,728 B. Live admission derived four workers from 18 logical CPUs and
47,357,788,160 available bytes; no worker count is latched in code.

## Identity receipts and retained payloads

- n4 result:
  `/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/runs/prefix_n4_optimized_r2/result.json`,
  50,012 B, SHA-256 `a7f7e46342b66f1ad35279924ebd91adbf22d2b2f499fff3f97a5ac6fffeb1d3`.
  Token payload: 786,432 B, SHA-256
  `4a5047eeba814a3db2dbfccfce5a45fa5aee8282e85170d09500bdc322bda78a`.
  Raw and retained render-stage payload: 24,416,064 B each, SHA-256
  `6e20f75ed08dd8b7274db6e1b9f4e28e259da961419262a80f20bddec06d3a9e`.
- n32 result:
  `/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/runs/prefix_n32_optimized/result.json`,
  57,743 B, SHA-256 `276112724b01468df053362a8115fd8a9177370410f2f68816bbcd7ed194d2a5`.
  Token payload: 6,291,456 B, SHA-256
  `870cc4d3d4d4d10aa4bad281b09bc5a5397fead4f3863f2afe226aff13f959f7`.
  Raw and retained render-stage payload: 195,328,512 B each, SHA-256
  `bd5fc43b2230e5700daa791577a5513402946c9792ee08bdf347ebc9eee11ebc`.
- Full-attempt blocker:
  `/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/receipts/FULL_N600_BLOCKED.json`.
  It binds the watched launch, terminal done receipt, exact governor pause event, current
  pressure/floor facts, the two prefix results, the absence of materialized decode payloads, and
  the fire order.

The canonical MP2 base runtime pin is the charter's full
`4718834fe2e589f4be998061a8d9cba552ba4814b584e0efcc77a41aa2cb6680`.
The older PQ1 base copy is `2da70653...`; it is not the canonical MP2 launch surface. The lifted
driver preserves MP2's real WANS1/SD1M/SM3R semantic variants before adding the default-off levers.

## Implemented surfaces

- `experiments/ddm_wc1_advisory_runtime.py` implements canonical-JSON, full-payload-rehashed,
  lock-protected, atomic token-cache entries and crash-resumable process-parallel rendering.
- `experiments/ddm_f26p_f26_inflate_cpu.py` adds explicit advisory flags for native tokens,
  content-addressed cache reuse, and parallel rendering. All default to the incumbent path.
- `experiments/ddm_wc1_advisory_decode_wallclock.py` copies source generations without mutating
  them, removes AppleDouble files before launch, retains every payload, emits frame manifests,
  uses per-run single-flight locking, writes watcher configs, and owns the full admission gate.
- `experiments/ddm_mp2_advisory_queue.py` treats a missing gate as slow-path operation, treats an
  invalid present gate as a blocker, and stages a private generation plus the admitted flags only
  from a complete gate whose source hashes still match.

LH2 is a typed non-consumer. `tools/fire_watched_continuation.py` composes RX2/WC2 HPAC training
continuations and never invokes `contest_auth_eval.py`, an F26 generation `inflate.sh`, or the F26
decoder. Adding WC1 flags there would not cause any real decode work and would be a fake consumer
wire. No LH2 code was changed.

## MEASURED, DERIVED, and NOT-MEASURED boundaries

- **MEASURED, INSTANCE scope:** n4 and n32 native-token plus four-worker render prefixes match the
  retained hv1 token and raw prefixes byte for byte. These validate the actual token, uint8,
  resize, selector, and canonical assembly mechanisms on those prefixes.
- **MEASURED, INSTANCE scope:** the governed n600 launch was paused at 0.2 GiB RSS under sustained
  WARN pressure before any decode payload materialized. This is a current-host launch blocker, not
  a decoder-family negative.
- **DERIVED, provisional:** linear scaling of the n32 subprocess wall is about 612 s. This makes
  the 700 s bar plausible, but fixed startup/copy costs and full-field scheduling make it
  non-authoritative. It is not an n600 timing row.
- **NOT MEASURED:** full hv1 raw SHA reproduction, full native token/logit/CDF/bit-position gates,
  forced-scalar parity, end-to-end n600 wall clock, a cache hit on a real micro-edit candidate, and
  cached-versus-fresh raw parity.
- **NOT MEASURED:** any SegNet, PoseNet, contest-CPU, or contest-CUDA score. The exact pointer did
  not move.

## RECALL EVIDENCE

The recall pass searched `.omx/research/`, `.omx/state/`, `docs/`, `tools/`, `src/tac/`, the
canonical equation registry, research index, full `sub015_DAG_*` FEED surface, task/lane stores,
and Git history for `F26P|F26Q|F26R|decode wallclock|INFLATE_WORKERS|content-addressed cache|HPAC`.

Beyond the charter seeds, it found:

- `.omx/research/decode_parallel_cache_fix_20260721T013658Z.md` (#592), which already defined the
  exact atomic content-addressed SSD-cache contract. WC1 reused its canonical JSON, full payload
  rehash, lock, and partial-to-rename discipline instead of creating a weaker cache.
- `verdict_parallel_workers_speedup_v1`, a measured scorer-forward parallelism equation. It
  justified live resource derivation as a precedent but did not transfer its 5.686x number to the
  F26 renderer.
- `.omx/research/ddm_ua2_upstream_defenses_and_budget_surface_20260731.md`, which kept decoder
  timing separate from score authority and reinforced the 30-minute runtime boundary.
- The existing F26 token checkpoint contract in the canonical runtime and the #214/#592 process
  parallel precedents. These changed the implementation from a monolithic pool call to retained
  token and per-chunk checkpoints.
- The actual LH2 composer and its memo, which closed the proposed LH2 wire as a mechanism mismatch
  and left MP2 as the real consumer.

## Landing custody

All seven scoped source/memo artifacts are validated but **UNCOMMITTED**. The required serializer
failed while staging with `error: unable to create temporary file: Operation not permitted`; Git
HEAD remains `bef8e76dbc1eefe6c998f717499fc5a61443bb51` and the index remains empty. This is a
managed Git-write boundary, not a code-review failure. No direct `git commit` bypass was attempted.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN. Consumer store: `/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/receipts/`. Fire trigger: three consecutive `.omx/state/memory_blackbox.jsonl` rows report `pressure=normal` and `available_gib>=64`, and the prior `wc1_base_optimized_n600.done` receipt is terminal. Action: launch a new watched base optimized n600 run ID, then serialize the forced-scalar row and the fresh/cached `score_gated_film_row_prune_keep75_minus_keep87` rows; run `finalize` only after all four identity gates pass.**
- **Disposition: READY-TO-LAND. Owner: MAIN. Consumer store: Git history for `/Users/adpena/Projects/pact`. Fire trigger: the managed environment permits Git index/object writes. Action: recompute all seven post-edit SHA-256 values and rerun `tools/subagent_commit_serializer.py` with the same explicit file list; do not use direct Git commands or absorb unrelated dirty work.**

## LIVE-HYPOTHESES

- Full n600 will clear the 700 s bar because n32's real four-worker/native path took 32.631 s and
  its linear wall projection is about 612 s; this remains plausible, not proven, because full-field
  copying and scheduling may scale differently.
- The MP2 `score_gated_film_row_prune_keep75_minus_keep87` candidate will hit the base token cache
  because its token stream, HPAC model, and correction table are inherited while its semantic
  section changes; the cache key binds those actual section hashes rather than the outer archive.
- Measured worker RSS plus 25 percent will continue to admit four workers on this 18-core host once
  pressure returns to normal because the observed maximum was about 1.50 GB per worker.

## DEAD-ENDS

- Treating the n4/n32 rows as a full-field verdict is closed: both are prefix populations and the
  charter requires a full raw SHA plus full decoder digests.
- Launching another detached n600 row under current WARN pressure is closed: the governor paused
  the first tree before decode, and its 64 GiB floor remains above current availability.
- Manually overriding the governor is closed: its reversible pause is the canonical host safety
  mechanism, and the charter supplied no operator override authority.
- Wiring WC1 into LH2 is closed: LH2 does not call the advisory decoder, so flags there have no real
  consumer.
- Concurrent producers for one WC1 run ID are closed: an early duplicate prefix preparation race
  exposed the missing seam; the runner now holds a nonblocking single-flight lock for the whole
  run, and the incomplete duplicate preparation tree remains retained rather than silently deleted.

Own-vehicle frontier remains `S=0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]`.
