# G55 — public typed selected-preimage codec closure

## Status

`IMPLEMENTED SHELL / EMPIRICAL CANDIDATE CLOSURE BLOCKED`.

Pointer delta is **UNMOVED**. The 2026-07-26 G52 yuv420p payload is
scorer-catastrophic and was derived by reading historical C1 planes. It is not a
production operand and cannot be promoted or used to establish this closure.

## Production input gate

G55 accepts only the exact G52 aggregate schema
`taskspace_fresh_selected_plane_codec_aggregate.v1`, recursively backed by the
exact G51 aggregate schema
`tac.taskspace_fresh_scorer_plane_aggregate.v1`. The G51 loader reopens all five
immutable 120-pair stages. The G52 aggregate and counted bundle must:

1. derive all 600 scorer-native `Y1` and `Y0|Y1` selected-preimage planes
   directly from sealed `gt_f0` / `gt_f1` caches through the generic exact
   `DisjointResizeOperator`;
2. binds fresh batch16 Seg-target custody and sealed source-frame custody;
3. binds compiler source, config, input, and a complete ordered set of preserved
   stage checkpoints;
4. declares candidate lineage allowed, fresh current scorer-plane compilation,
   generic V10 realization, and zero historical payload reuse; and
5. carries only newly encoded stream operands in the counted bundle.

Equal output hashes do not imply reuse. Derivation identity is authoritative.
The strict G51 loader rejects any forbidden historical producer/input lineage.
G55 additionally requires the G52 `DIRECT_TASK_LAYERED` truth fields, all five
external encoder checkpoint SHAs, the final whole-population recode receipt,
and the exact G51 outer receipt identity.

Cached historical `gt_poses` are advisory only and are not required as fresh
compiler custody. Pose authority arrives from the final upstream batch16
evaluation of the exact public raw bytes.

This is the truthful `DIRECT_TASK_LAYERED` architecture. It does not claim V15
composition, consume V15 program outputs, or embed V15 bytes. The semantic
object is the freshly derived scorer-native plane pair, followed by its counted
codec and generic V10 realization.

## Counted archive ABI

`archive.zip` contains canonical `manifest.json` and only declared video-derived
streams. The current G52 adapter carries two whole-population packed-RGB
streams: `y1_base` and `y0_given_y1_q2`. The manifest types every member by
exact bytes/SHA, actual codec name, actual native pixel format, public decoded
pixel format, deterministic RGB conversion path, semantic channels, frame
count, and decoded parse-back SHA. Requested encoder/pixel format metadata is
not decode authority.

Each chronological chunk has typed `base` and `enhancement` layers. A layer is
either:

- one packed `rgb24` decode; the coded pixel format is deliberately unconstrained
  and typed, so AV1 `yuv444p`, FFV1 RGB, or another verified codec is legal; or
- three independent `gray` streams in exact R/G/B semantic order.

Chunking is a resumability table, not representation semantics. Any positive
number of chunks forming one exact contiguous 0..600 cover is accepted,
including one long-GOP base stream and one long-GOP enhancement stream.

## Public runtime closure

The public runtime imports only Python standard-library modules, NumPy, and the
officially locked PyAV package. No `ffmpeg` or `ffprobe` shell executable is
required. G52 and G55 bind the exact `av` package pin derived from
`upstream/uv.lock` (currently 17.0.0), one decoder thread, and the exact decoded
parse-back SHA. Codec, native coded pixel format, and 512×384 geometry are typed
by PyAV. `libx264rgb`/`gbrp` is decoded by direct native G,B,R plane extraction
and RGB reorder, without a libswscale color conversion. Other admitted streams
use the explicitly typed `VideoFrame.to_ndarray(rgb24)` path. The generic layer
inverse is:

`Y0 = clip(Y1 + 2 * (enhancement - 128), 0, 255)`.

The exact factor-2 disjoint half-pixel support fill emits 1200 ordered
874×1164×3 uint8 frames. G55 independently reconstructs `Y0`/`Y1`, then derives
the raw hash in pair-ascending `Y0,Y1` order through the shared authoritative
`realize_factor2_uint8_scorer_plane` helper. It refuses unless its independent
plane and raw hashes equal G52. The generic public implementation is unit-proven
equal to that helper.

Each chunk is an atomic, hash-reopened stage checkpoint. Final raw assembly is
atomic. Resume invocations are explicitly labelled and remain useful for crash
recovery, but they do not satisfy authority. The double-inflate authority gate
requires two distinct output-root identities, both absent/empty at entry, with
every chunk freshly decoded, zero checkpoint reuse, and fresh final assembly.
Only stable fields (manifest/raw bytes and SHA, counts, PyAV identity, and stage
contents) are compared across the two receipts. Storage preflight requires stage
plus final capacity and 1 GiB headroom. Atomic scratch is success-cleaned; stage
receipts remain durable.

## Promotion gate

The builder first emits `archive.preview.zip`. Its explicit
`--stage-exact-eval` action copies those exact bytes to the evaluator-required
name `archive.zip` and seals
`taskspace_layered_public_exact_eval_staging_receipt.v1`. That receipt is
research-only, score-claim false, promotion-ineligible, pointer-unmoved, and
candidate-lineage-capable: the filename is staging, not promotion. This is
necessary because `upstream/evaluate.sh` requires `archive.zip` before exact
authority can exist.

Promotion only validates the already-staged bytes after a separately sealed
`taskspace_layered_public_auth_receipt.v1` binds:

- the exact preview archive SHA and exact public raw SHA;
- two byte-identical fresh inflates from distinct clean roots;
- `upstream/evaluate.py`, all 600 pairs, contest-CPU or contest-CUDA;
- candidate lineage allowed; and
- exact score strictly below the live canonical pointer's recomputed
  `effective_frontier`.

The builder and promoter each load and reverify
`.omx/state/canonical_frontier_pointer.json`, bind its exact SHA and selection
rule, and refuse if it is stale or changes during the operation. No target score
is hardcoded.

G51 is present and recursively reopens as five exact 120-pair stages. The fresh
official-PyAV G52 x264rgb bundle is also sealed at 184,052 bytes and passes the
G55 recursive lineage gate, with expected public raw SHA
`ccddfd9ab3606ed7a1b8e6bc0d2213028e9408f0dc082c1cba62587118a2ca3d`.
The G55 build, two distinct-clean-root inflates, and upstream exact evaluation
have not yet run. Therefore there is still no public exact row and no frontier
progress.
