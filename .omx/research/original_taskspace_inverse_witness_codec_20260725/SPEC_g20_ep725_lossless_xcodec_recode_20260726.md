# G20 specification — exact ep725 lossless cross-tensor recode

Date: 2026-07-26  
Lane: `ep725_lossless_xcodec_recode_20260726`  
Status: executable research-only recode; no score, candidate, promotion, or pointer claim

## Objective

Re-encode the exact original ep725 n600 `LVLS1` predictor without changing any
decoded quantized tensor.  The optimization variable is the complete counted
object, not an isolated Brotli section:

```text
base storage axes x code chronology x manifest bytes x outer DEFLATE profile
    -> exact archive.zip bytes
```

The source archive, member, and runtime are custody-bound.  Public archives,
weights, checkpoints, codebooks, targets, and scorer products are forbidden.
This is a reversible recoding of our own frozen payload using a receiver path
already present in its frozen runtime.

## Frozen source contract

- archive SHA-256: `149fefd097c1fa85c4afb6cb2d8ab20311035d7ba8063f1e72137b843a9b89f3`
- `0.bin` SHA-256: `f0c3e648f00f52e48c7be98997fb7dd57c2e5a607ed385846931af68f88cc78c`
- `inflate.py` SHA-256: `4b54d512565f7275c53f697a931dd087222a36a69495b6e536a6b65dede36224`
- strict four-section `LVLS1`: canonical JSON manifest, Brotli base, Brotli
  code, empty pose; no optional block and no pre-existing `xcodec`
- population/code shape: 600 pairs and `[1200,32]`

Any drift fails closed.  The source ZIP must have exactly one safe `0.bin`
member, exact EOF/CRC reopen semantics, and metadata that can reproduce the
source bytes under at least one searched DEFLATE level.

## Finite exact search

Let `M` be the ordered set of non-degenerate 2-D base tensors.  Exhaustively
measure every `2^|M|` storage-axis mask, both code modes, and the Python/zlib
default DEFLATE profile plus explicit levels 1 through 9.  The default profile
is a distinct exact byte program and must not be silently equated to numeric
level 6.  Base mode stores each selected matrix transposed; receiver
dequantization transposes it back.  Code mode 1 stores frame-separated
modulo-256 temporal differences and receiver cumulative sums reconstruct the
original signed-int8 rows.  Each point adds canonical manifest field
`"xcodec":{"p":[ordered tensor indices],"c":mode}` and is selected by the
actual final ZIP byte length.  Tie order is exact archive bytes, member bytes,
identity-before-transform, mask, code mode, DEFLATE level, then SHA-256.

The unchanged source archive is an explicit control and remains selected if no
transformed point strictly improves the exact archive price.

## Proof obligations

Before writing an artifact, the implementation must:

1. reproduce the exact source ZIP bytes under the recorded metadata/profile;
2. strict-parse the selected member with exact section consumption;
3. decode every base tensor and all 1,200 code rows from both objects;
4. prove array equality, shapes/dtypes, pose bytes, and a domain-separated
   whole-state digest; manifest equality is exact after removing only `xcodec`;
5. build the winning archive twice and prove deterministic byte identity;
6. reopen it as a one-member ZIP and verify exact member bytes and CRC; and
7. run the frozen runtime with `INFLATE_MAX_PAIRS=1` on source and recode,
   compare the two complete uint8 raw outputs, and auto-delete temporary raw
   data.

The bounded runtime replay proves the deployed parser/receiver consumes the
new spelling.  Full n600 replay and authoritative contest CPU/CUDA evaluation
remain owed.  Until those close, the durable archive name ends in
`.not_a_candidate.zip`, `score_claim=false`, and `promotion_eligible=false`.

## Full-stack wire-in

- sensitivity: functional delta is exactly zero at the full quantized-state
  surface; only exact rate moves;
- Pareto: the recode may dominate only the identical-state source object;
- bit allocator: expose the exact section/archive delta as a zero-distortion
  `MERGE_SHARE/REQUANTIZE_STORAGE` proposal, not nominal savings;
- autopilot: route the materialized same-runtime object to full replay before
  any score or candidate claim;
- continual learning: record local-section versus whole-archive savings so
  future actions price recompression interactions;
- disambiguator: the exhaustive final-ZIP search arbitrates all lawful masks,
  code modes, and compression levels.

This landing is the first substitutive-rate control for G17: information moves
between storage coordinates and entropy contexts while being counted exactly
once.  It does not establish that the ep725 representation itself is optimal.

## Materialized result

The reviewed materializer completed on 2026-07-26 and wrote the durable
research-only receipt at
`.omx/research/original_taskspace_inverse_witness_codec_20260725/ep725_lossless_xcodec_recode_20260726/receipt.json`.
The finite whole-object search selected base transpose indices `[0,8]`, code
mode `frame_delta_mod256`, and the default DEFLATE profile:

- exact archive: `83,838 -> 81,027` bytes (`-2,811` bytes);
- exact rate term: `-0.0018717295172264237` score units;
- exact member: `84,536 -> 81,738` bytes;
- source and selected full quantized-state digest:
  `5485d0d94c5c834e059837e74ae5320fe9d2b526604c47008a6bfdb74144adf6`;
- selected archive SHA-256:
  `8e9c7ba0fdd1fc0fdff696c639821d6e64a3110bb8744f47ae0ab3d287cd70d8`;
- bounded frozen-receiver output: complete two-frame uint8 equality.

This is structural/macOS-CPU receiver evidence, not an evaluator score.  The
rate delta is indivisible whole-object evidence: downstream composition must
re-run final archive coding and must not add `-2,811` bytes to another action's
nominal section savings.
