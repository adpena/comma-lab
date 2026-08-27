# QBW1 packet schema v1 — frozen first-rung wire contract

Status: **FROZEN BEFORE PAYLOAD**.  This document and its parser are the stage-00/01 fire
precondition for `ddm_qbw1_builder_first_rung`.  Any wire change requires a new versioned schema
memo before another candidate is encoded.

## Rule-118 and object boundary

`QBW1` carries video-derived quotient coordinates.  The generic crack integrator, chain verifier,
soft-cell rasterizer, Lane-event rasterizer, varint routines, and raw-DEFLATE receiver are free
algorithmic code.  The shared compression dictionary, every reset record, every renderer/model
weight, terminal pose state, residual, and archive/framing byte are counted.  No scorer weight,
GT field, per-pair lookup table, or hard-coded video fact is receiver code.

The decoded rate-bearing object has two coupled layers:

1. A four-label base field in canonical label order `{Road=0, Undrivable=2, Movable=3,
   MyCar=4}`.  Lane is merged into Road only for this base layer.  Oriented inter-class cracks are
   decomposed into deterministic maximal chains.  Chain births use signed deltas between canonical
   lattice-vertex ranks; chain bodies use cardinal steps and left/right labels; chain termination is
   the declared step count.  Region seed labels are attached to cells in the integrator's canonical
   row-major discovery order.
2. Lane dash events are native coordinates of the Road graph.  Each event names a canonical
   Road-boundary edge by a delta in graph order, then stores signed tangent/normal offsets, two
   half-extents, and a quantized angle.  The receiver obtains its origin and local basis from the
   decoded Road graph.  Lane is therefore not a standalone bitmap or separately addressed patch.

The categorical Lane raster is an observability product.  Stage 03 must feed the crack/cell and Lane
event coordinates to separate boundary/interior renderer branches from birth; it may not convert
the object into a fixed RGB palette.

## Address discipline

The only pair identifier is the reset-record header.  No event carries a `(pair,x,y)` tuple.  Crack
positions arise from a chain birth followed by causal cardinal steps.  Lane positions arise from a
Road-edge traversal rank followed by signed local offsets.  Region positions arise from generic
integration and canonical cell discovery.  The rung-1 format has no residual-address section.

## Shared model (`QBM1`)

All integers are big-endian unless they are varints.

| bytes | field |
|---:|---|
| 4 | ASCII magic `QBM1` |
| 1 | schema version `1` |
| 1 | coder id `1` = independently reset raw DEFLATE |
| 1 | level `9` |
| 1 | reserved, zero |
| 4 | dictionary length, unsigned |
| 4 | CRC32 of dictionary bytes |
| N | exact preset dictionary bytes |

The complete serialized `QBM1` bytes are `B_shared`.  A new zlib compressor and decompressor are
created for every counted section with `wbits=-15`, level 9, and the exact selected dictionary.
There is no cross-record or cross-section coder state.

Stage 01 may race dictionary capacities `{0,4096,16384,32768}` only.  Each candidate model and every
candidate coded section is retained.  Selection minimizes the measured `QBM1` bytes plus exact n32
record bytes.  Candidate dictionaries are deterministic: concatenate uncompressed n32 sections in
ascending pair-id then section-id order and retain the final `capacity` bytes.  The winning model is
then immutable for stage 02.

## Reset record (`QBR1`)

| bytes | field |
|---:|---|
| 4 | ASCII magic `QBR1` |
| 1 | schema version `1` |
| 1 | flags, exactly `1` (`RESET_SNAPSHOT`) |
| 2 | section count, exactly `3` |
| 2 | pair id, 0–599 |
| 2 | height, exactly 384 in this rung |
| 2 | width, exactly 512 in this rung |
| 2 | reserved, zero |
| 32 | SHA-256 of the complete selected `QBM1` bytes |
| variable | three counted section envelopes in ascending section-id order |

Each section envelope is `section_id:u8, reserved:u8=0, raw_len:u32, coded_len:u32,
raw_crc32:u32, coded_bytes[coded_len]`.  The required sections are:

- `1 BASE_CRACK_CHAINS`: unsigned chain count; per chain a signed birth-rank delta, unsigned step
  count, then one byte per step.  Step bits are `direction[1:0]`, `left_label[3:2]`,
  `right_label[5:4]`, with high bits zero.  Directions are `N=0,E=1,S=2,W=3` on the image lattice;
  left/right are geometric sides of that directed step.  Label codes map in order to `{0,2,3,4}`.
- `2 REGION_SEEDS`: unsigned cell count followed by packed two-bit base-label codes in canonical
  cell order, earliest cell in the low two bits of each byte.
- `3 LANE_DASH_EVENTS`: unsigned event count; events sorted by Road-edge anchor and stable geometry.
  Each event is anchor delta (unsigned varint), tangent offset q4 (signed varint), inward-normal
  offset q4 (signed varint), major half-extent q4 (unsigned varint), minor half-extent q4 (unsigned
  varint), and angle `round(turns*256) mod 256` (`u8`).

Unsigned varints are minimal LEB128.  Signed values use zigzag then minimal LEB128.  Non-minimal
encodings, overflow, duplicate crack edges, illegal labels/directions, missing/duplicate sections,
length drift, CRC drift, model-SHA drift, reserved-bit use, and trailing bytes are refusals.

## Receiver invariants and mutation contract

The receiver expands chain steps to the exact crack-edge set, integrates cells without using seed
labels, assigns seeds in canonical discovery order, then re-derives all base-field interfaces.  The
re-derived interface set must equal the decoded crack set exactly; this is the closed-chain
consistency verdict.  Lane events are rasterized only from decoded Road-edge geometry and event
coordinates.

Parseback passes only when decoded chains, seed codes, Lane events, integrated base cells, and
renderer-input arrays equal the encoder-declared objects.  The primary and deterministic repeat must
be byte-identical.  Stage 02 flips one bit in the shared model, record framing, and every coded
section of every selected record; each mutation must either be refused or change the declared
decoded object.  The reference parser refuses all checksum-bearing payload mutations.

## Accounting and non-authority boundary

For selected pair `i`, `b_i` is the exact `QBR1` file size.  `B_shared` is the exact `QBM1` size.
The Horvitz–Thompson projection is the no2 §5 formula.  Stage 02 additionally reports exact source
interface length and bytes per interface.  The inherited `53,076 B` renderer/pose/framing envelope
is a projection only until stages 03–05 materialize and recount those objects.

All n32 rows are `[macOS-CPU scorer-free advisory]`, `score_claim=false`, and cannot promote, rank,
kill the family, or move a frontier.
