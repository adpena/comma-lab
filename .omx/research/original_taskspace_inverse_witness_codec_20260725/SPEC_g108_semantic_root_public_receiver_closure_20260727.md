# G108 — SemanticRootY1V1 public receiver closure

Date: 2026-07-27  
Lane: `lane_g108_semantic_root_public_receiver_closure_20260727`  
Authority: receiver/package implementation only; research-only; no score or source-closure claim

## Result

G108 closes the public, scorer-free decode boundary for the committed G103
`SemanticRootY1V1` wire.  A clean extracted archive plus a separately copied
public runtime now emits exactly one `0.raw` containing 1,200 chronological
`Y0,Y1` frames at `874×1164×3 uint8`, or 3,662,409,600 bytes.

This is not a candidate.  The current frame-0 owner is explicitly
`duplicate_y1_research_only_v1`; G107/final conditional Y0 must replace it
after final Y1 freeze.  The fresh source compiler, source lineage, G17
whole-object placement, and exact post-R Seg/Pose closure remain owed.

## Counted archive grammar

The archive has exactly one member:

| member | compression | ownership |
|---|---:|---|
| `semantic_root_y1_v1.bin` | ZIP DEFLATE level 9 | counted video-specific packet |

No dispatch manifest or proof metadata is counted.  The packet is already
self-describing through its strict magic, version, architecture, typed section
headers, exact EOF, and CRC32.  Adding another manifest would duplicate
information and spend rate.  The archive parser refuses extra members,
encryption, symlinks, non-DEFLATE members, packet mutation, and malformed G103
wire.

Measured on the nontrivial committed G103 parity fixture:

- packet: 2,351 bytes,
  SHA-256 `2109ba56382f535c656dbc3b771a413a67e7403e68b44fb808e649b7e57dca76`;
- whole archive: 685 bytes,
  SHA-256 `11acdaac0bee81be91c808b7b20d836da4daa114510ede246d653918bb70523b`;
- whole archive, not packet-only bytes, is the rate authority.

The small fixture is only a deterministic format/rate measurement.  It is not
source-derived and has no score authority.

## Public runtime

`submissions/robust_current/g108_semantic_root_receiver` is a standalone
Python/NumPy runtime.  It imports neither `tac` nor the private repository and
does not load SegNet, PoseNet, source video, or evaluator state.  `inflate.sh`
implements the exact three-argument public contract:

```text
inflate.sh ARCHIVE_DIR OUTPUT_DIR VIDEO_NAMES_FILE
```

The runtime:

1. verifies the extracted root has exactly the counted packet;
2. loads public semantic plugins and requires exactly one to accept the
   packet;
3. strictly parses the G103 wire, including all tensor ABI, topology, temporal
   Rice, RGB-gauge, quotient, CRC, cardinality, and dense/foreign-payload
   exclusions;
4. renders scorer-grid Y1 through integer NumPy semantics matching G103;
5. realizes the scorer plane through the exact disjoint V10 factor-2 camera
   preimage;
6. invokes exactly one frame-0 owner;
7. streams `Y0,Y1` in upstream order to an atomic temporary file, fsyncs it,
   checks the exact byte count, and renames it to `0.raw`;
8. deletes incomplete temporary output on failure and refuses insufficient
   disk headroom.

Runtime dependencies are Python 3, NumPy, and the standard library, all
available in the contest environment.

## Variant ABI

Semantic variants are additive public plugin files and do not duplicate or
modify the dispatcher:

```text
VARIANT_ID: str
accepts_packet(packet: bytes) -> bool
parse_packet(packet: bytes) -> parsed
render_scorer_y1(parsed, pair_id: int) -> uint8[384,512,3]
```

The exact V9 adapter reserves:

```text
tac.semantic_root_y1.v9_hosc_dual_head_odd_y1.v1
```

Its public plugin can select the typed `SV9Y1V1` magic and reuse the same
archive extraction, output order, V10 realization, storage preflight, atomic
write, and frame-0 dispatch.  Exactly-one matching semantics prevents silent
fallback or ambiguous interpretation.

Frame-0 variants independently implement:

```text
VARIANT_ID: str
prepare(packet: bytes, archive_root: Path) -> state
render_camera_y0(state, pair_id, scorer_y1, camera_y1) -> uint8[874,1164,3]
```

The current duplicate-Y1 implementation is intentionally named research-only
and is not acceptable for candidate promotion.

## G102 interface

`taskspace_g108_semantic_root_public_product_v1.py` provides all seven
callables required by
`tac.semantic_root_y1.compiler_receiver.v2`.

Receiver/archive calls are real.  Producer-side calls raise
`G108_FRESH_SOURCE_COMPILER_LINEAGE_AND_EXACT_POST_R_CLOSURE_OWED`.
The capability dictionary has the exact G102 key set and exact public runtime
tree hash, but truthfully reports:

- `own_lineage = false`;
- `exact_post_r_seg_closure = false`;
- `exact_post_r_pose_closure = false`.

Therefore G102 still refuses S01 at its capability gate.  That refusal is the
correct remaining source/candidate boundary, not a runtime gap.

## Evidence

Standalone parity against committed G103 and V10 is byte-exact for pairs
`0,1,137,599`.

The opt-in full-n600 proof ran actual `inflate.sh` twice from independent clean
extracted roots under the G102 external-repository import guard:

- synthetic wire-valid full-n600 packet: 612 bytes,
  SHA-256 `a25c30d45d0c9049f40c4c07568fe3e2bc78482a70a337cc956324ed2433b954`;
- exact archive: 354 bytes,
  SHA-256 `eabfb3b29367e4beba8e4979603e9ba2fd15616761177c3d88bd66fee089128e`;
- public runtime tree:
  SHA-256 `23dd52fbd9035f236d6585ff7ca7e67a7062788ac1ff4b9b6fa9176eabc84ce5`;
- each output: 3,662,409,600 bytes;
- both output SHA-256:
  `8d6a921953885b1d85e15a0c9f1cb00a81e38f2f46dae91dc83983007faa7fb3`;
- double-decode equality: true;
- elapsed for two decodes plus two full SHA passes: 188.36 seconds;
- SSD scratch: removed in the success/failure `finally` path.

This full proof establishes receiver geometry, ordering, clean-root
independence, and determinism.  Its source-independent invariant packet is not
a distortion, score, candidate, or semantic-quality result.

## Remaining blockers

1. G105/fresh producer must emit a source-custodied own-lineage full-n600
   packet.
2. Exact post-R Seg and Pose closure must be measured on that same packet and
   archive.
3. G17 whole-object ownership and source-lineage evidence must close.
4. G107/final conditional Y0 must replace the research-only duplicate frame.
5. The same selected archive must pass public double decode and full
   `upstream/evaluate.py` batch-16 authority.

The canonical frontier pointer remains the external 0.172 row.  G108 does not
move it.
