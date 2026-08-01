# G48 — V15/MS1 coordinate compatibility: exact outer seam, no native state cast

Date: 2026-07-26  
Lane: `lane_g48_v15_ms1_coordinate_compatibility_20260726`  
Authority: read-only compatibility audit; no heavy launch, eval row, candidate, promotion, or pointer movement

## Forest-level verdict

The promising composition is real, but the direct interpretation was wrong.
V15 and MS1/C1 meet exactly at the final camera-byte coordinate:
`(600,2,874,1164,3) uint8` in pair/frame/row/column/channel order. They do not
share a latent or a native compact state. MS1 did not manufacture a new
selected-preimage program; every local-CVP proposal lost, so it selected the
unchanged C1 camera raw. Therefore no V15-field-to-MS1-field cast exists.

The missing frontier-moving object is narrower and more useful:

> a fresh counted V15-equivalent semantic program plus a factorized
> selected-preimage correction, including an independent two-frame pose gauge,
> decoded through the exact V10 factor-2 receiver.

This is not a request for another seam container. The exact semantic-to-C1
container is already implemented by `c0b_semantic_quotient.py`. The open
problem is making that quotient sparse/factorized and using the 133,941-byte
direct V15 base rather than the already-over-budget 339,094-byte E1 runtime
base.

## Exact custody and types

### V15 compact grammar

- Archive: 133,941 bytes, SHA-256
  `759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df`.
- Receipt: 169,900 bytes, SHA-256
  `5ed6f830b3749a51e0d300a9104fda9a77e86bbeb3b81428a20e1ec0d3dcfcb8`.
- Measured n600 row: `d_seg=0.027470296224`,
  `d_pose=163.061327281443`, batch 16.
- Receiver output contract:
  `(N,2,874,1164,3) uint8`.
- Native state: semantic role masks, lane and movable worldsheet grammar,
  shared row-band RGB templates, and Pose6 codes. The realization paints the
  same semantic/template state into both frames; it does not own two arbitrary
  independent RGB planes.
- The sealed producer receiver source SHA was
  `1a3622a64b307c8b5a6b1987f8bdb86d9df441d0159a3a420bd9c283d41f0824`.
  The current dirty source SHA is
  `3e1f69bb168da7b42a55ac4ba1a573c0c291a559f6622b40cffa71c219d76d48`.
  Thus the historical archive remains sealed evidence, but a fresh candidate
  requires fresh receiver/source custody and parse-back.

### MS1 is unchanged C1, not a new latent

- MS1 ingest receipt SHA-256:
  `1b7063a44574b0839ede08c807f348ad417be0492ac32d68634b124b9c2b1e97`.
- Immutable historical receipt SHA-256:
  `546a7fddb0225edb15b2254ab73e362758b7b0f244e4ff39cb7bfef25f779098`.
- Selected raw: 3,662,409,600 bytes, SHA-256
  `31d77be9ab9f00e9f814542368396a35ffa119a32571e701636d4747540e255b`.
- Exact shape inferred from custody:
  `(600,2,874,1164,3) uint8`.
- `B_previous_frame`: zero changed pairs.
- `C_xi_motion`: zero changed pairs.
- Local member selection: zero proposal wins under both conditioning
  expansions.
- Frozen-scorer diagnostic on the unchanged member:
  `d_seg=0.0001519690619574653`,
  `d_pose=0.00010184327939026322`; all 600 Seg argmax outputs and all 600
  Pose6 outputs were identical to the canonical input member.

The strong distortion is real. It belongs to the dense C1 witness, not to an
MS1-produced compact representation.

### C1 exact two-plane state

- Archive: 409,526,925 bytes, SHA-256
  `e4cd154f79a30e2b1d759af0d26e54444d22807f81700565e475392eae064f42`.
- Predictor payload: 409,525,473 bytes, SHA-256
  `b3a792e1d838673b9047b9bd7dea93f0946a57871d484aec017650b7c1b3846e`.
- Prepare receipt SHA-256:
  `f5a0334002b0c212a994c1bc8135da449a3be247eec9b75e9b7830a92ed54183`.
- Receiver receipt SHA-256:
  `a8096119ec8a009d1bab6e42fbee55a81635c50a3d8d96acfe950b1754267e1f`.
- Native state:
  two independent `(600,384,512,3) uint8` scorer-plane arrays, `Y0` and
  `Y1`, followed by exact deterministic V10 factor-2 camera realization.

## Compatibility matrix

| Coordinate | Verdict | Meaning |
|---|---|---|
| Pair identity/order | exact | Both cover source pairs 0 through 599 |
| Camera raw | exact shared | Both can inhabit `(600,2,874,1164,3) uint8` |
| Scorer-plane shape | shape only | Both refer to 384x512 RGB cells, but own different state semantics |
| Compact latent | incompatible | V15 grammar has no arbitrary independent `Y0/Y1`; MS1 emitted no compact latent |
| Pose/gauge | incompatible | V15 Pose6/shared photometry is not C1's independent pose-legible pair |
| Lossless field cast | falsified | A real render/factor/quotient transform is required |

## The exact bridge already under our nose

The tracked C0B substrate is an honest exact seam:

- `src/tac/witness_dsl/c0b_semantic_quotient.py`, SHA-256
  `f3316c68b5a402be110a4fa722f0610a4386eef21d71bd42977ada5cf79ca29b`.
- `tools/build_c0b_semantic_quotient_archive.py`, SHA-256
  `88f3fd1165361c59f7bbf5cb38ace2a75f34753e454fd2f7bedd451e46edecca`.
- `src/tac/witness_dsl/v10_production_receiver.py`, SHA-256
  `84d4ce09e8611a185c6aa53c073a90cc5b6970793222a52f66c1f2672842cd47`.

It stores a counted semantic program, deterministically renders base planes,
applies exact LZMA-compressed bytewise-XOR quotient chunks, and feeds the
independent planes to V10. It is correctly labeled
`NONPROMOTABLE_DENSE_C1_QUOTIENT_SCIENTIFIC_SEAM_BASELINE`.

This settles the type question. Building another general bridge duplicates
finished work. The rate problem is:

1. replace the 339,094-byte E1 semantic packet with a fresh direct
   V15-equivalent packet at roughly 133,941 bytes;
2. represent only the selected-preimage difference that matters, using
   factorized/global/conditional fields rather than dense XOR;
3. admit residual bytes by exact coupled score value, including preservation
   of the independent frame-0/frame-1 pose gauge.

## Coupled score budget

At the measured MS1 distortion, canonical arithmetic gives:

- distortion-only contribution: `0.04710980004607969`;
- target `0.172` largest archive: 187,562 bytes;
- score at 187,562 bytes: `0.1719996370115804`;
- score at 187,563 bytes: `0.17200030287053353`;
- sub-0.15 largest archive: 154,522 bytes.

Hypothetical arithmetic only—never an eval or score claim:

| Base | Bytes | Score if MS1 distortion survived | Headroom to 0.172 | Headroom to 0.15 |
|---|---:|---:|---:|---:|
| direct V15 | 133,941 | 0.13629561408621643 | 53,621 B | 20,581 B |
| V15 + J2 seed | 134,211 | 0.13647539600355943 | 53,351 B | 20,311 B |

This is the crux: the direct V15 grammar and C1 distortion would comfortably
cross the frontier if their missing quotient/factor state fits in about 53 KB.
The frontier question is therefore an empirical extreme-factorization problem,
not a coordinate uncertainty.

## First executable non-toy adapter contract

Contract ID: `tac.taskspace_selected_preimage_program.v1`.

Encoder input:

- a freshly compiled, counted, own-lineage V15-equivalent semantic program;
- freshly recomputed n600 selected-preimage teacher planes;
- exact frozen scorer-recursive Seg argmax and Pose6 obligations;
- typed deterministic factorization and rate-allocation configuration.

Counted output:

- semantic base program;
- global/shared selected-preimage factors;
- per-pair coefficients/selectors/gauge state;
- only those residual/quotient bytes admitted by exact score-unit value.

Decode ABI:

`decode_selected_preimage_pair(program, pair_index) -> SelectedPreimagePair`

where `SelectedPreimagePair` owns independent `frame0` and `frame1` arrays,
each exactly `(384,512,3) uint8`. The generic decoder then applies the existing
exact V10 factor-2 realization to produce `(2,874,1164,3) uint8`.

Admission is full n600 only: byte-canonical parse-back, all counted sections
receiver-effective, public `inflate.sh` closure, realized-through-R exact
`d_seg/d_pose`, and finally `upstream/evaluate.py` on the exact archive bytes.

## Lawful boundary

Generic semantic rendering, factor synthesis, deterministic repair, and V10
integer realization may live in `inflate.py`. Fresh video-derived semantic
state, factor coefficients, selectors, gauge values, and residuals are counted
payload. Scorer weights, GT/target tables, hidden C1 planes, or historical MS1
arrays disguised as decoder code are forbidden. Historical MS1/C1 remains an
encoder-side teacher and factor-family oracle, not candidate payload.

## Durable audit

Machine-readable receipt:
`v15_ms1_coordinate_compatibility_receipt_20260726.json`

- receipt identity SHA-256:
  `1b72cff2c99775a944e6abf22434a66b08465a83da45d3d920263376d4980940`;
- complete file SHA-256:
  `3ad9192ae2fc2c99d7d2f33c7780a1edade4f7e3df443bc5705303be9299a0aa`.

The receipt strictly parses the entire small V15 ZIP envelope and nested
predictor manifest, checks every sealed receipt and linked payload identity,
verifies the live dynamic frontier selection rule, and computes the exact
integer byte ceilings. It intentionally does not live-rehash or load the
3.662 GB raw; that identity is transitively bound by the sealed MS1 and C1
receipts.

Pointer delta: unchanged. This audit removed a false composition and made the
next candidate-producing object explicit; it did not lower the score.
