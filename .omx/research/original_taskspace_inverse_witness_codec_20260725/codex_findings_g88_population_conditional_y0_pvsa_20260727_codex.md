# G88 — Population conditional Y0 successor over exact PVSA Y1

Date: 2026-07-27  
Lane: `lane_g88_population_conditional_y0_pvsa_20260727`  
Status: new-only receiver/wire landing; research-only; no candidate, score, or
pointer claim

## Outcome first

G88 lands the missing population-shaped transition:

```text
exact PVSA1 member
  -> exact corrected uint8 Y1
  -> counted Y0 | exact Y1
  -> chronological uint8 (Y0,Y1)
```

The exact successor member carries the complete G82 PVSA member once and one
strict G88 operand. The operand has a closed population default
(`PASS_P0` or `XIP2_SE3_FRAME0_WARP`) plus sparse typed overrides
(`PASS_P0`, `COPY_CONDITIONAL_Y1`, `ROLE_TRANSLATE_RGB`, or XIP2). Therefore a
future n600 XIP2 trajectory pays a fixed 160-byte operand frame plus its exact
XIP2 payload; it does not repeat 600 identical 38-byte mode rows.

This is a real actuator seam, not a solved pose actuator. A source-custodied
fresh n600 SE3 xi trajectory does not exist in the inspected V15/G82 lineage.
The older counted XIP2 packet is foreign-keyed to a bounded PASS-G source, and
V15 Pose6 values cannot be relabeled as xi. G88 therefore fails closed on:

```text
G88_FRESH_V15_N600_XIP2_TRAJECTORY_CUSTODY_OWED
```

The successor task is to compile a fresh n600 xi/XIP2 trajectory from our own
inverse/full-lattice teacher surfaces and bind it to exact member
`d50aac6e...`. Synthetic trajectories are not admissible evidence.

## Exact wire

The COPY pair-0 operand and successor parse/re-encode exactly under both outer
ZIP encodings:

```text
base PVSA member       133363 B  d50aac6eab8114c2...
G88 operand               198 B  f9170051a46ac71f...
successor member       133646 B  809481dd63acedbc...
member delta              283 B

outer STORE            133754 B  dfc90a84806547e2...
outer DEFLATE selected 129606 B  caee21878c9da3f9...
G85 archive baseline   129392 B
selected outer delta      214 B
```

Every operand binds:

- the exact embedded PVSA member SHA-256;
- the exact semantic-P SHA-256;
- the complete pair range `[0,600)`;
- its control-body byte count and SHA-256;
- canonical fp32 pitch;
- its exact XIP2 length and SHA-256, or the sole empty-XIP2 digest;
- CRC32 and exact EOF.

An active XIP2 program is parsed through `SE3XiTransportV2` as exact
`int16[600,6]` codes plus six positive fp32 scales. Raw Pose6 bytes, trailing
bytes, malformed XIP2, wrong shape, nonpositive scales, and digest mutation
are refused.

## Original XIP2 mechanism, new source binding

The global XIP2 mode composes the existing mechanics from:

- `taskspace_counted_xip2_chronological_a3.py`;
- `taskspace_chronological_a3_encoder.py`; and
- `warp_real_luma_frame0.py`.

It preserves the existing numeric reference
`NUMPY_FP32_SAMPLE_FP64_EON_RNE_U8_V1`: exact corrected PVSA Y1 is the
byte-custodied camera source, `SE3XiTransportV2.xi[pair]` drives the EON
ground-homography warp, sampling is NumPy fp32 with fp64 geometry, and the
camera output is round-to-nearest-even uint8. The scorer's frozen resize `R`
remains downstream, hence CAMERA-THEN-R.

This is a new PVSA source foreign key. It does not translate or forge the
bounded PASS-G binding of the older G13 packet.

## Real fresh-G82 pair-0 execution

Input custody:

```text
base pair SHA-256  caf69dade383564ef8123149d193052b8b5b641711fed232b25dd67b29af25db
P0 SHA-256         754ce88b494bfbc3bd560b23ee26cdbccffb10d5829fa09d882e04f918aa9126
exact Y1 SHA-256   65ca46b182ef52d4cedffb56ec48576bd19610802b60f88027a9e7e46158a037
```

COPY is a real whole-frame control:

```text
changed Y0 values  828605
changed Y0 pixels  429630
output Y0 == Y1    true
exact Y1 preserved true
double decode      identical
```

The Road `ROLE_TRANSLATE_RGB(+1,-1,+1)` run is only an ownership regression,
not a capstone or pose result:

```text
owned/changed Y0 values       690762
changed Y0 pixels             230254
preserved unowned P0 values  2361246
exact Y1 preserved                yes
double decode                     identical
```

COPY owns the whole Y0 frame, so its unowned set is empty. The role-local
regression supplies the nonempty proof that P0 outside owned support remains
byte-identical.

## Pair-0 frozen CPU scorer receipt

The permitted local CPU PoseNet comparison used the exact G85 pair-0 decode,
frozen upstream weights SHA-256 `0f3a0874...`, seed 1234, two Torch threads,
deterministic algorithms, and two identical forwards:

```text
PASS_P0 d_pose            195.38666847693818
COPY_Y1 d_pose            195.27929640452930
COPY - PASS                -0.10737207240888

pair-local sqrt term PASS  44.20256423296483
pair-local sqrt term COPY  44.19041710648693
pair-local delta           -0.01214712647790
```

Seg is exactly unchanged because G88 never writes Y1 and upstream SegNet reads
only the final frame. This one-pair result is `[macOS-CPU local advisory]`.
It is not a population average, score, candidate selection rule, or evidence
that COPY is a viable pose family.

The exact +214-byte outer price would add `0.00014249381596814466` to the n600
archive rate term, but combining that n600 byte term with one pair's nonlinear
pose term would be invalid. No combined score or score-per-byte claim is made.

No XIP2 scorer comparison was run: the only locally executable XIP2 trajectory
was a deliberately synthetic test fixture. Scoring it would manufacture
evidence. The fixture proves only exact parsing and deterministic execution:
176 XIP2 bytes, 336 operand bytes, global n600 default, changed pair-0 uint8
Y0, and exact Y1 preservation.

## Triality

DSL:

```text
PVSC2(
  exact_PVSA1_member,
  G88(
    default in {PASS, XIP2},
    sparse_overrides[pair] in {PASS,COPY,ROLE,XIP2},
    optional exact XIP2[600,6]
  )
)
```

DAG:

```text
fresh G82/G85 custody
  -> strict successor parse
  -> strict embedded PVSA parse
  -> exact corrected Y1
  -> resolve population default + sparse override
  -> PASS / COPY / role-local / XIP2 camera transition
  -> assert exact Y1
  -> assert unowned Y0 == P0
  -> deterministic second decode
  -> bounded chronological stream batches <= 16
```

Equations:

```text
Y0'(p) =
  P0(p),                                      mode PASS
  Y1(p),                                      mode COPY
  P0 outside M; T(Y1)+delta on M,             mode ROLE
  round_even(clip(W_EON(Y1(p), xi[p]))),       mode XIP2

Y1'(p) = Y1(p) exactly

global_XIP2_operand_bytes = 160 + exact_XIP2_payload_bytes
```

## Verification

```text
focused pytest       7 passed in 35.62s
Ruff check/format    passed
Mypy focused         passed
py_compile           passed
```

Adversarial coverage includes unknown modes, default aliases, unordered IDs,
CRC/EOF mutation, XIP2 digest mutation with a resealed CRC, missing/foreign
XIP2, raw Pose6 masquerading as XIP2, foreign base-member keys,
noncontiguous batches, and base double-decode disagreement.

## Pointer-delta honesty

The effective frontier remains the official-leaderboard `0.172`. G88 did not
produce a fresh solved xi trajectory, public `inflate.sh` closure, full-n600
successor score, or contest-CPU/CUDA row. The exact remaining blockers are:

```text
G88_FRESH_V15_N600_XIP2_TRAJECTORY_CUSTODY_OWED
G88_PUBLIC_INFLATE_RUNTIME_GRAPH_LINK_OWED
G88_EXACT_POSE_OR_UPSTREAM_EVAL_OF_SUCCESSOR_ARCHIVE_OWED
```

Machine-readable receipt:
`g88_population_conditional_y0_pvsa_receipt_20260727.json`.
