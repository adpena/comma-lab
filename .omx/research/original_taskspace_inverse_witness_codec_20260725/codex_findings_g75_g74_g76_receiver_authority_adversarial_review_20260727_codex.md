# G75 adversarial review — G74 native overlay and G76 exact preimage

Date: 2026-07-27  
Reviewer role: read-only adversarial review; no G74/G76 source edits  
Git HEAD observed: `64f728d1cbc0ee7995bb221b12137d57f8cd200c`  
Verdict scope: bounded retained-V15 pairwise receiver mechanics on local macOS CPU only

## Executive verdict

**G74: PASS, bounded primitive only.** G74 closes the structural error in the
old G49 coordinate: it preserves the shearlet role and global source-pair
address, merges the counted operand into the V15 receiver before role paint,
executes the real `CarrierComposeReceiverV1.render_camera_pairs` path, and owns
camera taps independently per frame, scorer coordinate, and RGB channel. Its
selected output is bit-identical to native `P+A` at both the exact integer
resize numerator and local CPU Torch float32 pre-scorer surfaces.

**G76: PASS for its mechanical preimage contract; the required semantic
correction was applied before commit.** G76 genuinely reduces camera-domain byte
disturbance while retaining exact donor numerators and local CPU Torch
pre-scorer tensors. Because both PoseNet and SegNet begin from that same resized
RGB tensor, the reduction is in the evaluator's nullspace: it cannot by itself
preserve or improve Pose or Seg relative to the G74 donor. The initial G76 spec
language at lines 95–99 leaked beyond that proof; the current spec now labels
the result evaluator-nullspace/coding freedom and explicitly denies a present
Pose or Seg improvement.

Neither unit moves the exact frontier or proves a score. There is no n600
network evaluation, candidate archive, public runtime closure, contest-CPU/CUDA
row, or Pose result.

## G74 findings

### What is proven

1. The counted `G74RA1` operand preserves the donor G2SH role and global
   source-pair address, and strict parse/re-emit plus caller-supplied SHA closes
   the bounded operand bytes.
2. Immutable `P` and counted `A` remain distinct custody objects. The combined
   receiver is execution-only; it is not mislabeled as mutated archive custody.
3. Existing and operand shearlets are combined in donor canonical key order and
   collisions fail closed. The native receiver consumes the role before ordered
   realization paint and scorer-solved templates.
4. Both base and mutated states call `render_camera_pairs`; legacy
   `render_pairs` is diagnostic-only.
5. Ownership is derived from **any donor-tap difference**, not from rounded
   bytes or changed integer numerators. It is channelwise:
   `frame × scorer row × scorer column × RGB channel`.
6. The replacement policy is sealed to exact donor-tap copy. Decode independently
   reconstructs the expected ownership mask, preserves all unowned base camera
   bytes, verifies selected native integer numerators, and verifies local CPU
   Torch float32 bilinear equality.
7. Constructor sealing prevents direct dataclass construction/replacement.
   Global operand banks may be decoded in pair subsets; atoms are checked
   against the full receiver source window.
8. Receipt/result hashes bind the archive, operand, decoder source, camera
   output, integer numerators, Torch tensors, and ownership masks. The final
   receipt records Torch `2.12.1` and explicitly sets cross-host Torch parity to
   false.

The hard nullspace falsifier is correctly retained: at scorer cell `(13,0)`,
coefficients `(220968,387288,64728,113448)` and tap deltas
`(+29,0,-99,0)` produce zero exact numerator delta but change CPU Torch float32
from `128.0` to `127.99990844726562`. G74 owns that support and donor-copy
restores Torch bit equality.

The retained Road atom proof reports 207 changed support values per frame, 828
owned camera values per frame, and 633 camera values actually changed per
frame. An additional read-only live-role falsifier placed an
`UndrivableBoundary` atom at pair 0, center `(174,420)`, scales `(8,24)`, and
amplitude `64 q4`; it produced 156 changed support values and 456 actual camera
changes per frame. Thus both permitted roles are receiver-live; the role field
is not metadata-only.

### G74 debt and exact scope limits

- The retained fresh V15 archive has six scorer-solved templates, but no static
  realization-rule member and no pre-existing G2SH member. The code path keeps
  static rules and existing atoms inside the same native render, but those
  interactions are not fixture-proven here.
- Nonzero `source_pair_start` arithmetic is unit-tested as a pure mapping. A
  real nonzero-window V15 archive is not exercised. The retained n600 archive
  starts at zero.
- Local same-host double decode is not cross-host determinism. The receipt now
  states this explicitly.
- G74 does not add a public decoder or a new additive G49 factor mode. The old
  role-stripped `SHEARLET_BOUNDARY_TRANSPORT_Q4` behavior remains untouched and
  must remain frozen. A future mode must aggregate all native-prepaint factors
  before one render and define mixed old/new ordering or refuse the mix.
- No Pose/Seg model is invoked. Exact donor pre-scorer equality proves only
  equality to the donor at that surface, not preservation relative to semantic
  base or source ground truth.

Final G74 source SHA-256:
`96de0f530db1394383b83976371f727d4c4a0cac8a10a2f38dc79842fc3ce1b8`.
Final outer receipt self-hash:
`6a55d4d4a56a6e51336916831b1e84f7aa37f741a935faebd97f1930bcdea9ec`.

## G76 findings

### What is proven

1. Ownership is derived from changed donor taps per selected
   frame/scorer-coordinate/channel, closing the zero-integer-delta Torch
   falsifier.
2. Each owned disjoint 2×2/channel block receives a deterministic
   base-preferred exact integer solve. Budget misses copy the known-feasible
   donor taps.
3. A second local CPU Torch gate detects integer-nullspace float differences
   and replaces those blocks with donor taps. A whole-selected-frame replay must
   be bit-identical to the donor.
4. Unselected frames and every unowned camera value remain base-identical.
5. Receipt parse-back is canonical and duplicate-key rejecting. The result now
   binds the exact ownership-mask SHA, closing the same-count/wrong-mask
   substitution found during review.
6. The retained pair-0 result is internally closed: 414 owned scorer values,
   168 base-preferred Torch-exact blocks, 246 Torch-parity donor fallbacks, zero
   budget fallbacks, and matching donor/output numerator and Torch hashes.
   Camera changes fall from 1548 in the donor to 1345 in G76, a reduction of
   203 values (`13.113695090439276%`).
7. The receipt records Torch `2.12.1`, explicitly denies cross-host parity,
   denies Pose/score/candidate claims, and binds the current source SHA.

Final G76 source SHA-256:
`fe6eacb8715cfc3081ae88d1a81afd1d0537733f81efb81b5438ccf4449ca416`.
Final outer receipt self-hash:
`d8a97bbf0d6d6070cadc13f18d174b2a5a8405dfd4bbb80c9602e5906c268bc0`.

### Evaluator-nullspace correction

`upstream/modules.py` applies the same bilinear resize before both authority
paths: PoseNet resizes RGB and then computes YUV6; SegNet selects the last frame
and resizes RGB. G76 requires its selected resized RGB tensor to be bit-identical
to the G74 donor. Therefore:

`PoseInput(G76) = PoseInput(G74 donor)` and
`SegInput(G76) = SegInput(G74 donor)`

on the proven local CPU Torch axis.

The 203 avoided camera changes are consequently invisible to both current
evaluator networks. They may become useful only to a later camera-domain
composition or to a separately measured coding mechanism. They are not a Pose
preservation result, a score lever, or evidence that the semantic base's
scorer-visible photometry survived.

### Required pre-commit spec correction — satisfied

**Recommendation: correct G76 spec lines 95–99 before commit.** The initial
claim that G76 avoided a choice between native V15 semantics and preservation of
the “semantic/Pose base” needed language equivalent to:

> G76 preserves more semantic-base camera bytes while reproducing the G74
> donor's complete selected pre-scorer RGB tensor bit-for-bit. Because PoseNet
> and SegNet both begin from this tensor, the current reduction is
> evaluator-nullspace-only and makes no Pose, Seg, or score claim. Its value is
> a future camera-domain composition or coding coordinate that must be measured
> separately.

This recommendation was accepted during the audit. Current spec lines 95–102
now state that the 203 avoided camera disturbances are evaluator-nullspace /
coding freedom, not a current Seg or Pose improvement, because both networks
receive the donor tensor. That correction is adequate for this bounded verdict.

### Remaining G76 receiver/authority gaps

- The generic API accepts base and donor arrays; it does not itself bind a G74
  archive, operand, or donor receipt. The retained outer proof supplies that
  custody for one pair, but production composition must bind the exact G74
  donor camera SHA and role-aware operand.
- The float-parity branch is proven only for local macOS CPU Torch `2.12.1`.
  It must not run as a supposedly host-portable public decoder until
  contest-CPU/CUDA and cross-host output determinism are closed, or until the
  selected camera bytes are fixed independently of host float branching.
- There is no additive G49 wire/public runtime, n600 run, network forward,
  counted archive result, or exact score.
- “Base-preferred” is a deterministic feasibility preference, not a global
  L0/L1/L2, byte, or Pose-costate optimum.

## Verification

- `python3 -m py_compile` on the G74 and G76 modules/tests: PASS on stable
  sources.
- `uv run ruff check` on the G74 and G76 modules/tests: PASS on stable sources.
- G74 focused suite: 7 tests passed on the stable source.
- G76 focused suite: 6 tests passed.
- Frozen legacy G49/G17 compatibility suites:
  `test_taskspace_selected_preimage_program_v1.py` and
  `test_taskspace_g17_g49_active_a_abi.py`: 27 tests passed.
- The sealed G74 receipt additionally records the adjacent
  `test_taskspace_g17_g49_selected_program_product_bridge_v1.py` suite:
  31 tests passed across all three adjacent files.
- G74 and G76 outer receipt self-hashes independently recomputed: exact.
- G76 inner typed receipt self-hash independently recomputed: exact.

## Pointer delta

None. These are receiver/preimage mechanics. The exact frontier is unchanged,
and no score claim is admissible from this review.
