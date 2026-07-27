# G74 — V15-native role-aware support overlay decoder

Status: implemented and bounded-real-source proven at the frozen pre-scorer
resize surface. Research-only. No Pose, frozen-network, score, candidate,
public-runtime, n600-evidence, or frontier claim.

## Structural correction

The first G49 analytic wire is not a V15-compatible actuator. It strips
`BoundaryShearletAtomV1.role`, paints an RGB residual after a legacy
scorer-grid render, and realizes a whole factor-2 camera plane. V15 instead
consumes role before ordered role painting, applies the counted realization
profile and scorer-solved templates at camera resolution, and only then reaches
the frozen scorer resize. Reinterpreting the old wire would silently change its
meaning, so G74 adds a distinct counted operand and leaves G49 unchanged.

The G74 operand is:

- magic/version `G74RA1` / `1`;
- one counted `SelectedPreimageFrameSelectorV1` (`Y0`, `Y1`, or `BOTH`); and
- donor-exact canonical G2SH bytes, including each atom's global source pair
  and role.

Parse-back must reproduce the exact operand bytes and SHA. The immutable
semantic `P` archive and counted operand `A` remain separate custody objects.
At decode, G74 creates an ephemeral receiver state `P+A` only for execution.
Its retained `.archive` is still `P` and is never claimed as mutated archive
custody.

## Native execution and ownership

All immutable-P atoms and counted-A atoms are merged once in donor G2SH
canonical order. Address collisions fail closed because atom painting is
sequential and noncommutative. Both `P` and ephemeral `P+A` execute the actual
V15 `CarrierComposeReceiverV1.render_camera_pairs`; the legacy
`render_pairs` path is diagnostic-only.

The frozen resize has disjoint camera supports and channel-separable RGB
values. G74 derives ownership independently for every:

`frame × scorer row × scorer column × RGB channel`.

A value is owned if any camera tap in the native `P` versus `P+A` support
differs. The donor-copy policy copies all native-mutated taps in that owned
support/channel and preserves every unowned camera byte from `P`.

Ownership cannot be reduced to changed exact integer numerators. A bounded
hard falsifier at scorer cell `(13,0)` uses coefficients
`(220968,387288,64728,113448)` and tap deltas `(+29,0,-99,0)`: the rational
numerator delta is exactly zero, while CPU Torch float32 bilinear changes
`128.0` to `127.99990844726562`. Therefore the load-bearing gate is bit
equality of the actual CPU
`torch.nn.functional.interpolate(..., size=(384,512), mode="bilinear")`
pre-scorer tensor. Exact integer numerators and rounded-u8 planes are retained
as diagnostics and hashes, not mislabeled as evaluator inputs.

G74 is support-minimal donor-copy, not a globally minimum camera preimage.
G76 is the compatible next stage: it receives exact G74 base/donor camera
states, searches base-preferred local integer preimages, tests live Torch
parity, and falls back to donor taps when needed.

## Custody and streaming contract

`V15RoleAwareOverlayDecoderV1` has a sealed `.open()` constructor. It binds:

- exact semantic archive bytes, byte count, and SHA-256;
- the exact retained receiver object and its V15 realization/template surface;
- the exact disjoint-resize operator; and
- the G74 v1 donor-tap-copy policy.

Direct construction and dataclass replacement fail. Decode rechecks the seal.
The counted atom bank is validated against the full receiver source window,
then only requested local pair IDs are rendered. An operand may therefore be a
global n600 bank while decode streams a bounded pair subset. Requested atoms
with zero native support effect fail closed; a streamed subset with no atoms
returns the immutable base.

Bounded coverage debt is explicit: the retained real V15 fixture begins at
source pair zero and does not contain a pre-existing shearlet bank exercising
the merge path. Nonzero source-window mapping has a pure unit test, but no real
nonzero-window V15 decode is claimed. Likewise no separate retained fixture
proves composition with both pre-existing shearlets and static rules. Those are
required integration fixtures before public/additive G49 closure, not facts
inferred from this pair-zero proof.

The typed receipt has canonical JSON parse/re-encode, decoder source SHA,
archive/operand custody, numerator/Torch/camera/mask hashes, and exact bounded
truth flags. Result arrays validate exact dtype, shape, ownership hashes,
camera hash, rounded-diagnostic hash, output numerator hash, and output Torch
hash against the receipt. Decode runs twice and refuses drift.
Because the float32 nullspace is version-sensitive, the receipt also binds the
local CPU Torch version (`2.12.1`) and explicitly sets
`cross_host_torch_parity_claim=false`. Local double decode is not represented
as cross-host determinism.

## Bounded retained-V15 result

Source archive:

- bytes: `133941`
- SHA-256:
  `759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df`
- V15 scorer-solved templates: `6`

Bounded operand: one role-preserving `Road` atom at global source pair `0`,
center `(160,256)`, scales `(24,96)`, shear `0`, amplitude `64 q4`, selector
`BOTH`.

Measured across Y0 and Y1:

- operand: `39` bytes, SHA-256
  `39f99ce0e8283a4e3f42ab8bfc4b9d3aa319128f9f62ed1e8d753a1f5ca1f75e`;
- changed support values: `(207,207)`;
- changed support RGB cells: `(69,69)`;
- changed integer numerator values: `(207,207)` for this particular atom;
- owned camera values: `(828,828)`;
- camera values actually changed from P: `(633,633)`;
- preserved unowned camera values: `6102360`;
- exact resize denominator: `786432`;
- selected output integer numerators equal native `P+A`;
- selected output CPU Torch float32 scorer inputs are bit-equal to native
  `P+A`;
- unselected-frame base preservation is selector-covered;
- operand and receipt parse/re-encode are exact; and
- deterministic double decode is equal.

The legacy-coordinate diagnostic on the same retained pair changes
`(17570,17732)` rounded-u8 RGB values relative to native V15. Whole factor-2
realization changes `(745497,745958)` camera values and creates `708180` zeros
per frame versus `20680` in native V15. These are mechanism diagnostics only.

## Composition boundary and honest next gate

G74 closes the missing semantic-coordinate receiver primitive but does not
compile G49 factors, invoke Pose/Seg networks, create a public archive, or
measure n600. G72 remains fail-closed until a source-backed compiler emits this
distinct role-aware operand. A future additive G49 factor mode must preserve
role and global source addressing, aggregate all pre-paint factors before one
native render, and define explicit mixed-stage ordering. The old role-stripped
mode remains frozen and must never be reinterpreted as G74.

Pointer delta: none. The exact frontier is unchanged.
