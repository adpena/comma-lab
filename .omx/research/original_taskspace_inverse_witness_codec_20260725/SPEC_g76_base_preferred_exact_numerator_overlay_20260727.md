# G76 — base-preferred exact numerator overlay

Status: implemented and bounded-real-source proven through the live Torch
pre-scorer resize. Research-only; no frozen-network Pose, score, candidate, or
pointer claim. The current proof is local macOS CPU with Torch `2.12.1`;
cross-host/contest-CUDA bit parity remains explicitly unclaimed.

## The structural gap

The old generic factor-2 realization fills every camera support from a scorer
byte and leaves every coordinate outside those supports at zero. That proves
distortion feasibility, but it discards the semantic base's camera photometry
and therefore suppresses Pose signal.

The first G74 repair makes the semantic actuator role-aware and renders it in
the exact V15 camera coordinate before the exact integer downsample. Copying
the actuator-rendered donor taps on changed supports is exact, but it still
chooses one arbitrary feasible preimage.

The factor-2 operator gives a stronger decomposition. Its scorer supports are
disjoint 2x2 camera blocks, and RGB channels do not mix. Therefore every
donor-changed frame/scorer-coordinate/channel begins as one independent
bounded integer equation:

`c00*x00 + c01*x01 + c10*x10 + c11*x11 = n_donor`

with `xij in {0,...,255}` and exact common denominator `786432`.

Adversarial G75 review found the crucial floating-point refinement. Exact
integer numerator equality is necessary but not sufficient for the real
evaluator: different 2x2 taps can have the same rational numerator yet round
differently in float32 Torch bilinear interpolation. The regression fixture at
scorer cell `(13,0)` applies tap deltas `(+29,0,-99,0)` to coefficients
`(220968,387288,64728,113448)`. The integer delta is exactly zero, while the
live Torch result changes from `128.0` to `127.99990844726562`.

Therefore G76 ownership is derived from any donor-tap difference, not merely
integer-numerator difference, and live Torch parity is a second mandatory
gate.

## Receiver policy

`taskspace_g76_base_preferred_exact_numerator_overlay_v1.py`:

1. renders or receives an exact V15 base camera pair and an exact role-aware
   actuator donor camera pair;
2. marks every Y0/Y1 and R/G/B scorer support whose four donor taps differ;
3. calls the existing GCD-pruned bounded integer block solver with the four
   base taps as its deterministic preference;
4. runs the exact CPU float32 `torch.nn.functional.interpolate` operation used
   by both upstream PoseNet and SegNet;
5. falls back to the exact donor taps on any block that misses either the
   search budget or bit-identical Torch parity; and
6. proves whole selected-frame Torch scorer-input equality, donor numerator
   equality, and bit-identical base camera bytes outside owned channels.

The current solver is a deterministic base-preferred exact feasibility search,
not a proof of globally minimum L0/L1/L2 or Pose-costate energy. That stronger
objective is the next compatible extension; the ABI deliberately does not
overclaim it.

## Real source-bound proof

The proof reopens the retained fresh V15 n600 semantic archive:

- archive bytes: `133941`
- archive SHA-256:
  `759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df`
- bounded operand: one role-preserving Road `BoundaryShearletAtomV1` at source
  pair 0, center `(160,256)`, scales `(24,96)`, amplitude `64 q4`.

Across both chronological frames:

- donor-tap-owned scorer values: `414`;
- donor-tap-owned scorer RGB cells: `138`;
- integer-numerator-changing values: `414`;
- base-preferred blocks retaining live Torch parity: `168`;
- Torch-parity donor fallbacks: `246`;
- solver-budget fallbacks: `0`;
- maximum solver nodes for one block: `26`;
- donor camera values changed from base: `1548`;
- G76 camera values changed from base: `1345`;
- camera disturbances avoided: `203`, or `13.113695090439276%`;
- output numerator and live Torch scorer-input SHAs equal their donor SHAs
  exactly; and
- deterministic double decode is equal.

The machine-readable receipt is
`g76_real_v15_pair0_base_preferred_exact_numerator_overlay_receipt_20260727.json`
with self-hash
`d8a97bbf0d6d6070cadc13f18d174b2a5a8405dfd4bbb80c9602e5906c268bc0`.

## Composition consequence

The logical factor determines the desired role-aware actuator state; G74
renders its exact native V15 donor; G76 selects a camera preimage that preserves
more base-domain bytes while remaining bit-identical to that donor at the live
Torch scorer input. Therefore the measured `203` avoided camera disturbances
are evaluator-nullspace/coding freedom, not a present Seg or Pose improvement:
both frozen networks receive exactly the donor tensor. The freedom becomes
useful only when a later camera-domain composition or coder can exploit the
lower disturbance, and it remains bounded to the measured local Torch runtime.
Conditional A3 can then optimize Y0 given the already-realized Y1, while later
same-transition coding costates can replace the base preference with a measured
compression preference without changing the counted factor syntax.

This is the local, separable form of the previously intractable full-lattice
solve: solve distortion exactly per tiny block, preserve the useful base
signal, and train only the remaining factor/program quotient.
