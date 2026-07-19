# VJP custody, positive anisotropic bands, and rung-E rate points

**Date:** 2026-07-19 UTC
**Axis:** `[Darwin-arm64 CPU advisory] NON-PROMOTABLE`
**Authority:** build + local measurement only; no launch, paid compute, contest score,
promotion, or pointer authority.
**Pointer:** `0.1910828242 [contest-CPU]` **UNMOVED**.
**Verdict scope:** 24 selected unique pairs from the real frozen `gt_n600.npz` cache;
native-float32 CPU-Torch SegNet/PoseNet hard oracle. No receiver-closed archive,
contest-Linux CPU, contest-CUDA, or score claim. `research_only=true` until those
promotion surfaces exist.

## Outcome

The missing real derivative custody is closed and the first positive rung-E
range-coordinate curve is measured. Twenty-four immutable per-pair VJP sidecars
carry a frozen Seg active-field VJP and PoseNet first-six Jacobian. The final
wide sweep has 96 accepted pair/operating-point observations, four n24 curves,
96 content-hashed full bindingness maps, `d_seg=0` at every accepted row, and
all 96 frozen-hard Pose constraints inactive/slack at their declared `tau_pose`.

The lowest measured rate point is **MEASURED** at Seg scale `1e-4`,
`tau_pose=2.5e-4`: `1,474,579.92` Brotli-Q11 bytes/pair,
`1,662,768.38` zstd-19 bytes/pair, `d_seg=0`, and mean
`d_pose=2.521975392375284e-5`. Relative to the same-pair tiny-`tau=1e-7`
calibration at that Seg scale (`2,363,386.21` Brotli bytes/pair), the rate
reduction is **DERIVED** as `37.6073%`. This is a range-coordinate residual
measurement, not an `archive.zip` byte result.

The composed waterfill remains `INCONCLUSIVE_FLAT_OR_NOISY` at measured-instance
scope. Pose secants are traceable, but all four hard-admitted Seg points have
identical `d_seg=0`, so there is no measured Seg secant and no honest non-null
KKT allocation. No allocation is forced.

## Authority and frozen sources

The implementation follows `docs/operating_manual_craft_handoff.md`: artifact-backed
claims are labeled, negatives retain verdict scope, and an unchanged pointer is
reported literally. The source hierarchy was re-derived from:

- `.omx/research/joint_seg_pose_inverse_solve_20260719_codex.md` and the landed
  joint solver/measurement tool;
- `.omx/research/segnet_recursive_fractal_factorization_20260715.md`, equation
  `segnet_head_rank4_linear_flipdist_v1` and measured rank-4 head;
- `.omx/research/frozen_scorer_exact_factorization_20260715.md`, especially the
  shared bilinear resize `A`, last-frame Seg path, and two-frame Pose path;
- frozen cache SHA-256
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`;
- frozen `modules.py`, `frame_utils.py`, SegNet weights, and PoseNet weights with
  hashes `065961ba...`, `d689aca7...`, `68956e32...`, and `0f3a0874...` recorded
  in every VJP manifest.

The cache contains cached winners and margins, but no cached logits or rival IDs.
The producer therefore uses cached `lstars` as winner, verifies them against a
fresh native-float32 frozen forward, regenerates the highest non-winner rival
from that forward, and labels it as regenerated. It never calls that rival cached.

## Custodied derivatives

### SegNet active-field VJP

For each scorer cell with cached winner `w` and fresh rival `r`, the exact head
normal is

`n_hw = (W_w - W_r) / ||W_w - W_r||_2`.

The producer backpropagates the sum of the active unit-normal winner/rival logit
differences through frozen SegNet. It persists scorer-plane
`g_y` with shape `(384,512,3)` and camera pullback `g_x=A^T g_y` with shape
`(874,1164,3)`, both fp32. This is one field-level aggregate VJP, **not**
196,608 independent full-frame per-cell Jacobians; the hard oracle remains the
only admission authority.

The local field is **DERIVED**, not guessed:

`Lip_local = ||g_y||_2`, `q = g_y / Lip_local`, and `g_y = Lip_local * q`.

Across the final 24-pair corpus, measured `Lip_local` had min
`0.0019665483850985765`, mean of pair means `0.8891905631699905`, max
`21.897525787353516`, and zero count `0`. The maximum measured unit-`q` norm
error was `1.534561697713599e-7`.

The real pair-0 derivative smoke recorded unit-head-normal max error
`5.960464477539063e-8`, Seg directional-finite-difference relative error
`6.788555013327042e-6`, and `A^T` max-absolute residual `0`.

### PoseNet-6 Jacobian

The solver consumes scorer-plane coordinates, so the active representation is

`J_y = d pose[:6] / d(y0,y1)`, fp32 `(6,2,384,512,3)`.

The producer also persists `J_x=A^T J_y`, fp32 `(6,2,874,1164,3)`, for camera
custody and adjoint review. The real pair-0 smoke measured Pose forward parity
max-absolute error `3.814697265625e-6` and Pose `A^T` residual `0`. `J_y` is
consumed by the joint solver; `J_x` is retained as custody, not counted payload.

### Immutable manifests and storage

The final real pair IDs are `[0..10,24]` and `[12..17,19..23,25]`. Their 24
per-pair VJP NPZ files total `3,641,507,444` bytes on the primary SSD tier.
Every producer invocation is capped at 12 pairs, resumes from an atomic manifest,
validates source/tensor/config hashes, recovers a valid orphan after rename, and
never overwrites final sidecars.

- `/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/chunk_000_010_024_composed/manifest.json`,
  file SHA-256 `3d1218a52ededc4b347ae94c5c2bf58d06d70dd8f530bec67bf9cab36ee00694`.
- `/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/chunk_012_017_019_023_025_composed/manifest.json`,
  file SHA-256 `200e8cfa375cbdb8154777156441ae6adadf33e75668c86cc52b816f79488e94`.

The manifests are zero-copy compositions: they validate and reuse settled
immutable sidecars rather than re-deriving already-settled pairs.

## Rung E and explicit bindingness

Rung E jointly inverse-solves exact reachable scorer-plane integer numerators
for both frames. The declared free predictor is the generated piecewise-constant
fill of the counted scorer-plane description. The measured payload is the
signed little-endian int32 numerator residual `chosen_yhat - predictor_yhat`,
compressed with actual Brotli-Q11 and zstd-19. Camera residuals and `ker(A)` are
not serialized.

Every accepted row writes an immutable compressed bindingness NPZ containing:

- full frame-0 and frame-1 interval maps (`0=slack, 1=lower, 2=upper`);
- the full positive Seg-radius channel map;
- frame-0/frame-1 conservative exact-source fallback maps;
- embedded config, dtype, shape, tensor hash, file hash, and byte custody.

All 96 sidecar file hashes were re-read successfully. They total `23,827,016`
bytes. Across 96 rows, all `56,623,104` Seg radius channels were positive.
Frame-0 binding counts were `26,239,283` slack, `15,266,647` lower, and
`15,117,174` upper; frame-1 counts were `51,885,708` slack, `2,711,481` lower,
and `2,025,915` upper. Conservative exact-source fallback counts were
`3,040,841` frame-0 and `6,017,108` frame-1 channel blocks.

## Wide positive-band measurement

All values in this table are **MEASURED** on the same final n24 corpus. Bytes
are mean two-frame residual bytes per pair. `repairs` is the sum of hard-oracle
repair levels over 24 accepted rows; `rejects` counts proposal attempts rejected
by the frozen hard oracle before repair.

| Seg scale | `tau_pose` | Brotli-Q11 | zstd-19 | mean `d_seg` | mean `d_pose` | repairs (max) | linear active | hard Pose inactive | rejects |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1e-4` | `1e-4` | 1,588,353.75 | 1,789,521.75 | 0 | 2.0176594279141168e-5 | 9 (5) | 3/24 | 24/24 | 9 |
| `1e-4` | `2.5e-4` | 1,474,579.92 | 1,662,768.38 | 0 | 2.521975392375284e-5 | 9 (5) | 0/24 | 24/24 | 9 |
| `1e-3` | `1e-4` | 1,895,298.79 | 2,126,669.67 | 0 | 7.665108768125358e-6 | 41 (9) | 1/24 | 24/24 | 41 |
| `1e-3` | `2.5e-4` | 1,810,404.58 | 2,030,793.54 | 0 | 1.4835970267926648e-5 | 40 (9) | 0/24 | 24/24 | 40 |

The pre-registered authority hypothesis is **CONFIRMED 96/96 at this instance
scope**: the frozen-hard Pose constraint was inactive/slack at every accepted
source-centered Seg-band solution. The measured `d_pose/tau_pose` ratio ranged
from `7.859581504140086e-7` to `0.9221268935685756` (mean
`0.10965998180984583`). The linear proposer was more conservative and active in
4/96 rows; that is recorded separately and is not promoted into a false
hard-constraint refutation.

Across the four points the runner evaluated 195 proposals, admitted 96 final
rows, rejected 99 proposals before repair, and had zero unevaluated proposals.
Every accepted row had zero hard Seg mismatches. The custodied gradient-mode VJP
winner agreed with the inference-mode hard winner at every row. Regenerated
rivals differed at 12 cells per operating point across pairs
`{2,17,19,20,21,22,23,25}`; rival identity is proposal-only, so these differences
were recorded but did not override hard winner/Seg authority.

The tiny-`tau={1e-8,1e-7}` grid completed before the later operator reframe and
is retained only as historical calibration. It is not used by the final wide
composition or waterfill verdict.

## Receipts

The composed receipt binds all eight exact-tool-hash source receipts, 96 rows,
24 unique pairs, four n24 curves, the hypothesis count, and the waterfill result:

`/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/vjp_positive_n24_wide_composed.json`

SHA-256 `9c90483937114ae18bb4b516cc8296e8e3ab4d578a2de44f86604b55cd5755a0`.
The executed measurement-tool hash in every source receipt is
`11b51c33e2497e42bc87d56b7de4040172c9d2f4961f3d49b268d780934818c8`.

Source receipt hashes:

- scale `1e-3`, tau `1e-4`: chunk A `10d7fa3b...85f60`, chunk B
  `80c28a42...a29d4`;
- scale `1e-3`, tau `2.5e-4`: chunk A `2f2791e7...e6856`, chunk B
  `209c3e90...6a1ac`;
- scale `1e-4`, tau `1e-4`: chunk A `dd56fd4d...f9b22`, chunk B
  `10c856f3...460b`;
- scale `1e-4`, tau `2.5e-4`: chunk A `e4b65af9...ddf43`, chunk B
  `054679d7...12f7`.

## Scoped refusals and replacements

- Pair 11 refused derivative production because its cached/native active winner
  differed at one pixel. Durable refusal:
  `/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/refusal_pair_0011/pair_0011.vjp_refusal.json`,
  SHA-256 `8780a9fe8fe07b39a354c7ab082e9ad0e0a34c06a49ac48348fb25d45f7402db`.
  Scope is pair 11 and that frozen arrangement only; pair 24 replaced it.
- Pair 18 retained one hard Seg flip after repair level 12 at an early
  positive operating point (`d_seg=5.086263020833333e-6`,
  `d_pose=8.756841548181644e-11`). Durable refusal SHA-256
  `68da458e534cb116d2213f8fe0838615d6c7c2f4137bd29a071571150451f365`.
  Scope is pair 18 and that operating point only; pair 25 replaced it.
- Intermediate wide-v1 rows used pre-final instrumentation bytes. They were
  preserved but superseded after provenance review; only wide-v2 receipts with
  exact tool hash `11b51c33...` enter the composed result.

These are not family negatives.

## Round-1 adversarial self-review

1. **Is the pullback truly the unit normal's?** The producer normalizes exact
   frozen 3x3 head winner/rival weight differences, audits active-pair norms and
   seed cotangent signs, performs a real directional finite-difference check,
   and retains the `A^T` residual. The pair-0 checks above are MEASURED.
2. **Is this secretly a per-cell full Jacobian claim?** No. Seg custody is one
   active-field aggregate VJP. The memo and metadata state that limitation;
   hard frozen SegNet arbitrates every candidate.
3. **Can the band admit a bad candidate?** Yes. It is proposal-only. The final
   run rejected 99/195 proposals and repaired them; no rejected proposal was
   silently admitted.
4. **Is `Lip_local` invented?** No. It is the stored pointwise norm of measured
   `g_y`, and raw `g_y`, `q`, `Lip_local`, factorization hashes, and unit errors
   are custodied.
5. **Does Pose authority come from the linear model?** No. `J_y` proposes;
   full frozen PoseNet MSE admits. Linear and hard bindingness are reported
   separately. This distinction changed the final hypothesis count and was
   regression-tested, including zero-`tau` equality.
6. **Are full binding maps real bytes?** Yes. Review caught an initial metadata
   error that described `-1/0/+1`; the actual solver uses `0/1/2`. The metadata
   was corrected, tests pin the semantics, and all 96 final NPZ hashes were
   re-read.
7. **Do receipts bind landed source bytes?** Yes for the final grid. An initial
   wide pass was invalidated when its tool hash preceded the final semantics
   correction. The authoritative 96 rows were rerun against one exact reviewed
   tool hash.
8. **Is a KKT allocation now justified?** No. All measured Seg points are flat
   at zero distortion. The scoped `INCONCLUSIVE_FLAT_OR_NOISY` verdict is the
   correct output; the family remains open.

Focused verification after the final code review: `39 passed`, Ruff clean,
`py_compile` clean, CLI help clean, and `git diff --check` clean. The sacred
result tree remained unchanged. No upstream file, live run, score pointer,
remote provider, or paid substrate was modified.

## Triality and system wire-in

- **Equations:** `segnet_head_rank4_linear_flipdist_v1`,
  `g_y=Lip_local*q`, `J_x=A^T J_y`, the Pose quadratic step, and measured
  waterfill secants are explicit.
- **DAG/evidence:** immutable derivative manifests, per-stage binding NPZs,
  resumable state, scoped refusals, and the composed receipt form the executable
  evidence chain.
- **DSL/control:** additive sidecar/`tau_pose` controls preserve the legacy
  zero-band path; no invented trainer flag or launch control was added.
- **Sensitivity/allocator:** custodied `q`, `Lip_local`, and `J_y` now feed the
  landed joint solver. The measured allocator consumes four curves but refuses
  a KKT allocation because the Seg secant is absent.
- **Autopilot/continual learning:** no dispatch is authorized; this memo and
  `research_only` lane state are the durable posterior. Linear and hard Pose
  interpretations are both emitted, with the hard oracle as disambiguator.

**Pointer delta:** none. `0.1910828242 [contest-CPU]` remains unchanged.

## Remaining blockers

- `[Darwin-arm64 CPU advisory]` is not `[contest-CPU]` Linux x86_64 or
  `[contest-CUDA]`; axis parity is unmeasured.
- Measured bytes are scorer-numerator residual bytes, not receiver parse-back
  `archive.zip` bytes. Receiver closure, archive custody, and exact contest eval
  are absent.
- The aggregate Seg field VJP is a first-order proposal surface, not complete
  independent per-cell nonlinear custody.
- The four admitted Seg observations are flat at `d_seg=0`; a traceable Seg
  secant is still owed before a non-null KKT allocation.
- Pair-11 and pair-18 refusals retain only their literal instance scopes.
- Canonical lane validation remains globally blocked by 110 historical
  missing-evidence paths outside this lane. This lane itself is registered
  `research_only=true`, marks only `impl_complete`, and computes to L1; the
  unrelated registry debt was not rewritten or suppressed.

## MAIN landing review required

MAIN must review the isolated branch diff before merge. Review emphasis:
(a) unit-normal/field-VJP semantics and the aggregate-field limitation;
(b) `J_y` consumption versus `J_x` custody; (c) immutable binding-map and resume
validation; (d) hard-oracle admit/reject behavior and exact fallback scope;
(e) the 96/96 hard-Pose inactivity interpretation versus the more conservative
linear proposer; and (f) the honest no-KKT/no-pointer verdict. Do not promote
these advisory range bytes into archive or contest-score evidence.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; delegated authority prompt SHA-256
`4cb7251603e1e992608f96cd506766ea2801214723057767bc5a8f463a602114`;
`docs/operating_manual_craft_handoff.md`; the three research/source authorities
named above; real `gt_n600.npz`; VJP/binding/measurement SSD artifacts; lane,
task, and progress state; per-arm inbox including the 2026-07-19T06:27:52Z
operator reframe; fleet broadcast inbox.
