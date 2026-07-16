# Curvelet optimal-form crux — implementation and measurement spec (2026-07-15)

`research_only=false`; `lane_id=curvelet_optimal_form_crux_20260715`;
`pointer_move_authority=false`; `paid_or_remote_launch=false`.

## Objective and authority

Recover the clean finite polar-frequency-wedge curvelet as executable source, compose it with the
level-set witness's boundary-placement operators, close every train/checkpoint/NumPy/MLX/generated-
inflate consumer, and compare it with the legacy polar-Fourier control at **identical final
`archive.zip` bytes** through the real `R` operator and frozen SegNet on all 600 pairs.

The authority row is the full decoded archive, not the N-term proxy and not the historical
saved-OFF receiver row.  The prior `d_seg=0.5048239560` curvelet row is retained only as
`MEASURED_ADVISORY / receiver_rgb_nterm_formulation / equal_values_not_equal_bytes`; it is not a
family verdict.  The score pointer `0.19108/0.18804` remains unchanged unless the canonical exact
contest evaluator independently authorizes a move.

## Stores consulted and source recovery

- `experiments/train_levelset_witness_realized_through_R_mlx.py`
- `tools/levelset_byte_close_and_eval.py`
- `src/tac/boundary_math/windowed_curvelet_frame.py`
- `src/tac/boundary_math/ground_frame_chart.py`
- `src/tac/boundary_math/dseg_aware_fourier_taper.py`
- `src/tac/witness_dsl/optimal_basis_20260714.py`
- `.omx/research/genuine_curvelet_shearlet_build_measure_build_spec_20260714.md`
- `.omx/research/genuine_curvelet_shearlet_structural_proof_v2_polar_frequency_wedge_20260714.json`
- `.omx/research/genuine_curvelet_shearlet_receiver_n600_receipt_v4_polar_frequency_wedge_deploy_int8_batch32_20260714.json`
- `.omx/research/genuine_curvelet_spatial_window_dictionary_invalidation_20260714.json`
- `.omx/research/codex_findings_c1_deepmath_integration_20260715_codex.md`
- live inbox through `2026-07-14T20:32:37Z`

The receipt-custodied original file
`src/tac/boundary_math/localized_basis_frames.py` (source SHA-256
`8a2e8befde890f769997f5efdd917cb0eee52219c7e86393dc016604b8674697`) is absent from every
reachable ref and from the Git object store.  Recovery therefore means a fresh, reviewable
reimplementation from the sealed finite-dictionary contract below.  It MUST reproduce the sealed
atom metadata/proof invariants; it MUST NOT claim byte identity with the lost source unless its
source hash actually matches.

The checked-in `windowed_curvelet_frame.py` spatial Gabor packet stays historical and invalidated
as a literal curvelet.  Do not silently change the meaning of a resume checkpoint carrying
`basis=windowed_curvelet`.  The recovered implementation uses the new honest family identifier
`literal_polar_curvelet` and a versioned atom-spec hash.

## Crux verdict

`VERDICT_SCOPE=DERIVED_CURRENT_OPERATOR_GRAPH_ONLY; LITERAL_CURVELET_SCORE_FAMILY_OPEN`

The blanket guard is **over-broad**.  Parabolic scaling and shearing do not conflict with placement.
The live implementation has narrower defects:

1. the windowed MLX tensor is built before taper/IPE and selected unchanged;
2. ground-chart recomputation drops taper and, with self-orientation, drops the directional width;
3. supersample hardcodes the legacy Fourier feature builder;
4. scalar IPE attenuation `exp(-2*pi^2 b^T Sigma b)` is exact only for a global plane wave;
5. generated inflate persists no chart/taper/AA operator state;
6. the clean literal source is absent.

Items 1–3 and 5 are implementation/custody gaps.  Item 4 is a genuine conflict only for that
**scalar-IPE formulation**, not for curvelets or output-space supersampling.

## Selected mathematical form

Let `x` be frame coordinates and `y = H_p(x)` the per-pair ground-chart pullback.  With
`J_p = D H_p(x)`, boundary tangent vectors and normal covectors transform differently:

```text
t_y = J_p t_x / ||J_p t_x||,
n_y = J_p^(-T) n_x / ||J_p^(-T) n_x||.
```

For a literal curvelet column `lambda=(j,l,k)` with direction `theta_lambda`, use decoder-native
orientation gating instead of appending another directional Fourier bank:

```text
q[p,lambda](x) = exp(kappa*cos(2*(theta_lambda-arg(n_y)))) / Z[p,j](x),
Z[p,j](x)^2 = sum_{lambda' in directional scale j}
              exp(2*kappa*cos(2*(theta_lambda'-arg(n_y)))).
```

Scaling columns have `q=1`.  The fixed point starts with `q=1`, decodes its own argmax, derives
normals, updates `q`, and stops on argmax equality or the persisted iteration cap.  No GT label or
normal is available to inflate.

The ordered feature program is

```text
Phi_p(x) = D_w Q_p(x) Psi_literal(H_p x)
Y_p      = R[A_s G_theta(Phi_p(X_s), z_p)].
```

`D_w` is the existing per-column taper.  It is structurally compatible with any fixed dictionary,
but it is GT-derived at compression time.  To avoid free-data leakage, fold it exactly into the
first linear layer at deploy:

```text
W_in,deploy[:,lambda] = W_in,train[:,lambda] * w_lambda;
Phi_deploy            = Q_p Psi_literal(H_p x).
```

Thus `W_in,train (D_w v) == W_in,deploy v` up to the declared fp32 tolerance, and no taper vector is
needed by inflate.  The fold hash and parity receipt are mandatory.  Resume keeps the taper config
and unfolded training weights; deploy checkpoints are explicitly marked folded so a second fold is
refused.

`A_s` is the exact post-render `s x s` box average (`A_1=identity`).  Supersampling is universally
well formed but must follow the nonlinear renderer.  Scalar Fourier IPE remains fail-closed for
`literal_polar_curvelet`; a future wedge-integrated IPE is a different formulation.

Coefficient-space placement was rejected because `C H* C^dagger` is generally dense.  A separate
post-placement residual payload is reserved only if direct chart composition measures a residual
worth its rate.  Native directional gating is selected over concatenated self-orientation because
it preserves the same 80 input columns/learned parameter count as the control and does not duplicate
orientation degrees of freedom.

## Literal finite-dictionary contract

Implement `src/tac/boundary_math/localized_basis_frames.py` with NumPy-fp32 authority and an MLX
mirror.  The family has exactly 80 scalar columns: four common Q1 scaling columns and 76 directional
columns.

For scale/orientation:

```text
M[j,l](xi) = W_j(|xi|) V_j(wrap_pi(arg(xi)-theta_l)),  xi=q/2.
```

`W_j` and `V_j` are compact `cos(pi*s/2)^4` windows.  Use scales `j=(0,2,4)`, radial centers
`3*2^j`, radial half-widths `2*2^j`, orientations `(4,8,16)`, and rotated translations with normal
spacing `2^-j` and tangent spacing `2^(-j/2)`.  The half-cycle frequency lattice gives period two on
`[-1,1]`.  Arbitrary coordinates use the sparse trigonometric polynomial; a complete inclusive
Cartesian grid may use the mechanically equivalent alias-summed inverse FFT and exact copied
period-two endpoints.

Required proof gates include literal polar factorization, even/Hermitian wedge, DC exclusion,
radial/angular overlap, shrinking angular support, translation lattice, measured aspect `(1,2,4)`,
energy concentration/tail decay, direction alignment, direct-vs-FFT parity, and inclusive endpoint
parity.  The reimplementation must publish its own atom-spec/source hashes.  It may compare its
numeric proof values with the sealed receipt, but may not reuse the lost source's hashes as if
reproduced.

## Runtime and custody design

One versioned `BasisProgramConfig` must be consumed by trainer NumPy, trainer MLX, resume, deploy
fold, byte-close oracle, and generated inflate.  It contains:

- `family`, `basis_version`, `atom_spec_sha256`, `feature_width`, `seed`;
- chart on/off, reference pair, regime, calibration, and pose-section dependency;
- native orientation on/off, `kappa`, fixed-point iteration cap, normal-estimator version;
- taper on/off plus train-config hash and deploy-fold receipt hash;
- AA mode/factor.

The full config and its canonical SHA-256 are persisted.  Resume refuses semantic/version/hash
drift even if tensor shapes match.  Generated inflate refuses an unknown atom-spec hash.

Ground chart at decode MUST derive from counted receiver state.  Preferred closure is decoded pose-
carrier `xi`: add a `GroundFrameChart.build_from_xi` path that composes the same plane homographies
directly.  `--ground-frame-chart` therefore fails closed at byte-close unless the archive has the
required pose-carrier section or an explicitly counted chart payload.  Both A/B arms use identical
pose bytes and chart config.

## Exact equal-byte receiver

Equal learned scalar count is necessary but insufficient.  Both basis families expose exactly 80
input columns so all learned tensor shapes match.  After each arm is independently byte-closed and
parse-back verified:

1. construct deterministic copies of both `archive.zip` files;
2. add a clearly named, `ZIP_STORED`, fixed-timestamp `rate_match.bin` entry to each archive;
3. choose payload lengths so the two final ZIP sizes are exactly equal;
4. record pre/post archive SHA-256, member hashes, padding bytes, and target size;
5. inflate both matched archives and prove every decoded frame SHA is unchanged from its unpadded
   source archive.

The padding is counted experimental rate matching, never a candidate optimization and never hidden.
The receiver must tolerate but not consume it.  The receipt must state `equal_archive_bytes=true`,
not merely equal values/support.  A byte difference of one is a failed comparison.

## Owned implementation surfaces

Kernel unit:

- `src/tac/boundary_math/localized_basis_frames.py`
- `src/tac/boundary_math/tests/test_localized_basis_frames.py`
- `src/tac/boundary_math/curvelet_placement.py`
- `src/tac/boundary_math/tests/test_curvelet_placement.py`

Consumer unit:

- narrow hunks in `experiments/train_levelset_witness_realized_through_R_mlx.py`
- narrow hunks in `tools/levelset_byte_close_and_eval.py`
- narrow hunks in `src/tac/boundary_math/ground_frame_chart.py`
- `src/tac/witness_dsl/optimal_basis_20260714.py`
- focused tests under existing test modules

Rate/measurement unit:

- `src/tac/through_r/equal_archive_budget.py`
- `src/tac/through_r/tests/test_equal_archive_budget.py`
- a governed receipt driver under `tools/` that refuses non-n600, non-parseback, non-equal-byte, or
  non-batch32 scorer custody.

Triality:

- typed DSL `literal_polar_curvelet` basis/placement lever with real argv and LawRef;
- canonical equation for the transfer verdict, populated only from the final receipt;
- DAG FEED with explicit verdict scopes and pointer delta.

## Required tests before any long run

1. literal structural proof and fixed atom ordering/hash;
2. NumPy direct/grid parity and NumPy/MLX parity;
3. current legacy Fourier path byte identity when the new family is OFF;
4. identity chart and AA `s=1` byte identity;
5. affine/projective chart tangent-vector and normal-covector covariance;
6. nontrivial native orientation changes only directional columns and fixed-point argmax agrees
   across NumPy/MLX/generated inflate;
7. nontrivial taper train/deploy fold equality and double-fold refusal;
8. supersample uses selected basis and generated inflate reproduces it;
9. scalar IPE + literal curvelet strictly refuses;
10. checkpoint/resume refuses family/version/atom/operator-hash drift;
11. receiver mutation test: changing literal atom/config changes decoded frames;
12. deterministic equal-archive-byte padding preserves every inflated frame hash;
13. synthetic full-pipeline NumPy/generated-inflate exact uint8 equality.

## Measurement gate and verdict ladder

Run storage waterfall first and keep outputs on `/Volumes/VertigoDataTier/pact` when available.  A
training launch must use the governed launcher, fixed seed, stage-preserved atomic checkpoints, and
resume from disk.  No paid/remote dispatch is authorized by this spec.

The final n600 receipt must bind both exact archives, source/config hashes, stage checkpoints,
hardware/axis, exact command, batch-32 CPU-torch scorer custody, all decoded frame hashes, component
counts, archive bytes, and pointer delta.  Report one of:

- `MEASURED_TRANSFER`: literal optimal form has lower `d_seg` at identical archive bytes;
- `MEASURED_NO_TRANSFER / FORMULATION x INSTANCE`: no lower `d_seg`; family remains open unless the
  pre-registered optimal form and all custody gates were actually satisfied;
- `BLOCKED_FAIL_CLOSED`: name the first missing producer, data, runtime, governor, or custody gate.

No toy/small-n or stripped negative may be promoted to a curvelet-family `NO-GO`.
