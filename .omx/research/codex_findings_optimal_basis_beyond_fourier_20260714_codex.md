# Codex findings — optimal basis beyond Fourier — 2026-07-14

**Pointer status:** submittable `[contest-CPU]` **0.1910828242 UNCHANGED**;
non-submission defensive bank **0.1880443979880752 UNCHANGED**. No exact archive
row was created.

## Verdict first

The `$0` real-n600 through-R verdict already existed and was independently
re-derived from its preserved rows and checkpoints. It **does not re-confirm
the historical -48% directional prior**:

| matched ep675 arm | d_seg | trainable values incl. codes | delta vs OFF | verdict |
|---|---:|---:|---:|---|
| global polar directional Fourier, self-orient OFF | **0.004244** | **109,559** | — | **MEASURED winner at this formulation** |
| self-oriented Fourier, along=8 | 0.0042590586 | 111,095 | +0.00001506 (+0.35%) | no measured benefit |
| self-oriented Fourier, along=26 (3.2x-deficit reformulation) | 0.004286 | 111,095 | +0.000042 (+0.99%) | no measured benefit |

Axis: `[macOS-CPU advisory]`, frozen CPU-torch SegNet argmax after actual R,
real `n=600`, seed 0. Scope: **FORMULATION** — bounded warm-start from a parent
trained with self-orient ON. The fresh-start family remains **OPEN** because the
parent may have internalized directional structure. Noise floor is unmeasured;
the small within-instance ordering is not a family kill. The -48% claim is
nonetheless not transferred: it was a circular-GT `n96` direct-partition proxy,
not this realized surface.

Archive-byte equality is **NO-VERDICT**: owed16 preserved checkpoints, not
byte-closed archives. OFF has 1,536 fewer scalar values and a 6,144-byte smaller
FP32 NPZ, but neither is an archive-byte claim.

Durable receipt:
`.omx/research/optimal_basis_saved_n600_audit_20260714.json`. Reproducer:

```bash
PYTHONPATH=src .venv/bin/python tools/probe_optimal_basis_saved_n600.py
```

The probe is read-only. It runs no training, scorer forward, or evaluator and
does not load the 5,078,017,610-byte GT cache.

## Cargo-cult finding: the current `curvelet` bank is not a curvelet frame

`curvelet_directional_B` builds a polar set of frequency vectors. The feature
map is exactly `sin(2*pi*X@B) || cos(2*pi*X@B)`. For every column and every
coordinate, the paired envelope is `sin^2 + cos^2 = 1`; the measured numerical
envelope span is at most `1.462174630262325e-07`. There is no spatial window,
translation index, scale-dependent anisotropic support, or localized tight
frame normalization.

Therefore the honest label is **global polar directional Fourier**. Doubling
orientation count on alternate scales is curvelet-inspired angular sampling,
but does not create curvelet atoms. The previous naming over-credited the
implementation and made “Fourier vs curvelet” appear settled when no genuinely
different frame had been run.

## Ranked basis table

Ranks are **next-measurement priority**, not claimed score ranks. Numerical
d_seg appears only where measured; no percentage is guessed for an unrun frame.

| rank | candidate | fit to curved codim-1 anisotropy | byte cost | MLX portability | equal-budget d_seg |
|---:|---|---|---|---|---|
| 1 | hybrid Fourier interior + windowed curvelet boundary | Best synthesis: Fourier is hard-earned for smooth interiors; localized parabolic atoms spend capacity only on the annulus | generic windows/scales free; learned coefficients counted | primitives portable; implementation absent | **PREDICTED/UNMEASURED** |
| 2 | true windowed curvelet | Optimal N-term class for C2 curved edges; width approximately length squared | generic frame free; learned/selected coefficients counted | NumPy direct; MLX window/scatter owed | **UNMEASURED** |
| 3 | compact shearlet | Same curved-edge approximation class; shear indexing is GPU-friendly | generic frame free; learned coefficients counted | high after separable shear implementation | **UNMEASURED** |
| 4 | steerable/Gabor | Local tangent selectivity; fixed aspect ratio weaker than parabolic multiscale | generic kernels free; learned coefficients counted | high, dense elementwise ops | **UNMEASURED** |
| 5 | wavelet | Excellent interior/local multiresolution; axis-aligned atoms overspend on curves | transform free; retained coefficients counted | high, separable filterbank | **UNMEASURED** |
| 6 | B-spline/RBF | Local support avoids global ringing; adaptive boundary knots may cost bytes | generic knots free; video-derived amplitudes/locations counted | high for fixed knots | **UNMEASURED** |
| 7 | SIREN/FINER/FINER++ | Learned frequencies can adapt; global atoms and init saturation remain risks | init code free; learned weights counted | already compiler-portable | **UNMEASURED fresh-start** |
| 8 | hash-grid / Instant-NGP | Strong multiresolution locality; collision/table rate can dominate | learned hash tables counted | feasible; deterministic collision parity owed | **UNMEASURED** |
| 9 | Laplacian/geometric eigenfunctions | Metric/manifold adapted, but operator and boundary-condition dependent | eigenvectors counted unless analytically regenerated | dense matmul easy; eigencustody absent | **UNMEASURED** |
| 10 | NTK/Fisher-optimal features | Can align with reachable decision geometry; checkpoint/metric dependent | derived eigenfeatures generally counted | HVP/Lanczos feasible | **UNMEASURED** |
| 11 | spherical/Zernike | Global orthogonality mismatches planar perspective boundaries without a chart | generic functions free; coefficients counted | high | **UNMEASURED** |
| 12 | current global polar directional Fourier, OFF | Good smooth-interior fallback; global atoms are not boundary-local | bank free; existing weights/codes counted | **implemented train + inflate** | **0.004244 MEASURED**, scoped |
| 13 | self-oriented Fourier | Local tangent coordinates but still global phase/no window; measured memory tax | generic feats free; +1,536 decoder values; about +47-57 GiB live RAM | implemented, expensive | **0.004259 / 0.004286 MEASURED**, scoped |

## Typed DSL and inflate compilation

`tac.witness_dsl.optimal_basis_20260714.BasisLeverSpec` is the basis-family
stage/config surface and composes as an ordinary `Lever` with the existing
`n_dir_freqs`, directional-rebalance, and FINER flags.

| family | DSL compile | generated inflate compile | status |
|---|---|---|---|
| `polar_directional_fourier` | real `--bank-*`, `--max-bank-freq`, `--no-self-orient` | `_curvelet_B` + `_curvelet_feats` (legacy function names) | deterministic, rule-118 generic state free |
| `self_oriented_fourier` | adds real `--self-orient`, `--n-dir-freqs`, `--freq-*` | adds `_dir_feats` fixed point | deterministic; measured non-winner at scoped formulation |
| `siren_finer` | real periodic activation/SIREN/FINER init flags | existing decoder activation/weights | compile-supported; n600 basis verdict owed |
| true curvelet/shearlet/wavelet/Gabor/hash/spline/Zernike/Laplacian/NTK/hybrid | **REFUSE** | **REFUSE** | missing train+inflate op parity and equal-budget n600 receipt |

This confirms that the **measured fallback** compiles to `inflate.py` at
approximately zero generic-basis bytes. It does **not** falsely confirm that a
true curvelet compiles; that implementation is the next build.

## Canonical equation and basis-metric duality

`optimal_basis_equal_budget_through_r_v1` is implemented in
`tac.canonical_equations.optimal_basis_selection_20260714`:

`B*(K,A) = argmin_B d_seg(SegNet(R(G_{theta_B,B})))`

subject to parameter budget `K` and exact archive budget `A`. A row with
missing archive bytes cannot win an archive-constrained selection.

The #500 duality is explicit but non-owning: the basis supplies atoms `psi_i`;
the metric arm supplies the renderer/Fisher pullback `G_q`; the coupling is the
Gram `<psi_i, G_q psi_j>`. The live handoff identifies the canonical law as
`argmax_native_vjp_fidelity_v1`, its provider as
`tac.scorer_surrogate.vjp_fidelity`, the state receipt as
`reachable_decision_geometry_fidelity.v1`, the selector receipt as
`reachable_decision_preconditioner_selection.v1`, and the candidate
preconditioner as `winner_rival_margin_fisher_natural`. Full-n600 metric
selection remains `NO-VERDICT_DATA_CUSTODY`. A curriculum-varying basis is the
primal dual of a curriculum-varying metric, but this lane does not own or
implement metric annealing.

## Ranked next actions

1. **Build the actual family, not another label:** fixed-width (`in_feat=80`)
   localized curvelet and compact-shearlet feature ops with NumPy/MLX/generated-
   inflate parity and deterministic receiver tests. No learned frame tables.
2. **Pre-register a fresh-start three-arm disambiguator:** current polar Fourier
   vs windowed curvelet vs shearlet; same seed, schedule, optimizer, 109,559
   trainable values, stage checkpoints, and full real-n600 through-R verdict.
   This is a heavy run and remains operator-GO gated.
3. **Only if a localized frame wins:** byte-close all arms and compare exact
   archive bytes/runtime. Generic generator code is free; learned coefficients
   are counted. No pointer claim before exact contest-axis replay.
4. **Test the honesty-gate hybrid:** decoder-own coarse partition defines a
   deterministic boundary partition of unity; low global Fourier modes serve
   interiors and localized parabolic atoms serve the annulus. Match total
   feature width and counted parameters.
5. **Consume #500, do not duplicate it:** rank atoms using the registered
   pullback Gram once the metric provider publishes its API/receipt.

## Verification

```text
9 tests passed
saved-artifact probe exit 0
legacy atom envelope span <= 1.47e-7
no heavy launch / scorer forward / evaluator run
```

Triality is complete as an additive research-only landing: standalone DAG FEED,
typed DSL, canonical equation, probe, and regression tests. Pointer remains
unchanged.

## Landing status

The equation was appended through the locked canonical registry helper as
`optimal_basis_equal_budget_through_r_v1`. The required commit serializer was
attempted with post-edit SHA-256 guards and `dag,dsl,equations` triality. It
failed at `git add` with rc 128: `unable to create temporary file: Operation
not permitted`; no commit SHA exists. No bypass or direct commit was attempted.
All owned files remain in the shared working tree for privileged harvesting.
