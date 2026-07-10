# ADVISORY — v7.5.3 separate texture-trunk audit — 2026-07-10

`research_only=true`

**Disposition:** v7.5.3/A3 is **DESIGN/BUILD-ONLY; HOLD any launch or scored A/B**. v7.5.2 is
unaffected while `--texture-trunk` stays OFF. The current “BUILT” label is too strong because the
training, verdict, archive, and shipped receiver do not implement one texture-enabled vehicle.

This is an advisory means artifact. No training, dispatch, frontier move, or task-code mutation was
performed.

## Answer first

The separate-trunk hypothesis is worth testing: use G for low-description-length class geometry and
T for scorer-legible texture that does not deserve geometric capacity. The present implementation
cannot test that hypothesis. It trains a texture addend in MLX, omits the addend from every deploy
forward, counts a large deterministic bank as payload, and does not implement the home law described
in the v7.5.3 design. A green parity test can currently certify the wrong, trunk-blind decoder.

## P0 findings

### P0-1 — MLX training, deploy verdict, and inflate are three different vehicles

MLX adds the texture trunk in
`experiments/train_levelset_witness_realized_through_R_mlx.py:1370-1380`. Both sync/async realized
verdict paths call `levelset_rgb_forward_numpy` (`:5832-5844,6672-6688`). That claimed one-codepath
forward has no texture branch at
`src/tac/boundary_math/lever_b_levelset_generator.py:717-780`; the band forward at `:783-846` is also
trunk-blind. The inline shipped receiver omits T at
`tools/levelset_byte_close_and_eval.py:1112-1163`.

The supposedly independent oracle calls the same omission at `tools/levelset_byte_close_and_eval.py:
1883-1968`. Therefore MLX can train nonzero texture coefficients while the realized verdict and
inflate ignore them, and parity can still pass because both comparison sides share the same omission.

**Required falsifier:** set a synthetic nonzero `tex_trunk.w_tex`, zero `out_tex`, and require MLX,
canonical NumPy fp32, int8-dequant oracle, and fresh-process inflated `.raw` to show the same known
change through R. One manifest-aware deploy forward must serve sync, async, byte-close, and inflate.

### P0-2 — byte close counts the fixed bank, then inflate ignores it

`_load_levelset_ckpt` loads every non-`__` array (`tools/levelset_byte_close_and_eval.py:259-328`).
`build_levelset_blob` excludes pose-carrier fields but does not exclude names ending `_B`
(`:397-407`). The canonical in-memory quantizer does exclude deterministic banks at
`src/tac/boundary_math/lever_b_levelset_generator.py:701-714,956-975`, so accounting and shipping use
different definitions. The builder reports `accounting_matches_canonical` but does not refuse
(`tools/levelset_byte_close_and_eval.py:471-475`).

On the actual 384×512×24 bank, the read-only audit measured:

- 18,874,368 fp32 checkpoint bytes;
- 4,718,592 raw int8 bytes;
- 430,878 bytes after isolated Brotli-q11;
- approximately 0.286904 score units in rate alone.

That is larger than the whole current frontier score. Inflate then ignores both the bank and the
counted texture parameters.

**Required gate:** no deterministic bank samples or `_B` keys in the archive; count only learned
coefficients plus exact regeneration metadata; fail closed on accounting mismatch; prove those
coefficients change receiver pixels.

### P0-3 — the implementation does not satisfy the v7.5.3 texture home law

The full-stack design homes T on frame 1 in a grid-aligned, luma-null, chroma-high-frequency
subspace pre-imaged through exact D
(`.omx/research/fullstack_fractal_optimal_synthesis_20260710.md:61-70,93-104`; canonical equation
`src/tac/canonical_equations/fullstack_home_assignment_20260710.py:11-24`).

The current trunk emits unconstrained RGB, runs for every frame/code, and is added before sigmoid
(`src/tac/boundary_math/texture_trunk.py:343-356`; trainer `:1370-1383`). It has no frame-1 gate,
luma-null parameterization, exact-D preimage, or luma fallback switch. It can therefore spend bytes
in the Seg-dead frame0 subspace and in Pose-sensitive luma.

**Required falsifier:** T contribution exactly zero on frame0; exact post-D/R 384-grid luma null;
measured U/V block behavior; n600 d_pose non-inferiority; and a scorer-legibility gate for Seg.
Until then the generic RGB trunk is a diagnostic formulation, not v7.5.3.

## P1 findings

### P1-1 — the event curriculum and optimizer geometry do not exist

The design requires island-birth engagement plus T-specific learning rate/decay
(`fullstack_fractal_optimal_synthesis_20260710.md:172-184`). Current code has constructor/forward
flags and a DSL factory (`src/tac/witness_dsl/curriculum_dsl.py:1726-1752`), but T is active from
initialization with nonzero random coefficient scale and shares the global optimizer.
`TextureTrunk(window=...)` changes epoch count, not engagement.

Require pre-event output and gradient exactly zero; a persisted event state; zero-to-active amplitude
ramp; and separate optimizer-group state that round-trips through every checkpoint.

### P1-2 — the claimed matched-byte A/B does not exist

There is no `derive_crucible_v753_config`, and the A2 matched-capacity control remains owed
(`fullstack_fractal_optimal_synthesis_20260710.md:232-260,405-409`). The design memo sizes an A2 head
using hidden=256 while the canonical G trunk uses hidden=96. Replacing a 96→3 head with
96→H'→3 changes the parameter count by `100H' - 288`; H'=6 and 7 produce +312 and +412 parameters,
not an exact +375. Raw parameter equality would still not imply equality after tensorwise scaling and
Brotli.

Compile every arm and match **exact archive.zip bytes** within a preregistered tolerance. A zero-init
parallel residual bottleneck provides finer capacity increments than replacing the head.

### P1-3 — texture bias duplicates the palette gauge

Before sigmoid,

```text
soft @ palette + soft @ texture_bias = soft @ (palette + texture_bias).
```

The equations are visible at `src/tac/boundary_math/texture_trunk.py:281-284,343-350` and trainer
`:1373-1380`. The 5×3 bias is an exact duplicate home, adds a Jacobian null direction, and contaminates
the capacity match. Remove it or impose an explicit gauge constraint. Add the metamorphic test
`palette += delta; bias -= delta => identical output`.

### P1-4 — parameter separation does not imply optimization separation

T reads differentiable `softmax(phi/tau)` for placement
(`src/tac/boundary_math/texture_trunk.py:14-19,347-355`). Its loss therefore backpropagates into G.
The current formulation is **coupled placement**, not G×T optimization decoupling.

Ship both explicit modes for the disambiguation probe: coupled masks and stop-gradient masks. Report
G-gradient cosine/norm, per-class d_seg, island births, d_pose, and bytes.

### P1-5 — A2 vs A3 confounds basis with temporal conditioning

A3 is a stationary, image-coordinate, per-video bank. A2 is pair-conditioned through the shared
trunk/FiLM. The design also describes T as ego-advected by xi, but current T never reads code or xi.
Compare stationary and xi-coordinate-warped T at matched exact bytes; already-counted xi can advect
coordinates for free, with a residual phase admitted only if it earns its bytes.

### P1-6 — the `[4,8]` passband is an empirical hypothesis, not a theorem

A stride-2 stem establishes a sampling boundary; it does not establish period 8 as a hard lower
frequency cutoff. The current “Gabor” bank is actually a global sinusoidal/Fourier bank without a
spatial envelope (`src/tac/boundary_math/texture_trunk.py:146-172`). Its band report checks peak
frequency with a tolerance, not total out-of-band energy (`:178-212`). Read-only finite-grid checks
found feature means up to 0.00293 and leakage outside the tolerance-period band up to 0.00513, so
“exact zero mean” is also too strong.

Use scorer-JVP spectral response across period, orientation, channel, class, and scene context, then
waterfill frequencies. Do not promote a single hard-fill price list into a universal transfer law.

### P1-7 — the annulus guard is neither geometric nor resume-complete

`annulus_power` defaults to zero, and the implemented guard is confidence/temperature based rather
than signed distance based (`src/tac/boundary_math/texture_trunk.py:241-253`; DSL `:1726-1749`). Its
amplitude changes under a tau sweep even when phi is fixed, and confident wrong pixels are not
protected. The annulus settings are not covered by `_resume_lever_divergences` in the trainer
(`:827-930`), so resume can silently change render semantics.

Require a fixed-phi tau sweep, correlation with actual SDF boundary distance, per-class harm caps,
and fail-closed resume checks for annulus/routing changes.

## Additive formulations to test after P0 closure

1. **Exact evaluator-null texture basis.** Parameterize local RGB perturbations in the nullspace of
   PoseNet's YUV6 preprocessing, then solve the camera-grid preimage through exact resize/uint8/R.
   This is stronger than hoping generic chroma HF is Pose-invisible.
2. **Localized complex Gabor/WIRE control.** WIRE supplies spatial-frequency localization and avoids
   the global ringing of a Fourier bank. Primary source and official code:
   [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Saragadam_WIRE_Wavelet_Implicit_Neural_Representations_CVPR_2023_paper.html),
   [official repository](https://github.com/vishwa91/wire).
3. **Fourier frequency waterfill.** Use the tunable bank from
   [Fourier Features](https://proceedings.neurips.cc/paper_files/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html)
   as a measured control, not as proof of a cutoff.
4. **SIREN initialization/control.** [SIREN](https://proceedings.neurips.cc/paper_files/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html)
   is useful for periodic initialization, but changing the shared activation is not a clean T-only arm.
5. **Low-rank learned coefficient payload.** After training, SVD the 24×15 coefficient matrix. Rank r
   costs roughly 39r raw int8 coefficients versus 360; r=4 is 156. Admit only on exact archive delta
   and scorer parity, after removing the 15-parameter gauge bias.

## Smallest convincing proof matrix

1. OFF path: argv, checkpoint, and inflated bytes unchanged.
2. ON unit: a nonzero T causes a known output; bias-gauge and fixed-bank regeneration tests pass.
3. Integrated forward: MLX ↔ NumPy parity for both frames, with coupled/stop-gradient mode explicit.
4. Archive: `_B` absent; exact bank metadata counted; accounting mismatch refuses; fresh-process
   inflate is bit-identical to oracle.
5. Resume: stage checkpoint reload produces the same next loss/frame; missing or changed routing,
   annulus, event, or optimizer-group state refuses.
6. Home law: frame0 T=0; frame1 lies in the intended exact evaluator subspace after D/R.
7. Science: A1/A2/A3 exact-byte matched, paired seeds, n600 through-R; report per-class d_seg, island
   birth, d_pose, and archive bytes. Verdict scope is formulation-only.

## Scorer-derived additions — frozen evaluator/video pass

The exact scorer geometry strengthens the exact-D formulation and falsifies any generic
"high-frequency chroma is Pose-null" shortcut.

1. **Frame-role law.** SegNet reads only frame1, while PoseNet reads both frames. Therefore frame0 T
   must be identically absent unless it earns exact Pose improvement; frame1 T carries both Seg and
   Pose obligations. A symmetric two-frame texture renderer pays an unnecessary Seg representation
   on frame0.
2. **Exact local preprocessing kernel.** At the 384x512 scorer grid, away from clamp/round
   discontinuities, require per pixel
   `0.299*dR + 0.587*dG + 0.114*dB = 0` and per 2x2 block
   `sum(dR)=sum(dB)=0`. The six atoms made by the three zero-sum Haar patterns crossed with
   `c_U=(0,-0.3441362862,1.772)` and `c_V=(1.402,-0.7141362862,0)` span the six-dimensional local
   kernel. This is the candidate home law; the current generic period-[4,8] RGB bank is not proven
   to inhabit it.
3. **Projection order is binding.** Soft class placement, annulus gates, unequal sigmoid
   derivatives, bicubic camera lift, evaluator bilinear resize, uint8, and clamp can all leak out of
   that kernel. Compose placement first, project after the final nonlinear RGB construction, solve
   a camera-grid preimage, and require the first six Pose outputs bit-stable after fresh `.raw`
   reload.
   The evaluator preimage itself is block-separable: each scorer RGB sample has a disjoint 2x2
   camera footprint. Solve bounded integer 4-to-1 lifts per sample/channel; a global bicubic lift is
   an optional renderer constraint, not the scorer inverse.
4. **Texture effects are not proven spatially independent.** SegNet's SqueezeExcite/deep-U-Net path
   is source-level global. A recovered margin-Jacobian summary reports nonzero full-input support and
   tail energy outside radii 64/128/192, but its raw receipt is missing. Reproduce a remote-patch
   block response matrix and decoded composite interaction receipt; a class mask or annulus is an
   empirical routing device, not a scorer factorization.
5. **Byte admission remains severe.** One archive byte costs `6.6585895312e-7` score and must save
   more than 0.785479 Seg cells absent Pose. The 430,878-byte deterministic bank costs about
   0.286904 score if shipped. If it is fixed/generic, regenerate it as receiver code; if it is
   video-derived, count and consume it. Counting it and ignoring it remains a hard refusal.
6. **Raw blind coordinates are not texture homes.** 230,904 camera pixels per frame never enter
   either scorer resize and may be filled generically; a learned/video-derived T there cannot earn
   score. Preserve the strict 3,662,409,600-byte / 1,200-frame raw cardinality guard so this exact
   sparsity cannot be confused with a short-raw evaluator truncation.

**Scorer-derived launch disposition:** **DESIGN/BUILD-ONLY unchanged.** The exact-D basis is a new
replacement formulation; it does not repair MLX/NumPy/inflate divergence, event/resume state, or
matched-byte controls.

## Stores consulted

- `.omx/research/fullstack_fractal_optimal_synthesis_20260710.md`
- `.omx/research/fable_synthesis_texture_partition_20260710.md`
- `.omx/research/texture_trunk_p0_design_20260710.md`
- `src/tac/boundary_math/texture_trunk.py`
- `src/tac/boundary_math/lever_b_levelset_generator.py`
- `src/tac/witness_dsl/curriculum_dsl.py`
- `experiments/train_levelset_witness_realized_through_R_mlx.py`
- `tools/levelset_byte_close_and_eval.py`
- `.omx/research/ADVISORY_evaluator_video_geometry_20260710.md`
- focused texture, deploy, checkpoint, and byte-close tests
- the primary sources linked above

**Pointer delta:** none. v7.5.3 has no scored or receiver-closed candidate.
