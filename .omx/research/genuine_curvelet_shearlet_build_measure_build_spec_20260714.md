# Genuine localized-frame build and measurement spec (2026-07-14)

`research_only=false`; `operator_go_containment=true`; `paid_or_heavy_launch=false`.

## Objective

Replace the fake/global polar-Fourier interpretation of the existing basis lever with two real,
deterministic, spatially localized, parabolically scaled 80-channel dictionaries while preserving
the sealed 109,559-value witness budget. The curvelet branch is now a finite discrete periodized
polar curvelet-system truncation built from literal radial and angular frequency wedges, not the
invalid round-1 compact spatial bump around one carrier. The shearlet branch is a distinct finite
cone-adapted compact shearlet-system truncation. Wire both into the one typed `tac.witness_dsl`
basis lever, the live level-set trainer, and the generated NumPy inflate/byte-close path. Measure
the strongest honest no-training equal-value surfaces at `n600`; prepare but do not fire the
fresh-start training comparison.

The family IDs are the Task #497 IDs `windowed_curvelet` and `compact_shearlet`. These names denote
finite deterministic 80-column dictionaries/truncations. The curvelet exactness claim is limited
to the declared discrete period-two dictionary and its two equivalent evaluators. Neither family
is an exact continuum FDCT, a proved Candès-Donoho tight frame, a complete frame for the
196,608-dimensional raster space, or a dictionary with established frame bounds or continuum
approximation rates. Those claims remain forbidden unless separately proved.

## Round-1 invalidation and current structural status

The first `windowed_curvelet` implementation was a compact parabolic spatial bump multiplied by a
single oriented carrier. It distinguished itself from a global plane wave but did not derive its
atoms from radial-annular and angular frequency wedges. It is therefore invalidated under NO-FAKE
by `.omx/research/genuine_curvelet_spatial_window_dictionary_invalidation_20260714.json`; its
structural receipt and any comparison rank containing that curvelet row are historical only.

The replacement construction passed the round-3 adversarial structural gate **CLEAN**. Its durable
proof is
`.omx/research/genuine_curvelet_shearlet_structural_proof_v2_polar_frequency_wedge_20260714.json`
with SHA-256
`677a2252c43c1272ec0e2e83d65ce1b82d23b8ddb089d73a111a5f0b26d46d25`. This is a structural
receipt only: it makes no `n600`, through-R, archive-byte, score, launch, or pointer claim. Fresh
measurement is a separate active custody surface, so this spec records no comparison ranks.

## Authority and containment

- NumPy-fp32 is the portable/generation authority; MLX is a parity backend.
- A score claim requires exact receiver RGB, the real R operator, frozen CPU-torch SegNet, all 600
  pairs, and archive custody. MPS and mask-space disagreement are never scores.
- Direct one-hot/winner-rival fitting is a `MASK_SPACE_PROXY`, because no custodied inverse maps
  SegNet output targets to receiver-valid RGB. It must never be relabeled through-R.
- The receiver-valid no-training surface may rematerialize the saved owed16 OFF witness, perform
  equal-value N-term RGB approximation, and score reconstructed RGB through real R + CPU SegNet.
  Its verdict scope is `receiver_rgb_nterm_formulation`, not fresh-start training or family closure.
- Receiver authority uses the custodied CPU-torch scorer kernel at `receiver_batch_size=32` with
  its recorded source/checkpoint/deploy-int8 configuration. Two distinct predecessors remain
  invalidated: the v2 raw-fp32 EMA-shadow path used the wrong receiver parameters, while the later
  v3 deploy-int8 `receiver_batch_size=8` path used the right parameters but differed from the
  historical batch-32 scorer geometry by three razor-tie pixels. Preserve both as separate
  invalidation records; neither can be resumed, ingested, or mixed with the batch-32 receipt.
- No paid/GPU/heavy training launch. The ticket is dry-start-gated and `operator_go_required=true`.
- Do not touch `witness_autoconfig`, the shared config/provenance bijection, or pose-gate surfaces;
  `provenance_canonicalize_fix_all_fakes` owns them.

## Kernel contract

Owned files:

- `src/tac/boundary_math/localized_basis_frames.py`
- `src/tac/boundary_math/tests/test_localized_basis_frames.py`

Required API: `FEATURE_WIDTH=80`, deterministic atom metadata, NumPy/MLX feature evaluators,
anti-fake proof receipt, deterministic generated-inflate source/contract, and atom-spec SHA-256.
Directional atoms must have actual translations, parabolic width/length laws, scale-refining
orientations/shears, and either exact compact support (shearlet) or measured spatial concentration
and tail decay (curvelet). The proof must distinguish them from the legacy `sin/cos(X@B)`
constant-envelope global plane-wave bank. Existing polar-Fourier output is unchanged.

### Finite discrete periodized polar curvelet construction

For each registered scale/orientation, the curvelet frequency mask is exactly

```text
M_{j,l}(xi) = W_j(|xi|) V_j(wrap_pi(arg(xi) - theta_l)),  xi = q/2.
```

`W_j` and `V_j` are literal compact `cos(pi*s/2)^4` windows. The registered frequencies use the
half-cycle lattice `xi=q/2`, whose inverse discrete Fourier series has period two on the full
coordinate domain `[-1,1]`; using an integer-frequency lattice here would create a dishonest
second copy of the atom within the domain. The declared scales are `j=(0,2,4)`, with centers
`f_j=3*2^j`, radial half-widths `2*2^j`, orientations `(4,8,16)`, and rotated parabolic translation
lattice spacings `2^-j` normal to the atom and `2^(-j/2)` tangentially. This yields the expected
spatial aspect sequence `(1,2,4)` and refining angular half-widths `(pi/4,pi/8,pi/16)`.

The NumPy evaluator has two mechanically equivalent paths:

- arbitrary coordinates evaluate the sparse trigonometric polynomial directly;
- complete inclusive Cartesian grids alias-sum the registered spectrum, run the inverse FFT on
  the non-duplicated base grid, then copy the exact period-two endpoints.

The anti-fake proof fails closed unless the literal polar factorization, Hermitian/even wedge,
DC exclusion, radial/angular overlap, shrinking angular support, parabolic translation lattice,
measured spatial second-moment aspect, spatial energy concentration/tail decay, direction
alignment, direct-vs-FFT agreement, and period-two endpoint agreement all pass their declared
tolerances. Passing these gates proves only the finite discrete dictionary described above.

### Compact shearlet construction

The shearlet branch remains a finite cone-adapted compact shearlet-system truncation with atoms
`2^(3j/4) b'(u_n)b(u_t)`, parabolic dilation, integer shears, horizontal and vertical cones, and an
explicit cone-seam ownership policy. Its proof gates exact zero outside compact support,
parabolic support aspect, scale-refining shear directions, unique seam directions, translations,
and the normal vanishing moment. It does not establish continuum frame bounds, tightness,
completeness, or approximation rates.

### Portability and rule 118

NumPy-fp32 is the deterministic portable authority. Generated inflate source embeds the generic
family/seed/fixed atom construction and reproduces the NumPy reference; under rule 118 that generic
deterministic code is free while all learned/video-derived coefficients, downstream decoder
weights, and per-pair codes remain counted. An MLX implementation exists against the same atom
specifications, but Metal was unavailable on this host, so MLX parity is **UNMEASURED** and cannot
be inferred from NumPy or used as authority.

Acceptance:

```bash
python3 -m pytest -q src/tac/boundary_math/tests/test_localized_basis_frames.py
```

## Live-consumer integration contract

Owned surfaces for the integration unit:

- narrow hunks in `experiments/train_levelset_witness_realized_through_R_mlx.py`
- narrow hunks in `tools/levelset_byte_close_and_eval.py`
- narrow hunks in `src/tac/witness_dsl/optimal_basis_20260714.py`
- focused additions to `src/tac/tests/test_optimal_basis_20260714.py`

Add `--basis-family` and `--basis-frame-seed` as real trainer flags; default must reproduce the
legacy polar path. Every train-time feature construction and every ground/fine/AA evaluation path
must route through the selected family. Persist family and seed in complete/stage checkpoints;
categorical family drift is refused even for same-width weights-only resume. Generated inflate and
NumPy byte-close must use the same constants/order and persist family, seed, and atom-spec hash.
The sole typed `BasisLeverSpec` compiles the real argv and records the canonical #500 metric ID
`argmax_native_vjp_fidelity_v1` plus receipt schemas without duplicating the metric implementation.

Acceptance:

```bash
python3 -m pytest -q src/tac/tests/test_optimal_basis_20260714.py
python3 -m py_compile experiments/train_levelset_witness_realized_through_R_mlx.py tools/levelset_byte_close_and_eval.py
```

## Measurement contract

Owned files:

- `tools/probe_genuine_frame_nterm_n600.py`
- `src/tac/tests/test_probe_genuine_frame_nterm_n600.py`

The tool must provide:

1. structural anti-fake proof for both genuine families;
2. equal-value direct one-hot N-term decay at `n600`, explicitly `MASK_SPACE_PROXY`;
3. streaming receiver-valid approximation of the saved OFF witness RGB, with polar Fourier,
   windowed curvelet, compact shearlet, and a fixed target-independent comparison mix of four
   common Q1 columns, 38 polar-Fourier columns, and 38 windowed-curvelet columns;
4. real R + frozen CPU-torch SegNet measurement on exact reconstructed RGB for all 600 pairs;
5. deterministic receipt with source/checkpoint/model hashes, counts, support-index/metadata byte
   caveat, authority labels, and explicit verdict scope.

Use matrix-free/streaming fitting; never cache `600 x H x W x 80`. Equal `109,559` scalar values is
not equal archive bytes. The fourth arm is **not** a decoder-boundary partition-of-unity hybrid:
it is the fixed 4-Q1 + 38-polar + 38-curvelet column mix above, with no claimed interior/annulus
partition and no target-derived routing. Any target-derived boundary mask is oracle-only and
inadmissible. A literal Fourier-interior/curvelet-boundary-annulus PoU requires a counted,
self-consistent decoder `phi`/annulus state and separate equal-value **and** equal-byte custody;
the required source-phi mask is currently invalidated. This PoU formulation is fail-closed and
remains OPEN, not a negative verdict on the frame family. If a required receiver surface is
unavailable, emit a machine-readable blocker rather than fabricate RGB or a score.

Acceptance:

```bash
python3 -m pytest -q src/tac/tests/test_probe_genuine_frame_nterm_n600.py
python3 tools/probe_genuine_frame_nterm_n600.py --help
```

## Review and landing

The structural unit has a round-3 **CLEAN** gate against the legacy Fourier envelope and the
invalidated round-1 spatial wave packet. The broader landing review must still verify every feature
path uses the selected family, re-derive 71,159 decoder + 38,400 codes = 109,559 values, and check
every negative is formulation/instance scoped with an explicit optimal-form queue. Commit only
owned files through `tools/subagent_commit_serializer.py` with expected SHA-256 values. If the
sandbox blocks git object writes, emit the exact file allowlist and hashes for privileged harvest.
