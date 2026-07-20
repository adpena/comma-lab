# Codex findings — R1b boundary-generator solve — 2026-07-20

`lane_id=r1b_boundary_generator_solve_20260720T161946Z` · `$0 local` · `research_only=true` · no training/provider dispatch · pointer unchanged

## Verdict

**MEASURED N600 CONTROL; R1B CANDIDATE ABSENT.** The exact existing C2 production receiver archive is `94,344 B` and therefore passes both the task byte gate (`<=286,680 B`, margin `192,336 B`) and the fixed-C1 cap (`<=216,223 B`, margin `121,879 B`). Its full n600 hard CPU-Torch through-R row on the local advisory axis is `d_seg=0.003515794640406966`, `d_pose=127.36588287353516`, and `S=36.10275630841103`. The Seg gate `d_seg<=0.000339` fails by `0.0031767946404069663`, or `10.3710756354x`; the joint gate is false.

This is a **production-control result only**, not an R1b candidate and not a family negative. The archive contains no counted boundary-coordinate packet and no `xi` pose receiver. The exact current blocker is:

`R1B_COUNTED_MODERATE_MARGIN_BOUNDARY_GENERATOR_XI_INTERSECTION_UNMEASURED`

Machine-readable receipt: `.omx/research/r1b_boundary_generator_solve_20260720.json`.

## Exact n600 row and custody

| term | measured value | authority |
|---|---:|---|
| archive | `94,344 B` | exact ZIP bytes; SHA-256 `d633e6bfbbb5963b638f6f469ed1298ac86dbe3e04e5eae1b06b08cf64397539` |
| pairs / batches | `600 / 38` | batch size `16`, CPU threads `8`, seed `1234` |
| `d_seg` | `0.003515794640406966` | hard CPU-Torch, exact decoded uint8 through R, `[macOS-CPU advisory]` |
| `d_pose` | `127.36588287353516` | hard CPU-Torch, exact decoded uint8 through R, `[macOS-CPU advisory]` |
| Seg component | `0.3515794640406966` | `100*d_seg` |
| Pose component | `35.688357047296975` | `sqrt(10*d_pose)` |
| rate component | `0.06281979707335814` | `25*94344/37545489` |
| advisory action | `36.10275630841103` | component sum; no contest score claim |
| decode / hard score / total | `738.9803 / 487.2737 / 1234.0451 s` | measured wall clock; decode workers `8` |

The receiver produced `3,662,409,600 B` of raw n600 uint8 output with SHA-256 `dbfcdcfa9c2ea361cfa51eb6b6e26379b20ad5591fb0fe399ace496315628a97`. The canonical NumPy scorer-input bytes and receiver parse-back bytes both hash to `f09527720969b6552a29db13ff68efe3fd55c908ebac44dd829cfa6b3ec3f6f8`; factor-2 realization is exact. Storage preflight required `9,324,819,200 B` and measured `779,043,606,528 B` free on the SSD tier. Success scratch was removed only after the durable receipt captured command, source hashes, runtime, and cleanup proof.

This axis is deliberately advisory: `torch=2.12.1`, macOS arm64 CPU. It does not move or reproduce the contest-CPU pointer.

## Per-term byte and debt decomposition

The archive is already rate-cheap at `157.24 B/pair`; the task and fixed-C1 caps allow `477.8` and `360.3717 B/pair`, respectively. Its exact compressed sections are:

| section | compressed bytes | archive share | role |
|---|---:|---:|---|
| `0.bin` | `83,730` | `88.7497%` | counted base-video payload |
| `ipe_manifest.json` | `1,036` | `1.0981%` | counted manifest |
| `seg_head_target.pdw2` | `105` (`138` raw) | `0.1113%` | counted conditioning only; not a spatial receiver |
| `ipe_codes.f16` | `8,874` | `9.4060%` | counted residual/video codes |
| `ipe_quotient_residual_head.f16` | `27` | `0.0286%` | counted residual head |
| ZIP overhead | `572` | `0.6063%` | counted container |

The aggregate `d_seg` corresponds to approximately `414,740` mismatched scorer pixels (`DERIVED`, nearest integer from `d_seg*600*384*512`). Reaching the task gate requires a `90.3578%` Seg-debt reduction at the current denominator. The archive has ample gross rate headroom, but none of that headroom is yet a legal measured R1b carrier.

The missing intersection decomposes into four independently checkable terms:

1. `R1B_N600_COUNTED_BOUNDARY_PACKET_ABSENT_FROM_PRODUCTION_ARCHIVE` — the implemented packet/parser/generic localized receiver exists in code, but the measured ZIP does not contain it.
2. `R1B_MODERATE_MARGIN_FIRST_ORDER_PLUS_REALIZED_SECANT_JACOBIAN_CUSTODY_ABSENT` — the deterministic corrected-QP engine is built; production batch-16 Jacobian and secant values are not yet materialized.
3. `R1B_FULL_KERNEL_MIN_DESCRIPTION_SELECTION_NOT_YET_COMPILED_INTO_PRODUCTION_ARCHIVE` — the full compiler is now mandatory in the solve, but its selected preimages are not represented in an exact production packet.
4. `R1B_FULL_KERNEL_SELECTION_RECEIVER_RUNTIME_AND_COMPACT_REPLAY_NOT_CLOSED` — the imported constant-preference measurement took `34.7690 s` for one frame. `600x` is `20,861.4 s` (`DERIVED`, not a production timing row), so replaying the bounded search per frame in `inflate.py` would not be admissible. The selected null coordinates need a compact counted replay grammar.
5. `R1B_XI_POSE_RECEIVER_ABSENT_FROM_THIS_CONTROL_ARCHIVE` — no pose sidecar or `xi` manifest is present, so pose savings and bytes are both unclaimed.

## Landed solve boundary

`src/tac/optimization/boundary_coordinate_joint_solve.py` now provides the reusable fail-closed solve boundary:

- a strict counted `boundary_coordinate_packet.v1` parser/encoder with shared atom indices, n600-capable per-pair/channel int8 coefficients, float16 scales, canonical hashes, CRC, and no Fourier alias;
- receiver regeneration through the existing genuine `windowed_curvelet` or `compact_shearlet` frame code;
- corrected first-order plus mandatory secant weighted least-norm QP with deterministic dual active-set KKT diagnostics;
- exact bounded uint8 factor-2 realization and verification inside candidate evaluation;
- mandatory `FullResizeKernel` minimum-description preimage selection, followed by a second exact projection check and then a fresh hard oracle;
- the contained ERM route only for `STALLED/CYCLE/BUDGET` unknown: exactly `4x16` cheap Fisher/margin evaluations, one independently decoded uint8 hard terminal per replica, and no adoption for degenerate spread;
- radius-one-first reverse-waterfill with strict score-per-byte threshold `>25/37,545,489`.

The full-kernel selector includes the inherited zero-weight-mask candidate and admits a full-kernel fill only when neither Brotli nor LZMA worsens. Its frame-level coder diagnostic is **not** archive-carrier byte authority; the final packet and full ZIP must still be counted. The bounded selector is an offline encoder search, not a decode-time algorithm: its selected null coordinates must be represented compactly enough to replay under the receiver runtime cap.

## Live dependency and interaction evidence consumed

The full-kernel compiler landed upstream at `da64a5bc8e` and was integrated here as `9b50eb4aeb` with schema `resize_null_preimage_full_kernel.v1`. Its measured real-linear kernel coverage is `80.6742%` versus `22.6969%` for the mask-only form. Its canonical primitive-basis bounded-uint8 reachability is only a `34.1931%` lower bound, and the height-rowspace × width-null family measured `0/589,824` feasible on frame 0. A naive constant preference was byte-negative by `+512,550 B` Brotli / `+546,524 B` LZMA, so the implementation consumes coder-admitted minimum-description selection rather than assuming a preferred family.

The sibling R2b receipt measured the interaction term directly: blanket per-cell rounding fixes realized only `1,585/16,751 = 9.462%` of scheduled recovery (`8.68%` of the upper bound). Of the refined `17,926`-flip gap, tie-tight `<1e-3` accounts for only `1,607` flips / `0.00136 S`, whereas `[1e-3,1)` accounts for `16,319` flips / `0.01383 S`; no flips have margin `>=1`. `93.4%` of flip cells have an exact bounded-uint8 realization, so lattice feasibility is not the present crux. This evidence changes the next solve from independent tie repair to joint moderate-margin boundary displacement.

Pose is effectively one coordinate: delivered per-dimension MSE is `[6.107e-4, 2.1e-7, 1.4e-7, 1.7e-8, 3.9e-9, 2.6e-8]`, so the next `xi` leg targets dimension 0 only. At the measured `9.462%` realization fraction, the carrier must be `<=1,852 B` to break even against the `27,313 B` feasible stream. This is a preregistered numeric target, not an assertion that the current packet meets it.

## Next named coordinate

`R1B_MODERATE_MARGIN_FULL_KERNEL_MDL_XI0_COMPILE`

Derive the batch-16 winner-rival first-order and realized-secant Jacobians over the `[1e-3,1)` boundary band; solve them jointly in the counted localized grammar; use `FullResizeKernel` offline to select exact bounded-uint8 minimum-description preimages before the hard oracle; add only `xi[0]`; and compile the selected null coordinates into one compact replay grammar and strict archive. Target `<=1,852 B` of carrier at the currently measured realization fraction, or first remeasure realization and recompute the break-even threshold, and require receiver decode `<=1,800 s`. Then rerun the same full n600 receiver/scorer path and evaluate both byte caps and the Seg gate.

## Triality and system wiring

- **DSL:** no trainer flag was invented. The solve is a typed library API; measurement is an explicit CLI with source/weight custody and SSD preflight.
- **DAG:** settled C2 archive + counted packet + corrected QP + exact uint8 preimage + full-kernel MDL selection + fresh hard oracle + strict archive admission. The missing packet/Jacobian/`xi` edges remain visibly open in `.omx/research/r1b_boundary_generator_solve_DAG_FEED_20260720.md`.
- **Equations:** no new canonical law lands. This implementation consumes the settled hard-admission, exact-resize, full-kernel, KKT, and reverse-waterfill predicates; the n600 result is a control blocker, not a new law.
- **Pointer delta:** none. `0.19108 [contest-CPU]` remains unchanged.
- **Lane maturity:** intentionally L0. Code-path implementation is real, but `impl_complete` and `real_archive_empirical` are false because no R1b production archive candidate exists.

## Verification

Focused joint-solve, full-kernel, measurement-wrapper, inherited shared-receiver, and canonical-gate verification is `55 passed`; Ruff, `py_compile`, JSON parse, and `git diff --check` are clean. Two clean `review_tracker` passes were recorded after the final Python diff. The bounded Sol-high review initially failed the landing for a stale projector blocker, possible frame-coder/archive-byte ambiguity, missing compact-replay runtime scope, and an untested fail-before-oracle geometry path. Those were corrected; final verdict is `PASS_SCOPED_L0_CONTROL_BLOCKER`. Durable receipt: `.omx/research/r1b_boundary_generator_solve_round1_review_20260720.json`.

The global lane-registry validator still reports `110` inherited missing-evidence paths outside this lane. The R1b row asserts no maturity gate, remains L0, and introduces no missing evidence path.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, and `.omx/research/operating_manual.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md` and `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`
- `.omx/research/shared_receiver_r1_20260720T154328Z.md`, `.omx/research/shared_receiver_r1_20260720.json`, and the inherited fail-closed admission code
- existing curvelet/shearlet frame and exact uint8 feasibility implementations under `src/tac/boundary_math/` and `src/tac/optimization/`
- `.omx/research/null_compiler_full_kernel_20260720T163500Z.{md,json}` and source commit `da64a5bc8e`
- per-arm directives through `2026-07-20T17:15:45Z`, including sibling `r2b_sparse_target_selection_20260720.{md,json}` measurements
- top-ten Claude memory entries and the latest Codex/Claude findings/design/council surfaces required by preflight

## MAIN landing review required

MAIN must review the branch diff before merge. The merge review should specifically verify: (1) `FullResizeKernel` is mandatory for both corrected-QP and ERM terminals and is followed by exact projection re-verification; (2) the frame-level coder selector is neither treated as final archive-byte authority nor proposed as a per-frame decode-time search; (3) the n600 receipt is labeled control-only and cannot promote an absent R1b packet; (4) no `xi` or Jacobian custody is inferred; and (5) the conflict-resolved registry preserves both the inherited R1b lane and the imported full-kernel equations/lane. No pointer movement, promotion, paid dispatch, or long run is authorized.
