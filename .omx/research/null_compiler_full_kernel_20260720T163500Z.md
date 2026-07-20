# Full shared-resize null-compiler closure — 2026-07-20

Authority: `codex_delegate:null_compiler_fix:20260720T160856Z`  
Lane: `lane_null_compiler_full_kernel_20260720`  
Schema: `resize_null_preimage_full_kernel.v1`  
Verdict scope: one SHA-pinned real fixture frame plus the exact structural law  
Status: `research_only=true`; `score_claim=false`; `promotion_eligible=false`

## Outcome

**DERIVED:** the implicit compiler now spans the complete real-linear kernel of
the shared `(874,1164) -> (384,512)` bilinear resize. It closes the former
mask-only gap from `230,904 / 1,017,336 = 22.6969260893%` to the exact nullity
`820,728 / 1,017,336 = 80.6742315223%` per channel, adding `589,824` dimensions
or `57.9773054330` percentage points.

**MEASURED `[Darwin-arm64 CPU advisory]`:** on decoded frame 0 of the SHA-pinned
#49 fixture, `841,898 / 2,462,184 = 34.1931390993%` of the compiler's canonical
three-channel primitive basis directions admit at least one signed unit move
within uint8 bounds. This is a lower bound for a particular nonredundant basis,
not the cardinality or dimension of the full bounded integer-lattice
intersection.

**MEASURED `[Darwin-arm64 CPU advisory]`:** a deterministic constant-preference
cell search produced an exact-resize candidate in `589,823 / 589,824` cells,
with one budget fallback and exact integer-numerator equality overall. That
candidate was larger than the existing #49 zero-weight fill under both coders,
so coder admission correctly retained the old control. This falsifies only the
tested constant-preference, 128-node bounded MDL heuristic on this one source
frame. It does not falsify the full-kernel family or alternative lattice,
preference, allocation, and coder-conditioned searches.

The contest-CPU frontier pointer remains `0.19108`; it was not moved. No score
was measured or claimed.

## Exact law and implicit compiler

Let the separable resize be `A(X)=A_h X A_w^T`. With

`Q_h=A_h^T(A_hA_h^T)^-1A_h`, `P_h=I-Q_h`, and the analogous `Q_w,P_w`,

the exact real projector is

`P_ker(X) = X - Q_h X Q_w = P_h X + Q_h X P_w`.

The two terms are an orthogonal direct sum. The nonredundant implicit basis is

`K(U,V) = N_h U + A_h^T V N_w^T`,

with `U:(H-h,W)` and `V:(h,W-w)`. The implementation never materializes a
`(HW) x (HW-hw)` dense basis. Each disjoint two-tap rational axis row with
integer numerators `(a,b)` contributes primitive exact null atom
`(b/g,-a/g)`, `g=gcd(a,b)`; unowned indices contribute coordinate atoms.
Float32/float64 projectors are explicit numerical surfaces. Exact uint8
authority is equality of integer resize numerators, never rounded float output.

| Quantity, per channel | Exact value | Fraction |
|---|---:|---:|
| Camera domain | 1,017,336 | 100% |
| Resize rank | 196,608 | 19.3257684777% |
| Full nullity | 820,728 | 80.6742315223% |
| Legacy zero-weight mask | 230,904 | 22.6969260893% |
| Newly represented kernel | 589,824 | 57.9773054330 pp |
| Left tensor `ker(A_h) tensor R^W` | 570,360 | — |
| Right tensor `row(A_h) tensor ker(A_w)` | 250,368 | — |

Per-axis nullities are `490 / 874 = 56.0640732265%` and
`652 / 1164 = 56.0137457045%`. The implementation asserts
`570,360 + 250,368 = 820,728 = 1,017,336 - 196,608`.

## Bounded uint8 reachability

The measured basis test asks whether either sign of each primitive atom fits
around the exact source byte vector. It does not enumerate arbitrary integer
combinations, so every value below is explicitly a lower bound on the bounded
lattice intersection.

| Canonical direction family, three channels | Total | Feasible signed unit move |
|---|---:|---:|
| Zero-weight coordinate | 692,712 | 692,712 |
| Height-null, cell column 0 | 589,824 | 74,469 |
| Height-null, cell column 1 | 589,824 | 74,717 |
| Height-row-space x width-null | 589,824 | 0 |
| **Total** | **2,462,184** | **841,898 (34.1931390993%)** |

Frozen scorer class labels and margins were not loaded, so a class/margin
stratification was not measured. The receipt instead supplies the exact kernel
family split. Any later Fisher/margin allocation must join scorer custody to
this geometry rather than infer it from the fixture-only lower bound.

## Minimum-description measurement

Fixture:
`/Volumes/VertigoDataTier/pact/molab_witness_machine_upstream_20260709/videos/0.mkv`,
SHA-256
`2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`
(verified). Decoded frame-0 SHA-256:
`47492a5864f0521f0ab6b129e4b172717139ba202fc3265a5220b6d1b15c24ed`.

| Candidate | Brotli q11 bytes | Delta vs old | LZMA bytes | Delta vs old |
|---|---:|---:|---:|---:|
| Original source | 1,447,659 | +183,778 | 1,424,617 | +167,582 |
| Legacy #49 mask fill | **1,263,881** | 0 | **1,257,035** | 0 |
| Full-kernel constant candidate | 1,776,431 | +512,550 | 1,803,559 | +546,524 |
| Coder-admitted result | **1,263,881** | **0** | **1,257,035** | **0** |

The admitted legacy control remains `183,778 B` smaller than original under
Brotli (`12.6948404286%`) and `167,582 B` smaller under LZMA
(`11.7633019963%`). The full-kernel candidate visited `3,363,065` search nodes,
had one `NOT_FOUND_BUDGET` fallback, no proven-infeasible cells, exact integer
numerator equality, and maximum diagnostic float residual
`5.684341886080802e-14`. These are raw frame-coder measurements, not counted
archive bytes and not an evaluator score.

Machine-readable receipt:
`.omx/research/null_compiler_full_kernel_20260720T163500Z.json`
(SHA-256 `76af5e7f8d155363a6668b4ee7bca576ea448472ea0cd1e7f938577bb4adfd74`).

## Callable routing

- **r2b sparse target-selection:** call
  `FullResizeKernel.project_kernel(proposed_camera_residual)` to separate the
  resize-free component before charging sparse bytes. This is a geometry
  primitive; admission still belongs to exact receiver/coder evidence.
- **R1 `d_B` preimage cells:** use `FullResizeKernel.synthesize(left,right)` for
  implicit real-null proposals or
  `FullResizeKernel.compile_min_description_preimage(...)` for exact bounded
  uint8 cell alternatives. Hard decoded evidence remains the selector.
- **#401 blind fill:** replace mask-only candidate enumeration with
  `compile_min_description_preimage`; its old `measured_best` mask fill is an
  included control and wins every tie or regression.

## Triality and solver-stack wire-in

- **DSL:** no new config or invented flag. The callable takes typed geometry,
  an explicit finite preference sequence, and a bounded node budget.
- **DAG:** `.omx/research/null_compiler_full_kernel_DAG_FEED_20260720.md` routes
  the compiler behind exact numerator verification and coder admission.
- **Equation:** canonical ID `separable_resize_full_kernel_direct_sum_v1`
  records the projector, parameterization, exact dimensions, and bounded-uint8
  caveat.
- **Sensitivity map:** `project_kernel` exposes the free geometric component;
  no class sensitivity is invented without scorer custody.
- **Pareto/bit allocator:** the old mask remains the zero-regression control;
  full-kernel candidates are admitted only when measured coder bytes improve.
- **Cathedral/autopilot:** no dispatch is authorized. A future consumer may
  schedule preference/allocation probes only after MAIN adoption and exact
  receiver/counted-byte custody.
- **Continual learning:** the canonical equation plus this scoped negative
  prevents repeating “real nullity implies cheap uint8/coder realization.”
- **Probe disambiguator:** candidate preferences are a callable finite family;
  the coder measurement, not prose, arbitrates among them.

## Stores consulted

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, and
  `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`.
- `reports/latest.md`, `.omx/state/lane_registry.json`,
  `.omx/state/subagent_progress.jsonl`, and canonical equation/ledger searches.
- Existing #49, #391, #401, and #532 code, receipts, memos, and tests; frozen
  scorer factorization §8 B1; latest per-arm and fleet broadcast inboxes.
- SHA-pinned #49 SSD fixture above. No live run, paid provider, scorer, pointer,
  or submission state was mutated.

## MAIN landing review required

MAIN must independently review tensor/index orientation, exact numerator
authority, direct-sum nonredundancy and count identities, the lower-bound-only
uint8 language, old-control coder fairness, and the scope of the one-frame
negative. MAIN must also rerun focused tests and decide whether/where the three
consumers adopt the callable. This branch has no promotion authority.
