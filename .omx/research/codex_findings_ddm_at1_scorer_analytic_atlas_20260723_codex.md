# Codex findings — DDM AT1 scorer analytic atlas Phase 0

**Date:** 2026-07-23  
**Lane:** `lane_ddm_at1_scorer_analytic_atlas_20260723`  
**Authority:** `research_only=true`; `score_claim=false`;
`execution_allowed=false`; `[macOS-CPU frozen-scorer advisory]`  
**Verdict:** **STRUCTURAL MODERNIZATION LANDED; LOCKED-SOURCE FACTOR
MATERIALIZATION BLOCKED; FULL n600 GAZE OWED.**  
**Pointer:** unchanged. No candidate archive, contest-CPU/CUDA replay, score
claim, launch, or promotion authority.

## Executive disposition

The old #36 atlas contract is now a typed, hash-fresh Phase-0 substrate, and
the live costate organ no longer computes its own pair/site λ. The single
producer is
`tac.optimization.scorer_analytic_atlas.build_ddm_lambda_bundle`; the organ is
the advisory consumer/controller. This implements doctrine points 4, 5, 7,
and 8 without pretending that an execution-disabled lane materialized the
expensive n600 gaze/Jacobian tensors
(`ddm_scorer_native_doctrine_and_synthesis_20260723.md:29-50`).

The upstream closure audit found a real custody blocker before closed-form
materialization: the available local imports differ from the current
`upstream/uv.lock` on seven packages. Immutable evaluator sources and frozen
checkpoint names/shapes/dtypes are exact; observed third-party source lines are
not the lock-selected evaluator source. Therefore network closed forms are
typed but not emitted as consumable locked-source factors. This is a
**source-binding blocker**, not a negative verdict on the atlas, factor
families, or frozen scorer.

The three SHA-pinned receipts are:

| Receipt | Body SHA-256 | Result |
|---|---|---|
| `ddm_at1_scorer_analytic_atlas_20260723T194312Z/scorer_module_inventory_receipt.json` | `f02031d6025f869cceef65ca4eef365483f6c43d2378a6a629ec6868d7b53b0a` | exact immutable/checkpoint inventory; observed-library binding blocked |
| `ddm_at1_scorer_analytic_atlas_20260723T194312Z/scorer_semantic_divergence_receipt.json` | `807933c67c0567f052834f5594c169911e7e79a2e53e5871edb0726017d2bf06` | ranked closure divergence table; fp32 envelope quantified |
| `ddm_at1_scorer_analytic_atlas_20260723T194312Z/atlas_receipt.json` | `a446465df33af0e490f3a5dca4b78d56ea4574353a345e32d791928495f7e27a` | atlas/bridge/λ/triality closure |

## Phase-0 upstream closure — what actually exists

### PoseNet

The observed graph contains 616 named modules and the frozen checkpoint
contains 510 tensors. Checkpoint names/shapes are load-compatible; the only
graph-only buffers are 88 `num_batches_tracked` counters, which PyTorch
BatchNorm load compatibility defaults and which do not enter eval-mode BN
closed forms.

| Mechanism | Exact observed count | Consequence |
|---|---:|---|
| `SEModule` | 1 | Pose SE is one final-conv mechanism, not the Seg-wide pattern |
| `BatchNormAct2d` | 52 | primary frozen spatial affine/statistics surface |
| `BatchNorm2d` | 28 | additional spatial eval-BN surface |
| `BatchNorm1d` | 8 | head-side BN surface inside `AllNorm` |
| `AllNorm` | 8 | upstream wrapper is `BatchNorm1d(1)` over `x.view(-1,1)`, not per-feature normalization |
| `LayerScale2d` | 24 | learned per-channel scale mechanism omitted by the recalled charter list |
| `GELUTanh` | 19 | Pose nonlinear contrast mechanism; not SiLU |
| `ReparamLargeKernelConv` | 3 | large-kernel spatial/frequency mechanism |
| attention | 0 | no Pose attention stage exists |

Pose consumes both frames after bilinear resize and full-range BT.601 YUV6
construction; it produces 12 head coordinates and distortion consumes exactly
coordinates `0:6`. The discarded six do not enter the distortion expression.
Upstream `rgb_to_yuv6` is decorated `torch.no_grad`, so exact gaze work must
use the hash-stamped differentiable mirror in `tac.scorer` and retain its
forward-fidelity proof.

### SegNet

The observed graph contains 540 named modules and the checkpoint contains 562
tensors with exact names and shapes.

| Mechanism | Exact observed count | Consequence |
|---|---:|---|
| `SqueezeExcite` | 23 | Seg-wide global gate surface |
| `SiLU` | 68 | Seg contrast nonlinearity |
| `BatchNormAct2d` | 68 | primary frozen affine/statistics surface |
| `BatchNorm2d` | 10 | decoder/head BN surface |
| `Attention` wrappers | 10 | all contain `Identity`; name artifact, zero active attention |

Seg consumes `x[:, -1]` only. Frame 0 has exact zero Seg influence. Its
per-pair distortion is the mean over a uniform 384×512 site grid; hence
mean-of-pair-means and the global pixel mean are mathematically identical and
the live `1/600` pair currency is correct.

### Version custody

The immutable upstream lock selects, on this macOS-CPU axis, Torch 2.10.0,
TorchVision 0.25.0, timm 1.0.22, AV 17.0.0, einops 0.8.1, NumPy 2.3.4,
and safetensors 0.6.2. The observed shared environment instead has Torch
2.12.1, TorchVision 0.27.1, timm 1.0.27, AV 17.1.0, einops 0.8.2,
NumPy 1.26.4, and safetensors 0.8.0. SMP matches at 0.5.0.

No `upstream/.venv` or other materialized exact-lock environment exists.
Installed third-party source line references in the inventory are therefore
exact for the observed graph but **not** locked evaluator-source authority.
The consumer gate is
`BLOCKED_LOCKED_LIBRARY_SOURCE_NOT_MATERIALIZED`.

## Adversarial divergence table

The complete machine-readable table is in the divergence receipt. The
highest-impact rows are:

1. **Locked source drift — HIGH binding blocker.** Seven dependency versions
   differ. Rebuild the inventory under the exact lock and require zero drift
   before consuming library-source-bound network factors.
2. **Pose preprocessing derivative barrier — HIGH derivative requirement.**
   The evaluator's `@torch.no_grad` is correct for forward scoring but severs
   input derivatives. Atlas materializers must use the exact differentiable
   mirror; this is a derivative-port requirement, not evaluator modification.
3. **Internal E1/E2 meter versus upstream evaluator — MEDIUM exact-score
   custody.** Internal meters use cached GT cells/poses and fp64 aggregation.
   E1 internal minus upstream printed-8dp distortion terms is
   `+2.4288482106255005e-06` score units; E2 is
   `+5.3895744825394054e-08`. These comparisons include upstream
   eight-decimal report quantization. Separate official-harness receipts remain
   the authority.
4. **fp32 zero-dimensional aggregation — LOW but real.** On the measured v19b
   38-batch row, changing only batch scalar order spans:
   Pose term `3.7787208455597465e-06`, Seg term
   `5.587935447692871e-07`, joint upper envelope
   `4.337514390329034e-06` score units. Exact mirrors must preserve batch size,
   pair order, fp32 accumulator/divisor, and evidence axis.
5. **Score helper algebra — clean.** `tac.contest_score.compute_contest_score`
   and the E1/E2 `score_row` helper are exactly equal on both named internal
   rows. This certifies the real-arithmetic formula only, not forward or
   aggregation custody.
6. **Composition seeds — clean.** n600 batches are 37×16 plus 8; Pose prices
   output `0:6`; Seg is frame-1 only; uniform Seg sites preserve pair weights;
   the explicit bilinear resize call omits `align_corners` and `antialias`,
   selecting their false defaults at the executed source.

Ground-truth CPU decode uses the manual limited-range BT.601 YUV-plane path
with bilinear chroma upsampling and uint8 rounding. It does not use PyAV
`rgb24`, preserving the settled phantom-pose avoidance law.

## Atlas, bridge, and no-fake gates

The new typed module provides:

- `SourceHashStamp`, `TensorArtifactRef`, and `AnalyticFactor` with mandatory
  first-rung, pair interval, content hash, validity horizon, named consumer,
  and counted-inert states;
- frozen BN, SE, kernel-DFT, and BN∘SiLU closed-form builders plus direct
  reference evaluators;
- exact gaze registration contracts: six Pose VJPs per pair and rank-4 Seg
  head pullback, with total n600 coverage refusal;
- exact Jacobian composition and gaze pullback;
- axis projections for amplitude/frequency/phase/contrast/channel energy/
  texture statistics, with amplitude refusing any row lacking one complete
  uint8-surviving R-projection receipt;
- typed non-additive KKT pools, with no invented empty pool before real
  competing factors exist;
- preserved atomic stage checkpoints and source-hash revalidation on resume;
- an explicit SDWL1↔E2 coordinate bridge.

The bridge is correctly `LOSS_ACCOUNTED_NOT_INVERTIBLE`:

- SDWL1 declares 45,600 scalar facts;
- E2 has 117,964,800 semantic-role coordinates and 702,000 chart
  coordinates;
- E2 carries no counted pose member;
- background role code 0 is unresolved on the SDWL1 class side;
- no price crosses a lossy row;
- U1, mode re-race, and ξ direction stay blocked on the named missing maps.

#580 is reused rather than re-derived. It certifies a spatial direct-sum
resize kernel, not a global circular DFT diagonalization. Consequently the
requested exact spectral dead-band set is empty and
`frequency_band_admission` must refuse zero-byte truncation. This is a
**spectral-certificate formulation gap**; frequency/residual families remain
open.

No network closed-form, gaze, Jacobian, or axis tensor is reported as
materialized. The atlas receipt records counts of zero for those rows. The
only emitted factor is the reused #580 certificate, marked blocked/waiting
consumer. This obeys the doctrine's “unconsumed = counted-but-inert” law rather
than manufacturing completeness.

## λ unification and exact scope

`ddm_costate_organ.py` now imports the atlas producer and does not derive
pair/site λ locally. Its registered source fleet also recognizes current E1,
E2, and SDWL1 receipt schemas.

The current atlas-produced bundle has content SHA-256
`90ebdbb9af557a0d2e79572d0fc583d72573fab7b7c9ff941bb123ce17987bef`:

| Field | Exact current value |
|---|---:|
| g3 pair coverage required | 600 |
| exact v19-joined pair λ rows | 8 |
| exact site λ rows | 40 |
| missing exact pair λ rows | 592, explicitly counted inert |
| Spearman ρ on the eight-pair exact backtest | 0.9027075674773932 |
| NDCG@4 | 0.9268617843989323 |
| positive realized pairs | 5 |

This changes the old `BACKTESTED-FAIL` premise only at the named
`V19_EIGHT_PAIR_EXACT_RECEIVER_REPLAY_X_G3_ATLAS` instance. It does **not**
license a general adjoint success claim and does not fill the missing 592
pairs. The organ remains `_dev`, `LIVE_DDM_ADVISORY`, `actuation=NONE`.

## Directive-consumption table

| Directive | Disposition | Durable evidence / named successor |
|---|---|---|
| Doctrine 4 — derive > measure | **PARTIAL** | typed closed-form builders + exact module/checkpoint inventory landed; locked-source factor shards deferred to `at1_locked_source_factor_materializer` after zero-drift inventory |
| Doctrine 5 — total influence map | **PARTIAL** | gaze/Jacobian schemas, exact composition, n600 gates landed; tensor materialization deferred to `at1_n600_gaze_jacobian_materializer` because this lane has `execution_allowed=false` |
| Doctrine 7 — atlas/organ unification | **CONSUMED** | atlas is sole pair/site λ producer; organ is consumer/controller |
| Doctrine 8 — philosophy checklist | **CONSUMED** | KKT pool type, amplitude R gate, counted-inert law, n600 refusal, stage checkpoints, freshness stamps, first-rung fields |
| MAIN 19:52 module inventory correction | **CONSUMED** | full Pose/Seg inventory, separate mechanism tables, attention name artifact adjudicated |
| MAIN 19:55 transitive closure | **PARTIAL** | evaluator composition, checkpoints, names, decode, lock rows, observed source lines landed; exact locked third-party source unavailable and fail-closed |
| MAIN 19:56 adversarial divergence | **CONSUMED** | ranked machine-readable divergence receipt |
| MAIN 19:57 fp32 aggregation | **CONSUMED** | measured v19b 38-batch order envelope; E1/E2 and canonical helper comparisons |
| sn1 validation residual | **DEFERRED** | `at1_sn1_factor_residual_validator` joins only after an sn1 receipt lands with matching factor IDs/hashes |

No later per-arm directive was present at the final stage boundary. The
fleet-wide inbox was checked; its older unrelated campaign directives were
not reinterpreted as authority for this lane.

## STORES CONSULTED

- `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`
- canonical doctrine:
  `.omx/research/ddm_scorer_native_doctrine_and_synthesis_20260723.md`
- `reports/latest.md`; lane registry; subagent progress; current DDM receipt
  families selected by `ddm_costate_organ.discover_sources`
- immutable `upstream/evaluate.py`, `modules.py`, `frame_utils.py`,
  `pyproject.toml`, `uv.lock`, model checkpoints, and video-names file
- #580 full-kernel receipt; rank-4/current scorer ports; E1/E2 export,
  verification, and upstream-harness receipts; v19/v19b/g3/dv1/dv2/g4
- latest sister costate findings/session summaries and canonical memory
  preflight described by the operating contract

## MAIN landing review — required

MAIN should not rubber-stamp the branch. Review:

1. whether the seven-version drift is accepted as the correct exact
   materialization blocker rather than bypassed;
2. every upstream source-line/source-hash claim in the 758 KB inventory;
3. the #580 decision to refuse global DFT dead bands;
4. the λ producer subtraction from the organ and the exact E1/E2/SDWL1 source
   selectors;
5. the lossy bridge counts and forbidden price transfers;
6. the fp32 reduction simulation's use of fp32 division by the fp32 n600
   accumulator;
7. authority flags, counted-inert omissions, and the absence of any score or
   promotion claim.

Only after that review should MAIN merge the isolated worktree commit.

