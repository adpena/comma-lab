# V9 CGauge gauge/symmetry/homotopy warm-start — implementation spec

Date: 2026-07-14  
Lane: `warmstart_gauge_symmetry_homotopy`  
Authority: operator mission; `$0` local CPU only; no training, provider, GPU, evaluator, pointer, or hot-owner mutation.

## Outcome

Build a deterministic, resumable, NumPy-authoritative probe that measures the specified V9 EMA-best receiver surface at n600 through the canonical `R` and frozen CPU-torch SegNet, then answers:

1. D37: whether `M` is sufficient for the flip `F` given `(class, xi)`, using pair-blocked held-out conditional codelength and charged table overhead;
2. LieFlow/FINO warm-start: which candidate metadata refinements shrink the empirically supported within-context permutation group, without claiming a neural LieFlow model was trained;
3. D38: whether exact local restrictions of the realized flip field glue, while refusing to promote that tautological exact-section result into a receiver-rate twist law without typed encoded local sections and transition maps;
4. Noether: whether an explicitly supplied discrete density/flux satisfies a continuity equation between events, while refusing to call the observed flip-mass proxy a conserved Noether charge because V9 has no executable action momentum;
5. per-class realized d_seg and the existing scalar `d_seg=d_cov+d_gauge` decomposition, preserving `MEASURED` versus `DERIVED` and refusing a pointwise split.

## Ownership and collision constraints

Create only these new files:

- `src/tac/boundary_math/gauge_symmetry_homotopy_20260714.py`
- `tools/probe_v9_cgauge_symmetry_homotopy_n600_20260714.py`
- `tests/test_gauge_symmetry_homotopy_20260714.py`

Do not edit `preflight.py`, `src/tac/canonical_equations/**`, `src/tac/witness_dsl/**`, `src/tac/scorer_surrogate/vjp_fidelity.py`, the level-set trainer, launchers, configs, or any existing probe. Do not absorb or revert unrelated dirty files.

## Mathematical contract

### Conditional symmetry discovery / D37

For binary realized flip `F`, base context `B=(Q_M,Q_xi[,Q_phi])`, and candidate refinement `C`, estimate

`Delta L = L_cv(F | B) - L_cv(F | B,C)`

with deterministic nested pair-blocked cross-fitting. Report gross held-out gain, pair-bootstrap confidence interval, incremental receiver-table overhead, and net confidence interval. The estimator is a conditional-codelength proxy for `I(F;C|M,xi)`, not an exact MI identity.

The candidate within-cell permutation group is admitted only when the refinement's net upper confidence bound is non-positive. A positive lower bound means the base group is too large and the refinement carries residual non-gauge structure. This is a theorem-compatible support test inspired by LieFlow, not the paper's trained flow-matching model.

Use the same D37 operational variables as the registered predecessor for comparability:

- `F`: V9 EMA-best through-R frozen-SegNet argmax disagreement against GT;
- `M`: frozen GT top1/top2 margin;
- `C`: directed unlike four-neighbour GT class edge;
- `Qxi`: train-fold PCA quantization of cached official PoseNet six-vectors;
- optional `Phi`: `M_self/(M_self+M_adjacent)`.

Make every ASSUMED proxy explicit. Do not transfer the epoch-50 baseline verdict.

### D38 descent

Provide a generic finite-cover checker. Each local section is an array plus a boolean cover mask over one common finite base. Require exact equality on every overlap for an exact-section PASS. Also compute transition mismatches and a triple-overlap cocycle residual. Empty overlap is not evidence. The V9 n600 fire uses restrictions of the globally realized flip field, so it may prove only `EXACT_SECTION_GLUES`; the rate-level verdict must remain `GLOBAL_RATE_DESCENT_NOT_TYPED` until encoded local sections, receiver restriction maps, and changing-isotropy bands exist.

### Discrete Noether continuity

For density `rho[t,s]`, flux `j[t,e]`, signed incidence matrix `D[s,e]`, and event mask `E[t,s]`, compute

`residual[t,s] = rho[t+1,s]-rho[t,s] + (D @ j[t])[s]`.

Continuity is tested only off events. NumPy is authority; provide an optional MLX parity function for the pure tensor operation. The n600 probe may feed per-class flip mass with zero flux only as `OBSERVABILITY_PROXY_NOT_NOETHER_CHARGE`.

### d_seg decomposition

Measure total and per-GT-class d_seg directly from the realized argmax maps. Consume, do not remeasure, the registered scalar gauge anchor only if its provenance is explicitly supplied; derive `d_cov=max(0,d_total-d_gauge)`. Label the result `SCALAR_ONLY_NOT_POINTWISE`. If no compatible gauge anchor is supplied, report `NO_VERDICT_DECOMPOSITION_CUSTODY` rather than inventing one.

## Probe execution contract

- Fixed checkpoint: `experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_BEST.npz`, selected via `levelset_best.json` and hash-bound.
- Fixed GT: `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`, hash-bound.
- Exactly 600 pairs; no subset can emit the registered final receipt.
- Reuse canonical functions from `tools/witness_per_stage_annulus_attribution.py` for checkpoint parsing, deploy int8 round-trip, frame1 render, `R`, and frozen SegNet readout. Do not reimplement the receiver/scorer path.
- Process bounded batches. Prefer open argmax memmaps on the selected SSD waterfall (`/Volumes/VertigoDataTier/pact`, then `/Volumes/APDataStore/pact`), and never persist camera frames. If the managed sandbox refuses SSD writes, the only permitted contained fallback is bit-packed receiver `F` plus scalar per-pair counts under `experiments/results`, capped at 64 KiB per pair and 32 MiB total; a local full-map fallback remains forbidden.
- Atomic stage receipts after storage preflight, each render/scorer interval, render completion, D37, D38, and final aggregation. Resume from the done bitmap; preserve all stage receipts.
- Fail closed if no SSD has enough room, any input hash changes, checkpoint path disagrees with `levelset_best.json`, shape/count is not n600, MPS/GPU is requested, or source/receipt custody is incomplete.
- Small final receipts may be mirrored under `.omx/research`; bulky maps stay on SSD with path/bytes/SHA-256 and deterministic rebuild command.
- `score_claim=false`, `promotion_eligible=false`, pointer unchanged.
- Bind Torch intra-op geometry explicitly. This historical selector predates the 2026-07-13 one-thread standard; its source run used the then-default six-thread path. A replay can proceed only with that provenance stated and must reproduce all `117,964,800` selector argmax cells exactly before D37 is released.

## Tests

Cover at minimum:

1. class-independent synthetic data gives no admitted positive residual after overhead;
2. class-dependent synthetic data has positive held-out conditional gain;
3. folds are pair-blocked and deterministic;
4. exact local restrictions glue; altered overlap fails; triple cocycle mismatch is detected;
5. constant density/zero flux conserves; an injected event is excluded only where marked;
6. NumPy/MLX continuity parity when MLX is available;
7. per-class d_seg and scalar-only decomposition labels;
8. storage waterfall refuses local output and insufficient space;
9. final receipt refuses fewer than 600 complete pairs and refuses checkpoint-selector/hash drift.

## Held integration contract

This lane cannot touch exclusive provenance/DSL owners. Its final memo must request exactly one live V9 observational lever:

- factory: `GaugeSymmetryHomotopyProbePolicy.from_receipt(...)` (name held, owner may adapt to canonical naming);
- DSL lever: observational receipt path/hash only, default OFF, no trainer-loss mutation;
- LawRef: held `v9_empirical_gauge_refinement_d37_v1`, consuming the registered conditional-codelength equation plus the existing scalar CGauge decomposition; no new equation is claimed landed;
- consumers: V9 provenance bijection and the asynchronous/local frozen-SegNet verdict-forward audit surface;
- acceptance: content-bound n600 receipt, exact checkpoint/GT/tool hashes, D37 estimator schema, no score claim, and whole-V9 strict source closure green.

## Verdict ladder

- Naive/first-cut negative: `INSTANCE` only.
- This estimator on the V9 EMA-best: `FORMULATION x EMPIRICAL-SURFACE` only.
- No paper family kill, no group-family kill, no CGauge formulation kill.
- V9 runtime covariance remains `IMPLEMENTATION_CUSTODY_GAP_ONLY` until the exclusive owner lands a typed gauge-transform pair and pre/post action/divergence equality receipt with `(R,xi)` chart custody.
