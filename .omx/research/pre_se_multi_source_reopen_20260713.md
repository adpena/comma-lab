# #484 PRE-SE composition reopen — measured retained-mass family kill

Date: 2026-07-13  
Lane: `lane_replace_round5_pre_se_locus_20260713`  
Subagent/checkpoint identity: `pre_se_reopen_a`  
Authority: `[macOS-CPU advisory; NumPy-fp64 convex fit; CPU-Torch nonlinear]`  
Research only: `true`  
Score claim: `false`  
Promotion eligible: `false`  
Pointer moved: `false`

## Verdict

`RETAINED-MASS-FAMILY-KILL`

`verdict_scope = FAMILY x CHEAP-PRE-SE-LOCALIZATION x SINGLE-AND-MULTI-SOURCE x CONVEX-AND-NONLINEAR-RUNGS x FIXED-n600-REPLAY x 4.70%-AREA`

The two coupled reopen bars were tested separately and then conjoined. `tileable-modulo-cheap-globals` is **MEASURED-CONFIRMED**. Retained mass is **MEASURED-FAILED** for both preregistered multi-source rungs. Because both bars were required, the #484 cheap-localization family does not reopen. Route the surviving family to the whole-teacher distilled student #455, which does not require tileability. The #484 whole-teacher-over-boundary hedge is not reopened by this result.

## Apples-to-apples custody

- **MEASURED-INHERITED:** the same deterministic seed-455 n600 assignment, consisting of 480 immutable Round-5 compact exact targets and the same 120 registered heldout states.
- **MEASURED:** all 120 heldout exact costate array hashes matched the protected PRE-SE receipt; 120 starts, 120 completions, zero retries. The protected receipt stored the hashes and per-state aggregates but not raw support vectors, so those 120 states had to be replayed locally to score a new feature ordering.
- **SEALED:** requested area 0.047; realized area `2311/49152 = 0.047017415364583336`; retained-mass bar 0.47; same-area oracle 0.5278150212253758.
- **ONLY FEATURE DELTA:** shared shallow/base 42 columns once, block2 PRE-SE 144 columns, block3 PRE-SE 288 columns, and shared sensitivity 2 columns once: 476 columns total.
- **UNCHANGED FITS:** twenty exact float64 pair-block RankRLS Moore-Penrose optima; three deterministic 476→32→20 pair-gated ReLU MLP seeds with train-only dev early stopping.
- **CONTAINMENT:** `$0` local cached replay only. No witness training, paid/remote dispatch, provider mutation, live-run mutation, scorer-score claim, or pointer move.

Receipt: `experiments/results/pre_se_multi_source_reopen_20260713/receipt.json`  
Receipt SHA-256: `a092dd5cf791ab060a4300ac3b9c1d49a196ddd83b158121b70fae6a130dc643`

## Retained mass at 4.70% — MEASURED

| frozen PRE-SE feature formulation | exact convex MP | nonlinear MLP ensemble | clears 0.47? |
|---|---:|---:|---:|
| block2 single-source, protected prior | 0.20233024422907497 | 0.2736871496424692 | no |
| block3 single-source, protected prior | 0.09314654496850622 | 0.31323809443347944 | no |
| shallow + block2 + block3 joint multi-source | 0.11225888402810756 | **0.31562159104967574** | **no** |
| same-area exact oracle | 0.5278150212253758 | 0.5278150212253758 | yes |

The best joint result misses the bar by **DERIVED** `0.47 - 0.31562159104967574 = 0.15437840895032423`. Its nonlinear gain over the best protected single-source result is only **DERIVED** `0.00238349661619630`. The convex joint result is worse than the block2 convex result, even though its MP normal-equation optimum is certified; simply adding correlated frozen charts does not manufacture the missing ordering signal.

The individual nonlinear seed retained masses were **MEASURED** 0.2995934738746486, 0.289790392967775, and 0.2897676787268581; population standard deviation was **DERIVED** 0.004626579748100376. The miss is not an unstable one-seed accident.

## Cheap-global tileability — MEASURED and DERIVED

The operator snapshot described “~11” upstream globals. Executable-order re-derivation shows that block2's four SE ancestors are a subset of block3's seven. Therefore:

- **MEASURED:** 4 + 7 = 11 branch incidences if the two branches are counted separately.
- **MEASURED:** 7 unique upstream SE reductions for the composed deepest-prefix execution.
- **MEASURED:** 864 broadcast gate scalars per frame.
- **DERIVED from executable kernels/strides:** receptive field 111 input pixels, output stride 8, radius 55, stride-aligned sufficient halo 56.
- **MEASURED-CONSTRUCTION-PROOF:** on one real registered 384×512 state, every 2×2 tile core for both block2 and block3 is bitwise identical to the full-frame core under a same-shape zero embedding and the seven donated full-frame gates. This rules out any nonlocal dependency beyond the donated gates within the derived halo.
- **MEASURED:** physically cropped CPU tensors differ from the full-shape reference by at most `4.57763671875e-05`. This is recorded as shape-dependent convolution accumulation rounding, not silently called bitwise equality.
- **REQUIRED EXECUTION SEMANTICS:** tiles are independent only between SE barriers. Each channel reduction must finish across all tile cores, then its gate is broadcast before the next local-convolution stage.

### FLOP accounting

Convention: one MAC is two FLOPs; the input VJP charges twice the forward convolution FLOPs. Batch norm, pointwise activation/sigmoid, interpolation, localizer matrix multiplies, and autograd bookkeeping remain outside this inherited convolution-plus-pool model.

- **DERIVED from measured real shapes:** SE MLPs once = 13,504 forward MACs → 54,016 forward+VJP FLOPs.
- **DERIVED from measured real shapes:** global pooling once = 8,404,992 forward FLOPs → 16,809,984 forward+VJP FLOPs.
- **DERIVED:** total once-only globals `G_SE = 16,864,000` FLOPs.
- **MEASURED/DERIVED:** four cropped halo tiles contain 1,045,272,384 local forward MACs total → 4,181,089,536 local forward+VJP FLOPs.
- **DERIVED true operational average:** `(4,181,089,536 + 16,864,000) / 4 = 1,049,488,384` forward+VJP FLOPs per tile, including 4,216,000 amortized global FLOPs per tile.
- **DERIVED no-overlap equal-area accounting:** 668,210,368 FLOPs per tile including amortized globals.
- **DERIVED measured-overlap cost ratio:** 1.5705957798008905× relative to the full deepest-prefix cost split equally over four tiles.

The multi-source extractor costs the deepest block3 prefix once. Summing block2 and block3 cut costs would double-count the shared prefix and is forbidden.

## Req-R family evidence

Requirement R asks for at least two formulations plus a structural reason before a family-level negative.

1. **MEASURED formulation 1:** block2 single-source; convex and nonlinear both fail.
2. **MEASURED formulation 2:** block3 single-source; convex and nonlinear both fail.
3. **MEASURED formulation 3:** shallow+block2+block3 multi-source; exact convex MP and nonlinear ensemble both fail.
4. **STRUCTURAL reason:** the exact same-area oracle clears 0.47, so selected area is not the limiter. Donated globals clear tileability, so global reductions are not the retained-mass limiter. Exact convex optima and the nonlinear ensemble over every frozen cheap PRE-SE chart still fail, locating the binding deficit in target-ordering information absent from this frozen feature family.

This does **not** kill all possible learned localizers, attention models, new feature providers, or whole-teacher students. It kills the scoped cheap frozen PRE-SE localization family on retained-mass grounds. Reactivation requires a provider outside these frozen charts that adds target-ordering information and is preregistered against the same gate.

## Research and OSS cross-check

- Hu et al., [Squeeze-and-Excitation Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html), supports the decomposition used here: spatial global pooling produces channel statistics, then a small excitation network generates broadcast channel recalibration.
- Hariharan et al., [Hypercolumns for Object Segmentation and Fine-grained Localization](https://openaccess.thecvf.com/content_cvpr_2015/html/Hariharan_Hypercolumns_for_Object_2015_CVPR_paper.html), and Lin et al., [Feature Pyramid Networks](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html), motivate the preregistered multi-depth composition but do not guarantee costate-order retention.
- Ronneberger et al., [U-Net](https://arxiv.org/abs/1505.04597), supplies the overlap-tile precedent; this landing derives its own receptive field and measures its own halo cost rather than importing a claimed number.
- OSS implementation surfaces consulted: [timm EfficientNet](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py) and [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch). No external code was copied.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`
- `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`
- `.omx/state/canonical_task_status.jsonl`, `.omx/state/probe_outcomes.jsonl`
- `reports/latest.md`
- latest Codex findings/session summary, latest T3 council, latest design memo, and all <24-hour directive files required by AGENTS.md
- protected PRE-SE source, output, receipt, per-state heldout hashes, nonlinear chunks, and Round-5 compact targets
- primary papers and OSS surfaces listed above

## Triality and unified wire-in

- DSL: `tac.witness_dsl.pre_se_multi_source_reopen_policy_20260713` (default off; no live argv).
- Equation: `tac.canonical_equations.pre_se_multi_source_reopen_20260713`.
- DAG: `.omx/research/pre_se_multi_source_reopen_DAG_FEED_20260713.md`; canonical shared DAG append deferred because the shared surface was sibling-held/dirty.
- Sensitivity map: the retained-mass capability ceiling is exposed through the canonical empirical anchor; no pixel-level score authority is claimed.
- Pareto constraint: do not spend on this frozen PRE-SE family without the reactivation criterion.
- Bit allocator: non-binding because no archive or score actuator was admitted.
- Cathedral/autopilot: canonical `KILL` probe outcome blocks re-dispatch of this formulation.
- Continual learning: this memo, equation anchor, findings memo, and probe outcome carry the negative forward.
- Probe disambiguator: convex MP versus nonlinear ensemble, single-source versus multi-source, and strict tileability versus tileability-modulo-cheap-globals were all kept distinct.

## Limitations and pointer delta

The tile FLOP result is a real-shape analytical accounting backed by executed hooks; it is not a measured wall-clock speedup. One real state supplies the concrete tile equality proof, with graph locality supplying the structural generalization to aligned tiles of sufficient halo. NumPy emitted transient overflow/divide warnings inside inherited balanced-statistic and MLP matrix operations, but every durable Gram/RHS was finite, every convex normal-equation certificate passed, and all admitted per-state predictions passed the existing finite-output guards.

`pointer_delta = NONE`. No contest score axis was evaluated or inferred.
