# DDM EC1 — implicit decoder-derived edge conditioning (2026-08-14)

## Decision

**DESIGN COMPLETE; TRUE-CUDA TRAINING QUEUED; NO FAMILY VERDICT.** The first
shipping-form family should be the **oriented decoded-token latent adapter**.
It derives context from the semantic-token plane already decoded by CP135 and
injects a counted rank-4 adapter before CP135's four nonlinear `TokenBlock`s.
It ships no edge mask and no per-edge sidecar.

This arm did not own the full-n600 scorer slot and did not use Metal or Modal.
It therefore did not train a candidate, measure realized candidate flips, or
seal an archive-SHA-specific T4 measurement. Sealing that row before a trained
archive exists would be fake. The exact pointer did not move.

## Measured result

Axis for the context race: **[contest-CUDA T4 retained-field analysis, n600,
scorer-free]**. Denominator: all 600 pairs and all **117,964,800** scorer pixels.
Target: the pinned CP135 T4 base's **34,970** errors against the pinned GT field.
Each row is a pair-parity two-fold cross-fit: every pixel is ranked by a table
fit only on the opposite pair parity. This is full-population analysis, not a
prefix or a sampled negative.

| receiver-computable family | buckets | cross-fit AUROC | expected base errors in top 34,970 | top-budget precision | lift over prevalence |
|---|---:|---:|---:|---:|---:|
| class only | 5 | 0.8252455 | 441.56 | 0.0126269 | 42.59x |
| undirected adjacent-class set | 160 | 0.9909637 | 2,992.03 | 0.0855599 | 288.62x |
| **oriented four-neighbor classes** | 3,125 | **0.9956550** | **8,380.30** | **0.2396425** | **808.39x** |

The fractional counts are expected counts under exact score ties at the
top-34,970 cutoff. They are targeting/selectivity evidence, not flips a decoder
has realized.

All three shipping adapters use the same 25-channel input width, hidden width
4, and 1,424 parameters; unused context channels are zeroed in the smaller
controls. Their deterministic int8/float16 design-price payloads were put into
real two-member ZIPs so break-even uses exact archive delta, not module bytes:

| family | module member | exact archive delta vs 186,252 B CP135 | candidate? |
|---|---:|---:|---|
| class only | 1,616 B | 1,718 B | no, seeded design-price control |
| undirected | 1,635 B | 1,737 B | no, seeded design-price control |
| **oriented** | **1,605 B** | **1,707 B** | no, seeded design-price control |

At the charter's **0.785 flips/B** law, the oriented reference must realize at
least `ceil(1,707 * 0.785) = 1,340` fewer flips. Its 8,380-error cross-fit target
mass is 6.25x that bar, but only a true-CUDA trained receiver can determine how
much of that mass is reachable without collateral damage or pose loss.

## Actual receiver mechanism probe

Axis: **[macOS-CPU exact CP135 renderer mechanism surface, no scorer]**. On the
seeded random pair 125, using the decoded CP135 WANS1 weights and semantic
tokens:

- The counted zero-head oriented adapter was bit-identical to the unmodified
  CP135 pre-R output.
- The seeded nonzero capacity control changed **589,814** pre-R float values;
  maximum absolute change was **5.9328537**.
- The nonzero control is not trained and is not a candidate. This probe proves
  that the new archive member reaches the intended latent surface inside the
  actual decoder; it says nothing about sign, flips, pose, or score.

The correctly classified identity packaging control is a deterministic
**187,559 B** archive (`+1,307 B`), with members `p` and `ec1_latent.br`. Its
adapted runtime compiles and consumes the counted member before the real CP135
`TokenBlock` stack. It is explicitly `IDENTITY_CONTROL`, `is_candidate=false`,
and has no exact row.

## Mechanism and receiver boundary

The generic receiver computes one of three contexts from its decoded token
plane:

1. center-class one-hot;
2. center class plus an orientation-free pool of neighboring classes that
   differ from the center;
3. center class plus separate left/right/up/down neighboring-class channels at
   token boundaries.

The selected oriented form feeds a counted `25 -> 4` 3x3 convolution, rank-4
depthwise 3x3 convolution, and `4 -> 96` head. Its bounded output is added after
`token_embed + coord_mix` and before the four CP135 nonlinear blocks. The
algorithm and decoded-token context are generic receiver code. Quantized
weights are the only video-derived addition and live in the counted archive.

This is not SA1 under a new name. SA1 corrected pre-R RGB after the semantic
renderer and was exactly T4-inert. EC1 changes the renderer's latent state
before four nonlinear, dilated blocks can propagate and amplify it.

## RECALL EVIDENCE

Sources and content queries searched before design:

- Charter seeds: `ddm_js1c_cuda_custody_stage0_verdict_20260814.md`, JS8 memo
  SHA `f9486d646ba7...`, `#982`, the full JS1C retained CUDA store, and the CP135
  runtime.
- Full research corpus and task surfaces with `implicit edge`,
  `edge.condition`, `decoder-derived`, `Road.hub`, `oriented incidence`,
  `latent condition`, `HPAC`, `#978`, `#982`, `ddm_js3`, and `ddm_sa1`.
- Canonical equations via
  `.venv/bin/python tools/list_canonical_equations.py --json`; the relevant
  `ddm_lp1_deepest_home_context_waterfill_v1` rule requires same-object,
  receiver-closed negative marginals and charges video-derived parameters.
- `CANONICAL_RESEARCH_INDEX*`, the `sub015_DAG_*` FEED blocks, the live hot
  state, and the active lane registry.

Findings beyond the charter's seed list, and how they changed the plan:

- `ddm_sr1_implicit_edge_conditioning_20260811.md` measured only **-2 B** for
  additive causal-edge probability calibration. That closed another rate-table
  treatment, so EC1 does not touch HPAC probabilities.
- `ddm_js2_implicit_edge_conditioning_20260812.md` found a 44% local-vs-T4
  field mismatch. JS1C now supplies matched retained fields, but the lesson
  forbids local scorer admission.
- `ddm_js3_learned_implicit_conditioning_20260812.md` established a real tiny
  learned receiver surface; `ddm_sa1_shipping_axis_seg_actuator_20260813.md`
  then shipped it and measured **0 T4 flip movement**. This changed the EC1
  injection point from post-render RGB to the latent state before TokenBlocks.
- `ddm_gca1_graph_calculus_crosswalk_20260813.md` pre-registered the
  class/undirected/oriented context race once a matched directed decomposition
  existed. JS1C satisfied that trigger; the full-n600 race above executed it.
- `ddm_fd135_fractal_decomposition_20260810.md` and the actual F26 runtime show
  that decoded tokens are nearly the semantic target while most remaining
  errors arise in RGB realization. This made token-derived latent modulation a
  better surface than another token edit.
- `ddm_cn5_arc_consolidation_20260813.md` consolidates the JS2B/JS3/JS4/JS5/SA1
  instance/formulation closures. The new mechanism changes representation and
  therefore does not silently reopen those rows.

## Road-hub derivation

The matched base has 28,549 Road-incident errors, **81.64%** of all 34,970.
Road<->Lane alone has 15,178 (43.40%), Road<->Undrivable 6,972 (19.94%),
Road<->Movable 4,205 (12.02%), and Road<->MyCar 2,194 (6.27%). The directed
cells are asymmetric, especially Road/Lane. That is why the first family keeps
neighbor class and direction instead of collapsing to a single edge flag.

## Payload custody and catch-and-fix

Durable consumer store:
`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/ddm_ec1_20260814/`.
It holds the full context and cross-fit probability fields for all three
families, the derived base-error field, raw/NPY/Brotli LUTs, raw/coded/repeat and
decoded adapter tensors, exact design-price archives and repeats, the actual
receiver probe fields, the correctly classified identity runtime control,
`FINAL_RESULT.json`, and `MAIN_CUDA_FIRE_ORDER.json`.

The first analysis retained the LUT NPY and Brotli bytes but omitted the exact
raw float32 buffer passed to Brotli, and it recomputed the base-error boolean
field without retaining that derived field. Review caught both P0 defects. The
raw LUT buffers were recovered byte-exactly from the retained NPYs and verified
against Brotli decompression; each family has a `RETENTION_REPAIR.json`. The
117,964,800-value base-error field was deterministically rematerialized from the
pinned retained GT/base fields and is now retained as
`base_error_n600.bool.npy` (SHA `03e0d178...`). No scorer reran. The source now
persists both objects before measurement.

The first generic packaging-control receipt also used the overbroad field name
`trained_module` for an identity module. It never claimed a score, but the name
was wrong. An append-only classification receipt marks it non-candidate, and
the authoritative v4 packaging control uses schema
`ddm_ec1_packaged_module.v1`, status `IDENTITY_CONTROL`, and
`is_candidate=false`.

## Verification

- 6 EC1 tests pass: context separation, directed-code distinction, zero-adapter
  identity, nonzero latent reach, deterministic counted archive grammar, and
  exact module parse-back/retention preflight.
- 18-test focused regression passes with the JS3 and SA1 sibling suites.
- Ruff passes on all three new Python files.
- Strict `check_no_measure_and_discard_payload` returns no findings.
- The four patched identity-control runtime Python files compile.
- The identity archive repeats byte-identically and parses as exactly
  `['p', 'ec1_latent.br']`.

## What was not measured

No learned EC1 adapter exists. No candidate render, camera uint8 field, SegNet
field, PoseNet result, realized flip delta, candidate archive, contest-CPU row,
contest-CUDA candidate row, or exact score was measured. The context race is
not admission authority. The seeded nonzero module is capacity pricing only.
There is no negative family verdict. The full implicit family routes to #978
only if an optimal-form true-CUDA trained adapter fails the **1,340 realized
flip** break-even gate after exact archive pricing and pose accounting.

Own-vehicle frontier unchanged: **S = 0.7539807296911207 @ 357,836 B
[macOS-CPU advisory] n600**. Contest pointer unchanged and borrowed.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN true-CUDA trainer and exact-row
  owner; consumer store:
  `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/ddm_ec1_20260814/main_cuda/`;
  fire trigger: MAIN owns the sole full-n600 scorer lane, no competing n600
  scorer job is active, and a reviewed true-CUDA trainer uses the oriented
  pre-TokenBlock adapter with distinct live+EMA stage checkpoints and full
  payload retention. Train equal-parameter controls, package the selected
  module with the exact command in `MAIN_CUDA_FIRE_ORDER.json`, then adapt the
  proven re1t/js1b worker to the resulting archive/runtime SHAs. Admit only if
  the same T4 instrument falls from 34,970 to at most 33,630 flips when the
  trained archive delta is 1,707 B (otherwise recompute
  `ceil(0.785 * actual_delta_bytes)`) and joint pose/rate accounting is
  negative. A first-instance miss is INSTANCE-only;
  route Seg to #978 only after the optimal-form capacity/family ladder also
  fails the same break-even test.

## LIVE-HYPOTHESES

- Oriented token context may make the 1,340-flip rate hurdle reachable because
  its cross-fit top-budget contains an expected 8,380 base errors, 6.25x the
  hurdle, and it preserves direction plus class identity on the Road hub.
- Pre-TokenBlock injection may survive where SA1 did not because four existing
  nonlinear/dilated blocks can turn a small latent displacement into a
  partition-local RGB change before camera rounding; the actual receiver probe
  proves that path is live, though not that its sign is useful.
- Equal-parameter class and undirected controls remain necessary because the
  AUROC gain could be target-selection capacity rather than uniquely oriented
  realization capacity; only true-CUDA joint training separates those causes.

## DEAD-ENDS

- Explicit edge overlays are closed at the JS1C instance: 55,807 candidate
  flips versus 34,970 base, rho -2.727 versus 0.827795 required.
- Frozen-receiver singleton edits are closed at the JS8 formulation: 38 flips
  realized versus 4,314 needed.
- Additive causal-edge probability calibration is closed at the SR1
  formulation: only -2 bytes after charges.
- The shipped JS3/SA1 post-render hidden-4 conditioner is closed at that
  instance/formulation: it was exactly T4-inert and pure rate cost.
- The seeded EC1 nonzero design module and identity packaging controls are not
  candidates and must never be sent to a scorer as if trained.
