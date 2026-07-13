# Round-5 successor: block2/block3 PRE-SE feature locus

**Date:** 2026-07-13 UTC  
**Lane:** `lane_replace_round5_pre_se_locus_20260713`  
**Mode:** `$0` local build + measurement only; `research_only=true`  
**Authority:** `[macOS-CPU advisory; NumPy-fp64 convex; CPU-torch exact costates]`  
**Score authority:** `NONE`; no evaluator, archive, paid dispatch, training run, or live-run mutation  
**Pointer delta:** `NONE`

## Result

`WIDER-FAMILY-KILL`, with `verdict_scope = FAMILY x TESTED-SINGLE-SOURCE-LOCI x
FIXED-REPLAY x STRICT-END-TO-END-RGB-TILEABILITY`.

At the identical realized `0.047017415364583336` area (`2311/49152` prefix cells), all four
preregistered rungs fail the `0.47` retained exact input-costate L2-square mass gate:

| Feature source | Convex: 20 pair-specific RankRLS MP heads | Nonlinear: 3-seed pair-gated MLP ensemble | Strict RGB tileability | Joint reopen gate |
|---|---:|---:|---|---|
| block2 PRE-SE | **MEASURED** `0.20233024422907497` — FAIL | **MEASURED** `0.2736871496424692` — FAIL | **MEASURED-CONSTRUCTION** `N` | FAIL |
| block3 PRE-SE | **MEASURED** `0.09314654496850622` — FAIL | **MEASURED** `0.31323809443347944` — FAIL | **MEASURED-CONSTRUCTION** `N` | FAIL |
| same-area exact oracle | **MEASURED-INHERITED** `0.5278150212253758` | same ceiling | diagnostic | n/a |

The nonlinear per-seed heldout values are:

- block2 PRE-SE: `{0.25575363895519304, 0.26102436701697396,
  0.2463958844547387}`, population standard deviation `0.006049248306522052` — stability PASS;
- block3 PRE-SE: `{0.2917730459955019, 0.2931756134277998,
  0.29563933902293554}`, population standard deviation `0.0015981016858292142` — stability PASS.

The exact same-area oracle remains above the bar, so the negative is not a claim that the annulus
lacks support mass. It is a scoped kill of the tested cheap single-source feature-locus family.

## The captured tensors are genuinely PRE-SE

The implementation captures the `forward_pre` input of the last MBConv SE module in each Round-5
encoder stage:

| Locus | Hook | **MEASURED** NCHW shape | Equal to preceding depthwise activation | Own SE applied |
|---|---|---:|---|---|
| block2 PRE-SE | `encoder.model.blocks.1.2.se` | `(1,144,96,128)` | `true` | `false` |
| block3 PRE-SE | `encoder.model.blocks.2.2.se` | `(1,288,48,64)` | `true` | `false` |

This confirms the requested feature-extraction point: the depthwise-convolution activation
immediately before that MBConv's own squeeze-excite recalibration. Hook-vs-depthwise equality is
checked on the real frozen model before measurement and is regression-tested on the same timm
EfficientNet-B2 graph.

## Tileability hypothesis: falsified by the full dependency graph

The hook is local **relative to its current MBConv input**, but it is not independently tileable
from RGB. The stage sequentials contain earlier MBConvs with their own SE global means:

- block2 PRE-SE has **MEASURED-CONSTRUCTION** `4` upstream SE global reductions;
- block3 PRE-SE has **MEASURED-CONSTRUCTION** `7` upstream SE global reductions.

Therefore the exact tensor in a tile still depends on full-frame means/gates computed upstream.
"Before its own SE" is necessary but not sufficient for strict end-to-end tileability. The
construction receipt records every upstream module. The clean successor law is

`T_strict(c) = 1[N_global_upstream(c) + N_global_own(c) = 0]`.

For both cuts `N_global_own=0`, but `N_global_upstream` is nonzero, hence tileability is
**CONFIRMED N**. This corrects the prompt hypothesis without changing the measured feature cell.

## Pre-SE cut FLOP cost

The Round-5 cost convention is reused unchanged: one multiply-add is one MAC/two FLOPs; a
forward-plus-input-VJP convolution path is charged four FLOPs per forward MAC. Batch norm,
pointwise activation, decoder interpolation, loss/argmax, localizer matrix multiplies, and
autograd bookkeeping remain excluded. Fractions are **DERIVED from MEASURED real tensor shapes**.

| Cut | Forward conv MACs | Forward + input-VJP conv FLOPs | Fraction of full teacher conv FLOPs | Upstream global reductions |
|---|---:|---:|---:|---:|
| block2 PRE-SE | `375131200` | `1500524800` | `0.03785634855148739` (`3.7856349%`) | `4` |
| block3 PRE-SE | `664007872` | `2656031488` | `0.0670083252029248` (`6.7008325%`) | `7` |

The matched-area conditional composition `c_label=p+(1-p)q` is **DERIVED**, not a wall-clock
claim: block2 gives `0.083093856252039` (`12.034584084854789x` conditional variable-cost ratio),
and block3 gives `0.11087518230855714` (`9.019150897241158x`). Because retained-mass and strict
tileability both fail, neither conditional sparse path is admitted.

## Apples-to-apples controls and n600 custody

Everything except the feature tap is inherited from the sealed Round-5 contract:

- identical seed `455`, V9 real `n600` renderer states, checkpoints `{ep150, ep251, ep275}`;
- identical deterministic `480` train / `120` untouched heldout split and `60`-state train-only dev;
- identical `0.047` requested area, `0.047017415364583336` realized area, `0.47` bar, and
  `0.5278150212253758` same-area oracle;
- identical 20 ordered source/competitor blocks, symmetric eigendecomposition, rank-truncated
  Moore-Penrose minimum-norm RankRLS optimum, with `40/40` certificates across the two loci;
- identical three seeds `{455,456,457}`, `32` hidden units, optimizer, batch size, 60-epoch cap,
  patience, train-only dev early stopping, and NumPy-fp32 inference reference;
- no post-heldout rung or tuning.

**MEASURED custody:** the `480` exact train support/mass targets are hash-closed read-only Round-5
artifacts, and the untouched `120` heldout states received fresh exact frozen-SegNet costates.
The union is exactly all `600` real states. Campaign-honest accounting is `600` exact starts,
`600` unique completed states, and `0` retries. Reuse avoids repeating settled exact gradients;
it does not reduce evidence coverage.

## Scoped wider verdict

The exhausted tested cells are:

1. shallow pre-first-SE: strictly tileable but too primitive; Round 4 best convex retained mass
   **MEASURED-INHERITED** `0.20172451295048283`;
2. block2/block3 post-SE: Round 5 retained mass **MEASURED-INHERITED** `0.13046753525944724`
   convex / `0.29462633883840517` nonlinear, plus structural non-tileability;
3. block2/block3 pre-own-SE: the four new values above, all below `0.47`, and strict
   non-tileability from upstream SE.

Thus no tested single-source locus simultaneously supplies enough retained mass and strict RGB
tileability. The verdict does **not** kill:

- a genuinely SE-free or local-attention deep feature extractor with zero upstream global paths;
- a charged, explicitly cached/donated full-frame SE-gate broadcast;
- multi-source, dense-label, larger-attention, or evaluator-inverse localizers;
- transition-complete on-policy/FORE successors or other replay distributions/seeds;
- the evaluator-equivalent witness paradigm.

No `#455` cheap-localization reopen is registered because the joint gate fails.

## Triality and wire-in

- DSL leg: `tac.witness_dsl.pre_se_locus_policy_20260713` compiles the sealed Round-5 inheritance
  and the only permitted feature-locus delta.
- Equations leg: `tac.canonical_equations.pre_se_locus_20260713` registers the structural
  upstream-global-dependency tileability law and the n600 retained-mass anchor through an isolated
  locked-registry test.
- DAG leg: `.omx/research/pre_se_locus_DAG_FEED_20260713.md` is the standalone FEED node.

The shared canonical DAG and `.omx/state/canonical_equations_registry.jsonl` were already dirty
under live sibling ownership, so their append is `DEFERRED_MAIN`; neither shared file was hand-edited.

- Sensitivity-map contribution: exact per-cell costate mass, ordered class-pair blocks, and the
  block2/block3 PRE-SE channel charts.
- Pareto constraint: retained mass × matched area × strict tileability × cut fraction. Archive
  bytes and evaluator score remain unmeasured/non-binding.
- Bit allocator: non-binding; failed localization cannot expose an admitted sparse support budget.
- Cathedral/autopilot: `REFUSE`; no live/paid/trainer/evaluator dispatch.
- Continual learning: typed receipt + equation + DAG FEED; shared registry append deferred to main.
- Probe disambiguator: both block2 and block3 loci and both convex/nonlinear interpretations were
  shipped and measured rather than chosen by intuition.

## Durable custody

- Receipt: `experiments/results/pre_se_locus_20260713/receipt.json`, SHA-256
  `660a5763831539715d8593df0ba40a0f50f660af93c0e5bcd1d399ea340d1abb`.
- Completion: `experiments/results/pre_se_locus_20260713/complete.json`, SHA-256
  `7686ebd872b16b2e73dbafd2d2f748df1bd1afb0b43276ec2c8bcda3db2f5faa`.
- Cleanup manifest: `experiments/results/pre_se_locus_20260713/cleanup_manifest.json`, SHA-256
  `8921581370341a084d6532aa8cc21ce861b046e3b6bae6a87959489e196d3a8a`.
- Preregistration: `experiments/results/pre_se_locus_20260713/preregistration.json`, SHA-256
  `3360182c2ea5e920fadfb79f0ecf7130eed29e8555edda33a08b66e1b32e1b6f`.
- Preserved artifact footprint: **MEASURED** `721M`, `945` files; no destructive cleanup.
- Command: `.venv/bin/python tools/probe_pre_se_locus_20260713.py --resume`.
- Focused verification: `13 passed` across formulation and equation suites; scoped Ruff and
  `py_compile` pass; terminal resume revalidates the sealed receipt without new teacher calls.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; `docs/operating_manual_craft_handoff.md`;
the v7.5/v8 operating SPECs; `reports/latest.md`; `.omx/state/lane_registry.json`;
`.omx/state/subagent_progress.jsonl`; the 2026-07-13 replace directives; latest Codex session,
design, and council memos; Round-4/5 formulation, policy, probe, receipt, cost, target, and DAG
artifacts; frozen GT/scorer/checkpoint custody surfaces named in the run contract.
