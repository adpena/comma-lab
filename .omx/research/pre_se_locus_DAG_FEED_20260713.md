# Standalone DAG FEED — Round-5 block2/block3 PRE-SE feature locus

**Date:** 2026-07-13 UTC  
**Lane:** `lane_replace_round5_pre_se_locus_20260713`  
**Node:** `FEED-455/replace-round5-pre-se-locus`  
**Status:** `WIDER-FAMILY-KILL`; `research_only=true`; shared-DAG append `DEFERRED_MAIN`  
**Verdict scope:** `FAMILY x TESTED-SINGLE-SOURCE-LOCI x FIXED-REPLAY x STRICT-END-TO-END-RGB-TILEABILITY`  
**Pointer delta:** `NONE`

## Settled incoming edges

```text
Round 4 shallow pre-first-SE source
  -> MEASURED retained mass 0.20172451295048283 at 0.047017415364583336 area
  -> strict tileability yes; retained-mass gate fail

Round 5 block2/block3 post-SE source
  -> MEASURED retained mass 0.13046753525944724 convex / 0.29462633883840517 nonlinear
  -> strict tileability no after first SE; retained-mass gate fail
```

Those cells remain read-only. The new edge changes only the feature tap.

## Preregistered PRE-SE edge

```text
sealed V9 n600 replay, seed455, checkpoints {ep150,ep251,ep275}
  -> immutable exact Round-5 train targets: 480 states
  -> fresh untouched exact CPU-SegNet heldout costates: 120 states
  -> identical top-2311 / 49152 area and 0.47 retained-mass bar
  -> SOURCE A: blocks.1.2.se forward-pre input (144x96x128)
  -> SOURCE B: blocks.2.2.se forward-pre input (288x48x64)
  -> each source fitted independently
  -> RUNG 1: 20 exact pair-block RankRLS Moore-Penrose heads
  -> RUNG 2: 3 deterministic pair-gated width-32 MLP seeds
  -> train-only dev early stop; no heldout tuning or third rung
  -> joint reopen iff retained mass >=0.47 AND strict RGB tileability
```

Preregistration SHA-256:
`3360182c2ea5e920fadfb79f0ecf7130eed29e8555edda33a08b66e1b32e1b6f`.

## Measurement append

| Locus | Convex retained mass | Nonlinear retained mass | Strict RGB tileability | Verdict |
|---|---:|---:|---|---|
| block2 PRE-SE | `0.20233024422907497` | `0.2736871496424692` | `N` (`4` upstream SE reductions) | FAIL |
| block3 PRE-SE | `0.09314654496850622` | `0.31323809443347944` | `N` (`7` upstream SE reductions) | FAIL |

Same-area oracle: `0.5278150212253758`. Campaign custody: `480` inherited exact train targets +
`120` fresh exact heldout targets = real `n600`; `0` retries.

## Structural equation edge

`T_strict(c) = 1[N_global_upstream(c) + N_global_own(c) = 0]`.

Both captured tensors are exactly before their own SE (`N_global_own=0`), but their upstream
counts are `4` and `7`. The prompt's local-relative-to-MBConv-input property is true; the required
end-to-end independent RGB tileability is false.

Canonical successor module: `tac.canonical_equations.pre_se_locus_20260713`.

## Cost edge

| Cut | DERIVED fraction of full teacher conv FLOPs | DERIVED conditional `c_label` at matched area |
|---|---:|---:|
| block2 PRE-SE | `0.03785634855148739` | `0.083093856252039` |
| block3 PRE-SE | `0.0670083252029248` | `0.11087518230855714` |

These are shape-derived FLOP compositions, not implemented sparse-kernel or wall-clock claims.

## Terminal routing

```text
joint PRE-SE gate FAIL
  -> no #455 cheap-localization reopen
  -> WIDER-FAMILY-KILL within tested single-source loci / fixed replay / strict RGB tileability
  -> admissible reactivation only for:
       zero-upstream-global deep extractor
       charged cached/donated SE gates
       multi-source or dense-label localizer
       transition-complete on-policy successor
       different replay distribution/seed
```

Shared DAG and equation-registry appends are `DEFERRED_MAIN` because both canonical shared
surfaces were dirty under live sibling ownership. This standalone node and the registerable
equation module are the durable handoff; no shared hot file was edited.

Receipt: `experiments/results/pre_se_locus_20260713/receipt.json`, SHA-256
`660a5763831539715d8593df0ba40a0f50f660af93c0e5bcd1d399ea340d1abb`.

