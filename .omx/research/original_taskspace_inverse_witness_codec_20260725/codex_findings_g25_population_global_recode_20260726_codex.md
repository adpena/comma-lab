# Codex findings — G25 population-global same-solution recode v2

Date: 2026-07-26  
Lane: `lane_g25_population_global_same_solution_recode_20260726`  
Verdict: `PROCEED` to standalone full-n600 receiver replay; research-only, not a candidate or score row

## Pointer-delta honesty

The exact competitive frontier pointer did not move.  G25 invoked no scorer or
contest evaluator.  It produced a smaller exact same-quantized-state research
archive and therefore remains a means, not goal completion.

## STORES CONSULTED

- frozen ep725 n600 source archive/runtime on `VertigoDataTier`;
- G20 spec, implementation, 81,027-byte same-state control, and receipt;
- G21 lattice-teacher/compaction-homotopy spec, especially H1;
- G17 selected-solution compiler and G19 action/costate vocabularies;
- canonical lane registry and subagent ownership ledger; and
- full CLAUDE.md/AGENTS.md/PROGRAM.md plus current Claude MEMORY index.

No public/donor archive, weights, code, targets, selectors, dictionaries, or
scorer products entered the payload.

## Exact result

The durable receipt is
`ep725_population_global_recode_v2_20260726_r2/receipt.json`.  R2 supersedes
the append-only first materialization receipt because the four Python sources
were mechanically formatted after that run; the selected archive bytes and
SHA-256 are identical, while R2 binds the receipt to the formatted source
hashes.

| complete object | exact archive bytes | SHA-256 |
|---|---:|---|
| frozen source | 83,838 | `149fefd097c1fa85c4afb6cb2d8ab20311035d7ba8063f1e72137b843a9b89f3` |
| G20 same-state control | 81,027 | `8e9c7ba0fdd1fc0fdff696c639821d6e64a3110bb8744f47ae0ab3d287cd70d8` |
| G25 V2 selected | **80,238** | `68351f57781d8fe60c05ab59fc250e48d6bb03e7cdf95b3d00987328d08d2a98` |

G25 is `-789` bytes versus G20 and `-3,600` bytes versus the frozen source.
The exact rate-coordinate changes are `-0.0005253627140133931` and
`-0.0023970922312398168`, respectively.  They are zero-distortion
same-state coordinates, not evaluator score rows.

Source and selected full-state digests are exactly equal:

`122b9928568c8e5ad4cf57424b0cf71bb2d0bd17659f9b26b825c7a77bc5c38e`.

All 17 named base tensors, all 38,400 signed-int8 population-code values,
and pose bytes compare equal.  Strict parse, exact EOF, canonical member
re-encode, deterministic archive rebuild x2, ZIP member reopen, and CRC pass.

## What the dynamic whole-object search selected

The structured search measured 6,669 complete archives and terminated only
after a full five-coordinate cycle was unchanged.  It selected:

- base storage transposes for `in_proj.weight` and `hidden.1.weight`;
- signed-zigzag storage for `hidden.1.weight`, `hidden.2.weight`, and
  `hidden.3.weight`;
- modulo-256 previous-pair same-slot differences over the full n600 code;
- storage permutation `(frame, latent_dimension, pair) = (1,2,0)`, which makes
  each latent dimension's entire 600-pair trajectory contiguous;
- separate base and population Brotli quality-10 streams; and
- default ZIP DEFLATE.

This result changed during composition.  G20's independently selected base
transposes were not the best base coordinates after population chronology and
coder layout changed.  Conversely, the early joint-stream provisional winner
lost to separate quality-10 streams after the complete knot was iterated.  The
selection therefore could not lawfully have been assembled from isolated
section winners.

## Interaction custody

For exact complete archive bytes, G25 retains

`I_B(a,b|x) = B_11 - B_10 - B_01 + B_00`.

The selected base-transform x population-transform four-corner hyperedge is
`-1` byte.  The selected transform-bundle x coder/container-bundle hyperedge is
`-274` bytes.  The receipt preserves all eight corner archives' hashes, bytes,
and configs.  These observations are indivisible; neither is distributed as
per-tensor, per-section, or additive action credit.

## Representation and geometry finding

The rate coordinate is a gauge problem over identical decoded state.  G20
changed tensor axes and pair chronology.  G25 exposed the missing population
chart: the entropy geometry is better represented as 64 long trajectories
(frame x latent dimension) than as 1,200 short row vectors.  The generic decoder
inverts that chart for free; the chosen chart/coder IDs are counted in the
archive manifest.

This is the concrete H1 lesson under our current ep725 selected solution:
population, chronology, tensor storage, and entropy container form one knot.
The correct action is not “save N bytes on tensor i,” but replace one complete
same-solution archive program with another and reprice the entire object.

## Triality and system wire-in

- Equation: exact same-state `REQUANTIZE_STORAGE`, with `Delta D_Q = 0` and
  `Delta B = -789` relative to G20.
- DSL: strict versioned `LVPG2` parser/encoder plus the complete-object action
  receipt in `ep725_population_global_recode_v2.py`.
- DAG: G21 H1 identity-population-recode -> standalone n600 receiver replay ->
  selected-solution compiler admission; scorer/eval remains downstream.
- Controller: the receipt exports one G17/G19-vocabulary substitutive action and
  two separate interaction hyperedges.  G19 v1 explicitly requires a schema
  extension because it currently accepts only the G20 receipt; V2 is not
  silently laundered through that parser.
- Continual learning: probe
  `g25_ep725_population_global_same_solution_recode_v2_20260726` is registered
  `PROCEED` on exact complete-object bytes.

## Exact blocker and next rung

`parse_population_global_member` is a real generic quantized-state receiver,
but the frozen ep725 `inflate.py` does not yet consume `LVPG2`.  Therefore full
n600 uint8 runtime equality remains owed.  The next rung is to compile this
generic V2 inverse into a standalone inflate runtime, replay all 600 pairs
against source/G20 output bytes, and reprice that exact archive/runtime object.
Only after that receiver closure may the same bytes approach candidate or
contest CPU/CUDA custody.
