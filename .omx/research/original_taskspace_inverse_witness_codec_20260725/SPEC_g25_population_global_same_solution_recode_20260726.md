# G25 specification — population-global same-solution recode v2

Date: 2026-07-26  
Lane: `lane_g25_population_global_same_solution_recode_20260726`  
Status: research-only exact-state recode; no candidate, score, promotion, or pointer claim

## Objective

Construct and measure a new versioned spelling of our frozen ep725 n600
`LVLS1` quantized state.  The state is the complete named base-tensor map plus
the `[600,2,32]` population code and pose bytes.  The search variable is a
complete archive object:

```text
(base storage coordinates, population chronology, population storage order,
 inner coder/layout, outer ZIP spelling) -> exact archive.zip bytes.
```

No per-tensor or per-section byte saving is authoritative.  Every retained row
is a complete ZIP object.  Base/code, transform/coder, and inner/outer effects
are nonadditive and selected dynamically by exact final bytes.

## Frozen custody and originality

- Source archive SHA-256:
  `149fefd097c1fa85c4afb6cb2d8ab20311035d7ba8063f1e72137b843a9b89f3`.
- Source `0.bin` SHA-256:
  `f0c3e648f00f52e48c7be98997fb7dd57c2e5a607ed385846931af68f88cc78c`.
- Source runtime SHA-256:
  `4b54d512565f7275c53f697a931dd087222a36a69495b6e536a6b65dede36224`.
- G20 same-state control archive SHA-256:
  `8e9c7ba0fdd1fc0fdff696c639821d6e64a3110bb8744f47ae0ab3d287cd70d8`.

Only our source state is read.  No public/donor archive, code, weights,
checkpoint, target, scorer output, dictionary, or selector enters the payload.
Generic inverse transforms and generic lossless decoders are lawful decoder
mechanisms.  Every selected transform ID, fitted ordering, header, frequency
table, dictionary, or other video-derived operand would be counted.  V2 uses
only a sealed generic transform menu whose IDs are counted in its manifest.

## New wire and real receiver

V2 uses new magic `LVPG2\0`; it never silently mutates `LVLS1`.  A strict parser
owns exact EOF, canonical JSON, safe ZIP, dtype/shape, transform, coder, and
inverse checks.  The member has canonical manifest, base payload, population
payload, and pose payload.  Layout can keep base/population separate or code
their concatenated raw streams jointly.  In the joint case the logical split is
derived from the manifest's exact tensor shapes and `[1200,32]` code shape.

The generic inverse supports:

- exhaustive C/F storage choices for every nondegenerate 2-D base tensor;
- reversible signed-zigzag choices over the same tensor set;
- full-population axis permutations over pair/frame/channel;
- reversible modulo-256 differences, with sealed segment resets, and XOR
  controls;
- raw, Brotli, zlib, bz2, and LZMA inner coders available in the current
  environment;
- separate and joint payload layouts; and
- ZIP STORE, default DEFLATE, and explicit DEFLATE levels 1 through 9.

Decode reconstructs every named signed-int8 base tensor, every one of 38,400
population-code bytes, and the pose bytes exactly.  Canonical parse/re-encode
and two independent builds must reproduce the selected member and archive.

## Structured whole-object search

The finite surface is searched by deterministic exact coordinate descent.  A
coordinate sweep physically builds and prices every complete archive while all
other coordinates remain at the current complete-object parent.  Sweeps cover
base transpose, base zigzag, population transform, inner layout/coder, and
outer ZIP profile.  A strict whole cycle with no selected-object change is the
termination proof; strict size/SHA tie order prevents oscillation.  Source and
G20 are explicit complete-object controls.  This is honestly a structured
recode search, not a solver/compiler claim and not proof of the global minimum
over codecs outside the sealed menu.

Every stage is atomic and preserved.  The materializer writes preflight, each
coordinate sweep, interaction closure, and finalization checkpoints; resume
continues from the last complete stage.  Since the source is 84 KB, no bulky
artifact is produced.  Temporary build bytes are memory-resident.  The durable
selected archive is explicitly `not_a_candidate`.

## Exact interaction custody

After selection, build complete four-corner lattices for:

1. selected base transform x selected population transform under one fixed
   selected coder/container; and
2. selected transform bundle x selected coder/container bundle.

For exact archive bytes `B`, retain only

```text
I_B = B_11 - B_10 - B_01 + B_00.
```

The four archive hashes/bytes and `I_B` are one indivisible hyperedge.  They
must not be spread over tensors, sections, or actions.

## Substitutive action and downstream contract

The selected row is one `REQUANTIZE_STORAGE` substitutive action over unchanged
`base`, `population_code`, and `entropy_context` reservoirs.  Its authority is
the exact parent/candidate archive bytes plus whole-state equality.  The action
exports G17/G19 vocabulary and declares:

- zero complete quantized-state delta;
- exact whole-object rate delta;
- section-marginal attribution forbidden;
- interaction hyperedges retained separately;
- full n600 runtime replay and contest CPU/CUDA remain owed; and
- G19 v1 receipt ingestion needs an explicit schema extension rather than
  silently laundering this V2 receipt as G20.

The decoder-equality proof depends on archive bytes, member mapping, V2 receiver,
pair order, and equality algorithm.  It does not depend on the volatile frontier
pointer.  Admission against a competitive target is a later, separate dependency
domain.

## Acceptance

1. Strict source/G20 custody passes.
2. The real n600 state drives the search; synthetic fixtures are unit tests only.
3. Selected V2 archive is smaller than or control-equal to G20 or the control is
   retained.
4. Full named-array/code/pose equality and domain-separated state hashes match.
5. Canonical member re-encode, deterministic archive rebuild x2, ZIP reopen/CRC,
   and exact EOF pass.
6. Full structured-search checkpoints, candidate-set roots, exact whole-object
   controls, and interaction corners are durable.
7. No scorer/eval is invoked and no candidate/score/frontier claim is emitted.

## Canonical-vs-unique decision per layer

- Reuse G20 source parser and exact ZIP custody because their premises exactly
  fit this source object.
- Use a new V2 wire/parser/search because population-global transforms, joint
  layout, resumability, and interaction custody are absent from G20.
- Reuse G17/G19 action vocabulary, but do not modify or impersonate their frozen
  schemas; export an explicit typed handoff with a named ingestion blocker.
- Reuse standard/bundled lossless mechanisms only as generic algorithms; all
  selected IDs and bytes remain inside the priced archive.

## Six-hook wire-in

- Sensitivity: decoded-state delta is exactly zero; rate only.
- Pareto: only same-state complete objects may dominate.
- Bit allocator: one indivisible `REQUANTIZE_STORAGE` action.
- Autopilot: full n600 runtime replay is the next required gate.
- Continual learning: retain complete-object stage rows and two interaction
  hyperedges, never section credits.
- Disambiguator: the measured whole-object coordinate search selects the sealed
  transform/coder/container knot.
