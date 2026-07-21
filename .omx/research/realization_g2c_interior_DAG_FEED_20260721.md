# DAG FEED — realization G2c interior fills

`research_only=true` · Task #578 · `[macOS-CPU advisory]` · pointer unmoved.

## Nodes

- `G2C_SEED_CELLS`: canonical `seed_compose_b2` class field and constraints.
- `G2C_R1_FIXED`: generic fixed-radius RGB class codebook, zero bytes.
- `G2C_R2_MARGIN`: frozen constant-tile max-margin RGB codebook, zero bytes.
- `G2C_R3_HOPFIELD`: class-conditioned frozen-bank prox with local cell query,
  zero bytes.
- `G2C_R4_EXCEPT`: R2 plus parse-backed R2-dying ordinal/RGB records, 16,568
  bytes at n600.
- `G2C_FACTOR2`: #580 full-kernel and #583 exact uint8 feasibility/verification.
- `G2C_HARD`: native CPU-Torch SegNet/PoseNet hard oracle.
- `G2C_ADMISSION`: `predict_project_realization_admissibility_v1`.
- `G2C_NEXT_SPATIAL`: spatial contextual cell optimizer plus explicit frame0
  pose carrier; unbuilt reformulation.

## Edges and decisions

`G2C_SEED_CELLS -> {R1,R2,R3} -> G2C_FACTOR2 -> G2C_HARD -> G2C_ADMISSION`.
All three zero-byte edges reach exact factor-2 transport but admission remains
false.  `R2 -> R4_EXCEPT -> FACTOR2 -> HARD` is priced and non-improving.
`G2C_ADMISSION -> G2C_NEXT_SPATIAL` is the reactivation edge; no family closure.

## Unified-stack wire-in

1. Sensitivity map: target-logit winner/rival margin bucket is recorded per
   declared write; R2 has 114 positive-margin survivors and 3,074 nonpositive
   failures.
2. Pareto constraint: the existing hard admission equation prevents semantic
   or pose laundering despite exact transport.
3. Bit allocator: R4's 16,568 counted bytes buy no admission and reduce write
   survival from 114 to 42; do not allocate to this single-pixel formulation.
4. Cathedral/autopilot: consume the second n600 empirical anchor and route only
   to the explicit spatial/frame0 reformulation edge.
5. Continual-learning posterior: the canonical equation now retains both the
   charged source-RGB control and zero-byte R2 receiver anchors.
6. Probe disambiguator: R1/R2/R3/R4 are callable sibling modes in one runner;
   the hard oracle, not policy naming, arbitrates them.

## Triality

- DSL/ABI leg: existing seed grammar plus decoder-derived RGB custody contract;
  no scorer weights or raster payload enter R1–R3.
- DAG leg: this file and immutable n16/n64/n600 checkpoint graph.
- Equations leg: #583 exact resize feasibility plus updated
  `predict_project_realization_admissibility_v1` empirical anchors.

Durable receipt:
`.omx/research/realization_g2c_interior_receipt_20260721.json`.
