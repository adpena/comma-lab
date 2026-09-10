Implemented, measured, documented, and committed as `01b13a69b`; serializer returned rc 0.

The move43 arithmetic was stale:

- Held-distortion archive cap: **154,507 B**, not 137,986 B.
- Required saving: **25,959 B**, not 42,480 B.
- Replaceable tail: 119,969 B, or 66.4773% of the archive.
- Success inequality: `P_program + R_exact <= 94,010 B`.
- GF1 screening inequality: `P_program + 0.2909*M < 94,010.265605 B`.

The measured scanline construction is `FORMULATION-NO-GO`. Its best n600 point was K=6:

- Program: 223,494 B
- Mismatches: 234,295
- Real tile64-time residual: 105,628 B
- Complete replacement: **329,122 B**
- Excess over gate: **235,112 B**
- Full-field decode: 3.63 seconds
- Exact correction back to the move43 field: passed

The three constructed vehicles are:

- Ordered scanline partition program: implemented and closed at the six tested capacities. Its geometric motivation is anchored by [PolyLaneNet](https://arxiv.org/abs/2004.10924) and [DiffVG](https://people.csail.mit.edu/tzumao/diffvg/).
- Categorical Cool-Chic distillation: train a counted coordinate decoder on the retained K=8 teacher. It must compress 306,042 B to at most 68,322 B while preserving the teacher’s 88,304 mismatches—a required 4.4794× reduction. This uses the [COOL-CHIC paper](https://arxiv.org/abs/2212.05458) and [official implementation](https://github.com/Orange-OpenSource/Cool-Chic) as mechanism anchors.
- Occlusion-aware tropical SDF atom dictionary: a class-protected, temporally reused local atom bank, anchored by [DeepSDF](https://github.com/facebookresearch/DeepSDF) and [msdfgen](https://github.com/Chlumsky/msdfgen). Its 124,021-mismatch prediction remains hypothetical.

Artifacts:

- [Main memo](/Users/adpena/Projects/pact/.omx/research/ddm_gdc1_generator_door_construction_design_20260910.md)
- [Result receipt](/Users/adpena/Projects/pact/.omx/research/ddm_gdc1_20260910/RESULT.json)
- [Falsifier implementation](/Users/adpena/Projects/pact/experiments/ddm_gdc1_scanline_partition_program.py)
- Retained payload store: `/Volumes/VertigoDataTier/pact/ddm_gdc1_generator_door/scanline_v1/`
- Retention: 167 files, 1,648,591,638 B; independent rehash and terminal scientific replay passed.

Validation: Ruff clean, 6 tests passed, 32/32 Python entities reviewed twice. No training, scorer, Modal, MPS, candidate archive, or score measurement occurred.

Composition frontier remains **S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43)**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN-assigned GDC2 arm; consumer store: `/Volumes/VertigoDataTier/pact/ddm_gdc2_categorical_coolchic_k8_distill/`; fire trigger: explicit MAIN authorization of the training burn. Execute the governed categorical K=8 distillation specification; allocate no scorer unless physical packet plus real residual is ≤94,010 B and n600 decode is ≤900 seconds.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `.omx/state/main_hot_state.md`; fire trigger: the next managed hot-state refresh consuming commit `01b13a69b`. Replace the stale 42,480 B premise with 25,959 B and fold the finite scanline formulation.

## LIVE-HYPOTHESES

- Categorical Cool-Chic distillation may compress the useful K=8 teacher structure by 4.4794× because it replaces explicit transition addresses with a shared learned coordinate function. This remains aggressive and unmeasured.
- Local class-protected SDF atoms may succeed where global Laguerre and explicit worldsheets failed because 86.84% of HG1 errors lie within four pixels of a boundary.
- The K=8 render is a useful teacher despite its losing packet: it reaches 88,304 mismatches, mostly concentrated in Lane, and supplies an exact n600 target for implicit address compression.

## DEAD-ENDS

- The exact-DP ordered scanline grammar at K=4/6/8/12/16/24 is closed at FORMULATION scope; every packet alone exceeds the 94,010 B gate.
- The move43 137,986 B cap and 42,480 B demand are stale-premise arithmetic.
- GF1’s 0.2909 B/mismatch is not an actual successor rate; K=6 measured 0.450833 B/mismatch.
- Persistence, rigid shared-field warps, class-blind dyadic overlays, one global Laguerre diagram, exact Lane contours, and the registered explicit worldsheet should not be renamed and repeated.

