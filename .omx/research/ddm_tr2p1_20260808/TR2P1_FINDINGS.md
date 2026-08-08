# TR2P1 TROT residual/range-AC byte race - 2026-08-08

Tags: [no-triality] [p0-ledger-ok]

## Answer First

No scorer, no evaluator, no Metal/GPU, no paid job, and no archive promotion ran.
Best reference-form TROT AC challenger: `trot_q2p0_ac` at `2637555` B (brotli-q11), delta `+2172998` B vs best incumbent control `cr1_edge_conditioned_lzma1_raw_incumbent` at `464557` B.
Verdict: `LOSS-w/-bytes; FORMULATION falsifier met for q-family joint-from-marginals conditional range-AC coding of CR1 selected edge support`.

| arm | coding family | best bytes | delta vs CR1-LZ | delta vs best incumbent | decode equality | verdict |
|---|---|---:|---:|---:|---|---|
| identity_container_control | delta_container | 464660 B (lzma1-raw) | +103 B (0.022%) | +103 B (0.022%) | True | CONTROL |
| incumbent_edge_context_ac | conditional_range_ac | 2582804 B (brotli-q11) | +2118247 B (455.971%) | +2118247 B (455.971%) | True | CONTROL |
| marginals_only_residual | lz_residual_baseline | 1133907 B (lzma1-raw) | +669350 B (144.084%) | +669350 B (144.084%) | True | INSTRUMENT-LZ-BASELINE |
| marginals_only_ac | conditional_range_ac | 2593397 B (brotli-q11) | +2128840 B (458.252%) | +2128840 B (458.252%) | True | CONTROL |
| trot_q0p5_residual | lz_residual_baseline | 1291866 B (brotli-q11) | +827309 B (178.086%) | +827309 B (178.086%) | True | INSTRUMENT-LZ-BASELINE |
| trot_q0p5_ac | conditional_range_ac | 3036649 B (brotli-q11) | +2572092 B (553.666%) | +2572092 B (553.666%) | True | LOSS-w/-bytes |
| trot_q0p8_residual | lz_residual_baseline | 1314955 B (brotli-q11) | +850398 B (183.056%) | +850398 B (183.056%) | True | INSTRUMENT-LZ-BASELINE |
| trot_q0p8_ac | conditional_range_ac | 2691482 B (brotli-q11) | +2226925 B (479.365%) | +2226925 B (479.365%) | True | LOSS-w/-bytes |
| trot_q1p0_residual | lz_residual_baseline | 1315291 B (brotli-q11) | +850734 B (183.128%) | +850734 B (183.128%) | True | INSTRUMENT-LZ-BASELINE |
| trot_q1p0_ac | conditional_range_ac | 2665580 B (brotli-q11) | +2201023 B (473.790%) | +2201023 B (473.790%) | True | LOSS-w/-bytes |
| trot_q1p2_residual | lz_residual_baseline | 1321772 B (brotli-q11) | +857215 B (184.523%) | +857215 B (184.523%) | True | INSTRUMENT-LZ-BASELINE |
| trot_q1p2_ac | conditional_range_ac | 2653884 B (brotli-q11) | +2189327 B (471.272%) | +2189327 B (471.272%) | True | LOSS-w/-bytes |
| trot_q2p0_residual | lz_residual_baseline | 1329394 B (brotli-q11) | +864837 B (186.164%) | +864837 B (186.164%) | True | INSTRUMENT-LZ-BASELINE |
| trot_q2p0_ac | conditional_range_ac | 2637555 B (brotli-q11) | +2172998 B (467.757%) | +2172998 B (467.757%) | True | LOSS-w/-bytes |

## Inputs

- GT argmax: `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/gt_argmax_n600.npy` (b74a14b226a5aceb5824899898bcb06e5413c54b7db2441268da7bc91a10db5d)
- Current argmax: `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_argmax_n600.npy` (5e903de650e60ec6a64b34eb455fa1bc911223551d0b31e9ae45cc906e1490be)
- CR1 incumbent: `/Volumes/VertigoDataTier/pact/ddm_cr1_20260808/payloads/p2_edge_conditioned_support.lzma1-raw.bin` (0a53f649768c61912399ccab14e4d3323998e47235992091e2a9e28cf7259fe1)
- Axis: `[byte-only scorer-free]`.
- Selection: `n600_all_pairs_no_prefix`.

## Recall Evidence

| source or query | result | impact |
|---|---|---|
| MEMORY.md query: ddm_tr2p1\|TR2P1\|tr2p1\|ddm_tr2\|common_contract\|20260808 | No TR2P1-specific prior memory entry was found in MEMORY.md. | Used live charter, TR2, CR1, and corpus receipts rather than a recalled shortcut. |
| .omx/research/ddm_tr2_20260808/TR2_CROSSWALK.md and TR2_ROWS.jsonl rows 1, 3, 6 | TR2 pre-registered only this CR1 same-payload q-family residual race; sparse-plan claims were lesson-only. | Kept verdict scoped to this formulation and did not claim TROT sparsity or metric replacement. |
| .omx/research/ddm_tr2p1_20260808/CHARTER.md amendment 2026-08-08 | The operator correction requires conditional arithmetic/range coding; LZ residual rows alone are instrument-scoped. | Added incumbent edge-context AC plus TROT conditional range-AC rows and made the FORMULATION verdict depend only on the AC race. |
| .omx/research/ddm_cr1_20260808/CR1_FINDINGS.md, CR1_RECEIPT.json, CR1_ROWS.jsonl row 2 | CR1 P2 measured 464557 B edge-conditioned lzma1-raw with exact decode equality on selected n600 supports. | Set the pass threshold and re-decoded the incumbent artifact instead of re-measuring CR1. |
| content query over GDL1/RL1/SX1/TR2/BD1 scopes: joint-from-marginals\|TROT\|Tsallis\|Sinkhorn\|edge-conditioned\|same-coder\|#940\|Road<->Lane\|separatrix | Found GDL1-P2 edge-conditioned fire-order, SX1 separatrix concentration, RL1 interface pricing, and TR2 same-coder requirement beyond the charter seeds. | Used top-five class-pair edges, n600/no-prefix selection, same coder set, and exact decode equality. |
| content query: #940\|same-coder\|races-not-reputation\|same payload\|decode equality | Found repeated same-coder race doctrine and TR2 citation of ddm_sv2 as the #940 blocker. | Reported all coders per arm and selected by real compressed bytes, not representation reputation. |
| tools/list_canonical_equations.py --json | Canonical registry was consulted; relevant hits reinforced scorer-free byte-race and transfer/equality boundaries, with no TR2P1 superseding equation found. | Receipt stays a byte-only measurement with score_claim=false and promotion_eligible=false. |

## Solver Validation

Fixture `3x4 active support embedded in scorer grid`: q=1 matched the independent Sinkhorn/IPF reference with max plan delta `0.000e+00`; row/column max errors `0.000e+00` / `2.220e-16`; residual decode equality `True`; conditional range-AC decode equality `True`.

## Counted Bytes

| arm | compressed total | raw total | marginals raw | side-info raw | residual raw | AC stream raw | tags raw | framing raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| identity_container_control | 464660 | 2732163 | 0 | 0 | 2716738 | 0 | 3409 | 12016 |
| incumbent_edge_context_ac | 2582804 | 4552519 | 2096981 | 0 | 0 | 2428240 | 3764 | 23534 |
| marginals_only_residual | 1133907 | 6944244 | 2096981 | 0 | 4825553 | 0 | 3728 | 17982 |
| marginals_only_ac | 2593397 | 4563681 | 2096981 | 0 | 0 | 2439420 | 3758 | 23522 |
| trot_q0p5_residual | 1291866 | 6946003 | 2096981 | 0 | 4827329 | 0 | 3711 | 17982 |
| trot_q0p5_ac | 3036649 | 5041031 | 2096981 | 0 | 0 | 2916793 | 3741 | 23516 |
| trot_q0p8_residual | 1314955 | 6946313 | 2096981 | 0 | 4827639 | 0 | 3711 | 17982 |
| trot_q0p8_ac | 2691482 | 4669094 | 2096981 | 0 | 0 | 2544840 | 3741 | 23532 |
| trot_q1p0_residual | 1315291 | 6944494 | 2096981 | 0 | 4825820 | 0 | 3711 | 17982 |
| trot_q1p0_ac | 2665580 | 4639822 | 2096981 | 0 | 0 | 2515558 | 3741 | 23542 |
| trot_q1p2_residual | 1321772 | 6942203 | 2096981 | 0 | 4823529 | 0 | 3711 | 17982 |
| trot_q1p2_ac | 2653884 | 4626844 | 2096981 | 0 | 0 | 2502577 | 3741 | 23545 |
| trot_q2p0_residual | 1329394 | 6938749 | 2096981 | 0 | 4820075 | 0 | 3711 | 17982 |
| trot_q2p0_ac | 2637555 | 4609380 | 2096981 | 0 | 0 | 2485106 | 3741 | 23552 |

## Boundaries

- These are byte-only scorer-free measurements over cached argmax labels.
- The q-family solve is generic decode-time algorithm; counted payload includes marginals, AC streams or residuals, tags, and framing.
- Conditional range-AC rows are the reference-form verdict carrier required by the 2026-08-08 amendment.
- Residual-through-LZ rows are retained only as instrument baselines, not as formulation verdict carriers.
- No video-derived side-information matrix was hidden in code or omitted from bytes; side-info bytes are zero because the cost is derived only from counted marginals.
- No RGB receiver, archive parse-back, SegNet/PoseNet scorer survival, or score improvement is claimed.
- Negative verdict scope is the pre-registered conditional-AC FORMULATION only, not the TROT family outside this CR1 selected-support payload.

## Follow-On Disposition

| item | disposition | fire order |
|---|---|---|
| TR2-P1 | FIRED | This receipt is the pre-registered byte-only q-family residual plus amendment-required conditional range-AC race against CR1's incumbent. |
| TR2-P1-implementation-reference | FIRED | The local q-family solver passed deterministic fixtures before the CR1 payload race; no unpinned dependency was vendored. |
| #984 rate axis / CR1 successor consumer | QUEUED-WITH-FIRE-ORDER | Consume this row only as a scorer-free byte-race negative unless a future arm supplies a new counted side-info source or conditional probability model and repeats same-payload decode-equality AC racing. |

## Frontier Honesty

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`. Contest pointer remains `borrowed/unmoved 0.1910828242 [contest-CPU]`.
