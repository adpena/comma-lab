# TR2P1 TROT residual byte race - 2026-08-08

Tags: [no-triality] [p0-ledger-ok]

## Answer First

No scorer, no evaluator, no Metal/GPU, no paid job, and no archive promotion ran.
Best TR2P1 challenger: `trot_q0p5_residual` at `1292070` B (brotli-q11), delta `+827513` B vs CR1's `464557` B incumbent.
Verdict: `LOSS-w/-bytes; FORMULATION falsifier met for q-family joint-from-marginals residual coding of CR1 selected edge support`.

| arm | best bytes | delta vs CR1 | decode equality | verdict |
|---|---:|---:|---|---|
| identity_container_control | 464660 B (lzma1-raw) | +103 B (0.022%) | True | LOSS-w/-bytes |
| marginals_only_residual | 1130973 B (lzma1-raw) | +666416 B (143.452%) | True | LOSS-w/-bytes |
| trot_q0p5_residual | 1292070 B (brotli-q11) | +827513 B (178.129%) | True | LOSS-w/-bytes |
| trot_q0p8_residual | 1316311 B (brotli-q11) | +851754 B (183.348%) | True | LOSS-w/-bytes |
| trot_q1p0_residual | 1316436 B (brotli-q11) | +851879 B (183.374%) | True | LOSS-w/-bytes |
| trot_q1p2_residual | 1322174 B (brotli-q11) | +857617 B (184.610%) | True | LOSS-w/-bytes |
| trot_q2p0_residual | 1336591 B (brotli-q11) | +872034 B (187.713%) | True | LOSS-w/-bytes |

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
| .omx/research/ddm_cr1_20260808/CR1_FINDINGS.md, CR1_RECEIPT.json, CR1_ROWS.jsonl row 2 | CR1 P2 measured 464557 B edge-conditioned lzma1-raw with exact decode equality on selected n600 supports. | Set the pass threshold and re-decoded the incumbent artifact instead of re-measuring CR1. |
| content query over GDL1/RL1/SX1/TR2/BD1 scopes: joint-from-marginals\|TROT\|Tsallis\|Sinkhorn\|edge-conditioned\|same-coder\|#940\|Road<->Lane\|separatrix | Found GDL1-P2 edge-conditioned fire-order, SX1 separatrix concentration, RL1 interface pricing, and TR2 same-coder requirement beyond the charter seeds. | Used top-five class-pair edges, n600/no-prefix selection, same coder set, and exact decode equality. |
| content query: #940\|same-coder\|races-not-reputation\|same payload\|decode equality | Found repeated same-coder race doctrine and TR2 citation of ddm_sv2 as the #940 blocker. | Reported all coders per arm and selected by real compressed bytes, not representation reputation. |
| tools/list_canonical_equations.py --json | Canonical registry was consulted; relevant hits reinforced scorer-free byte-race and transfer/equality boundaries, with no TR2P1 superseding equation found. | Receipt stays a byte-only measurement with score_claim=false and promotion_eligible=false. |

## Solver Validation

Fixture `3x4 active support embedded in scorer grid`: q=1 matched the independent Sinkhorn/IPF reference with max plan delta `0.000e+00`; row/column max errors `0.000e+00` / `2.220e-16`; residual decode equality `True`.

## Counted Bytes

| arm | compressed total | raw total | marginals raw | side-info raw | residual raw | tags raw | framing raw |
|---|---:|---:|---:|---:|---:|---:|---:|
| identity_container_control | 464660 | 2732163 | 0 | 0 | 2716738 | 3409 | 12016 |
| marginals_only_residual | 1130973 | 6944205 | 2096981 | 0 | 4825553 | 3689 | 17982 |
| trot_q0p5_residual | 1292070 | 6945964 | 2096981 | 0 | 4827329 | 3672 | 17982 |
| trot_q0p8_residual | 1316311 | 6946274 | 2096981 | 0 | 4827639 | 3672 | 17982 |
| trot_q1p0_residual | 1316436 | 6944455 | 2096981 | 0 | 4825820 | 3672 | 17982 |
| trot_q1p2_residual | 1322174 | 6942164 | 2096981 | 0 | 4823529 | 3672 | 17982 |
| trot_q2p0_residual | 1336591 | 6938710 | 2096981 | 0 | 4820075 | 3672 | 17982 |

## Boundaries

- These are byte-only scorer-free measurements over cached argmax labels.
- The q-family solve is generic decode-time algorithm; counted payload includes marginals, residuals, tags, and framing.
- No video-derived side-information matrix was hidden in code or omitted from bytes; side-info bytes are zero because the cost is derived only from counted marginals.
- No RGB receiver, archive parse-back, SegNet/PoseNet scorer survival, or score improvement is claimed.
- Negative verdict scope is the pre-registered FORMULATION only, not the TROT family outside this CR1 selected-support payload.

## Follow-On Disposition

| item | disposition | fire order |
|---|---|---|
| TR2-P1 | FIRED | This receipt is the pre-registered byte-only q-family residual race against CR1's incumbent. |
| TR2-P1-implementation-reference | FIRED | The local q-family solver passed deterministic fixtures before the CR1 payload race; no unpinned dependency was vendored. |
| #984 rate axis / CR1 successor consumer | QUEUED-WITH-FIRE-ORDER | Consume this row only as a scorer-free byte-race negative unless a future arm supplies a new counted side-info source or residual model and repeats same-payload decode-equality racing. |

## Frontier Honesty

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`. Contest pointer remains `borrowed/unmoved 0.1910828242 [contest-CPU]`.
