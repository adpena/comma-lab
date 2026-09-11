**Charter partially executed.** Census and lossless pricing are complete; HPAC pruning with tail re-encoding and renderer precision cuts with re-solve remain unmeasured.

[Full memo and evidence](/Users/adpena/Projects/pact/.omx/research/ddm_ntb1_non_tail_bytes_price_and_resolve_20260911.md)

| Non-tail component | Bytes |
|---|---:|
| ZIP framing | 100 |
| RX1M header | 14 |
| HPAC | 11,911 |
| Renderer | 29,862 |
| Carrier | 18,610 |
| **Total** | **60,497** |

**13 formats, 26 retained twin encodes; no savings.** Shipping-parser decoded sections remained identical.

| Lever | Δbytes |
|---|---:|
| ZIP stored / deflate / bzip2 / LZMA | 0 / +58 / +1,214 / +2,492 |
| HPAC Brotli q11 / q10 / XZ | +22 / 0 / +357 |
| Renderer Brotli q11 / q10 / plane toggle | +91 / +1 / +591 |
| Carrier Brotli q11 / q10 / plane toggle | +7 / +2 / +16 |

These are **[macOS-CPU advisory; scorer-free byte measurements]**. Zero distortion deltas are derived from identical decoded inputs, not newly scored. The exact bar requires **31 saved bytes**.

No composed candidate, projected candidate score, or seal was produced. The common contract assigns no scorer slot to this arm. HPAC pruning additionally needs an RLC1-aware encoder; that implementation remains owed.

Serializer rc 17 preserved verified bundle commit `e5a4b5779`; all 13 intended files verified. **Not landed on main.** [Landing receipt](/Users/adpena/Projects/pact/.omx/research/ddm_ntb1_20260911/LANDING.json)

composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44) unchanged.

## NEXT_IF_RESUMED

- **QUEUED — MAIN:** land the verified bundle; consumer: Pact main branch; trigger: harvest of the landing receipt.
- **QUEUED — ddm_ntb1:** implement and price HPAC pruning jointly with its tail; consumer: `ddm_ntb1_non_tail_bytes/hpac_joint` on Vertigo; trigger: next continuation after reproducing the current stream with the RLC1-aware encoder.
- **QUEUED — ddm_ntb1:** renderer precision cuts and real re-solve; consumer: `ddm_ntb1_non_tail_bytes/renderer_resolve`; trigger: MAIN assigns the sole scorer slot.
- **QUEUED — ddm_ntb1:** compose and cold-verify winners; consumer: `ddm_ntb1_non_tail_bytes/composed`; trigger: a measured joint treatment clears the bar. MAIN owns exact dispatch.

## LIVE-HYPOTHESES

- Structural HPAC pruning may pay: prior multiplier/seed negatives did not test those coordinates.
- Remaining 4-bit renderer layers may pay after re-solve: the wire already supports mixed precision, but recovery and pose cost are unmeasured.

## DEAD-ENDS

- These 13 lossless formats on move 44: best saving **0 B**.
- ZIP hygiene: one required `p` member; extras, comments and padding already absent.
- Unchanged JG2 encoder reuse: omits move 44’s counted RLC1 mixer path.
- Fresh savings from frame-embedding/block-0 FiLM 4→3-bit cuts: those tensors are already 3-bit.
<!-- # FORMALIZATION_PENDING: measurement memo; no score row -->
