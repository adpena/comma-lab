# CR1 GDL1 coder races - 2026-08-08

Tags: [no-triality] [p0-ledger-ok]

## Answer First

No scorer, no evaluator, no Metal, no GPU, no paid job, and no archive promotion ran.
Both races use cached real n600 argmax payloads and real coder round-trips.

| race | baseline best | treatment best | delta | verdict |
|---|---:|---:|---:|---|
| phase-coset stride-2 | 265906 B (lzma1-raw) | 367743 B (brotli-q11) | 101837 B (38.298%) | LOSS-w/-bytes |
| edge-graph conditional carrier | 575095 B (lzma1-raw) | 464557 B (lzma1-raw) | -110538 B (-19.221%) | WIN-w/-bytes |

## Inputs

- GT argmax: `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/gt_argmax_n600.npy` (b74a14b226a5aceb5824899898bcb06e5413c54b7db2441268da7bc91a10db5d)
- Current argmax: `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_argmax_n600.npy` (5e903de650e60ec6a64b34eb455fa1bc911223551d0b31e9ae45cc906e1490be)
- Axis: `[byte-only scorer-free]`.
- Selection: `n600_all_pairs_no_prefix`.

## Recall Evidence

| source or query | result | impact |
|---|---|---|
| .omx/research/ddm_gdl1_20260807/GDL1_CROSSWALK.md rows 1-2 | Found the two queued ADOPT-CLASS probes and their falsifiers. | Executed the registered phase-coset and edge-conditioned byte races instead of redesigning them. |
| .omx/research/ddm_rl1_roadlane_interface_price_20260803.md | RL1 supplies the non-prefix n32 Road<->Lane price line and W=1.273108 B/flip denominator. | Phase race reports bytes per settled Road<->Lane flip and waterline status. |
| .omx/research/ddm_sx1_separatrix_carrier_20260803.md | SX1 supplies full-population separatrix edge denominators and Road<->Lane hub evidence. | Edge race uses n600 edge-labeled support instead of a prefix or class-row proxy. |
| experiments/ddm_bd1_class_field_receiver.py and experiments/ddm_pe1_per_edge_partition_race.py | Existing BD1/PE1 coder primitives already provide Brotli q11, raw LZMA1, and SMEVR with decode checks. | Reused landed coder surfaces; did not introduce a synthetic or unvalidated codec. |
| content search: phase_coset, edge_graph, Road<->Lane, SMEVR, #920, #984 over .omx/research and experiments | Found PE1/ST1/SM2/BD1 precedents beyond the charter seeds; no finished CR1 race artifact existed. | Built a narrow new CR1 measurement and kept scorer consumers queued. |
| tools/list_canonical_equations.py --json | Canonical registry was consulted for relevant byte/coder equations; no CR1-specific equation superseded the charter. | Receipt remains a scorer-free measured byte race, not a score or equation promotion. |

## Race 1 - Phase-Coset Stride-2

Road<->Lane support pixels: `1143497`; cx1 direct Road<->Lane flips: `235148`; RL1 settled flips: `235148`.
Treatment best bytes per RL1 settled flip: `1.563879` vs W `1.273108`.
Decode equality: flat and phase-coset records both decode exactly to the same Road<->Lane n600 support arrays.

Coder rows:

| side | codec | bytes | sha256 | artifact |
|---|---|---:|---|---|
| baseline | lzma1-raw | 265906 | `5b1684df6acc235f78de5437e4c2bfe6973391d95679f4cfeab46d3e7fc93a16` | /Volumes/VertigoDataTier/pact/ddm_cr1_20260808/payloads/p1_flat_road_lane_support.lzma1-raw.bin |
| baseline | brotli-q11 | 273416 | `f39daab508d68685cb2de278882846a92515e38648a92abbf6f6d26bb802f27f` |  |
| baseline | zlib-9 | 319074 | `3e434229fb5d6cb2865622dc1d25183f9633a68bc808b7d45463e553217a11c9` |  |
| baseline | smevr-r7-nibble | 378812 | `b619482d7c9a47f487b9b4298a584b3b045e07ddd9242740610dbf65689c8d54` |  |
| phase | brotli-q11 | 367743 | `5e98dc107efcb87cb957ac467e3d94dafe53c42b5c8a8c2da51cac47ae1c7638` | /Volumes/VertigoDataTier/pact/ddm_cr1_20260808/payloads/p1_phase_coset_road_lane_support.brotli-q11.bin |
| phase | lzma1-raw | 369274 | `a4a9c9db3601bb19784091f976af5ab68cb9780541ed3ab55f388c8157cdcf7a` |  |
| phase | zlib-9 | 441920 | `c0f42aa835447cdf01cffebc47835efe9a7cf4faf776fa6d0b325e2a1f2a7d4d` |  |
| phase | smevr-r7-nibble | 567734 | `619f7f0b1fa0f917da568cb8afbbc1cf4e4ee1b4cba44782f8a21addd447bf47` |  |

## Race 2 - Edge-Graph Conditional Carrier

Selected support pixels: `2554360`; selected cx1 direct flips: `506837`.
Decode equality: pooled and edge-conditioned records both decode exactly to the same edge-labeled n600 support arrays.

Selected edge denominators:

| edge | support px | support share | cx1 direct flips | flip share |
|---|---:|---:|---:|---:|
| Road<->Lane | 1143497 | 0.447665 | 235148 | 0.463952 |
| Road<->MyCar | 606290 | 0.237355 | 63027 | 0.124354 |
| Road<->Undriv | 511976 | 0.200432 | 89545 | 0.176674 |
| Undriv<->Movable | 150302 | 0.058841 | 61892 | 0.122114 |
| Road<->Movable | 142295 | 0.055707 | 57225 | 0.112906 |

Coder rows:

| side | codec | bytes | sha256 | artifact |
|---|---|---:|---|---|
| pooled | lzma1-raw | 575095 | `e0b4d7860bf4fc6baf3be290cafaaf3d9fe712f44fbaabf09dfa9dca39438104` | /Volumes/VertigoDataTier/pact/ddm_cr1_20260808/payloads/p2_pooled_edge_blind_support.lzma1-raw.bin |
| pooled | brotli-q11 | 577243 | `8d3d9065de5dc96436b3123d660e40e69a655930e294fcf6e067d639ad9ad5b2` |  |
| pooled | zlib-9 | 685445 | `1dd2ebe35e35a7dac3cbb0dc9d7edf106ca7fc8f35a7153933f722770bd4b475` |  |
| pooled | smevr-r7-nibble | 1460476 | `ef951f8c603b79eff549b96dcb2a034df73801bc4ee66831de5a291aae414ed9` |  |
| edge-conditioned | lzma1-raw | 464557 | `0a53f649768c61912399ccab14e4d3323998e47235992091e2a9e28cf7259fe1` | /Volumes/VertigoDataTier/pact/ddm_cr1_20260808/payloads/p2_edge_conditioned_support.lzma1-raw.bin |
| edge-conditioned | brotli-q11 | 476117 | `addb7e7758db654bc65e05d1e8aaae80e718b8e5289633f9f31c70a302fa07a6` |  |
| edge-conditioned | zlib-9 | 555651 | `dfc8ea67a3c6fb216cf8a7f836a6867bfdd256089b938cc3627b04ccda5d128b` |  |
| edge-conditioned | smevr-r7-nibble | 669851 | `45c662edd2298d9b8481d85bf37a9f12e65a1775bfda04360889e061a640aaad` |  |

## Boundaries

- These are byte-only scorer-free measurements over cached argmax labels.
- Support denominators are 4-neighbor endpoint pixels, not separatrix crack lengths.
- No RGB receiver, archive parse-back, or n600 scorer survival is claimed.
- Negative verdicts are formulation-scoped only, exactly as pre-registered.
- Follow-ons named here exit this run as FIRED by these CR1 measurements; any scorer consumer remains queued behind a future owner.

## Follow-On Disposition

| item | disposition | fire order |
|---|---|---|
| GDL1-P1 | FIRED | This receipt is the scorer-free phase-coset byte race; any receiver/scorer consumer must claim a future lane separately. |
| GDL1-P2 | FIRED | This receipt is the scorer-free edge-conditioned byte race; #984/ty1 may consume only the measured rows and must not infer RGB survival. |

## Frontier Honesty

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved at `0.1910828242 [contest-CPU]`.
