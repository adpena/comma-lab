# ddm_fo1 RECEIPT — real coder on sr1's waterfilled seg-correction support

`[macOS-CPU advisory]` · scorer-free · $0 · `score_claim=false` · `promotable=false`
Tool: `experiments/ddm_fo1_waterfill_real_coder.py` · n600 · wall 45.4 s (selection) + 26.7 s (race)

## VERDICT — the channel CLEARS the pre-registered bar

| quantity | bytes |
|---|---:|
| pre-registered bar (FROZEN) | **5,066** |
| **real coded total, pre-registered M0–M7 only** | **4,310** |
| **real coded total, incl. the added M8** | **4,308** |
| sr1's IDEAL entropy limit | 4,276.17 |
| margin under the bar | **756 B (14.9%)** |
| real / ideal | **1.0074** (0.74% above ideal, against 18.47% of headroom) |

**REAL SUPPLIER.** sr1's waterfilled channel survives a real, round-trip-verified coder. The
verdict does not depend on my one added coder: the pre-registered M0–M7 set alone clears the bar
by 756 B.

Bar provenance (pn2 §5/§6, frozen, not recomputed here): `η_projected_pooled_n12 = 0.6111 ×
6,512 flips × 1.273108 B/flip = 5,065.6 B`. My charter quoted η 0.6224; pn2's final n=12 memo
carries 0.6111 and that is the η the 5,066 B figure is built from. The bar value is unchanged.

## Selection reconstruction — EXACT against sr1's retained payloads

Rebuilt from the transmitted labels alone using sr1's `cell_definition`
(own class × lowest differing 4-neighbour class × min(degree,4) × row band, 1,200 cells).

| control | result |
|---|---|
| `cell_band_px` vs sr1 retained | **byte-identical**, sha `739d33c27e739416…` (sum 2,551,464) |
| `cell_flip_px` vs sr1 retained | **byte-identical**, sha `2d0b71464d3901e3…` (sum 34,666) |
| target-class rate | `0.22530701479359683` bits/flip — identical to sr1 to 17 decimals |
| cells selected | **41** (sr1: 41) |
| flips described | **6,512** (sr1: 6,512) |
| ideal total | **4,276.171156196116 B** (sr1: 4,276.171156196069 B; rel. diff 1.1e-14) |
| rt1 `free_band_mask` == boundary(labels) | verified every frame, fail-closed |
| rt1 `flip_mask_vs_gt` == (argmax_base != gt) | verified every frame, fail-closed |

Support geometry: **94,124** band pixels (3.689% of the 2,551,464-pixel band) carrying **6,512**
of 34,666 band flips (18.785%). Density **6.9185%** vs the band's 1.3587% — **5.09× denser**.

## Receiver-derivability of the support

Every **per-pixel** cell factor is a deterministic function of the decoded label field, so
evaluating membership costs zero archive bytes. The premise holds. The one item that is **not**
receiver-derivable is the **set** of 41 cell ids — it was chosen using GT-derived flip densities,
so it is this-clip side information and is COUNTED. sr1 excluded it and the bar comparison above
excludes it too; priced, it is **52.4 B** (41 × log2(1200) index list) or 150 B (1,200-cell
bitmap). At the cheaper price the total is **4,360 B — still under the bar.**

## Mask coder race — 9 coders, every payload decoded back through the same context machine

| tag | bytes | bits/described flip | pre-registered | round-trip | coder |
|---|---:|---:|:--:|:--:|---|
| M8 | **4,123** | 5.065 | no (added) | ✅ | CABAC full-band-walk order, support-only symbols (pair × run × temporal) |
| M7 | **4,125** | 5.068 | yes | ✅ | CABAC support-walk (pair × run × temporal), 88 contexts |
| M5 | 4,140 | 5.086 | yes | ✅ | CABAC raster (pair × causal-neighbours × temporal), 88 contexts |
| M6 | 4,230 | 5.197 | yes | ✅ | CABAC support-walk (run × temporal), 8 contexts |
| M3 | 4,274 | 5.251 | yes | ✅ | static binary AC (i.i.d. realized) |
| M4 | 4,281 | 5.259 | yes | ✅ | adaptive binary AC, order-0 |
| M1 | 4,354 | 5.349 | yes | ✅ | brotli(packed) q11 |
| M2 | 5,288 | 6.496 | yes | ✅ | lzma(packed) preset 9\|EXTREME |
| M0 | 11,766 | 14.455 | yes | ✅ | raw packed support bits (no coder) |

Ideal **mask-only** entropy is 4,092.77 B; M7 is +0.79%, M8 +0.74%.

## Target-class coder race — 3 coders, all round-trip verified

| tag | bytes | bits/flip | round-trip | coder |
|---|---:|---:|:--:|---|
| T2 | **185** | 0.227 | ✅ | adaptive binary-tree AC, context = (own, partner) — both label-derived, free |
| T1 | 1,386 | 1.703 | ✅ | adaptive binary-tree AC, order-0 |
| T0 | 2,442 | 3.000 | ✅ | raw 3 bits/flip packed |

Ideal target term is 183.40 B; T2 is +0.87%. No model is transmitted — the coder learns online.

## Joint arithmetic on the real bytes (4,308 B, 6,512 flips)

| η | net ΔS, real coder | net ΔS, sr1 ideal |
|---|---:|---:|
| 0.6235 (sr1 selection η) | **−0.000573** | −0.000595 |
| 0.6111 (pn2 projected, the bar's η) | **−0.000505** | −0.000526 |
| 0.5651 (pn2 **unprojected**) | **−0.000251** | — |
| 1.0000 | −0.002652 | — |

Break-even η on the real bytes = **0.5196**, below pn2's unprojected pooled η (0.5651) as well as
its projected one. 96.4% of sr1's headline ΔS survives the real coder.
Share of the 0.0095973 S gap to 0.15, at η = 0.6111: **5.26%**.

## Determinism + payload custody

**Three independent n600 runs produced byte-identical coder payloads** (original, a determinism
repeat into a separate directory, and a re-run after a code tightening). Repeat receipts kept as
`FO1_*.repeat.json`; the duplicate blobs were certified-rebuildable by recorded sha and removed.
Only the two receipt JSONs differ between runs, in `wall_s`.

The round-trip proof is auditable from disk without trusting a flag:
`roundtrip_decoded_mask_best.npy` and `restricted_mask_bits.npy` share sha
`8bca66ec89eb830a2c3ee0fe4df55149fb1652d4127529d313e0b043f0b7bce5`. That file is sourced only
from a coder that produced it by DECODING (`decoded_by: M8`), never from the truth field — the
harness refuses to seed it from the non-arithmetic coders, which would make the artifact circular.

| artifact | bytes | sha256 |
|---|---:|---|
| `retained/mask_M0.bin` | 11,766 | `db74d7f662a6078b58d18499ffa46f789fc2c3929edc0288f86f110c677344c0` |
| `retained/mask_M1.bin` | 4,354 | `5fd08ff0687ce1d1c27bf9e481dd480cc6f70f2308a099db5a11f7b40d80ea1a` |
| `retained/mask_M2.bin` | 5,288 | `aa1a1bb3365b1e30b19c8bb46c85d849539407fb69f97949910b1351545c03f4` |
| `retained/mask_M3.bin` | 4,274 | `1d941093700c05cfe0179928cb03b96c655a8aca37d664d0578d417e8e1c9944` |
| `retained/mask_M4.bin` | 4,281 | `2b100eedbfb9db96a2ad89831fda547bd37279952d62f1b108ad00b76e2b39ca` |
| `retained/mask_M5.bin` | 4,140 | `4ffadb403465c5050c1fa9dfa1a181963861acdccc83d43f69c34e06830f9c36` |
| `retained/mask_M6.bin` | 4,230 | `fb3de125be399be3478c985c5a011c503910510a77b704166228ec331be378e9` |
| `retained/mask_M7.bin` | 4,125 | `e1822d95c662ac4a1853a7ff6b7d7442de453b6ed701a54ee8786dae4893799e` |
| `retained/mask_M8.bin` | 4,123 | `22759c6d4aba2dac3bbb4eb8e8ff5c2bde6bfc2b77ca32ab8b69561da9d9a26e` |
| `retained/target_T0.bin` | 2,442 | `dc012bf2c6222e8f1e0fb1abd6a293fa033e3cb0a26ab353a43b5df7dd158685` |
| `retained/target_T1.bin` | 1,386 | `c8b7086a20502cea8426f6221ed5b0b37f34391d6bc0fca232746966a2be41e7` |
| `retained/target_T2.bin` | 185 | `49c91c72d9dca674861fb59ae0cac8672cdc676499aa9ddb6c7a3590aeb1c52f` |
| `retained/restricted_mask_bits.npy` | 94,252 | `8bca66ec89eb830a2c3ee0fe4df55149fb1652d4127529d313e0b043f0b7bce5` |
| `retained/roundtrip_decoded_mask_best.npy` | 94,252 | `8bca66ec89eb830a2c3ee0fe4df55149fb1652d4127529d313e0b043f0b7bce5` |
| `retained/restricted_support_index.npy` | 376,624 | `feb59a95c9296b38ae926d5cc10dba701648ae967f170d481051d7360a820392` |
| `retained/restricted_support_frame_offsets.npy` | 4,936 | `542cd33174da6e410fa7f39c817d1dfa4022d74cf1bcbbe80f4759949559f6b2` |
| `retained/target_values.npy` | 6,640 | `cf81248de9c7fa066775be11c1c6e37ca6965fbb0cb81f63c2fb02c77e7fe85d` |
| `retained/selected_cells.npy` | 456 | `d3c7f9c32d4c02a4c900e0bb37b45efafc6c08ba99c85f52e69902e3a9b21969` |
| `retained/cell_band_px.npy` | 9,728 | `739d33c27e739416177b9a89dde939b492e5735ce86ee9d63c0e57ef34b8ab9f` |
| `retained/cell_flip_px.npy` | 9,728 | `2d0b71464d3901e3364b45fad9d4626ad58b047db8e4087f1dfad1097281ae35` |
| `retained/tgt_counts.npy` | 1,328 | `3785a53a8ed1f96360820210c93deee91fe1dc3f7ee10373306076e89e3ab381` |
| `FO1_SELECTION.json` | 3,519 | `f32665cd2fc4b0b8c98def50b396c054db0e1c8021f7bcdf61bd42675946b09e` |
| `FO1_CODER_RACE.json` | 11,900 | `4c651c24dd3a1cc0006ec6e5611a8ff3a7e6796c6187c55c2e6382579966ed9b` |
| `FO1_SELECTION.repeat.json` | 3,464 | `89d16df4ca86650435d0384e7f0642d76a2471a4080b49c2f67e821c8c86d7fc` |
| `FO1_CODER_RACE.repeat.json` | 11,638 | `8ef873a42f3dc0be1aa6af8f4f4d6ba3dc11e596afd8358ca8d8a1887ba6418c` |

`cell_band_px.npy` / `cell_flip_px.npy` carry sr1's own shas because the reconstruction is
byte-identical — that identity is the control, not a copy.

## Consumed, unmodified

rt1 `/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816/{argmax_base,flip_mask_vs_gt,free_band_mask,flip_target_class}.npy` ·
sr1 `/Volumes/APDataStore/pact/ddm_sr1_manufactured_seg_recovery_20260816/{cell_band_px,cell_flip_px}.npy` + `SR1_WATERFILL.json` ·
hv1 `…/ep0634/retained/coders/s1p25_c1p0/decoded_spatial_tokens.rc64.bin` ·
qs3 `/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy` (read-only).
`upstream/` was never read or written by this arm.

## verdict_scope

**formulation** — the M0–M8 coder family on sr1's 41-cell waterfilled support of rt1's free label
boundary at n600. A better coder could lower 4,308 B further; nothing here bounds the family, and
nothing here is a score.
