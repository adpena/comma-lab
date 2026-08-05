# OD8 native-DOF receipt - 2026-08-05

Status: `SCORER_FREE_NATIVE_DOF_PRICED / OD6_FORMAT_GAP_SELF_CONTAINED / NO FRONTIER MOVE`.

Axis: `[macOS-CPU cache-derived advisory / scorer-free byte pricing]`.
`score_claim=false`, `promotion_eligible=false`, `scorer_forwards_run=0`, `upstream_evaluate=false`.

## Answer First

OD8 identifies JS1 Stage-1 `cprime`'s native payload as sparse scorer-lattice frame_1 paint: the solved object is RGB `uint8` paint values on the block16 target band, plus the support needed for a decoder to apply those values.  On OD2's recorded n32 pair set, the actual recomputed support is `18734` scorer pixels (`585.44`/pair).  The optimized value DOF alone is `56202` u8 values at n32; receiver-applicable sparse support+values is `112404` raw n32 B under a fixed u24 flat-index accounting, projected `2107575` raw n600 B before entropy coding.

Stage-2 `cheapdct4` is confirmed from code and OD2 records as `k*k*3` int16 coefficients per pair.  For `k=4` that is `48` coefficients, `96` B/pair, and `57600` raw B at n600.

The best OD8 scorer-free native stream estimate is `base_rgb_proxy`: `66432` exact n32 coded B, linearly projected `1245600` B at n600.  That is `+1155600` B vs the top of GC18's 45-90K conjectured boundary-grammar floor and `+1169296` B vs OD6's 76,304 B packet projection.  This is an ESTIMATE, not a score: OD2/OD7 did not store actual solved paint or DCT coefficients.

OD8 fixes the OD6 b1024 format gap by producing a self-contained sibling packet at `/Volumes/VertigoDataTier/pact/ddm_od8_20260805/run_20260805T_od8_codex_r4/packets/od8_od6_b1024_self_contained.od5.raw_packet`.  The table qlogits are unchanged, and the packet now carries the sections required by its coverage hash: `pe1_generator_coords_n32`, `pe3_hybrid75_coords_n32`, and `pe3_hybrid_knee_coords_n32`.  Exact n32 best-coded bytes rise by `+3214` B vs the OD6 incumbent.  A conservative component-sum n600 projection is `247169` B (`+170865` B vs 76,304 B).

## Native Price Table

| stream | exact n32 best B | projected n600 B | vs 45K | vs 90K | vs OD6 76,304 B | best coder |
|---|---:|---:|---:|---:|---:|---|
| base_rgb_proxy | 66432 | 1245600 | +1200600 | +1155600 | +1169296 | brotli-q11 |
| deterministic_noise_proxy | 72070 | 1351313 | +1306313 | +1261313 | +1275009 | brotli-q11 |

## DOF Identification

| surface | object | dimensionality | quantization | receiver note |
|---|---|---:|---|---|
| Stage-1 `cprime` optimized values | RGB paint on frame_1 scorer-lattice support | `3 * band_px` values (`56202` on OD2 n32) | uint8 selected through rounded proxy flips | support+values must be shipped for decoder application; offsets alone are not enough without scorer/GT argmax |
| Stage-1 receiver support | scorer-lattice flat indices for the band | `band_px` indices (`18734` on OD2 n32) | sparse varint in OD8 packet; u24 raw accounting in table above | applies via `realize_scorer_paint_to_camera`, not stamps |
| Stage-2 `cheapdct4` | frame_0 low-frequency DCT coefficients | `4*4*3 = 48` coefficients/pair | int16 | basis is generic/free; coefficient values are counted |

## OD6 Format Gap Fix

| quantity | value |
|---|---:|
| source OD6 incumbent exact n32 best B | 7334 |
| OD8 self-contained exact n32 best B | 10548 |
| exact n32 delta | +3214 |
| OD6 incumbent projected n600 B | 76304 |
| OD8 conservative component-sum projected n600 B | 247169 |
| conservative projected delta | +170865 |

Parse-back proof: OD5 packet reserializes exactly; OD6 table qlogits parse back unchanged; `coverage_source_sections` maps all three coverage columns to shipped sections.

## Determinism Check

No scorer was run.  OD8 recomputed the deterministic cprime target/band from cached `cx1_argmax_n600.npy` and `gt_argmax_n600.npy` for all `32` OD2 pairs.  `flips_before`, `band_px`, and `n_described` match OD2 recorded rows exactly.  OD8 also verified every recorded `cheapdct4` row carries `96` B/pair and `value_measured_through_int16_quantiser=true`.

## RECALL EVIDENCE

| source/search | recalled fact | plan impact |
|---|---|---|
| `.omx/tmp/codex_runs/od8_prompt.md`, `_common_contract.md` | OD8 is scorer-free, must not touch JS1 running harness or od3 run dirs, and must use SSD for bulk. | Added a sibling tool only; ran only cache/byte pricing; wrote bulk packets under `/Volumes/VertigoDataTier/pact/ddm_od8_20260805/`. |
| `.omx/state/main_hot_state.md` | OD3 owns scorer slot; OD8 must persist native DOF and fix OD6 self-containment. | Kept the actual scorer re-derive in the post-OD3 fire order. |
| `OD7_RECEIVER_CLOSE_RECEIPT.md` | Stamp realization is formulation-dead; OD2/OD3 discard terminal fields; OD6 b1024 packet lacks coverage feature homes. | Priced native support+paint payloads and added missing OD6 coverage sections instead of stamping. |
| `OD2_STAGE12_RECEIPT.md` and OD2 JSON | Stage-1 outcomes are recorded but solved params are absent; k=4 carriage is 96 B/pair. | Treated native price as DERIVED/ESTIMATE until post-OD3 re-derive stores real values. |
| `OD6_DECODER_LEGAL_RECEIPT.md` and source packet | Incumbent b1024 exact n32 best is 7,334 B; projected n600 is 76,304 B; table uses the missing coverage features. | Produced a self-contained sibling packet and priced the delta. |
| `GC18_CONVOCATION_RECEIPT.md` | Boundary-grammar floor conjecture is 45-90K; OD6-style packet is the comparison point. | Reported native estimates directly against both bars. |
| canonical equations search (`receiver`, `payload`, `format`, `coefficient`, `counted`) | Receiver support, format-vs-search, and decoder-derived-context equations all require counted receiver-visible payloads. | Kept support bytes explicit and refused score/promotion wording. |
| bounded `rg` over `.omx/research`, `.omx/state`, `docs`, `experiments`, `src/tac`, `tools` for OD8/native/cprime/cheapdct/OD6 gap terms | Found no prior OD8 receipt beyond the live hot-state/charter; found OD6/OD7 gap statements and JS1 code. | Built the first OD8 receipt and scoped absence to the searched surfaces. |

## SHA Table

| artifact | bytes | sha256 |
|---|---:|---|
| `/Users/adpena/Projects/pact/experiments/ddm_od8_js1_persist.py` | 63042 | `21cb31a32449a3b7906653eb0ec92af2913fe78ad91d26157d7a534a2bb1be63` |
| `/Users/adpena/Projects/pact/.omx/research/ddm_od2_20260805/od2_js1_n32_cprime_k4.json` | 103690 | `fd1016751e4668ff786692f52f91d924be97081a70a20d11e470150aaf85c6af` |
| `/Users/adpena/Projects/pact/.omx/research/ddm_od2_20260805/PAIR_SELECTION.json` | 2388 | `0a8ac26a1cd39c7dc425dbb4922d0dda6f71227b205241d3d771ea9791c2d4f9` |
| `/Users/adpena/Projects/pact/.omx/research/ddm_od6_20260805/ddm_od6_decoder_legal_receipt.json` | 550769 | `4775cb3fc8925de33a70945ba1469b962f0934b5527a032e39cb60e0589566b1` |
| `/Volumes/VertigoDataTier/pact/ddm_od6_20260805/run_20260805T030410Z/packets/base_rgb_generator_geometry_b1024.od5.raw_packet` | 12577 | `52bd534cea34578bb8d76fb079befa951d26f914347bd51b7078c398e2ff6e2a` |
| `/Users/adpena/Projects/pact/.omx/research/ddm_od8_20260805/od8_native_dof_receipt.json` | 28375 | `753189ea30a53561704e28da8b7ff8cded0015de88afe91c675e5f34fd8d55d5` |
| `/Volumes/VertigoDataTier/pact/ddm_od8_20260805/run_20260805T_od8_codex_r4/packets/od8_native_dof_base_rgb_proxy.od5.raw_packet` | 83466 | `17ff7e478f53d711d76fdb558e243224e1b6613769dc06cb211a3ed6cbcd5f35` |
| `/Volumes/VertigoDataTier/pact/ddm_od8_20260805/run_20260805T_od8_codex_r4/packets/od8_native_dof_deterministic_noise_proxy.od5.raw_packet` | 83499 | `e0db0f5d0282b57ba67ec8924fa321c59e28144c59c3c7179b91f3055522b708` |
| `/Volumes/VertigoDataTier/pact/ddm_od8_20260805/run_20260805T_od8_codex_r4/packets/od8_od6_b1024_self_contained.od5.raw_packet` | 34444 | `dce4efdd6a06ab0846b2da538162eef27e4828cb18bf9c1eb37104614fe93d95` |

## NEXT_IF_RESUMED

1. Wait until OD3 releases the scorer slot and its terminality receipt is stable.  Do not read or mutate OD3 run dirs while it is active.
2. If OD3's terminal artifact contains native payload fields, pass them as warm-start/custody inputs; if it contains only outcome JSON, re-run the same pair set with OD8 persistence because there are no coefficients to warm-start.
3. Fire:

```bash
.venv/bin/python experiments/ddm_od8_js1_persist.py solve-persist \
  --allow-scorer \
  --sub-dir /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2 \
  --gt-mkv upstream/videos/0.mkv \
  --pairs-npy .omx/research/ddm_od2_20260805/od2_pairs_n32_seed20260805.npy \
  --argmax-cache /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache \
  --out .omx/research/ddm_od8_20260805/od8_js1_persist_n32_cprime_k4.json \
  --packet-out /Volumes/VertigoDataTier/pact/ddm_od8_20260805/native_dof/od8_native_dof_n32_cprime_k4.od5.raw_packet \
  --block 16 --rmax 5 --seg-steps 100 --pose-steps 40 --eval-every 5 \
  --dct-k 4 --threads 4 --resume
```

4. Price the persisted native packet:

```bash
.venv/bin/python experiments/ddm_od8_js1_persist.py price-persisted \
  --persist-json .omx/research/ddm_od8_20260805/od8_js1_persist_n32_cprime_k4.json \
  --packet /Volumes/VertigoDataTier/pact/ddm_od8_20260805/native_dof/od8_native_dof_n32_cprime_k4.od5.raw_packet \
  --out .omx/research/ddm_od8_20260805/od8_persisted_native_dof_price.json
```

5. Replace ESTIMATE native bytes with MEASURED real-solved bytes, then stage the receiver candidate whose realization is coefficient/support application.

## Boundaries

- No SegNet/PoseNet forward, `upstream/evaluate.py`, contest-CPU, contest-CUDA, MPS, or n600 scorer job was run by OD8.
- Native price rows use actual OD2 support and base/noise proxy values because OD2/OD7 do not store solved paint or coefficients.
- OD6 self-contained packet is a format repair and byte price, not receiver-closed RGB/inflate/scorer survival.
- The frontier did not move.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
