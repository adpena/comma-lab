# OD9 receipt - 2026-08-05

## ANSWER FIRST
Stage-1 does **not** price inside the GC18 45-90 KB corridor. On the OD2/OD3 n32 pair set, the best real persisted Stage-1 packet is `stage1_only_absolute_u8` at `64747` exact coded B, projected `1214007` B at n600 by linear scaling. That is `+1124007` B over the 90 KB corridor top, `+1137703` B over OD6, and `+1139599` B over PE3.

The delta-entropy fork falls **LARGE_DELTA_ENTROPY_SHIP_THE_SOLVE_RATE_DEAD**. The best combined Stage-1 plus k=4 carriage packet is `combined_stage1_absolute_u8_plus_stage2_cheapdct4` at `66785` exact coded B on n32, projected `1252219` B at n600. Base-relative coding did not rescue it: `base_delta_mod_u8` projects `1262157` B, worse than absolute. This is a byte-only representation verdict, not an archive or score.

Verdict scope: **FORMULATION** - flat sparse solved-paint support plus absolute/base-delta RGB values, with cheapdct4 qcoeff carriage. It does not kill shared carrier, CPWL/task-description, or learned TR1 families. It does close the ship-the-solve OD-line as rate-dead at this representation.

## DENOMINATORS AND AXES
- Solve axis: `[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE`.
- Pricing axis: `[macOS-CPU byte-only persisted-native pricing]`.
- Selection: `OD2 n32 seed20260805 stratified pair set inherited by OD3/OD9`.
- Denominator: 32 measured pairs projected linearly to 600 pairs for byte comparisons only.
- No `upstream/evaluate.py`, n600 archive, contest-CPU, contest-CUDA, or MPS authority was run by OD9.
- Score claim: `false`; promotion eligible: `false`.

## LEG A - STAGE-1 REPRESENTATION PRICING
| variant | coder | n32 coded B | projected n600 B | vs 90KB corridor | OD5 parseback |
| --- | --- | --- | --- | --- | --- |
| absolute_u8 | brotli-q11 | 64747 | 1214007 | 1124007 | True |
| base_delta_mod_u8 | brotli-q11 | 65232 | 1223100 | 1133100 | True |
| base_delta_zigzag_varint | brotli-q11 | 66452 | 1245975 | 1155975 | True |
| base_delta_i16_raw | brotli-q11 | 72302 | 1355663 | 1265663 | True |

Measured support: `18734` scorer-lattice pixels, `56202` raw RGB u8 values, and `112404` raw i16 delta bytes across n32.

## LEG B - PERSISTENCE AND DELTA ENTROPY
| packet | n32 coded B | projected n600 B | vs OD6 | vs PE3 | vs 90KB corridor |
| --- | --- | --- | --- | --- | --- |
| combined_stage1_absolute_u8_plus_stage2_cheapdct4 | 66785 | 1252219 | 1175915 | 1177811 | 1162219 |
| combined_stage1_base_delta_mod_u8_plus_stage2_cheapdct4 | 67315 | 1262157 | 1185853 | 1187749 | 1172157 |
| combined_stage1_base_delta_zigzag_varint_plus_stage2_cheapdct4 | 68409 | 1282669 | 1206365 | 1208261 | 1192669 |
| combined_stage1_base_delta_i16_raw_plus_stage2_cheapdct4 | 72271 | 1355082 | 1278778 | 1280674 | 1265082 |
| stage2_only_cheapdct4_qcoeffs | 2157 | 40444 | -35860 | -33964 | -49556 |

Delta stats against the decoded qo1 base on the same support: channels `56202`, min `-252`, max `239`, mean abs `38.223142`, median abs `30.0`, zero fraction `0.008612`, within +/-4 `0.075442`, within +/-16 `0.277374`. The deltas are not concentrated enough for base-relative coding to beat absolute solved values under Brotli q11.

The prior OD8 proxy was `1245600` B projected. OD9's real persisted combined absolute packet is `1252219` B projected, same order and still far outside the corridor.

## POSE SCOPE
OD3 labeled this n32 set pose-easy at `0.42628664334579025x` population and seg-matched at `1.0099888594483923x`; that caveat binds every pose number here. OD9 re-derived the same-pair persistence and measured mean `d_pose_after_stage1 = 0.033106106524428` and mean `d_pose_after_stage2_cheapdct = 0.000791809037082` on this subset only. I do not project those pose numbers to n600.

## RULE 118 SPLIT
Free generic receiver/code side:
- OD5 parser and inflate.py algorithm.
- Base raw decode already produced by the submitted generator/codec.
- Deterministic camera-to-scorer resize for the base lattice.
- Delta add/reconstruct logic.
- Fixed cheapdct4 basis and inverse transform algorithm.

Counted archive.zip side:
- Support flat-index gaps for edited scorer-lattice pixels.
- Video-derived RGB absolute values or base-relative deltas for those support pixels.
- Video-derived cheapdct4 int16 qcoefficients.

No video-derived table is treated as free code in this receipt.

## RECALL EVIDENCE
| query | source | finding | plan impact |
| --- | --- | --- | --- |
| MEMORY.md OD9 OD8 #899 #904 Pact | /Users/adpena/.codex/memories/MEMORY.md | No OD9-specific prior memory hit found in the quick pass; Pact memory did flag #899/#904 and live Pact discipline as separate apparatus context. | Kept OD9 as a fresh price/receipt unit and did not import #899/#904 apparatus claims into the representation verdict. |
| common contract plus CLAUDE AGENTS PROGRAM operating manual main_hot_state | .omx/tmp/codex_runs/_common_contract.md, CLAUDE.md, AGENTS.md, PROGRAM.md, docs/operating_manual_craft_handoff.md, .omx/state/main_hot_state.md | Scorer slot discipline, rule-118 split, and live own frontier line bind this unit; OD9 scope is scorer-free pricing first. | Did not launch n600 or upstream/evaluate.py; measured packet bytes and kept score_claim=false. |
| OD3 terminality all five files native payload fields | .omx/research/ddm_od3_20260805/{OD3_TERMINALITY_RECEIPT.md,NEXT_IF_RESUMED.md,CHARTER_ADDENDUM_PREREGISTERED_PREDICTION.md,OD3_AGGREGATE.json,od3_seal.json} | OD3 closed terminality but persisted outcome rows only, not native stage1/stage2 payload fields. | Re-ran solve-persist on the same OD2 n32 pair set and stored real solved values on SSD before pricing. |
| OD4 OD6 PE3 OD8 GC18 Stage-1 native packet corridor | .omx/research/ddm_od4_20260805, ddm_od6_20260805, ddm_pe3_20260805, ddm_od8_20260805, ddm_gc18_20260805 | Baselines are OD4 104,775 B projected, OD6 76,304 B, PE3 74,408 B, OD8 proxy 1,245,600 B, and GC18 45-90 KB coherent boundary corridor. | Compared the real persisted OD9 packets against all four measured/preregistered byte bars instead of against the OD8 proxy alone. |
| AM1 IG1 SD1 OD8 post-OD3 persisted values shared context fire order | .omx/research/ddm_am1_20260805, .omx/research/ddm_ig1_20260805, .omx/research/ddm_sd1_20260805 | AM1 and IG1 name post-OD3 persistence as producer for smooth/shared-context A/Bs; SD1 queues a receiver CPWL/task-description packet against the 45-90 KB corridor. | Flat/base-delta OD9 is rate-dead, so follow-ons are queued behind shared/task-description carrier routes rather than treated as OD-line ship-the-solve polish. |
| canonical equations receiver native format coefficient payload decoder counted entropy | .venv/bin/python tools/list_canonical_equations.py --json | No OD9-specific equation found; relevant registry context warns entropy-coded archives require post-decompress grain rather than raw-byte locality assumptions. | Used receiver packet parse-back plus real Brotli/LZMA coder races; did not infer rate from raw array size or byte-gradient heuristics. |
| CANONICAL_RESEARCH_INDEX sub015_DAG OD8 delta entropy TR1 rate-dead scorer-free | .omx/research/CANONICAL_RESEARCH_INDEX*, .omx/research/sub015_DAG_* | Found prior rate-dead direct-correction family cautions and TR1/shared-carrier routing context; no completed OD9 persisted-delta receipt existed before this unit. | Scoped the negative to this flat/base-delta persisted solved-paint formulation and did not kill shared carrier or task-description families. |

## ARTIFACTS
| artifact | bytes | sha256 | path |
| --- | --- | --- | --- |
| persist_json | 142330 | ed3420f40b89136040663584e9d0540cef3c6070c5ef1bce18f418920aee1ea6 | /Users/adpena/Projects/pact/.omx/research/ddm_od9_20260805/od9_js1_persist_n32_cprime_k4.json |
| persisted_native_price_json | 1913 | f77869d3c401ed9b43222d230650eddadc819e9f7e96e662c00e913b30f7ed46 | /Users/adpena/Projects/pact/.omx/research/ddm_od9_20260805/od9_persisted_native_dof_price.json |
| delta_entropy_price_json | 26315 | cdd94808c9f58f3382eeb6b96bbc2c353f50c3bd758ca482f718e6f208b0aa5f | /Users/adpena/Projects/pact/.omx/research/ddm_od9_20260805/od9_delta_entropy_price.json |
| ssd_payload_manifest | 10817 | a409cd3598e99d9af9ffd891bbf84c815b733079837e9d94b31cb26c641175e8 | /Users/adpena/Projects/pact/.omx/research/ddm_od9_20260805/od9_ssd_payload_manifest.json |
| best_combined_packet | 82958 | ad996c55aa26bfe194063bcefbb854f6d281a94b85664efd25290835aa8dcfdb | /Volumes/VertigoDataTier/pact/ddm_od9_20260805/native_dof/od9_combined_absolute_u8_plus_stage2_cheapdct4.od5.raw_packet |
| best_stage1_packet | 79721 | b615df3dc390bb8246233153254507720ef5ed7d39e34a5425cace3a60860464 | /Volumes/VertigoDataTier/pact/ddm_od9_20260805/native_dof/od9_stage1_only_absolute_u8.od5.raw_packet |
| absolute_persisted_packet | 82955 | 7219f6442c06ea00ba4793f905b38bb9ad8b98e37d8b4d9da2a0ebe039f5c18f | /Volumes/VertigoDataTier/pact/ddm_od9_20260805/native_dof/od9_native_dof_n32_cprime_k4.od5.raw_packet |

SSD payload manifest tree SHA: `7a127223953ab330f5539d48ceefd4173972d5728d0d71f6803fb0567c63c4a8` over `42` packet/pair-payload entries.

## FOLLOW-ONS
| id | disposition | fire order |
| --- | --- | --- |
| AM1-smooth-residual-packet | QUEUED-WITH-FIRE-ORDER | Only after a shared/task-description carrier emits a lower-entropy stream; compare acceleration-coded residual against OD9 best absolute and base-delta packets with exact decode equality. |
| IG1-shared-context-vanishing-AB | QUEUED-WITH-FIRE-ORDER | Use this persisted n32 stream as the independent baseline; fire a shared-context/generator packet only if the same decoded output parses back and projects below the 90 KB to 190 KB corridor. |
| SD1-CPWL-task-description-packet | QUEUED-WITH-FIRE-ORDER | Build scorer-free receiver packet first, require exact bytes, packet hash, parse-back equality, double-decode equality, and explicit 45-90 KB corridor falsifier before any scorer slot. |
| OD9-Leg-C-receiver-closed-archive | QUEUED-WITH-FIRE-ORDER | Do not fire n600 from flat solved paint. Fire only after a shared/task-description packet is receiver-closed and under the TR1 190 KB ceiling with a concrete archive path. |

## NEXT_IF_RESUMED
1. Treat flat/base-delta solved-paint shipping as closed for this formulation unless a new shared context changes the decoded object or support.
2. Route next work to a receiver-consumed shared/task-description carrier first: CPWL/boundary grammar, TR1 learned carrier, or a shared-context generator packet.
3. Require exact parse-back, double-decode equality, and a projected byte ceiling below 190 KB before any Leg C n600 scorer slot.
4. If a shared packet survives bytes, then fire n>=32 receiver survival before n600; recompute S from components, not the rounded evaluate.py print.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
