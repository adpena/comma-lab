# PK1 Receiver-Closed Boundary Grammar Receipt - 2026-08-05

Schema: `ddm_pk1_receipt.v1`
Created UTC: `2026-08-05T07:41:35Z`
Charter: `.omx/tmp/codex_runs/pk1_prompt.md`
Common contract: `.omx/tmp/codex_runs/_common_contract.md`
Axis: `[macOS-CPU advisory]` for scorer rows; `[macOS-CPU advisory / scorer-free receiver-byte custody]` for byte rows.

## Verdict

PK1 found a byte-positive boundary grammar representation but a scorer-negative receiver realization.

The PE3 hybrid generator-coordinate packet is receiver-closed and double-decode deterministic at 74,408 counted section bytes. It sits inside the GC18 45-90 KB segmentation grammar corridor and beats the OD6 76,304 B byte bar by 1,896 B on the segmentation section alone. When composed with the OD9 cheapdct4 pose carriage projection, the packet is 114,852 B, which misses the hard 90 KB composed gate by 24,852 B but remains inside the broader GC18 90-155 KB legal task-description corridor.

The bounded scorer gate was fired after the PE2 v2 batch was harvested. The PE4 receiver-closed PE3 75KB candidate scored `S = 1.852721897902562 @ 432,428 B`, with `d_seg = 0.00660216`, `d_pose = 0.08182466`, rate ratio `0.011517442215228572`, rate contribution `0.2879360553807143`, pose contribution `0.9045698425218475`, and seg contribution `0.660216`. This is a local advisory negative and is not promotion-eligible.

No contest pointer move. The own-vehicle line remains:

`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.`

## Byte Race

| Row | Denominator and selection | Bytes | Verdict |
|---|---:|---:|---|
| PE3 hybrid segmentation grammar | n600 all pairs; no prefix; PE3EDGE1 section | 74,408 section B | Inside GC18 45-90 KB; 1,896 B below OD6 76,304 B; scorer-free byte-positive |
| PE3 hybrid archive as staged by PE4 | n600 full archive with existing sections | 432,428 archive B | Receiver-closed and scored; scorer-negative |
| OD9 cheapdct4 pose carriage | n32 measured, linear n32-to-n600 projection | 40,444 projected B | Compose-only pose carriage; not a new n600 proof |
| PE3 seg + OD9 cheapdct4 pose | mixed measured n600 seg section plus n32-to-n600 pose projection | 114,852 projected section B | Misses 90 KB by 24,852 B; inside 90-155 KB corridor |
| OD9 flat solved paint Stage 1 | n32 measured, linear n32-to-n600 projection | 1,214,007 projected B | FOLDED: flat solve-paint rate-dead |
| OD9 best combined flat + cheapdct4 | n32 measured, linear n32-to-n600 projection | 1,252,219 projected B | FOLDED: flat solve-paint rate-dead |
| PE4 subset conditional transport | n600 all PE1 generator tracks; real coder race | 180,190 B vs 164,831 B independent | FOLDED: +15,359 B encoded delta; transport not promoted |

## Receiver Closure

Source candidate: `/Volumes/VertigoDataTier/pact/ddm_pe4_20260805_r2/sub_auto_pairbit_pe4_pe3_hybrid_75kb_receiver`

Primary scorer receipt: `/Volumes/VertigoDataTier/pact/ddm_pe4_20260805_r2/scorer_batch/pe3_hybrid_75kb_n600_cpu.json`

PE3 section parse-back:

| Field | Value |
|---|---:|
| section magic | `PE3EDGE1` |
| section bytes | 74,408 |
| section sha256 | `5cc024ad32df7fedb18afb75dbed6be9c1af948dac826a1736cb1084949855c2` |
| body codec | `lzma1-raw` |
| body bytes | 74,351 |
| body sha256 | `792697d0ddf3692a9242fbecf4aa374cbeb293f97b0cf9de3416e4329d7ce0a8` |
| raw bytes | 169,975 |
| raw sha256 | `beecc444dac58e7b345df3783a8b38e20c8c74e8b011ac82bd4cb02c24e697a8` |
| frame records | 600 |
| component records | 8,644 |
| modes | `generator_pair_bisector=7894`, `depth_conditioned_curve=750` |

PE4 runtime consumption:

| Field | Value |
|---|---:|
| payload reencodes identically | true |
| byte ledger closes | true |
| residual bytes | 0 |
| painted pairs | 600 |
| painted pixels total | 540,058 |
| smoke pair | 0 |
| smoke pair painted pixels | 724 |
| camera pixels changed in smoke | 2,896 |
| deterministic raster hash | `1661535005f09a8dcd864fb54d20d18be618455bb7cf0c5801fec3c4efe83818` |
| old qo1 absent-section identity under PE3-capable receiver | true |

Double-decode determinism:

| Decode | Path | Bytes | sha256 |
|---|---|---:|---|
| scorer-gate inflate | `/Volumes/VertigoDataTier/pact/ddm_pe4_20260805_r2/sub_auto_pairbit_pe4_pe3_hybrid_75kb_receiver/inflated/0.raw` | 3,662,409,600 | `0c6c7e68ee1364d78241d2d5ba00cae7109eb171e7bf35720291094e829316df` |
| PK1 second inflate | `/Volumes/VertigoDataTier/pact/ddm_pk1_20260805/double_decode/pe3_hybrid_75kb_second_inflated/0.raw` | 3,662,409,600 | `0c6c7e68ee1364d78241d2d5ba00cae7109eb171e7bf35720291094e829316df` |

Second-inflate receipt: `/Volumes/VertigoDataTier/pact/ddm_pk1_20260805/double_decode/pe3_hybrid_75kb_second_inflate_skip_eval.json`. It reran `experiments/ddm_fz2_byteclose_and_eval.py` with `--skip-eval`, re-extracted `archive.zip`, rechecked byte-ledger closure, and inflated in 194.8257291316986 seconds.

Runtime custody caveat: three staged receiver files are pinned vendored copies that diverge from current repo HEAD (`ddm_r7_token_coder.py`, `ddm_tr1_runtime.py`, `repair_entropy_coder_runtime_adapters.py`). The scored artifact uses those staged bytes; restaging from HEAD would be a different receiver.

## Coverage

Representation coverage is not realization quality.

| Denominator | Numerator | Fraction |
|---|---:|---:|
| flip mass `461,271` | `383,557` | `0.8315220336851872` |
| components `22,338` | `8,644` | `0.38696391798728624` |
| source band pixels `2,569,387` | `102,968` | `0.04007492837785822` |

The packet spends sparse per-edge where-tax rather than dense raster-order where-tax. The 8,644 selected components carry 83.1522% of the extracted flip mass with only 4.0075% of the source-band pixels, using 7,894 generator-pair bisector records plus 750 depth-conditioned curve residual records. This is why the section wins the byte race against denser full-curve rows. The scorer result shows that the current realization still fails through R, uint8, SegNet, and PoseNet: `d_seg` is `0.0022903737722439234` worse than the current own-vehicle `d_seg = 0.004311786227756077`, and pose is not preserved.

## Scorer Gate

The PE2 v2 receiver batch was harvested before firing PK1:

| Candidate | Archive B | d_seg | d_pose | S | Axis |
|---|---:|---:|---:|---:|---|
| PE1 full explicit curve k8 | 478,612 | 0.01733365 | 1.64659417 | 6.109877835060689 | `[macOS-CPU advisory]` |
| PE1 surgical generator pair waterfill 75KB | 425,627 | 0.00638793 | 0.03455114 | 1.510002726247929 | `[macOS-CPU advisory]` |
| BF1 lane crop r3 | 563,256 | 0.00677187 | 1.66379786 | 5.131203885626703 | `[macOS-CPU advisory]` |
| PE4 PE3 hybrid 75KB receiver | 432,428 | 0.00660216 | 0.08182466 | 1.852721897902562 | `[macOS-CPU advisory]` |

Command fired by PK1 after the scorer slot was free:

```bash
bash .omx/research/ddm_pe4_20260805/stage_pe4_fourth_candidate_scorer_batch.sh cpu
```

No scorer row here is a contest score claim; all four rows have `score_claim=false` and `promotion_eligible=false`.

## Rule-118 Split

Free generic receiver side:

- PE3EDGE1 parser and section dispatcher.
- Generator-pair bisector and depth-conditioned curve rasterization algorithms.
- Deterministic byte-ledger parsing, archive extraction, and receiver execution code.

Counted archive side:

- PE3 video-derived component records and mode payload.
- Existing staged renderer, selector, pose_warp, frame0_pose_repair, and PE3 section bytes.
- Any future learned or video-derived transport/statistic payload.

No video-derived table was moved into free code by PK1.

## Recall Evidence

Searched and used:

- `.omx/tmp/codex_runs/_common_contract.md`, `.omx/tmp/codex_runs/pk1_prompt.md`, `PROGRAM.md`, `CLAUDE.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, `.omx/state/active_lane_dispatch_claims.md`.
- Memory quick pass for Road-Lane real-coder and scoped projection discipline; this changed PK1 wording to use explicit denominators and to avoid iid/global claims.
- `.omx/research/ddm_od9_20260805/OD9_RECEIPT.md` and `.omx/research/ddm_od9_20260805/OD9_RECEIPT.json`.
- `.omx/research/ddm_sd1_20260805/SD1_CROSSWALK_RECEIPT.md`.
- `.omx/research/ddm_sj1_20260805/SJ1_CROSSWALK_RECEIPT.md`.
- `.omx/research/ddm_pe1_20260805/PE1_RECEIPT_20260805.md`.
- `.omx/research/ddm_pe2_20260805/PE2_RECEIPT_20260805.md` and PE2 v2 run log under SSD scorer custody.
- `.omx/research/ddm_pe3_20260805/PE3_RECEIPT_20260805.md` and `.omx/research/ddm_pe3_20260805/ddm_pe3_hybrid_receipt.json`.
- `.omx/research/ddm_pe4_20260805/PE4_RECEIPT_20260805.md` and `.omx/research/ddm_pe4_20260805/ddm_pe4_runtime_transport_receipt.json`.
- `.omx/research/ddm_gc18_20260805/GC18_CONVOCATION_RECEIPT.md`.
- `.omx/research/operator_directive_per_edge_optimality_criteria_20260805.md`.
- `.venv/bin/python tools/list_canonical_equations.py --json` filtered for frontier and rate/byte equations; no equation already settled PK1.

Plan changes from recall:

- PE2 and PE4 had already landed receiver consumption, so PK1 fired the PE4 scorer gate instead of rebuilding the receiver.
- PE4 had already repriced conditional transport through real coders, so PK1 folded transport rather than promoting PE3-r2.
- OD5 targeter bytes stayed out of the PK1 receiver verdict because OD5 is targeter-only and not decoder-legal receiver-closed RGB/scorer survival.
- OD9 flat solved paint was folded as rate-dead; only the cheapdct4 pose carriage projection was composed as the pose-side byte comparison.

## Follow-Ons

| Item | Disposition | Fire order |
|---|---|---|
| PE3 receiver scorer gate | FIRED | Completed by PK1 through PE4 staged script after PE2 v2 harvest. |
| PE3 current realization | FOLDED | Scorer-negative at `S = 1.852721897902562`; do not spend another n600 slot on this exact receiver packet. |
| PE4 conditional transport | FOLDED | Encoded subset conditional is +15,359 B versus independent; no PE3-r2 archive promotion. |
| OD9 flat solve-paint shipment | FOLDED | Best projected n600 packet is 1,214,007 B stage1 / 1,252,219 B combined, far outside corridor. |
| Next boundary grammar realization | QUEUED-WITH-FIRE-ORDER | Fire only after a receiver-closed n>=32 scorer-survival probe proves R/uint8/SegNet improvement over current own-vehicle `d_seg`; then stage one n600 advisory row with the exact archive bytes. |

## Required Final Line

`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.`
