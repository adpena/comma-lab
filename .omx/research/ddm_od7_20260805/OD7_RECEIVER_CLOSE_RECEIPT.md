# OD7 receiver-close receipt - 2026-08-05

Status: `RECEIVER_CONSUMED_SELECTED_SET_MEASURED / OD6_TARGETER_FORMAT_GAP_HELD / NO FRONTIER MOVE`.

Axis: `[macOS-CPU frozen-scorer n32 advisory]`.
`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`, `upstream_evaluate=false`.

## Answer First

OD7 staged a real `sub_od7` receiver artifact at `/Volumes/VertigoDataTier/pact/ddm_od7_20260805/run_20260805T_od7_codex/sub_od7`.  The staged `archive.zip` is `460585` B with sha256 `f2b8f4392e2642c2e3f7fb892815b2f59248242e31b26d568fc5a849543c497c`.  The selected OD6 n32 mask set was consumed by an OD7 sparse RGB section and measured through the edited RGB receiver path with frame_0 recomputed from edited frame_1.

Best measured value mode: `sparse_gt_rgb`.  Its n32 advisory projection is `S=1.067512050`, `delta_vs_live=+0.313531321`, and `delta_vs_od6_mask_projection=+0.323911280`.  This is not a contest score and does not move the pointer.

OD7 also found a hard format gap for pure OD6 packet receiver recomputation: the OD6 b1024 hash header requires `pe1_generator_coverage` and `pe3_hybrid_knee_coverage`, while the OD6 packet row stores only the PE3 hybrid75 n32 subset and the bucket table.  The staged artifact therefore proves receiver consumption of the materialized selected set, not a self-contained OD6 targeter.

## Value Menu

| mode | section B | records | exceptions | flips after | retained | d_pose mean | projected S | delta vs live |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_proto` | 5585 | 6111 | 0 | 28355 | -1105 | 0.010753564 | 1.068153354 | +0.314172624 |
| `sparse_gt_rgb` | 20987 | 6111 | 0 | 28182 | -932 | 0.010228725 | 1.067512050 | +0.313531321 |
| `hybrid_proto_gt_exceptions` | 16218 | 6111 | 3766 | 27932 | -682 | 0.013137705 | 1.103058796 | +0.349078066 |

## Receiver Proofs

- Candidate parse-back re-encoded the IX2 payload identically: `True`.
- Appended OD6 guard section sha256: `52bd534cea34578bb8d76fb079befa951d26f914347bd51b7078c398e2ff6e2a`.
- Appended OD7 value section sha256: `827d0b8a1d1e468059d9a1b50270422b87ee62a40552bd77b74d38cab07a829d`.
- Absent-section identity on n32 qo1 raw using the patched runtime: `True`.
- Runtime changed-pixel proof total: `24444` changed frame_1 camera pixels for `6111` selected records.
- Projected full decode wall clock from n32 loop: `197.524` seconds.

## RECALL EVIDENCE

| source | recalled fact | plan impact |
|---|---|---|
| `.omx/tmp/codex_runs/od7_prompt.md`, `_common_contract.md` | OD7 must not run `upstream/evaluate.py` or n600, must preserve protected files/staged index, and must end with the live frontier line. | Ran only n32 frozen-scorer advisory and wrote serializer-ready artifacts. |
| `.omx/state/main_hot_state.md` | od3 owns the scorer slot and OD7 headline is realization tax. | Did not fire the queued n600/full scorer gate. |
| `OD6_DECODER_LEGAL_RECEIPT.md` and its raw packet | OD6 incumbent is `base_rgb_generator_geometry_b1024`, exact n32 packet bytes `7334`, projected n600 packet bytes `76304`, projected `S=0.743600771`. | Reconstructed the exact selected n32 mask set and measured RGB realization tax against that projection. |
| `OD2_STAGE12_RECEIPT.md` | k=4 carriage byte credit is `57,600` n600 bytes but the JSON stores outcomes, not coefficients. | Reported sparse pose damage and queued fresh carriage derivation instead of pretending OD2 coefficients were reusable. |
| `experiments/inflate_runner_v4d.py` | Optional IX2 sections are parsed fail-closed and frame_0 is computed from materialized frame_1. | Generated an OD7 runtime patch that consumes OD7 sparse RGB and recomputes frame_0 from edited frame_1. |

## SHA Table

| artifact | bytes | sha256 |
|---|---:|---|
| `/Users/adpena/Projects/pact/experiments/ddm_od7_receiver_close.py` | 68973 | `8d65fc9dfb25a4a2f88a2485144e8a5c2e3a26d12a42f45adeaff47b51b2e34b` |
| `/Volumes/VertigoDataTier/pact/ddm_od6_20260805/run_20260805T030410Z/packets/base_rgb_generator_geometry_b1024.od5.raw_packet` | 12577 | `52bd534cea34578bb8d76fb079befa951d26f914347bd51b7078c398e2ff6e2a` |
| `/Users/adpena/Projects/pact/.omx/research/ddm_od6_20260805/ddm_od6_decoder_legal_receipt.json` | 550769 | `4775cb3fc8925de33a70945ba1469b962f0934b5527a032e39cb60e0589566b1` |
| `/Users/adpena/Projects/pact/.omx/research/ddm_od2_20260805/od2_js1_n32_cprime_k4.json` | 103690 | `fd1016751e4668ff786692f52f91d924be97081a70a20d11e470150aaf85c6af` |
| `/Users/adpena/Projects/pact/.omx/research/ddm_od2_20260805/PAIR_SELECTION.json` | 2388 | `0a8ac26a1cd39c7dc425dbb4922d0dda6f71227b205241d3d771ea9791c2d4f9` |
| `/Volumes/VertigoDataTier/pact/ddm_pe3_20260805/pe3_20260805T000000Z/sub_auto_pairbit_pe3_hybrid_75kb/archive.zip` | 432428 | `3f08c7fdd1c2746fa456ef8b6d8005e850d1a3acac5665a5d08b2ef17585b5e0` |
| `/Volumes/VertigoDataTier/pact/ddm_od7_20260805/run_20260805T_od7_codex/sub_od7/archive.zip` | 460585 | `f2b8f4392e2642c2e3f7fb892815b2f59248242e31b26d568fc5a849543c497c` |
| `/Volumes/VertigoDataTier/pact/ddm_od7_20260805/run_20260805T_od7_codex/sub_od7/inflate_runner.py` | 59308 | `0299cad26eca4b5f2bc94514cf644c01c007090bbcf224b5156ddd85ec1cad99` |
| `/Users/adpena/Projects/pact/.omx/research/ddm_od7_20260805/od7_receiver_close_receipt.json` | 84297 | `b8be8ec5a2b8c839edf71aa7cb88a1ae2300cb5de2c964d1e3332353326d5f46` |
| `/Users/adpena/Projects/pact/.omx/research/ddm_od7_20260805/NEXT_IF_RESUMED.md` | 809 | `d8fc60d13259914abb951c764a5f2c931bba93855f7fa06cda94ab84e9001c10` |

## NEXT_IF_RESUMED

See `/Users/adpena/Projects/pact/.omx/research/ddm_od7_20260805/NEXT_IF_RESUMED.md`.  First gate: close the OD6 packet self-containment gap or explicitly route a counted n600 selected-value stream; then use the queued n32/full receiver gate only when the scorer lane is free.

## Boundaries

- No `upstream/evaluate.py`, contest-CPU, contest-CUDA, MPS, or full n600 scorer run.
- The staged value section is n32 selected-set materialization; non-selected pairs remain unedited by OD7 values.
- The OD6 table/packet is guarded and parsed, but pure receiver recomputation of OD6 b1024 is held because the packet lacks all hash feature coverage sources.
- OD2 k=4 pose credit was not re-applied; a fresh OD7 frame_0 carriage is queued.
- This does not move the frontier.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
