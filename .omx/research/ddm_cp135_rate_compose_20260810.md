# DDM CP135 — exact lossless rate composition on PR135

**Axis:** `[macOS-CPU advisory, scorer-free lossless composition]`  
**Verdict scope:** one exact custodied PR135/F26 archive and the HP3-step2 state derived from it  
**Base:** `186,724 B`, SHA-256 `12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004`  
**Winner:** `186,252 B`, SHA-256 `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`

## Result

| Exact whole-container state | Archive bytes | Incremental delta | Total delta vs PR135 | Status |
|---|---:|---:|---:|---|
| Custodied PR135/F26 | 186,724 | — | — | exact contest-CUDA ancestor row |
| VP1 split-model, control RC64 | 186,547 | −177 | −177 | receiver-closed |
| + CAP1 fixed-field metadata pack | 186,468 | −79 | −256 | receiver-closed |
| + HP3 IHS2 step2 and fresh RC64 | **186,252** | **−216** | **−472** | receiver-closed winner |

The winning archive is retained at:

`/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/retained/candidates/hp3_step2/split_brotli_per_section_opt_cap1_metadata__rc64/archive.zip`

The adapted receiver tree, with the same archive, is retained at:

`/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/`

If the exact PR135 distortion components survive the queued CUDA render, the byte-only score derivation is
`0.16195513827824176`. This is **derived, not an exact score**. This arm ran no renderer, scorer, Modal job,
or `upstream/evaluate.py`; the qualifying pointer therefore did not move.

## Per-lever byte ledger

| Lever | Whole-container effect | Absorbed? | Exact finding |
|---|---:|---|---|
| VP1 split-model Brotli per physical section | −177 B | no | PR135's joint raw-LZMA2 model prefix does not contain the best per-section split form |
| CAP1 fixed-field metadata pack | −79 B after VP1 | no | 40 raw bytes become 79 archive bytes after the independently selected section coder changes |
| HP3 IHS2 step2 plus exact F26 probability recode | −216 B after VP1+CAP1 | no | model −741 B, fresh RC64 +525 B, net −216 B |
| LC2 same-state stack ANS | +6 B control; +9 B HP3 | yes | RC64 wins on both exact F26 probability states |
| R7 SMEVR | 0 wins / 14 sections | yes | it loses every physical/decomposed PR135 section race winner |
| LOTTO shared dictionary | +136 B vs selected WANS+Brotli | yes | exact WANS1/F12 restoration, but larger |
| LOTTO supermask + shared dictionary | +254 B vs selected WANS+Brotli | yes | exact WANS1/F12 restoration, but larger |

All deltas above are complete deterministic ZIP recounts. No row is a linear sum of projected section
savings. The winner uses Brotli qualities `[10, 11, 11]` for HPAC, semantic, and packed carrier sections,
respectively; its counted payload is `70,825 B` model + `96 B` compact residual + `115,231 B` RC64 +
`100 B` ZIP framing.

## Receiver closure

The final parse-back receipt is
`/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/PARSEBACK_RESULT.json`.

- The adapted receiver restores the exact PR135 semantic blob: `36,051 B`, SHA-256
  `b489c73567046e64a1644eb1bca5cb5ed86d690f2f98f703e22424ab97505521`.
- It restores the exact PR135 carrier blob: `22,242 B`, SHA-256
  `196f0e5136f4d6bfd22c4cf24ad779eee55f6e95a4f5f5994ae09a4fc268b6ef`.
- It restores the intended HP3-step2 IHS2 blob: `16,599 B`, SHA-256
  `4fc531c47919e48f276988be50459e3094c6f1bafb04f2b96f5ce03c1f992d0e`.
- It restores the exact PR135 residual payload: `100 B`, SHA-256
  `bd27a2ddb17067995f4cf6ac085d35299970862b88b623e42ac19d1579eaff46`.
- It consumes the retained fresh HP3 RC64 payload: `115,231 B`, SHA-256
  `8fe9bb3cd4dc42668730690bddad091a86ebed6b2c74e7773a11ee951d2bd15d`.

The fresh HP3 RC64 decoder reproduced all `117,964,800 / 117,964,800` source symbols. Its event-order
SHA-256 is `8eb51ab7a2884c9d7b6e73ee60f78ded38c691d6b82e639b75dddec6e0ac1366`; its spatial-token SHA-256 is
`c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`. The parallel HP3 stack-ANS
decoder also reproduced all symbols and terminated empty. Thus the output-driving semantic, carrier,
residual, and decoded token values are equal by construction. Literal CUDA frame rendering remains the
queued authority boundary.

The adapted runtime contains 25 durable files and has tree SHA-256
`f56139f82447018765879ce5ba6138d087911dacaa1f181a79cabea03b790996`. Its receiver adds only the
split-Brotli and packed-CAP1 inverses before the unchanged F26 decode/render path. `inflate.sh` pins
`Brotli==1.2.0`, installs it through `uv` into a success-cleaned temporary directory when absent, and
compiles the unchanged RC64 decoder. The locked T4 fire must pass this dependency preflight before the
one exact row is consumed.

## Coder races

The corpus receipt is
`/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/CORPUS_CODEC_RACE_RESULT.json`. It retains 15 real coder
payloads for each of 14 exact sections: identity, raw LZMA2, Brotli qualities 0–11, and recalled R7 SMEVR.
SMEVR helps the raw control and HP3 IHS2 bodies relative to identity, but never wins the complete coder
race: on HP3 IHS2, SMEVR is `14,273 B`, raw LZMA2 is `14,103 B`, and Brotli q10 is `13,910 B`. On the
dominant RC64 token stream, SMEVR costs `+11,090 B`.

The renderer receipt is
`/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/LOTTO_RENDERER_RACE_RESULT.json`. Both tested LOTTO forms
restore canonical WANS1 and the exact F12 physical bytes. The exact PR135 W4 state contains `63,936` code
values and is `85.465%` nonzero, so the supermask is not sparse enough to pay. The best shared-dictionary
form is `34,899 B`, versus `34,763 B` for the selected WANS+Brotli section.

## Payload retention and reproducibility

Every materialized probability code lattice, coder output, candidate archive, repeated archive, model
section, residual section, token stream, corpus-race payload, LOTTO payload, decoded symbol stream, and
checkpoint is retained below `/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/`. The 600 HP3
probability-frame checkpoints are individually reloadable. Both ANS and RC64 encoders preserve
stage checkpoints; the checkpointable RC64 backend passed a resumed-versus-direct byte-identity test.
Every admitted archive has a deterministic repeat with the same bytes and SHA-256.

Primary machine receipts:

- `FINAL_RESULT.json` — winner, derived score boundary, and per-lever ledger.
- `BUILD_RESULT.json` — all 12 complete candidate archives and whole-container counts.
- `PARSEBACK_RESULT.json` — adapted receiver values and runtime tree custody.
- `retained/coders/hp3_step2/FRESH_RC64_RESULT.json` — full n600 symbol identity.
- `retained/coders/hp3_step2/ANS_RESULT.json` — full n600 symbol identity and empty terminal state.
- `CORPUS_CODEC_RACE_RESULT.json` and `LOTTO_RENDERER_RACE_RESULT.json` — negative race denominators.
- `EXACT_EVAL_FIRE_ORDER.json` — single-flight queue disposition for MAIN.

## Borrowed substrate accounting

The PR135 archive, learned HPAC/semantic/carrier state, F26 receiver, and CUDA renderer are codexblack's
borrowed PR135 substrate. This arm does not claim that learned state or vehicle as ours-original.

Our original contribution in this successor is the concrete composition: VP1's split physical-section
representation on this base, the CAP1 fixed-field metadata pack and inverse, the HP3-step2 composition
with exact F26 probability export and fresh resumable recodes, the same-state ANS/RC64 race, the recalled
SMEVR and exact LOTTO races, and the receiver-equality/whole-container harness. LC2 ANS did not survive
the exact PR135 race and is not present in the winner.

## RECALL EVIDENCE

Before pricing or building, recall searched `.omx/research/` memos and arm receipts, `.omx/state/`,
`CANONICAL_RESEARCH_INDEX*`, the sub-0.15 DAG FEED surfaces, canonical equations from
`.venv/bin/python tools/list_canonical_equations.py --json`, the task ledger, and `main_hot_state.md` by
content. Queries included `lc2`, `VP1`, `HP3`, `split-model`, `SMEVR`, `LOTTO`,
`renderer_weight_codec`, `#940`, `races-not-reputation`, `WANS1`, `CAP1`, and `RC64`.

Findings beyond the charter seeds changed the implementation:

- `.omx/research/ddm_fd135_fractal_decomposition_20260810.md` supplied the exact F26 physical-section
  boundaries, ZIP floor, RC64-near-ideal closure, and the 40-raw-byte CAP1 proposal. That added the
  CAP1 pack, forbade RC64 parameter tuning, and prevented reopening direct camera-residual reclaim.
- `.omx/research/ddm_hp3_20260810/FINAL_REPORT.md` and its retained code manifest supplied exact
  HP3-step2 selected-logit codes and receiver-closed IHS1 custody. That made HP3 a real probability
  recode rather than an additive eight-byte projection.
- `.omx/research/ddm_r7_smevr_liveness_on_v4d_20260801.json` and
  `experiments/ddm_bd1_class_field_receiver.py` supplied the recalled SMEVR record machinery. That
  expanded the race to 14 exact PR135 sections rather than testing only the token bulk.
- The PR135 ExperimentBook's `renderer_weight_codec` showed that its selected semantic state already
  used WANS. Together with the prior LOTTO/supermask records, this changed the addendum into two exact
  restoration forms raced against the selected WANS section, not a reputation-based replacement.
- `.omx/research/ddm_ah2_arm_harvest_20260810.md` established that VP1 and HP3 savings could not be
  added linearly. That forced 12 retained complete-container candidates and an incremental byte ledger.

No additional canonical equation displaced the exact real-coder/whole-container protocol for this
surface.

## Exact-eval disposition

`QUEUED-WITH-A-FIRE-ORDER` to **MAIN exact contest-row owner**. The consumer store is
`/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/exact_eval`. Fire only after MAIN owns the sole exact
lane, verifies archive `186,252 B` / SHA-256 `6eb1a3b7…edb6`, verifies the adapted runtime tree, and passes
the locked T4 `Brotli==1.2.0` preflight. Then run exactly one n600 contest-CUDA `upstream/evaluate.py`
row, harvest all payloads, recompute score from components, and promote only the identical pinned archive.

Own-vehicle frontier remains **lc2 `S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, adjudicated,
n600]`**. This scorer-free unit did not move it or reach sub-0.15.

## ADDENDUM 3 (MAIN, 2026-08-10 19:55Z) — EXACT ROW CONFIRMED SUB-BAR + MAIN ADJUDICATION

The one contest-CUDA row fired per the fire order LANDED and CONFIRMED the derivation exactly:

- **S = 0.16195513827824176** recomputed from components [contest-CUDA T4, n600, locked env]
- avg_segnet_dist **0.00029643** · avg_posenet_dist **6.88e-06** — BIT-FOR-BIT identical to the
  PR135 replay's distortions. The 117,964,800-symbol identity proof HELD through the literal T4
  render; the one untested link is now tested.
- Custody: archive sha `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`
  @ 186,252 B (expected==evaluated, gpu_t4_match=True, validation_errors=[]); runtime tree
  `36712078…`; call `fc-01KZPK1KHE01QWDSVH7BQBEA3W`; modal_elapsed 403.6 s; result at
  `experiments/results/modal_auth_eval/ddm_cp135_composed_paired_modal_auth_20260810T193605Z_cuda/`.
- **vs the custodied bar** (PR135 replay 0.16226942370411543): **−0.00031429. FIRST SUB-BAR ROW.**
  Also below PR135's published 0.16226842169958583. Best-known point on the board.
- MAIN ADJUDICATION: ACCEPT. Grounds: sha+size+GPU custody exact, n600, zero validation errors,
  distortion identity with the base replay (the strongest possible cross-check). CPU leg
  redispatched for the honest per-axis record (their f26 renderer is CUDA-locked; expected fast
  refusal, same as the base). Axis label stands [contest-CUDA]; contest-CPU remains
  REFUSED-BY-VEHICLE for this family.
- Honesty (NO-FAKE #7 unchanged): PR135/codexblack base + F26 receiver + CUDA renderer are
  borrowed-granted; ours-original = VP1/CAP1/HP3 composition + exact recodes + receiver-equality
  harness. This row is a COMPOSED-ON-GRANTED-BASE bank, not an own-vehicle row; lc2 remains the
  own-vehicle frontier. Next composition: ps135's pose re-solve on top of this base (js1).
