# ddm_wd4 — warm-lineage nested width on the exact fx5 semantic state

**Date:** 2026-08-21  
**Disposition:** `CEILING-PASS_GATE-BLOCKED`  
**Authority:** `[macOS-CPU scorer-free exact fx5 byte/container + receiver parse-back]`  
**Score claim:** false  
**Frontier:** **UNMOVED** — `S = 0.14823186109359`, `180,386 B`, fx5 archive SHA-256
`4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841`
on `[contest-CUDA T4 n600]` remains the live row.

## Verdict

The byte ceiling is open, but the training/scorer gate is not adjudicated.

Width 64 is the largest nested dense four-block child that clears the charter's
`12,155 B` complete-archive requirement. The inherited-GroupNorm-group-salience
child is a real, retained, receiver-closed archive of **166,465 B**, saving
**13,921 B** versus fx5. Its counted rate credit is
**−0.009269422486413747 S**. Width 72 saves only **10,879 B** under the same
selector and is ceiling-dead.

The governed bounded Metal launch was attempted through preflight and stopped before
step 0. This sandbox reports PyTorch `2.12.1` with MPS built but unavailable, and MLX
reports no Metal device. CPU substitution is forbidden. No training step, scorer call,
Modal call, or paid action occurred. The byte result is not a `READY` verdict and the
pointer did not move.

## Exact lineage

The conditioned shipping lineage is not the e960 checkpoint bank.

- fx5 ships a **36,130-byte SM3R mode-6 decoded semantic packet**, SHA-256
  `17e0fd0b197ac147afe98397ef38f02f7915b69372d03c042e6be6fa0f992e50`,
  carried in a **30,856-byte** CK2+Brotli section, SHA-256
  `39d1be52ba62933498395c48ce4d9482f37db097d504da76c2a321efe3e4a76f`.
- Its receiver-decoded state has exactly **38 tensors**. Per-tensor float32 SHA-256,
  shape, element count, and nonzero count are retained in
  `/Volumes/APDataStore/pact/ddm_wd4_warm_lineage_width/retained/source_fx5/fx5_semantic_38_tensor_manifest.json`.
- The training ancestor is stage-08
  `semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt`, SHA-256
  `3948ccfcd44778dc42affee18a10c3f3baa434d1a2eb2345a013146c1dbfb647`.
  Stage-07 is the sibling terminal checkpoint
  `semantic_renderer_w96_b4_qat4_12k.pt`, SHA-256
  `1549607db224ea2c4681738dbcc80d2ba9dd453de72db1cf60309985d0602eaf`.
- The exact fx5 state is a post-training SM3R representation descendant, not a
  byte-identical stage-08 state. Against stage-08's q4 realization it differs in
  **5/38 tensors and 9,414 elements**: `frame_embed.weight` and the four
  `blocks.{0..3}.film.weight` tensors. This is the expected mixed-depth/row-pruned
  surface, and it is why the archive-decoded state is the only exact warm parent.
- There is **no exact optimizer checkpoint or EMA checkpoint** for that SM3R state.
  Stage-07/stage-08 are terminal Torch checkpoints; SM3R is a later representation
  transform. A warm continuation must therefore initialize model state from the exact
  archive and initialize optimizer/EMA state explicitly.
- `tools/select_hpac_checkpoint.py` and the e480→e960 bank concern the HPAC probability
  model only. The selected e960 artifact is epoch 634,
  `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/full_e480b_e960/checkpoints/full_mps_e960.checkpoints/periodic/epoch_0634.pt`,
  SHA-256 `5007beae7af7789758092f12f49096e13692e2e59850c85eb4642cd6fad147ec`.
  It did not change the semantic state.

The maximum semantic-section credit on fx5 is **30,856 B**, the limiting case where
the entire current compressed section disappears. That is an upper bound, not a viable
candidate. All admissibility decisions below use complete deterministic ZIP bytes.

## Real serialized width rows

Every row uses the exact decoded fx5 SM3R tensors, the unchanged four-block dense
mechanism, uniform int4 deployment, the additive `WD2S` counted packet, whole-section
CK2, Brotli quality 11, and a stored single-member deterministic ZIP. The salience
selector ranks whole inherited 8-channel GroupNorm groups and preserves channel order
inside each group; the index-prefix row is the matched nested control. Every packet, coded semantic
stream, member `p`, archive, and repeat archive is retained.

| width | selector | packet B | semantic stream B | archive B | saving vs fx5 B | clears 12,155 B |
|---:|---|---:|---:|---:|---:|---|
| 88 | salience | 35,353 | 26,925 | 176,455 | 3,931 | no |
| 88 | prefix | 35,353 | 26,920 | 176,450 | 3,936 | no |
| 80 | salience | 30,761 | 23,269 | 172,799 | 7,587 | no |
| 80 | prefix | 30,761 | 23,405 | 172,935 | 7,451 | no |
| 72 | salience | 26,489 | 19,977 | 169,507 | 10,879 | no |
| 72 | prefix | 26,489 | 19,962 | 169,492 | 10,894 | no |
| **64** | **salience** | **22,537** | **16,935** | **166,465** | **13,921** | **yes** |
| 64 | prefix | 22,537 | 16,921 | 166,451 | 13,935 | yes |
| 56 | salience | 18,905 | 14,162 | 163,692 | 16,694 | yes |
| 56 | prefix | 18,905 | 14,211 | 163,741 | 16,645 | yes |
| 48 | salience | 15,593 | 11,531 | 161,061 | 19,325 | yes |
| 48 | prefix | 15,593 | 11,528 | 161,058 | 19,328 | yes |
| 40 | salience | 12,601 | 9,233 | 158,763 | 21,623 | yes |
| 40 | prefix | 12,601 | 9,269 | 158,799 | 21,587 | yes |
| 32 | salience | 9,929 | 7,229 | 156,759 | 23,627 | yes |
| 32 | prefix | 9,929 | 7,266 | 156,796 | 23,590 | yes |

Selection mode is complete archive bytes. There is no first-order compression-factor
projection in this table.

## Width-64 receiver and score bar

The retained gate runtime is
`/Volumes/APDataStore/pact/ddm_wd4_warm_lineage_width/retained/runtime_gate_candidate_group8_salience/`.
Its archive is **166,465 B**, SHA-256
`29c26e428647a8feb1a9614e55be3d12a88a66351443f7f98de23622982f7a06`.
The semantic packet is **22,537 B**, SHA-256
`7cacb8fd4d2f4a18ce2358757eab6e518ff685acdbf9f5adb6dec068ab28b1ce`.
The patched parser returns that packet byte-identically and the receiver strict-loads
`dense, depth=4, width=64`. WANS1, SD1M, and SM3R branches remain present and unchanged;
`WD2S` is a narrow additive dispatch. This is a byte/receiver proof, not an output or
score proof.

Using only the measured fx5 components and the real archive size:

- rate credit: `13,921 * 25 / 37,545,489 = 0.009269422486413747 S`;
- projected score after the **full** current Seg term is removed:
  `0.14823186109359 - 0.009269422486413747 - 100*0.00020139`
  `= 0.11882343860717624`;
- total realized Seg/Pose degradation headroom to remain below 0.12:
  **0.0011765613928237523 S**;
- charter's early-checkpoint `1.5x` gate:
  **0.0017648420892356284 S** degradation versus matched fx5.

These are first-order arithmetic bars, not predicted model quality. The first scorer
gate must measure the realized joint Seg/Pose delta on a deterministic stratified
population.

## Training apparatus and blocked fire

`experiments/ddm_wd4_warm_lineage_width.py train-gate` is the sealed bounded child entrypoint;
it refuses an ungoverned launch and requires a matching active local lane claim.
It starts from the retained width-64 packet, uses the exact fx5 decoded uint8 teacher
and MC36 token field, runs only on PyTorch MPS with fallback disabled, and is capped at
64 steps. The default window is 32 deterministic stratified pairs, 32 steps, AdamW
`2e-7`, and checkpoints every 8 steps. It writes atomic distinct initial, periodic,
and terminal stage checkpoints; each checkpoint includes live model state, optimizer,
RNG binding, EMA shadow, source hashes, and a retained byte-closed EMA candidate plus
repeat archive. `--resume-from` restores the whole state.

Prelaunch evidence is retained at:

- `/Volumes/APDataStore/pact/ddm_wd4_warm_lineage_width/METAL_PROBE.json`;
- `/Volumes/APDataStore/pact/ddm_wd4_warm_lineage_width/TRAIN_GATE_BLOCKED.json`;
- `/Volumes/APDataStore/pact/ddm_wd4_warm_lineage_width/GOVERNED_PRELAUNCH_STATUS.json`.

The probe failed before any lane claim or training launch because no local Metal device
is visible. Separately, `.omx/state/active_lane_dispatch_claims.md` currently has MAIN's
`ddm_jo1_payload_unblock` scorer/materializer active. The common contract does not grant
WD4 scorer ownership, so no scorer was fired and no duplicate lane was created.

## Payload custody and verification

The consumer store is
`/Volumes/APDataStore/pact/ddm_wd4_warm_lineage_width/`. It currently contains 549
files, 12,941,218 logical bytes (80 MiB allocated). The retained tree includes exact
source fields, all 16 width/subset candidates, deterministic repeats, the gate runtime,
the Metal probe, and the prelaunch blocker. `RESULT.json` is the machine-readable
authority. No materialized payload was discarded.

Two earlier receiver copies remain retained as explicitly superseded build evidence:
`runtime_gate_candidate/` used individual-channel reordering, and
`runtime_gate_candidate_group8/` bound the prefix control during review. Neither is the
selected gate runtime and neither may be scored as WD4's candidate.

Verification completed:

- `python -m py_compile experiments/ddm_wd4_warm_lineage_width.py` — pass;
- `ruff check` on builder and tests — pass;
- WD4 + WD2 tests — **10 passed**;
- relevant WD3 receiver/packet/score/negative tests — **26 passed**;
- exact candidate parse-back and packet parse/repack — pass;
- deterministic archive repeat — pass for every candidate.

## RECALL EVIDENCE

Recall was corpus-wide before the build, not seed-memo-only.

- Governing contract: `CLAUDE.md`/`AGENTS.md` (identical), `PROGRAM.md`,
  `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, the WD4
  charter, and `_common_contract.md`.
- Campaign/equations: `.omx/research/ddm_gs3_unbridled_gestalt_20260821.md` and
  `.omx/research/ddm_r012_rate_representation_20260821.md` supplied the current
  `12,155 B` bar, full-Seg-win arithmetic, and warm-lineage fire order.
- Negative bank: `.omx/research/ddm_wd3_n120_family_disposition_20260816.md`, the D56
  and F64 verdict memos, and retained n120 receipts close only fresh instances. They do
  not close a nested warm descendant.
- Frozen-section census: the mz2 memo/result and `ddm_wd2_width_distillation_build` show
  38/38 semantic tensors are live, recoding/precision are closed, and width is the only
  multi-kilobyte semantic axis left.
- Checkpoint lineage: hv1 endpoint-closure receipts, `tools/select_hpac_checkpoint.py`,
  PR130 stage-07/stage-08 custody, and the B2E sealed ticket separate e960 HPAC custody
  from semantic custody.
- Index/DAG/task surfaces: the research index/DAG entries, `main_hot_state.md`, active
  lane ledger, task ledger, and arm-final messages were searched for warm width,
  D56/F64, fx5, e960, SM3R, and the 12,155-byte gate.

The beyond-seed change that altered the plan is decisive: current fx5 ships SM3R mode-6
at 30,856 compressed bytes, not the older WANS semantic section assumed by WD2-era
pricing. It also changes 5 tensors/9,414 elements versus the stage-08 q4 state. Therefore
WD4 bound the archive-decoded state and repriced complete fx5 archives instead of
transferring the older e480b width table or treating e960 as a semantic parent.

## Sealed fire order

1. On a host with visible PyTorch MPS, claim a unique local Metal lane and run the
   retained 32-step `train-gate` entrypoint with fallback disabled. Resume only from its
   atomic checkpoints.
2. After JO1 is terminal and MAIN grants a unique scorer lane, score the matched fx5
   baseline and the first retained WD4 checkpoint on the same deterministic stratified
   `n>=32` population. Stop as `GATE-FAIL` if joint degradation exceeds
   `0.0017648420892356284 S`.
3. Only if the early gate passes, run the charter-required advisory n600 gate checkpoint
   in chunks `<=120`, retain all scorer payloads, and compare against the
   `0.0011765613928237523 S` final degradation ceiling.
4. Only an n600 pass may produce `READY` and a sealed longer-burn fire order. No fresh
   D56/F64 rerun, long burn, Modal training, or exact contest eval is authorized by this
   memo.

## Own-frontier line

**Pointer unchanged:** fx5 remains `S = 0.14823186109359 @ 180,386 B`
`[contest-CUDA T4 n600]`. WD4 produced a 166,465-byte receiver-closed unscored candidate,
not a lower exact row.
