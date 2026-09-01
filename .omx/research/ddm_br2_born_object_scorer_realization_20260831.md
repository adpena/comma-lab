# DDM BR2 — born-object n600 scorer realization

**Date:** 2026-08-31  
**Arm:** `ddm_br2_born_object_scorer_realization`  
**Axis:** `[macOS-CPU advisory]`, `score_claim=false`  
**Verdict:** **DISTORTION-REFUSED**, scope **INSTANCE** — the exact retained 106,832-byte archive only.

## Result first

| component | measured value |
|---|---:|
| archive | **106,832 B**, SHA-256 `0e2ffdfaa5fe481d481dd70a9672a67f80b9aad7648f0c775fe2956dd3a4841d` |
| rate, `25*B/37,545,489` | **0.07113504367994782** |
| `d_seg` | **0.17077688429090712** |
| Seg term, `100*d_seg` | **17.077688429090713** |
| `d_pose` | **115.83742417077798** |
| Pose term, `sqrt(10*d_pose)` | **34.03489740997877** |
| distortion | **51.11258583906948** |
| `S` recomputed from components | **51.18372088274943** |
| delta versus 0.12 | **+51.06372088274943** |
| delta versus AFR1 `0.14797617125559104` | **+51.035744711493834** |

The exact 106,832-byte rate leaves a sub-0.12 distortion allowance of
`0.12 - 0.07113504367994782 = 0.04886495632005218`. The realized distortion is
**1,045.996756945733x** that allowance. Seg alone is 349.4874387533777x the allowance and
Pose alone is 696.5093181923553x. The measured Pose term already makes the implied Seg budget
negative (`d_seg < -0.33986032453658716`), so this instance cannot be a sub-0.12 candidate.

The charter's abbreviated `0.071125...` rate and `0.048734` allowance are corrected here by exact
recomputation from the physical archive byte count. The older QBZ1 `B_hat=122,062 B` remains a
separate HT reset projection; BR2 scores the charter-named physical 106,832-byte archive and does
not substitute the projection.

## Per-class Seg decomposition

The denominator is 117,964,800 SegNet pixels: 600 last frames x 384 x 512. Contributions below
sum to global `d_seg` up to floating-point rounding.

| target class | target pixels | errors | conditional error | contribution to global `d_seg` |
|---|---:|---:|---:|---:|
| Road | 27,407,372 | 16,822,018 | 0.6137771253661242 | 0.14260201348198784 |
| Lane | 690,754 | 690,754 | 1.0 | 0.005855594211154514 |
| Undrivable | 58,413,067 | 9,033 | 0.00015464005682153275 | 0.00007657368977864584 |
| Movable | 1,460,386 | 1,460,386 | 1.0 | 0.012379845513237847 |
| MyCar | 29,993,221 | 1,163,470 | 0.03879109882863198 | 0.009862857394748265 |
| **all classes** | **117,964,800** | **20,145,661** | — | **0.17077688429090712** |

This member completely misses Lane and Movable, and most of the global Seg error comes from Road.
That description is about this archive's realized output, not a family theorem.

## Stage 0 — custody, receiver identity, and AFR1 SHA resolution

- Input container: 2,723,840 B, SHA-256
  `4c16e6c045768b2dee62f59ac9a2a27b7386280dfccff3dd5331a8d9509d95f7`.
- `FIT_RESULT.json`: 433,162 B, SHA-256
  `69b33e5d393deff7f1fcd76844cf524d7c19691f431aa399a876b2ad1ce227bf`.
- `archive.zip` and `archive.repeat.zip`: both 106,832 B and byte-identical at SHA-256
  `0e2ffdfaa5fe481d481dd70a9672a67f80b9aad7648f0c775fe2956dd3a4841d`.
- The archive's `0.qbf` packet and repeat: both 106,724 B and byte-identical at SHA-256
  `8c26684d33313ca44f3d4f02cf3c369f0f33d6de37eeba42ae4220faed3e6d38`.
- Receiver decode/re-encode reproduced both the packet and deterministic archive bytes exactly;
  the complete section set and all 600 latent records parsed successfully.
- The NX1 memo's AFR1 SHA `cbb8e900...` is a non-authorizing transcription error. The canonical
  frontier pointer and AFR1 authority memo agree on
  `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.
  This discrepancy does not touch any BR2 input.
- Stage-0 storage preflight observed 6,138,626,048 free bytes against the required 5,000,000,000;
  no retained payload was deleted or moved.

Stage-0 receipt: `/Volumes/APDataStore/pact/ddm_br2/checkpoints/stage_00_receiver.json`, 2,981 B,
SHA-256 `94a849c5ae7d03eb03a06d03af3d4c26f7d7f6c8c5cd8522c0390b4b8204e5ac`.

## Stage 1 — full retained scorer realization

The governed detached run completed rc=0 in 484.769 s under launch counter 709. It decoded the
exact packet, rendered all 600 two-frame pairs, applied the real camera round trip, and ran frozen
CPU SegNet and PoseNet against registered DALI-aligned authority targets. Denominators are:

- 600/600 pairs, with pair IDs exactly `0..599`;
- 117,964,800 Seg pixels;
- 3,600 Pose values, six per pair;
- 20 retained 30-pair shards, so no scorer chunk exceeds 120 pairs.

Each shard retains the uint8 two-frame camera render, fp16 SegNet logits, generated and target
argmax, generated Pose6, and target Pose6. The 20 shards total 1,058,094,084 bytes under
`/Volumes/APDataStore/pact/ddm_br2/realized_n600/`. Their individual paths, byte counts, and
SHA-256 values are in `REALIZED_RESULT.json`; all 20 hashes were independently rechecked after
the run. The result was independently recomputed directly from the retained arrays: 20,145,661
Seg errors and Pose SSE 417,014.7270148007 reproduce the stored `d_seg`, `d_pose`, per-class sums,
rate, and `S`.

Primary receipts:

- `/Volumes/APDataStore/pact/ddm_br2/REALIZED_RESULT.json`, 13,517 B, SHA-256
  `a7ae997a75cd86fa1e36552cd83c5b7b208874438832ebc1555e24666e9a4c8e`;
- `/Volumes/APDataStore/pact/ddm_br2/PAIR_ROWS.json`, 481,901 B, SHA-256
  `dba07e8f281df09dc667d15884a4246c8a75efe2eaaf3af478d7b8d5a27df104`;
- `/Volumes/APDataStore/pact/ddm_br2/launch/launch_manifest.json`, 5,034 B, SHA-256
  `67bc3d6c477dcce24bbba7c7a8534a864ff095ffa6ca0e692a5ab5647caa488d`;
- `.omx/tmp/codex_runs/ddm_br2_born_object_scorer_realization.done`, 402 B, SHA-256
  `95301c14adcb2fa7c36bab3ba6df9ad981a097ef80630cba64e5956880d3c4f2`, rc=0.

## Typed disposition and boundaries

`DISTORTION-REFUSED` closes only this exact archive instance. It does not close the continuous
implicit partition family, establish a capacity/optimization fork, or transfer the measured
numbers to QX1. No contest CPU/CUDA evaluator, Modal call, seal, public runtime tree, or pointer
mutation occurred. The physical archive is locally receiver-closed, but it is not a promotable
contest archive without a sealed public receiver runtime and an authority evaluation.

NX1 rank 1 is now terminal and refused. Its rank-2 trigger is therefore satisfied: the next bounded
action is the scorer-free QX1 transitive section census, not another qbt2b n32 doubling chase and
not an exact-address residual. The native-coefficient `100/100` versus `100/0` A/B remains the
required experiment before anyone labels this family capacity-limited versus optimization-limited;
it is not needed to close this already-refused archive instance.

## RECALL EVIDENCE

I searched the full bounded corpus rather than only the charter seeds:

- content queries `qbt2b|qbflow|born object|capacity ceiling|n600`,
  `QBW|QBMIX|QBCERT|quotient`, `task.space|witness|preimage|two-plane`,
  `Lane|topology|residual|generator`, `pose-null|bit-identical|pose-priced`, and
  `106832|0e2ffd|69b33e|realize --scorer|B_hat` across `.omx/research/`, arm-final receipts,
  `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, `.omx/state/`, task-ledger surfaces, and the QBT/QBZ
  instruments;
- `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for
  `qbt|qbflow|born|capacity|configuration|allocation|descent|alignment|score_marginal`;
- current hot state, canonical pointer, lane claims, QBT/QBZ AP receipts, and the AFR1 authority
  memo.

Beyond the seeds, three findings changed execution. First, the inherited QBZ1 `realize` path prices
the HT `B_hat=122,062 B`, while BR2 is chartered to realize the exact 106,832-byte archive; this
required a BR2 receiver path that could not silently substitute the projection. Second, the live
hot-state mention of WWC1 was stale relative to its terminal queue/done receipt, so the unique BR2
scorer claim was safe to append and later close terminal. Third, canonical pointer and AFR1 authority
custody resolved the NX1 SHA discrepancy as transcription, avoiding a false input blocker. No
registered QBT-specific capacity equation was found in the canonical-equation search; the
native-coefficient A/B remains an experiment, not a recalled theorem.

## Frontier

BR2 produced no pointer movement. The exact contest frontier remains AFR1:
`S=0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]`, archive SHA-256
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.

## NEXT_IF_RESUMED

- **Disposition `QUEUED_WITH_FIRE_ORDER`; owner MAIN-assigned QX1 derivation arm; consumer store `/Volumes/VertigoDataTier/pact/ddm_qx1/SECTION_CENSUS.json`; fire trigger:** BR2 rank 1 is terminal `DISTORTION-REFUSED` and MAIN assigns the owner; run the scorer-free transitive section census first, classifying every retained QBT section as reused, replaced, or forbidden, and refuse implementation unless the section-complete real-coder envelope is below 137,986 B with no explicit address, GT, or scorer-weight stream.

## LIVE-HYPOTHESES

- **QX1 may escape both measured walls.** A continuous implicit partition avoids exact-address
  carriage, while a jointly optimized two-plane RGB/YUV preimage directly addresses the catastrophic
  Pose and class-birth failure measured here.
- **Optimization versus capacity is still distinguishable inside the QBT family.** The preregistered
  same-start, same-seed native-coefficient `100/100` versus `100/0` scorer-finish A/B can test whether
  native loss competes with realized Seg descent, although it cannot rehabilitate this fixed archive.

## DEAD-ENDS

- **This exact 106,832-byte archive is closed.** Its measured distortion is 1,045.997x the lawful
  allowance, with both Seg and Pose independently over budget.
- **The scorer-free native 0.0141554 row is not a realized ceiling.** Real R/uint8/frozen scorers
  produce `d_seg=0.1707768843` and `d_pose=115.8374242` on the shipped packet.
- **Do not continue the qbt2b n32 doubling chase or buy an n600 Metal extrapolation.** BR2 bought the
  decisive full-n600 terminal for the retained member; more proxy trajectory does not change it.
- **Do not append exact masks, Lane addresses, topology events, or an exact residual.** Existing
  real-coder measurements price those representation forms above the lawful budget.
- **Do not compose this object's rate with another object's distortion.** BR2 measures one exact
  receiver object; a cross-object score would be fake.

