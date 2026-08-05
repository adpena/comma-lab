---
schema: ddm_sl1_staging_law_correct_measurements.v1
date_utc: 2026-08-05
arm: sl1
axis: "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE"
score_claim: false
promotion_eligible: false
pointer_moved: false
tokens: [no-triality, p0-ledger-ok]
---

# SL1 - staging-law-correct measurements

## Answer First

No exact row was produced and the contest pointer did not move. SL1 measured the DQ1 tail confirmation and the LC1 per-record label curve, and blocked the ET1/SQ2 terminal-pose composition leg on missing composed-byte custody rather than composing onto scalar summaries.

| leg | status | denominator | result | disposition |
|---|---|---:|---|---|
| Leg 1: ET1/SQ2 composed terminal pose | BLOCKED | ET1 n32 receipt; SQ2 n32 receipt | prior receipts preserve scalar metrics, not edited frame_1 bytes or paint tensors | QUEUED-WITH-FIRE-ORDER to rerun realizers with per-pair composed bytes persisted |
| Leg 2: F2 DQ1 tail confirmation | MEASURED | 17/17 tail pairs; corrected n120 aggregate | pose term `0.07024019179644458 -> 0.058293481255628804`; tail mean `0.0014145876354304256 -> 0.00033066675070032587` | DQ1 wall is BUDGET-CONDITIONAL, not convergence-closed; still above the `0.05` term bar |
| Leg 3: F1 LC1 per-record curve | MEASURED | 8,644/8,644 PE3 records; n600 frozen SegNet argmax | 107 positive records; best static positive subset `+619` net fixed; all-record net `-245443` | F1 has narrow trust-gate headroom, but PE3 target-labels remain globally negative |

## Leg 1 - Composition Blocker

SL1 did not measure composed ET1/SQ2 terminal pose because the required post-seg-correction frame bytes are absent from custody.

The consumed receipts contain rows, scalar d_pose values, and convergence curves:

| source | rows | SHA-256 |
|---|---:|---|
| `/Volumes/VertigoDataTier/pact/ddm_et1_20260803/et1_b16_realization_n32.json` | 32 | `4c2d01a7af9bfc1cbff0e6f72188db218a5ea01573e818f5d5434b0a731f47ff` |
| `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_stage_n32_uncap100_sq2.json` | 32 | `dc7ecfe5c1578cc6a7f2668c070f04251b7e570a3e288d2789364d4e8ecead0b` |
| `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_aggregate_n32_uncap100_sq2.json` | aggregate | `f6d5ef091fd574d34fbc06cf4230c13a4b1654db94600b6fcf822d221f1c113a` |

They do not contain the edited frame_1 camera bytes, scorer-lattice paint tensors, or sufficient deterministic replay state for the selected best iterate. Terminal pose must be applied to the actual corrected frame bytes. Composing onto `d_pose_after` scalars would be a fake measurement.

Blocker artifact: `/Volumes/VertigoDataTier/pact/ddm_sl1_20260805/sl1_leg1_composition_blocker.json`, SHA-256 `f840b8ef7ab349bd45eb2ef6124c153ec44f833a5843210c85202e49bacc8dfc`.

## Leg 2 - F2 Tail Confirmation

SL1 reran exactly the DQ1 tail pairs with `d_pose_free_u8 > 1e-3` at a 1,600-iteration budget.

| metric | DQ1 160-iters | SL1 1,600-iters |
|---|---:|---:|
| tail pairs | 17 | 17 |
| tail mean d_pose | `0.0014145876354304256` | `0.00033066675070032587` |
| n120 mean d_pose | `0.0004933684543601322` | `0.0003398129956900347` |
| n120 pose term | `0.07024019179644458` | `0.058293481255628804` |
| final-10% plateau passes | not measured | 1/17 |
| max tail d_pose after rerun | `0.0027784248191924494` | `0.0008455590141169927` |

Verdict scope: FORMULATION, tail-only 10x rerun plus DQ1 non-tail rows. The 160-iteration DQ1 wall was too strong, but SL1 did not refute the wall at convergence because the corrected pose term remains above `0.05` and 16/17 tail rows did not satisfy the final-10% plateau criterion.

Artifacts:

| artifact | bytes | SHA-256 |
|---|---:|---|
| `/Volumes/VertigoDataTier/pact/ddm_sl1_20260805/sl1_f2_tail_confirmation.json` | 227,249 | `4cdfa1790fdd4f67ba92b09e1349b990adebd4b0dbeffa3266d76eaa461008bc` |
| `/Volumes/VertigoDataTier/pact/ddm_sl1_20260805/sl1_f2_tail_confirmation.partial.jsonl` | 104,925 / 17 rows | `4ab3122d34ea15028394553114f13dd58f5efd82e494bb845e03d0ff7fa64e2e` |

## Leg 3 - F1 Per-Record Curve

SL1 regenerated the PE3 per-record local-net curve from the byte-closed PE3 section and frozen CPU SegNet argmax surfaces over all 600 pairs.

| metric | value |
|---|---:|
| PE3 records | 8,644 |
| positive records | 107 |
| negative records | 8,344 |
| zero records | 193 |
| all-record net_fixed | `-245443` |
| best static positive subset net_fixed | `+619` |
| best sorted-prefix records | 107 |
| positive subset fixed / introduced | `1299 / 680` |
| Lane-to-Road introduced pixels | `109412` |

Mixture split:

| component family | records | fixed | introduced | net_fixed | Lane-to-Road introduced |
|---|---:|---:|---:|---:|---:|
| `generator_pair_bisector` | 7,894 | 10,201 | 230,423 | `-220222` | 92,663 |
| `depth_conditioned_curve` | 750 | 5,097 | 30,318 | `-25221` | 16,749 |

Lane-to-Road split:

| bucket | records | fixed | introduced | net_fixed |
|---|---:|---:|---:|---:|
| has Lane-to-Road introduced | 4,145 | 7,805 | 145,085 | `-137280` |
| rest | 4,499 | 7,493 | 115,656 | `-108163` |

Artifact: `/Volumes/VertigoDataTier/pact/ddm_sl1_20260805/sl1_lc1_per_record_curve.json`, 6.7 MiB, SHA-256 `876f0f9dc824e5f8d100420b3bf7dca9aee6c570fb2d08d0b06a53b42f0a1ff3`.

## RECALL EVIDENCE

| source searched | finding | impact |
|---|---|---|
| Governing docs | `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, `docs/operating_manual_craft_handoff.md`, and `.omx/state/main_hot_state.md` loaded the no-fake, no full-n600, protected-path, serializer, review-gate, and current frontier boundaries. | Kept work to bounded CPU scorer passes, no launches, no exact score claim, no protected-file edits. |
| Charter inputs | `.omx/tmp/codex_runs/sl1_prompt.md`, `_common_contract.md`, `.omx/research/ddm_audit_naive_binary_20260805.md`, and `.omx/research/ddm_dq1_20260805/DQ1_RECEIPT.md`. | Drove the three-leg structure and the F2 tail-pair list. |
| ET1/SQ2 receipts and SSD dirs | ET1/SQ2 scalar receipts exist, but no edited frame_1 or paint payloads were found in `/Volumes/VertigoDataTier/pact/ddm_et1_20260803` or `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts`. | Leg 1 blocked fail-closed instead of manufacturing a composed measurement. |
| EG1/PFS1/P3V2 producers | `terminal_pose_gn.py`, `pfs1_warp_receiver.py`, `tools/rehearse_terminal_pose_gn.py`, and `experiments/ddm_p3v2_optimal_form_pose_resolve.py` expose the banked terminal-pose machinery. | Reused p3v2 helpers for F2; did not rebuild pose machinery. |
| RZ1/LC1 producer | `experiments/ddm_rz1_pe3_head_solve.py` exposes `extract_pe3_section`, `parse_pe3_components`, and `effective_component_ownership`; LC1 anchored PE3 archive and base raw hashes. | Regenerated all-record F1 curve from parsed PE3 records rather than using LC1's aggregate only. |
| Memory registry | Pact memory reminded that current-state surfaces must be reread and that frontier work must remain lane-claimed, artifact-producing, and score-custody honest. | Kept labels advisory/non-promotable and carried exact pointer honesty. |

## Boundaries

No `upstream/evaluate.py` was run. No full n600 authority score, contest-CPU row, contest-CUDA row, remote dispatch, GPU launch, or training launch was produced. The LC1 curve uses n600 frozen SegNet argmax as advisory measurement only. The F2 rerun uses bounded CPU PoseNet on 17 tail pairs plus DQ1's non-tail rows. The contest pointer is borrowed/unmoved.

`experiments/ddm_sl1_measurements.py` SHA-256: `593e7f322df73ae5b165e77479d385de6ff5806f56cc75e0d2a39fc7243c0a73`. The file passed `py_compile` and two `tools/review_tracker.py mark-file ... --status reviewed` passes after the final edit.

## NEXT_IF_RESUMED

```json
{
  "sl1_status": "partial_complete_no_score_claim",
  "leg1": {
    "status": "QUEUED_WITH_FIRE_ORDER_BLOCKED_MISSING_COMPOSED_BYTES",
    "fire_order": "Rerun ET1 block16 and SQ2 uncap100 realizers with per-pair edited frame_1 bytes or paint tensors persisted, then apply terminal pose to those actual bytes and run R8 on the composed n32 objects."
  },
  "leg2": {
    "status": "MEASURED_BUDGET_CONDITIONAL",
    "corrected_n120_pose_term": 0.058293481255628804,
    "next": "If this formulation remains live, extend the tail solves or change solver form until the final-10% plateau criterion closes; do not cite the current result as convergence-closed."
  },
  "leg3": {
    "status": "MEASURED",
    "positive_subset_net_fixed": 619,
    "next": "Use the 107 positive-record trust gate as an oracle upper-bound cue for a learned conditioner; do not ship PE3 target-label substitution."
  }
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
