# ddm_jg2 — the sub-0.15 chain: replace the modelled rate leg with a measurement

- **arm** `ddm_jg2` (task #1139 — the successor to `ddm_jg1`'s joint solve)
- **date** 2026-08-19
- **axis** every number is `[macOS-CPU advisory]` unless it carries an explicit DALI-lineage
  tag. `score_claim=false` · `promotable=false`. This arm fires **no Modal job**; MAIN owns
  the T4 slot.
- **cost** $0.
- **store** `/Volumes/APDataStore/pact/ddm_jg2/`
- **status** IN PROGRESS — written incrementally, committed at every stage boundary.
  **Pointer UNMOVED** at contest-CUDA `0.15652626435208142` until a T4 row says otherwise.

## THE BASE (re-read from `.omx/state/canonical_frontier_pointer.json` at arm start)

| term | value | S contribution |
|---|---:|---:|
| `d_seg` | 0.00030309 | 0.030309 |
| `d_pose` | 7.649246787e-06 | 0.008746 |
| archive | 176,420 B | 0.117471 |
| **S** | | **0.15652626435208142** |

`archive.zip` sha `7ce46fd7a845d5987903a0d85a56581961eb7716a55c38a7361e3b5ecae94b5f`.
**Gap to sub-0.15 = 0.006526.**

## THE INHERITED PROJECTION, AND THE ONE LEG THAT IS NOT MEASURED

`ddm_jg1` (memo `.omx/research/ddm_jg1_joint_solve_20260819.md`) established, at $0:

1. a **validated** local contest-axis seg instrument (`0.99995x` of the T4 seg leg);
2. the **move-class law** — single-cell token coordinate moves repair ~1.5 argmax cells per
   changed token and compose within a sparse pass; block/dilation moves realize worse;
3. the **hard negative and its reversal** — token seg edits destroy pose (`x387`), but
   re-running the carrier's own coordinate descent against the edited frame recovers
   `d_pose` to `1.073x` of original at ~0 bytes. **The actuators compose.**

Its rate leg is **modelled, not measured**: `+4.718 bits` per changed token, computed from
the **hm1/182,759 B body's** probability model, then transferred to the to1/up3 body we
actually ship. jg1 names three reasons that constant is suspect, and **all three point the
same way — the real price is likely HIGHER**:

| # | risk (jg1 S1d caveats 3-5) | direction |
|---|---|---|
| 3 | cross-body transfer: to1's model is **sharper** (0.007446 vs 0.007603 bits/token) | costs MORE |
| 4 | context coupling: the HPAC model decodes in 190 groups, feeding decoded tokens forward | costs MORE |
| 5 | the table correction is omitted from the marginal number | unknown sign |

Two extrapolations exist and they disagree, which is itself information:

| source | repaired cells | changed tokens | net S |
|---|---:|---:|---:|
| jg1 §S1e "honest extrapolation" | ~11,400 | ~7,800 | **-0.0066** |
| jg1 §S2 first-pass scale-up (the charter's headline) | ~18,000 | ~11,600 | **-0.0104** |

The gap is 0.006526. **The honest one barely clears it; the headline clears it with room.**
Both rest on the same modelled constant. That is why S1 runs before anything else.

## STAGE LEDGER

| stage | what it settles | status |
|---|---|---|
| S1 | REAL `ΔB` for jg1's retained 3-pair edit set, through a real encoder on the to1 body | IN PROGRESS |
| S2 | n600 joint solve, seeded-random pair order, rate-aware acceptance | GATED on S1 |
| S3 | byte-close + identity control + determinism + seal | GATED on S2 |

**HONESTY RAIL (charter, binding).** `-0.0104 S` is a 3-pair extrapolation. Realized-vs-
projected is printed at every scale rung. A smaller honest win still seals and fires; an
honest refusal with the measured curve is a first-class landing.

STORES CONSULTED: `.omx/state/canonical_frontier_pointer.json` (re-read at arm start) ·
`.omx/research/ddm_jg1_joint_solve_20260819.md` (full) ·
`/Volumes/APDataStore/pact/ddm_jg1/JG1_RETENTION_MANIFEST.json` + all 12 retained files ·
memory `pose_gap_was_gt_cache_lineage_not_cuda_20260819` ·
memory `the_denominator_and_the_falsifier_can_both_be_vacuous_20260816` ·
memory `concavity_helps_when_you_pay_the_axis_upward_20260818`.

---

## S1 — THE REAL RATE

(in progress; findings land here before S2 starts)
