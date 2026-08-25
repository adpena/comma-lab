# s1e Stage-A both-OFF floor verdict — ENTERED_AND_REFUSED, ON not authorized, diagonal closed at family scope (2026-08-25)

**STORES CONSULTED:** the adjudication receipt
`/Volumes/APDataStore/pact/ddm_s1e_off_floor_adjudicator/both_off_endpoint.json` (35 SHA-pinned rows,
live-content-verified archives) · the sealed Stage-A memo `.omx/research/ddm_s1a_stage_a_adapter_20260825.md`
(MAIN_LAUNCH_ORDER sha 708eae6a…, NEXT_IF_RESUMED fire triggers, LIVE-HYPOTHESES) · task ledger rows
#1270/#1215/#1239/#1222/#1230/#1088/#1252 · the gestalt memory
[[dx2-block-ceilings-are-measured-and-sum-to-5-percent]] · the resumability-cure memo
`.omx/research/ddm_s1a_seed2_sigkill_adjudication_and_wd3_resumability_cure_20260825.md`.

## The measurement

Both OFF seeds ran to stage end (seed 20260815: ep65 clean; seed 20260816: ep95 after the crash-resume,
matched at the ep65-window checkpoints plus its extension). The pre-registered GB1 renderer-corner
falsifier (`tools/s1a_off_floor_adjudicator.py`, `[Darwin-mps frozen-scorer advisory]`, n60
evenly-strided pairs, non-authoritative by construction) returned:

**`ENTERED_AND_REFUSED_ALL_POINTS` — 0 of 35 checkpoints cross the corner.**

Best point (seed 20260816, ep75), composed ΔS vs break-even **+0.14815737243836**:

| leg | value | vs gb1 reference |
|---|---|---|
| seg damage | +0.06046 S | hard_d_seg 8.060e-4 vs 2.0139e-4 (4.00×) |
| pose damage | +0.08238 S | d_pose 8.164e-4 vs 6.37e-6 (128×) |
| rate | +0.00532 S DEBIT | packet 38,847 B vs renderer 30,856 B (−7,991 B shed) |

The controller's own quantization race explains the rate debit: uniform int4 (38,847 B) is the ONLY
allocation passing all gates; int3 (31,491 B) and int2 (24,135 B) fail pose + road/lane + hard-cell
gates; adaptive (38,105 B) fails road/lane. The trained renderer never sheds a byte against the
incumbent.

**Prior-law prediction CONFIRMED 35/35:** pose damage exceeds seg damage at every checkpoint
(0 falsifying points) — the #1222/#1230 law (the RENDERER carries pose) reproduced on a fresh
two-seed floor.

## The ON-authorization review (sealed-order obligation) — NOT AUTHORIZED

The sealed order queues `on_seed_20260815` (sampled real uniform int3/int2 rungs) behind MAIN's review
of "the measured seed floor plus its unresolvable delta-S." That review is now arithmetic. Grant the
ON arm BOTH of its best-case outcomes simultaneously:

1. QAT rungs make int2 pass every gate → bytes_shed +6,721 B → rate swing −0.00979 S (from +0.00532
   debit to −0.00447 credit);
2. The LIVE-HYPOTHESES compensation recovers ALL pose damage → −0.08238 S.

Composed best case: 0.14816 − 0.00979 − 0.08238 = **+0.0560 S still above break-even**, carried
entirely by seg — and #1088 measured this seg training regime asymptoting ABOVE its own init (5× the
window does not reach parity), so longer training does not cure it. The ON burn (~3.5 h Metal +
review chain) is structurally dominated at its own best case. Per the #1252 lesson (never authorize a
run structurally guaranteed to refuse): **ON is not authorized. Stage B and Stage C do not fire** —
their triggers require a Stage-A window that crossed, and none exists.

## Verdict and scope

**The S1 trained-renderer diagonal (#1270) closes at FAMILY scope**: family = wd3 W96
width-distillation from the dx2 teacher (Film-W96, flattened arm, this trainer, both seeds, 35
matched checkpoints, controller-raced quantization). The refusal is measured at the floor
(+0.148 S), with the pre-registered falsifier fired on every point and the prior law confirmed on
every point. This is NOT a closure of every conceivable joint object — it closes THIS family and,
with it, the last named route.

**The #1215 2×2 is now fully populated and every cell refuses.** Both diagonal entries are
measured: the semantic joint move refused at 686× (#1239); the trained-renderer diagonal refused at
+0.148 S floor (this memo). The sharp-optimum law (#1214 — the HPAC optimum is sharp in every
direction) now extends to the trained-renderer direction.

## Campaign state after this verdict (honest)

Sub-0.12 has **no measured-live route** on the dx2-lineage object or any named alternative object.
The measured remainder: model-axis conditional structure ~2,009 B (marginal shrinking 153→23 B per
family) + coder 88 B + the banked jt21 −23 B rider. Fully collected these move S 0.148118 → ~0.1467
— real pointer moves, nowhere near 0.12. Per the operator's standing "any progress on the way down":
the next concrete fire is the 22nd conditioning family composed with the jt21 bank across the ~30 B
solo-fire bar → one T4 row. New-object invention remains open as exploration, with every named family
now bracketed by a measured refusal multiple.

verdict_scope: FAMILY (wd3-W96 width-distillation diagonal on the dx2 teacher; instrument =
n60 frozen-scorer advisory, non-authoritative by construction; the family closure rests on the
pre-registered falsifier + confirmed prior law + best-case-granted ON arithmetic, not on a single
instrument row).

## Routing

- #1270 → diagonal ENTERED_AND_REFUSED at the Stage-A floor; ON/B/C not fired; family closed.
- ON request `on_seed_20260815` (config sha 163cb5d5…) → RETIRED-NOT-AUTHORIZED with this memo as
  the recorded review; reactivation only via a mechanism that attacks the SEG leg below ~2.5e-4
  hard_d_seg at ≤ renderer bytes (nothing named does).
- Metal + scorer lanes → RELEASED; next occupant = the 22nd-family model-axis build (Opus slot).
- All 35 evaluated packet/archive payloads retained per ALWAYS-KEEP-THE-PAYLOAD (receipt pins
  sha256 + bytes for every row; live-content-verified).
