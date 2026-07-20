# Codex findings addendum — Einstein-Kolmogorov crux v3 FIX_ONCE closure

**UTC:** 2026-07-20  
**Lane:** `lane_einstein_kolmogorov_crux_v3_20260720`  
**Authority:** `[macOS-CPU advisory] NON-PROMOTABLE`  
**Classification:** **A — measured, terminal negative**  
**Pointer:** **UNMOVED**

This append-only addendum supersedes only two evidence statements in
`codex_findings_einstein_kolmogorov_crux_v3_20260720_codex.md`:

1. the strict gate is now truly **n24**: all 24 pairs / 48 frames, zero
   differing frames, maximum absolute uint8 difference zero;
2. capped `pose_carrier_confirm` now consumes the exact extracted
   `archive/0.bin` used by shipped decode and is bit-exact across all 24
   frame-zero pairs, maximum difference zero.

The sole external review classified the landing `FIX_ONCE`. Commit
`261a2c8296049d6bef6a43f1aa545d653618a398` also makes strict-n24 validation
reject two-pair receipts, binds inflate resume to the receiver hash, binds score
reuse to GT/oracle/source hashes, and makes completed-result resume revalidate
all durable stages plus cleanup state. Regression tests cover each condition.

The scored packet did not change:

| Quantity | Value | Evidence class |
|---|---:|---|
| archive | 91,062 B, `3555bafc...37c6a` | MEASURED, identical bytes |
| strict gate | 24 pairs / 48 frames / max diff 0 | MEASURED |
| `d_seg` | 0.003555730183919271 | MEASURED, hard CPU-Torch, n600 |
| `d_pose` | 126.30360158587386 | MEASURED, hard CPU-Torch, n600 |
| projected `S` | 35.955425463668846 | DERIVED by `tac.contest_score` |
| delta vs 0.1910828242 | +35.76434263946884 | DERIVED, worse is positive |
| headroom vs 264,320 B | 173,258 B | DERIVED |

No n600 rerun was warranted: the review fix does not change `archive.zip`,
`inflate.py`, or the hard scorer's input raw. The n600 score and certified 3.66
GB raw cleanup therefore remain the exact prior measured evidence. There are no
assumed score components.

## Operator redirect consumed

Per-arm directives dated 2026-07-20T08:54:58Z through 08:59:03Z explicitly
end further iteration on this from-scratch 91,062-byte packet because it
collapses the already solved distortion. The next campaign target is the joint
three-axis KKT optimum formed by composing the proven C1 distortion capstone
with the frontier rate solver, using the registered #536 marginal waterfill and
real byte-closed evaluation. This row remains valuable only as exact terminal
negative evidence.

## Durable custody

- Machine correction receipt:
  `.omx/research/einstein_kolmogorov_crux_v3_fix_once_20260720.json`
- External review:
  `.omx/research/einstein_kolmogorov_crux_v3_external_review_20260720.md`
- Original full score receipt:
  `.omx/research/einstein_kolmogorov_crux_v3_20260720.json`
- Full decode and score stages:
  `.omx/research/einstein_kolmogorov_crux_v3_n600_inflate_stage_20260720.json`,
  `.omx/research/einstein_kolmogorov_crux_v3_n600_score_stage_20260720.json`

This branch is not MAIN. MAIN must review the immutable commit chain, sole
external review, post-review machine receipt, regression tests, and operator
redirect before merge. A merge must not move the pointer.
