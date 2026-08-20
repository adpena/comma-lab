# ddm_nv1 — numeric-claims verification over the PR packet (harvest adjudication)

`date_utc: 2026-08-20` · `owner: MAIN (harvest of the nv1 verifier arm)` · `score_claim: false` · `frontier_moved: false`

**Own-vehicle frontier: S = 0.14839100138338618 @ 180,625 B `[contest-CUDA T4, n600]`.** Unmoved.

## Verdict

15 load-bearing numeric claims in the packet surfaces (PR_BODY_DRAFT / README_PUBLIC /
REPORT_PUBLIC / BORROWED_SUBSTRATE_ACCOUNTING / CONTRIBUTIONS_INVENTORY) were traced to
primary receipts. **14 of 15 VERIFIED against their originating artifacts.** One substantive
defect, one provenance correction, two caveat-quoting obligations.

## The four actionable findings

1. **The 95.72% wrong-referent (the one defect) — public surfaces ALREADY FIXED, internal
   surfaces still carry it.** 95.72% is the token-decode share of the 1,401.58 s
   instrumented-stage SUM; against the 1,419.900 s inflate elapsed the true figure is 94.5%.
   pq5's fix landed on all three public surfaces (94.5% primary, 95.72% parenthesized with
   its denominator named — verified by grep 2026-08-20). Still wrong-referent, on #1157's
   fix list: `ddm_pq4_packet_completeness_20260820/{MAIN_HANDOFF.md:71,
   COMPRESSION_SCRIPT_VERDICT.md:95, CONTRIBUTIONS_INVENTORY.md:101,125}` (internal only).

2. **The residual-band "disagreement" is RESOLVED — it was never two measurements.** Both
   bands come from `ddm_wc2_wall_clock_pass_20260820.md`, not one from cw1: `[822, 1302] s`
   is the ua2-derived CUDA **residual** window (wc2:31,86,273,394; source ua2:189), and
   `[890.6, 1430.6] s` is its **evaluate-corrected derivative**, derived at wc2:653-655 as
   `[822,1302] + (120…180 − 51.4)`, consumed by pq3:147. One measurement, two frames
   (residual vs evaluate-corrected). **Freeze action:** the packet should present them as
   such — the "one of these bands is wrong" framing in pq4's MAIN_HANDOFF §4.1 is
   withdrawn. The REFUSE-vs-WARN grade difference is a framing difference: the 1,419.9 s
   inflate elapsed sits above the residual band's ceiling (REFUSE frame) and inside the
   evaluate-corrected band (WARN frame). Quote both with the derivation, grade against the
   evaluate-corrected band, keep the rr2 native port as the cure either way.

3. **Claim-2 caveat owed at freeze:** the 1.774–1.834x local speedup range straddles the
   1.804x PASS bar; the receipt refuses to call PASS because run variance exceeds the
   distance to the bar. Any packet sentence quoting the range must keep that refusal.

4. **Claim-10 caveat owed at freeze:** the contest-CPU 3,422.711146813 s figure is labeled
   `inherited, not measured` in PACKET_TARGET.json:137 — no contest-CPU row exists on the
   jg5 bytes. Any CPU-axis sentence must carry the inherited label.

## Also verified (no action)

- gen-4 same-sha dual-score anomaly (0.15710… authority tree vs 79.402… mixed-lineage
  desync tree) correctly documented as receiver-tree difference, rc 0 garbage decode.
- rc64_backend.c census (241 copies, 4 distinct bodies, roles registered), jg5 edit
  telemetry (455/573, 600/600 no_improving_step, 3.8373 bits/token), ck2 −657 B,
  to1 −105 B, br1 measured-null, additivity <3%, PR#135 rank-1 0.162, wc2c 24/34 census,
  13.4x pose-vs-seg edit loss — all trace to their receipts exactly.

## Stale-arm disposition (same harvest)

`ddm_fo2h` (η-hardening, chartered 08-17 off FO-1) died on that day's usage limit with no
landing. Its subject — hardening the pose-null seg channel's η (n=12) behind a −0.000505 S
waterfill margin — was overtaken by the jg5 edits-waterfill line (fifteenth move) and is
NOT on the submission critical path. Parked, not killed: if the pose-null seg channel is
ever re-admitted to a waterfill, η must be hardened from n=12 first (the FO-1 verdict,
DAG FEED-2026-08-17e, stands as the reactivation condition).

## Consumers

#1157 wave-end review (items 1–4 above are named checks) · the freeze-boundary re-stage
(MAIN_HANDOFF §4.1 band framing) · rr2 port charter (band framing does not change its
critical-path status).
