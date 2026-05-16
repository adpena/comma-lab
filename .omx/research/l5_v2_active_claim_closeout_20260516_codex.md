# L5 v2 Active Claim Closeout - 2026-05-16

## Context

`tools/operator_briefing.py --json` was correctly suppressing L5-v2/PR106
frontier target rows when live dispatch claims existed. Three nonterminal rows
remained active even though the underlying smoke artifacts had already been
recovered and were non-promotional. That made the briefing look blocked by
in-flight work rather than by the real L5-v2 evidence gaps.

## Closeout

Recorded terminal claim rows in `.omx/state/active_lane_dispatch_claims.md`:

- `lane_d4_wyner_ziv_paired_full_modal_dispatch_20260515` /
  `substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch_20260515T194805Z__smoke__100ep`
  -> `completed_smoke_recovered_no_paired_auth_eval_spawned`
- `lane_d4_wyner_ziv_paired_full_modal_dispatch_20260515` /
  `d4_wyner_ziv_paired_full_modal_20260515T194700Z`
  -> `stale_superseded_by_recovered_smoke_no_paired_auth_eval_spawned`
- `lane_z4_v2_wunderkind_e1_tier1_modal_paired_20260515` /
  `z4_v2_wunderkind_e1_pending_smoke_20260515T194341Z`
  -> `stale_superseded_by_recovered_z4_step2_smoke_no_score_claim`

## Evidence

D4 smoke evidence:

- call id: `fc-01KRPJZ9FY7N1HJH6HMEK6TX6C`
- rc: `0`
- elapsed seconds: `6914.784993056`
- archive ZIP SHA-256:
  `88658722701bfc2970123ee9b0e701196746a49a1509dbf92e51b58e8b3f2741`
- archive ZIP bytes: `2421202`
- score claim: `false`
- promotion eligible: `false`
- paired auth eval spawned: `false`

Z4 contemporaneous smoke evidence:

- call id: `fc-01KRPJVEMQ5S7Q8EKGKQWKCS93`
- rc: `0`
- elapsed seconds: `2930.851619733`
- archive SHA-256:
  `48dc2f77ee12eb149c410ce7ffa703a0755940f97d99632318786c27078f4dc2`
- archive bytes: `166512`
- diagnostic CPU score: `90.16108088426107`
- score claim: `false`
- promotion eligible: `false`

## Post-Closeout Check

`tools/claim_lane_dispatch.py summary --ttl-hours 24` reported:

```text
CLAIM_SUMMARY active=0 stale_nonterminal=0 terminal_latest=921 unparsable_timestamp=0 invalid_lane_id=4
```

`tools/operator_briefing.py --json --skip-provider-readiness` then reported
`dispatch_claim_summary.active_count=0` and `dispatch_claim_gate_blocked=false`
for L5-v2. The remaining L5-v2 blockers are now real evidence blockers, not
phantom active-dispatch blockers.
