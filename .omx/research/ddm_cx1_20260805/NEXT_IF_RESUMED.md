# ddm_cx1 Next If Resumed

This arm is complete as a scorer-free static pre-fire gate. Resume only after MAIN publishes the current v3 smoke receipts or asks for a follow-up gate decision.

Fire order:

1. FIRED: Consume both v3 smoke receipts from MAIN: entry ep1336 and refuse-final ep1354. Require `jd1_realized_hold_latch`, live gate telemetry, stage-window EMA re-anchor, and at least one post-latch gate per smoke.
2. FIRED: Apply `CX1-FG2` and `CX1-FG3`. Missing realized-hold rows, missing per-pair scatter, or rollback without a post-rollback gate blocks checkpoint selection.
3. QUEUED-WITH-FIRE-ORDER: If both smokes show slack lane guard plus Lane/seg give-back, run a separate ratchet-on v3b smoke before full FIRE. Do not mutate the live v3 primary label.
4. QUEUED-WITH-FIRE-ORDER: Before long full FIRE, resolve deterministic-R policy on the Metal host or record an explicit accepted noise-floor decision.
5. QUEUED-WITH-FIRE-ORDER: Only after v3 adjudication, consider EN1 margin-weight A/B, SL2 teacher distill, and PE3 conditioning-only as v4 riders.
6. FOLDED: Do not tune `jd1-seg-hold-weight` for realized-hold strength; in realized mode it is an enablement validation value, not the actuator.

Do not run training, scorer, or Metal work under this cx1 arm unless a new charter explicitly changes the owner and lane claim.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
