# PR130 lift wave — recursive adversarial review ROUND 1 (MAIN, 2026-08-06)

Operator-convened. Scope: every surface this wave changed (et4 cache rebuild + driver edit ·
hb1 host chain · mx1 renderer lift/port/parity · mx2 pose lift · landings + skip receipt).
Council perspectives applied: call-site tracing · provenance/calibration-lineage · assumption-
challenge (axis 8, mandatory) · measured-runnability/scored-quantity (axis 9) · phase/resume
paths · default-override hunting. Clean-pass counter: **0/3** (round found findings; fixed
in-round below). score_claim=false.

## Findings

**F1 — MEDIUM-HIGH (assumption axis): Row-1 token-source conflation.** The mx1 smoke config
(input=tq1c, target=GT) measures error-correction, not receiver capacity; feeding tq1c tokens
caps the vehicle at tq1c's own d_seg unless correction reach is separately proven. FIXED:
binding two-arm amendment (ARM-CAP GT→GT vs ARM-VEH tq1c→GT) appended to LAUNCH_TICKET.md —
ARM-CAP is the EH1 Row-1 discriminator; ARM-VEH prices the composed vehicle.

**F2 — MEDIUM (provenance): mx1 ticket args unverified at source.** I had closed calibration
lineage for hb1 (e2e.py:1220-1247) but not mx1. VERIFIED this round: 6000 steps / lr 2e-7 /
bits 4 / eval-every 250 = their stage-08 TAIL exactly (e2e.py:419-424), resumed from their
retained 12k QAT checkpoint — their-form-faithful. CLOSED.

**F3 — LOW (harvest hygiene): hb1 driver stages 3/4 are non-blocking on failure.** A pack or
decode failure logs rc but the chain continues; the race table MUST be read only after
checking the driver.log rc lines (exact bytes come from pack/encode receipts, never the
trainer's estimated_* fields). Recorded here as the harvest contract; no code change (failure
is visible, not silent).

**Axis-8 record:** the wave's operating assumption = "their recipe at their settings is the
optimal first measurement on our labels." Held legal as a RACE framing; the two named
residuals (horizon/λ not payload-optimal; token-source choice) are now explicit — F1 fixed,
horizon sweep pre-registered as the follow-up IF HPAC/receiver win their races.

**Axis-9 record:** et4 = running at full scale (the strongest runnability proof); hb1 =
resumable, live at epoch 6 with descending joint bytes (123,082 vs incumbent 142,001,
estimates); Row-1 = n32 bounded entry with Metal s/step measured at rung start + abort bar
(amendment). No unmeasured-scored-quantity claims found: all numbers this wave are labeled
estimated/external/advisory, exact bytes deferred to pack/decode/evaluate stages.

## Disposition

Round 1: 3 findings (0 CRITICAL / 2 MEDIUM / 1 LOW), all fixed or contracted in-round.
Counter resets to 0/3 per protocol. Round 2 = independent fresh-eyes codex adversary
(ddm_rr2) over the same scope; round 3 fires only after round 2 lands clean or its findings
are fixed.
