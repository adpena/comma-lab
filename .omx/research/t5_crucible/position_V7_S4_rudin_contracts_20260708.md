# Position V7_S4 — Rudin (gates/declarations as contracts, auditability)

STORES CONSULTED: CONVENING_T3_v7 (my contract) · DRAFT_v7_restart_config_synthesis_20260708 §1/§6 ·
crucible_v7_authored_20260708 (diff table + HONEST wiring-gap block) · witness_native_schedule_
derivation_20260709 (continuous verdict, my blind-safe items 1+§Phase-1) · CODE READ: witness_autoconfig.py
`_crucible_v7_schedule_governance` + `crucible_v7_wiring_gaps` (L1840–1892) · tools/schedule_provenance_gate.py
`RECOGNISED_EVENT_SENSORS` + classifier precedence (L60–231) · curriculum_dsl.py `TailCycles`/`LadderIsland
Homotopy` (L1636–1817) · witness_control/tail_cycles.py `step`/`TailStep.reason` (L128–208) · CLAUDE.md NO-FAKE
#1 + value-provenance ladder (L22) · MEMORY L22 (bare-literals bug class). [no-triality]

## The face question: can every trigger's firing be reconstructed post-hoc from telemetry ALONE?

Element-by-element, that is the explanation-as-contract bar, and it is MOSTLY MET:
- **3 fixed start-epochs (muon 726 / lane-band 500 / chroma 450):** fire at a CONSTANT epoch → perfectly
  reconstructable from the epoch counter alone. Auditability is not the defect here — the caps are the MOST
  auditable objects in the config.
- **TAIL cycles/stop:** `TailController.step` is a PURE function of the recorded verdict trace `rows=[(ep,d_seg)]`
  + config threshold; `TailStep.reason` stamps the fired branch AND the threshold (`"powerplay stop … marginal
  ΔS/ep < {stop_marginal_s}"`). Reconstructable from the persisted verdict trace. ONE gap: it stamps `< thr`, not
  the marginal NUMERATOR value — stamp the computed marginal too so a reader reconstructs the inequality, not just
  its outcome.
- **LADDER per-class births:** fire on λ_c (#315 per-class verdict) vs release r*(t)=coeff·σ_eff. Reconstructable
  IFF the per-class λ trace is persisted (perclass_verdict) — CONFIRM that stamp lands per epoch.

## The NO-FAKE line: is a tagged-CAP-with-owed-wiring an acceptable launch state?

YES — launch-with-caps, NOT build-wirings-first. Falling-rule:
1. IF class=`event` (claims runtime sensor-fired) AND the sensor does not move the trigger → FAKE (a claim the run
   cannot honor) → REFUSE. **This is not the case here.**
2. ELSE IF class=`cap` (fixed epoch, self-documenting) AND rationale names the sensor as a co-emitted BACKSTOP
   (not the firer) AND the gap is published in `crucible_v7_wiring_gaps()` → the run HONORS exactly what it
   declares (a fixed cap co-running a wired controller); firing is auditable from a constant. → **ACCEPT.**
The author took branch 2 correctly: class="cap" everywhere, rationale says "governing wired event … the specific
sensor→start wiring is an owed build", and all three gaps are enumerated verbatim. That is honest declaration,
not a fake event-trigger. An unwired sensor NAMED IN A CAP is a backstop reference, not an unhonored claim.

## The residual contract defect (why PROCEED_WITH_REVISIONS, not clean PROCEED)

The `ScheduleGovernance.sensor` field is MIS-READABLE: on a CAP it means "backstop I co-run with," but a reader
(or a future gate) can read `sensor:--curriculum-nucleus-guard` as "fired-by nucleus-guard" — the exact
event-vs-cap ambiguity the gate exists to kill. The gate today only checks the sensor is RECOGNISED + co-emitted;
it does NOT assert a CAP's sensor is documented-as-backstop. Interpretability-as-contract fix = add a `role`
discriminator (`fires` | `backstops`); gate asserts `cap.role=='backstops'`. Then a CAP structurally cannot be
misread as an event claim from the declaration alone.

## Position (council_pending knobs)

1. **Three event-sensor choices + caps 726/500/450** — ACCEPT as tagged CAPS (auditable, honest, gap-published).
   REVISION R1: add `role: backstops` to each CAP's ScheduleGovernance + a gate assertion, so the sensor field is
   un-misreadable. Launch-with-caps; the sensor→start wirings are a legitimately-deferred OWED build, not a launch
   blocker.
2. **TAIL k_max=2 / stop-marginal-s 1e-4** — ACCEPT k=2. REVISION R2: `stop_marginal_s=1e-4` is a BARE CONSTANT
   (value-provenance-ladder violation, req-T/L22) — TailCycles cites laws for cycle_floor/dwell/cycles_max but NOT
   for this stop; either LawRef it (attribution-floor derivation vs λ_bytes) or tag HARDCODED-WITH-WAIVER. Same for
   `tau_halving 0.5`. Also stamp the marginal numerator in `TailStep.reason` (telemetry-completeness).
3. **LADDER gate thresholds** — ACCEPT builder defaults. λ-gate 0.0=OPEN is auditable-neutral (self-documenting
   falling rule: "λ never binds; births are release-law-driven"). release_coeff 0.95 / sigma_eff 1.5 already claim
   LawRef — GOOD. Minor: the two λ-gate 0.0 literals should carry the same DERIVED/DEFAULTED tag as their siblings
   rather than sit bare. Recalibrating from run-1's per-class λ trace is moot (run-1 produced no trajectory).
4. **Structure round** — defer to S6 (binding blind seat); not my face.
5. **Run-1 stop point** — auditability-neutral; ACCEPT "first Muon-cap OR seal-complete, whichever first"
   (checkpoints preserved either way).

## Assumption tags (#363)
- "CAP firing is fully reconstructable from a constant epoch" — VERIFIED_VIA_SOURCE_INSPECTION (fixed gate).
- "TAIL stop is a pure function of persisted verdict rows" — VERIFIED_VIA_SOURCE_INSPECTION (tail_cycles.py step).
- "per-class λ trace is persisted so LADDER births are reconstructable" — ASSUMED_AWAITING_VERIFICATION (confirm
  the perclass_verdict stamp lands each epoch before relaunch).
- "sensor field on a CAP will not be misread downstream" — INFERRED_FROM_DOMAIN_LITERATURE → gated by R1.

## Verdict contribution
**PROCEED_WITH_REVISIONS.** Launch-with-caps is the correct state — the caps are honestly tagged, published in the
gap list, and MORE auditable (constant epochs) than the sensor-fired form would be; building the three wirings
first is NOT required and is genuinely-deferred. Revisions are declaration-layer only, cheap, and pre-launch:
R1 (CAP `role:backstops` discriminator + gate assertion, so an unwired sensor cannot read as a firing claim) and
R2 (stop-marginal-s / tau-halving / λ-gate 0.0 carry a value-provenance tag, not bare literals; stamp the TAIL
marginal numerator). Pointer 0.19110 UNMOVED — MEANS until the byte-closed n600 row.
