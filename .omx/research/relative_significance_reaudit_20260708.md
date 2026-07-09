# Relative-significance RE-AUDIT (2026-07-08) — every "too small / weak / negligible / noise" dismissal, re-ranked at the sub-0.15 operating point

**Operator directive (RECURRING correction):** *"there are OTHER things you orphaned or deferred
because of negligible impact or noise but that was really significant in RELATIVE terms if not in
absolute terms."* Anchor lesson:
`relative-not-absolute-significance-near-goal-dont-orphan-small-deltaS_20260708`.

**Pointer 0.19110 UNMOVED** (read-only research + re-ranking sweep — nothing built, nothing launched,
live #205 run untouched).

## The metric (so the verdict is not eyeball)

S is LINEAR in each term (`100·d_seg + √(10·d_pose) + 25·bytes/37.5M`), so a fixed ΔS is a fixed
Δcomponent at ANY operating point. What changes near the goal is the DENOMINATOR you judge it against.
Two canonical denominators, both reported below:

- **rel-sig(S) = ΔS / (S_current − S_target)** = ΔS / (0.19110 − 0.15) = **ΔS / 0.0411** = fraction of
  the remaining descent to sub-0.15 this lever buys (pointer-anchored).
- **rel-sig(d_seg) = Δd_seg / target_d_seg** = Δd_seg / **0.0009** = the operator's own trigger framing
  (frontier seg-term is d_seg 0.00056; a competitive witness needs ~0.0009-class d_seg). A d_seg lever
  measured at Δd_seg 0.0001 is ~11% of the *entire* target d_seg budget — not "polish."

Frontier decomposition (Lane A, FEED-bh, harness-validated to 8 decimals): d_seg 0.00056 (term 0.05598)
+ d_pose 0.0000294 (0.01715) + rate 0.004719 (0.11797) = S 0.19110. Rate is DEAD (entropy floor);
pose is BANKED (R1 dxi shippable, √(10·d_pose)=0.106–0.127, FEED-238resolved). **d_seg is the entire
remaining fight** → every d_seg lever's relative significance is now maximal.

---

## RE-OPEN — dismissed on ABSOLUTE magnitude, relatively significant near the goal (ranked, highest first)

| rank | lever | ΔS (label) | dismissal-as-written | rel-sig(S)=ΔS/0.0411 | rel-sig(d_seg)=Δd_seg/0.0009 | action |
|---|---|---|---|---|---|---|
| 1 | **#169 horizon-weighted margin (0-byte IN-TRAINING)** | ΔS **0.012–0.024** (MEASURED cap; Δd_seg 0.00012–0.00024) | I called it "weak" (the trigger); Lane-A wrote the horizon *oracle* as "video-derived → dead-rate NO-GO" | **29–58%** | **13–27%** | BUILD it. In v7.5 plan item B but FEED-v75Aactuated flags "B.5 horizon-margin #169 NOT a built trainer flag — GAP." Land the DSL `Lever` + trainer flag; the orphan is a *missing wire*, not a dead lever. |
| 2 | **#121 d_seg-aware taper** | ΔS **~0.03** if the sign holds (ESTIMATED; −8% of witness d_seg 0.0045 ≈ Δd_seg 0.00036) | recorded "+18% NO-GO" | **~70% (ESTIMATED)** | **~40% (ESTIMATED)** | RE-VALIDATE at convergence (cheap disk A/B on a converged ckpt). NO-GO is SUSPECT/RETRACTED: +18% was ge300/3000 under-converged; converged anchors FLIP the sign to −8% (may HELP). verdict_scope: instance-under-convergence — sign is UNMEASURED-at-convergence. |
| 3 | **activation: step-native / FINER++ vs sine** | ΔS **~0.013** (MEASURED FINER −4.5% n600; Δd_seg ~0.00013) | "modest / weaker at scale" (was −18.7% n100 → −4.5% n600) | **~32%** | **~14%** | ADOPT the winner: the step-native screen (hosc-b4/b8/step_basis-k8 arms) is LIVE; the adopt-verdict is still OWED. "Modest" ≠ orphan near-goal — land the screen verdict and set the generator chart. |
| 4 | **#274 seg down-weight lever** | ΔS not-yet-measured-at-optimal (BUILT, the standing seg play) | held as "the seg play" while attention went to rate/pose | — (measure owed) | — | MEASURE at optimal form inside v7.5; ensure it's in the optimal-combination set, not left default-off. |
| 5 | **D18 latent-table TRUNCATE-at-export (k90 free rate)** | rate ΔS ESTIMATED (`k90_truncate_bytes_estimate` emitted per verdict) | rate treated as "DEAD" globally | small but PURE-S (rate term 0.118) | n/a (rate) | Run the truncate A/B at run-1 stop (ARMED, sensor landed). Near-goal, any real byte cut is undiluted S — "rate is dead" is a floor claim about the *representation*, not a licence to skip a measured free-byte cut. |
| 6 | **19-neutrality mod-32 rate-saving A/B** | rate ΔS small (MEASURED-pending) | "non-blocking, open" | small, PURE-S | n/a (rate) | Non-blocking ≠ never. Fold into the stop-time byte-close A/B alongside D18. |

**Honest read:** the STRONG relative-significance re-opens are **#1 (#169 horizon-margin) and #2 (#121
taper)** — both are d_seg levers I dismissed on absolute magnitude that clear ~30–70% of the remaining
descent. #3 (activation) is real but already has a live screen. #4–#6 are near-goal-relevant tidy-ups
that "absolute-small" reasoning would have skipped. Most of the operator's other named levers turned
out already-folded (see SUPERSEDED) or measured-un-recoverable (see below) — the discipline worked
*except* on the eyeball-"weak" cases, which is exactly the recurring bug.

---

## GENUINELY-UN-RECOVERABLE — deferred because MEASURED un-achievable, NOT because "too small" (stay deferred)

These have a MEASUREMENT and an exit/reactivation criterion; the ΔS is UNREACHABLE, not un-worth-it.

- **#307 contour-string flip coding** — MEASURED NO-GO n600: **0.820 B/flip > 0.65 GO bar** (mod32cap
  ep425, all 600 pairs, decode-verified). Residual is fragmented confetti (mean 3.1 px comps, 44.6%
  singletons); reaching 0.65 needs ~3× larger mean component ⇒ **coherence is a TRAINING outcome, not
  a coder trick**. Reactivation: re-measure after a coherence-improving training arm. *Measured, legit.*
- **#139 hood static clamp (DECODE-SIDE)** — MEASURED ~0: "19 flips in 25% of frame, clamp saves ~0."
  The hood interior is already correct; there is no gain to recover. *(Distinct from #139-as-static-
  COMPONENT, which is SUPERSEDED — promoted into the level-set as a 0-byte structure that frees
  capacity/rate.)*
- **Horizon-BAND capacity chase** — MEASURED NO-GO: flip-matrix rank **547/600** = high-entropy /
  label-noise-like; spending capacity chasing horizon-band flips is NEGATIVE-EV. *(Distinct from the
  0-byte horizon-MARGIN weighting, RE-OPEN #1 — different lever, do not conflate.)*
- **Generic 0-byte decode-side perturbations on the frontier inflate.py** (upsample-kernel swap, pre-
  sharpen, PR98 ±1) — MEASURED HURT (corrected-R CPU-torch): the frontier decoder is trained
  end-to-end THROUGH the exact R → sits at a trained optimum → generic perturbations move AWAY.
  *Measured-negative, not small.*

### "label-noise / noise" dismissals — do any LACK a measurement? (the trap to catch)

Audited: every "noise / high-entropy / flicker" dismissal in scope carries a measurement — horizon-band
rank 547/600 (measured flip-matrix rank), #205 CE-residual "44% spikes = LANE / temporal flicker"
(measured), comma10k 4.67% lane-edge residual "GENUINE" (measured same-rig). **No un-measured
label-noise kill found in scope.** One nuance worth flagging: the #205 CE temporal-flicker residual was
treated as a floor, but FEED-undrivrecall MEASURED it is **pose-explainable inter-frame jitter,
removable by the temporal-screw #360** (clean-canonical floor 0.0016 vs live 0.082) — so "flicker" is
NOT un-recoverable; it is folded into the horizon fix (SUPERSEDED), not deferred.

---

## STRUCTURALLY-SUPERSEDED — already folded into v7.5 actuation or a live lever (no action)

- **#291 lane paint-then-SDF** — actuated v7.5 item A.1 (`--lane-prior-phi1-mode replace`→`paint`);
  MEASURED lane FN 0.00713→0.00211 (~3×). FEED-v75Aactuated.
- **#287 along-tangent dash comb** — actuated v7.5 item A.2 (`DashComb` factory, `--lane-band-dash-comb`);
  oracle 0.00695; operator overrode the defer, DEFAULT-ON.
- **#149 sub-pixel — IN-TRAINING survival-wall reduction** — folded into the step-native activation
  screen (step_basis lowers the ~16% round-trip survival-wall). *(Sub-pixel-as-sidecar = dead rate,
  correctly dropped.)*
- **#139 hood static — as level-set COMPONENT** — promoted to the level-set static core (0-byte).
- **#360 temporal-screw + sky=rotation-only** — the horizon fix (Undriv 0.082→floor 0.0016); v7.5 item B.
- **#128 / PR98 offset / FECa / DQS1** — NOT orphaned: MEASURED to be ALREADY IN the 0.19110 frontier
  inflate.py (FEED-bh REFUTED the "cheap levers never byte-closed" hypothesis). Banked, not deferred.

---

## STRUCTURAL FOLD — make the apparatus compute relative significance (so this stops recurring)

**Root cause of the recurrence:** `duty_to_measure()` in
`src/tac/witness_dsl/activation_ledger.py` (L187) ranks owed levers by **state then alphabetical name**
— it has NO value axis, so "which owed lever matters most" falls back to my eyeball, which keeps
anchoring on ABSOLUTE ΔS. Levers carry no ΔS estimate anywhere (confirmed: `Lever`/lever_registry have
flags + composability only).

**The fold (Results-must-become-system-intelligence + default-off-is-orphaned-signal):**

1. **New canonical store** `.omx/state/lever_relative_significance.jsonl` — one append-only row per
   lever/finding: `{lever, est_delta_s, label: MEASURED|ESTIMATED, source_anchor, ts}`, populated from
   the measured DAG/equation anchors (seed it from the RE-OPEN table above). This is the missing
   ΔS field; keep it beside the activation ledger, not a parallel registry.
2. **New function** `duty_to_measure_ranked(s_current, s_target=0.15, path=None)` in
   `activation_ledger.py` — joins `duty_to_measure()` against that store, computes
   `rel_sig = est_delta_s / (s_current − s_target)` (reading `s_current` from
   `.omx/state/canonical_frontier_pointer.json`), and returns rows sorted by rel_sig **descending**
   (never-fired ties broken by rel_sig, not by name). Levers with no ΔS row sort as `unknown` and are
   surfaced as a duty-to-*estimate* queue (an un-estimated lever is itself orphaned signal).
3. **Consumer** `tools/costate_digest.py::section_duty_to_measure()` (L168) — print the rel_sig number
   next to each lever so the session-start digest ranks by fraction-of-remaining-descent, not
   alphabetically. This is the SENSE-layer change that makes the controller (not the operator) hold the
   ranking.

Net: the duty-to-measure queue becomes value-ranked by ΔS/remaining-gap, computed, surfaced, and
append-only — the eyeball is removed from the loop.

**Triality:** DAG FEED-relsig (this re-audit + the recurring-lesson anchor) · DSL leg = the
`duty_to_measure_ranked` fold (owed build, not done here — this is a read-only sweep) · equations leg =
the est_delta_s store is the measured-anchor join (no new equation; it consumes existing anchors).

**Pointer 0.19110 UNMOVED.**
