# POSITION — SEAT S5 (Assumption-Adversary) — T3 v7 INCLUSION SYMPOSIUM

BLIND. I did NOT read any `position_INCL_S*.md`. Stores consulted: CONVENING docket +
DOCKET STATUS UPDATE · ORCHESTRATION_LEDGER (last ~250 lines + wave-1/R-7 folds) ·
`v7_compute_exploitation_audit_20260708.md` (lever table + wall-clock) · the LIVE run-1
argv/log (`levelset_n600_crucible_v6_run1_20260708T095730Z`, pid 63069) · operating manual.
Pointer contest-CPU **0.19110 UNMOVED** — everything below is MEANS; only a byte-closed n600
`upstream/evaluate.py` row < 0.19110 moves it. Items 1–2 are OPERATOR-DECIDED; I do not
challenge the decisions, only the assumptions their EXECUTION rides on.

## MY OPERATING-WITHIN ASSUMPTION (surfaced first, per the seat contract)
I am operating within the assumption that **the inclusion classes are being assigned per-item,
against the SEALED config, on the honest live budget** — i.e., that "is this affordable / safe /
evidenced" is answered against the run that will actually launch, not against the pre-fold audit
narrative or an optimistic anchor. If that assumption is false (e.g., classes are being read off
the docket's expected-class column without re-deriving against the live argv), then several of my
verdicts below flip, and that itself is the finding. I flag where I caught the docket reasoning
off a premise the live config contradicts.

---

## THE BACKDROP — shared assumptions the whole exercise rides on

| # | Shared assumption | Class | #363 tag | Note |
|---|---|---|---|---|
| A1 | "default-OFF + duty-to-measure is the SAFE class" | **CARGO-CULTED** (for trajectory-affecting speed levers behind a long baseline) | app'n = ASSUMED_AWAITING_VERIFICATION | HARD-EARNED as the general no-untested-trajectory-change doctrine; CARGO-CULTED when it silently prices deferral at ~5 days of wall and the A/B comparator is the very baseline you're paying for. |
| A2 | "the 3.1 min/ep anchor" (⇒ 6.5-day budget) | **CARGO-CULTED (optimistic)** | VERIFIED_VIA_EMPIRICAL_ANCHOR (live > 3.1) | I MEASURED live 3.62 min/ep incl-startup @ep79 → 7.55 d; sealed rc=8 budget = 7.427 d (=3.565 min/ep); the ledger's own cadence check measured 4.2–4.5 under fleet contention. 3.1 is BELOW all three. Contention is the operating mode, not an anomaly. |
| A3 | "bit-identical ⇒ safe to include" | **HALF** — HARD-EARNED at single-call scale, CARGO-CULTED at COMPOSITION scale | single-call = VERIFIED_VIA_EMPIRICAL_ANCHOR; composition = ASSUMED_AWAITING_VERIFICATION | Parity measured per-kernel (D16, #330, fused-R, cache-gt). NOT measured for: crash-during-subprocess-verdict × resume; resume bit-identity of the 4 NON-gate controllers (`_rng_`/`_cl_`/`tau_advance_`/`_evt_`); and `--skip-admission-gate` softens "safeguarded". |
| A4 | "IN-v7 / v7.1-ARM / REGISTERED is the complete ontology" | **mostly HARD-EARNED; one gap** | INFERRED_FROM_DOMAIN_LITERATURE | A real FOURTH class exists: **IN-v7-with-bounded-auto-revert** (default-OFF-byte-identical AND self-guarding: safe-compile per-chip fingerprint; #330 subprocess fallback). The 3-class ontology forces self-guarding score-neutral levers into "off until A/B," which is deferral-momentum. |
| A5 | "each item's class can be set in isolation from the LIVE sealed argv" | **CARGO-CULTED** | VERIFIED_VIA_SOURCE_INSPECTION | The docket set item 5 (D16) to REGISTERED "because the consuming loss term is default-off in v7." The LIVE argv carries `--persistence-loss-weight 1.0` and the log shows `persistence: 0.37` in loss_terms — the term is **ACTIVE**. The class was reasoned off a stale premise. |

**The two momentum poles I am testing every item against** (per the mandate): *it-was-built-today-
so-include-it* (inclusion-momentum) AND *it-was-built-today-so-defer-it-to-be-safe*
(deferral-momentum). Deferral-momentum is the subtler one here and it is where the exercise is
most exposed — "we just built it, park it default-OFF, A/B later" FEELS rigorous while quietly
costing a week of wall and leaving self-guarding score-neutral levers on the shelf.

---

## PER-ITEM VERDICT (class + is-it-derived-or-momentum)

**1. DirectionalBasisRebalance — IN-v7 (DECIDED; not challenged).** Execution flag only: the
waterfill "admitted" peak is **71.54 GiB** but the live `safe_run` carries `--projected-gib 67.61
--skip-admission-gate`. The 4-GiB discrepancy is harmless on a 128 GiB sole-workload box, but the
admission projection is NOT the same number the waterfill certified, and the gate is SKIPPED — so
"safeguarded @ 71.54" reads stronger than the live posture. DERIVED (operator + waterfill), one
provenance seam noted.

**2. --tau-advance-mode EVENT — DECIDED (operator, EVENT, risk accepted).** Not challenged.
Advisory: the telemetry-on-the-wiring requirement (per the docket) is the correct locus.

**3. Micro-batch-pairs — my class: v7.1-ARM *only with a NAMED BOUNDED A/B*; else FOURTH-class
(IN-v7-eligible-pending-bounded-recess). NOT baseline-gated.** This is my headline. The docket's
"v7.1-ARM, n600 trajectory A/B" is **deferral-momentum IF the A/B is scheduled after the full
baseline** — because the baseline IS the 7.55-day slow run, and micro-batch is the 2–4× lever
that would fix exactly that slowness (6.5→~2–3 d per the compute audit). Deferring the fix to get
a clean baseline, when the baseline costs the un-accelerated wall the fix would remove, is the
circular argument the mandate names. **HONEST COUNTERWEIGHT (this is why it is not a clean
IN-v7):** micro-batch>1 is *trainer-refused* while `--logit-adjust-loss-tau 1.0` is active
(live argv confirms it IS active), and routing logit-adjust into the batched twin is an UNBUILT
item, and batched-fp reduction is genuinely trajectory-affecting. So the block is real, not
cosmetic. **The derived resolution:** the A/B must be a **bounded empirical recess** — a short
n600 d_seg A/B to the first curriculum landmark (~ep300–350), not a full 3000-ep baseline — that
measures whether batched-fp reduction is d_seg-neutral once the twin routes ALL active v7 losses.
Bounded-neutral ⇒ IN-v7 (cut the run to ~2–3 d). Not-neutral ⇒ clean v7.1. Class = v7.1-ARM with
the A/B EXPLICITLY bounded and NOT gated on baseline completion.

**4. Safe-compile certified regions — v7.1-ARM (accept).** Default-OFF byte-identical (0.0),
per-chip fingerprint fail-closed. Derived deferral. Flag: "certified 1.41× on the hosc `_act`
region ⇒ worth flipping" assumes the region is a material fraction of the whole step — UNMEASURED
(whole-step bench owed, correctly named as the run-1 stop-checklist gate). HARD-EARNED on
bit-identity; step-fraction-materiality = CARGO-CULTED-until-benched. Candidate for the FOURTH
class (its per-chip fingerprint IS an auto-revert) but the missing whole-step number keeps it at
v7.1-ARM honestly.

**5. D16 Metal kernels — my class: IN-v7 CANDIDATE (re-derive vs live argv). NOT REGISTERED.**
The docket's REGISTERED rests on "the consuming loss term is default-off in v7" — **FALSIFIED by
the live config**: `--persistence-loss-weight 1.0`, `--cldice-iters 5`, `--persistence-recall-
weight 1.0`, and the log shows `persistence: 0.37` every step. The persistence/clDice pool path
is ACTIVE. D16 is bit-identical (max|Δ|=0 vs numpy authority, N=5 cross-process deterministic)
and delivers a MEASURED speedup on that active hot path. That is the SAME class as fused-R and
#330 (score-neutral speed on an active path) — which are IN-v7. Synthesis MUST verify D16 is
wired to the live persistence path AND that bit-identity holds in-composition; if both hold, D16
is IN-v7, not shelved. This is a momentum-of-stale-reasoning catch (A5).

**6. #330 verdict reclaim — IN-v7 (accept the class; correct the rationale + flag one seam).**
Score-neutral by single-verdict bit-identity; subprocess parent +0.0 vs +4.6 GiB ratchet. Accept
IN-v7. But the docket rationale "the memory it frees funds the basis raise" is **weak/circular**:
the basis raise already sealed at peak 71.54 with `--skip-admission-gate`, so nothing was gated on
the freed memory. #330's REAL value is **ratchet/paging avoidance** over 7.5 days (the cadence
check saw RSS page 18→7.2 GiB under contention — that is what #330 prevents), NOT funding the
basis raise. Seam (A3): bit-identity was measured for a CLEAN verdict; the ~7 GiB transient npz
SSD-hop × crash-during-subprocess × resume is UNTESTED. Accept IN-v7; owe a crash-composition
parity test.

**7. Adaptive-ε (#320) — REGISTERED-duty-to-measure (accept; the CORRECT use of the class).**
This is the clean contrast to A1: its A/B never ran because v6's λ=0.01 redesign structurally
removed its failure mode (eikonal stable ~0.009 in the LIVE log, no re-entry). REGISTERED with a
real telemetry trigger (eikonal re-entry signature) is dormant-not-orphaned — derived. Equation
`adaptive_eps_cfl_edge_tracking_v1` stays ASSUMED_AWAITING_VERIFICATION, correctly.

**8. R-7 finishers — SPLIT.** Rewarmup: the BASE mechanism is ALREADY IN-v7 (live argv:
`--stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-floor 0.1`); R-7 only makes the
window DERIVED-AT-CONFIG (β2-horizon), and that β2 law is INFERRED/PROVISIONAL → **v7.1-ARM** for
the derived version. Polyak finisher: an ADDITIONAL stop-time checkpoint candidate (EMA never
replaced; byte-close picks the winner) → **REGISTERED-duty-to-measure** (stop-time). Flag the
named footgun: `start_epoch=0` arms Polyak from run START — a live default that must be sized
before any flip. Derived deferral (sizing INFERRED), accept the split.

**9. Resume registry — IN-v7 (hardening) CONTINGENT on #358 + one UNTESTED seam.** Correct as
crash-resume correctness (16 tests incl. crash-resume bit-identity, fail-closed
ResumeIntegrityError). BUT the crash-resume bit-identity test covers **GATE controllers only**;
the four NON-gate controllers (`_rng_`/`_cl_`/`tau_advance_`/`_evt_`) are asserted-persisted but
their crash-resume bit-identity is NOT in the tested set — `_rng_` especially (a divergent RNG
stream on resume ⇒ non-bit-identical continuation, invisible until a byte-close mismatch). Task
#358 brings them under the static gate and "lands BEFORE seal round 2." So IN-v7 is CONTINGENT: if
#358 slips, the resume path ships with a known unguarded seam and the "crash-resume is
bit-identical" claim is HARD-EARNED for gates, ASSUMED for the four controllers (A3).

**10. GPU-verdict hybrid — REGISTERED (accept).** MLX/MPS NEVER a score (L53); a GPU verdict
gated on a CPU-agreement probe is the only compliant path. Derived. Accept.

**11. fp16 cf-feats — REGISTERED (accept; enforce consistency with item 6).** Memory-save, not
wall-clock; low urgency on a 128 GiB sole-workload box. But the SAME "ample memory" premise that
makes fp16-feats low-value ALSO undercuts item 6's "frees memory for the basis raise" rationale.
The council must price the memory envelope CONSISTENTLY across 6 and 11: memory headroom is ample
(→ both memory levers low-urgency), so #330 earns IN-v7 on RATCHET-avoidance, not headroom, and
fp16-feats is correctly REGISTERED.

---

## VETO — the violation hypotheses the synthesis MUST engage (or I withhold consensus)

1. **Circular-baseline / bounded-recess (item 3, A1).** The synthesis MUST either (a) name a
   BOUNDED n600 d_seg A/B for micro-batch (short, to ~ep300–350, twin routes all active losses)
   that does NOT wait on the full 7.55-day baseline, OR (b) explicitly justify why no micro-batch
   measurement is possible before the full baseline. A bare "v7.1-ARM, A/B later" that does not
   engage the circularity is a VETO trigger — it prices a week of wall as if free.

2. **The 3.1-anchor is optimistic (A2).** Any inclusion set justified as "affordable in the
   budget" MUST state the budget against the LIVE cadence (≥3.6 min/ep, ~7.55 d, contention-
   inclusive — I measured it this session), not the 3.1 anchor. The campaign's own continuous
   apparatus operation (this 5-agent symposium included) is what makes the machine contended;
   "clears as agents land" is a HOPE, not a measurement, and there will be agents during the run.

3. **D16's class is set off a falsified premise (A5).** The synthesis MUST re-derive D16 against
   the live argv (`--persistence-loss-weight 1.0`, persistence term ACTIVE), not the "term off"
   audit line. If D16 is wired to the active persistence path and bit-identity holds
   in-composition, it is IN-v7, not REGISTERED. Shelving an active-path score-neutral speed win is
   deferral-momentum.

4. **(secondary) Composition-bit-identity seam (A3).** Before items 6 and 9 are load-bearing
   IN-v7, a crash-resume/subprocess bit-identity test must exercise the WHOLE included set
   (subprocess verdict mid-write + the four non-gate controllers, `_rng_` foremost). Single-call
   parity does not license composition parity.

I also propose the council adopt the **FOURTH class** (IN-v7-with-bounded-auto-revert) for
self-guarding, default-OFF, byte-identical levers (safe-compile, #330), so the ontology stops
forcing self-protecting score-neutral work into "off until A/B."

---

## ONE-LINE CLASSES (my seat)
1. Basis rebalance — **IN-v7** (DECIDED; flag 67.61-vs-71.54 + skip-admission softness)
2. Event-vs-clock — **DECIDED-EVENT** (not challenged)
3. Micro-batch — **v7.1-ARM with NAMED BOUNDED A/B, NOT baseline-gated** (VETO #1)
4. Safe-compile — **v7.1-ARM** (whole-step bench owed; FOURTH-class-eligible)
5. D16 kernels — **IN-v7 candidate** (re-derive vs live argv; premise falsified; VETO #3)
6. #330 reclaim — **IN-v7** (rationale = ratchet-avoidance not basis-funding; crash-seam owed)
7. Adaptive-ε — **REGISTERED-duty-to-measure** (correct dormant-not-orphan)
8. R-7 finishers — **rewarmup v7.1-ARM / Polyak REGISTERED** (base rewarmup already IN; size start_epoch)
9. Resume registry — **IN-v7 CONTINGENT on #358** (`_rng_` crash-resume UNTESTED; VETO #4)
10. GPU-verdict hybrid — **REGISTERED** (accept)
11. fp16 cf-feats — **REGISTERED** (enforce memory-envelope consistency with item 6)

Pointer 0.19110 UNMOVED. All of the above is MEANS; the number moves only through a byte-closed
n600 exact row.
