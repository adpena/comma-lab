# POSITION — SEAT S2 (Dykstra: composition / feasibility lens) — T3 v7 INCLUSION SYMPOSIUM

BLIND. I did not read any position_INCL_S*.md. Pointer contest-CPU **0.19110 UNMOVED** — everything
below is APPARATUS/MEANS; certifying feasibility does NOT move the exact score. Only a byte-closed
n600 `upstream/evaluate.py` row < 0.19110 does.

## Operating-within assumption (stated per Council-conduct Fix-7)
My lens is the CONVEX-FEASIBILITY intersection of the COMPOSED lever set, not any item in isolation.
The assumption I operate within: **"a set of individually-landed, individually-byte-identical-when-off
levers composes feasibly iff no pair shares a live actuator surface (τ / LR / the resume sidecar / the
memory envelope) in the same epoch without an explicit ownership guard, and iff every schedule anchor
survives event-mode's fire-vs-cap ambiguity."** The Assumption-Adversary should test whether I have
cargo-culted "landed + tested ⇒ composes" — I have tried to defeat that by tracing each shared surface
to source (file:line) rather than trusting the landing memos.

Items 1 (basis) + 2 (event mode) are OPERATOR-DECIDED. I certify the set they anchor; I do not
relitigate them. I DO certify the wiring feasibility ON them (my remit per docket line 54).

---

## PER-ITEM CLASS (feasibility lens)

| # | Item | S2 class | Why (feasibility) |
|---|------|----------|-------------------|
| 3 | Micro-batch-pairs | **v7.1-ARM** | Trajectory-affecting; waterfill-B pinned-1 UNMEASURED. `offset None ⇒ byte-identical` (bd6219a0a) so it is safe-to-hold-off; as an ARM it puts ZERO burden on the launch. Its bit-exactness × `unify_tau` is a v7.1 A/B question, not a v7-launch question. |
| 4 | Safe-compile regions | **v7.1-ARM** | Default-OFF byte-identical 0.0 verified (`_act` flip); evidence gate = the run-1 stop-time GPU re-cert + whole-step bench (D17). No launch interaction while off. |
| 5 | D16 Metal kernels | **REGISTERED-duty-to-measure** | Bit-identical, 2.3–3.9×, but the CONSUMING loss term (margin-map / curvelet / soft_skeleton) is default-off in v7 ⇒ NO hot path in the sealed launch ⇒ nothing to A/B. Register with trigger = "consuming term armed". (Prefer REGISTERED over v7.1-ARM precisely because there is no armed consumer in v7.) |
| 6 | #330 verdict reclaim | **REGISTERED-duty-to-measure** (I DISSENT from the docket's "candidate for IN-v7") | See COND-4. The sealed launch is cap-only, in-process verdict, mem-preflight 67.61 GiB PASS; the basis raise's 71.54 GiB peak was WATERFILLED without relying on subprocess reclaim. The verdict transient is a BOUNDED per-epoch high-water (chunked at `--verdict-batch`), not a monotone ratchet, so the preflight peak already captures it. Turning subprocess ON adds a ~7 GiB SSD-hop transient + a child-crash surface the launch does not need. Trigger = observed RSS growth beyond the preflight projection in v7 telemetry. |
| 7 | Adaptive-ε | **REGISTERED-duty-to-measure** | Its failure mode (eikonal re-entry) is structurally ABSENT at the sealed λ=0.01 fixed; default-off; A/B never ran. Trigger = eikonal re-entry signature. Concur with the orchestrator recommendation. |
| 8 | R-7 finishers | **SPLIT** — β2-window rewarmup → **v7.1-ARM**; Polyak finisher → **IN-v7-eligible IFF start_epoch is sized, else REGISTERED** | See COND-2/COND-3. β2-rewarmup is trajectory-affecting AND its sizing law is INFERRED/PROVISIONAL ⇒ ARM with A/B gate. Polyak `observe()` only READS live weights + writes its OWN sidecar keys ⇒ SCORE-NEUTRAL to the trained trajectory ⇒ safe to arm as a free extra candidate, but only useful if `start_epoch` is set to the finishing window. |
| 9 | Resume registry (event fired-state) | **IN-v7 (binding PRECONDITION)** | Not merely "hardening" — it is a HARD DEPENDENCY of item 2. Event mode without it = a crash between an event-fire and the next checkpoint restores a fired gate to OFF (a config that never existed). Landed + `test_crash_resume_all_gates_bit_identical_to_uninterrupted` + `test_vanished_event_state_fails_closed` + `test_all_cap_only_registry_emits_nothing_byte_identical` PASS. Certified. |
| 10 | GPU-verdict hybrid | **REGISTERED-duty-to-measure** | Sealed verdict is CPU-torch (authority-anchored); GPU path is fast advisory, NON-PROMOTABLE, gated on the D1 stop-time agreement probe; default = cpu ⇒ byte-identical to #205. |
| 11 | fp16 cf-feats | **REGISTERED pending fresh waterfill** | See COND-5. Competes with the basis raise for the SAME already-consumed envelope (basis took +3.93 GiB → peak 71.54 GiB admitted). The launch fits at 67.61 GiB WITHOUT fp16-feats; composing it needs a re-run mem-preflight proving post-basis + fp16 peak < 0.70×RAM (89.6 GiB on 128 GB). Do not compose blind. |

---

## THE INTERACTION MATRIX (the COMPOSED IN-v7 set)

**IN-v7 set certified for feasibility:** 4 DSL levers {unify_tau, TAIL_k, LADDER, basis} × 3 event
wirings {muon, lane_band, seg_chroma_boundary} × item 9 (resume registry) × item 8-Polyak
(conditional). All of items 3,4,5,6,7,8-β2,10,11 enter DEFAULT-OFF (ARM/REGISTERED) ⇒ byte-identical
to the trained trajectory ⇒ zero launch-interaction burden by construction.

Pairwise interactions traced to source:

**A. LADDER × muon-EVENT stagger — THE composition question of the event decision. FEASIBLE ✓**
Two independent layers enforce `max(ladder arm windows) < muon_entry`:
- STATIC (REV-A): `curriculum_dsl.py:1461-1477` — `ladder_muon_stagger_violation(...muon_start_epoch=cap)`
  refuses a config where an arm window ≥ the muon CAP. `[VERIFIED_VIA_SOURCE_INSPECTION]`
- RUNTIME (REV-B): the muon event CANNOT fire before nucleation completes. `event_wirings.py:290`
  `fired = meat_exhausted AND nucleation_complete`, and `nucleation_complete =
  ladder_arms_complete(ep, arm_windows)` (`:266`, `all(ep >= w)`). Trainer wires it at
  `train_…mlx.py:7372-7374` with `_ladder_arm_windows` built from the SAME birth+hold+anneal sums
  (`:5206-5213`) the DSL validates. `[VERIFIED_VIA_SOURCE_INSPECTION]`
  ⇒ Even though `muon_start` is a SENSOR fire in event mode, the sensor's OWN gate carries the
  nucleation predicate, so `muon_fire_epoch ≥ max(arm windows)` by construction. The stagger
  invariant SURVIVES event mode. This is the load-bearing certification.

**B. TAIL_k × muon-EVENT — FEASIBLE ✓ (with a named schedule-anchor note)**
`_tail_start_epoch = muon_start_epoch(CAP) + tail_dwell_min` (`:7010-7011`); TAIL gated on
`muon_switched AND ep >= _tail_start_epoch` (`:7726`). The τ double-driver is structurally guarded:
`assert _tau_ctrl is None or _tau_ctrl.frozen` (`:7731`) — the event-mode τ-advance ladder freezes AT
the muon switch, which strictly precedes `_tail_start_epoch`. `[VERIFIED_VIA_SOURCE_INSPECTION]`
NAMED NOTE (feasible, not a blocker): under an EARLY event fire (`ep_fire < cap`), the Muon-only
preamble [`ep_fire`, `cap+dwell_min`] STRETCHES; TAIL cycles still occupy [`cap+dwell_min`, `epochs`]
identically. This is a DELIBERATE resume-determinism choice (schedule geometry anchored on the cap so a
resume rebuilds it) — internally consistent, no double-driver, no crash.

**C. unify_tau × LADDER × TAIL — FEASIBLE ✓ (sequential τ ownership)**
unify_tau makes τ live/render-coupled by-ref for the PRE-Muon phase (D15: "missing callable RAISES" —
the latent silent-CE is killed). The τ-advance ladder owns τ until the muon switch freezes it; TAIL is
the SOLE τ driver thereafter (guard B). No epoch has two τ writers. `[VERIFIED_VIA_SOURCE_INSPECTION]`

**D. basis × memory envelope — FEASIBLE ✓**
Waterfilled: peak 71.54 GiB ADMITTED both envelopes (docket line 51); sealed cap-only mem-preflight
67.61 GiB PASS (ledger v6.2 dry-run) — under the 0.70×RAM = 89.6 GiB refuse line
(`witness_memory_preflight.py:54 DEFAULT_SAFE_FRAC=0.70`). `[VERIFIED_VIA_EMPIRICAL_ANCHOR: dry-run
mem-preflight]`. This is why items 6 and 11 must stay OFF (COND-4/COND-5): they would re-open a
settled waterfill.

**E. resume registry × 3 event gates — FEASIBLE ✓ (and the enabler of B/A under crash)**
`resume_registry.py`: symmetric write/restore iteration (`:159-252`), manifest lists ONLY controllers
that wrote keys, VANISHED event-gate key ⇒ `ResumeIntegrityError` fail-closed (`:239-241`), cap-only ⇒
`{}` no-manifest byte-identical (`:181-182`). The atomic sidecar write (`_atomic_savez` tmp+os.replace,
`train_…mlx.py:463-476`) guarantees you get either the old OR the new sidecar, never a truncated one —
so `ResumeIntegrityError` fires only on genuine external corruption (correct fail-closed).
`[VERIFIED_VIA_SOURCE_INSPECTION + test_crash_resume_all_gates_bit_identical_to_uninterrupted]`

**F. Polyak accumulator × per-stage checkpoints (the atomic-savez claim) — FEASIBLE ✓, source-verified**
The scalar sentinel (count/start/arm) is emitted through the REGISTRY
(`resume_registry.state_arrays()` merged at `:698`) and the HEAVY fp64 running-mean at
`train_…mlx.py:6062` (`_polyak.heavy_state_arrays`) — BOTH land in the SAME `resume_arrays` dict → ONE
`_atomic_savez(levelset_resume_state.npz, resume_arrays)` (`:6065`). No `observe()` runs between the two
emissions (observe is at `:8317`, epoch-end, outside the checkpoint closure) ⇒ count and mean are a
CONSISTENT snapshot ⇒ **no cross-file, no in-file desync possible across a crash.** The per-stage
PRESERVED copy writes the same `resume_arrays` (`:6086`) atomically. Restore is FAIL-OPEN
(`polyak_finisher.py:147-159`, `173-184`): count-without-heavy or heavy-without-count ⇒ LOUD-safe n=1
seed; the EMA shadow (resume-critical) is untouched. `[VERIFIED_VIA_SOURCE_INSPECTION]`

**G. #330 subprocess verdict × resume registry (child death mid-verdict) — FEASIBLE ✓ IF item 6 default-OFF**
Even though item 6 is REGISTERED (default-off) in my classing, I certify the interaction so an ARM is
safe: the subprocess I/O is on DEDICATED tmpfiles (`_verdict_subproc/verdict_in_*.npz` /
`verdict_out_*.json`, `verdict_reclaim.py:205-207`) — DISJOINT from `levelset_resume_state.npz`. The
verdict NEVER writes the resume sidecar. On child death the caller catches and FALLS BACK to the
in-process chunked verdict (`train_…mlx.py:5343-5354`, never a fabricated verdict); the `finally`
killpg's the group + removes tmpfiles (`verdict_reclaim.py:255-261`). The subprocess is proven
bit-identical to in-process ⇒ the d_seg trajectory feeding the muon/annulus event sensors is IDENTICAL
whichever path runs ⇒ **event fires are unperturbed by verdict-path choice.** A missing verdict simply
omits that epoch's point; the sensors are fail-safe (`too-few points ⇒ NOT fired`), so a child crash
cannot SPURIOUSLY fire an event. `[VERIFIED_VIA_SOURCE_INSPECTION]`

**No infeasible interaction found in the composed IN-v7 set.**

---

## VERDICT: **CERTIFY** the composed v7 set, conditional on the named conditions below.

The IN-v7 intersection {unify_tau ∩ TAIL_k ∩ LADDER ∩ basis ∩ 3 event wirings ∩ resume-registry} is
NON-EMPTY and feasible: every shared actuator surface (τ, LR, resume sidecar, memory envelope) has a
single-owner-per-epoch guard traced to source, and every schedule anchor survives the event-mode
fire-vs-cap ambiguity via the two-layer stagger (A) + cap-anchored determinism (B).

### Binding conditions (a REFUSE if any is not met at compile)
- **COND-1 (item 9, binding):** the resume registry MUST be IN-v7. It is the precondition that makes
  event mode (item 2, operator-decided) crash-resumable. Verified landed+tested ⇒ satisfied. Absent it,
  I REFUSE event mode as un-launchable.
- **COND-4 (item 6, binding for the launch):** `--verdict-subprocess` stays DEFAULT-OFF in the sealed
  launch. The launch fits WITHOUT it; arming it re-opens the settled waterfill (SSD-hop transient) and
  adds a child-crash surface for zero launch-necessity. REGISTERED, trigger = RSS ratchet beyond
  preflight. (This is my one dissent from the docket status note.)
- **COND-5 (item 11, binding for the launch):** fp16 cf-feats stays OUT of the launch pending a FRESH
  mem-preflight against the post-basis 71.54 GiB peak proving < 0.70×RAM. Do not compose blind.

### Non-blocking named conditions (certify-with-note)
- **COND-2 (item 8 Polyak):** IF armed IN-v7, `--polyak-finisher-start-epoch` MUST be sized to the
  finishing window (via `polyak_finisher_window_provenance`, `frac∈[0.1,0.3]`). Unsized (`start=0`) the
  candidate averages pre-convergence weights — harmless (never replaces EMA, fail-open) but useless. If
  the operator does not want to size it, class it REGISTERED (stop-time byte-close duty-to-measure).
  Arming it is SCORE-NEUTRAL to the trained trajectory (observe reads-only), so IN-v7 is admissible.
- **COND-3 (items 2×8 anchor):** the Muon finisher LR-final-frac anneal budget and the TAIL start are
  anchored on the muon CAP, not the fire epoch (`:7399`, `:7010`). Feasible + deterministic by design.
  BUT if `--muon-lr-final-frac < 1.0` is ever armed under event mode, an early fire mis-anchors the
  anneal budget. Sealed default is `final_frac=1.0` ⇒ byte-identical ⇒ moot for the launch; flag it so
  a future arm sizes the anneal on the fire epoch.

### #363 evidence-status summary
All five load-bearing composition claims (A/B/E/F/G) are `VERIFIED_VIA_SOURCE_INSPECTION`; the memory
envelope (D) is `VERIFIED_VIA_EMPIRICAL_ANCHOR` (dry-run mem-preflight 67.61 GiB). The β2-sizing law
(item 8) and adaptive-ε equation (item 7) remain `INFERRED/ASSUMED_AWAITING_VERIFICATION` — which is
exactly why I class them ARM/REGISTERED, not IN-v7.

Pointer 0.19110 UNMOVED — this certification is a feasibility gate on APPARATUS, not an exact row.
