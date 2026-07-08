# SEAL v7.4 · ROUND-3 · LENS: DEEP-MATH + STRUCTURE (fix-wave diff) — [no-triality]

- **UTC:** 20260708 · **Agent:** SEAL ROUND-3 DEEP-MATH+STRUCTURE (Opus, hostile) · **Authority:**
  `[macOS advisory / pure-math + structure review]` — $0, NO launch, live pid 63069 UNTOUCHED. Pointer
  contest-CPU **0.19110 UNMOVED** — this review is a MEANS; the END is a byte-closed n600 exact row
  `< 0.19110` from `upstream/evaluate.py`.
- **Scope:** `git diff 106e77b84..HEAD` = builder A (`5ea59a1f1`) + builder B (8 commits) + hook patches.
  ROUND 3 reviews ONLY the fix-wave diff (fixes are UNREVIEWED NEW CODE). I was the round-2 deep-math
  lens that filed the β_end BLOCKER + the two MAJORs.

## STORES CONSULTED
- `SYNTHESIS_seal_v73_round2_20260708.md` (fix charter) + all four round-2 lens reports
  (`seal_v73_r2_{bugs,deepmath,confound,structure_phase1,structure_phase2}`) + fix-wave memos
  `r2_fixwave_{A,B}_20260708.md` + `LAUNCH_PACKAGE_v7` + `crucible_v73_compile` + `ORCHESTRATION_LEDGER`.
- CODE read from source (not memo prose): `tau_advance.py:240-299` (octave_fraction β/LR coupling — the
  A1 mechanism) · `witness_autoconfig.py` (`_CRUCIBLE_V7_HOSC_BETA_END_EVENT`, `_build_crucible_v7`
  L2280-2349, `crucible_v7_polyak_start_provenance`, `crucible_v7_registered_off_levers`) ·
  `curriculum_dsl.py` (`persistence_classes_for_basis_regime`, `DirectionalBasisRebalance`) ·
  `scorer_throughput_gate.py` (anchor) · `resume_registry.py:185-254` (state_arrays manifest chain,
  `build_gate_resume_registry`) · `event_wirings.py` (sentinel writer) · `metal_persistence_pool.py`
  (B2 fingerprint gate) · `typed_config.py` (B4) · trainer L4050-4130 (persistence/island setup) +
  L7285-7322 (per-class-λ ladder homotopy).
- MEASURED myself: run-1 `levelset_n600_crucible_v6_run1_20260708T095730Z/run.log` wall-clock
  re-reconstruction (A2); independent module-constant re-derivation; 202 targeted tests + ruff F.
- **review_status:** fresh-eyes round-3 diff review; every number RE-DERIVED from the primary artifact,
  not the memo. Pointer 0.19110 UNMOVED.

---

## THE β(t) TRAJECTORY CHECK (the round-2 BLOCKER — the headline)

**Mechanism (source, `tau_advance.py:282,285`):** event-mode
`β(rung) = β_start + (β_end − β_start)·octave_fraction()`, `octave_fraction = rung/N`, **β and τ ride the
SAME octave_fraction** and both FREEZE at the Muon switch. With the fix `β_start=1.0, β_end=3.177`:

```
β(rung) = 1.0 + 2.177·(rung/N)          rung ∈ [0, N]
  rung 0     (CE / τ=1, coarse):  β = 1.000   ← soft, CORRECT at the coarse partition
  octave mid (rung/N≈0.5):        β = 2.089   ← healthy mid-sharpen
  rung N     (τ=τ*=0.31, Muon):   β = 3.177   ← FROZEN for the whole ~2274-ep tail
```

**Verdict on the two failure modes the prompt names:**
- **Early-run β too low (under-annealed)?** NO. Early β = 1.0 is the DESIGNED soft value at CE/coarse-τ.
  Because β and τ share `octave_fraction`, β is ALWAYS the correct sharpness for the current τ-coarseness
  at every rung — this is MORE correct than the clock (β keyed to actual τ-progress, not wall-clock ep).
  β-low-when-τ-high is not a pathology; it is the co-annealing intent. There is no rung at which β is
  saturated-high while τ is still coarse.
- **Trajectory dominates the healthy clock pointwise?** NO. The trajectory is BOUNDED in **[1.0, 3.177]**
  with `max β = β_end = 3.177 ≤ 4.0` (the anneal-β divergence bound). It never enters the forbidden
  fixed-high-β tanh-saturation regime anywhere. The frozen tail value (3.177) EQUALS the mod32cap control's
  frozen β(726) — the fix restores the exact healthy control state the round-2 BLOCKER destroyed (frozen
  β≈10 over 76% of the run). The fix does NOT trade the late freeze for an early pathology.
- **CE→tau handoff / mid-tau checked:** handoff (octave_fraction≈0) → β≈1.0 (soft CE, correct); mid-tau
  (octave_fraction≈0.5) → β≈2.089 (healthy). Both inside [1, 3.177].
- **[1,10] GPU bit-cert coverage:** range ⊆ [1,10] verified at BOTH endpoints (1 ≥ 1 ✓; 3.177 ≤ 10 ✓);
  bit-identity is β-value-invariant within the domain, so the superset cert bounds the new range. ✓
- **Provenance:** I INDEPENDENTLY re-derived `β(726) = 1 + 3·725/999 = 3.17718 → 3.177` from the control's
  den-1000 trajectory (the primary artifact), NOT the intermediate 10.0. DERIVED-AT-CONFIG, clean.

**A1 = CLEAN.** The round-2 BLOCKER is correctly and completely fixed at the trajectory level.

---

## DEEP-MATH FINDINGS (items 2–5)

### A2 budget — CLEAN (one MINOR note). I re-reconstructed run-1's wall-clock MYSELF.
From `run.log` verdict `ts` (launch 09:57:30Z): ep0=24.43 / ep25=137.77 / ep50=219.02 / ep75=312.33 /
ep100=396.62 min — **EXACT match** to builder A's table. r_ss(ep75→100)=(396.62−312.33)/25=**3.371**;
S=396.62−337.16=**59.5**; amortized(3000)=3.371+59.5/3000=**3.39**; budget=3.39·3000/1440·1.15=**8.122 d**;
refuse ceiling=3.39·1.15=**3.90** = a TRUE 15% gate (independently confirmed 3.898 from the live code).
The REASONED DEVIATION from the synthesis's 3.12 is CORRECT: run-1's measured steady slope (3.37)
contradicts the memo's untrusted r_ss=3.1 lower bound; the value-provenance ladder forbids anchoring on a
lower bound. **MINOR (non-blocking):** r_ss=3.37 is the ep75→100 window; the fuller ep25→125 average is
~3.45 (slopes bounce 3.25–3.73, NOT declining to 3.1), so 3.39 is ~0.06 min/ep optimistic. Direction is
FAIL-SAFE for a refuse gate (ceiling 3.90 is marginally tighter → refuses marginally-too-eager, never
lets a too-slow run through); budget projection ~0.2 d low. verdict_scope: INSTANCE.

### A3 Polyak 2546 — CLEAN. `726 + (2274 − round(0.2·2274)) = 726 + 1819 = 2545`; the inclusive loop
`[2545, 3000]` = **456** epochs (the off-by-one the round-2 MINOR-2 caught). Fix `epochs − window + 1 =
3000 − 455 + 1 = 2546` → `[2546, 3000]` = **exactly 455** — I VERIFIED by simulating the loop
(`sum(1 for ep in range(2546,3001)) = 455`). Degenerate `epochs+1` is GENUINELY inert — I ran the real
`PolyakTailAverager` over `range(1,4)` and confirmed `count==0` (vs old `start=epochs` observing once).
Averaging-window theory: the `0.2` is LawRef-tagged (`muon_finisher_schedule_warmstart_and_lr_anneal_v1`,
band [0.1,0.3]) — inherited-within-a-justified-band, not a bare literal; the specific 0.2 within-band is a
default (round-2 accepted cap-sizing as variance-CONSERVATIVE, not a v7.4 blocker).

### A5 persistence law — CLEAN. `persistence_classes_for_basis_regime` is CORRECT against the born-empty
erasure-tail class set {1=Lane, 3=Movable} (structure phase-1 D0 fact 5). `lane_offloaded → "3"` (movable
only; lane rides the FREE analytic band, MEASURED lane d_seg 0.00087, and the freq_along≈6 cartoon basis
cannot represent the ~25-cyc dash comb → a lane-skeleton RECALL there is unsatisfiable → wasted gradient +
Road↔Lane jitter). `lane_carried → "auto"` (keep lane at freq_along≈26). fail-closed on an unknown regime.
Counter-arm coherently specified: flipping `_CRUCIBLE_V7_BASIS_REGIME → lane_carried` drives BOTH surfaces
off ONE constant (freq_along via `freq_along_for_regime` AND persistence via the derived law) → no drift;
the freq_along≈26 √-optimum is honestly labeled ASSUMED_AWAITING_VERIFICATION.

### B5 closed-by-construction — CLEAN, and I verified the FULL chain + all three gate shapes.
`build_gate_resume_registry` registers muon/lane_band/seg_chroma_boundary as `EventBackstopGate`s under
`__mg_/__lbg_/__cbg_`. An event-mode gate's `state_arrays` returns `{prefix+fired_epoch: -1,
prefix+fired_by: ""}` even UNFIRED (guarded `if not self.event_mode: return {}`) → **all three write a
sentinel pre-fire**. `ResumeRegistry.state_arrays` stamps the manifest `if any(w["event"] for w in wrote)`
where `event = e.event_active` (independent of firing) → stamped from checkpoint 1 → every co-writing
non-event controller (rng/closed-loop/tau/evt) is vanish-protected across the whole ep0..muon-fire window.
Necessary-and-sufficient condition (≥1 event gate writing a sentinel) is over-satisfied (all three do).
The ONLY manifest-free case (clock/cap-only) is the LIVE run's byte-identity contract, correctly preserved.
Test `test_event_mode_unfired_gate_stamps_manifest_no_window` PROVES it (all-unfired event registry stamps;
sister cap-only stays manifest-free). Behavior test, not a constant assertion.

---

## STRUCTURE FINDINGS

### Lane-regime coherence across the THREE surfaces — CONSISTENTLY applied (one MINOR conditional).
- **Surface 1 (basis lever)** + **Surface 2 (persistence-recall)** are HARD-gated off the single
  `_CRUCIBLE_V7_BASIS_REGIME="lane_offloaded"` constant (basis freq_along=6 via
  `DirectionalBasisRebalance`; `--persistence-classes="3"` via the derived law) — verified in the emitted
  argv (`--persistence-classes 3`, `--freq-along 6.0`).
- **Surface 3 (island-amplify)** is SELF-gated via `LadderIslandHomotopy` (COMPOSED in v7 at
  `witness_autoconfig.py:2349`): the per-class-λ homotopy (`trainer:7298-7314`) drives each class's island
  radius by its MEASURED marginal cost `λ_c` (`step_radius(arm, ep, λ, …)`), so the lane arm auto-shrinks
  when lane's cost falls. This is a REAL mechanism (λ-driven radii), NOT a hand-wave. The design DECISION
  — hard-gate the fixed-weight term, self-gate the λ-adaptive term — is internally articulated in the A5
  docstring. **Coherence is applied to all three surfaces via the mechanism appropriate to each.**
- **MINOR (non-blocking, NOT a new gap):** surface-3 self-gating is CONDITIONAL on λ_lane actually
  falling, which requires the `lane_render_band` to composite (run.log: "composited PRE-R", `start_epoch:
  350`). So in the early window [0, 350] the homotopy grows lane islands under the lane_offloaded basis —
  the pre-existing round-2 M1 residual, BOUNDED and already WATCHED via the registered `lane_carried`
  counter-arm + the Road↔Lane jitter watch-list signal. Arguably CORRECT (birth born-empty lane early,
  hand off to the band at ep350, homotopy self-de-emphasizes). verdict_scope: INSTANCE.

### A6 road_boundary_fallback — CLEAN (registered-off, no silent activation).
`road_boundary_fallback` + `lane_carried_basis_regime` appear ONLY in `crucible_v7_registered_off_levers()`
(a metadata registry, `default:"off", state:"registered_duty_to_measure"`) — they wire NO argv flag into
`base` (grep-confirmed; consumed only by the observability test). Duty-to-measure triggers are pinned
(Road flip-rate > 0.30 @ep200). No silent activation.

### Value-provenance ladder — CLEAN (no bare literals).
Every new knob carries provenance: `3.177` DERIVED (control β(726), full derivation comment); `3.39`
MEASURED (re-derived from run-1's log); `lane_offloaded` operator-DECIDED (approved basis rec); `"3"`
DERIVED (from regime); `--per-group-grad-clip=True` justified (R3, run-1 gnorm_hijack telemetry). Consistent
with the existing base-argv pattern (derivation in the module-constant, budget wrapped in `Provenanced`).

### Blinded-derived topology PRESERVED.
The fix-wave touches endpoints/gates/coherence, NOT the shape. The round-2 blind topology
(CE→continuous-L_τ→Muon-on-powerlaw_meat→tail_k at τ*→EMA+Polyak, l7 dissolved, event + backstop caps)
is unchanged; A5's coherence fix REDUCES an antagonism the blind derivation flagged, moving toward the
blind ideal, not away.

---

## TESTS + BUILD (verified myself)
- ruff `--select F` CLEAN on all 9 touched source files.
- 202 tests PASS across the 7 touched suites (crucible_v7_config · resume_registry · event_wirings ·
  wallclock/perfenv · metal_persistence_pool · weights-arm · closed_loop_control).
- Tests are BEHAVIOR-oriented (NO-FAKE class-2 avoided): the Polyak tests SIMULATE the inclusive loop and
  COUNT observations (`observed == window == 455`; degenerate `count==0` over the real averager), not bare
  constant asserts; β_end test asserts the EMITTED argv value + the event-mode coupling + the ≤4.0 bound.
- Independent re-derivation of every headline constant matched the emitted config exactly (3.177 / 3.39 /
  8.122 / 2546 / persistence "3" / refuse 3.90).
- Triality/decision-record: the operator EVENT override verbatim is present in ledger + launch package +
  compile memo (A4); all three legs carry the updated values consistently.

---

## VERDICT: **CLEAN** — 0 BLOCKER · 0 MAJOR · 2 MINOR (both non-blocking, both already in the watch-list).
The fix-wave closed the round-2 BLOCKER (β_end) and every MAJOR/MINOR without introducing a new pathology.
The β(t) trajectory is bounded [1.0, 3.177], co-annealed with τ, no early under-anneal and no saturation
— the fix restores the healthy control frozen-β exactly. The budget/Polyak arithmetic re-derives exactly
from the primary artifacts. The lane-regime coherence is consistently applied across all three surfaces
(hard-gate + articulated self-gate). B5's sentinel→manifest chain is logically sound and test-proven.
The two MINORs (A2 r_ss window marginally optimistic in the fail-safe direction; A5 surface-3 self-gating
is band-composite-conditional post-ep350) are INSTANCE-scope and already pre-registered in the launch
watch-list — they do not gate the launch. **verdict_scope:** the fixes are FORMULATION/INSTANCE level; the
witness paradigm + event mode + finishing structure are sound and DECIDED. Pointer contest-CPU **0.19110
UNMOVED** — every line here is MEANS; the END is a byte-closed n600 row < 0.19110 AFTER the run.
