# SEAL v7.3 · Round-2 · STRUCTURE lens · PHASE-2 DIFF (blind-derivation vs as-built)

**Seat:** STRUCTURE (anti-cargo-cult). **Phase:** 2 (diff, post-memo).
**Phase-1 blinding proof:** committed BEFORE any memo read at sha
`1e91081c77a743b09b8c672205eddce96e7f85fb` (`seal_v73_r2_structure_phase1_20260708.md`).
**Pointer 0.19110 [contest-CPU] UNMOVED — everything here is MEANS.** Only a byte-closed n600 exact
row `< 0.19110` from `upstream/evaluate.py` (contest-CPU/CUDA, NEVER MPS) moves it.

## STORES CONSULTED (phase-2 — full)
- Phase-1 file (my blind derivation) + its allowlist stores.
- `.omx/research/crucible_v73_compile_20260708.md` (the compile + Polyak derivation + dry-run gate chain).
- `.omx/research/t5_crucible/SYNTHESIS_INCL_symposium_20260708.md` (T3 class table + arbitrations + deltas).
- `src/tac/witness_autoconfig.py` `_build_crucible_v7` (L2148-2308) + `crucible_v7_polyak_start_provenance`
  + `crucible_v7_registered_off_levers` + the `_CRUCIBLE_V7_*` caps (MUON 726, LANE_BAND 500, CHROMA 450,
  TAIL_CYCLES_MAX 2).
- `src/tac/witness_dsl/curriculum_dsl.py` `DirectionalBasisRebalance` (regime law) — the lane_offloaded vs
  lane_carried freq_along derivation.
- v6-base inherited flag audit (`witness_autoconfig` L759/764/907/912/1455-1457): persistence-loss-weight
  1.0, amplify-weight 1.0, logit-adjust-loss-tau 1.0 all ON in the v7 base.

## What I did NOT re-litigate (operator-DECIDED, per charter)
Event mode · basis lever inclusion · new-baseline-risk acceptance · pose block (verbatim v6) · the
compute levers (D16 pool, safe-compile GPU cert, micro-batch A/B). These are settled; I audit only the
COMPOSED STRUCTURE.

---

## HEADLINE

**verdict:** v7.3 is a STRUCTURALLY SOUND, level-set-native composition — my blind topology (CE →
continuous-L_τ → Muon-on-powerlaw_meat → tail_k at τ* → EMA+Polyak, l7 dissolved, event-driven with
backstop caps) MATCHES the as-built on every load-bearing axis. **No BLOCKER.** Two MAJORs and three
REVISE/MINORs, all fixable without re-architecting. The decisive one is the **lane-regime coherence
mismatch (b)** — a `lane_offloaded` basis composed with four still-full-weight lane-CARRYING learned
losses, which can waste gradient on a frequency the basis cannot represent and may jitter the binding
Road↔Lane boundary. The **Road-convergence question (d)** is real but softer than my blind derivation
feared: Road IS served (the lane_offloaded basis is exactly the all-class cartoon-edge allocation) — the
gap is the ABSENCE of a fallback if the single basis bet underperforms, against a class that barely moved
in v6.

**verdict_scope:** FORMULATION-level (the v7.3 composition as drafted). NOT paradigm/family — the witness,
event mode, and finishing structure are sound and DECIDED.

---

## DIVERGENCE TABLE  {my-derivation · as-built · which-is-right + why · severity}

### AGREEMENTS (blind derivation independently reproduced the as-built)
| topic | my derivation | as-built | verdict |
|---|---|---|---|
| l7 terminal | OFF (measured DEFECT in a viscous flow) | DISSOLVED via `seg_form_unify_tau` (no l7 stage) | **AGREE** — as-built achieves it by dissolving the discrete switch, cleaner than my "omit the stage". |
| discrete CE→τ switch | dissolve into one continuous `L_τ=τ·logsumexp(φ/τ)−φ_y` | `SegFormUnifyTau()` lever = exactly this | **AGREE** (independent match). |
| sensor→transition graph | 3: muon←powerlaw_meat(+nucleation gate), lane-band←lane_nucleus, chroma←annulus_plateau | identical 3 sensors + backstop caps (726/500/450) | **AGREE** — exact match, incl. the `cap_fired_before_event` falsification-signal discipline. |
| budget form | caps + tail cycles, extra budget extends the τ* tail | k_max=2, caps as backstops, 8.673d derived from live 3.62 min/ep | **AGREE**. |
| finishing = EMA + uniform tail mean | both candidates, byte-close picks | Polyak armed as an extra candidate, EMA untouched, fail-open | **AGREE**. |
| INERT levers OFF | uniward / msal_uni OFF | not composed | **AGREE** (verdict_scope: formulation — the msal_uni texture-proxy formulation, prior measured L76/#268; the margin-saliency family stays open via the exact S_R successor). |

### MAJOR
**M1 — lane-regime coherence mismatch (prompt item b).**
- *my derivation:* the analytic lane band + LADDER-lane + basis lane-freq all touch lane capacity — check
  for a redundant/antagonistic pair.
- *as-built:* `DirectionalBasisRebalance(regime="lane_offloaded")` sets `freq_along = round(√32) ≈ 6`
  (cartoon-edge / Candès–Donoho scaling — the regime that assumes lane is HANDLED by the free rule-118
  analytic band, so the basis carries only C²-cartoon boundaries). BUT the v7 base STILL runs, at full
  weight: `--persistence-loss-weight 1.0` (clDice skeleton RECALL on lane class 1, dashes included),
  `--amplify-weight 1.0` (island-amplify class 1), `lane_render_band`, PLUS the new `LadderIslandHomotopy`
  lane VP-tangent curve prior. So the BASIS is set to "lane offloaded" while FOUR learned losses drive the
  witness's OWN render toward lane structure.
- *which is right + why:* the two must be made COHERENT. The LADDER lane term (a smooth VP-tangent
  centerline) IS cartoon-scale-compatible with freq_along=6 — fine. But **persistence-RECALL[1] on the GT
  lane skeleton and island-amplify[1] demand the learned render reproduce lane structure the basis cannot
  represent at freq_along=6** (the dash comb is ~25 cyc/unit — the very reason the `lane_carried` regime
  uses freq_along=26). An unsatisfiable recall target on a frequency-starved basis is at best wasted
  gradient and at worst injects boundary JITTER on the Road↔Lane separatrix — which is part of the binding
  Road residual (68% of flips). This connects M1 to (d): a lane-carrying loss under a lane-offloaded basis
  can HURT the binding class. Cheap resolution (no re-arch): either (i) commit to `lane_offloaded` and gate
  the lane RECALL/island terms to the cartoon-scale structure (rely on the analytic band for dashes), or
  (ii) commit to `lane_carried` (freq_along≈26) and keep the learned lane losses. v7.3 mixes the two.
- *severity:* **MAJOR** (structural incoherence with a plausible mechanism to raise the binding-class
  d_seg; INFERRED-from-mechanism, not MEASURED — a $0 check on whether the lane recall term is satisfiable
  at freq_along=6, or a per-class d_seg watch on Road↔Lane, would settle it).

**M2 — Road is served but has NO fallback, against a class that barely moved (prompt item d).**
- *my derivation (blind):* "every specific seg lever is boundary/rare-class-focused; Road (68%) has no
  Road-first lever → likely the decisive MAJOR."
- *as-built:* Road IS served — the `lane_offloaded` basis rebalance is precisely the all-class cartoon-edge
  (Road↔Undrivable horizon + Road↔Lane) allocation (−48% directional was measured ALL-class), and eikonal
  + seg-chroma-boundary (annulus) are all-class. So my blind "no Road lever" fear is **partly WRONG** and I
  correct it: Road is addressed by the generic boundary suite.
- *which is right + why:* the residual concern is narrower but real. (1) v6 MEASURED Road flip-rate barely
  moved (0.44→0.40 over 100 ep) under the generic suite — so "generic levers touch Road" is not the same as
  "generic levers CRUSH Road". (2) v7.3's entire Road bet rides on ONE new mechanism (basis freq_along 4→6
  + geometric anneal) with **no contingency lever** if it underperforms. (3) `--logit-adjust-loss-tau 1.0`
  (Menon) actively DE-weights the 68%-binding Road class in the loss (offset −1.37 vs lane −5.14 boosts
  rare classes) — defensible for rare-class recall, but it is spending loss-gradient away from the binding
  class. Together: Road convergence is the #1 empirical risk of the run, and the composition has a single
  point of failure on it.
- *severity:* **MAJOR** as a WATCH/REVISE — not launch-blocking (the run will produce the answer), but the
  seal should (a) name Road per-class d_seg as the primary run-abort/continue signal, and (b) register a
  Road-fallback lever in the duty-to-measure queue (e.g. a Road↔Undrivable margin term, or a Menon-offset
  audit) so a Road underperformance at ~day-1 has a pre-staged response rather than a cold restart.

### REVISE / MINOR
**R1 — Polyak start-epoch: fixed-cap sizing vs event-anchoring (prompt item a).** *my derivation:* anchor
the Polyak window to the muon-fire EVENT (else a fixed epoch mis-places the uniform mean). *as-built:*
`start_epoch = muon_CAP(726) + (2274 − round(0.2·2274)) = 2545`, window 455 ep ending at 3000, explicitly
sized off the muon CAP (the LATEST possible entry) to GUARANTEE post-Muon / never-pre-turnpike. **This
RESOLVES my concern the safe way** — it deliberately accepts the "shorter tail fraction" failure mode
(higher variance if the event fires early) to eliminate the "descending-prefix bias" failure mode, which
is correct because a uniform mean is unforgiving of a descending prefix and Polyak is only a free EXTRA
candidate (byte-close discards it if EMA wins). Residual: if the muon EVENT fires well before 726, the
2545 start discards most of the turnpike orbit (higher variance than an event-anchored window would give).
*severity:* **MINOR** — the cap-sizing is defensible and resumable-simple; event-anchoring is a v7.4
refinement, not a v7.3 fix. Which is right: as-built (for v7.3).

**R2 — running unify-τ AND flipping τ-advance to event in the SAME run confounds two schedule changes.**
The v7.3 base sets `--tau-advance-mode=event` (self-paced octave ladder) SIMULTANEOUSLY with the first-ever
`seg_form_unify_tau` run. The compile's own cited memo (`self_paced_tau_advance_20260708`) RECOMMENDS the
first unified-L_τ run be in CLOCK mode to isolate the unify-τ variable, then flip to event for run-2
("one continuation parameter at a time"). The as-built emits `event`, diverging from its own memo's
recommendation (the comment flags this and defers to "council/seal"). *severity:* **REVISE** — the SEAL
should make the explicit call: clock-first (clean attribution of the unify-τ effect) vs event-now
(operator conversion directive). Isolating one schedule variable is the from-scratch-correct choice; I'd
launch run-1 in CLOCK mode. Which is right: my derivation (isolate) for a first unify-τ run.

**R3 — early gnorm-hijack transient watch.** run-1 telemetry fired `gnorm_hijack` 3× at ep1 (gnorm
143–167 vs grad-clip 1.0; note: "one gradient group scaling the whole step down — seg starvation risk")
with `island_amplify=8.25` = ~20% of ep1 total loss. The note says `--per-group-grad-clip` bounds it.
*severity:* **MINOR** — verify `--per-group-grad-clip` is actually ON in the v7 base (it is referenced in
the alarm), else the large early island/eikonal terms can starve the seg gradient during the very window
where the coarse partition (and Road) forms. A one-line launch-flag confirmation.

### SELF-CORRECTION (attacking my own phase-1, per manual §6)
My phase-1 twice under-valued the island suite ("rare classes are a rounding error", "budget aimed at the
wrong class"). On re-derivation that is WRONG as stated: lane & movable are BORN-EMPTY at init
(part_frac=0.0, d_seg=1.0) and the island/persistence losses drive them 1.0→0.01–0.09 by ep25 — they are
LOAD-BEARING, not wasted (without them lane+movable would each contribute ~1.0 to d_seg). The correct
framing (which M1/M2 use): the island suite is justified for BIRTHING the born-empty classes; the defect
is not "island losses are wasted" but "a lane-CARRYING recall/amplify under a lane-OFFLOADED basis is
incoherent and may jitter the binding boundary." I flag my own phase-1 overstatement rather than let it
travel.

---

## OFF-LEVER QUEUE (prompt item c) — what a from-scratch design would toggle vs v7.3
- **Road-fallback lever** — WANTED, MISSING (M2). Register in duty-to-measure with trigger = Road per-class
  d_seg plateau > threshold by ~day-1.
- **lane recall/island regime-gate** — WANTED (M1): gate the lane-class learned losses to the chosen basis
  regime.
- v7.3's registered-off levers (micro-batch, #330 verdict-reclaim, adaptive-ε, GPU-verdict, fp16 cf-feats)
  are COMPUTE/apparatus levers, not d_seg score-movers — correctly OFF with named triggers. No d_seg
  structural lever is wrongly parked. **AGREE.**
- Menon logit-adjust: keep ON but **audit the Road offset** (part of M2) — it de-weights the binding class.

---

## SEAL RECOMMENDATION (this lens)
**REVISE-then-PROCEED.** No BLOCKER; the topology, sensor graph, finishing, and budget are sound and match
a blind from-scratch derivation. Before the SEAL closes: resolve **M1** (make the lane regime coherent —
basis vs learned-lane-losses) and **M2** (name Road per-class d_seg as the primary run signal + register a
Road-fallback lever), decide **R2** (clock-first vs event τ-advance for the first unify-τ run — I recommend
clock), and confirm **R3** (`--per-group-grad-clip` ON). R1 stands as-built for v7.3.

**verdict:** REVISE-then-PROCEED · **verdict_scope:** FORMULATION (v7.3 composition). Pointer 0.19110
UNMOVED — MEANS; the END is a byte-closed n600 exact row < 0.19110 after the run.
