# T5 CRUCIBLE — PROBE WAVE-A: CONTROL/SCHEDULE recess probes (P-CT3 · P-CT1 · P-CT2 · τ-CONFIRM)

review_status: fresh-eyes-measured(1)
axis: [macOS advisory] NON-PROMOTABLE — trace analysis on the completed mod32cap run; no score claims; pointer 0.19110 UNMOVED.
run under test (READ-ONLY): `experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/`
trace: `levelset_train_result.json` `history` = the 41-row verdict trace (epochs 0,25,…,1000; d_seg per row).

STORES CONSULTED: `.omx/research/t5_crucible/ORCHESTRATION_LEDGER.md` (req A–R) ·
`DRAFT_OPTIMAL_STACK_v5_20260707.md` §2.2f/§2.5/§3.5/§7c · `DRAFT_OPTIMAL_STACK_v3_20260707.md`
§2.2b/§2.2c/§2.2d (forfeit table + m_q derivation + pre-GO confirm rule) ·
`DRAFT_OPTIMAL_STACK_v2_20260707.md` §2.2b (shipped co-predicate backtest table — the estimator
anchor) · `ct_deepresearch_1_training_campaign_control_20260707.md` §3.2/§3.3/§4.2/§5.1/§10.2 ·
`src/tac/witness_control/powerlaw_exit.py` (fitting machinery, reused) ·
`tools/witness_annulus_convergence.py` + `tools/witness_annulus_live_monitor.py` (cached-map
provenance: 16-pair strided subset, stride = 600//16 = 37) · run-dir `annulus_live.jsonl` +
`annulus_live_maps/maps_{CE_ep299,BEST_ep300}.npz` · `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`
(lstars + margins) · `costate_shadow.jsonl` (rel-slope rows, units cross-check) · corpus_query
("margin quantile m_q").

## DURABLE INSTRUMENTS (requirement Q) — run-2 replays these for free

| surface | path |
|---|---|
| library (laws; pure stdlib; witness_control containment honored) | `src/tac/witness_control/trace_probes.py` |
| CLI (pre-registered bands + verdicts + artifact emit; takes ANY run dir) | `tools/witness_trace_probes.py` |
| tests (17; incl. the bit-for-bit mod32cap estimator anchor) | `src/tac/witness_control/tests/test_trace_probes.py` |
| artifact (this run, full precision) | `.omx/research/t5_crucible/artifacts/trace_probes_levelset_n600_witness_mod32cap_20260706T115554Z.json` |

Reproduce: `.venv/bin/python tools/witness_trace_probes.py --run-dir experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z`
All .py review-tracked (reviewed, no override); ruff F clean; 17/17 tests pass.

**Estimator-form anchor (pins "exactly as the shipped trigger does"):** the shipped co-predicate
(v2 §2.2b) is the trailing-V=4-row endpoint slope. My reimplementation reproduces the v2 table
BIT-FOR-BIT on this trace: first fire **ep625**, rel slope **−1.3693665e-3**/25ep, **n_fires = 8**.
The forfeit arm uses the same window form in per-epoch-normalized S/ep
(s = −100·(d[i]−d[i−3])/(ep[i]−ep[i−3])); a least-squares line fit over the same window is the
robust second estimator (both reported per the charter).

---

## 1. P-CT3 — forfeit-matched TAU→FIN arm backtest — **VERDICT: PASS** (arm promotes to firing arm, injection test still owed)

Law: fire when s < s\* = ν·forfeit = **1.4154e-5 S/ep** (v5 §2.2f / CT-1 §3.3). Stage window
[300, 725], cap 726. Band: first sustained fire ep670–700; EMA-best-at-fire within 1 cadence of
ep650. Kill: fires < ep650 or > 726.

Measured (both estimators AGREE):

| epoch | endpoint slope [S/ep] | LS slope [S/ep] | fires? |
|---|---|---|---|
| 600 | 1.124177e-4 | 1.044210e-4 | no |
| 625 | **1.866093e-5** | 2.763536e-5 | **no — 1.32× above s\*** (the shipped arm fired here) |
| 650 | 4.340278e-5 | 3.057522e-5 | no |
| **675** | **−5.481861e-6** | **5.733914e-6** | **FIRE (both)** |
| 700 | −1.927129e-5 | −2.116903e-5 | FIRE (both) |
| 725 | −6.356698e-5 | −6.983439e-5 | FIRE (both) |

- **First SUSTAINED fire = ep675** (fires hold 675/700/725, no flapping) — inside band [670, 700]. Not < 650, not > cap 726.
- **EMA-best at fire = ep650, d_seg 0.0033661906 = the TAU-stage true best exactly** (0 cadences away) — band condition PASS.
- **The +5.4e-4 S recovery claim — VERIFIED at full precision** (v3 §2.2c forfeit table re-derived from the trace):
  - fire@625, v2-as-written (current θ, no restore): d_seg 0.0033928596 → forfeit **+2.666897e-3 S**.
  - fire@625, v3 restore-EMA-best law (best ≤ 625 = ep600, 0.0033716414): forfeit **+5.450779e-4 S** ← the "+5.4e-4".
  - fire@650 or fire@675 (EMA-best = ep650): forfeit **0 exactly**. Recovery of the forfeit-matched arm vs the ep625 shipped fire = **+5.450779e-4 S**. Claim VERIFIED (not just ≈0 — exactly 0 forfeit on this trace).
- **Robustness to the P-CT1 ν dispute:** under the refit ν = 0.012653 (below), s\* = ν·forfeit = 6.8971e-6 S/ep — the arm STILL first-fires at ep675 (−5.481861e-6 < 6.8971e-6). The fire epoch is invariant across s\* ∈ [6.9e-6, 1.42e-5] because the slope crosses ZERO between ep650 and ep675.

**v5 change:** §2.2f's P-CT3 gate is PASSED — the arm promotes from would-fire-only to the firing
arm CONTINGENT ONLY on the req-B injection test (still owed; not a $0 trace probe). Fail-safe cap
726 untouched.

---

## 2. P-CT1 — ν refit per stage — **VERDICT: BAND-FAIL (CE, tau_softplus) · KILL for muon_fin's window laws** — scope: FORMULATION

Law: d_seg(t) = a + b·exp(−ν t) per stage, profile-LS fit via the existing
`powerlaw_exit.fit_tail_models` (deterministic; AIC vs the power-law alternative). Band
ν ∈ [0.02, 0.035]/ep; kill ν < 0.01 ⇒ recompute ALL window laws.

| stage | rows | ν [/ep] | exp a | AIC pref (ΔAIC exp−pow) | powerlaw α | verdict |
|---|---|---|---|---|---|---|
| CE (25–275) | 11 | **0.019955** | 4.803804e-3 | exponential (−20.32) | 0.120 | BAND-FAIL by 0.2% (marginal) |
| tau_softplus (300–725) | 18 | **0.012653** | 3.376827e-3 | exponential (−46.83) | 0.056 | **BAND-FAIL** (above kill floor) |
| muon_fin (750–1000) | 11 | **0.003289** | 3.238845e-3 | exponential (−42.29) | 0.050 | **KILL** (< 0.01) |

Scope discipline (req R): this is a **FORMULATION**-level negative at most — the single-ν
exponential with window origin at stage start, on THIS trace. AIC PREFERS the exponential over
the power-law in all three stages (window-law *thinking* is not dead; the ν VALUE is what
contradicts the registered 0.026210). Untested reformulations: per-class ν_c (F3 rows, run-1),
window origin at anneal-engage rather than stage start, two-exponential mixture.

**Provenance flag (units, MINOR-9 bug class, audit-grade):** CT-1's ν = ln(13.75)/100 = 0.026210
rests on "erosion slope +3.3e-3 → +2.4e-4 S/ep over ep350→ep450". From this trace: single-cadence
slope at ep450 = **2.385e-4 S/ep ✓** (matches), but at ep350 = **1.4812e-3 S/ep — NOT 3.3e-3**
(no window over this trace's d_seg yields 3.3e-3). The 3.3e-3 magnitude matches the costate
rel-slope rows (−2.89e-3…−3.42e-3 **/ep RELATIVE** over ep300–350, `costate_shadow.jsonl`) — a
suspected relative-vs-absolute units mix. The trace-reproducible two-point ratio gives
ν = ln(1.4812e-3/2.385e-4)/100 = **0.018258**; the LS fit gives 0.012653.

**Recomputed window laws (kill contract; emitted per stage in the artifact) — at the binding tau_softplus ν = 0.012653:**
- settle 3/ν = **237.1 ep** (was 115) · TAIL cycle floor = settle + 150 = **387.1 ep** (was 265) · dwell_TAIL ≥ **237.1 ep** · LPV ramp floor ≥ 237 ep.
- k_max after ep650 in 3000 ep ≈ floor(2350/387) = **6** (turnpike claim "3–7" survives, at its edge).
- s\* = ν·forfeit = **6.8971e-6 S/ep** (P-CT3 fire epoch invariant, above).
- v5 §10.2 consequence: the co-predicate window V=5 (125 ep) no longer covers settle (237 ep); pure 3/ν coverage needs V ≈ 10–11 verdicts — flag to B1's spec, do not silently recalibrate.

**v5 change:** every 3/ν-derived constant (settle 115 · cycle floor 265 · dwell ≥ 115 · s\* 1.4154e-5)
is amendment-grade pending either adoption of the refit values above or a run-1 re-derivation from
the arm's OWN trace (the F3 per-class ν_c rows already planned — this instrument replays on run-1
for free).

---

## 3. P-CT2 — self-triggered cadence replay — **VERDICT: BAND-FAIL (5 skipped, band 12–17); kill NOT triggered** — B-CT3 stays unbuilt — scope: FORMULATION

Law: Δt_next = clamp(floor_S/|Ŝ′|, 25, 100), floor_S = 0.00178, self-triggered (slope from
VISITED verdicts only), floor-rounded to the 25-ep grid (v5 §2.5 / CT-1 §4.2).

Measured:
- window estimator (trailing-4 visited): **5 of 41 skipped** (ep 650, 675, 725, 750, 775) = 12.2% savings, not 30–40%.
- pair estimator: 6 skipped (600, 650, 675, 700, 750, 775).
- **No missed prefix-best > 1 cadence** (kill condition NOT triggered); global best ep650 sits exactly 1 cadence (25 ep) from the nearest visited verdict.
- floor_S sensitivity sweep (what WOULD pass): ×1→5, ×2→6, ×3→6, ×4→7, ×6→8 skipped, never a missed best. **No floor_S reaches the 12-skip band on this trace** — the band is unreachable in the clamp/floor dimension here. Mechanism: the −30–40% projection assumed monotone late-run exhaustion; this trace RESTARTS descent at the ep726 Muon switch (FIN slopes 1.6e-4–3.3e-4 S/ep ≫ floor-binding 7.1e-5), so the 25-ep floor re-binds for the whole FIN stage. The law's savings materialize only on runs whose post-TAU stages are ALSO exhausted (TAIL_k regime, slopes < ~3.6e-5 S/ep for a dozen consecutive rows) — i.e. run-2's longer schedule, not this 1000-ep control.
- **SEAM FINDING (fold-1 interaction, sharper than v5's declared one):** the replay SKIPS ep650 —
  the TAU best. Composed with the forfeit-matched arm on visited verdicts: the arm first fires at
  ep700 (windowed slope over visited {575,600,625,700} = −6.856e-6 < s\*), and the EMA-best among
  VISITED TAU verdicts is ep600 (0.0033716414) → forfeit **+5.450779e-4 S** — the cadence law
  hands back EXACTLY the forfeit the forfeit-matched arm exists to recover (net ≈ 0 composed).
  v5 §2.5's declared seam (window on verdict count) does not cover this; the amendment needed is
  either (a) cadence never stretches while the exit co-predicate is within one decision band of
  firing, or (b) per-cadence EMA snapshots are still SAVED (cheap) even when the n600 verdict that
  RANKS them is skipped — noting honestly that restore-best selection needs the verdict, so (a) is
  the clean fix.

Scope (req R): FORMULATION — this clamp form (floor_S = 0.00178, clamp [25,100]) on this trace
class. Self-triggering as a family is untouched (the replay degraded safely exactly as designed —
floor = today's cadence, zero missed best).

**v5 change:** B-CT3 stays unbuilt (P-CT2 gate not passed). Re-probe on run-1's trace once TAIL
cycles exist; add the seam-(a) guard to any future B-CT3 spec.

---

## 4. τ-CONFIRM — τ\*_end = m_q/ln5 = 0.062 inputs — **VERDICT: PARTIAL — arithmetic ✓; the m_q = 0.10 anchor does NOT reproduce on this run's cached maps; decisive end-checkpoint confirm BLOCKED-cheaply-with-path** — scope: INSTANCE

- Arithmetic: 0.10/ln5 = **0.0621335** ✓ (claimed 0.062; ln5 = 1.6094379).
- Anchor under test (v3 §2.2d, from `birth_death_persistence_dseg_20260630`): "per-pixel flip-rate
  0.764 for GT-margin < 0.10 and ~0.000 above ⇒ the flip annulus lives at m < 0.10" (m_q = 0.10
  support edge). v3's own pre-GO rule: *"recompute m_q from THIS vehicle's cached annulus margin
  field — if the support edge differs materially, re-derive."*
- Recomputed on THIS run's cached artifacts (16-pair strided advisory subset, stride 37; witness
  argmax from `annulus_live_maps/` vs `gt_n600.npz` lstars; GT-cache margin field):

| cached map | flip-rate GT-margin<0.10 | flip-rate ≥0.10 | **flip-MASS share <0.10** | GT-margin at flips q50/q90/q99 |
|---|---|---|---|---|
| CE_ep299 | 0.419846 | 0.00362831 | **0.243520** | 0.25135/0.89490/3.09964 |
| BEST_ep300 | 0.419388 | 0.00363118 | **0.243173** | 0.25406/0.93914/3.36112 |

  (full precision in the artifact JSON.) **75.7% of the flip mass sits ABOVE the
  0.10 edge at this state** — the "all flip mass below 0.10" premise does not hold here; the rate
  above the edge is 0.0036, not ~0.000 (115× lower than below, but the above-edge area is huge).
- Honest limits: these cached maps are ep299/ep300 (early-TAU, d_seg 0.00478, 16-pair advisory
  subset) — τ_end applies at anneal END where the flip population is smaller and more
  boundary-concentrated, so this measurement does NOT falsify m_q = 0.10 at the end state; it
  shows the anchor is STATE-DEPENDENT and unverified on this vehicle. The run's annulus monitor
  emitted only 2 rows/maps (ep299/300) before stopping — no end-state margin artifact exists.
- **BLOCKED-cheaply-with-path** for the decisive check: `tools/witness_annulus_convergence.py
  --ckpt END=<run>/levelset_ckpt_stageTau_muon_ep1000.npz --pairs 16 …` — a 16-pair advisory
  scorer forward, ~minutes of CPU. NOT $0-trivial (scorer re-run) → not executed in this wave; it
  is also NOT n600 (a full-fidelity m_q needs the n600 field, which is the SC-3 live-m_q row v5
  already mandates — this finding CONFIRMS SC-3's necessity).
- **v5 change:** τ_end = 0.062 is PROVISIONAL per v3 §2.2d's own re-derive rule until the
  end-checkpoint m_q is recomputed (cheap path above, or SC-3 on run-1). Scope: INSTANCE — the
  anchor VALUE's transfer to this vehicle/stage; the law τ\* = m_q/ln5 is untouched.

---

## Verdict summary (bands binding, measured first, req J)

| probe | most consequential measured number | band | verdict | disposition in v5 |
|---|---|---|---|---|
| P-CT3 | first sustained fire **ep675**; EMA-best-at-fire = ep650 (stage best, forfeit 0); recovery vs ep625 fire = **+5.450779e-4 S** (claim VERIFIED) | fire 670–700; best ≤1 cadence | **PASS** | arm → firing arm, pending injection test only |
| P-CT1 | ν(tau_softplus) = **0.012653**/ep (CE 0.019955, muon_fin 0.003289); registered 0.026210 not reproducible (suspected rel/abs units mix at the "3.3e-3" input) | ν ∈ [0.02, 0.035]; kill < 0.01 | **BAND-FAIL** (tau, CE) / **KILL** (muon_fin) — scope FORMULATION | recompute 3/ν laws: settle 237 · cycle floor 387 · dwell ≥237 · s\* 6.8971e-6; P-CT3 robust to this |
| P-CT2 | **5 of 41 skipped** (window est.), 0 missed best; no floor_S in ×1–×6 reaches 12 | 12–17 skipped; kill = missed best | **BAND-FAIL**, kill not triggered — scope FORMULATION | B-CT3 stays unbuilt; seam finding: composed with the forfeit arm it returns the full +5.450779e-4 S |
| τ-CONFIRM | flip-mass share below the 0.10 edge = **0.2432** at ep300 (anchor expects ~1.0) | n/a (input verify) | **PARTIAL**; end-ckpt **BLOCKED-cheaply-with-path** — scope INSTANCE | τ_end 0.062 PROVISIONAL per v3's own re-derive rule; SC-3 necessity confirmed |

No run-dir writes; all probes < 1 min CPU, < 1 GiB. None was MISSPECIFIED-AS-$0 (the τ-CONFIRM
end-state leg is the one non-$0 item and is marked BLOCKED with its cheap path rather than run).
