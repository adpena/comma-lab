# T5 CRUCIBLE — PROBE WAVE-B: geometry/class probes on cached fields (Q1 · P-CON · P-DZ · P-MP)

`[macOS-numpy advisory . NON-PROMOTABLE]` — pointer 0.19110 UNMOVED (these are $0 cached-field
gate probes; means, not ends).

STORES CONSULTED: ORCHESTRATION_LEDGER.md (reqs A–S, esp. R verdict-scope + Q toolbelt) ·
DRAFT_OPTIMAL_STACK_v5_20260707.md §1.3/§3.4/§0.0a/§0.0b/§7c/§11-row-16 ·
negatives_scale_validity_review_20260707.md (Q1 spec §7) ·
ct_deepresearch_2_pde_geometric_topological_control_20260707.md §7/§13 ·
birth_death_persistence_dseg_20260630T172510Z.md (filtration cross-check) ·
corpus_query (FEED-08l · comb-registration — straggler section) · MEMORY.md L65/L76.
NOT consulted: run dirs beyond the read-only caches named per probe.

review_status: fresh-eyes-measured(1); instruments inherited from credit-killed predecessor,
reviewed as own (one defect found + fixed at review: Q1 correlator could declare `robust_dead`
with major class-pair sides MISSING/NaN — missing sides now block the kill verdict).

Instruments (committed this session, req Q toolbelt): `tools/signed_flip_asymmetry_correlator.py`
· `tools/conley_persistence_certifier.py` · `tools/uint8_deadzone_census.py` ·
`tools/maxplus_annulus_fit.py`. Artifacts: `experiments/results/t5_probe_waveB_20260708/`.
Common substrate: gt cache `experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz` (96
witness frames, stride-2 → gt, stride-3 → global n600; alignment byte-asserted per frame) +
witness maps `experiments/results/witness_per_stage_attribution/maps_{Tau,MuonBest}.npz`
(τ=0.0500065329 @ep599 · τ=0.2156894835 @ep900). Subset caveat (req L): 96-frame witness
attribution subset, NOT n600 — these are gate probes on cached artifacts as pre-registered;
scope tags below are set accordingly.

---

## P-CON — Conley persistence certificate backtest

- method: per-island (4-conn GT-class components, same filtration as the 20260630 birth-death
  ledger: H0 superlevel birth = component peak GT top1-top2 margin) vs witness survival
  (majority: flip_rate < 0.5), certificate pers > τ_k·ln5, at both known τ points.
- artifact: `experiments/results/t5_probe_waveB_20260708/pcon_conley_backtest.json` (+ per-island
  `.ledger.npz`, 3454 islands × 2 stages — the B17 row format).
- numbers (full precision):
  - Tau (τ=0.0500065329, thr=0.0804824100): 3454 islands, 3126 certified,
    P(survive|cert) = **0.4408189379**, P(survive|uncert) = 0.0853658537.
  - MuonBest (τ=0.2156894835, thr=0.3471388320): 2360 certified,
    P(survive|cert) = **0.5635593220**, P(survive|uncert) = 0.1106032907.
  - Pixel-weighted certified survival: 0.9997628103 (Tau) / 0.9998880585 (MuonBest) — the
    failures are TINY (lane-dash) islands.
  - Per-class: Lane is the entire failure — P(s|cert) 0.3056 (Tau) / 0.4081 (MuonBest) on 2691
    lane islands; Road/Undrivable/Movable/MyCar all 0.81–1.00.
  - Safety-factor fits (the pre-registered reformulation): Tau s = 21.75 → threshold
    1.7504924172 logit, P(s|cert)=0.9573901465 (751 certified); MuonBest s = 3.75 → threshold
    1.3017706202 logit, P(s|cert)=0.9505783386 (951 certified).
- band: PASS ≥0.95, KILL <0.80 → **0.44/0.56 = KILL at both τ points; fit-s branch executed.**
- VERDICT: **KILL, verdict_scope: FORMULATION** — the raw threshold pers > τ_k·ln5 (+Δ_dec=0)
  FAILS as a sufficiency certificate at both τ points. The certificate FAMILY survives by
  construction via the fitted safety factor (pre-registered reformulation, req R): reformulation
  queue = {fitted s per stage (measured above) · ABSOLUTE-threshold form (see finding) ·
  per-class thresholds (lane-specific) · Δ_dec^logit > 0 once SC-7 measures it · size-weighted
  certificate}.
- FINDING (v6-grade, beyond the band): the two fitted thresholds are nearly τ-INDEPENDENT
  (1.75 vs 1.30 logit while τ varies 4.3×) — survival behaves like an ABSOLUTE persistence bar
  ~1.3–1.8 logit, not ∝ τ·ln5. The τ-scaling of the law, not just its constant, is what failed.
  Discriminative signal is REAL (cert vs uncert survival separates 5.2×/5.1×), so B17's alarm
  keeps its sensor value — but B17's alarm semantics must ship with the fitted absolute
  threshold (or per-class s), NOT the raw τ·ln5 form; "certified-death = controller failure"
  would false-alarm ~50% of lane islands as-written.
- what changes in v5/v6: §3.4/B17 threshold law amended to fitted-s / absolute form; per-class
  (lane) certificate row added; consistency row (a) (τ_end·ln5 ≈ 0.10 flip edge) is UNTOUCHED
  (different law — flip-support width, not island survival).

## P-DZ — uint8 deadzone census

- method: CT-2 §13 M1 estimator on cached fields — deadzone(x) ⟺ |H_R|·g_I(x)·(m/|∇m|) < 1
  (0-255 units), |H_R| = 0.842 (census-maximizing, conservative-for-DEFER) and 1.0 variant;
  GT-margin-geometry form (witness margins not cached — stated form limitation).
- artifact: `experiments/results/t5_probe_waveB_20260708/pdz_deadzone_census.json` (+
  `.gi_hists.npz` per-pair g_I edge-contrast histograms = the SC-16 seed, 128 bins).
- numbers (full precision): 96 frames, total_px 18,874,368; flips 77,706 (flip d_seg
  0.0041170120 on this subset/stage MuonBest); deadzone flips HR = 29,813 →
  **d_seg-equivalent 1.5795495775e-3**; H1 variant 27,475 → 1.4556778802e-3;
  deadzone fraction of flips = 0.3836640671.
- per-pair concentration (top rows, HR): Lane→Road 7,019 (24.7% of pair flips; rows 176–224
  dominate = far-range lane) · Road→Lane 5,647 (39.0%) · Road→Undrivable 4,362 (77.5%, rows
  176–224 = horizon/shadow edges) · Undrivable→Road 3,190 (58.8%) · Road↔MyCar 5,686 combined
  (67–85% of pair flips, rows 224–384 = hood boundary — #139 hood-clamp territory, free).
- band: <5.34e-6 DEFER / >1.78e-5 duty queue → measured value is **88.7× ABOVE the bind
  threshold** (and 296× above DEFER). **VERDICT: #149 ENTERS THE DUTY QUEUE**,
  verdict_scope: INSTANCE→duty (census, no kill by design; positive disposition). Scope per
  req H/L: binding is per-class-pair per-range — far-range lane + horizon + hood boundary;
  near-range lane (rows 224+ on Lane↔Road: 1,588/6,470 deadzone) partially healthy.
- honest boundary: the estimator is the CT-2 formula's own GT-geometry form; it counts flips
  whose required smooth through-R intensity change is sub-quantum — an in-principle
  unreachability census, not a measured #149-fix yield. The YIELD upper bound if #149-class
  dither/phase tricks recover the whole census: ~1.58e-3 d_seg ≈ 0.158 S-units of headroom —
  vastly above the crossing margin; even 1% recovery ≈ 0.89× margin.
- what changes in v5/v6: #149 (sub-pixel/dither closed-form) moves DEFER → duty-to-measure
  queue with a real measured prior; SC-16 g_I histograms now exist as seed data; the M1
  impossibility bound gets its deciding measurement row (family floor term is REAL at this
  stage of training).

## Q1 — signed per-class-pair per-DIRECTION ρ (the B16 gate)

- method: point-biserial Pearson ρ(field, flip) per ORDERED class pair on the GT-class-i side of
  the (i,j) boundary annulus (chebyshev r=2), population restricted to {witness ∈ {i,j}}; field
  under test = texprox (the LEVER-4 msal_uni UNIWARD cost 1/(1+4·tex), exact lever formula);
  positive controls sR (cached #268 through-R reachability sidecar) + GT margin. 96 frames,
  alignment byte-asserted; all 8 major sides evaluated (none missing).
- artifact: `experiments/results/t5_probe_waveB_20260708/q1_signed_asymmetry.json`.
- numbers (full precision, ρ_texprox / ρ_margin): Road→Lane −0.0713726498 / −0.2448159833 ·
  Lane→Road −0.1115970020 / −0.3786970261 · Road→Undrivable +0.0139239649 / −0.2930434855 ·
  Undrivable→Road +0.0275365717 / −0.2847148814 · Road→Movable −0.0272292227 / −0.3092093454 ·
  Movable→Road **+0.1147104715** / −0.3075756246 · Undrivable→Movable −0.0421340661 /
  −0.2982510727 · Movable→Undrivable **−0.1241516781** / −0.3293063754.
- band: FIRE |ρ|≥0.3 any side → **max |ρ| = 0.1242, NO FIRE**. KILL |ρ|<0.1 both sides all
  major pairs → three sides ≥0.1 (Lane→Road 0.1116, Movable→Road 0.1147, Movable→Undrivable
  0.1242) → **NOT robust-dead either**.
- VERDICT: **BETWEEN (no-fire, no-kill), verdict_scope: FORMULATION** (this cost field + linear
  point-biserial response). B16 stays DEFAULT-OFF/gated — it does NOT enter the duty queue with
  a real prior. The requirement-L mechanism IS qualitatively confirmed: per-direction effects
  exist with OPPOSITE SIGNS across pairs (raw-texture ρ: Lane→Road erasure +0.1346 = flips in
  HIGH texture; Movable→Road −0.1027 = flips in LOW texture) — the pooled −0.033 was averaging
  opposite-signed weak effects, exactly as the shape-gradient theorem predicted — but at
  ~0.10–0.12 magnitude, 2.4–2.9× below the fire bar. Reformulation queue (req R, enumerated at
  non-fire): other cost fields (S_R-composited, per-range-conditioned) · hinge vs linear
  response · rank-ρ (Spearman) · witness-margin-side conditioning (winner vs runner-up side
  proper, needs per-class logits cached — SC-16's full form).
- instrument caveat (honest, per the pre-registered sentinel): the sR positive control is
  itself at chance (max |ρ_sR| = 0.0808) while the margin control passes strongly (−0.24 to
  −0.38, right sign, all sides). Per the instrument's strict letter the sR sentinel failed; the
  verdict is admitted WITH CAVEAT because the margin control demonstrates instrument
  sensitivity on the same populations + labels. The sR-at-chance is itself a NEW measured row
  echoing L76: the through-R reachability field does not predict realized flips within the
  boundary annulus (annulus flips are margin-driven, not reachability-driven).
- what changes in v5/v6: B16 remains gated (state: never-fired → Q1-adjudicated-no-fire in the
  activation ledger); §1.3's "efficacy bound 2.2× margin" must NOT be cited as live — the
  measured prior is ~0.12/0.3 = 40% of the fire bar; SC-16 keeps its build slot (the per-logit
  winner-side form is the untested reformulation with the most headroom).

## P-MP — max-plus annulus fit

- method: per class per frame, fit logit_c on the flip-support annulus (GT margin < 0.10; fit
  support < 0.5) as max of K concave quadratics (Magnani-Boyd alternating, NSD-projected,
  curvature-capped); frozen CPU SegNet logits (path self-validated: argmax agreement vs cached
  lstars asserted ≥0.99/frame); 8 frames evenly spread; K grid 4–64; ORACLE-SELECTION
  capacity control (min-residual assignment) + BULK control (M5).
- artifact: `experiments/results/t5_probe_waveB_20260708/pmp_maxplus_fit.json`.
- numbers: annulus argmax agreement = 0.352255 (K=4) · 0.390021 (K=8) · 0.413596 (K=16) ·
  0.412451 (K=32) · 0.407874 (K=64) — plateaus ~0.41, band is 0.95: **KILL at K≤64**.
  Decomposition (the kill-scope sharpener): ORACLE-selection rms residual at K=64 =
  0.0761–0.1306 logit per class (≈ the τ_end coupling bound 0.0998) — K quadratics CAN track
  the fields; max-ENVELOPE rms residual = 2.30–58.18 logit (Road 56.16, Lane 58.18, p95 to
  157.8) — the concave-max SELECTION mechanism fails catastrophically on Road/Lane annulus
  fields. Bulk control: 0.68–0.94 agreement (no clean two-semiring blow-up signature at these
  K; M5 unresolved by this form). Bytes: K=64 fp16 n600 = 2,304,000 B = **1.5341 S** — even a
  PASSING K=64 would be rate-dead by 862× the crossing margin; only K≈1–2/class is ever
  λ_bytes-viable.
- VERDICT: **KILL, verdict_scope: FORMULATION** — max-of-K≤64-concave-quadratics at the
  annulus. NOT max-plus as a family (req R). Reformulation queue, now RANKED by the measured
  decomposition: (1) band-residual hybrid m_lane = max(band, m_INR) — the K=1 special case
  already in the stack, UNAFFECTED by this kill; (2) log-sum-exp at finite τ (soft selection
  may fix exactly what hard-max broke); (3) tropical RATIONAL (difference of max-plus — lifts
  the concavity restriction the oracle bound shows is the binding one); (4) larger K is NOT the
  fix (agreement flat 16→64 and rate-dead anyway).
- what changes in v5/v6: §11 row 16 "max-plus expansion lever (if P-MP passes)" does NOT enter;
  the row's surviving essence stays exactly the band-residual decomposition (K=1 specials:
  band/clamp/comb); the λ_bytes consequence table gains the 1.5341 S @ K=64 receipt.

## Stragglers (report-only)

### (a) FEED-08l fresh-eyes re-review — verdict: UPHELD-WITH-EVIDENCE-CORRECTION

Re-derived from the durable JSON
(`experiments/results/freq_along_ladder_probe_20260707/freq_along_ladder_n600_20260707.json`):

- CONCLUSIONS UPHELD: net-d_seg ladder is flat (0.00731211 / 0.00755527 / 0.00726753 /
  0.00715998 / 0.00713790 across f_along 0→32; spread 4.2e-4, non-monotone at the start);
  cCOMB best (0.00695123); "raise freq_along" NOT a confirmed simple win. S1's lane_carried
  demotion may stand on the corrected evidence.
- EVIDENCE CORRECTION (the recovery-written verdict mis-read its own JSON): the
  `band_scoreable`/`gt_sep` gates are indexed by FORWARD-DISTANCE RANGE BAND (4–10m: gt_sep
  0.1894 · 10–20m: 0.2557 · 20–35m: 0.0224 · 35–55m: 0.0113 · 55m+: nan), NOT by freq rung.
  The memo's table attributed the range-band failures to conditions cF16/cF25/cF32 ("NO,
  gt_sep 0.022/0.011/nan") — wrong axis. ALL FIVE rungs are d_seg-scoreable at n600 (n_done
  600 each); the gt_sep gate applies to the CLOSURE metric per range band. This STRENGTHENS
  the flat-ladder claim (5 scoreable points, not 2) while limiting the closure analysis to
  ranges <20m (beyond 20m even GT has no mark/gap contrast — registration undecidable there).
- MISSED STRUCTURE (new, from the same JSON): the flatness is a COMPENSATED TRADE, not
  absence of effect — contrast-closure toward the comb rises monotonically with f (band 4–10m:
  0.149→0.908→0.959→0.978→0.985; band 10–20m: 0.173→0.376→0.672→0.810→0.854) and dash-gap FP
  falls monotonically (0.0013246→0.0007200→0.0005516→0.0004814→0.0004550 vs cCOMB 0.0003600),
  while lane_recall degrades (0.7948→0.7353). Higher along-frequency buys comb-like contrast
  and gap-FP but pays it back in recall — net zero. Consistent with req-K format-not-capacity:
  the dash wants the comb's REGISTRATION, not its spectrum.
- STANDING CAVEAT (was honest in the original): oracle-injection form only; every injected
  condition is WORSE than the plain witness (c1_witness 0.00314626) — comparative ranking
  within a degraded family; form-a (retrain) remains the only in-training discriminator.

### (b) Comb-registration audit — deciding measurement, and it is $0

- What is at stake: "cCOMB best 0.00695" + the gap-FP removal are load-bearing for the comb
  lever; the comb gate is ANALYTIC (line-fits × global ego-phase, never GT-conditioned) —
  mis-phase would gate OFF real lane marks (recall loss scored against real lane).
- DECIDING MEASUREMENT (both halves on the frozen ep650 probe state + gt cache, the existing
  probe-render instrument — $0/CPU, but each condition is a full n600 render-score pass, i.e.
  the cost class of the ladder probe itself, NOT this session's <10-min budget → report-only):
  (1) REGISTRATION SCORE: P(comb-gate ON | GT mark px) − P(comb-gate ON | GT gap px) on the
  4–20m range bands only (beyond 20m GT itself has no mark/gap contrast — measured above);
  (2) PHASE-SWEEP CONTROL: re-render cCOMB at shifted comb phases (+T/4, +T/2, +3T/4) and
  score n600 — if measured-phase is not strictly best on {d_seg, gap-FP, recall}, the comb's
  win is a gating-AREA artifact (any mask that removes band area removes gap FP at recall
  cost), not registration.
- New receipt from this session's re-derivation: THIS JSON reproduces the comb's gap-FP
  removal vs solid at **79.3283%** (0.0017417→0.0003600), not the compendium's 86%; cCOMB's
  own gap_FP reproduces FEED-08c c3 EXACTLY (0.000360) → the 86%-vs-79% discrepancy sits in
  the SOLID-baseline construction between probes, not in the comb condition. The audit should
  pin one solid baseline before the number is cited again.

---

## Verdict summary (scopes per requirement R)

| probe | verdict | scope | consequence |
|---|---|---|---|
| Q1 | BETWEEN: no-fire (max abs rho 0.1242 < 0.3), no-kill (3 sides >= 0.1) | FORMULATION non-fire | B16 stays default-off; asymmetry mechanism real but weak; sR-at-chance new row |
| P-CON | KILL 0.4408/0.5636 << 0.80 | FORMULATION (raw tau*ln5) | fitted s=21.75/3.75 => ~tau-INDEPENDENT absolute bar 1.30-1.75 logit; B17 ships fitted form; lane = the whole failure |
| P-DZ | FIRES: 1.5795e-3 d_seg-eq = 88.7x bind band | census (no kill) | #149 DEFER -> duty queue; far-range lane + horizon + hood; SC-16 g_I seed exists |
| P-MP | KILL: agreement ~0.41 << 0.95 at K<=64 | FORMULATION (concave-max K<=64 @ annulus) | capacity is NOT the binder (oracle rms ~=0.0998 coupling) — the max-envelope selection is; band-residual K=1 specials unaffected |
| FEED-08l | UPHELD-WITH-EVIDENCE-CORRECTION | — | scoreability axis mis-mapped (range bands, not rungs); flat ladder now 5-point; compensated-trade structure found |
| comb-reg | measurement named, $0 (but n600-pass cost class) | — | phase-sweep + registration score, 4-20m only; 86% -> 79.33% on this JSON (solid-baseline discrepancy) |

Most consequential number: **1.5795495775e-3 d_seg-equivalent** of the current flip mass
(38.37% of all flips) sits in the uint8 deadzone — 88.7× the duty-queue band; the #149
sub-pixel/dither class graduates from DEFER to a first-order lever with measured headroom two
orders of magnitude above the crossing margin.

