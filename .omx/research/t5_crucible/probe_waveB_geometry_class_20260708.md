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

(pending — instrument committed, run follows this checkpoint)

## P-MP — max-plus annulus fit

(pending)

## Stragglers

(pending)
