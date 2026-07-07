---
doc_type: t5_crucible_position
seat: S5 (LEVER-LEDGER — Fridrich/Yousfi charter)
date: 2026-07-07
anti_anchoring: no other position_S*.md read
axis_tag: "[$0 read/analysis probes only; every number labeled; pointer 0.19110 UNMOVED — MEANS]"
---

# S5 POSITION — NEVER-FIRED BUILD LEDGER RESOLUTION (36/36) + scorer-geometry ranking

## Position

### 0. The headline theorem (drives the whole ranking)

**Island birth is NECESSARY for T_3 — this is arithmetic, not preference.** MEASURED (FEED-07c,
`lane_share_probe_ep225_n600.json` in the mod32cap run dir, n600, [macOS advisory]): the un-born
island region (Movable 44.8% + Lane 19.1% of d_seg; 63.9% of ALL flips on 1.8% of px) plus big-3
boundary jitter (Road 20.3 / Undriv 12.7 / MyCar 3.1 ≈ 36.1%). mod32cap best d_seg = 0.0033662
(MEASURED, ep650 EMA). Perfect removal of ALL big-3 jitter with zero island birth leaves
d_seg ≈ 0.639 × 0.0033662 ≈ **0.00215 > the 0.00092 T_3 need**. Caveat WELDED: shares are
witness-alone ep225 upper bounds (probe self-flags), but the within-class un-born fractions
(lane 83.9% / movable 93.1%) independently transfer (live part_frac lane=movable=0 all run).
⇒ the ledger resolution below is island-first, basis-second, loss-geometry-third, everything else
conditioned. ~97% of d_seg mass in the ~4.7%-area annulus (#333) says the SAME thing from the other
side: recovery-per-byte is maximized by levers that act ON the annulus (birth + boundary position),
not on bulk capacity.

### 1. Ledger-semantics ground truth (honesty-critical, verified against run configs)

The activation ledger (`.omx/state/lever_activation_ledger.jsonl`, 7 rows: DashComb fired+measured,
LaneBandResCoder fired+measured, WeightEntropyPenalty torch-backfill) records ONLY
`--dsl-lever`-path launches (VERIFIED-VIA-SOURCE `tools/launch_witness_run.py:1075-1082`). Every
historical run launched via generated raw-flag `launch.sh` → recorded nowhere. Ground truth from
`experiments/results/levelset_n600_witness_*/launch.sh` + `run.log`:

- **Raw-flag FIRED (ledger wrong):** Muon (mod32cap @726 + #205), PoseDecouple (mod32cap IS
  `--w-pose 0`), AnalyticLaneRenderBand (#205 `--lane-render-band` active from ep300, log stage
  `lane_render_band` recall 0.5475), AmplifyIsland (#205 + v5/v6, weight 1.0 UNIFORM),
  PersistenceTopology (#205 + v5/v6, weight 1.0), SeedIslandBirth components (`--seed-islands`
  paintseedON + v5/v6; `--witness-alone-island-loss` v5/v6 arms), EikonalViscosity (v6 arms 0.3
  anneal/adaptive), StiefelW (`--film-stiefel` v5/v6), BoundaryDistance (0.2, v5/v6),
  CacheGtSkeleton (v6 arms), WarpRealLumaFrame0 mechanism (#205 ran `--w-pose 1.0` + the
  store-nothing dispatch).
- **Flag-present but mechanism NEVER ENGAGED:** MuonWarmStart (in v5/v6 launch.sh, but those runs
  froze/died before ep726 — the l7→Muon switch never happened).
- **No-op emission:** AdamBeta2 (runs emitted 0.999 = the trainer DEFAULT,
  VERIFIED-VIA-SOURCE `train_levelset_witness_realized_through_R_mlx.py:7387` — never fired at a
  non-default value).
- **Staged-but-never-executed:** the islands_treatment dir
  (`levelset_n600_witness_islands_treatment_20260707T000000Z/`) contains ONLY a launch.sh
  (SeedIslandEased, EventTriggeredCurriculum, MuonWarmStart, SegFocalGamma 2.0, amplify 1.0,
  witness-alone pair) — zero logs/ckpts. Those levers did NOT fire.
- **Genuinely never fired anywhere:** AACoverageRender (`--render-aa none` in every run),
  DirectionalBasis (lane-edge loss), DirectionalBasisRebalance, LanePrior (lane-thin), FiLMFix,
  CodeSpectralEntropy, DM1Minimal, MarginSaliency, UniWARD, GroundFrameChart, TauFrozen,
  SoftBoundary, StepNativeActivation, FinerBiasInit, LengthSigma, LogitAdjust, MarginFieldHead,
  WeightEntropyPenaltyMLX (MLX port), MicroBatch (n600; n24 governed probe only), SegFocalGamma,
  SeedIslandEased, EventTriggeredCurriculum, Mod32SegOnlyControlBase, AdamBeta2 (non-default).
- **The grounding packet's caveat list is itself partly wrong:** it names MicroBatch + LanePrior as
  "demonstrably fired" — MicroBatch never appears in any n600 launch.sh (its firing was an n24
  governed throughput probe, FEED 8283); "LanePrior" conflates the DSL lever (`--lane-thin-*`,
  never fired) with `--lane-prior-phi1` (a different mechanism, fired-and-measured-no-op in
  `replace` mode). Per-lever ground truth below.

**CRITICAL evidence caveat (Fridrich charter):** every v5/v6-era raw firing (20260704–20260705)
is NON-LOAD-BEARING — those runs were spike-guard median-frozen (ep92–114, `ep_loss: 0.0`,
FEED-05j) and 18-confound-era. "Fired in a poisoned run" earns a `fired` ledger row, NEVER a
`measured` verdict. #205 (20260703T120444Z) was live (descended to 0.004752@ep300) but ran the
FULL stack simultaneously — nothing in it is attributed per-lever.

### 2. THE 36-ROW RESOLUTION

Verdicts: **FIRE** (built; needs an activation arm) · **BUILD** (missing piece then compose) ·
**DEFER(reason)** · **RETIRE(reason)** · **BACKFILL** (fired-in-fact; ledger correction only).
Bands are pre-registered PREDICTED Δ(seg-term = 100·d_seg) or ΔS at matched epochs vs the
Mod32SegOnlyControlBase control unless noted; all advisory-surface until byte-closed exact rows.
Rank = expected annulus-mass recovery / byte / training-cost.

| # | Lever (flags) | Ground truth fired? | Measured evidence | Verdict | Band · measurement · kill |
|---|---|---|---|---|---|
| 1 | **SeedIslandBirth** (`--seed-islands` + `--witness-alone-island-loss` PAIR) | Components raw-fired: seed in paintseedON (live) + v5/v6; wa-island in v5/v6 (poisoned). Never as clean DSL pair | paintseedON ep0 d_seg −36% (0.474 vs 0.744, MEASURED DAG 8366) but part_frac[lane] still 0 at init + ~8× slower/ep (#300 co-grad); wa-island lane within-flip −45% while d_seg 0.162→0.122 (T3 memo; frozen-era caveat); probe: PROCEED-class (lane 83.9%/movable 93.1% un-born) | **FIRE** — the treatment-arm core, paired with #315 guard (T3 REVISE satisfied: probe landed PROCEED); FRESH not warm-start (CE-converged basin has ≈0 island gradient) | Movable-first band: Δseg −0.05..−0.11 (44.8% share × 30–70% birth). Measure: matched-epoch A/B vs control + per-class within-flip. Kill: movable un-born still >80% by ep150 OR total d_seg > control at ep100 gate |
| 2 | **AnalyticLaneRenderBand** (`--lane-render-band` + 7 `--lane-band-*`) | RAW-FIRED in #205 (active ep300+, recall 0.5475 logged) — packet's "never fired" is WRONG for the mechanism; never fired ISOLATED | Offline analytic lane authority d_seg 0.00087 (V-A packet); post-hoc compose on mod32cap ep425 ≈0/slightly WORSE on flips (DAG 9054) ⇒ value ONLY trained-with; naive form HURT +0.00082, non-naive ~break-even post-hoc (factory); counted payload MEASURED: LBND2 41,526 B → LBND4 30,892 B, rate_term 0.0206 (ledger LaneBandResCoder row) | **FIRE (trained-with, isolated arm)** — the lane lever; rule-118 generator free but coeff payload is NOT near-zero (0.0206 rate) — packet's "near-zero byte" needs this correction | Δseg −0.02..−0.05 (lane 19.1% × band recall 0.55) MINUS rate +0.0206 ⇒ net −0.005..−0.03. Measure: trained-with vs control matched-epoch + lane-flip attribution + REAL LBND4 bytes. Kill: net ΔS ≥ −0.005 at ep300 checkpoint |
| 3 | **EventTriggeredCurriculum** (`--curriculum-event-triggered --curriculum-nucleus-guard`) | Staged only (islands launch.sh, unexecuted) | #315 wired end-to-end VERIFIED (T3 memo: L1508/L4417, resume-persisted, cap-ceiling ⇒ byte-identical unfired); tau erosion MEASURED in #205 (0.004752→0.006568, +38%) but ABSENT in mod32cap (clean descent) — guard = insurance for BORN islands, not refuted | **FIRE** with row 1 (the PAIR is the only config where birth pressure pays — T3 lens B) | ~0 solo; enabler. Measure: tau-onset epoch + per-class nucleus stats telemetry. Kill: none needed (byte-identical when unfired); ABORT if handoff never fires by cap |
| 4 | **SeedIslandEased** (`--seed-island-eased`) | Staged only; wired 705afea84, default-OFF byte-identical | Movable SDF-dilation = PROVEN 1-Lipschitz transfer (GO independent of probe); lane VP-tangent manifold-preserving (isotropic-of-a-curve NO-GO avoided) | **FIRE** as modifier on row 1 (movable half unconditional; lane half sized by 19.1% share) | Rides row 1's band. Measure: eased vs plain seed A/B if row 1 pays. Kill: inherits row 1's |
| 5 | **LogitAdjust** (`--logit-adjust-loss-tau`) | Never (built 2026-07-07) | Priors MEASURED n600 [.232,.0059,.495,.0124,.254]; lane/movable log-priors −5.13/−4.39 = gradient aimed exactly at the un-born mass; byte-identity boundary PROVEN (τ=0 same-object; deployed argmax reads RAW logits) | **FIRE** in the islands arm (zero-byte, deterministic, textbook rare-class cure) | Δseg −0.01..−0.05 composed (solo unproven). Measure: island grad-share + within-flip vs control. Kill: total d_seg regression >2% at matched epoch. CAVEAT: fails closed with micro-batch>1 (moot — MicroBatch retired, row 32) |
| 6 | **AACoverageRender** (`--render-aa supersample --aa-supersample 2`) | NEVER (every run `--render-aa none`) — genuinely never-fired | "#1 measured islands lever" per the #220 gate ($0 probe, V-A factory/FEED-07c); compose-after-downsample guard RESOLVED 2026-07-07 (composes with seed/band/residual); self-orient×supersample fine-mode memory gate still applies | **FIRE** — sub-pixel island coverage is the render-side half of birth; separate arm or composed after the fine-mode gate is checked | Δseg −0.01..−0.04 (sub-pixel coverage of 1.8%-px island region). Measure: matched-epoch A/B; memory preflight FIRST (ss=2 ⇒ 4× render px). Kill: wall-clock >1.5× control at equal Δd_seg ≤ noise |
| 7 | **AmplifyIsland** (`--amplify-weight`) | RAW-FIRED #205+v5/v6 at weight 1.0 UNIFORM | UNIFORM form = the measured net-negative class (full-stack 0.121 / paint-seed 0.026, T3 memo); margin-GATED support = net-positive BY CONSTRUCTION (Δd_seg ∝ n_big3−n_isl) but the hard costate-gated arm is DESIGNED-ONLY (reachability sensor inert, #268 owed) | **FIRE gated / DEFER un-gated** — fire ONLY the #300-soft-gated form (witness-alone) inside row 1's pair; the hard margin-gate is a BUILD behind #268 | Rides row 1. Kill: island share of d_seg RISES (over-amplification signature) or big-3 within-flip degrades |
| 8 | **DirectionalBasisRebalance** (`--self-orient --n-dir-freqs 4 --freq-across/along`) | NEVER. NOTE: mod32cap ALREADY runs self-orient/n-dir-freqs 4/across 32/along 8 — the −48% directional anchor is BANKED in the base; this lever's MARGINAL content is only the along-freq realloc | FEED-08l oracle ladder (n600, $0): d_seg FLAT across along 0→32 (0.00731→0.00714), comb BEST (0.00695) ⇒ "raise along" NOT confirmed; ≥16 INDETERMINATE-at-resolution; lane_offloaded √-law ASSUMED_AWAITING (registered eq) | **FIRE lane_offloaded regime only (along 32→6), coupled to row 2**: if the band carries lane, Candès–Donoho says spend along-budget down; **DEFER lane_carried (along 26)** — oracle probe leans against; in-training comb A/B (FEED-08c) is the lane arbiter | Δseg −0..−0.02 marginal (honest: the big win is already in base). ~0 byte. Measure: form-a retrain A/B at along∈{6,8} under band-ON. Kill: along-6 worse than along-8 at ep300 |
| 9 | **MuonWarmStart** (`--muon-warm-start-momentum --muon-lr-final-frac 0.1`) | Flags in v5/v6 launch.sh but runs died pre-ep726 ⇒ mechanism NEVER ENGAGED | +8% cold-Muon transient MEASURED (factory/FEED-07b); corroborating: mod32cap cold-Muon finisher ENDED WORSE than its own tau-stage best (final ep1000 ≈0.0036 vs 0.00337@ep650 — MEASURED, grounding packet) | **FIRE-as-is** — free, default-off, fires at l7→Muon switch; in EVERY next arm | Δseg −0.003..−0.01 (recover the transient + finish below tau-best). Measure: Muon-stage best vs tau-stage best within-run. Kill: none (byte-identical unfired); drop if Muon-stage still > tau-best twice |
| 10 | **LengthSigma** (`--length-sigma-matrix fitted-20260707`) | NEVER (built; fit landed 3571e5b65) | DERIVED from the frozen scorer's OWN junction geometry (Young's-law fit: σ[Road-Lane]=0.377 [0.317,0.441]) — uniform length weight over-penalizes lane boundary ~2.7× = a NAMED lane-erasure mechanism. The anti-cargo exemplar | **FIRE** — zero-byte, loss-geometry, scorer-derived; composes with everything | Δseg −0.005..−0.02 (lane-boundary length tax removal). Measure: junction-local d_seg attribution A/B (the registered owed anchor). Kill: total d_seg regression at matched epoch |
| 11 | **Muon** (`--muon-start-epoch` etc.) | **FIRED-in-fact** (mod32cap @726, #205) | −32% d_seg vs AdamW (measured anchor); cold-start transient +8% | **BACKFILL** fired+measured; stays in base config; tuning = row 9 | n/a (base) |
| 12 | **PoseDecouple** (`--w-pose 0`) | **FIRED-in-fact** — mod32cap IS the pose-decoupled control | mod32cap = d_seg-COMPONENT run by design (council clean-config memo) | **BACKFILL**; control-arm marker, not a build item | n/a |
| 13 | **Mod32SegOnlyControlBase** (8 deltas over proven_base) | Marker; the config it encodes RAN (mod32cap) | Control best 0.0033662@ep650 (MEASURED) | **FIRE-as-base** — the composition base of every treatment arm (clean matched-epoch A/B); trivially "fires" with any arm | n/a (base reproduction) |
| 14 | **CacheGtSkeleton** (`--cache-gt-skeleton`) | Raw-fired v6 arms (frozen-era) | BIT-IDENTICAL PROVEN (#260 n=8 CPU A/B: EMA max_abs=0) | **FIRE-as-is** whenever PersistenceTopology on (pure speed); no-op otherwise | n/a (bit-identical). Sister note: dash-COMB registration audit (FEED-08l/L65) belongs to DashComb (already fired+measured in ledger; render-composite net-negative +0.0038 ⇒ comb must be IN-TRAINING; mis-phase risks real lane — the in-training `--lane-band-dash-comb` arm stays owed and REGISTRATION-AUDIT-gated) |
| 15 | **PersistenceTopology** (`--persistence-loss-weight`) | RAW-FIRED #205+v5/v6 (w 1.0, warmup 300) unattributed | In #205's full stack, lane was partially birthed then tau-eroded (DAG 8014); never isolated; clDice targets exactly the 1/persistence erasure tail | **FIRE in the islands arm only** (with row 3's guard so births aren't taxed); DEFER solo | Rides rows 1–3. Kill: wall-clock (clDice cost) >1.3× control at Δd_seg ≤ noise — mitigate w/ row 14 |
| 16 | **SegFocalGamma** (`--seg-focal-gamma`) | Staged γ=2.0 (unexecuted) | **CONFLICTED measured record:** #301 calibration (FEED-05h, seeded run ep50/75, gt_n24): γ*=0 HOLD — focal is a weak bulk-boundary reallocator, NON-monotone island share (γ=3 < pixel share), γ=1 least-harmful. FEED-07c (mod32cap ep225 n600): ANALYTIC island-weight share γ=2→0.599. Different surfaces (analytic weight vs realized grad), different ckpts | **DEFER-until-recalibrated**: #301's binding rule — "any arm re-runs the probe on ITS checkpoint first." The staged γ=2.0 in islands launch.sh VIOLATES the calibration; fire only at the per-ckpt γ* (likely 0 or 1) | $0 probe first (RECESS R5). Kill: realized island grad-share gain < +0.5pp at chosen γ |
| 17 | **BoundaryDistance** (`--boundary-distance-weight`) | Raw-fired 0.2 in v5/v6 (poisoned ⇒ non-load-bearing) | w_bd* 9.05%-of-total calibration (frozen-era run, weak); #301 names `0.05` as the geometry-native A/B, UNMEASURED; Kervadec SDF-native term = annulus-aligned by construction | **FIRE small (0.05) as the loss-geometry A/B** after rows 1–10 land, or compose in round-2 — the family (16/17/5/18) must not all fire at once (attribution) | Δseg −0..−0.02. Kill: matched-epoch regression; family cap: ≤2 loss-geometry levers per arm |
| 18 | **MarginFieldHead** (`--margin-field-head-weight`) | NEVER | #218 byte-free head; shares the `_signed` margin surface with rows 5/16/17 | **DEFER(round-2)** — loss-geometry family crowding; fire after LogitAdjust+LengthSigma measured | Named reason: attribution budget (max 2/arm) |
| 19 | **StepNativeActivation** (hosc β 4→8 anneal) | NEVER (mod32cap anneals 1→4; this extends toward step limit) | Deep-math L∞-at-edge optimality (registered eq); best non-step 0.004445 ≈ 4.4× need (V-A); fixed high-β diverges (MEASURED) — anneal only | **FIRE paired with FinerBiasInit** as the #310 sweep arm (capstone lever #5, the named headroom item) | Δseg −0.01..−0.05 (edge-error confinement). Measure: β_end∈{4,6,8} short ladder then best-arm full. Kill: any divergence signature (gnorm/liveness) or worse than 1→4 at ep200 |
| 20 | **FinerBiasInit** (`--finer-bias-init --finer-bias-k 10`) | NEVER (built 2026-07-07) | Published fix (FINER++ 2407.19434) for the MEASURED fixed-β saturation-death; dedicated RNG stream, byte-identical OFF | **FIRE with row 19** (from-scratch arms only — resume overwrites it, trainer stamps applied:false) | Rides row 19. Kill: inherits |
| 21 | **WeightEntropyPenaltyMLX** (`--weight-entropy-penalty-lambda`) | NEVER on MLX (torch sibling fired+measured, ledger-backfilled) | Torch: λ50 cut live-decoder −19.6% bytes / −1.55 bits/wt; EMA-lag caveat; ema0.9 proof; λ* open {5,15,30}; DOES NOT TRANSFER (NO-FAKE firewall welded in factory). Rate stake: blob ~99KB ⇒ rate term ~0.066; λ_bytes 6.659e-7 S/B | **FIRE in the byte-close/finishing arm** (λ=15 center) — the rate half of S the seg levers can't touch | ΔS −0.005..−0.015 via bytes at equal d_seg. Measure: measured_symbol_entropy_bits + REAL quantize_levelset_blob bytes, λ-on vs 0. Kill: d_seg harm > byte win (λ50-class overshoot) |
| 22 | **DirectionalBasis** (lane-edge LOSS term `--lane-edge-weight`) | NEVER (BASELINE holds 0; absent from mod32cap argv) | Lane-only directional loss measured weak (−8% vs −48% all-class, packet); superseded in spirit by rows 2+8 | **DEFER(dominated)** — if band carries lane, a lane-edge hinge is redundant; revisit only if row 2 kills | Named reason: dominated-by-band |
| 23 | **LanePrior** (thin-lane `--lane-thin-*`) | NEVER (grounding packet's "fired" claim conflated it with `--lane-prior-phi1`) | Design basis measured (57% Road↔Lane confusion, 52.7% lane components missed) but same target as row 2 at higher training cost | **DEFER(dominated-by-band)**; reactivate if row 2 nets ≥ −0.005 but lane within-flip stays >50% | Named reason + reactivation criterion recorded |
| 24 | **StiefelW** (`--film-stiefel`) | Raw-fired v5/v6 (frozen ⇒ non-load-bearing) | FiLM PR collapse MEASURED (3.34→1.19, 91.8% var in one axis) on the OLD vehicle/runs; unknown on the clean mod32cap | **DEFER-pending-$0-telemetry** (RECESS R4: recompute FiLM PR on mod32cap stage ckpts; fire DM1Minimal only if collapse reproduces) | $0 probe decides; band if fired: Δseg −0..−0.03 |
| 25 | **CodeSpectralEntropy** (`--code-spectral-entropy-weight`) | NEVER | Same design memo as row 24 (DM1b) | **DEFER** with row 24 (fire as DM1Minimal pair if PR-collapse reproduces) | — |
| 26 | **DM1Minimal** (composite 24+25) | NEVER | — | **DEFER** (= rows 24/25) | — |
| 27 | **FiLMFix** (`--film-per-layer --film-concat-code`) | NEVER | Third competing cure for the same collapse | **DEFER**; the $0 telemetry picks ONE of {24+25, 27} — never both (double-cure confound) | — |
| 28 | **MarginSaliency** (`--margin-saliency-*`) | NEVER | Capacity-routing pays ONLY after basis-match (measured lever ranking #2); late-stage by design | **DEFER(sequencing)** until rows 2+8 land and pay; then fire late-stage (l7/Muon window) | Named reason: basis-prior-to-capacity (measured) |
| 29 | **UniWARD** (`--margin-saliency-uniward`) | NEVER | The texture-saliency proxy (msal_uni) MEASURED AT CHANCE vs through-R reachability (memory L76; #268 exact θ-indep S_R owed). On-theme (the contest IS inverse steg — my charter) but firing a chance-level sensor is theater | **BUILD(#268 exact S_R) then FIRE**; DEFER until the sensor is real. Fridrich-seat honesty: I want this lever most and it is the one I must hold back | Post-#268 band: Δseg −0..−0.03 (error-hiding in texture). Kill: S_R-gated saliency still ≤ chance vs reachability |
| 30 | **GroundFrameChart** (`--ground-frame-chart` + gfc-*) | NEVER | v0 FAIL-CLOSES with `--self-orient` (V-S factory) — and every base config uses self-orient ⇒ structurally unfireable today | **DEFER(incompatible-as-built)**; BUILD = self-orient-in-chart-coords composition; revisit with pose face (dual-use ξ) | Named reason recorded; no band until composable |
| 31 | **EikonalViscosity** (`--eikonal-viscosity` float + adaptive) | Raw-fired v6 (0.3) — the FROZEN runs; verdicts non-load-bearing (confound hunt) | Baseline eik-weight=0 BY DESIGN (scorer reads only the zero level set — council memo); viscosity is INERT without eik>0 | **DEFER(inert-in-baseline)** — belongs ONLY inside the pre-registered Ballé eik-0.01 A/B arm, which is itself queued behind higher-band arms | Named reason: dominated queue position; kill for the eik arm: R-roundtrip d_seg gap arbitration per Ballé dissent |
| 32 | **MicroBatch** (`--micro-batch-pairs`) | NEVER in n600 runs; MEASURED in n24 governed probe (FEED 8283) | B∈{1..4} noise-swamped, B=8 SLOWER at the real 384×512+wa config; "2–4×" did NOT transfer; also fails closed with seed-islands + logit-adjust | **RETIRE(measured-no-win)**, reactivation: >15% wall-clock win on a clean multi-epoch median or smaller-render regime | Ledger: record retired-with-reason (drains the queue honestly) |
| 33 | **SoftBoundary** (fixed hosc β=2) | NEVER | Fixed-β class is the measured divergence family; sub-pixel-softness hypothesis is served properly by row 6 (real AA coverage) and the 1→4 anneal | **RETIRE(superseded)** — anneal + AA cover both mechanisms; reactivation: none foreseen | — |
| 34 | **TauFrozen** (temp start==end) | NEVER | Diagnostic isolation arm for l7-vs-tau attribution; l7 now parked at 1001 ⇒ the dispute is moot | **DEFER(diagnostic; moot)** — keep as an attribution tool, no duty-to-measure | — |
| 35 | **AdamBeta2** (non-default β₂) | NEVER at non-default (0.999 emissions were the default = no-ops) | No witness-side measurement; β₁<√β₂ guard exists | **DEFER(low-band micro-sweep)** — optimizer β₂ is a generic knob; Muon owns the finisher; spend attribution budget on islands | Named reason: expected Δ ≪ arm cost |
| 36 | **WarpRealLumaFrame0** (`--w-pose>0` + carrier wire-in) | Mechanism RAN in #205 (`--w-pose 1.0`, store-nothing dispatch; d_pose descended to ~0.0019–0.0023 advisory in tau stage — log rows) | Pose OPEN on the witness (L68/L69); warp floor s_t_fit best 2.562; §5B = the pose seat's ON/OFF decision | **BACKFILL fired (engagement-verified via run.log)**; verdict on FIRING NEXT belongs to the §5B pose face — my ledger only certifies the ground truth | Interface row (see §Interfaces) |

**Composition into arms (my recommendation to the synthesis):**
- **ARM-ISLANDS (the primary):** rows 13+1+4+3+5+7(gated)+9+10+15 — FRESH, seg-only, γ per row 16's
  probe. Predicted composed band: Δseg −0.08..−0.20 vs control at ep650-matched (ceiling −0.24 per
  FEED-07c arithmetic, upper-bound caveat welded). This is the only arm whose ceiling reaches T_3.
- **ARM-LANE-BAND (isolated):** rows 13+2+8(lane_offloaded)+9. Net ΔS band −0.005..−0.03 incl. the
  measured +0.0206 rate.
- **ARM-STEP (paired):** rows 13+19+20+9. Independent axis; cheap ladder first.
- **FINISH/RATE:** row 21 + LBND4 inflate inlining (interface to the rate face) at byte-close.

### 3. THE APPARATUS GAP + minimal fix (RECESS/build item R1)

**Gap (VERIFIED-VIA-SOURCE):** `activation_ledger` is populated only by
`launch_witness_run.py:1075-1082` on explicit `--dsl-lever` args. Raw-flag launch.sh runs (ALL
historical runs) record nothing ⇒ the "36 never-fired" is ledger truth, not run truth.
**Minimal fix, 3 parts (~1–2h build, $0):**
1. **Launcher reverse-map:** after argv emission, map argv→levers via
   `lever_registry.lever_factories()` (each factory's emitted-flag set) with an ENGAGED predicate
   (flag present AND value ≠ trainer argparse default — kills the AdamBeta2-0.999 false-positive
   class); `record_activation(fired, reason="raw-flag reverse-map")` for every match, on EVERY
   launch, not just `--dsl-lever` ones.
2. **One-shot backfill tool** (`tools/backfill_lever_activation_from_runs.py`): scan
   `experiments/results/levelset_n600_witness_*/launch.sh` + `run.log`; launch.sh-only dirs (the
   staged islands arm) → NOT fired; runs with logs → `fired`; engagement-verify stage-gated levers
   against run.log stage rows (MuonWarmStart needs the l7→Muon switch row; band needs the
   `lane_render_band` row) else annotate `engagement-unverified`; `measured` ONLY where a verdict
   artifact exists AND the run was live (liveness stamp) — v5/v6 frozen-era rows get
   `fired` + reason `poisoned-run, non-load-bearing` and STAY on duty-to-measure.
3. **Close-the-loop hook:** the byte-close tool calls `record_measured_for_run` (already built,
   `activation_ledger.py:229`) — wire it into `tools/levelset_byte_close_and_eval` on verdict land.

## Derivations + assumption tags

- Ledger records only --dsl-lever launches: VERIFIED-VIA-SOURCE (`tools/launch_witness_run.py:1075-1082`; `activation_ledger.py:16-23`).
- Ledger current state 7 rows / 36 owed: VERIFIED-VIA-ANCHOR (`.omx/state/lever_activation_ledger.jsonl`, read inline; `known_levers()`=37, `never_fired()`=36 executed inline).
- Per-lever fired ground truth: VERIFIED-VIA-ANCHOR (grep of every `levelset_n600_witness_*/launch.sh` + `run.log` of 20260703T120444Z, executed inline this session; islands_treatment dir = launch.sh only, ls verified).
- #205 band/persistence/amplify/pose-carrier active; part_frac lane=movable=0 @ep0: VERIFIED-VIA-ANCHOR (run.log stage rows quoted in §1).
- AdamBeta2 0.999 = trainer default: VERIFIED-VIA-SOURCE (`train_levelset_witness_realized_through_R_mlx.py:7387`).
- Island shares (44.8/19.1/63.9%), un-born fractions (83.9/93.1%), γ analytic surface, birth ceiling 0.0037→0.0013: VERIFIED-VIA-ANCHOR (DAG FEED-07c + `lane_share_probe_ep225_n600.json` path; upper-bound caveat WELDED as in source).
- Necessity theorem (0.00215 > 0.00092 without birth): DERIVED from the two anchors above; ASSUMES flip-share stability ep225→ep650 (unverified — RECESS R2 measures it; verdict PROVISIONAL on that assumption per #363, though the within-class un-born transfer is independently confirmed).
- mod32cap best 0.0033662@ep650, final ≈0.0036; Muon −32%; +8% cold transient: VERIFIED-VIA-ANCHOR (grounding packet shared facts + factory notes).
- Band post-hoc ≈0/negative at ep425; LBND4 30,892 B / rate 0.0206: VERIFIED-VIA-ANCHOR (DAG 9054; ledger LaneBandResCoder measured row).
- #301 γ*=0 / γ=1 least-harmful / bd 0.05 unmeasured: VERIFIED-VIA-ANCHOR (DAG FEED-05h/8100-8104).
- FEED-08l along-ladder FLAT, comb best: VERIFIED-VIA-ANCHOR (DAG 10030-10040).
- MicroBatch no-win: VERIFIED-VIA-ANCHOR (DAG 8283 verdict block).
- v5/v6 runs frozen/poisoned ⇒ non-load-bearing: VERIFIED-VIA-ANCHOR (FEED-05j deadlock row; confound-hunt memory L4/L5).
- UniWARD proxy at chance: VERIFIED-VIA-ANCHOR (memory L76; #268 owed).
- GFC fail-closed with self-orient: VERIFIED-VIA-SOURCE (factory docstring, curriculum_dsl.py:1809-1845).
- All predicted bands: INFERRED (share × plausible recovery fraction, Dykstra-feasible: no band exceeds its measured flip-mass ceiling; each carries a kill threshold). None is a score claim; pointer moves only via byte-closed exact rows.

## PR95 cargo-cult audit (my face)

- Muon@726 + cold finisher: INHERITED placement (PR95 stage-8 echo). Kept-with-derivation only via
  row 9 (warm-start + LR anneal = the witness-derived correction; the measured mod32cap
  finisher-worse-than-tau-best is the smoking gun that the verbatim PR95 placement is suboptimal).
- AdamBeta2 sweep: generic-optimizer reflex, not witness math → DEFER (audit-consistent).
- SegFocalGamma/BoundaryDistance: LITERATURE-cargo risk (standard segmentation imports).
  JUSTIFIED-KEPT only via the #301/#331 measured calibration surfaces + per-ckpt recalibration rule;
  the staged γ=2.0 was cargo (contradicts our own calibration) — corrected in row 16.
- LengthSigma: DERIVED-FROM-WITNESS-MATH (frozen-scorer Young's-law junction fit) — the exemplar.
- StepNativeActivation/FinerBiasInit: DERIVED (L∞-at-edge optimality + measured saturation
  mechanism), not PR95.
- Islands family (1/3/4/5/7): DERIVED (nucleus theorem, linear-in-flip-count Δd_seg, measured
  starvation) — the anti-PR95 core of the stack.
- SoftBoundary: DROPPED (superseded); fixed-β family = measured trap.
- UniWARD: our own on-ramp theme, but firing a chance-level sensor would be cargo of our OWN
  branding — held behind #268. (Assumption-Adversary discipline applied to my own charter.)

## RECESS measurement proposals

- **R1 — Ledger backfill + reverse-map (build, $0, ~1–2h):** as §3. Kill/proceed: n/a (apparatus).
  Pre-registered outcome: never-fired 36 → ~22 genuinely-never-fired + ~10 fired-unmeasured +
  2 retired.
- **R2 — Flip-share stability probe ($0 CPU, ~1–3h, <8 GiB, chunked/resumable):** re-run the
  lane-share probe functions on mod32cap ep650 BEST + 2 stage ckpts (same tool as ep225). Predicted:
  island share 55–70% (stable); kill for the necessity theorem: island share <35% at ep650 (then
  big-3 jitter levers re-rank first).
- **R3 — SegFocalGamma per-ckpt recalibration ($0, ~30–60min):** #301 realized-grad probe on ep650.
  Predicted: γ*∈{0,1}; proceed with γ=1 only if island grad-share gain ≥ +0.5pp; else γ omitted from
  the islands arm.
- **R4 — FiLM-PR telemetry on mod32cap stage ckpts ($0, ~30min):** participation ratio per stage.
  Predicted: collapse (PR<1.5 by tau) reproduces → DM1Minimal enters round-2; kill: PR≥2.5 (cure
  family stays deferred).
- **R5 — Composed-surface island arithmetic ($0, ~1h):** the FEED-07c owed "composed-surface
  ceiling" memo before any launch leans on the 0.0013 ceiling. Kill: composed ceiling >0.0025 ⇒
  islands arm band shrinks to −0.04..−0.10 (still #1 by rank).
- (Training A/Bs — ARM-ISLANDS / ARM-LANE-BAND / ARM-STEP — are launches: operator-GO items, spec'd
  in §2's arm table with matched-epoch gates; NOT run by this seat.)

## Interfaces

- **From schedule/curriculum face:** I assume #315's event-trigger semantics (row 3) and the
  rewarmup/reset behavior at the l7→Muon switch (row 9). If that face re-derives stage boundaries,
  rows 1/15 warmups must re-anchor to the derived tau onset, not ep300.
- **From pose face (§5B):** row 36's ON/OFF. Evidence I hand over: #205 ran `--w-pose 1.0` with
  the store-nothing carrier and logged advisory d_pose ~0.0019–0.0023 in tau stage (√(10·d_pose)
  ≈0.14 — 8× above the 0.018 ancestor target) while its d_seg stayed within the full-stack
  confound. Seg-side constraint from me: frame0 is seg-free, so pose-ON cannot disturb d_seg by
  construction — no seg-side veto.
- **From rate face:** LBND4 decode inlining into `_INFLATE_PY` (parity gate fails closed until
  then) is a PREREQUISITE for row 2's byte-close; row 21's λ pilot rides the same byte-close arm.
  The +0.0206 band rate is MY number for row 2's net-ΔS gate — if the rate face shrinks the lane
  payload further (comb world-phase ~2–6 floats vs per-pair fitted), row 2's band improves ~+0.01.
- **From basis face:** row 8's regime switch is COUPLED to row 2's verdict (band carries lane ⇒
  lane_offloaded along≈6; band killed ⇒ lane arbiter is the in-training comb A/B per FEED-08l, NOT
  along=26).
- **To costate face:** the ledger fix (R1) is what makes duty_to_measure a real SENSE input; until
  it lands, the DECIDE queue is ranking ledger fiction. Also: rows 7/29 both wait on #268 exact
  S_R — one build unblocks two levers.
- **To DSL face:** everything I mark FIRE already holds a validated `Lever` factory (verified in
  `curriculum_dsl.py` this session); the 113 unmapped trainer flags are orthogonal to these 36 —
  no stub-lever needed for my verdicts. Mod32SegOnlyControlBase (row 13) is the composition base
  every arm should compile over (config-generator SoT, no hand-edited launch.sh).

Pointer 0.19110 UNMOVED — everything above is MEANS until a byte-closed `upstream/evaluate.py`
n600 exact row lands.
