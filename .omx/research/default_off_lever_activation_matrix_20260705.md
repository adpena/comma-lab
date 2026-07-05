---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "Two of the three drift rows are bit-identical hygiene; the pose-carrier-source reversion is the only one that can move a scored quantity, and its restore is trajectory-affecting — do not smuggle it in as a 'rider'. It is an operator decision at a FRESH arm, full stop."
council_assumption_adversary_verdict:
  - assumption: "the master lever ledger (fresh_run_master_lever_ledger_20260704 + its Supersessions block) is the authoritative INCLUDE/KEEP gate for the live argv"
    classification: HARD-EARNED
    rationale: "It is the synthesis of sweeps A/B/C + 5 facets + the SEAL review, and its Supersessions block already models argv-over-ledger overrides (S1-S4). Where argv and ledger disagree, the discriminator is a NAMED later audit (SEAL M4, intervention package, FEED-05q) — absent that, it is drift."
  - assumption: "an intervention-package/review-audited argv delta is NOT drift even though the master ledger has no row for it"
    classification: HARD-EARNED
    rationale: "ce_window_intervention_package + its independent adversarial review (PROCEED_WITH_REVISIONS) + FEED-05q GO#3 are chronologically-later audited authority for the v4 resume tail; the ledger's own S-block pattern licenses exactly this supersession form. The residual obligation is LEDGER-side (append the rows), not argv-side."
council_decisions_recorded:
  - "op-routable 1: v5 rider set = --cache-gt-skeleton (drift-restore, bit-identical) + --fused-r-kernel (parity-gated) + --dm1-telemetry (observability); nothing else rides without an A/B"
  - "op-routable 2: pose-carrier-source generated restore = operator decision at the next FRESH arm (trajectory-affecting; matches ledger KEEP + store-nothing byte-close); never flip mid-resume"
  - "op-routable 3: append the v4-intervention rows (bd 0.2, tau 400, band 450, persist 275, lr 5e-4, seed-anneal 101-cosine, wa-island) to the master ledger Supersessions block so ledger==argv again"
related_deliberation_ids: [fresh_run_master_lever_ledger_20260704, per_lever_compute_audit_20260705, ce_window_intervention_package_20260705, council_grand_symposium_curriculum_derivation_20260705]
---

# DEFAULT-OFF LEVER ACTIVATION MATRIX + LEDGER↔ARGV DRIFT AUDIT (exhaustive)

**2026-07-05 · $0 local, read-only (live run pid untouched) · every classification cites its artifact
or says UNMEASURED · Pointer contest-CPU 0.19110 UNMOVED — everything here is MEANS.**
Operator question: *"Are there any other orphaned or default off techniques or levers or flags we
should integrate and wire up and config?"* Ground truth surfaces: master ledger
(`fresh_run_master_lever_ledger_20260704.md` incl. Supersessions S1–S4) × the live v4 argv
(`experiments/results/levelset_n600_witness_20260705T125950Z/launch.sh`, argparse last-wins) × the
trainer argparse (217 add_argument rows, grep-verified) × sweep-A (189-flag verdicts) × sweep-C
(task/orphan ledger) × the compute audit (drift catch #1).

---

## §1. DRIFT CHECK — master-ledger INCLUDE/KEEP rows × the v4 effective argv

**Headline: 3 drift rows (1 previously proven + 2 new), 2 ledger-stale rows, 0 unaudited argv
additions.** All ~34 other INCLUDE/KEEP/Supersession rows MATCH the effective argv (verified
flag-by-flag: seed-islands · paint · dilate 1 · eikonal 0.05→end 0.1 · length 0.001 · geometric-τ +
temp-end 1.0 · mod-19 · film-stiefel · muon 726/0.002/0.95/5 + warm-start + lr-final-frac 0.1 ·
rewarmup 20/0.1/cosine + reset-moments · verdict-batch 64 (S2) · closed-loop (S4) · l7 1001 · no
bank-n-scales flag = default 4 (S1) · verdict-pairs 0 + async · chroma · palette-anchor · self-orient
2/32/4/50 · persistence 1.0 + recall 1.0 + clDice 5 · amplify 1.0/hinge/1.0/inverse_thickness ·
structured-init + include-lane · lane-prior-phi1 + dash-gate · pose-carrier + residual table · EMA
0.997 · w-seg 100 / w-pose 1.0 / score-domain · accum 8 · grad-clip 1.0 · ckpt 25 + stage-checkpoints
· render 384×512 aa=none · lane-render-band + witness/0.85/0.35/1.0/55.0 · hosc 1.0-anneal + siren-init
· lr-schedule/warmup defaults · seed 0 / gt_n600 / epochs 1000 / eval 25).

### 1a. DRIFT — ledger row present, argv absent or diverges (the `--cache-gt-skeleton` class)

| # | flag | ledger says | v4 argv effective | severity / action |
|---|---|---|---|---|
| D1 | `--cache-gt-skeleton` | KEEP (§1 KEEP-list; sweep-A A6 INCLUDE "bit-identical speed") | **ABSENT** → off | KNOWN (compute-audit catch, FEED-05u). Bit-identical by construction + n64 A/B ⟹ zero-risk rider. Recovers part of the +47 s/ep tau-stage group from ep275+. **ADD at v5.** |
| D2 | `--pose-carrier-source generated` | KEEP-list verbatim: "pose-carrier generated **(store-nothing)**"; sweep-A A26 INCLUDE=generated; #205 argv HAD it (grep-verified `20260703T120444Z/launch.sh`) | **ABSENT** → default `real_keyframe` (trainer:6531) | **NEW CATCH, the one score-relevant drift.** Mechanism: `derive_fresh_seeded_config` derives from `derive_sealed_205_config` (real_keyframe base; flag emitted only when ≠ real_keyframe, witness_autoconfig:713-718) — the store-nothing variant lives in a SEPARATE `derive_store_nothing_205_config`, so the fresh_seeded family (v1…v4) silently reverted the pose source. Consequence: training-time pose warps the REAL frame0 (a decode-side input the store-nothing byte-close does NOT have) ⟹ the run's pose telemetry is optimistic vs what `levelset_byte_close_and_eval.py` can reproduce, and the "store-nothing (~0 marginal bytes)" claim no longer describes the live config. Pose is already OPEN/UNMEASURED on the witness (sweep-B #5, #238 owed) so no measured number is falsified — but the byte-close pose path diverges from the trainer path. **Trajectory-affecting ⟹ NOT a rider; restore `generated` at the next FRESH arm (operator decision), and fix `witness_autoconfig` so fresh_seeded inherits the store-nothing source (or emits it explicitly).** |
| D3 | `--dm1-telemetry` | sweep-A A13 INCLUDE ("cheap observability of the rank verdict when film-stiefel on — pure read") | **ABSENT** → off | Minor, observability-only, bit-identical. film-stiefel IS on in v4, so the PR(M) rank verdict it was meant to watch is running unobserved. **ADD at v5.** |

### 1b. LEDGER-STALE — argv is right (later audited authority), ledger text not updated

| # | flag | ledger says | argv (audited authority) | action |
|---|---|---|---|---|
| L1 | `--hosc-beta-end` | KEEP-list: "hosc ANNEALED β1→4 (NEVER fixed)" | **5.134** — deliberate SEAL-review value M4 baked into `_FRESH_SEEDED_DELTAS` ("beta(ep726 muon-freeze)=4.00 exactly"); the ce-window adversarial review quantified β(400)=2.651, β(726)=4.000, β(1000)=5.134 and flagged the >4 tail as an **ep726+ watch item** (pre-existing sealed behavior, not a restart trigger) | Append an S5 row to the ledger; keep the ep726+ β>4 watch in the monitoring cadence (the measured divergence anchor is FIXED β=4 from init, not annealed-β on trained weights — but β>4 late-run is beyond the sanctioned example). |
| L2 | S1 wording "NO bank flags emitted" | — | argv emits `--max-bank-freq 64` (sweep-A A9 INCLUDE=64, #205 had it; S1 only supersedes bank-n-scales 6→4) | Wording fix only — S1 should read "no bank-n-scales flag emitted". |

### 1c. INVERSE CHECK — argv flags with NO master-ledger row (all audited elsewhere; 0 unaudited)

The v4 resume tail is the CE-window intervention + the v2/v3/v4 escalation, each with named audit
authority — these need LEDGER rows appended (op-routable 3), but none is an unaudited addition:
`--resume-from/--resume-allow-lever-drift/--resume-clear-spike-guard` + `--boundary-distance-weight
0.2` + `--tau-softplus-start-epoch 400` + `--lane-band-start-epoch 450` +
`--persistence-warmup-epochs 275` (ce_window_intervention_package Option B; independent adversarial
review CONFIRMED effective config + collision ordering 275→400→450→726→never) · `--lr 5e-4 --lr-end
5e-5` (FEED-05q OPERATOR GO#3, stepping-instability escape MEASURED at n600) ·
`--seed-anneal-shape cosine --seed-anneal-epochs 101` + `--witness-alone-island-loss`
(seed-compose crutch fix, `seed_compose_island_gradient_starvation_…_20260704.md`; compute audit:
wa-island NECESSARY, ~free).

---

## §2. DEFAULT-OFF INVENTORY (every trainer flag defaulting off/0.0/None, joined with evidence)

217 argparse rows; ~70 default to off/0/None. Grouped by classification. **Counts: CORRECTLY-OFF 28
· A/B-CANDIDATE 14 · DEPLOY-PENDING 6 · DEAD/inert 12** (dependent sub-flags counted with their
parent; base-trainer-only 87 flags EXCLUDE per sweep-A §B).

### 2a. CORRECTLY-OFF — measured NO-GO / inert / dominated / gated (do NOT activate; cites)

| flag(s) | why off is correct | cite |
|---|---|---|
| `--margin-saliency-uniward` (+beta) | texture proxy MEASURED INERT (Pearson −0.033 vs S_R, Jaccard 0.024 = chance) | `msal_uni_texture_proxy_inert_…_20260703.md`; sweep-A A15 |
| `--mx-compile` | fp-contraction flips uint8-STE argmax Δ~4.8e-3; fails closed | sweep-A A7 MEAS 2026-07-03 |
| `--render-aa supersample` / `--aa-supersample>1` / `--aa-ipe-footprint` / `--aa-self-orient-fine-*` | brute supersample HURTS −49% (SIGNAL-B); fine-mode ~86 GB @n600 refused | `aa_signal_a_…_20260702.md`; sweep-A A24 |
| `--film-per-layer` / `--film-concat-code` / `--film-rank-floor-weight` (+target) | MEASURED: do not raise rank / grad blow-up + proxy-gaming; dominated by film-stiefel (PR 1.19→4.57 at 0 bytes) | sweep-A A13 M1/M2 |
| `--lane-edge-weight` family (4) | defends only 19% of the flip band; dominated by class-agnostic saliency | sweep-A A12; FEED-eq flip-band split |
| `--hardness-oversample` family (5) | per-pair GT-margin spread only 1.31× — not worth the wall-clock | sweep-A A18 |
| `--seg-focal-gamma` | **γ\*=0 HOLD** — measurably NOT an island lever at this stage (non-monotone, peak γ=1, live slope already ≫ fire threshold) | `focal_boundary_calibration_20260705.md` |
| `--gpu-reorient` | parity probe (cos>0.999) never passed; keep bit-faithful numpy reorient | sweep-A A9 |
| `--curriculum-event-triggered` (+3 params) | SEAL CRITICAL C1/C2: plateau-eps 1e-3 fires CE→tau ~ep150 mid-descent (15% CE-floor loss) + could converge-fire the l7 DEFECT; gated on recalibration + boundary re-anchor BUILD | `fresh_run_config_adversarial_review_20260704` (via autoconfig docstring); symposium §C.ii |
| `--residual-mode` (+target-npz) | v2 rate path; separate future arm, orthogonal to nucleation | sweep-A A1 |
| `--freeze-decoder-fit-codes` | amortize path; fresh runs train jointly | sweep-A A0 |
| `--wire-w0/--wire-s0` | wire activation not chosen (hosc A/B winner 0.221 vs 0.265) | sweep-A A10 |
| `--anneal-epochs` | warm-start-window tool only; global-horizon default correct (review §1b verified β/temp schedules) | adversarial review §1 |
| `--lambda-pre-probe-iters` (+fd-eps) | diagnostic probe; λ_pre 38/η law FALSIFIED at the restored state — do not derive lr from it | eik-stab memo §7.3 |
| `--profile-timing` | probe-only observability (used by the n24 harness) | compute audit §2 |
| `--spike-guard-mode rollback` (+4 params) | BUILT (89c2add13, 27 tests) but arbitration verdict pending AND `lr_scale`+snapshot NOT persisted in the resume sidecar — "persist before any n600 run that relies on it" | `eikonal_stabilizer_build_20260705.md` §5/§7.5 → moves to DEPLOY-PENDING once both close |

### 2b. A/B-CANDIDATE — built, never measured at optimal form (ranked by expected value; each names its gate)

| rank | lever | gate it needs | EV rationale (cite) |
|---|---|---|---|
| 1 | `--n-dir-freqs 4 --freq-across 8` (#277 along-tangent) | fresh arm (shape-change; Nyquist 8·2³=64) | MEASURED #1 root cause of the binding lane-dash residual (3.2× along-tangent deficit, FEED-03t); config-only; master ledger §3 names it the #1 secondary. **Highest-EV un-run A/B in the program.** |
| 2 | chroma-OFF arm (#227/D9) | isolated run-3 arm (chroma is ON but never A/B'd) | verdict-BLOCKING: the whole d_seg ledger is provisional until A/B'd (master §3; sweep-C §7.5); GREEN probe 7.54% Lane→Road on removal, 93.4% of chroma-flips in margin<1 annulus |
| 3 | `--margin-saliency-reachability` (#268) | BUILD the `sR` key into gt_n600 (`tools/precompute_sR_reachability.py`, $0) + NOT with micro-batch>1 | exact through-R S_R replaces the measured-inert texture proxy; VERIFIED-ACTIVE wiring; MODEST/secondary (FEED-03n/03p) |
| 4 | `--ema-decay-finisher 0.9995` (+start-epoch) | run-3 A/B vs None (deployed-checkpoint authority unchanged until byte-closed comparison) | symposium READY-NOW: 0.997 = 333-step window = averages only ~1.6% of a 274-ep finisher (π-group violation; measured 78× early shadow lag); flag EXISTS, default None = bit-identical |
| 5 | `--seg-spike-reweight` + `--seg-spike-downweight 0.3` (#274) | net-S A/B (n600) | the standing flicker play after Lever-D NO-GO; flicker = 88.6% irreducible, 44% of CE-residual spikes = LANE (`witness_converged_to_flicker_floor…`) |
| 6 | `--seg-chroma-boundary-weight` | isolated A/B, start-epoch 300, NOT with micro-batch>1 | GREEN probe a3e9f0bd: 93.4% of chroma-flips inside the margin<1 annulus; luma-orthogonal boundary sharpener (sweep-A A16) |
| 7 | `--head etf` / `--logit-adjust-per-class` / `--margin-field-head-weight` (#218) | $0 probe of ETF rate-win + rare-class lift first | neural-collapse minority-norm fix, byte-free; BUILT (Laguerre head) but no byte-closed row (sweep-A A12) |
| 8 | OT head-offset b\* (#288, tool not flag) | $0 gate: apply b\* to EMA-best → re-render R → d_seg vs Menon, n600 (was RSS-gated; runnable in a quiet window) | the PRINCIPLED nucleation/asymmetry cure, byte-free, replaces the Menon heuristic (sweep-C §2.7, paranoia #2) |
| 9 | `--seg-subpix-boundary-weight` | isolated A/B, start 300 | GREEN directional probe a8afad40; densest sub-pixel signal, reuses shared margin fwd (sweep-A A16) |
| 10 | `--code-spectral-entropy-weight` β∈{0.01,0.05,0.1} | calibration sweep (β is a GUESS) | the other half of the byte-free rank cure with film-stiefel (sweep-A A13/§G) |
| 11 | `--lane-thin-weight` ~0.5 (+4 params, start 300) | calibration (magnitude unmeasured) | per-class area surrogate holding dropped-dash mass; partially covered by persistence 1.0 in v4 (sweep-A A14; master §5 area-constraint measured-deferred) |
| 12 | `--micro-batch-pairs` B∈{2,4} (#261/#293) | BUILD: extend `levelset_micro_batch_loss.LeverConfig` with wa-island/bd/focal/eik-stab routing, then loss/grad-equivalence A/B + n600 RSS re-measure | the ONLY order-of-magnitude speed lever (−50-75% of the 152 s/ep core); currently FAIL-CLOSES vs the v4 lever set (compute audit §4.1) |
| 13 | `--eikonal-junction-relax` | A/B paired with the raised eikonal | leaves Herring-angle creases un-over-penalized (θ* STRETCH-1, sweep-A A19); UNMEASURED |
| 14 | `--adam-beta2 0.9999` arm | isolated A/B (NOT the primary) | symposium §C.ii-7; the 0.9999999 value was a MIS-ANCHOR (superseded to 0.999); the small-n law itself is un-A/B'd |
| — | `--code-nuclear-weight` (+2) | low priority | dominated by the stiefel+spectral path (sweep-A A19); UNMEASURED |
| — | `--hosc-beta-anneal geometric` / Muon event-trigger / B5 Muon-boundary rewarmup | BUILDs (~10/80/30 LOC, symposium §C.ii) | not flags yet — listed for completeness, run-3+ |

### 2c. DEPLOY-PENDING — measured-positive/neutral, not yet in config

| lever | what it rides on | cite |
|---|---|---|
| `--cache-gt-skeleton` | **v5 rider** (drift D1; bit-identical + n64 A/B) | compute audit §4.2 |
| `--fused-r-kernel` | **v5 rider optional** (startup grad-bit-identity gate fails closed; ~−1.8% step) | compute audit §4.3 |
| `--dm1-telemetry` | **v5 rider** (drift D3; observability-only) | sweep-A A13 |
| `--pose-carrier-source generated` restore | **next FRESH arm, operator decision** (drift D2; trajectory-affecting) + `witness_autoconfig` fix | §1a D2 |
| eik-stab winner (`--eikonal-steik-weight` / `--eikonal-viscosity` / rollback guard) | folds into the relaunch delta per the arbitration verdict (GO-gated; table pending in the eik-stab memo at write time); rollback additionally needs sidecar persistence | eik-stab memo §6/§7 |
| `witness_control_monitor` attach (external tool, not argv) | run alongside any relaunch (decision-only, never launches); would have caught #205's erosion at ep325 and is skip-blind-complementary to SC1' | master ledger §2 (BUILT 3db114735) |

### 2d. DEAD / inert-by-design (no action; successor named)

`--l7-start-epoch 1001` + `--l7-mult/--l7-threshold` (l7 DEMOTED — measured L∞-sharpening defect;
successor: tau_softplus is the drop) · `--tau-hold-frac` (cosine_hold not chosen; geometric) ·
`--margin-target-end` (margin_hinge form unused) · `--lane-prior-phi1-bias-scale` (bias mode
unused; paint) · `--additive-margin`/`--logit-adjust-tau` alone (parents gated in 2b-7) ·
`--pose-carrier-s-r/-pitch/-fit-pairs/-residual-scale` (defaults measured/auto-fit) ·
`--loss-term-log-every` (default 0 ALREADY = per-epoch per-term summary, #304 item 4 — the
ce-window review's "per-term telemetry unbuilt" is stale; -1 disables, N>0 densifies; no action) ·
`--verdict-batch 0` (A/B-parity mode only) · `--mlx-cache-clear-accum 0` (pre-fix A/B mode) ·
base-trainer 87 flags (sweep-A §B; the two conceptual gaps — MD-Decoupling port, plateau-trigger —
are superseded by measured under-stepping (canon C15) and the control monitor respectively).

---

## §3. ORPHANED TOOL / TECHNIQUE SWEEP (outside the trainer argparse)

| item | on disk | missing step | joins | EV (one line) |
|---|---|---|---|---|
| #204/#207 sig-proc | probes + memo (`project_sig_proc_filter_chain…`) | **nothing to deploy**: deconv/pre-emphasis/matched-filter MEASURED NEGATIVE (R all-pass to 2px, FEED-03t) — honest correction to the "inflate-side winners, deploy pending" framing. The surviving winner is **L3 NTK/multiscale band-pass whitening — UNBUILT** (Ch5-M2) | Tier-3 build, exponent-bet gated ($0 N-term slope probe first, master §4) | the only EXPONENT lever + ~3-10× speed if the tail is spectrum-limited |
| #220 coverage-AA | aa flags + probes | none — CLOSED correct (`--render-aa none` shipped; supersample HURTS −49%; SIGNAL-A 0.00086 = floor proof only) | — | 0 (door measured shut) |
| #226 margin_conditional_residual | `margin_conditional_residual.py` BUILT | none — MEASURED NO-GO (b=0.876>0.65, net ΔS +0.202 WORSE, 5.2% band capture) | — | 0 (confidently-closed; successor = #274 down-weight, 2b-5) |
| #227 chroma/D9 (seg⊥pose frees seg-frame chroma) | wired (`--chroma` on) | the ISOLATED A/B (chroma-OFF arm) | run-3 arm #2 | verdict-blocking — whole d_seg ledger provisional until run |
| #213 analytic lane band (+#287 dash-comb) | BUILT (band default-gated; comb rate codec `_line_row_params`/`serialize_lane_band_rd`) | net-S A/B of band + of dash-comb-as-d_seg-lever | run-3 A/B | recall 0.5475 measured; rate codec COUNTS 3 comb params/line |
| #268 S_R reachability | BUILT+WIRED, VERIFIED-ACTIVE | `sR` key into gt_n600 ($0 build) + A/B | run-3 arm #3 | replaces measured-inert proxy (2b-3) |
| #242 flat-minima/MDL | λ sweep artifacts | none — Ballé weight-entropy NET-NEGATIVE at every λ; rate not binding | — | 0 (keep λ=0) |
| #288 OT head-offset | BUILT (`damped_newton_ot_offsets`, 6 tests) | the $0 gate (was #205-RSS-gated; now runnable in a quiet window) | pre-run-3 $0 probe | principled nucleation cure, byte-free (2b-8) |
| #217 leap-residual | DESIGN only | BUILD (post-Muon micro-stage); sequence AFTER the ep726 #270 read | run-3+ | highest-EV finishing lever (FEED-03o) |
| #293 micro-batch wa-twin | serial path + spec filed | BUILD (LeverConfig wa-island leg) + equivalence A/B + RSS re-measure | v6/run-3 | the order-of-magnitude speed lever (2b-12) |
| n24 probe queue P8/P8b/P9/P10/P12 + P_min re-runs | harness landed (`compute_audit_n24_20260705/run_probe.sh`) | ~75 s/probe on an IDLE GPU (single-workload discipline; contamination measured +77%) | next quiet window | splits the +47 s/ep tau group; prices the skeleton-cache share |
| BUILD §6.1 in-loop skip-escape + STRICT gate | not landed | build before run-3 (review op-routable 3) | run-3 prereq | kills the absorbing-deadlock class (3× incident) |

---

## §4. THE v5-RIDER SHORTLIST (safe under the standing GO) vs run-3/A-B items

**RIDERS (recommend; measured-neutral or observability-only — do not stretch the GO):**
1. `+ --cache-gt-skeleton` — bit-identical by construction + n64 A/B (drift-restore D1).
2. `+ --fused-r-kernel` — startup parity gate fails closed; ~−3 s/ep (optional).
3. `+ --dm1-telemetry` — pure-read rank observability for the running film-stiefel (drift-restore D3).

**OPERATOR-DECIDES (not neutral; recommend at the next FRESH arm, not as silent riders):**
- `--pose-carrier-source generated` restore (D2) + the `witness_autoconfig` inheritance fix — trajectory-affecting; aligns trainer path with the store-nothing byte-close.
- eik-stab arbitration winner delta (per its own GO-gated §6; compose into the same relaunch argv).
- `--ema-decay-finisher 0.9995` — engages only at ep726 but changes the DEPLOYED EMA; the A/B-clean home is run-3 (symposium: "A/B vs None").

**EXPLICITLY NOT riders:** everything in §2b (each needs its named gate; stacking un-A/B'd levers
re-creates the attribution debt the isolated-arm discipline exists to prevent) and everything in
§2a (measured NO-GO/inert/gated — activating any of them contradicts a measured verdict).

## Observability surface
Drift table reproducible: `launch.sh` token-diff vs ledger §1/§Supersessions + `grep add_argument`
defaults; D2 verified from 3 primary sources (#205 launch.sh grep, witness_autoconfig:518/713-718,
v4 launch.sh absence). Classification joins cite sweep-A row ids + memo filenames; counterfactual =
re-run any probe cited. Diff-able across future argvs (repeat §1 against any new launch.sh).

## 6-hook wire-in (Catalog #125)
sensitivity-map: N/A (audit, no score axes) · Pareto: ACTIVE (feeds the relaunch-delta composition,
§4) · bit-allocator: N/A · cathedral autopilot: N/A · continual-learning: this memo + DAG FEED-05v ·
probe-disambiguator: the named gates per §2b row ARE the disambiguators.

**NO-FAKE:** no lever recommended against a measured verdict; UNMEASURED written where true (β
spectral weight, junction-relax, mod-19 d_seg-neutrality at n600). Pointer 0.19110 UNMOVED.
