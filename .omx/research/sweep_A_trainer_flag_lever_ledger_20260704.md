# MAX-SIGNAL PARANOIA SWEEP — Surface A: TRAINER FLAGS / BUILT LEVERS ledger (2026-07-04)

**Surface A of 3 (operator: "insanely detail oriented and meticulous… all levers that should be judged
for inclusion or not and may still need building or calibration or measurement"). This is the EXHAUSTIVE
per-lever verdict table to gate the FRESH SEEDED RUN config (the next #205 successor).** $0 research; NO
heavy/paid/GPU; **#205 (`levelset_n600_witness_20260703T120444Z`, pid live) SACRED READ-ONLY**. Pointer
contest-CPU **0.19110 UNMOVED** — everything here is **MEANS**; no lever below is a score until a
byte-closed `upstream/evaluate.py` n600 row beats 0.19110.

**NO-FAKE:** every "MEASURED" cell cites a real artifact/probe; where I have no measurement I write
UNMEASURED (never a fabricated number). Flags grepped from the live trainer — never invented.

## Scope + method
- **Live launch path = `experiments/train_levelset_witness_realized_through_R_mlx.py`** (the ~208KB entry
  point per CLAUDE.md §"Capstone θ* witness trainer canonical entry point"). **189 `add_argument` flags**
  (grep-confirmed count). The base `train_witness_realized_through_R_mlx.py` (87 flags) is imported for
  shared primitives ONLY and has its OWN standalone argparse whose flags are NOT on the live path —
  auditing the base alone MISSES the live launch levers (the 2026-07-01 wiring-audit near-miss). Base-only
  flags are listed in §B (all EXCLUDE-from-live).
- **Verdict tokens:** **INCLUDE**(=value) · **EXCLUDE**(why) · **MEASURE-FIRST**($0 gate owed) ·
  **CALIBRATE**(value uncertain, A/B) · **BUILD**(flag/underlying incomplete).
- **Grounding:** `lane_nucleation_failure_seed_above_critical_nucleus_20260704` (the nucleation crux) ·
  `scaling_law_facet{1,2,4,5}_*_20260704.md` · `deepmath_converged_next_run_config_20260704.md` · #205
  `launch.sh` (the current live config).
- Column key: **dfl-BI?** = is default byte-identical to pre-flag path · **B+T** = built+tested in-tree ·
  **MEAS** = measured artifact or UNMEAS · **#205** = value in the live run.

---

## §A. THE EXHAUSTIVE LEVELSET-TRAINER TABLE (all 189 flags)

### A0 — Run / IO / checkpointing (10)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --out-dir | run dir | (req) | n/a | Y | n/a | set | INCLUDE (required) |
| --num-pairs | P pairs | 24 | n/a | Y | n/a | 600 | INCLUDE=600 (n600 = only evidence) |
| --epochs | run length | 1500 | n/a | Y | n/a | 1000 | INCLUDE=1000 (see UNSURE — new τ dynamics may want longer tau stage) |
| --anneal-epochs | decouple anneal denom from epochs | None | Y | Y | n/a | unset | EXCLUDE (only for warm-start windows; fresh run = None) |
| --eval-every | verdict cadence | 25 | Y | Y | n/a | 25 | INCLUDE=25 |
| --ckpt-every | rolling ckpt | 0 | Y | Y | n/a | 25 | INCLUDE=25 (crash-window bound; launch non-negotiable) |
| --stage-checkpoints | per-stage preserved ckpt | True | Y | Y | n/a | on | INCLUDE (MANDATORY per resumable/per-stage non-negotiable) |
| --resume-from | resume a run | None | Y | Y | n/a | unset | EXCLUDE (fresh run — the seed MUST be at init, can't retrofit) |
| --resume-allow-lever-drift | permit lever drift on resume | False | Y | Y | n/a | off | EXCLUDE (fail-closed default correct) |
| --freeze-decoder-fit-codes | amortize: freeze decoder, fit codes | None | Y | Y | FEED-eo | unset | EXCLUDE (joint train for a fresh seeded run) |

### A1 — Residual-only v2 hybrid (2)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --residual-mode | INR-on-residual, bulk generated at decode | False | Y | Y | UNMEAS(no byte-closed row) | off | EXCLUDE (v2 rate path; orthogonal to the nucleation fix; a separate future arm) |
| --residual-target-npz | residual bundle input | None | Y | Y | n/a | unset | EXCLUDE (needs --residual-mode) |

### A2 — Render / architecture (5)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --render-h | render height | 384 | n/a | Y | config-review#1: 192 pre-caps 0.00085 | 384 | INCLUDE=384 (R-survival floor; 192 blocks sub-0.15) |
| --render-w | render width | 512 | n/a | Y | (as above) | 512 | INCLUDE=512 |
| --hidden-dim | decoder width | 96 | n/a | Y | config-review#2 RD-optimum | 96 | INCLUDE=96 (mod32×hidden96→~122KB RD-optimum) — but re-derive with mod-19 (UNSURE) |
| --n-hidden | decoder depth | 4 | n/a | Y | UNMEAS | 4 | INCLUDE=4 |
| --mod-dim | per-pair FiLM code dim | 32 | n/a | Y | **DERIVED-from-MEAS: Whitney 2m+1≈17-19, m≈8** (facet2 §1.5) | 32 | **CALIBRATE→19** (rate win 32→19; but shape-change→fresh arm; capacity-cliff risk if m>9 — see paranoia #4) |

### A3 — Softmax-temp / τ anneal (4)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --softmax-temp-start | anneal start (soft) | 1.0 | n/a | Y | config-review#4 | 1.0 | INCLUDE=1.0 |
| --softmax-temp-end | anneal end (sharp) | 0.05 | n/a | Y | facet4 §1.3: 0.05 = 40× sub-grid (wasted compute, unit-caveat) | 0.05 | **MEASURE-FIRST** {0.05,0.1,0.25} (direction=raise to resolution floor DERIVED; magnitude unit-convention UNVERIFIED — paranoia #3) |
| --tau-anneal-shape | anneal curve | cosine | Y(cosine=BI) | Y | facet4 §1.1-1.2: geometric = τ-Fisher-Rao geodesic under broadband margin | cosine | **INCLUDE=geometric** (Γ-optimal; equal epochs/octave; damps late-τ volatility) |
| --tau-hold-frac | cosine_hold floor frac | 1.0 | Y | Y | n/a | 1.0 | EXCLUDE (only for cosine_hold; geometric chosen instead) |

### A4 — Core optimizer / EMA / LR (9)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --lr | base LR | 1e-3 | n/a | Y | UNMEAS(stable) | 1e-3 | INCLUDE=1e-3 |
| --lr-end | cosine floor | 1e-4 | n/a | Y | UNMEAS | 1e-4 | INCLUDE=1e-4 |
| --weight-decay | AdamW WD | 1e-4 | n/a | Y | muon-deep-dive: keep WD | 1e-4 | INCLUDE=1e-4 |
| --adam-beta2 | 2nd-moment decay | 0.999 | Y | Y | #222 arXiv2603.02092 small-n opt ~0.9999999 | 0.999 | **CALIBRATE→0.9999999** (n~75 accum steps; 0.999 under-smoothed by ~4 orders — UNSURE, #205 kept 0.999) |
| --ema-decay | EMA shadow decay | 0.997 | n/a | Y | Quantizr 0.997 anchor | 0.997 | INCLUDE=0.997 (EMA non-negotiable) |
| --ema-decay-finisher | wider SWA EMA in finisher | None | Y | Y | UNMEAS | unset | CALIBRATE (θ* MUST-3 SWA; wider EMA over Muon oscillation — A/B, low priority) |
| --ema-decay-finisher-start-epoch | when it engages | None | Y | Y | n/a | unset | EXCLUDE (needs --ema-decay-finisher) |
| --lr-schedule | warmup+cosine | True | n/a | Y | n/a | on | INCLUDE (on) |
| --warmup-epochs | LR warmup len | 1 | n/a | Y | n/a | (default) | INCLUDE=1 |

### A5 — Loss weights / score-domain (5)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --w-seg | seg loss weight | 100.0 | n/a | Y | mirrors S=100·d_seg | 100 | INCLUDE=100 |
| --w-pose | pose loss weight | 0.0 | n/a | Y | fix-g: pose=stored sidecar | 1.0 | INCLUDE=1.0 (required by --pose-carrier; witness's binding job is still d_seg) |
| --score-domain-loss | w_seg·seg + w_pose·√(10·pose) | True | n/a | Y | HNeRV L6 | on | INCLUDE (on) |
| --pose-eps | √-stability eps | 1e-2 | n/a | Y | FEED-bd/be caps grad coeff | 1e-2 | INCLUDE=1e-2 |
| --hinge-weight | margin-hinge weight | 4.0 | n/a | Y | UNMEAS | (default) | INCLUDE=4.0 |

### A6 — Batching / throughput / verdict / device (8)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --accum-pairs | grad accum chunk | 8 | n/a | Y | FEED-bd variance-reduction | 8 | INCLUDE=8 |
| --micro-batch-pairs | batched scorer fwd (speed) | 1 | Y(1=BI) | Y | ~2-4× (FEED 2026-07-03c) NOT bit-identical | 1(unset) | **CALIBRATE** (S-neutral verdict; but trajectory-affecting + INCOMPATIBLE with --seed-islands (fail-closed at build) → the fresh seeded run CANNOT use it → EXCLUDE for THIS run) |
| --cache-gt-skeleton | cache constant GT skeleton (speed) | off | Y(BI) | Y | bit-identical n64 A/B | unset | INCLUDE (bit-identical speed; no-op unless persistence>0, which fresh run has) |
| --mlx-cache-clear-accum | Metal pool hygiene (OOM) | 1 | Y(score-neutral) | Y | #205 OOM fix, bit-identical n64 A/B | 1(default) | INCLUDE=1 (OOM insurance) |
| --grad-clip | grad-norm clip | 1.0 | n/a | Y | FEED-bd primary divergence fix | 1.0 | INCLUDE=1.0 |
| --spike-factor | spike-skip guard | 5.0 | n/a | Y | FEED-bd | 5.0 | INCLUDE=5.0 |
| --verdict-batch | CPU-verdict chunk (OOM) | 32 | Y(BN running stats) | Y | #205 REAL OOM fix, score-neutral | 32 | INCLUDE=32 (the n600 OOM law: chunk full-P scorer forwards) |
| --verdict-pairs | verdict subset (0=all) | 24 | n/a | Y | n/a | 0(all) | INCLUDE=0 (n600 = all pairs; NO subset per allergic-to-toys) |
| --async-verdict | background verdict thread | False | Y(BI train) | Y | ~4.7% wall reclaim (FEED-em) | on | INCLUDE (on; bit-identical training, verdict never read back) |
| --mlx-device | gpu/cpu | gpu | n/a | Y | MLX-GPU not bit-identical cross-proc (verdict CPU-locked) | gpu | INCLUDE=gpu (training gradient; CPU is verdict authority) |

### A7 — Compute-facet #252 kernels (3)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --fused-r-kernel | fused Metal R (speed) | False | Y(startup parity gate) | Y | bit-identical fwd, ~1 ULP VJP | off | CALIBRATE (S-neutral speed; parity-gated fail-closed; ADOPT after per-chip gate passes — low-risk speed, but OFF in #205 → keep OFF for clean A/B unless speed pressure) |
| --mx-compile | mx.compile'd R | False | Y(gated) | Y | **MEAS 2026-07-03: reintroduces fp-contraction, flips uint8-STE argmax (Δ~4.8e-3)→FAILS CLOSED** | off | **EXCLUDE** (measured non-bit-identical; prefer --fused-r-kernel) |
| --profile-timing | per-epoch phase split | False | Y(BI) | Y | advisory | off | EXCLUDE (advisory; optional observability) |

### A8 — Misc (4)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --seed | RNG seed | 0 | n/a | Y | n/a | 0 | INCLUDE=0 (deterministic-repro non-negotiable) |
| --gt-cache | shared GT npz | None | Y | Y | n/a | gt_n600.npz | INCLUDE (gt_n600 cache — skips 480s/arm precompute) |
| --chroma | 3 independent RGB planes (d_seg lever) | True | n/a | Y | operator "Chroma too"; d_seg actuator | on | INCLUDE (on; real argmax-flip actuator) |
| --palette-anchor | init palette to per-class GT mean | True | n/a | Y | DIAGNOSED FIX: breaks ~0.51 luma-ramp plateau | on | INCLUDE (on) |

### A9 — Curvelet front-end / bank (12)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --bank-n-scales | radial octaves | 4 | n/a | Y | facet2 §2.3: Nyquist budget = 6 (only 2 of 6 octaves used at 4) | **4(default,unset)** | **INCLUDE=6** (use the Nyquist radial budget; shape-change→fresh arm; §3.2 slope measurement owed to confirm finer octaves PAY) |
| --bank-n-orient0 | base orientations | 6 | n/a | Y | facet2: keep 6 | (default) | INCLUDE=6 |
| --bank-f0 | base freq | 2.0 | n/a | Y | n/a | (default) | INCLUDE=2.0 |
| --bank-base | freq base | 2.0 | n/a | Y | n/a | (default) | INCLUDE=2.0 |
| --bank-n-iso | isotropic atoms | 4 | n/a | Y | n/a | (default) | INCLUDE=4 |
| --max-bank-freq | Nyquist cap (cyc/unit) | None | Y(no-op@dflt) | Y | facet2: stem Nyquist=64 | 64 | INCLUDE=64 (cap at stem Nyquist; drops aliasing atoms, shrinks in_proj) |
| --self-orient | self-orientation directional feats | False | Y | Y | −48% on direct generator (byte-closeable) | on | INCLUDE (on; wavefront orientation; the decisive basis lever) |
| --n-dir-freqs | self-orient freq octaves | 6 | n/a | Y | facet2: default 6→1024 cyc/unit = 16× over Nyquist (aliases); use 2 | 2 | INCLUDE=2 (Nyquist-safe; #205 already correct) |
| --reorient-every | reorient cadence | 50 | n/a | Y | n/a | 50 | INCLUDE=50 |
| --gpu-reorient | reorient argmax on GPU | False | Y(parity-gated) | Y | UNMEAS(needs probe cos>0.999) | off | EXCLUDE (parity probe not passed; keep bit-faithful numpy reorient) |
| --freq-across | high freq across edge | 32.0 | n/a | Y | facet2 Nyquist-safe with n_dir=2 | 32 | INCLUDE=32 |
| --freq-along | low freq along edge | 4.0 | n/a | Y | lane-dash residual = along-tangent freq deficit 3.2× (2026-07-03) | 4 | INCLUDE=4 (BUILD note: M1 along-tangent #277 wants n_dir 2→4 @ freq-across 8 — a Tier-3 build arm, not this run) |

### A10 — Activation (8)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --activation | wire/hosc/relu | hosc | n/a | Y | config-review#3: HOSC only descent evidence (A/B 0.221 vs 0.265 wire) | hosc | INCLUDE=hosc |
| --wire-w0 | wire freq | 20.0 | n/a | Y | n/a | (default) | EXCLUDE (wire not chosen) |
| --wire-s0 | wire gaussian scale | 10.0 | n/a | Y | n/a | (default) | EXCLUDE (wire not chosen) |
| --hosc-beta | hosc sharpness | 4.0 | n/a | Y | FEED 2026-06-25: fixed β=4 DIVERGES | 1.0 | INCLUDE=1.0 (anneal start; NEVER fixed 4) |
| --hosc-beta-end | hosc anneal target | None | Y(=const) | Y | FEED-fb: β→∞ step-native | 4.0 | INCLUDE=4.0 (anneal 1→4; step-sharpens as SDF pins) |
| --hosc-beta-anneal | anneal shape | linear | Y | Y | n/a | linear | INCLUDE=linear |
| --hosc-omega | hosc base freq | 1.0 | n/a | Y | n/a | 1.0 | INCLUDE=1.0 |
| --siren-init | SIREN init for periodic | True | n/a | Y | from-scratch trainability fix (Sitzmann 2020) | on | INCLUDE (on) |

### A11 — Seg loss / PR95 curriculum (8)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --seg-loss | ce/tau_softplus/l7/margin_hinge | ce | n/a | Y | PR95 curriculum | ce(via curriculum) | INCLUDE=ce (curriculum drives the stage form) |
| --curriculum | run PR95 ce→tau→l7 curriculum | False | Y | Y | PR95 8-stage | on | INCLUDE (on) |
| --tau-softplus-start-epoch | ce→tau boundary | 300 | n/a | Y | PR95 ~CE300 | 300 | INCLUDE=300 (but this is the NUCLEATION-CRITICAL boundary — tau = MCF erosion of the un-seeded lane; the seed fix + geometric-τ must be paired) |
| --l7-start-epoch | tau→l7 boundary | 800 | n/a | Y | PR95 | 1000 | UNSURE=1000 (==epochs → l7 NEVER runs in #205; is that intended? see UNSURE list) |
| --tau-softplus-tau | tau_softplus tau | 0.3 | n/a | Y | PR95 0.3 | 0.3 | INCLUDE=0.3 |
| --l7-mult | l7 hard-pixel weight | 4.0 | n/a | Y | PR95 | (default) | INCLUDE=4.0 (inert if l7 never runs) |
| --l7-threshold | l7 small-margin threshold | 1.0 | n/a | Y | FEED-ca: measured median 0.42 (levelset default 1.0) | (default) | CALIBRATE→0.42 if l7 is actually run (UNSURE — inert at l7@1000) |
| --margin-target-end | margin_hinge target | 0.5 | n/a | Y | lensA 0.5 best | (default) | INCLUDE=0.5 (inert unless margin_hinge) |

### A12 — LEVER-3 lane-edge (3) + #218 margin-field head (6)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --lane-edge-weight | class-1 realized margin hinge | 0.0 | Y(0=off) | Y | LEVER-3 defends only 19% of flip band | off | EXCLUDE (dominated by LEVER-4 class-agnostic saliency; nobody runs both) |
| --lane-edge-class | class idx to up-weight | 1 | n/a | Y | comma10k Lane=1 CONFIRMED | (default) | EXCLUDE (needs lane-edge-weight) |
| --lane-margin-target | lane hinge target | 0.5 | n/a | Y | n/a | (default) | EXCLUDE (needs lane-edge-weight) |
| --head | softmax/etf/additive-margin | softmax | Y(BI) | Y(#218,Laguerre BUILT) | UNMEAS(no byte-closed row) | softmax | MEASURE-FIRST (ETF = neural-collapse minority-norm fix, byte-free rate-win; probe owed before INCLUDE) |
| --additive-margin | AM-softmax margin | 0.0 | Y | Y | UNMEAS | (default) | EXCLUDE (needs --head additive-margin) |
| --logit-adjust-per-class | Menon rare-class margin raise | False | Y | Y(BUILT) | UNMEAS | off | MEASURE-FIRST (byte-free rare-class/Lane lift; needs margin-field-head-weight>0) |
| --logit-adjust-tau | Menon tau scale | 1.0 | n/a | Y | n/a | (default) | EXCLUDE (needs logit-adjust-per-class) |
| --margin-field-head-weight | realized per-class margin hinge weight | 0.0 | Y(0=off) | Y | UNMEAS | off | MEASURE-FIRST (enables #218 facets 1b/3) |
| --lane-edge-start-epoch | LEVER-3 engage gate | 0 | n/a | Y | optimal-form: gate to tau stage | (default) | EXCLUDE (needs lane-edge-weight) |

### A13 — LEVER-A FiLM-rank-fix + DM1 (7)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --film-per-layer | per-layer residual FiLM (+25k params) | off | Y | Y | **MEAS(M2): does NOT raise rank** (PR(M)≤rank(codes)≤mod_dim) | off | EXCLUDE (capacity not rank; +0.01 rate for nothing; dominated by --film-stiefel) |
| --film-concat-code | additive code injection (+12k) | off | Y | Y | MEAS(M2): same rank ceiling | off | EXCLUDE (dominated by --film-stiefel) |
| --film-rank-floor-weight | soft PR floor penalty | 0.0 | Y | Y | MEAS(M1): grad blows up at small codes, proxy-games | off | EXCLUDE (dominated by --film-stiefel; kept ablation-only) |
| --film-rank-floor-target | PR floor target | 4.0 | n/a | Y | n/a | (default) | EXCLUDE (needs rank-floor-weight) |
| --film-stiefel | Stiefel orthonormal-column projection (byte-free) | off | Y | Y | **MEAS: PR(M) 1.19→4.57 at 0 added bytes** (byte-free rank fix) | **off** | **INCLUDE** (byte-free FiLM rank-collapse cure; facet2 recommends; NOT in #205 → key addition) |
| --code-spectral-entropy-weight β | keep code directions live (byte-free) | 0.0 | Y | Y | UNMEAS(β) | off | **CALIBRATE** (the other half of the byte-free cure; β unmeasured — A/B small β e.g. 0.01-0.1) |
| --dm1-telemetry | force PR telemetry row | off | Y(BI observability) | Y | n/a | off | INCLUDE (cheap observability of the rank verdict when film-stiefel on — pure read) |

### A14 — LEVER-B thin-lane (5)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --lane-thin-weight | thin-lane realized margin hinge | 0.0 | Y(0=off) | Y | 52.7% GT-lane CCs wholesale-missed; <5px 93% missed | off | **CALIBRATE→small** (facet4(d): per-class AREA surrogate that HOLDS dropped-dash mass — the auction-MBO cure; weight unmeasured; PAIR with the seed) |
| --lane-thin-class | lane class idx | 1 | n/a | Y | comma10k Lane=1 | (default) | INCLUDE=1 (if lane-thin on) |
| --lane-thin-radius | density window half-width | 4 | n/a | Y | UNMEAS | (default) | INCLUDE=4 (if on) |
| --lane-thin-target | hinge target | 0.5 | n/a | Y | n/a | (default) | INCLUDE=0.5 (if on) |
| --lane-thin-start-epoch | engage gate | 0 | n/a | Y | facet4: gate to tau@300 | (default) | **INCLUDE=300** (if lane-thin on; gate to tau/MCF stage to avoid margin-from-scratch starvation) |

### A15 — LEVER-4 margin-saliency (8)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --margin-saliency-weight | all-class fragility-weighted hinge | 0.0 | Y(0=off) | Y | FEED-eq: flip band Road47/Lane19/Undriv14/Mov9/MyCar11% | off | CALIBRATE (generalizes LEVER-3 to 100% of flip band; class-agnostic; A/B — but #205 leaned on amplify+persistence instead; secondary to the seed) |
| --margin-saliency-tau | saliency softness | 0.5 | n/a | Y | ~p1 GT-margin | (default) | INCLUDE=0.5 (if on) |
| --margin-saliency-target | hinge target | 0.5 | n/a | Y | n/a | (default) | INCLUDE=0.5 (if on) |
| --margin-saliency-start-epoch | engage gate | 0 | n/a | Y | optimal-form gate to tau/l7 | (default) | INCLUDE=300 (if on) |
| --margin-saliency-uniward | down-weight textured regions | off | Y | Y | **MEAS: texture proxy INERT (Pearson −0.033 vs S_R, Jaccard 0.024=chance)** (2026-07-03) | off | EXCLUDE (measured inert; mildly misdirects) |
| --margin-saliency-uniward-beta | uniward strength | 4.0 | n/a | Y | (as above) | (default) | EXCLUDE (needs uniward) |
| --margin-saliency-reachability | multiply by cached through-R S_R | off | Y | Y | S_R lives on fragile band (the real signal); needs 'sR' key in gt-cache | off | BUILD (needs `tools/precompute_sR_reachability.py` sR key + NOT with micro-batch>1; the exact-S_R replacement for the inert texture proxy — build then MEASURE-FIRST) |

### A16 — LEVER-4b sub-pixel (3) / LEVER-4c chroma-boundary (3)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --seg-subpix-boundary-weight | sub-pixel placement (t_wit−t_GT)² | 0.0 | Y(0=off) | Y | probe a8afad40 GREEN 2026-07-03 (directional) | off | CALIBRATE (denser sub-pixel signal, reuses shared margin fwd; NOT with micro-batch>1; A/B — secondary) |
| --seg-subpix-boundary-v-band | genuine-V flip band | 1.0 | n/a | Y | gt_n96: band1.0→2196 px/frame, t~Uniform | (default) | INCLUDE=1.0 (if on) |
| --seg-subpix-boundary-start-epoch | engage gate | 0 | n/a | Y | gate to tau/l7 | (default) | INCLUDE=300 (if on) |
| --seg-chroma-boundary-weight | chroma-match at margin annulus | 0.0 | Y(0=off) | Y | probe a3e9f0bd GREEN: 93.4% chroma-flips in margin<1 annulus (2026-07-03) | off | CALIBRATE (chroma = proven independent d_seg boundary sharpener; luma-invariant, orthogonal to luma levers; NOT with micro-batch>1; A/B — promising secondary) |
| --seg-chroma-boundary-margin-band | annulus band | 1.0 | n/a | Y | band1.0=93.4% of chroma-flips | (default) | INCLUDE=1.0 (if on) |
| --seg-chroma-boundary-start-epoch | engage gate | 0 | n/a | Y | gate to tau/l7 | (default) | INCLUDE=300 (if on) |

### A17 — Spike-aware seg reweight (3)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --seg-spike-reweight | down-weight flicker, up-weight coherent | off | Y(map=1@dflt) | Y | source-split MEAS n600 2026-07-03: flicker 88.6% irreducible; store-flicker net-NEG (rate+0.56>d_seg0.52) | off | CALIBRATE (MODEST 2nd-order reallocation; benefit ~popout floor; NOT with micro-batch>1; A/B arm not a claim — the flicker-floor path per `witness_converged_to_flicker_floor`) |
| --seg-spike-downweight | flicker px weight | 1.0 | Y(1=BI) | Y | (as above) | (default) | INCLUDE<1.0 (if reweight on; e.g. 0.3) |
| --seg-coherent-upweight | coherent px weight | 1.0 | Y(1=BI) | Y | (as above) | (default) | INCLUDE>1.0 (if reweight on) |

### A18 — LEVER-5 hardness waterfill (5)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --hardness-oversample | extra per-epoch steps (frac of P) | 0.0 | Y(0=off) | Y | FEED-eq: per-pair GT-margin spread only 1.31× (modest) | off | EXCLUDE (margin-source spread modest; 'realized' source sharper but adds wall-clock; not for the nucleation-focused run) |
| --hardness-weighted | draw extras ~hardness^power | off | Y | Y | (as above) | off | EXCLUDE (needs oversample) |
| --hardness-source | margin/realized | margin | n/a | Y | realized = sharper for code-fit | (default) | EXCLUDE (needs oversample) |
| --hardness-power | sampling exponent | 1.0 | n/a | Y | n/a | (default) | EXCLUDE (needs oversample) |
| --hardness-band | flip-prone px threshold | 0.5 | n/a | Y | n/a | (default) | EXCLUDE (needs oversample) |

### A19 — Level-set regularizers (7)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --eikonal-weight | \|∇φ\|→1 interface-width control | 0.01 | n/a | Y | **facet4 §2.2(b): raise 0.01→0.05 = interface-width control (#286); keeps thin lane in 94%-survival regime** | 0.01 | **INCLUDE=0.05** (enables the τ-floor; keeps the seeded lane sharp under MCF — the nucleation fix) |
| --length-weight | Chan-Vese boundary length | 0.001 | n/a | Y | **facet4 §2.2(c): length IS the MCF-erosion driver of the thin lane — KEEP small, do NOT raise** | 0.001 | **INCLUDE=0.001** (KEEP; the exact inversion of the naive "add smoothing" instinct — paranoia #5) |
| --code-nuclear-weight | smoothed nuclear-norm low-rank code | 0.0 | Y(0=off) | Y | UNMEAS | off | CALIBRATE (θ* MUST-2 rate lever; low-rank codes; A/B — but --film-stiefel + spectral-entropy is the preferred rank path; low priority) |
| --code-nuclear-eps | NS smoothing floor | 1e-3 | n/a | Y | ~0.3% bias | (default) | INCLUDE=1e-3 (if nuclear on) |
| --code-nuclear-ns-iters | NS iters | 25 | n/a | Y | converged ~25 for mod≤48 | (default) | INCLUDE=25 (if nuclear on) |
| --eikonal-junction-relax | down-weight eikonal near triple junctions | 0.0 | Y(0=off) | Y | UNMEAS | off | CALIBRATE (θ* STRETCH-1; leaves Herring-angle creases un-over-penalized; A/B with raised eikonal — pairs well with 0.05 eikonal) |
| --eikonal-junction-tau | junction relax scale | 0.5 | n/a | Y | n/a | (default) | INCLUDE=0.5 (if relax on) |

### A20 — Structured-init (7)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --structured-init | pretrain φ to static-core partition | False | Y | Y | **MEAS FEED-ef: NO epoch-0 realized win (texture-gated); trajectory A/B only** | on | INCLUDE (on — REQUIRED by --seed-islands + --lane-prior-phi1; BUT see paranoia #2: #205's structured-init seeded lane at part_frac 0.0) |
| --structured-init-include-lane | include static lane band | True | n/a | Y | (see nucleation memo) | on | INCLUDE (on) |
| --structured-init-thresh | majority-vote threshold | 0.5 | n/a | Y | n/a | (default) | INCLUDE=0.5 |
| --structured-init-steps | pretrain steps | 600 | n/a | Y | UNMEAS | (default) | INCLUDE=600 |
| --structured-init-lr | pretrain LR | 5e-3 | n/a | Y | 5e-3 converges, 8e-3 stalls | (default) | INCLUDE=5e-3 |
| --structured-init-subsample | px/step | 8192 | n/a | Y | n/a | (default) | INCLUDE=8192 |
| --structured-init-sdf-clip | SDF target clip | 20.0 | n/a | Y | argmax-preserving | (default) | INCLUDE=20.0 |

### A21 — Muon finisher (8)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --muon-start-epoch | AdamW→Muon switch | None | Y(None=AdamW) | Y | **muon-deep-dive: KEEP Muon (−32% d_seg vs AdamW); "Muon is THE drop"** | 726 | INCLUDE=726 (BUT cannot nucleate a zero-mass class → pair with seed; re-derive if epochs/tau change) |
| --muon-lr | Muon-group LR | None(→0.1·lr) | Y | Y | UNMEAS(0.002 not proven optimal) | 0.002 | CALIBRATE (muon-deep-dive: TUNE finishing schedule; ~1e-3-5e-3 typical) |
| --muon-adamw-lr | AdamW-fallback LR | None(→0.1·lr) | Y | Y | UNMEAS | unset | INCLUDE=None (default 0.1·lr) |
| --muon-momentum | Muon momentum | 0.95 | n/a | Y | Keller Jordan default | 0.95 | INCLUDE=0.95 |
| --muon-weight-decay | Muon WD | None(→wd) | Y | Y | muon-deep-dive: keep WD | unset | INCLUDE=None |
| --muon-ns-steps | Newton-Schulz iters | 5 | n/a | Y | Keller Jordan default 5 | 5 | INCLUDE=5 |
| --muon-lr-final-frac | cosine-decay Muon LR | 1.0 | Y(1=flat=BI) | Y | GAP1: flat LR can't self-reduce near min (river-valley 2606.21514) | 1.0 | **INCLUDE=0.1** (#270 A/B; lets the finisher settle — the transition-easing principle) |
| --muon-warm-start-momentum | warm-start Muon v from AdamW m | False | Y(cold=BI) | Y | GAP2: kills cold-start +0.000357 spike (measured, muon-restart memo) | **off** | **INCLUDE** (#270; removes the cold-start boundary thrash — NOT in #205, key addition) |

### A22 — BUILD-1 stage-transition treatment (4)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --stage-transition-rewarmup-epochs | LR re-warmup after AdamW→AdamW boundary | 0 | Y(0=off) | Y | FEED-fw; Ch.6 easing | 8 | **INCLUDE=20** (Ch.6: attacks the ep300 bump d_seg 0.0056→0.020 3.4×; facet4/deepmath §1) |
| --stage-transition-rewarmup-floor | LR fraction at boundary | 0.1 | n/a | Y | n/a | 0.1 | INCLUDE=0.1 |
| --stage-transition-rewarmup-shape | linear/cosine | linear | Y | Y | Ch.6 recommends cosine | linear | **INCLUDE=cosine** (Ch.6 reduced-step corrector) |
| --stage-transition-reset-moments | zero AdamW m/v at boundary | off | Y(off=BI) | Y | FEED-ft#3: stale momentum through loss-landscape change = tau-jump root cause | on | INCLUDE (on; #205 already sets it — stale-momentum fix) |

### A23 — BUILD-2 lane-prior φ1 (5)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --lane-prior-phi1 | init φ1 to openpilot deg-3 centerline SDF | False | Y | Y | FEED-fs: centerline IS Road↔Lane separatrix (residual 1.9e-5) | on | INCLUDE (on; the separatrix-SDF lane seed = facet3 SEED; rule-118 FREE generic structure) |
| --lane-prior-phi1-mode | replace/bias | replace | Y | Y | n/a | replace | INCLUDE=replace |
| --lane-prior-phi1-bias-scale | bias scale | 1.0 | n/a | Y | n/a | (default) | EXCLUDE (unused for replace) |
| --lane-prior-phi1-source-pair | which pair's L* fit | 0 | n/a | Y | n/a | (default) | INCLUDE=0 |
| --lane-prior-phi1-dash-gate | model dash period | True | n/a | Y | tropical max-plus comb BUILT (Wave-F #229/#234) | on | INCLUDE (on) |

### A24 — #224 render-AA (5)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --render-aa | none/supersample/ipe | none | Y(none=BI) | Y | **MEAS: SIGNAL-A real-frame ceiling 0.00086 (floor proof) but SIGNAL-B witness brute-supersample HURTS −49%** (2026-07-02) | none | INCLUDE=none (SHIP `--render-aa none`; NEVER brute supersample) |
| --aa-supersample | supersample factor | 1 | Y(1=BI) | Y | (as above) | (default) | EXCLUDE (supersample hurts) |
| --aa-ipe-footprint | ipe footprint std | 1.0 | n/a | Y | UNMEAS | (default) | EXCLUDE (needs --render-aa ipe) |
| --aa-self-orient-fine-mode | refuse/batch/full | refuse | n/a | Y | ss=2 → ~86GB @ n600 (CONTAINMENT forbids) | (default) | EXCLUDE (memory-unsafe at n600; refuse is correct) |
| --aa-self-orient-fine-cache-cap | fine dir-feat cache | 16 | n/a | Y | n/a | (default) | EXCLUDE (needs batch mode) |

### A25 — #224 analytic lane-render-band (8)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --lane-render-band | composite analytic lane band (class-1 render authority) | False | Y | Y | FEED-dv #203/#213/#215; band net-S #205-gated | on | INCLUDE (on; the class-1 render-time authority — analytic lane band per `analytic_lane_band_primary_authority_decomposition`) |
| --lane-band-softness | coverage ramp width | 1.0 | n/a | Y | n/a | 1.0 | INCLUDE=1.0 |
| --lane-band-dash-forward-max-m | dash-gate forward cutoff | 55.0 | n/a | Y | #215 SegNet-Nyquist | 55.0 | INCLUDE=55.0 |
| --lane-band-uncertainty-source | witness/gt/none | witness | n/a | Y | FP-killer gate | witness | INCLUDE=witness |
| --lane-band-tau | uncertainty threshold | 0.85 | n/a | Y | UNMEAS | 0.85 | INCLUDE=0.85 |
| --lane-band-eps | uncertainty ramp | 0.35 | n/a | Y | UNMEAS | 0.35 | INCLUDE=0.35 |
| --lane-band-weight | band strength | 1.0 | n/a | Y | curriculum ramp | 1.0 | INCLUDE=1.0 |
| --lane-band-start-epoch | engage epoch | 300 | n/a | Y | **Ch.6 deep-math: DECONFLICT from tau@300 → 350** | 300 | **INCLUDE=350** (Ch.6: band@300 collides with tau@300 at full LR + stale momentum = the ep300 bump; one homotopy param at a time) |

### A26 — #224 pose-carrier (8)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --pose-carrier | warp frame0 via SE(3) ground-homography | False | Y | Y | **POSE OPEN+UNMEASURED on witness; warp catastrophic 3.7-10.3; 3.4e-5 = ANCESTOR never witness-validated** | on | UNSURE (INCLUDE=on to match #205 but pose d_seg⊥ and the witness pose is HELD-until-measured-byte-close; the carrier is store-nothing `generated` so ~0 marginal bytes — keep as #205 but VERDICT is provisional) |
| --pose-carrier-source | real_keyframe/generated | real_keyframe | Y | Y | Track B store-nothing-but-ξ (18927a1ae) | generated | INCLUDE=generated (store-nothing: ~0 marginal bytes, rule-118 render is free) |
| --pose-carrier-residual-mode | table/film | table | n/a | Y | table = (P,6) byte-minimal | table | INCLUDE=table |
| --pose-carrier-residual-scale | dξ scale | 1.0 | n/a | Y | n/a | (default) | INCLUDE=1.0 |
| --pose-carrier-s-t | ground-homography translation scale | None(fit) | Y | Y | self-calibrating fit | unset | INCLUDE=None (auto-fit at startup) |
| --pose-carrier-s-r | rotation scale | 0.0 | n/a | Y | measured d_pose-optimal | (default) | INCLUDE=0.0 |
| --pose-carrier-pitch | ground-plane pitch | 0.0 | n/a | Y | n/a | (default) | INCLUDE=0.0 |
| --pose-carrier-fit-pairs | s_t fit grid | 24 | n/a | Y | n/a | (default) | INCLUDE=24 |

### A27 — #224 persistence/topology loss (5)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --persistence-loss-weight | soft-clDice + persistence island-recall | 0.0 | Y(0=off) | Y | facet4(d): the auction-MBO area constraint surrogate; "births finest-scale erasure tail" | 1.0 | INCLUDE=1.0 (per-class AREA driving force that raises the nucleus basin — pairs with seed) |
| --persistence-recall-weight | w_recall | 1.0 | n/a | Y | n/a | 1.0 | INCLUDE=1.0 |
| --cldice-iters | soft-skeleton peel iters | 5 | n/a | Y | n/a | 5 | INCLUDE=5 |
| --persistence-warmup-epochs | linear warm-up | 0 | Y | Y | coarse→fine | 300 | INCLUDE=300 (warm-up to full weight by tau stage) |
| --persistence-classes | auto/comma-list | auto | n/a | Y | self-detects thin/small tail | auto | INCLUDE=auto |

### A28 — #224 island seed / containment / amplification (10)
| flag | what | default | dfl-BI? | B+T | MEAS | #205 | VERDICT |
|---|---|---|---|---|---|---|---|
| --seed-islands | EARLY-SEED finest-scale islands into φ target | False | Y | Y | **#208; the nucleation-memo physics-required fix (seed above critical nucleus)** | **OFF** | **INCLUDE (on) — THE nucleation fix; #205 did NOT have it → the single most important addition (paranoia #1)** |
| --island-dilate-px | annulus dilation | 1 | n/a | Y | **MEAS probe: native 6px σ=1.5→0.489 survival; +2px→0.979** | 1 | **INCLUDE=2** (dilate above critical nucleus so MCF GROWS not erases; facet3 DIL=2.0) |
| --seed-blend | island-seed blend | 1.0 | n/a | Y | n/a | (default) | INCLUDE=1.0 |
| --seed-lr | separate island-seed AdamW LR | 0.02 | n/a | Y | UNMEAS | (default) | INCLUDE=0.02 |
| --containment-mode | freeze/damp/shield | shield | n/a | Y | shield=zero destructive same-sign grad component | (default) | INCLUDE=shield (protects seed grad from bulk-CE wash) |
| --containment-damp | damp factor | 0.1 | n/a | Y | n/a | (default) | INCLUDE=0.1 (inert for shield) |
| --amplify-weight | island-birth term on shared _signed | 0.0 | Y(0=off) | Y | #208/#224; rides shared LEVER-4 _signed | 1.0 | INCLUDE=1.0 (BUT paranoia #1: in #205 it ran WITHOUT seed-islands → the LOSS can't nucleate zero mass; MUST pair with --seed-islands) |
| --amplify-form | hinge/softplus | hinge | n/a | Y | n/a | hinge | INCLUDE=hinge |
| --amplify-margin-target | margin the island must win | 1.0 | n/a | Y | n/a | 1.0 | INCLUDE=1.0 |
| --amplify-persist | uniform/inverse_thickness | inverse_thickness | n/a | Y | up-weights thinnest tail | inverse_thickness | INCLUDE=inverse_thickness |

---

## §B. BASE-TRAINER-ONLY flags (NOT on the live path — all EXCLUDE-from-fresh-run)
`train_witness_realized_through_R_mlx.py` has its own standalone argparse (87 flags). These are NOT
consumed by the levelset entry point; several are conceptual ancestors under different names. Do NOT put
any of these in the fresh-run argv (they'd error). Notable ones (EXCLUDE, informational only):
`--n-fourier --fourier-sigma --witness-bytes --int8-verdict --ema-lag-warn-ratio --verdict-device
--verdict-which --ema-verdict-every --optimizer{adamw,md} --md-base --md-gain-lr-scale --n-restarts
--grad-log-every --basis{isotropic,directional} --siren-omega --finer-first-bias-scale --wire-scale
--activation{…,siren,finer,…} --margin-weighted-loss --margin-weight-fn --margin-weight-temp
--margin-weight-start-epoch --margin-weight-temp-start --margin-weight-anneal-epochs
--margin-stage-lr-warmup-epochs --margin-stage-lr-floor --muon-finisher-start-epoch --muon-adam-lr
--muon-all-params --plateau-trigger --plateau-slope-eps --plateau-window --tau-anneal-start/end
--l7-thr-anneal-start/end`.
**Note the two conceptual gaps worth a future levelset port:** (1) `--optimizer md` (MD-Decoupling,
arXiv 2606.25971 — warmup-free, width-transferable) is a base-only ablation with NO levelset flag; (2)
`--plateau-trigger` (advance a curriculum stage on MEASURED d_seg plateau) is base-only — the levelset
trainer uses FIXED `--*-start-epoch` caps. The FACET-5 control monitor (BUILD queue) is the levelset-side
answer to plateau-triggering.

---

## §C. FRESH-RUN INCLUDE-LIST DRAFT (the argv; net-S #205-gated, operator-GO-gated)
Built by layering the change-set on #205's optimal-form config. **CONTAINMENT: a config PROPOSAL, not a
launch.** Shape-changing flags (`--mod-dim 19`, `--bank-n-scales 6`) FORCE a fresh arm (cannot warm-start
#205's CE ckpt). Sub-items marked ⚠ are CALIBRATE/MEASURE-FIRST — resolve before or A/B within the run.

```
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 .venv/bin/python \
  experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir <fresh> --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 --mlx-device gpu --seed 0 --epochs 1000 --eval-every 25 \
  --verdict-pairs 0 --async-verdict --verdict-batch 32 --mlx-cache-clear-accum 1 \
  --curriculum --tau-softplus-start-epoch 300 --tau-softplus-tau 0.3 --l7-start-epoch 1000 \
  --muon-start-epoch 726 --muon-lr 0.002 --muon-momentum 0.95 --muon-ns-steps 5 \
  --muon-warm-start-momentum --muon-lr-final-frac 0.1 \                 # ADD (Ch.6/#270)
  --stage-transition-rewarmup-epochs 20 --stage-transition-rewarmup-floor 0.1 \
  --stage-transition-rewarmup-shape cosine --stage-transition-reset-moments \  # rewarmup 8→20, linear→cosine
  --w-seg 100 --w-pose 1.0 --score-domain-loss \
  --pose-carrier --pose-carrier-residual-mode table --pose-carrier-source generated \
  --mod-dim 19 \                                                        # ⚠ CALIBRATE 32→19 (Whitney; fresh arm)
  --hidden-dim 96 --n-hidden 4 \                                        # ⚠ re-derive RD-optimum with mod-19
  --activation hosc --hosc-beta 1.0 --hosc-beta-end 4.0 --hosc-beta-anneal linear --hosc-omega 1.0 \
  --siren-init --softmax-temp-start 1.0 \
  --softmax-temp-end 0.05 \                                             # ⚠ MEASURE-FIRST {0.05,0.1,0.25}
  --tau-anneal-shape geometric \                                        # ADD (cosine→geometric)
  --self-orient --n-dir-freqs 2 --freq-across 32 --freq-along 4 --reorient-every 50 \
  --bank-n-scales 6 --max-bank-freq 64 \                               # ADD bank-n-scales 4→6 (fresh arm)
  --film-stiefel --code-spectral-entropy-weight 0.05 \                 # ADD film-stiefel; ⚠ CALIBRATE β
  --chroma --palette-anchor \
  --eikonal-weight 0.05 \                                              # ADD 0.01→0.05 (interface-width #286)
  --length-weight 0.001 \                                              # KEEP (do NOT raise — MCF driver)
  --render-h 384 --render-w 512 --render-aa none \
  --lane-render-band --lane-band-start-epoch 350 \                    # band 300→350 (Ch.6 deconflict)
  --lane-band-uncertainty-source witness --lane-band-tau 0.85 --lane-band-eps 0.35 \
  --lane-band-softness 1.0 --lane-band-dash-forward-max-m 55.0 --lane-band-weight 1.0 \
  --persistence-loss-weight 1.0 --persistence-recall-weight 1.0 --cldice-iters 5 \
  --persistence-warmup-epochs 300 --persistence-classes auto \
  --lane-thin-weight 0.5 --lane-thin-start-epoch 300 \                # ⚠ CALIBRATE weight (area surrogate)
  --amplify-weight 1.0 --amplify-form hinge --amplify-margin-target 1.0 \
  --amplify-persist inverse_thickness \
  --seed-islands --island-dilate-px 2 --containment-mode shield \      # ADD seed-islands; dilate 1→2 (THE FIX)
  --structured-init --structured-init-include-lane \
  --lane-prior-phi1 --lane-prior-phi1-mode replace --lane-prior-phi1-dash-gate \
  --accum-pairs 8 --grad-clip 1.0 --ema-decay 0.997 --lr 1e-3 --lr-end 1e-4 --weight-decay 1e-4 \
  --adam-beta2 0.999 \                                                 # ⚠ CALIBRATE→0.9999999 (small-n)
  --ckpt-every 25 --stage-checkpoints --cache-gt-skeleton
```

---

## §D. #205-vs-FRESH DIFF (the actionable change-set — 12 changes)
| # | change | from (#205) | to (fresh) | why | verdict class |
|---|---|---|---|---|---|
| 1 | **--seed-islands** | OFF | **ON** | nucleation fix: seed the lane above critical nucleus at INIT (the LOSS alone can't birth zero mass) | INCLUDE |
| 2 | **--island-dilate-px** | 1 | **2** | +2px clears the MCF critical nucleus (probe: σ=1.5 survival 0.49→0.98) | INCLUDE |
| 3 | **--eikonal-weight** | 0.01 | **0.05** | interface-width control (#286): keep thin lane in 94%-survival regime, enable τ-floor | INCLUDE |
| 4 | **--tau-anneal-shape** | cosine | **geometric** | τ-Fisher-Rao geodesic; equal epochs/octave; damps late-τ volatility | INCLUDE |
| 5 | **--mod-dim** | 32 | **19** ⚠ | Whitney 2m+1 (m≈8); rate win; shape-change→fresh arm | CALIBRATE (measure N-term slope first) |
| 6 | **--bank-n-scales** | 4 (default) | **6** | use the Nyquist radial budget (only 2 of 6 octaves used at 4); shape-change→fresh arm | INCLUDE (slope-gated) |
| 7 | **--film-stiefel** | OFF | **ON** | byte-free FiLM rank fix (PR(M) 1.19→4.57 at 0 bytes) | INCLUDE |
| 8 | **--code-spectral-entropy-weight** | 0.0 | **~0.05** ⚠ | other half of byte-free rank cure; β unmeasured | CALIBRATE |
| 9 | **--muon-warm-start-momentum** | OFF | **ON** | kills cold-start +0.000357 spike at ep726 (#270) | INCLUDE |
| 10 | **--muon-lr-final-frac** | 1.0 | **0.1** | cosine-decay Muon LR; settle the finisher (#270) | INCLUDE |
| 11 | **--lane-band-start-epoch** | 300 | **350** | Ch.6: deconflict from tau@300 (the ep300 bump); one homotopy param at a time | INCLUDE |
| 12 | **--stage-transition-rewarmup**: epochs 8→20, shape linear→cosine | 8/linear | **20/cosine** | Ch.6 reduced-step corrector at boundaries | INCLUDE |
| (13) | **--lane-thin-weight** | (off) | **~0.5** ⚠ | per-class area surrogate holding dropped-dash mass | CALIBRATE (add) |
| (14) | **--softmax-temp-end** | 0.05 | **{0.05,0.1,0.25}** ⚠ | resolution-floor direction derived; magnitude unit-uncertain | MEASURE-FIRST |
| (15) | **--adam-beta2** | 0.999 | **0.9999999** ⚠ | small-n (n~75) optimum | CALIBRATE |

**UNCHANGED (correctly kept from #205):** render 384×512, hosc β 1→4 anneal, self-orient n_dir=2, chroma,
palette-anchor, persistence 1.0 warmup 300, amplify 1.0, structured-init + lane-prior-phi1, length 0.001
(KEEP), muon@726 momentum 0.95, verdict-batch 32, pose-carrier generated table, EMA 0.997, grad-clip 1.0,
accum 8, stage-checkpoints, ckpt-every 25, render-aa none.

---

## §E. QUEUES

### MEASURE-FIRST ($0 gate owed before INCLUDE)
1. **--softmax-temp-end resolution floor** — A/B {0.05, 0.1, 0.25}; unit-convention of the SDF pixel scale UNVERIFIED (facet4 §1.3 caveat). Direction derived (raise); magnitude measured.
2. **--mod-dim 19 N-term slope** — the $0 GT-margin N-term log-log slope (facet2 §3.2) confirms m≈8 BEFORE committing; if m>9, 19 UNDER-embeds → capacity cliff (sister of bc20 under-capacity).
3. **--head etf / --logit-adjust-per-class / --margin-field-head-weight** — #218 Laguerre head BUILT but no byte-closed row; probe the ETF rate-win + rare-class lift before INCLUDE.
4. **per-class d_seg attribution on real #205 @n600** — confirm the creep = lane at scale (memory-gated; concurrent n600 witness-eval risks the >128GB machine-crash P0 gate).
5. **--bank-n-scales 6 finer-octave payoff** — the §3.2 slope also tells whether the finer octaves PAY in d_seg or are wasted atoms.

### BUILD (flag/underlying incomplete — NOT config)
1. **Ch.5 M2 NTK/feature-Gram whitening** — UNBUILT (no flag); the EXPONENT lever + dominant SPEED lever (~3-10×, facet1/facet2). The real exponent-changer; #204/#207 sig-proc lineage.
2. **--margin-saliency-reachability sR key** — needs `tools/precompute_sR_reachability.py` to write the 'sR' key into gt-cache; the exact-S_R replacement for the measured-inert texture proxy.
3. **facet5 control monitor** — de-orphan `tools/render_witness_trajectory_dynamics.py` into a lean resumable τ-creep early-stop instrument (the levelset answer to base-only `--plateau-trigger`).
4. **shearlet front-end** (`front_end="shearlet"`) — facet2 genuine build (anisotropic-WIRE parabolic-shearlet); numpy ref + MLX fwd + argmax-parity.
5. **geometry-native solvers** (deepmath §Tier3): damped-Newton semi-discrete OT head-offset (asymmetry cure, replaces Menon heuristic), auction-MBO volume-preserving flow (the proven-erasure per-class area solver — the principled cure to law #8), RKMK Lie-group ξ-transport. Each $0-gated before wiring.
6. **step_basis MLX port** — LearnableStepBasis is torch-side only (base-trainer note); port if hosc transfers.

### CALIBRATE (value uncertain, A/B)
- --mod-dim 19 (also measure-first) · --code-spectral-entropy-weight β · --lane-thin-weight magnitude · --muon-lr 0.002 (finishing-schedule tune) · --adam-beta2 0.9999999 · --island-dilate-px (2 chosen; the exact critical-nucleus knee is a $0 probe owed) · --ema-decay-finisher (SWA) · --eikonal-junction-relax (pairs with 0.05 eikonal) · secondary d_seg levers as isolated arms: --margin-saliency-weight, --seg-chroma-boundary-weight (chroma 93.4% margin-annulus — promising), --seg-subpix-boundary-weight, --seg-spike-reweight (flicker-floor path), --code-nuclear-weight.

---

## §F. TOP-5 PARANOIA FLAGS (levers most likely WRONGLY in/out)
1. **--amplify-weight 1.0 ran WITHOUT --seed-islands in #205 (near-silent-no-op).** The island-birth LOSS is active (`island_weight_mx` built whenever amplify>0, verified `train_levelset…:2075-2090`), but it can only up-weight pixels the witness ALREADY has mass to win — it CANNOT nucleate a zero-mass lane. #205 had the amplification loss FIGHTING the MCF erosion of a zero-mass class. **Fresh run MUST pair amplify WITH --seed-islands AND verify `part_frac[lane] > 0` at ep0 in the log** (don't trust the flag — MEASURE the seed).
2. **--structured-init + --structured-init-include-lane + --lane-prior-phi1 seeded the lane at part_frac 0.0 in #205 (the flags LOOK like they seed the lane, MEASURED they didn't).** The nucleation memo's decisive log finding: `lane_px=0, lane_mean_iou=0.0` DESPITE all three lane-init flags on. The seed is subtle; the fresh run's acceptance gate is `part_frac[lane]≈0.006 (~0.59% known lane)` at ep0, NOT the presence of the flags. This is why --seed-islands + dilate-2 is the physics-required addition.
3. **--softmax-temp-end raise is DIRECTION-derived but MAGNITUDE unit-uncertain (facet4 §1.3 NO-FAKE caveat).** Raising it wrong stalls descent (too high → never sharpens) OR wastes compute (too low → sub-grid refinement of an unreadable boundary). It reads as an obvious INCLUDE but is a MEASURE-FIRST — putting 0.25 in blind could stall the whole run.
4. **--mod-dim 32→19 is a DERIVED-from-MEASURED rate win that risks a CAPACITY CLIFF.** m≈8 is a manifold-dim estimate; if the true intrinsic dim is >9, 19 under-embeds (non-injective) → the bc20 under-capacity trap under a new name. The $0 N-term slope must confirm m≈8 before committing; otherwise keep 32 (a rate loss beats a capacity cliff). Easy to wrongly INCLUDE=19 for the rate headline.
5. **--length-weight: the naive instinct is to RAISE it (smoother = better), but it IS the surface-tension term whose gradient is exactly the MCF `V=−κ` that ERODES the thin high-curvature lane** (facet4 §2.2(c)). Keeping it at 0.001 is correct and counter-intuitive; a reviewer optimizing "smoother regions" would wrongly raise it and accelerate the exact nucleation failure. Flagged LOUDLY: **do NOT raise length-weight.**

---

## §G. UNSURE — need operator / cross-surface-synthesis judgment
- **--epochs / --l7-start-epoch / --muon-start-epoch under the NEW τ dynamics.** #205 set `l7@1000==epochs` (l7 NEVER runs) and `muon@726`. With geometric-τ + raised eikonal + the seed, does the tau stage want to be LONGER (more MCF-with-driving-force to grow the lane) before Muon? Is skipping l7 intended, or should l7 run (with --l7-threshold→0.42)? The facet docs don't prescribe epochs — synthesis call.
- **--hidden-dim with --mod-dim 19.** #205's RD-optimum (config-review#2) was derived at mod-32×hidden-96→122KB. With mod-19 the rate drops; is hidden-96 still optimal or can it grow (more decoder capacity at the freed rate budget)?
- **--pose-carrier verdict is provisional.** Pose is OPEN+UNMEASURED on the witness (memory `project_pose_solved_screw…`: warp catastrophic 3.7-10.3; 3.4e-5 is ANCESTOR-only). Keeping it ON matches #205 and store-nothing `generated` is ~0 bytes, but whether the witness pose term HELPS or HARMS d_seg convergence at the fresh config is unmeasured. Include-to-match or drop-to-isolate-d_seg?
- **How many secondary d_seg levers to stack in ONE run vs isolate.** The seed + eikonal + geometric-τ + film-stiefel + persistence + amplify is already a big composition. Adding lane-thin / chroma-boundary / margin-saliency / subpix / spike-reweight risks un-attributable interactions (per-stage-treatment discipline). Recommend: fresh run = seed-fix + Ch.6 + DIM + geometric-τ ONLY; the secondary d_seg levers as SUBSEQUENT isolated A/Bs. Operator call on stack depth.
- **--adam-beta2 0.9999999 (small-n optimum) — #205 kept 0.999.** The theory (arXiv 2603.02092) is sound but the change was never A/B'd; is it worth folding into the fresh run or isolating?
- **--code-spectral-entropy-weight β = 0.05 is a GUESS** (facet2 says "small β", no measured value). Needs a calibration sweep {0.01, 0.05, 0.1} or drop to film-stiefel-only.

---

**Totals:** 189 levelset-trainer flags audited (+87 base-only, all EXCLUDE-from-live). Fresh-run INCLUDE
set ≈ #205's config + 12 primary changes (§D). MEASURE-FIRST: 5 · BUILD: 6 · CALIBRATE: ~12. Pointer
**0.19110 UNMOVED** — every verdict is net-S #205-gated + operator-GO-gated (CONTAINMENT; no autonomous
heavy launch; the P0 system-memory governor binds any n600 dispatch). Sisters: surfaces B/C of this sweep
· `lane_nucleation_failure_seed_above_critical_nucleus_20260704` · `scaling_law_facet{1,2,4,5}` ·
`deepmath_converged_next_run_config_20260704`.
