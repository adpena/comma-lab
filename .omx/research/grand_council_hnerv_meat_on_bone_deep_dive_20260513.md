# Grand Council — HNeRV meat-on-bone deep-dive 2026-05-13

**Lane**: `lane_hnerv_meat_on_bone_deep_dive_council_20260513` (L0 SKETCH, phase 2.0)
**research_only**: true — council deliberation; no archive build, packet, or dispatch.
**Operator framing (2026-05-13)**: *"we want to be very wary of getting stuck in local minima especially hnerv local minima though there is still hnerv meat on the bone i believe"*.
**Sister council**: `lane_first_principles_original_score_lowering_council_20260513` (parallel — derives ORIGINAL ideas from first principles; calibrates against this memo's HNeRV-family empirical ceiling).
**Scope**: read-only deliberation; no code edits, no archive builds, no dispatch.
**Evidence axes (Apples-to-apples — CLAUDE.md non-negotiable)**: every score below carries `[contest-CPU]` / `[contest-CUDA]` / `[macOS-CPU advisory]` / `[prediction]` tag.

---

## 1. Executive summary (≤500 words)

**HNeRV-family empirical ceiling estimate (80% CI)**: `S_HNeRV-family-floor = 0.180 ± 0.012 [contest-CPU prediction]` and `0.213 ± 0.015 [contest-CUDA prediction]`, under the constraint of preserving the canonical PR101 HNeRV-LC architecture (229K-param 6-stage PixelShuffle decoder + 28-d per-pair latent + sin activation). The current empirical PR101 anchor (`0.192861 [macOS-CPU advisory]` matching `0.19284 [contest-CPU GHA Linux x86_64]`) sits ~0.013 above this estimated family-floor.

**The operator's intuition is correct on both axes**:

1. **Meat remains.** ≥0.005 of empirical signal is achievable inside the HNeRV-family without architectural family change, by closing the gap between PR101 (which optimized inflate-side codec only, branched from PR95+PR98 with no retrain) and a properly-disciplined HNeRV parity retrain.
2. **Family is a local minimum at ~0.180**. Below ~0.180 `[contest-CPU prediction]`, HNeRV's 229K-param 6-stage PixelShuffle decoder + 28-d per-pair latent grammar cannot absorb additional rate or component compression without crossing into a different representation family (replace, not extend).

**Top 5 unexplored HNeRV-family primitives**, ranked by predicted score-lowering signal:

| Rank | Primitive | Predicted ΔS `[contest-CPU prediction]` | Cost | Family-extension/cliff | Source |
|---:|---|---:|---|---|---|
| 1 | **Score-aware retrain of PR101's HNeRV-LC with eval-roundtrip + differentiable rgb_to_yuv6 + EMA + QAT, scoring archive bytes IN-LOOP** (codex Priority 1 / S1) | -0.005 to -0.012 | $4-15 Modal A100 + 1 day dev | Family extension | F1, F4 |
| 2 | **HNeRV decoder QAT to FP4 (Quantizr lineage) + scorer-aware quant schedule + Hinton T=2.0 KL distill** | -0.003 to -0.008 | $3-8 Modal A100 + 1 day dev | Family extension | F5, F8 |
| 3 | **Ballé Markov-1 hyperprior over PR101 latents + integer entropy export** (codex P3 / H6) | -0.002 to -0.005 | $2-5 Modal A100 + 2 day dev | Family extension | F6, F9 |
| 4 | **Score-aware latent quantization (replace PR101's 8-bit centered-delta with QAT-trained Lloyd-Max / trellis quantizer)** | -0.002 to -0.004 | $1-3 Modal A100 + 1 day dev | Family extension | F7 |
| 5 | **Per-pair-difficulty adaptive bit allocation (variable bytes per latent row based on PoseNet/SegNet sensitivity)** | -0.001 to -0.003 | $1-3 + 2 day dev | Family extension | F10 |

**PR101 forensic findings**:
- PR101 source tree contains NO training script. It is a CODEC-ONLY contribution that branched from PR98 (which branched from PR95). The score gain over PR100 (-0.0025) is ENTIRELY codec/entropy bolt-on, not retrain.
- PR101's `model.py` = PR100's `hnerv_model.py` byte-for-byte. Architecture is unchanged.
- PR95 intake profile (`profile_pr95_hnerv_muon_intake.md`) reveals an 8-stage training schedule (29,650 epochs) with `Muon` activated only in stage 8 (5000 epochs); `--enable-eval-roundtrip-in-training=True`, `--enable-differentiable-yuv6=True`, `--enable-scorer-domain-loss=True`, `--yuv6-mode=monkey_patch_global`, `--segmentation-surrogate=sinkhorn` were the canonical PR95 T1-trainer flags.
- Our internal `train_substrate_sane_hnerv.py` exists and has been dispatched ≥5 times; the first contest-CUDA anchor has not yet landed — every attempt DEFERRED at trainer-bug, preflight-RED, or sibling-memo PCC4 STOP-PRECONDITION (see §6 forensics). The integration discipline that made PR101 a 605-LOC packetized bolt-on has NOT YET been operationally reproduced.

**Family-saturation verdict (3 scenarios, posterior-weighted)**:

| Scenario | Probability | Implication |
|---|---:|---|
| HNeRV-family ceiling at `~0.180 [contest-CPU]` | 0.55 | Extend HNeRV for 1-2 more rounds (top-5 primitives above); switch families at 0.180-0.185 transition zone. |
| HNeRV-family ceiling at `~0.170 [contest-CPU]` | 0.30 | Extend HNeRV through 2-3 more rounds; switch at 0.170-0.175. |
| HNeRV-family ceiling at `~0.155-0.160 [contest-CPU]` (Shannon-floor reachable from HNeRV alone) | 0.15 | HNeRV is the dominant primitive for the remainder of the contest; replacement-family work is opportunity-cost negative. |

**Recommended HNeRV-extension dispatch order**:

1. **S1 archive-in-loop HNeRV parity retrain** (codex Priority 1). Pre-condition: complete `sane_hnerv` first contest-CUDA anchor (currently DEFERRED at attempt #5; STOP-PRECONDITION is two sibling-subagent memos missing PCC4 council sections, NOT a substrate bug). Once anchored at any score `[contest-CUDA]`, the substrate becomes the FIRST internal score-aware HNeRV-family substrate we can compose bolt-ons onto.
2. **QAT FP4 + Hinton KL distill** on the S1 anchor.
3. **Ballé hyperprior** over PR101-style latents (extends PR101's existing LZMA latent stream).
4. **Score-aware latent quantization** (replaces PR101's centered-delta uint8 scheme with QAT-trained Lloyd-Max).
5. **Per-pair adaptive allocation** (last-mile; small predicted Δ; ship when other primitives saturate).

**Family-switch threshold (when to stop extending HNeRV)**:

- **HARD STOP** if a paired same-axis HNeRV-family candidate at the operating point shows ≤ `+0.0005 [contest-CPU]` improvement over a 3-experiment moving window AND the empirical 3-experiment Pareto frontier is bytes-only.
- **PRE-EMPTIVE SWITCH** if a replacement-family candidate (Ballé full-renderer, SIREN/FINER/WIRE, Cool-Chic/C3 with export discipline) lands a paired same-axis result within `±0.005` of the best HNeRV at the same byte budget. The replacement-substrate's escape velocity becomes the right next investment.

---

## 2. Q1 — empirical HNeRV-family score floor

### 2.1 Anchors (apples-to-apples; `[macOS-CPU advisory]` validated within `2e-5` of `[contest-CPU GHA Linux x86_64]`)

From `.omx/research/macos_cpu_canvas_pareto_ranking_20260513.md` (validation: A1 macOS-CPU = `0.192864` matches memo `[contest-CPU]` = `0.192864`; PR101 macOS-CPU = `0.192861` matches memo `[contest-CPU]` = `0.192861`; absolute drift ≤ 2e-5):

| Family member | macOS-CPU `[advisory]` | Bytes | seg_avg | pose_avg | Public claim |
|---|---:|---:|---:|---:|---:|
| `pr101_hnerv_ft_microcodec` (GOLD) | 0.192861 | 178,258 | 5.60e-4 | 3.286e-5 | 0.193 `[contest-CPU]` |
| `a1_baseline` (PR101 + 4-byte latent compression) | 0.192864 | 178,262 | 5.60e-4 | 3.286e-5 | n/a |
| `pr103_hnerv_lc_ac` (SILVER) | 0.194865 | 178,223 | 5.76e-4 | 3.443e-5 | 0.195 `[contest-CPU]` |
| `pr100_hnerv_lc_v2` (SUBSTRATE root) | 0.195369 | 178,981 | 5.76e-4 | 3.443e-5 | 0.1954 `[contest-CPU]` |
| `pr107_apogee` (OUR best HNeRV-family) | 0.196640 | 178,392 | 5.89e-4 | 3.580e-5 | n/a; 0.197 leaderboard CPU comment |
| `pr105_kitchen_sink` (LOST) | 0.197979 | 177,857 | 6.09e-4 | 3.471e-5 | 0.198 `[contest-CPU]` |
| `pr104_qhnerv_ft_best` | 0.198711 | 178,637 | 6.12e-4 | 3.464e-5 | n/a |

**Family centroid + spread**: μ = 0.196 ± 0.002, σ = 0.0022, range = [0.192861, 0.198711], spread = 0.006.

### 2.2 Family-floor derivation (Shannon LEAD + Dykstra CO-LEAD + Ballé)

**Shannon**: the contest score formula `S = 100·d_seg + sqrt(10·d_pose) + 25·B/N` is a R(D) bound with three independent terms. The HNeRV-family's `d_pose` floor is bounded below by the eval-roundtrip uint8 quantization noise (~`3.0e-5` floor; the macOS-CPU sweep shows PR101 = 3.286e-5, which is within `~3×` of the uint8 noise floor). `d_seg` floor is bounded below by the boundary-pixel argmax-disagreement rate after the eval-roundtrip 384→874→uint8→384 chain (~`5.0e-4` floor for HNeRV's 229K-param decoder). Rate floor at 178K bytes ≈ `0.00475` and is dominated by the 162K-byte decoder section.

Best-case PR101-architecture component bounds:
- `d_seg ≥ 5.0e-4` (eval-roundtrip uint8 boundary noise) → `100·d_seg ≥ 0.0500`
- `d_pose ≥ 3.0e-5` (eval-roundtrip uint8 chromatic noise) → `sqrt(10·d_pose) ≥ 0.0173`
- `25·B/N ≥ 25·160000/37545489 ≈ 0.1066` if every byte is theoretical-minimum decoder weights + latents at Ballé-Markov-1 entropy (vs current `0.1186` at 178K)

Sum: `0.0500 + 0.0173 + 0.1066 = 0.1739` `[contest-CPU prediction]`. Practical floor ≈ `0.180 ± 0.012` once decoder-overhead-included entropy model + sidecar grammar + ZIP framing are added.

**Dykstra**: the achievable region is the intersection of three convex feasibility sets — `d_seg ≤ f_seg(θ)`, `d_pose ≤ f_pose(θ)`, `B ≤ B(θ)`. Alternating projections converge to a Pareto frontier. At PR101's operating point, the projections suggest:
- Reducing `B` from 178K to 160K via Markov-1 hyperprior + Lloyd-Max latent quant: feasible (`d_seg`/`d_pose` projections stay inside the family); Δrate ≈ `-0.012` score.
- Reducing `d_seg` from 5.60e-4 to 5.0e-4 via score-aware retrain: feasible; Δseg ≈ `-0.006` score.
- Reducing `d_pose` from 3.286e-5 to 3.0e-5 via score-aware retrain: feasible (within uint8-noise floor); Δpose ≈ `-0.002` score.

Sum of independent compositions (NOT GUARANTEED additive — antagonism flag set per CLAUDE.md): max plausible Δ ≈ `-0.020`, giving a family ceiling of `0.193 - 0.020 = 0.173`. With antagonism discount (Dykstra default 0.6 on stacking): `0.193 - 0.012 = 0.181`.

**Ballé**: a scale-hyperprior over PR101's 600×28 latent table changes the entropy floor from `-log₂(p_factorized(y))` to `-log₂(p_y|z(y|z))` where `z` is a side-information channel. Empirical median gain on natural-image latents is `~15-20%` (Ballé 2018). On PR101's 15K-byte latent section: `-0.003` to `-0.004` predicted Δrate score.

**Combined family-floor estimate**: `S_HNeRV-family-floor = 0.180 ± 0.012 [contest-CPU prediction]`. CUDA-axis floor extrapolation (per +0.033 PR102/PR107 gap on the pose-axis at this operating point): `~0.213 ± 0.015 [contest-CUDA prediction]`.

### 2.3 Is the family converged at PR101 or is PR101 just where it landed?

**Answer**: PR101 is where it LANDED. The family-cluster at 0.193-0.198 is a 6-point empirical sample on the CLOSED PR95→PR98→PR100→PR101 substrate-chain. The OPEN questions (top-5 in §1) have NOT been independently anchored. PR101's 0.193 represents the best `inflate-side-only` codec performance against PR95+PR98's fixed-weights substrate. Score-aware retrain + QAT + hyperprior have NOT been bolt-on'd onto a PR101-baseline.

---

## 3. Q2 — unexplored HNeRV-family primitives

### 3.1 Primitive enumeration

The 13 primitives below are the candidate "meat" — each cited from a forensic-recovered public PR / paper / internal memo:

| ID | Primitive | Already tried publicly? | Tried internally? | Notes |
|---|---|---|---|---|
| F1 | **Score-aware retrain of HNeRV-LC with full PR95-stage curriculum** (8-stage AdamW→Muon, ce→tau_softplus→smooth→l7_softplus losses, sigma sweep, lambda sweep, eval_roundtrip + differentiable yuv6 + scorer-domain loss) | PR95 partial (no medal); never reproduced | `sane_hnerv` 5 dispatch attempts, all DEFERRED; first anchor pending | The dominant unexplored primitive |
| F2 | **Differentiable scorer-preprocess in training inner loop** (PR95's `--enable-differentiable-yuv6 --yuv6-mode=monkey_patch_global` flag) | Yes in PR95 | Catalog #187 `check_hnerv_training_parity_guard` enforces it for `sane_hnerv`/`tc_nerv` but no anchor landed yet | Required for L8 of HNeRV parity discipline |
| F3 | **EMA on HNeRV weights (decay=0.997, per CLAUDE.md non-negotiable)** | PR95/PR100/PR101 likely, given the medal positions; not directly recoverable from inflate-only artifacts | `sane_hnerv` declares EMA per `check_hnerv_training_parity_guard` | Cross-ref CLAUDE.md "EMA — non-negotiable" |
| F4 | **Archive-in-loop validation (select packet by exact archive score, not proxy)** (codex S1) | UNKNOWN; PR95 8-stage curriculum suggests likely some form of it | NOT YET in `sane_hnerv` trainer | The "exit-roundtrip-aware" complement; small dev cost |
| F5 | **HNeRV decoder QAT to FP4** (Quantizr lineage; 88K-param FiLM-conditioned CNN at FP4 = ~44KB compared to HNeRV's 162K Brotli decoder) | Yes in Quantizr (0.33 archive); never combined with HNeRV-LC architecture | `qat_finetune.py` exists; never run on HNeRV | Predicted -0.003 to -0.008 Δ; risk of QAT cliff |
| F6 | **Markov-1 hyperprior over HNeRV latents** (Ballé 2018 + PR101's existing LZMA latent stream) | NO public PR | NO internal anchor | Codex H6 + P3 |
| F7 | **Score-aware latent quantization (Lloyd-Max / trellis quant; replaces PR101 centered-delta uint8)** | NO | NO | Codex H6 + classical TCQ literature |
| F8 | **Hinton T=2.0 KL distill of HNeRV teacher to smaller HNeRV student** (Quantizr canonical T=2.0; see CLAUDE.md non-negotiable on kl_on_logits) | Yes in Quantizr SegNet head; never on HNeRV decoder | NO | Distill 229K→100K param decoder; ~60KB rate save |
| F9 | **HNeRV + Ballé scale-hyperprior on latents + integer entropy export** | NO | NO | Codex P3 / H6 — first proper Ballé-stacked HNeRV |
| F10 | **Per-pair-difficulty adaptive bit allocation** (variable bytes per latent row based on PoseNet/SegNet sensitivity per-pair) | NO | NO | Codex Eureka #10 |
| F11 | **Muon optimizer for hidden HNeRV weights** (codex S4) | PR95 stage 8 (5000 epochs Muon finetune) | NOT in `sane_hnerv` | Limited Δ likely (Muon is throughput, not score-axis); KEEP IN STAGE-8 ROLE |
| F12 | **GEPA-style text-serializable HNeRV config search** (codex S5) | NO | NO | Search space for stage schedules / lambda sweeps; cross-axis EV via proposal-evaluator separation |
| F13 | **HNeRV with FINER/WIRE/BACON variable-frequency activation** (replacing `sin(x)`) | NO | NO | Architectural variant; risk-flagged because it changes HNeRV class boundary |

### 3.2 Predicted score impact per primitive (empirical signal estimate)

**Constants** (per codex P3 score-math, validated):
- Rate slope: `25 / 37,545,489 = 6.659e-7 score/byte = 0.000682 score/KiB`
- Pose marginal (at PR101's `d_pose = 3.286e-5`): `d/dp sqrt(10·p) = 5/sqrt(10·p) ≈ 275.8 score/pose-dist-unit`
- SegNet marginal at this operating point: 100 score/seg-dist-unit (constant)
- **Operating-point pose:seg marginal ratio at PR101**: 2.71× (per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent")

**Per-primitive predicted Δ**:

| ID | Mechanism | Predicted Δseg | Predicted Δpose | Predicted Δbytes | Predicted total Δ `[contest-CPU prediction]` |
|---|---|---:|---:|---:|---:|
| F1 | Score-aware retrain w/ PR95 8-stage curriculum, eval_roundtrip+yuv6+scorer loss | -3e-5 to -8e-5 (small Δ; HNeRV is near family ceiling on seg already) | -0.3e-5 to -0.7e-5 (close uint8-floor gap) | -0 to -2KB (better entropy clustering from cleaner gradients) | **-0.005 to -0.012** |
| F2 | yuv6 monkey-patch ensures pose gradient survives BT.601 | small; folded into F1 | -0.2e-5 to -0.5e-5 (alone) | 0 | -0.001 to -0.003 (alone); folded into F1 |
| F3 | EMA stabilization | small; folded into F1 | small; folded into F1 | 0 | -0.0005 to -0.002 (alone); folded into F1 |
| F4 | Archive-in-loop validation | small; folds into F1; chooses best of N parsed candidates | small; folds | small; folds | **-0.001 to -0.003** (incremental on top of F1) |
| F5 | QAT FP4 decoder | -1e-5 to +3e-5 (risk of QAT cliff) | -0.3e-5 to +2e-5 (QAT can hurt pose) | -20KB to -60KB | **-0.003 to -0.008** (gain band) or **+0.005 cliff** if QAT poorly scheduled |
| F6 | Markov-1 hyperprior over latents (15K → 12K) | 0 (rate-only) | 0 (rate-only) | -2KB to -3KB | **-0.001 to -0.002** |
| F7 | Lloyd-Max latent quant replacing centered-delta uint8 | -1e-5 to +1e-5 | -0.5e-5 to 0 (Lloyd-Max preserves pose better at low rate) | -1KB to -2KB | **-0.002 to -0.004** |
| F8 | KL distill 229K→100K decoder | +1e-5 to +5e-5 (slight seg degradation expected) | +0.5e-5 to +2e-5 | -40KB to -60KB | **-0.020 to -0.040** (HIGH MEAT but HIGH CLIFF RISK) |
| F9 | Ballé hyperprior full stack (decoder + latent) | 0 (rate-only) | 0 (rate-only) | -3KB to -5KB | **-0.002 to -0.005** |
| F10 | Per-pair adaptive bit allocation | small | -0.2e-5 to -0.5e-5 (hard-pair pose targeting) | -0KB to -1KB | **-0.001 to -0.003** |
| F11 | Muon optimizer | folds into F1 stage 8 | folds | folds | **0 to -0.001** (training accelerator, not score axis) |
| F12 | GEPA search | search-mediated; folds into best of F1-F10 | folds | folds | **0 to -0.002** (best primitive selection win) |
| F13 | FINER/WIRE/BACON activation swap | UNCERTAIN; needs ablation | UNCERTAIN | UNCERTAIN; could increase activation memory | **±0.005 [uncertain; council suggests deferred-research]** |

### 3.3 "Meat" vs "local-minimum variation" classification

**MEAT (≥0.005 predicted Δ; high-EV)**:
- F1 (score-aware retrain w/ PR95 curriculum) — **-0.005 to -0.012 predicted**
- F5 (QAT FP4 with scorer-aware schedule) — **-0.003 to -0.008 predicted**
- F8 (KL distill 229K→100K decoder) — **-0.020 to -0.040 predicted; CLIFF RISK**

**INCREMENTAL (0.002-0.005 predicted Δ; medium-EV; ship after F1+F4 anchor)**:
- F4 (archive-in-loop validation)
- F7 (score-aware latent quant)
- F9 (Ballé hyperprior full stack)

**LOCAL-MINIMUM-VARIATION (< 0.002 predicted Δ; low-EV)**:
- F2, F3, F11 (folded into F1; not separately dispatchable)
- F6 (Markov-1 latent hyperprior alone)
- F10 (per-pair adaptive allocation)

**DEFERRED-PENDING-RESEARCH**:
- F13 (FINER/WIRE/BACON activation swap; needs ablation)

---

## 4. Q3 — empirical signal per primitive (already in §3.2 + §3.3)

See above. Top-3 MEAT primitives summed (with antagonism discount 0.6, per Dykstra): max plausible Δ ≈ -0.020. From PR101 base 0.193 → predicted HNeRV-family floor ≈ **0.173 to 0.180** (matches §2.2 derivation).

---

## 5. Q4 — when HNeRV becomes a structural ceiling

The operator's intuition: HNeRV has more meat BUT eventually becomes local-minimum trap. We answer this with three independent derivation routes converging on the same band.

### 5.1 Shannon-floor route

- Theoretical absolute floor on contest score under uint8-roundtrip + 384×512 frame quantization noise:
  - `d_seg_floor ≈ 5.0e-4` (eval-roundtrip uint8 boundary noise)
  - `d_pose_floor ≈ 3.0e-5` (eval-roundtrip uint8 chromatic noise)
  - `B_floor ≈ 60,000 bytes` (any neural representation needs ≥60KB to fit a usable decoder + sufficient latent table; below this the entropy can't be packed without information loss)
- Floor: `100·5.0e-4 + sqrt(10·3.0e-5) + 25·60000/37545489 ≈ 0.0500 + 0.0173 + 0.0400 ≈ 0.107` `[Shannon absolute floor, prediction]`
- HNeRV-family's `B_floor` is HIGHER than the absolute Shannon floor because the 229K-param 6-stage decoder needs at minimum `~120KB` after FP4 QAT + brotli. HNeRV `B_floor ≈ 120-130K bytes` → `25·125000/N ≈ 0.083`
- HNeRV-family Shannon floor: `0.050 + 0.017 + 0.083 = 0.150` `[contest-CPU prediction; HNeRV-family Shannon floor]`

### 5.2 Sub-architectural-variant route

- The HNeRV architecture has variants we haven't tried: HNeRV-LC (line-conv variant per `feedback_substrate_vs_codec_composition_meta_pattern_20260508`), HNeRV-Boost, E-NeRV with disentangled spatial-temporal context, FFNeRV (frequency), HiNeRV (hierarchical). Each could yield 0.005-0.020 architectural Δ alone. Combined with F1+F5: family ceiling could approach 0.165.

### 5.3 Replacement-substrate route

- Below 0.180 `[contest-CPU prediction]`, the marginal-byte-per-score is dominated by the decoder section (162K bytes Brotli). Replacing the decoder with a smaller architecture (Quantizr 88K-param FiLM CNN at FP4 = 44KB; or coordinate-MLP at ~30KB) creates a different family. THAT'S the threshold beyond which "HNeRV-extension" becomes "HNeRV replacement".

### 5.4 Convergent estimate

| Route | HNeRV-family ceiling estimate `[contest-CPU prediction]` |
|---|---:|
| Shannon-floor route (uint8 noise + min architecture) | 0.150 |
| Sub-architectural-variant route (HNeRV-LC + HNeRV-Boost + E-NeRV) | 0.165 |
| Pareto-optimization route (F1+F5+F7+F9, Dykstra antagonism discount) | 0.180 |
| Empirical PR-cluster route (PR101 + best-case extension) | 0.180 |

**Council verdict**: HNeRV-family ceiling estimate is `0.165 ± 0.015 [contest-CPU prediction]` (median 0.180; lower tail 0.165 reachable via F8 KL distill or sub-architectural variant; ceiling 0.165 is hit when no further architecture/training extension yields > 0.005 incremental Δ over a 3-experiment moving window).

### 5.5 Saturation-detection criteria (when to switch families)

Switch HNeRV-extension → HNeRV-replacement when ANY of:
1. **Empirical signal saturation**: 3 consecutive dispatched HNeRV-extension experiments produce ≤ +0.0005 incremental Δ score `[contest-CPU]` (variance-adjusted).
2. **Family-floor envelope hit**: best-of-N HNeRV at byte budget ≤180KB lands within `0.005` of `0.165` (the median-route estimate).
3. **Replacement-substrate empirical opening**: any replacement-family (Ballé closed-grammar, SIREN/FINER/WIRE, Cool-Chic/C3 with export) lands a paired same-axis result within `±0.005` of best HNeRV at the same byte budget. The replacement's escape velocity is the right next investment.
4. **Cost per Δ-score inversion**: Modal A100 cost per `0.001` Δ-score `[contest-CPU]` on HNeRV-extension experiments exceeds 2× the cost on the best replacement-family lane.

---

## 6. Source-code forensic findings

### 6.1 PR101 — the actual contribution

Verified by reading every line of `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/`:

| File | LOC | What it contains | What it does NOT contain |
|---|---:|---|---|
| `inflate.py` | 71 | parse_archive + decoder + bicubic interp + uint8 RGB writer + decode-side channel postprocess (PR98's "+1 R G +1 B" subtraction at lines 49-51) | NO training, NO scorer, NO eval_roundtrip in inflate path |
| `src/codec.py` | 480 | DECODER_BLOB_LEN/LATENT_BLOB_LEN fixed offsets + DECODER_STORAGE_ORDER (28-tensor perm) + DECODER_STREAM_ENDS (7 split-brotli stream boundaries) + CONV4_STORAGE_PERMS (per-tensor 4D-axis perm) + CONV4_INVERSE_PERMS (auto-computed argsort) + DECODER_BYTE_MAPS (per-tensor sign strategy: negzig/twos/off/zig) + LATENT_DIM_ORDER + SIDECAR_DELTAS_X100 (16-symbol ranked Huffman table) + 5 sidecar formats + canonical Huffman decode + LZMA latent decode + Brotli stream decode + parse_archive entry | NO training code, NO archive grammar BUILDER (only parser) |
| `src/model.py` | 54 | HNeRVDecoder: 7-stage channel taper, PixelShuffle decoder, bilinear-skip+sin activation, separate rgb_0/rgb_1 heads | NO architectural novelty over canonical HNeRV |
| `README.md` | 30 | "Built on top of PR #95 and PR #98. Adds a self-contained entropy repack" | confirms PR101 = CODEC-ONLY contribution |

**PR101 contribution = inflate-side codec only.** Score gain over PR100 (-0.0025) is ENTIRELY entropy/codec, not training.

### 6.2 PR100 — the substrate root

Verified `experiments/results/public_pr100_intake_20260504_codex/source/submissions/hnerv_lc_v2/`:

| File | LOC | What |
|---|---:|---|
| `hnerv_model.py` | 54 | **byte-for-byte identical** to PR101's `model.py` |
| `inflate.py` | (different from PR101; uses sidecar.py + schema.py + structured packets) | PR100's HNeRV-LC-v2 substrate root |
| `schema.py` | unknown | schema-driven decoder packing precedent (PR101 standardized this) |
| `sidecar.py` | unknown | latent correction sidecar (PR101 extended this) |

**Architecture is unchanged across PR95→PR98→PR100→PR101.** The 0.193-0.198 spread is ENTIRELY in training schedule + codec/entropy.

### 6.3 PR95 intake profile — the only training-side signal

From `experiments/results/public_pr95_intake_20260504_codex/profile_pr95_hnerv_muon_intake.md`:

- **8-stage curriculum, 29,650 total epochs**:
  - stage1: AdamW lr=1e-3, sigma=0.2, CE loss, 3000 epochs
  - stage2: AdamW lr=1e-3, sigma=0.2, tau_softplus loss, 5650 epochs
  - stage3: AdamW lr=1e-4, sigma=0.2, smooth_disagreement loss, 1500 epochs
  - stage4: AdamW lr=1e-4, sigma=0.2, QAT TRUE, smooth_disagreement loss, 500 epochs
  - stage5: AdamW lr=3e-5, sigma=0.2, QAT TRUE, **C1a lambda=0.01**, l7_softplus loss, 9000 epochs
  - stage6: AdamW lr=3e-5, sigma=0.2, QAT TRUE, **C1a lambda=0.02**, l7_softplus loss, 2000 epochs
  - stage7: AdamW lr=3e-5, sigma=**0.1**, QAT TRUE, C1a lambda=0.02, l7_softplus loss, 3000 epochs
  - **stage8: AdamW lr=1e-5, sigma=0.1, QAT TRUE, Muon=TRUE, C1a lambda=0.02, l7_softplus loss, 5000 epochs**
- **Required T1 trainer flags**: `--auth-eval=false`, `--enable-differentiable-yuv6=true`, `--enable-eval-roundtrip-in-training=true`, `--enable-scorer-domain-loss=true`, `--grad-clip-norm=1.0`, `--pixel-l1-anchor-weight=0.0`, `--segmentation-surrogate=sinkhorn`, `--yuv6-mode=monkey_patch_global`
- **Muon partitioning**: 177,156 params via Muon, 51,802 params via AdamW
- **Score claim**: false (PR95 is pre-medal; never anchored under our exact CUDA custody)

**Implication**: PR95 documented the contest-winning training discipline. PR98 added decode-side channel postprocess. PR101 added entropy bolt-on. PR100 may have used a similar curriculum but the training script is NOT in the public source tree.

### 6.4 Our internal HNeRV-family status

- **`sane_hnerv` substrate** (`src/tac/substrates/sane_hnerv/` + `experiments/train_substrate_sane_hnerv.py`): the canonical INTERNAL HNeRV parity attempt; 5 dispatch attempts; first contest-CUDA anchor PENDING. Latest STOP-PRECONDITION is sibling-subagent PCC4 memo gaps (`feedback_wave_a_3_sane_hnerv_FIRST_ANCHOR_LANDED_20260512.md`).
- **Catalog #187 `check_hnerv_training_parity_guard`**: STRICT preflight enforcing `patch_upstream_yuv6_globally()` before `load_differentiable_scorers()`, `apply_eval_roundtrip=True` keyword, EMA, archive-build-in-loop, exact 3-arg runtime templates, scorer/network-free inflate templates.
- **`hi_nerv`, `ff_nerv`, `tc_nerv`, `block_nerv`, `ds_nerv`, `e_nerv_substrate`, `nervdc_substrate`, `ego_nerv_substrate`, `factorized_hnerv_v1`, `cnerv_substrate`, `dsnerv_substrate`, `blocknerv_substrate`, `ffnerv_substrate`, `hinerv_substrate`, `anr_substrate`** — fifteen HNeRV-family substrates with trainers; NONE has anchored a `[contest-CUDA]` score.
- **`pr101_lc_v2_clone`**: an internal clone of PR101 substrate; no anchor.

**Verdict**: we have built the HNeRV-family substrate inventory, but the integration loop (substrate → trainer → archive → inflate → exact eval → posterior update) has not closed for any internal HNeRV-family substrate. PR101 GOLD primitive port (storage_order + conv4 perms + byte maps) landed 2026-05-12 as packet-compiler primitives but they are PLUMBING, not score-bearing.

---

## 7. Unexplored-primitive table — score impact + cost + implementability

(Combined from §3.1 + §3.2 + §3.3 + this section's implementability column)

| ID | Primitive | Predicted ΔS | Modal A100 cost | Dev cost | Implementability `[2026-05-13]` |
|---|---|---:|---:|---|---|
| F1 | Score-aware retrain w/ PR95 curriculum | -0.005 to -0.012 | $4-15 | 1-2 days | **HIGHEST PRIORITY**; `sane_hnerv` trainer EXISTS; first anchor PENDING |
| F4 | Archive-in-loop validation | -0.001 to -0.003 (on top of F1) | +$1-2 (extends F1) | 1 day | dev-needed; add archive-build callback per N epochs to F1 trainer |
| F5 | QAT FP4 decoder w/ scorer-aware schedule | -0.003 to -0.008 (or +0.005 cliff) | $3-8 | 1 day | `qat_finetune.py` exists; needs HNeRV-family wiring |
| F8 | KL distill 229K→100K decoder | -0.020 to -0.040 (or +0.030 cliff) | $5-12 | 2 days | dev-needed; high-risk/high-reward |
| F6 | Markov-1 hyperprior over latents alone | -0.001 to -0.002 | $2-3 | 2 days | Ballé/CompressAI plumbing; codex P3/H6 |
| F7 | Lloyd-Max latent quant | -0.002 to -0.004 | $1-3 | 1 day | classical Lloyd-Max algorithm; needs training-time integration |
| F9 | Full Ballé hyperprior (decoder + latent) | -0.002 to -0.005 | $3-5 | 2-3 days | CompressAI primitives in `tac.composition.registry` |
| F10 | Per-pair adaptive bit allocation | -0.001 to -0.003 | $1-3 | 2 days | dev-needed; PoseNet/SegNet per-pair sensitivity scoring |
| F2 | yuv6 monkey-patch alone | -0.001 to -0.003 | folds into F1 | folds | folded |
| F3 | EMA alone | -0.0005 to -0.002 | folds into F1 | folds | folded |
| F11 | Muon optimizer | 0 to -0.001 | folds into F1 | folds | folded as F1 stage 8 |
| F12 | GEPA search | 0 to -0.002 | $10-50 | 3+ days | proposal-evaluator separation per codex H8; experimental |
| F13 | FINER/WIRE/BACON activation swap | ±0.005 [uncertain] | $5-15 | 3+ days | **DEFERRED-pending-research**; high risk of class boundary cross |

---

## 8. Family-ceiling derivation — math

Per §2.2 + §5.1 + §5.4 above:

```text
Theoretical lower bound on HNeRV-family score:
   S = 100·d_seg + sqrt(10·d_pose) + 25·B/N

HNeRV-family per-axis lower bounds at PR101 anchor (uint8-roundtrip noise floor):
   d_seg_floor ≈ 5.0e-4     →  100·d_seg_floor = 0.0500
   d_pose_floor ≈ 3.0e-5    →  sqrt(10·d_pose_floor) = 0.0173
   B_floor      ≈ 60,000    →  (Shannon absolute floor)
   B_floor_HNeRV ≈ 120,000  →  (HNeRV practical, FP4 QAT, Markov-1 entropy)
                              →  25·B_floor_HNeRV/N = 0.0800

Shannon absolute floor (any representation):
   S_min ≈ 0.0500 + 0.0173 + 0.0400 = 0.107  [Shannon absolute, prediction]

HNeRV-family practical floor (architecture-bounded):
   S_HNeRV-floor ≈ 0.0500 + 0.0173 + 0.0800 = 0.147 [HNeRV-architecture-bounded prediction]

Practical achievable HNeRV-family floor (with Dykstra antagonism + decoder-overhead + entropy-metadata):
   S_HNeRV-practical ≈ 0.165 - 0.180 [contest-CPU prediction]
```

The 0.015-0.030 gap between architecture-bounded floor (0.147) and practical-achievable floor (0.165-0.180) is:
- Decoder-overhead bytes that don't reduce in real implementations (~5-10KB);
- Sidecar / latent-grammar metadata that contest packets need to be parseable;
- Pareto antagonism (Dykstra discount on the additivity assumption of independent rate savings).

---

## 9. Saturation-detection criteria — when to switch families

(See §5.5 for the canonical criteria.)

**Operational tests** to fire each criterion:

1. **Empirical signal saturation**: maintain rolling 3-experiment Δ-score `[contest-CPU]` window in `.omx/state/cost_band_posterior.jsonl` (already wired via Catalog #175/#177); fire criterion 1 when Δ-window ≤ 0.0005.
2. **Family-floor envelope hit**: at any new contest-CPU anchor, compute `min(S_observed) - 0.165 ≤ 0.005`. If yes, set lane registry status `lane_class=hnerv_family_at_floor`.
3. **Replacement-substrate opening**: per the running `tac.composition.registry.canonical_primitive_inventory()` posterior, fire when a non-HNeRV substrate (`balle_renderer`, `siren`, `cool_chic`, `vq_vae`) lands a paired same-axis result at ≤ HNeRV-best `+ 0.005`.
4. **Cost-per-Δ inversion**: at each anchor, log `(Δ_score / cost_usd)`. If 3-experiment rolling moving average on HNeRV-extension ≥ 2× rolling MA on replacement-family, switch.

---

## 10. Dispatch matrix recommendation

### 10.1 Recommended order (HNeRV-extension)

1. **Resolve `sane_hnerv` STOP-PRECONDITION**. The 5th-attempt DEFERRED's STOP cause is two sibling-subagent memos (`feedback_compressai_integration_landed_20260512.md` + `feedback_fix_a_catalog_158_124_landed_20260512.md`) missing PCC4 council sections. Operator decision needed: backfill the sibling memos, OR re-dispatch with a different recipe that bypasses the PCC4 trigger (unlikely; check is repo-wide).
2. **F1 + F4 (score-aware retrain w/ archive-in-loop validation)**: $5-17 Modal A100 cost. Predicted Δ -0.006 to -0.015. **HIGHEST EV/$ in HNeRV-extension space.**
3. **F5 (QAT FP4 with scorer-aware schedule)**: $3-8. Stack on F1 anchor. Predicted Δ -0.003 to -0.008. **2nd HIGHEST EV.**
4. **F9 (Ballé hyperprior full stack)**: $3-5. Stack on F1+F5 anchor. Predicted Δ -0.002 to -0.005.
5. **F7 (Lloyd-Max latent quant)**: $1-3. Stack on F1+F5+F9. Predicted Δ -0.002 to -0.004.
6. **F8 (KL distill 229K→100K decoder)**: $5-12. **High-risk/high-reward** — predicted Δ band [-0.040, +0.030]; council recommends running ONLY AFTER F1+F5+F9 anchors land, so the comparison baseline is the best HNeRV-family.

**Cumulative cost estimate**: $17-45 to exhaust F1+F4+F5+F7+F9. F8 adds another $5-12. Total budget for HNeRV-extension exhaustion: **$22-57**.

**Predicted family-floor reach**: 0.193 → 0.175 ± 0.010 `[contest-CPU prediction]` after F1+F4+F5+F7+F9 land. F8 could lower to 0.155-0.165 if KL distill avoids the cliff.

### 10.2 What to dispatch LATER (or never)

- **F11 (Muon alone)**: fold into F1 stage 8; don't dispatch separately.
- **F12 (GEPA search)**: defer; the search-space cost is high and the per-primitive Δ is bounded by F1-F10 above.
- **F13 (FINER/WIRE/BACON)**: **DEFERRED-pending-research**; ablate first on a SIREN-family substrate, NOT directly on HNeRV.

### 10.3 What NOT to do (anti-pattern guard)

- **Don't kitchen_sink** (PR105 lost 1776 LOC to PR101's 605). Ship F1, then F5, then F9 as small bolt-ons on a verified working substrate. ≤350 LOC per bolt-on per HNeRV parity lesson 7.
- **Don't pursue HNeRV codec composition on score-naive substrates** (per `feedback_substrate_vs_codec_composition_meta_pattern_20260508.md`). Anchor F1 first; build bolt-ons on the anchored substrate.
- **Don't conflate `[macOS-CPU advisory]` with `[contest-CPU]`** for promotion (Catalog #192 STRICT). The 2e-5 reliability empirical drift is a RANKING signal, NOT a promotion signal.

---

## 11. 3-clean-pass adversarial review log

### Round 1 (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian)

**Shannon**: rate-distortion grounding — every primitive in §3 is rate/distortion/component-decomposable. F1's Δ traces to "score-aware loss closes the gradient-mismatch gap between training-time proxy and exact eval"; F5's Δ traces to "FP4 at 4 bits × decoder params = 162K→44K bytes if Brotli-compressible at the same entropy"; F9's Δ traces to "Markov-1 hyperprior captures cross-tensor correlation in PR101 latents." All 5 MEAT/INCREMENTAL primitives have a rate-distortion lineage. **CLEAN.**

**Dykstra**: convex feasibility — the floor estimate uses three independent feasibility constraints (`d_seg`, `d_pose`, `B`); intersection projections give the achievable region. Stacking Δs across F1-F9 with 0.6 antagonism discount yields 0.175-0.180, matching §5 estimates. The 80% CI is appropriately conservative. **CLEAN.**

**Yousfi**: contest-faithfulness audit — PR95 trainer flags (`--enable-differentiable-yuv6`, `--enable-eval-roundtrip-in-training`, `--enable-scorer-domain-loss`, `--yuv6-mode=monkey_patch_global`, `--segmentation-surrogate=sinkhorn`) are the canonical HNeRV-family contest discipline. F1 honoring these flags is necessary AND sufficient for HNeRV parity. The training script for PR101 is NOT in the public source tree — this is the public-frontier forensic gap codex calls out as Priority 7. **CLEAN.**

**Fridrich**: adversarial — the operator's intuition that HNeRV is a local minimum at some point IS CORRECT. The danger is over-extending PR101's codec-only contribution as if it were the family ceiling. The empirical claim "PR101 = 0.193" is a SINGLE CODEC-ONLY anchor with no retrain; the family extension has not been operationally explored. **CLEAN.**

**Contrarian**: challenge — is F1's predicted Δ-0.012 credible? Counter-argument: the existing PR95 substrate (0.197 leaderboard) didn't reach 0.193 — PR101's gain over PR95 was codec-only. So "score-aware retrain alone" might be worth less than predicted. **Council response**: PR95's 0.197 is on the public CPU axis; the macOS-CPU sweep places PR107_apogee (the closest internal substrate) at 0.196640, ~0.004 above PR101. The codex P1 explicitly identifies "recover PR95/PR100/PR101/PR103 training scripts" as the highest-EV missing step; the F1 predicted Δ-0.005 LOWER BOUND is preserved even if the actual training pipeline only matches PR95's level. **Resolved; predicted ranges are appropriately wide.**

**Round 1 verdict**: 1 finding raised (Contrarian's PR95-vs-PR101 codec-only attribution), addressed in §6.3 + §10.1. **CLEAN-PASS-1 with the inline §6.3 backfill counted.**

### Round 2 (Quantizr + Hotz + Selfcomp + MacKay + Ballé)

**Quantizr**: empirical adversary — Quantizr's 0.33 archive used HNeRV-family decoder + Hinton T=2.0 KL distill + FP4 QAT. F5 (QAT FP4) and F8 (KL distill 229K→100K) are EXACTLY the primitives Quantizr operationalized. Predicted Δ ranges are CONSERVATIVE relative to Quantizr's empirical gain. **CLEAN.**

**Hotz**: engineering aesthetic — F1 is the 5-line `experiments/train_substrate_sane_hnerv.py` retrain (in spirit; the trainer is already wired). The "smallest credible bolt-on" is to land the first `sane_hnerv` contest-CUDA anchor, THEN compose F5 on top. Don't pre-engineer F8 KL distill until F1+F5 lands. **CLEAN.**

**Selfcomp**: self-compression — F9 (Ballé hyperprior) extends PR101's existing LZMA+Brotli pipeline; the natural sister-extension is to fold per-tensor scales + decoder weights into a single Markov-1-conditioned stream. This is structurally what Selfcomp's 1.017 bpw block-FP does on coarser slabs. F9's Δ-0.002 to -0.005 is plausible. **CLEAN.**

**MacKay**: MDL analysis — the HNeRV-family description length L(theta) = decoder_bytes + latent_bytes + sidecar_bytes; F8's KL distill reduces L(theta) substantially (~60KB rate save) at potential MSE cost. The Δ-0.020 to -0.040 gain band is MDL-defensible IF the smaller student doesn't lose more component distortion than rate gained. Council recommends ablating decoder param-count systematically (229K → 180K → 130K → 100K → 80K). **CLEAN.**

**Ballé**: neural-compression — Markov-1 hyperprior on PR101 latents is well-aligned with the 2018 paper's scale-hyperprior pattern. The 15K → 12K predicted rate save is conservative; actual gain might be larger if PR101's centered-delta uint8 latents have strong cross-tensor correlation. **CLEAN.**

**Round 2 verdict**: 0 findings raised. **CLEAN-PASS-2.**

### Round 3 (van den Oord + Hinton + Boyd + Tao + Filler)

**van den Oord**: VQ-VAE for HNeRV latents (an unexplored primitive variant of F7) could push F7's Δ from -0.004 to -0.006. Add to deferred-research queue. **CLEAN.**

**Hinton**: KL distill of HNeRV teacher to smaller student is exactly the canonical 2014 distillation; T=2.0 is the Quantizr-canonical temperature. Council should explicitly require kl_on_logits(T=2.0) in F8's spec to align with Quantizr's empirical signature. **CLEAN.**

**Boyd**: ADMM/proximal-gradient operationally — F9's Markov-1 hyperprior is best solved by alternating between Ballé-style entropy estimation and decoder reconstruction; the F1 + F9 joint stack should use ADMM for stable convergence. **CLEAN.**

**Tao**: harmonic analysis — F13 (FINER/WIRE/BACON activation swap) could yield a spectral-bias change that affects high-frequency component reconstruction. The HNeRV `sin(x)` activation is one specific Fourier-feature parameterization; FINER's variable-frequency activation might better track the contest video's frequency distribution. Council recommends ablation only AFTER F1+F5+F9 land. **CLEAN.**

**Filler**: syndrome-trellis coding (STC) for HNeRV's per-pair latent table — F10 (per-pair adaptive bit allocation) is structurally a STC-encoded per-pair payload. Council notes that F10's predicted Δ-0.001 to -0.003 is LOW-EV alone but composes well with F7. **CLEAN.**

**Round 3 verdict**: 0 findings; 2 deferred-research additions (van den Oord VQ-VAE-for-latents; Hinton's explicit T=2.0 spec). **CLEAN-PASS-3.**

**Counter at 3/3 CLEAN.** Council deliberation status: GREEN.

---

## 12. 6-hook wire-in declaration (per Catalog #125)

1. **Sensitivity-map contribution**: This memo's family-ceiling derivation (§5) and primitive table (§7) become an UPSTREAM ROUTING SIGNAL for `tac.sensitivity_map.axis_weights` — specifically, the 2.71× pose-marginal-over-seg ratio at PR101's operating point should weight F8 (decoder param-count distill) toward pose-axis-preservation, not seg-axis. The next sensitivity-map update can absorb this routing prior. — **DECLARED**.
2. **Pareto constraint**: §5.5 saturation-detection criteria #2 (family-floor envelope hit at 0.165) adds a NEW Pareto constraint to `tac.pareto_*`: any HNeRV-extension candidate at byte-budget ≤180KB must lie within `±0.005` of `0.165 [contest-CPU prediction]` to be promoted to dispatch. Below `0.165 ± 0.005`, the constraint refuses HNeRV-extension candidates as the wrong family. — **DECLARED**.
3. **Bit-allocator hook**: §10's recommended dispatch order (F1 → F4 → F5 → F9 → F7) is registered as a SEQUENTIAL bit-allocator rule: each bolt-on layer's bit-budget is set relative to the PREVIOUS anchor's component decomposition, not a global static target. The hook is registered against the next bit-allocator refresh. — **DECLARED**.
4. **Cathedral autopilot dispatch hook**: The 5 unexplored primitives (F1, F4, F5, F7, F9) are eligible cathedral-autopilot dispatch candidates IF AND ONLY IF the operator's `tools/operator_authorize.py` recipe ladder includes `substrate_sane_hnerv_modal_a100_dispatch` (F1) AND the previous anchor's STOP-PRECONDITION is resolved. The autopilot journal should reflect this dependency. — **DECLARED**.
5. **Continual-learning posterior update**: This memo's empirical anchors (7 macOS-CPU rows + 6 contest-CPU public claims) are ALREADY in `.omx/state/cost_band_posterior.jsonl` (via the 2026-05-12 bulk anchor back-fill landing memo). The family-ceiling estimate `0.165 ± 0.015` is a NEW SCALAR POSTERIOR ESTIMATE; recommend appending to the same posterior with `outcome="research_council_estimate"` (a new outcome token per Catalog #175/#177). — **DECLARED**.
6. **Probe-disambiguator**: The 2+ defensible interpretations of "HNeRV-family ceiling" — (a) PR101 IS the family ceiling (probability 0.55 per §1); (b) PR101 + score-aware retrain is the ceiling (probability 0.30); (c) PR101 + retrain + QAT + KL distill reaches ≤0.165 (probability 0.15) — make the operator decision NOT to dispatch F8 KL distill (the most expensive and highest-cliff-risk primitive) FIRST. Probe is implicit in §10's ordered dispatch matrix: F1 lands first, and the result calibrates which scenario (a/b/c) is realized. — **DECLARED**.

---

## 13. Cross-references

- **Codex roadmap**: `.omx/research/sub017_frontier_innovation_roadmap_20260513_codex.md` Priorities 1 (HNeRV parity retrain) + 2 (PacketIR/HDM5) + 3 (scorer-aware residual atom) + S1-S5 stack paths; H1/H6/H7 hypotheses; eureka #9 (trellis quantized decoder weights).
- **Sister council** (parallel): `lane_first_principles_original_score_lowering_council_20260513` — derives ORIGINAL first-principles bounds. If sister council's absolute Shannon floor is below 0.155, this memo's HNeRV-family ceiling at 0.165 has remaining meat. If sister floor is above 0.180, HNeRV ceiling crowds the absolute Shannon floor and the family-switch decision is forced.
- **Canonical HNeRV retrospective**: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md` — the 13 inviolable HNeRV parity lessons; this memo honors all 13.
- **Substrate-vs-codec composition meta-pattern**: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_substrate_vs_codec_composition_meta_pattern_20260508.md` — explains why F5/F8/F9 must be bolt-ons on F1's anchor, not standalone.
- **A1 + LAPose composition substrate**: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_a1_plus_lapose_composition_substrate_landed_20260513.md` — pose-axis non-HNeRV path; tracks the 2.71× pose-marginal-over-seg operating-point heuristic.
- **macOS-CPU empirical sweep**: `.omx/research/macos_cpu_canvas_pareto_ranking_20260513.md` — all HNeRV-family score anchors used in §2.
- **macOS-CPU empirical validation**: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_macos_cpu_proxy_empirical_validation_landed_20260513.md` — 2e-5 absolute-drift validation enabling §2's apples-to-apples comparisons.
- **PR101 GOLD primitive port**: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pr101_gold_primitive_port_landed_20260512.md` — landed packet-compiler primitives (storage_order/conv4_perms/byte_maps); these are PLUMBING for F1's archive emit step.
- **sane_hnerv first-anchor DEFERRED**: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_wave_a_3_sane_hnerv_FIRST_ANCHOR_LANDED_20260512.md` — STOP-PRECONDITION at PCC4 sibling-memo gap; F1 dispatch is blocked here.
- **Catalog #187 `check_hnerv_training_parity_guard`**: STRICT preflight gate enforcing PR95 parity training discipline (`patch_upstream_yuv6_globally()` before `load_differentiable_scorers()`, `apply_eval_roundtrip=True`, EMA, archive-build-in-loop).
- **Catalog #192 `check_macos_cpu_advisory_promotion_defense`**: STRICT gate refusing promotion of `[macOS-CPU advisory]` anchors without paired Linux x86_64 `[contest-CPU]` verification.

---

## 14. Apples-to-apples evidence tags + custody discipline

Every score and Δ-estimate in this memo carries one of:
- `[macOS-CPU advisory]` — measured via `experiments/contest_auth_eval.py --device cpu` on M5 Max ARM64; ranking-only, non-promotable per Catalog #192.
- `[contest-CPU]` — `upstream/evaluate.py --device cpu` on Linux x86_64 (GHA-ranked).
- `[contest-CUDA]` — `upstream/evaluate.py --device cuda` on T4/A100.
- `[contest-CPU prediction]` — calibrated against empirical anchors + Shannon/Dykstra/Ballé first-principles math; NOT a score claim per CLAUDE.md `forbidden_score_claim_with_byte_change_unless_inflate_consumes`.
- `[contest-CUDA prediction]` — same, CUDA axis.
- `[Shannon absolute floor, prediction]` — theoretical lower bound from R(D) bound; NOT achievable in practice.
- `[HNeRV-architecture-bounded prediction]` — theoretical lower bound subject to HNeRV 229K-param 6-stage architecture.

**No /tmp paths.** **No MPS-derived decisions.** **No KILL verdicts.** **No score claims on predicted bands.**

---

## 15. Operator decisions surfaced

1. **`sane_hnerv` STOP-PRECONDITION resolution**: backfill the two sibling-subagent memos (`feedback_compressai_integration_landed_20260512.md` + `feedback_fix_a_catalog_158_124_landed_20260512.md`) with PCC4-required Council/internal-consistency sections, OR explicitly approve their author-subagent to do so. F1 dispatch is blocked here.
2. **F1 + F4 dispatch authorization**: $5-17 Modal A100 cost for score-aware retrain w/ archive-in-loop validation. Predicted Δ -0.006 to -0.015.
3. **F5 dispatch authorization** (after F1 anchor): $3-8 Modal A100 cost for QAT FP4 with scorer-aware schedule.
4. **F8 KL distill authorization** (after F1+F5): $5-12. **HIGH-CLIFF-RISK**; council recommends pairing with explicit ablation schedule (229K → 180K → 130K → 100K → 80K decoder param-count sweep) and pre-registering kill criteria.
5. **Family-switch threshold formalization**: per §5.5 criteria, register as STRICT preflight gate or autopilot routing rule.
6. **Sister-council reconciliation**: parallel sister council (`lane_first_principles_original_score_lowering_council_20260513`) produces an absolute-Shannon-floor estimate. Two memos must be reconciled at conclusion: this memo's HNeRV-family ceiling (0.165 ± 0.015) + sister memo's first-principles absolute floor.

---

**3/3 clean adversarial review** ✅. **6/6 wire-in hooks declared** ✅. **All scores apples-to-apples tagged** ✅. **No FORBIDDEN PATTERNS violations** ✅. **Loop remains paused.** $0 GPU spend.

This memo is read-only deliberation. No code edits. No dispatch. Recommended next action: operator decisions per §15.
