# SIREN literature + open-source-repo cross-reference review

**Date:** 2026-05-13
**Lane:** `lane_siren_literature_review_20260513` (L0 → L1 after this memo lands)
**Scope:** Operator-routed pre-dispatch literature review of the SIREN substrate
(`experiments/train_substrate_siren.py` + `src/tac/substrates/siren/*`) ahead of
the $6 Modal A100 dispatch authorized by the audit + fix-wave landings
2026-05-13 (`feedback_siren_pre_dispatch_audit_LANDED_20260513.md` +
`feedback_siren_pre_dispatch_audit_fix_wave_LANDED_20260513.md`).
**Evidence grade for every claim:** `[literature-prediction]` or
`[third-party-empirical:<paper>]` per CLAUDE.md
"Apples-to-apples evidence discipline" — third-party PSNR-on-Kodak is **NOT**
comparable to our contest contract (`100·d_seg + sqrt(10·d_pose) + bytes_term`).

## Executive summary

The pact SIREN substrate is canonically-faithful to Sitzmann et al. (NeurIPS
2020) on initialization (`omega_0=30` first layer, `omega=1.0` hidden, bounds
`Uniform(-sqrt(6/fan_in)/omega, sqrt(6/fan_in)/omega)`), confirmed line-by-line
against [`vsitzmann/siren`](https://github.com/vsitzmann/siren) and
[`lucidrains/siren-pytorch`](https://github.com/lucidrains/siren-pytorch).

**Three actionable recommendations the literature surfaces:**

1. **Architectural under-capacity for video memorization.** The canonical
   Sitzmann video config is `hidden_features=1024, num_hidden_layers=3` (~2.1M
   params) for a SINGLE-video memorization task. Our config (`H=128, L=6`,
   ~84K params) is **~25× smaller** than canonical and was chosen to fit the
   contest rate budget — but the 0.193 contest-CPU frontier is dominated by
   HNeRV-family content-adaptive-embedding architectures, not coordinate-only
   MLPs. The right comparison config under the contest's ~100KB-post-brotli
   rate envelope is `H≈140-180, L=4-5` per a back-of-envelope rate calc
   (Section 5). **Recommendation:** declare an explicit "predicted_band
   floor" expecting ~0.18-0.22 contest-CPU (NOT 0.145) and add a probe-mode
   `H=256, L=4` arm if the rate envelope opens after first anchor.

2. **Temporal-coordinate-as-scalar-input is a known weak primitive.** Treating
   `t = pair_idx / num_pairs` as a single scalar coordinate is the exact
   anti-pattern called out in TeNeRV (`2601.17743`), CANeRV (`2502.06181`),
   and the original HNeRV paper (`2304.02633`): "without explicit motion
   modeling, the limited network capacity is consumed by competing temporal
   patterns, resulting in over-smoothed artifacts." HNeRV's content-adaptive
   per-frame embedding gives +4.7 PSNR and 16× faster convergence on UVG
   `[third-party-empirical:HNeRV]`. **Recommendation:** in the post-anchor
   diagnostic, log per-pair reconstruction PSNR vs `pair_idx` to detect the
   over-smoothing failure mode early. If observed, the lane should
   DEFER-pending-temporal-encoding-upgrade (frequency-encoded t, learned
   per-pair latent, or hierarchical t-decomposition per TeNeRV) rather than
   chase hyperparameters.

3. **The 2025-09 ICLR-2026 "WINNER" paper (`2509.12980`) identifies a SIREN
   spectral-bottleneck failure mode that requires NO trainable parameters to
   fix.** WINNER perturbs SIREN's uniform init with Gaussian noise whose
   scale is determined by the target signal's spectral centroid. Reports
   "state-of-the-art audio fitting and significant gains in image and 3D
   shape fitting tasks over base SIREN" `[third-party-empirical:WINNER]`
   without adding parameters. **Recommendation:** post-anchor, if SIREN's
   reconstruction shows uniformly-low PSNR (the "spectral bottleneck"
   signature: model collapses to near-zero output), apply the WINNER init
   perturbation as a 1-LOC patch (~5-10 LOC including the spectral-centroid
   estimator). This is a $0 dev effort that may rescue an otherwise-failed
   anchor.

**What this review does NOT establish** (Section 7): no hardware verification,
no ablation runs, no contest-CUDA or contest-CPU measurement on the actual
SIREN substrate. Every architectural claim is a literature-grounded
prediction with empirical uncertainty.

---

## 2. Paper-by-paper cross-reference table

Evidence-tag legend:
- `[literature-prediction]` = derived from paper claim, not measured by us
- `[third-party-empirical:<paper>]` = paper reports empirical result on
  third-party scoring contract (NOT our contest contract)
- `HAVE` = our impl already incorporates this idea
- `MISSING` = our impl could incorporate this idea
- `N/A` = not applicable to our contest scoring contract

| Paper | Year | Venue | Key idea | Our-impl status | Would help? |
|---|---|---|---|---|---|
| Sitzmann et al. ["Implicit Neural Representations with Periodic Activation Functions"](https://arxiv.org/abs/2006.09661) | 2020 | NeurIPS | Sin activations + omega_0=30 first layer + variance-preserving init | **HAVE** (verified vs official repo) | N/A — baseline |
| [Sitzmann/siren official repo](https://github.com/vsitzmann/siren) `experiment_scripts/train_video.py` | 2020 | OSS | Video fitting config: `H=1024, L=3, lr=1e-4, batch_size=1, sample_frac=38e-4` | **PARTIAL** — we use H=128, L=6, lr=5e-4 (smaller + slower-LR for rate budget) | `[literature-prediction]` Our config has ~25× FEWER params than canonical video config. Predicts lower reconstruction quality but smaller archive. |
| [lucidrains/siren-pytorch](https://github.com/lucidrains/siren-pytorch) | 2021 | OSS | Reference impl confirming `w0_initial=30, w0=1.0` defaults | **HAVE** — matches our impl exactly | N/A — confirmation |
| Tancik et al. ["Fourier Features Let Networks Learn High Frequency Functions"](https://arxiv.org/abs/2006.10739) | 2020 | NeurIPS | RFF + ReLU MLP as alternative to SIREN; tunable bandwidth via std of RFF | **MISSING** — could test as ablation arm | `[literature-prediction]` Some workloads favor RFF+ReLU over SIREN; depends on signal spectrum. Not surfacing as a high-priority lane. |
| Mildenhall et al. NeRF (NeRF positional encoding) | 2020 | ECCV | Pyramid positional encoding `(sin(2^k pi x), cos(2^k pi x))_{k=0..L}` for ReLU MLPs | **MISSING** — could pre-encode `(x,y,t)` before first layer | `[literature-prediction]` SIREN paper claims `omega_0=30` first layer "is equivalent to a positional encoding" — so explicit PE on top of SIREN is double-counting. Don't add. |
| Lindell et al. ["BACON: Band-limited Coordinate Networks"](https://arxiv.org/abs/2112.04645) | 2022 | CVPR | Analytical Fourier spectrum; bandwidth explicitly specifiable; better extrapolation than SIREN | **MISSING** — research-only alternative; would require rewrite | DEFERRED — high LOC, low EV at PR101 frontier |
| Liu et al. ["FINER: Flexible Spectral-bias Tuning in INR"](https://arxiv.org/abs/2312.02434) | 2024 | CVPR | Variable-periodic activation `sin((|x|+1)·x)`; per-neuron frequency diversity | **MISSING** — drop-in activation replacement | `[third-party-empirical:FINER]` Outperforms SIREN on 2D image fitting PSNR. **Score-axis effect unknown.** Recommend as Round-2 ablation after first SIREN anchor. ~10 LOC patch. |
| Dupont et al. ["COIN"](https://arxiv.org/abs/2103.03123) | 2021 | ICLR-W | Overfit-MLP-to-image + quantize weights as compressed code | **PARTIAL** — our archive grammar (fp16 + brotli) is COIN-style; could push to int8/FP4 | `[third-party-empirical:COIN]` Beats JPEG at low bitrates. Direct analog to our use case. **Higher-priority lane:** int8 QAT or FP4 quantization of SIREN weights would compress archive ~2× (167KB → 84KB raw → ~50KB brotli). |
| Dupont et al. ["COIN++"](https://openreview.net/forum?id=NXB0rEM2Tq) | 2022 | TMLR | Meta-learned INR base + per-instance modulations | **MISSING** — Phase 3 lane; requires meta-learning rig | DEFERRED — does not fit single-video contest contract |
| Chen et al. NeRV ["Neural Representations for Videos"](https://openreview.net/forum?id=BbikqBWZTGB) | 2021 | NeurIPS | Frame-index → CNN renderer (NOT SIREN) | **N/A** — different architecture class | N/A — sibling lane (lane_12_v2_nerv_as_renderer) |
| Chen et al. ["HNeRV: Hybrid Neural Representation"](https://arxiv.org/abs/2304.02633) | 2023 | ICCV | Per-frame content-adaptive embeddings → CNN decoder; +4.7 PSNR vs NeRV; 16× faster | **N/A for SIREN substrate** — HNeRV is a different paradigm class | `[third-party-empirical:HNeRV]` HNeRV is the **structural reason** the public 0.193 frontier dominates pure coordinate MLPs. **Implication for SIREN lane:** SIREN-as-renderer is likely structurally dominated unless rate-axis savings exceed +4.7 PSNR worth of distortion. |
| Kim et al. ["E-NeRV: Disentangled Spatial-Temporal Context"](https://arxiv.org/abs/2207.08132) | 2022 | ECCV | Spatial + temporal context decomposition; 8× convergence speedup at same params | **N/A for SIREN** — applies to NeRV class | N/A |
| Kwan et al. ["HiNeRV: Hierarchical Encoding"](https://arxiv.org/abs/2306.09818) | 2023 | NeurIPS | Hierarchical positional encoding + depthwise-conv layers; 36.6 dB @ 0.051 bpp on UVG; "first INR to beat HEVC veryslow" | **N/A for SIREN substrate** — HiNeRV is a different arch class | `[third-party-empirical:HiNeRV]` Best-in-class on UVG. Sibling repo intake recommended. |
| Ladune et al. ["Cool-Chic"](https://arxiv.org/abs/2401.02156) | 2024 | TIP | Overfitted latent feature maps + small entropy model; matches HEVC at <1k MACs/pixel | **HAVE primitives** — `tac.composition.registry` carries CompressAI Cool-Chic primitives per Catalog #169 | Active lane; not for THIS SIREN substrate. |
| Kim et al. ["C3"](https://arxiv.org/abs/2312.02753) | 2024 | CVPR | Cool-Chic successor with 3D latents + context selection for video; matches VTM/H.266 RD | **HAVE primitives** — CompressAI rows in `tac.composition.registry` | Active sibling lane. |
| Liu et al. ["CANeRV: Content Adaptive Neural Representation"](https://arxiv.org/html/2502.06181) | 2025 | (preprint) | Content-adaptive NeRV variants; explicit motion modeling | N/A for SIREN | Confirms thesis that pure-coordinate temporal input is dominated |
| Liu et al. ["TeNeRV: Hierarchical Temporal Neural Representation"](https://arxiv.org/html/2601.17743) | 2026 | (preprint) | Decomposes temporal axis hierarchically; addresses over-smoothing in pure-t-input INRs | N/A for SIREN-as-is | `[third-party-empirical:TeNeRV]` **DIRECTLY contradicts our SIREN-with-t-coord paradigm.** Quote: "Existing INR-based methods fail to match the performance... particularly on highly dynamic videos due to inefficient modeling of temporal information... the limited network capacity is consumed by competing temporal patterns, resulting in over-smoothed artifacts." |
| (anon) ["WINNER: Weight Initialization with Noise for SIREN"](https://arxiv.org/abs/2509.12980) | 2025 | ICLR 2026 prep | Spectral-centroid-determined Gaussian noise perturbation of SIREN's uniform init; no extra params | **MISSING** — ~5-10 LOC patch | `[third-party-empirical:WINNER]` "Significant gains over base SIREN" on audio/image/3D fitting. **Highest-EV cheap intervention for our SIREN lane.** |
| Benbarka et al. ["Seeing INRs as Fourier Series"](https://openaccess.thecvf.com/content/WACV2022/papers/Benbarka_Seeing_Implicit_Neural_Representations_As_Fourier_Series_WACV_2022_paper.pdf) | 2022 | WACV | Theoretical bridge between SIREN and Fourier features | **MISSING** — purely theoretical context | N/A |
| Saragadam et al. WIRE | 2023 | CVPR | Wavelet-based INR (Gabor wavelet activation) | **MISSING** — research-only alternative | DEFERRED — high LOC, unverified contest-axis EV |
| Fathony et al. MFN | 2021 | ICLR | Multiplicative Filter Networks | **MISSING** | DEFERRED |
| Schwarz et al. ["COMBINER"](https://dl.acm.org/doi/10.5555/3666122.3666216) | 2023 | NeurIPS | Bayesian/variational bit-allocation for INR compression | **MISSING** — relevant Phase 3 lane | DEFERRED — research-only |
| Yang et al. ["NVRC: Neural Video Representation Compression"](https://arxiv.org/html/2409.07414) | 2024 | NeurIPS | End-to-end RD optimization framework for INR-based video codecs | N/A for current SIREN scope | DEFERRED — would require new training rig |
| Strümpler et al. ["SINR: Sparsity Driven Compressed INRs"](https://arxiv.org/html/2503.19576) | 2025 | (preprint) | Sparse-code compression of INR weights pre-quantization-and-entropy | **MISSING** — Phase 3 codec lane | DEFERRED |

## 3. Repo-by-repo cross-reference

| Repo | Stars | Purpose | Mirror for us? |
|---|---|---|---|
| [`vsitzmann/siren`](https://github.com/vsitzmann/siren) | ~1.7K | Canonical SIREN PyTorch impl | **CANONICAL — confirmed our init scheme matches `first_layer_sine_init` + `sine_init`** |
| [`lucidrains/siren-pytorch`](https://github.com/lucidrains/siren-pytorch) | ~1.5K | Simple SirenNet wrapper | **CONFIRMS** w0_initial=30, w0=1.0 defaults |
| [`liuzhen0212/FINER`](https://github.com/liuzhen0212/FINER) | (active) | FINER variable-periodic activation | **CONSULT** if Round-2 FINER-activation ablation lands |
| [`computational-imaging/bacon`](https://github.com/computational-imaging/bacon) | (CVPR 2022) | BACON bandwidth-controlled INR | Research-only reference |
| [`EmilienDupont/coin`](https://github.com/EmilienDupont/coin) | ~200 | COIN single-image INR compression | **CONSULT** for quantization scheme — they ship int8 weights post-training, suggesting our fp16+brotli archive can be ~2× smaller. |
| [`hmkx/HiNeRV`](https://github.com/hmkx/HiNeRV) | (NeurIPS 2023) | HiNeRV reference impl | Sibling lane (not SIREN family) |
| [`Orange-OpenSource/Cool-Chic`](https://github.com/Orange-OpenSource/Cool-Chic) | (active) | Cool-Chic / C3 reference codec | Sibling lane; we already have CompressAI primitives |
| [`kyleleey/E-NeRV`](https://github.com/kyleleey/E-NeRV) | ECCV 2022 | E-NeRV reference impl | Sibling NeRV-family lane |
| [`haofeixu/NeRV`](https://github.com/haofeixu/NeRV) | (older) | Original NeRV impl | Sibling NeRV-family lane |

## 4. Architectural gap analysis

### 4.1 CONFIRMED-CANONICAL: omega init scheme

Our `experiments/train_substrate_siren.py` defaults (`--first-omega 30.0`,
`--hidden-omega 1.0`) and `src/tac/substrates/siren/architecture.py:106-118`
init bounds match Sitzmann's `first_layer_sine_init` (Uniform(-1/n, 1/n)) and
`sine_init` (Uniform(-sqrt(6/n)/30, sqrt(6/n)/30)) exactly. CANONICAL.

### 4.2 GAP: hidden-width / depth deviation from canonical video config

`src/tac/substrates/siren/architecture.py:62-65` declares `hidden_dim=128,
num_hidden_layers=6` (~84K params). Sitzmann's `train_video.py` uses
`hidden_features=1024, num_hidden_layers=3` (~2.1M params) — **25× larger**.
Our config was chosen to fit the contest rate envelope (~100KB post-brotli);
the smaller H/L choice is rate-motivated, NOT distortion-optimal. The
literature would predict significantly lower memorization PSNR at our scale
`[literature-prediction]`. The contest's `100·d_seg + sqrt(10·d_pose)`
distortion contract is NOT equivalent to PSNR, so the actual score impact is
empirical — but the literature strongly suggests this is the dominant risk.

**Action:** declare a probe-arm `H=180, L=4` if the first-anchor archive
post-brotli leaves rate headroom (i.e., archive_bytes < 80KB).

### 4.3 GAP: temporal coordinate as single scalar input

`src/tac/substrates/siren/architecture.py:178-183` builds coords as
`(spatial_x, spatial_y, t)` where `t = pair_idx / num_pairs`. This is the
EXACT primitive TeNeRV/CANeRV/HNeRV identify as the failure mode for video
INR. Quote from TeNeRV: *"the limited network capacity is consumed by
competing temporal patterns, resulting in over-smoothed artifacts and
substantial loss of fine-details"* `[third-party-empirical:TeNeRV]`.

For our 600-pair video with first-layer omega_0=30, the temporal frequency
support is `sin(30·t)` where `t ∈ [0, 1)`. The MLP can in principle encode
~30 temporal cycles, but each pair needs to be discriminable — that's 600
unique "signatures" with ~30 Hz support. The aliasing risk is real.

**Action:** in the post-anchor diagnostic, plot per-pair reconstruction PSNR
vs `pair_idx`. Look for: (a) uniform smoothing (low PSNR everywhere — the
WINNER spectral-bottleneck mode), (b) high-frequency content collapse
(PSNR drops on high-motion pairs — the TeNeRV mode), (c) aliasing
(periodic PSNR pattern aligned with `omega_0=30` cycles).

### 4.4 GAP: rate proxy is a constant during training

`experiments/train_substrate_siren.py:434-449` `_archive_bytes_proxy_closed_form`
returns `num_params * 2` (fp16 raw bytes), constant. The Lagrangian rate term
is therefore an additive constant during gradient updates, and the training
loop has NO gradient signal toward smaller archives. This is HONESTLY
DOCUMENTED in the code ("A future Phase 2 lane will replace this with a
differentiable rate proxy"), but it is a STRUCTURAL LIMITATION the literature
addresses: COIN++ + COMBINER + NVRC all train against a differentiable rate
proxy (e.g. quantization-aware bit accounting from
`Balle entropy_bottleneck`).

**Action:** for the first dispatch, accept this limitation — but record the
post-anchor archive bytes vs Sitzmann's `n_params * 2` proxy as
empirical evidence for whether the proxy is even a good upper bound. If the
brotli-actual is dramatically smaller than the proxy, the rate term during
training was effectively scaled wrong.

### 4.5 GAP: no quantization (fp16 only)

`src/tac/substrates/siren/archive.py:85` stores `state_dict` as `fp16 cpu` +
brotli. COIN ([Dupont 2021](https://arxiv.org/abs/2103.03123)) shows that
INR weights can be quantized to int8 or below with minimal quality loss
post-training. The marginal rate cost at our operating point is `25·1024/N =
6.8e-4 score per KB` (Section 5), so a 2× compression via int8 quantization
would yield `~50KB · 6.8e-4 = +0.034` score reduction on the rate axis.

**Action:** post-anchor, if distortion is acceptable, run a no-retrain int8
quantization of the EMA shadow weights as a +0 dispatch follow-up. The
expected score delta is `≈-0.034` (improvement) at zero additional GPU spend.

### 4.6 GAP: spectral-bottleneck failure-mode unguarded

Per `2509.12980` WINNER, SIREN with default init can enter a
"spectral-bottleneck" mode where output collapses to near-zero. Our impl has
NO probe for this; the val loss would just stay high. **Recommendation:**
after first anchor, if proxy val loss is suspiciously plateaued or output
mean is near 0.5 (sigmoid midpoint), apply WINNER's spectral-centroid noise
perturbation (~5-10 LOC) as Round-2.

### 4.7 NON-GAP: eval-roundtrip + EMA + score-aware Lagrangian + patched yuv6

`src/tac/substrates/siren/score_aware_loss.py:80-105` correctly threads
`apply_eval_roundtrip_during_training` through both rendered pairs and
imports through the canonical `score_pair_components` helper. CLAUDE.md
non-negotiables on EMA (0.997 default), eval_roundtrip (forbidden False),
score-aware Lagrangian (alpha·B/N + beta·d_seg + gamma·sqrt(d_pose)),
patched yuv6, NaN watchdog, scorer-not-at-inflate, contest-runtime emission
(Catalog #146) are ALL HONORED. CLEAN.

## 5. Recommendations (ordered by EV/dollar)

| # | Recommendation | LOC | Cost | Predicted impact | Priority |
|---|---|---|---|---|---|
| R1 | **Proceed with $6 Modal A100 dispatch as-is** — engineering is canonical; literature predictions are uncertain enough that empirical first-anchor data is the highest-EV next step | 0 | $6 | `[literature-prediction]` 0.18-0.22 contest-CPU more likely than recipe's 0.145 | **DO** |
| R2 | **Add post-anchor diagnostic** logging per-pair PSNR vs `pair_idx` to detect WINNER spectral-bottleneck / TeNeRV over-smoothing modes | ~30 LOC | $0 | Diagnostic only; informs Round 2 | **DO** (parallel to dispatch) |
| R3 | **If post-anchor score 0.18-0.25, apply WINNER init perturbation** (1-LOC override on `_SinLayer.__init__` weight noise; spectral centroid estimator ~10 LOC) | ~10 LOC | $6 re-dispatch | `[third-party-empirical:WINNER]` "significant gains" → predict 0.01-0.03 improvement | Conditional |
| R4 | **If post-anchor rate budget < 80KB**, add probe-arm `H=180, L=4` (~104K params, ~100KB post-brotli) | ~5 LOC + 1 recipe row | $6 re-dispatch | `[literature-prediction]` ~0.03 distortion improvement at same rate | Conditional |
| R5 | **Post-anchor int8 quantization** (no retrain) following [COIN](https://arxiv.org/abs/2103.03123) | ~30 LOC (quantize + inflate.py decode path) | $0 | `[literature-prediction]` ~0.034 score improvement on rate axis | Conditional |
| R6 | **FINER activation drop-in** if SIREN baseline FAILS to converge | ~10 LOC | $6 re-dispatch | `[third-party-empirical:FINER]` Outperforms SIREN on PSNR; score-axis effect unknown | Conditional |
| R7 | **Acknowledge structural disadvantage vs HNeRV-family** in lane registry notes; pre-commit to DEFERRED-pending-{HNeRV-parity-or-rate-headroom} if first anchor > 0.22 | 0 | $0 | Discipline | **DO** (this memo IS the acknowledgment) |
| R8 | **DEFERRED**: NeRF-style positional encoding on top of SIREN | 0 | $0 | Double-counting per SIREN paper; predicted harmful | **SKIP** |
| R9 | **DEFERRED**: BACON / WIRE / MFN architectural rewrites | 0 | $0 | High LOC, unverified contest-axis EV | **SKIP for now** |
| R10 | **DEFERRED**: Cool-Chic / C3 / hyperprior — already a sibling lane via `tac.composition.registry` CompressAI primitives | 0 | $0 | Not a SIREN-substrate concern | **SKIP** (sibling lane) |

## 6. Cross-cutting observations

### 6.1 The 0.193 frontier is HNeRV-family for a structural reason

Public PRs #100/#101/#103 (0.193-0.195 contest-CUDA) all use HNeRV-family
content-adaptive embeddings, NOT pure coordinate MLPs. HNeRV reports +4.7
PSNR over NeRV at same param budget on UVG `[third-party-empirical:HNeRV]`.
This is a STRUCTURAL prior favoring per-frame conditioning over per-frame
temporal coordinates for the contest's particular scoring contract (which is
SegNet/PoseNet on pairs of frames, not aggregate PSNR).

**Prediction with uncertainty:** SIREN-as-renderer at our rate budget is
likely structurally dominated by HNeRV-family. Estimated probability that
SIREN beats 0.193 contest-CPU: **15-25%** `[literature-prediction]`. The
recipe's `predicted_score_target: 0.145` is optimistic; the council Phase 5
prediction did not weight the temporal-coord limitation per the TeNeRV
analysis. **Operator-routable:** revise predicted_band downward to
`[0.18, 0.25]` before dispatch, OR accept the optimism with a documented
DEFERRED-pending-empirical reactivation clause.

### 6.2 Score-axis vs PSNR-axis: the apples-to-apples discipline matters

Every SIREN-family paper I surveyed reports PSNR / LPIPS / SSIM on
images/videos with no scoring-axis equivalent to the contest's
`100·d_seg + sqrt(10·d_pose) + bytes_term`. The Tancik/Lindell/Liu/Sitzmann
PSNR gains do NOT directly transfer to our scoring contract. The empirical
question is unresolved: **does an INR architecture that gains +4 dB PSNR also
gain on SegNet/PoseNet distortion?** Our score-aware Lagrangian training
attempts to bias toward the contest distortion directly, but the
distortion-loss-to-PSNR mapping is dataset-specific and architecture-specific.

The conservative read: **SIREN literature is informative on the spectral
side but cannot predict score-axis EV.** First-anchor empirical is required.

### 6.3 SIREN's "video" experiments were never compression experiments

Sitzmann's video config (H=1024, L=3) memorizes 8-second clips. Compression
was NEVER the design target — SIREN was an INR-quality paper. COIN
(`2103.03123`) is the first compression-axis SIREN-family paper, and even it
focuses on bit-rate-vs-PSNR, NOT bit-rate-vs-segmentation/pose-distortion.

The pact SIREN substrate is exploring a regime literature has NOT bench-marked:
score-aware Lagrangian training of a SIREN under <100KB rate budget for
SegNet+PoseNet distortion. **This is genuinely novel research territory.**
The lane's `research_only=true` tag is correct.

### 6.4 The HNeRV parity discipline (CLAUDE.md NON-NEGOTIABLE) is honored

The 13 inviolable lessons review in `src/tac/substrates/siren/__init__.py:29-46`
correctly declares `lane_class=substrate_engineering` exception for L7 (~520
LOC vs ≤350) and `research_only=true` opt-out at design time. CLEAN per
Catalog #124 archive-grammar gate.

## 7. What this review does NOT establish

- **No hardware verification.** Did not run SIREN training. Did not measure
  any score. Every architectural prediction is `[literature-prediction]` or
  `[third-party-empirical:<paper>]` on a NON-CONTEST scoring contract.
- **No ablation.** The FINER / WINNER / RFF predictions are derived from
  abstracts and partial paper reads, not full-paper analysis and not
  contest-axis replay.
- **No code review of FINER / BACON / WIRE official implementations.** Their
  drop-in feasibility into our `_SinLayer` is sketched, not confirmed.
- **Empirical SIREN parameter-vs-rate Pareto curve is unknown.** The Section
  4.2 prediction (smaller H/L → lower PSNR) is qualitative; exact crossover
  config for our 100KB envelope requires the post-anchor data.
- **Score-axis effect of any literature recommendation is uncertain.** PSNR
  gains do NOT directly transfer to our `100·d_seg + sqrt(10·d_pose)` axis.

## 8. Operator-routable decisions surfaced

1. **DECISION — Revise predicted_score_target?** The recipe declares 0.145;
   the literature predicts 0.18-0.22 is more likely. Either revise downward
   (with cross-ref to this memo) OR accept the optimism with a documented
   DEFERRED-pending-empirical clause.

2. **DECISION — Add post-anchor diagnostic per R2?** ~30 LOC patch to the
   trainer's val loop. Cost: 0 GPU. Benefit: directly informs whether
   Round 2 should target WINNER (spectral bottleneck), TeNeRV (temporal
   over-smoothing), or the rate axis. **Recommend YES.**

3. **DECISION — Pre-commit Round-2 conditional dispatches?** If first anchor
   lands in `[0.18, 0.22]`, the highest-EV Round 2 is WINNER + int8
   quantization (~$6 + ~$0). If it lands in `[0.22, 0.30]`, the lane should
   DEFER-pending-HNeRV-parity-renderer or rate-axis attack on the existing
   archive. If it lands >0.30, the lane DEFERS-pending-empirical-rescue.

4. **DECISION — Approve Round-2 EV/budget split?** The literature suggests
   ~$12-20 of probe budget post-first-anchor produces ~3-5 ablation arms
   covering the dominant uncertainty modes. Pre-authorize?

## Cross-refs

- `feedback_siren_pre_dispatch_audit_LANDED_20260513.md` — the audit that
  surfaced 2 CRITICALs + 3 DEFECTs on the dispatch chain
- `feedback_siren_pre_dispatch_audit_fix_wave_LANDED_20260513.md` — the
  fix-wave landing (Catalog #190 + #191 STRICT @ 0)
- `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md`
  — the 13 inviolable lessons; HNeRV-family structural dominance on this
  scoring contract
- `feedback_substrate_vs_codec_composition_meta_pattern_20260508.md` —
  substrate-engineering vs codec-composition framing
- CLAUDE.md: "HNeRV / leaderboard-implementation parity discipline",
  "Apples-to-apples evidence discipline", "Frontier target",
  "Meta-Lagrangian/Pareto solver", "Adversarial council review of design
  decisions"

## Sources

- [Sitzmann et al., NeurIPS 2020 — arxiv 2006.09661](https://arxiv.org/abs/2006.09661)
- [vsitzmann/siren official repo](https://github.com/vsitzmann/siren)
- [lucidrains/siren-pytorch](https://github.com/lucidrains/siren-pytorch)
- [Tancik et al., NeurIPS 2020 — arxiv 2006.10739](https://arxiv.org/abs/2006.10739)
- [Lindell et al., BACON CVPR 2022 — arxiv 2112.04645](https://arxiv.org/abs/2112.04645)
- [Liu et al., FINER CVPR 2024 — arxiv 2312.02434](https://arxiv.org/abs/2312.02434)
- [Anon., WINNER ICLR 2026 prep — arxiv 2509.12980](https://arxiv.org/abs/2509.12980)
- [Dupont et al., COIN ICLR-W 2021 — arxiv 2103.03123](https://arxiv.org/abs/2103.03123)
- [Dupont et al., COIN++ TMLR 2022](https://openreview.net/forum?id=NXB0rEM2Tq)
- [Chen et al., NeRV NeurIPS 2021](https://openreview.net/forum?id=BbikqBWZTGB)
- [Chen et al., HNeRV ICCV 2023 — arxiv 2304.02633](https://arxiv.org/abs/2304.02633)
- [Kim et al., E-NeRV ECCV 2022 — arxiv 2207.08132](https://arxiv.org/abs/2207.08132)
- [Kwan et al., HiNeRV NeurIPS 2023 — arxiv 2306.09818](https://arxiv.org/abs/2306.09818)
- [Ladune et al., Cool-Chic TIP 2024 — arxiv 2401.02156](https://arxiv.org/abs/2401.02156)
- [Kim et al., C3 CVPR 2024 — arxiv 2312.02753](https://arxiv.org/abs/2312.02753)
- [Liu et al., CANeRV 2025 preprint](https://arxiv.org/html/2502.06181)
- [TeNeRV 2026 preprint](https://arxiv.org/html/2601.17743)
- [Strümpler et al., SINR 2025 — arxiv 2503.19576](https://arxiv.org/html/2503.19576)
- [Yang et al., NVRC NeurIPS 2024](https://arxiv.org/html/2409.07414)
- [Benbarka et al., WACV 2022](https://openaccess.thecvf.com/content/WACV2022/papers/Benbarka_Seeing_Implicit_Neural_Representations_As_Fourier_Series_WACV_2022_paper.pdf)
- [Saragadam et al., WIRE CVPR 2023](https://github.com/vishwa91/wire)
- [Schwarz et al., COMBINER NeurIPS 2023](https://dl.acm.org/doi/10.5555/3666122.3666216)
- [EmilienDupont/coin](https://github.com/EmilienDupont/coin)
- [hmkx/HiNeRV](https://github.com/hmkx/HiNeRV)
- [Orange-OpenSource/Cool-Chic](https://github.com/Orange-OpenSource/Cool-Chic)
- [kyleleey/E-NeRV](https://github.com/kyleleey/E-NeRV)
- [liuzhen0212/FINER](https://github.com/liuzhen0212/FINER)
- [computational-imaging/bacon](https://github.com/computational-imaging/bacon)
