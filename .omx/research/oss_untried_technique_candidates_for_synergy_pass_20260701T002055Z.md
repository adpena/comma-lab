# OSS + literature mining — UNTRIED techniques for the level-set task-space witness (synergy pass, feeds task #201)

- **UTC:** 2026-07-01T00:20:55Z
- **git HEAD:** 45055f517604edb2322fd0659ee8c2c8748b9452
- **Axis / authority:** `[advisory only]` · `[literature-signal]` — NO exact eval run here. Pointer UNMOVED at contest-CPU **0.19110** (the SoT is `.omx/state/canonical_frontier_pointer.json`; this file asserts no score).
- **Budget:** $0, online + local-CPU, NO GPU. This is a research/design artifact, not a measured row.
- **Scope:** techniques / losses / levers / optimizers / curricula we have **NOT** tried, that could lower S or shorten train-time for the **level-set task-space witness** (coordinate-INR amortizing the frozen SegNet argmax partition + PoseNet-via-stored-sidecar; indirect rate-distortion / coding-for-machines).
- **NO-FAKE discipline applied:** every citation below was web-verified this session (arXiv id + repo exists — see the Verification Appendix). Every **EV number is a HYPOTHESIS** (derived / analogized from the cited paper's domain result), **NOT a measurement** on our scorer. Uncertainty flagged inline. Any adopt/kill verdict requires our own byte-closed measured row per CLAUDE.md.

---

## 0. How to read this + what we ALREADY ship (the untried delta)

**Already in-tree (do NOT re-recommend as "new"):** eikonal (0.01) + length (0.001) level-set regularizers; the **directional/curvelet (anisotropic) Fourier basis** (measured **−48% d_seg**, ~0 byte — `--n-dir-freqs / --freq-across/along / --max-bank-freq / --reorient-every`; per CLAUDE.md "ALL-CLASS DIRECTIONAL ... THE decisive lever"); deterministic Fourier features (seeded B-matrix, free in inflate.py); FiLM-per-pair; Muon finisher (Keller Jordan; "THE drop"); CE→tau_softplus→l7→Muon curriculum; Gauss/step activation (in-code but **UNSWEPT**); Cool-Chic/C3/HNeRV (parity banks); UNIWARD (named LEVER-4). These are the **baseline** — everything below is a delta on top.

**EV notation:** `Δd_seg` = relative change in the witness d_seg term (baseline d_seg ≈ 0.008257; best measured ≈ 0.004445; need ≈ 0.001). S-impact of the seg term ≈ `100 × Δd_seg_absolute` (so a 10% cut from 0.0044 ≈ **−0.04 S**). `Δrate` = counted `archive.zip` residual bytes → S-impact `25 × Δbytes / 37,545,489`. `train-time` = relative wall-clock to reach a fixed d_seg. **Maturity** ∈ {measured-elsewhere (paper), derived (first-principles for our task), untested-derivable}.

---

## Area 1 — Segmentation / argmax losses beyond CE

The witness's binding metric is **argmax-flip rate on a codim-1 boundary annulus** (flip mass ≈ 50% Road / 19% Lane / 13% Undrivable; lane = thin, IoU 0.263, the hard ~8-dim orbit). CE spreads gradient over all pixels; the flip metric only cares about the small-margin annulus. These losses re-aim the gradient at the boundary and at rare/thin classes.

**1a. Boundary loss (Kervadec et al.) — `arXiv:1812.07032` · repo `github.com/LIVIAETS/surface-loss` (MIDL 2019 best paper).**
- *Mechanism:* loss is a **distance metric on the space of contours**, implemented as a region integral weighted by the **signed distance function (level-set) of the GT boundary**: `L_B = Σ_p φ_G(p)·s_θ(p)`. It replaces unbalanced region integrals with an interface integral → designed for **highly-unbalanced, thin structures** (exactly the lane-marking orbit).
- *Why it fits us uniquely:* it is **literally an SDF/level-set integral** — the same object our witness already computes (eikonal/length on the SDF). Zero conceptual impedance; the φ_G map is precomputed once from the frozen GT argmax.
- *EV [HYPOTHESIS — derived]:* Δd_seg **−15% to −35%** (targets the annulus + rare thin classes where CE under-weights). train-time ≈ neutral to −10% (smoother boundary gradient). Δrate 0.
- *R-survival:* **robust** — it's a soft region integral, not a hard edge-detector op, so the bicubic↑/uint8-STE/bilinear↓ round-trip does not break it.
- *Composition:* add as a **curriculum stage** after CE (Kervadec's own recipe is `α·CE + (1−α)·L_B` with α annealed 1→~0.01) or as a joint term with tau_softplus. Synergizes with the SDF machinery and the curvelet basis (basis places capacity; boundary loss tells it *where the contour must land*).

**1b. clDice / soft-clDice (Shit et al.) — `arXiv:2003.07311` · repo `github.com/jocpae/clDice` (also in MONAI).**
- *Mechanism:* topology-preserving similarity on the **morphological soft-skeleton** of the mask; provably preserves connectivity of tubular/network structures up to homotopy. Differentiable (`soft_skeletonize`).
- *Why it fits us:* the **19% lane-marking flip mass** is precisely thin connected structures whose argmax flips *break the dash chain*. clDice penalizes exactly those connectivity breaks that pixel-CE ignores.
- *EV [HYPOTHESIS — derived]:* on the **class-1 (lane) sub-residual only**, Δd_seg **−20% to −40%** of the lane component (≈ −4% to −8% of total d_seg). Cheap.
- *Composition:* class-1-gated auxiliary term (`+λ·soft_clDice(pred_c1, gt_c1)`), λ small. Pairs with the finest curvelet scale (1e below) and the Morse-Smale birth-death dashes.

**1c. Region Mutual Information (RMI) loss (Zhao et al.) — `arXiv:1910.12037` · repo `github.com/ZJULearning/RMI` (NeurIPS 2019).**
- *Mechanism:* models each pixel by its k×k neighborhood as a high-dim point and **maximizes mutual information** between pred and GT neighborhood distributions → high-order structural consistency, sharper boundaries. ~zero test overhead.
- *EV [HYPOTHESIS — derived]:* Δd_seg **−8% to −20%** (structural/boundary consistency), low integration cost.
- *R-survival:* moderate (neighborhood covariance is smooth). *Composition:* joint term with CE; complementary to boundary loss (RMI = structural coupling; boundary = contour placement).

**Runners-up (area 1):** **Focal loss** (Lin et al. `arXiv:1708.02002`) — down-weights easy pixels, concentrates gradient on the small-margin annulus; **drop-in, R-agnostic, ~free**; strong *faster-convergence + tail* lever (see top-8 #7). **Lovász-Softmax** (Berman et al. `arXiv:1705.08790` · repo `github.com/bermanmaxim/LovaszSoftmax`, CVPR 2018) — direct IoU surrogate; helps the rare lane class (IoU-sensitive) but d_seg is argmax-*disagreement-rate* not IoU, so it's a looser surrogate than boundary/focal. **Active Boundary Loss** (Wang et al. `arXiv:2102.02696` · repo `github.com/wangchi95/active-boundary-loss`, AAAI 2022) — per-pixel direction vector pushing the predicted boundary onto GT; *most literal* "move the argmax boundary" loss, **but** relies on a boundary-detection op that is **fragile through R** → flag; prefer boundary loss (1a) which achieves the same intent robustly.

---

## Area 2 — Level-set / topology / boundary-aware losses

We already have eikonal + length. The untried deltas are **topology/persistence** losses (guarantee correct connected-component / Betti structure of the partition, which directly bounds a class of argmax flips) and the boundary-loss SDF term (covered in 1a — it is simultaneously a level-set and a segmentation loss).

**2a. Topology-preserving segmentation via persistent homology (Hu et al.) — `arXiv:1906.05404` (NeurIPS 2019) + companion `arXiv:1910.01877` · repo `github.com/HuXiaoling/TopoLoss`.**
- *Mechanism:* a differentiable loss on the **persistence diagram** that forces the prediction's Betti numbers (# components / holes) to match GT over all thresholds → kills broken-connection / spurious-blob flips at fine scale.
- *EV [HYPOTHESIS — derived]:* attacks a *specific* flip class (topology errors on thin lanes/dashes) → Δd_seg **−5% to −15%** concentrated on the lane long-tail. **Cost caveat:** persistent-homology computation is the slow part (train-time +10-30%); mitigated by the newer efficient variants below.
- *Composition:* late-stage aux, class-1 focus. Complementary to clDice (clDice = skeleton connectivity; persistence = full Betti spectrum).

**2b. Efficient Betti matching (Stucki et al.) — `arXiv:2407.04683`; spatial-aware persistent feature matching — `arXiv:2412.02076`.**
- *Mechanism:* modern, **much faster** topology-aware losses (Betti matching / spatially-localized persistence) that make 2a practical inside a training loop.
- *EV [HYPOTHESIS — derived]:* same d_seg intent as 2a with train-time overhead cut to +5-15%. Prefer these over vanilla TopoLoss if 2a's cost bites. Maturity: measured-elsewhere (medical seg); untested for us.

**Note:** IGR eikonal (Gropp et al. `arXiv:2002.10099` · repo `github.com/amosgropp/IGR`, ICML 2020) is the canonical eikonal-SDF regularizer — **we already ship its eikonal term**, so it's not a new lever, but its `unit-gradient` formulation is the reference if we want to harden our SDF head.

---

## Area 3 — INR representation advances (sharp boundaries @ low param, fast convergence)

We ship deterministic Fourier features + a Gauss/step activation that is **in-code but UNSWEPT**. The deltas are activations with **better space-frequency locality** (sharper piecewise-constant edges without Gibbs) and tunable spectral bias.

**3a. FINER — variable-periodic activation (Liu et al.) — `arXiv:2312.02434` · repo `github.com/liuzhen0212/FINER` (CVPR 2024).**
- *Mechanism:* `sin((|x|+1)·x)` variable-period activation; **initializing the bias in different ranges selects sub-functions of different frequency** → the *supported frequency set is tunable per-neuron*, alleviating SIREN's fixed-band limitation. Better 2D fit + 3D SDF + NeRF than SIREN/Gauss at equal params.
- *Why it fits us:* the witness must represent a **sharp piecewise-constant argmax partition at low param** (capacity is the binding trilemma constraint). FINER's per-neuron frequency tuning = more *effective* capacity on the boundary without adding bytes.
- *EV [HYPOTHESIS — derived]:* Δd_seg **−10% to −25%** at equal param count (sharper edges) **AND** train-time **−15% to −30%** (faster convergence vs SIREN reported in-paper). Δrate ≈ 0 (activation, not stored).
- *Composition:* drop-in activation swap in the coord-INR trunk; **sweep against the existing Gauss/step activation** (which is the deep-math-predicted step-native lever, best-so-far 0.004445). This is the cleanest "finish the UNSWEPT activation sweep" action.

**3b. WIRE — Gabor-wavelet activation (Saragadam et al.) — `arXiv:2301.05187` · repo `github.com/vishwa91/wire` (CVPR 2023).**
- *Mechanism:* complex **Gabor wavelet** activation — *optimally concentrated in space AND frequency* (Heisenberg-optimal). Combines sine's frequency-compactness with Gaussian's spatial-compactness → smallest, most spatially-compact error; robust to noise/undersampling.
- *Why it fits us:* an argmax boundary is a **spatially-localized high-frequency event** on a smooth background — Gabor's joint space-frequency locality is the theoretically-matched chart (sister of the curvelet basis, which is itself a directional wavelet). WIRE and the curvelet basis are the *same wavelet family* seen from activation-side vs feature-side.
- *EV [HYPOTHESIS — derived]:* Δd_seg **−10% to −20%**, and **robustness to the R low-pass** (its spatial compactness resists Gibbs aliasing — the exact failure mode noted for naive sine). train-time neutral to −15%.
- *Composition:* activation swap; **strongest theoretical synergy with our curvelet feature basis** (feature basis orients, WIRE activation localizes). Sweep FINER vs WIRE vs Gauss/step as one A/B arm.

**Runner-up (area 3):** **Gauss activation "Beyond Periodicity"** (Ramasinghe & Lucey `arXiv:2111.15135`, ECCV 2022) — the theory behind our in-code Gauss activation; positional-embedding-free, robust to init. Not new (we have it) but the paper's tuning guidelines are the reference for the UNSWEPT sweep. Also note **"A Sampling Theory Perspective on Activations for INRs"** `arXiv:2402.05427` — bandwidth-matched activation selection, useful prior for picking the activation given the R low-pass bandwidth.

---

## Area 4 — Optimizers / faster-train (upgrade the Muon finisher; cut sweep cost)

Muon is "THE drop." The deltas either **upgrade Muon directly** (fix its capacity-wasting failure) or **cut the A/B sweep cost**.

**4a. Aurora — leverage-aware Muon fix (Tilde Research) — repo `github.com/tilde-research/aurora-release` · blog `blog.tilderesearch.com/blog/aurora` (2026-05). ⚠ NO arXiv (blog + repo only) — claims unverified, flag.**
- *Mechanism:* Tilde found Muon's orthogonalization lets **>25% of MLP neurons permanently die early** (weak neurons keep getting weak updates). Aurora **redistributes update mass across rows** of the up/gate projections → prevents neuron death at ~6% overhead.
- *Why it fits us uniquely:* our witness is a **capacity-limited tiny MLP** — dead neurons = wasted capacity = *directly* the binding trilemma constraint. Preventing neuron death → more effective capacity at equal bytes → on the sub-0.15 path. And it upgrades the exact optimizer ("THE drop") we already rely on.
- *EV [HYPOTHESIS — derived, HIGH uncertainty]:* Δd_seg **−5% to −20%** (via recovered capacity); train-time neutral-to-better. **Uncertainty: HIGH** — very new, no peer review, designed for large transformer MLPs (row-mass redistribution on up/gate proj); needs a $0 CPU smoke to confirm it helps a tiny INR, not a large LM.
- *Composition:* drop-in Muon replacement in the l7/Muon finisher stage. **Do a $0 CPU sanity smoke first** (dead-neuron fraction with Muon vs Aurora on the witness trunk).

**4b. NAMO / NAMO-D — "Adam Improves Muon" (Schaeffer, Zhang, Liu) — `arXiv:2602.17080` (Feb 2026). [arXiv-backed alternative to 4a.]**
- *Mechanism:* integrates **orthogonalized momentum (Muon) with Adam-type norm-based noise adaptation**; NAMO uses a single adaptive stepsize preserving orthogonality; NAMO-D right-multiplies by a clamped diagonal for neuron-wise adaptation (matches near-block-diagonal Hessian).
- *EV [HYPOTHESIS — derived]:* improves Muon "at negligible cost" (per abstract) → train-time −5% to −15%, small d_seg gain. Lower ceiling than Aurora's capacity claim but **peer-verifiable / lower risk**.
- *Composition:* drop-in for the Muon stage. Use as the **safe** faster-train pick if Aurora's smoke disappoints.

**4c. muP / muTransfer — zero-shot HP transfer (Yang et al., Tensor Programs V) — `arXiv:2203.03466` (+ unit-scaled u-µP `arXiv:2407.17465`).**
- *Mechanism:* parametrize in **maximal-update parametrization** so the optimal LR (and other HPs) are **stable across width** → tune LR on a tiny proxy witness, **zero-shot transfer** to the full witness.
- *Why it fits us:* the 10-lever per-stage A/B campaign runs MANY arms, each needing an LR. muP collapses every arm's LR sweep to one tiny-proxy sweep → **large aggregate wall-clock savings across the campaign**.
- *EV [HYPOTHESIS — derived]:* not a per-run d_seg lever; a **campaign-level train-time −30% to −60%** by killing redundant LR sweeps. Caveat: muP's benefit is width-scaling; for a single fixed tiny width the gain is smaller (use it when sweeping width / capacity-routing).

**Runner-ups (area 4):** **Schedule-Free** (Defazio et al. `arXiv:2405.15682` · repo `github.com/facebookresearch/schedule_free`) — removes the LR-schedule/stopping-time tuning entirely, no extra HPs, AlgoPerf-winning; **safest drop-in train-time reducer** for the AdamW stages (not the Muon stage). **Sophia** (Liu et al. `arXiv:2305.14342` · repo `github.com/Liuhong99/Sophia`) — diagonal-Hessian second-order, ~2× over AdamW; alt finisher, but Muon+Aurora likely dominates for our matrix-structured trunk. **MD-Decoupling** (magnitude-direction decoupling; candidate `arXiv:2606.25971` "Improving NN Training by Decoupling the Magnitude and Direction of Weight Vectors" — ⚠ verify exact id/EPFL attribution; already on our radar per MEMORY) — makes stage *transitions* stable-by-construction (relevant to the "different stages need different treatment" rule).

---

## Area 5 — Training / scheduling / curriculum / stopping (FIRST-CLASS AXIS; "shortest-train" is an explicit objective)

**Scope discipline (NO-FAKE):** "shortest-train" is a stated objective, so this axis is central. But this file's mandate is **untried OSS/literature**. Below, **5.0 maps the in-tree BASELINE we already ship** (context, so the untried deltas are legible) and **5a–5d are the untried OSS deltas** (my verified contribution). In-tree measured-EV / task-IDs are cited **as reported in CLAUDE.md / git-log / coordinator relay** and are flagged **`[in-tree — measured EV in our ledgers, NOT re-verified in this $0 literature pass]`** — I do not re-assert them as this pass's measurements.

### 5.0 In-tree training-axis BASELINE (context, NOT untried — do not re-recommend)

| In-tree lever | What it is | Reported EV | Status |
|---|---|---|---|
| 4-stage curriculum CE→tau_softplus@300→l7@600→Muon@726 | distilled from PR95 8-stage/29,650ep (CLAUDE.md L14, task #176) | Muon is "THE drop"; smooth-stage RAISES d_seg (dropped) | `[in-tree]` shipped |
| per-stage different-treatment + transition RE-TREAT | `--stage-transition-rewarmup-epochs`, `--stage-transition-reset-moments` | prevents margin-stage inheriting base-stage treatment | `[in-tree]` (per `feedback_different_stages_need_different_treatment_...`) |
| softmax-temp / tau anneal 1.0→0.05 | `--tau-anneal-shape {cosine/geometric/cosine_hold}`, `--tau-hold-frac`, `--anneal-epochs` | CE+softplus LOWER d_seg | `[in-tree]` (#119 calibration) |
| Muon finisher LR | `--muon-lr 0.002` (MANDATORY, #164) | the d_seg drop | `[in-tree]` |
| EMA 0.997 shadow + EMA-best + per-stage ckpt | inference = EMA shadow; atomic per-stage save | crash-resume + early byte-close + per-stage A/B | `[in-tree]` (EMA non-negotiable) |
| eikonal(0.01)+length(0.001) + junction regularizers | level-set PDE regularizers | live derivative/integral terms | `[in-tree]` |
| curvelet/directional basis (STATIC `max-bank-freq`) | anisotropic Fourier basis | **−48% d_seg, ~0 byte (MEASURED, CLAUDE.md)** | `[in-tree]` (the lever 5a schedules) |
| θ* TIER-3 scale-curriculum + perspective-aware chart (#185); from-scratch openpilot-seeded (#191); training-signal regularizers latent-structure #110 / variable-grid QAT #111 / color-offset #113 | our own campaign levers | reported internally | `[in-tree — EV in ledgers, not re-verified here]` |

The untried OSS deltas 5a–5d act **on top of** this baseline.

### ⭐ 5a. CURVELET SCALING CURRICULUM (first-class NEW lever — coarse→fine ramp of `max-bank-freq`)

- *What we have:* the directional/curvelet basis is the **measured −48% d_seg** lever (CLAUDE.md: "ALL-CLASS DIRECTIONAL (anisotropic/curvelet) Fourier basis [THE decisive lever, ~0 byte] — −48% d_seg; lane-only is only −8%"), but it ships **STATIC** — a fixed `--max-bank-freq`, all curvelet scales active from epoch 0.
- *The untried delta:* **progressively ramp `max-bank-freq` (and add curvelet scales) coarse→fine over epochs** — start with only coarse directional bands, linearly unmask finer scales as training proceeds. This is the curvelet-basis instantiation of the **frequency-mask curriculum** proven in FreeNeRF and BARF.
  - **FreeNeRF** (Yang et al. `arXiv:2303.07418` · repo `github.com/Jiawei-Yang/FreeNeRF`, CVPR 2023): a **linearly-increased frequency mask** over positional-encoding bands based on training step — a "free lunch," zero compute cost, SOTA in the few-shot/low-data regime (our regime: overfit one clip).
  - **BARF** (Lin et al. `arXiv:2104.06405`, ICCV 2021): proves **naive full-frequency positional encoding hurts** the optimization landscape and coarse-to-fire frequency scheduling fixes it (theoretical connection to classical coarse-to-fine image alignment).
- *Transferable mechanism:* early epochs see a smooth low-frequency landscape (no high-freq noise to fight → the coarse Road/Undrivable/MyCar partition locks in fast); fine curvelet scales are introduced *only after* the coarse partition is placed, so their capacity goes to the **boundary annulus + dash long-tail** instead of polluting the bulk.
- *EV [HYPOTHESIS — derived, untested-derivable]:*
  - **train-time:** −20% to −40% (avoids the spectral-bias fight; the documented FreeNeRF/BARF mechanism — fewer wasted epochs on the annulus tail).
  - **d_seg:** additional **−5% to −15%** on top of the static curvelet basis (finest scales, added late, target the boundary/dash long-tail without early over-fitting). In S units on the witness: from d_seg ≈ 0.0044, a 10% cut ≈ **ΔS ≈ −0.04**; combined with the train-time cut this is a high ΔS-per-train-time lever.
  - **Δrate:** 0 (the basis remains a deterministic, seeded, free-in-inflate.py generator; only the *schedule* changes, which is compile-time not stored).
  - **maturity:** untested-derivable (mechanism measured elsewhere in FreeNeRF/BARF; the curvelet-scale instantiation is our own, unmeasured).
- *Synergy (per coordinator's note, all derivable):*
  - **NTK seed → annulus:** the coarse-to-fine schedule and NTK-aware Fourier-feature scaling are the *same knob* (NTK bandwidth = which frequencies are learnable when); ramping `max-bank-freq` is literally scheduling the effective NTK bandwidth toward the annulus.
  - **structured-init:** seed the coarse scales at a structured (boundary-tangent-oriented) init so the coarse partition lands on the true low-frequency boundary geometry from step 0.
  - **Morse-Smale birth-death:** dashes = the finest-scale 0-cells (births) in the Morse-Smale chart; introducing the finest curvelet scale *last* is exactly "resolve the birth-death pairs after the 2-cells/1-cells are fixed."
- *Composition with the loss levers:* pair the finest-scale unmask epoch with turning on clDice/boundary-loss (1a/1b) — capacity and loss arrive at the annulus together.

### 5b. Meta-learned initialization / warm-start (Strümpler; Tancik learned-init)

- **Strümpler et al.** `arXiv:2112.04267` (ECCV 2022, EPFL CVLab) — MAML meta-init reaches the INR encoding **in far fewer gradient updates** and improves rate-distortion. Full INR-compression pipeline (also relevant to Area 6). **Tancik et al.** "Learned Initializations for Coordinate-Based Representations" `arXiv:2012.02189` (CVPR 2021) — meta-learned init for coordinate MLPs.
- *Transferable mechanism:* meta-train a witness init across the **same-rig comma10k frames we already have** (30 frames / 26 drives, RAV4 same device — per `feedback_comma10k_membership_...`), then warm-start the contest-clip witness from that init → basin reached in a fraction of the from-scratch epochs.
- *EV [HYPOTHESIS — derived]:* **train-time −40% to −70%** for the per-clip fit (the dominant wall-clock sink across the A/B campaign); minor d_seg gain (better basin). Δrate 0 (init is deterministic/regeneratable, not stored — the *learned* init weights would be counted ONLY if they ship; keep the init in inflate.py as generic code, ship only the clip-specific residual).
- *Caveat / NO-FAKE:* meta-init weights derived from real frames are **video-derived** — if any meta-init weight ships in `archive.zip` it is COUNTED (rule 118). The compliant design: meta-init is a *training warm-start only*; the shipped artifact is still just the tiny clip-specific sufficient statistic. Higher build cost than an optimizer swap; higher ceiling.
- *Composition:* warm-start feeds EVERY A/B arm → multiplies the value of every other lever by cutting each arm's cost.

### 5c. Warm restarts for the stage-transition RE-TREAT (SGDR)

- **SGDR — Stochastic Gradient Descent with Warm Restarts** (Loshchilov & Hutter) — `arXiv:1608.03983` · repo `github.com/loshchil/SGDR` (ICLR 2017).
- *Mechanism:* cosine-annealed LR with periodic **warm restarts** (LR jumps back up, then re-anneals) → escapes the basin the prior stage settled into; the canonical OSS formulation of "re-treat at a stage transition."
- *Why it fits us:* we already do `--stage-transition-rewarmup-epochs / --stage-transition-reset-moments` (the "different stages need different treatment / transitions must RE-TREAT" rule). SGDR is the **published, tuned schedule** for exactly this, incl. cosine-restart shape + restart-period growth (T_mult) — a principled replacement for a hand-set rewarmup length, and it composes with the tau-anneal shapes we already sweep.
- *EV [HYPOTHESIS — derived]:* train-time neutral-to-−15% (fewer wasted post-transition epochs); small d_seg gain (better basin at the CE→tau and tau→l7 boundaries where flips re-form). Maturity: measured-elsewhere.
- *Composition:* schedule the SGDR restart to fire exactly at the curvelet-scale-unmask epoch (5a) and the boundary-loss-turn-on epoch (1a) — one coordinated re-treat instead of three uncoordinated ones. **Antagonism flag:** a restart can *un-place* an already-correct coarse partition if fired too late; gate the restart to before the l7 stage.

### 5d. ⏱ Stage-aware stopping / knee early-stop = THE shortest-train lever (critical-slowing + kneedle)

- **Critical Slowing Down Near Topological Transitions in Rate-Distortion Problems** (Agmon, Benger, Ordentlich, Tishby) — `arXiv:2103.02646` (+ Part I ResearchGate 349786895); foundational IB dynamics `arXiv:1503.02406` (Tishby & Zaslavsky).
- **Kneedle — knee/elbow detection** (Satopää et al., ICDCS-W 2011) · repo `github.com/arvkevi/kneed` (Python).
- *Mechanism:* Agmon-Tishby prove that near a **topological (bifurcation) transition of the rate-distortion solution, convergence CRITICALLY SLOWS with a power-law** — training near a class-boundary birth/death crawls. This is *precisely* the witness's 30k-epoch annulus tail (argmax topology transitions in an indirect-RD problem = our exact setting). The practical corollary: a **stage-aware stopping rule** — detect the power-law critical-slowing knee (kneedle on the d_seg-vs-epoch curve) and **early-stop / pivot at the knee** instead of grinding the flat tail.
- *Why it fits us:* directly operationalizes the in-tree "trajectory-dynamics early-stop-on-plateau/knee (#188)" as a **principled, cited** rule — and it is the **single cheapest wall-clock win** because it removes the longest (flat, low-return) segment of every run.
- *EV [HYPOTHESIS — derived, high-confidence mechanism]:* **train-time −30% to −60%** (kill the flat critical-slowing tail across every A/B arm) at **~0 d_seg cost** (by definition the tail's marginal d_seg is near-zero — that's why it's flat) → **highest ΔS-per-train-time of any lever here** (it buys wall-clock nearly for free). Δrate 0.
- *Composition:* the stage-aware variant fits per-stage ckpt + per-stage A/B — stop each stage at its own knee, warm-start the next. **Synergy with RL-lab idea (MEMORY):** the critical-slowing tail is exactly where the surrogate gradient vanishes → this quantifies *when* to hand the annulus to a non-differentiable optimizer instead of grinding surrogate SGD.
- *Caveat:* early-stop trades a little final d_seg for large wall-clock; when a run is a *frontier candidate* (not an A/B arm), run the tail. Use knee-stop for the campaign, full-run for the byte-close candidate.

**Runner-up (area 5):** **SWA — Stochastic Weight Averaging** (Izmailov et al. `arXiv:1803.05407`) — average weights along the trajectory for wider optima / better generalization; **note we already use EMA-shadow** (a related running-average), so SWA is a small delta (tail-averaging variant); low priority. Distillation warm-start from the bc36 PR95 inflate (we have it, d_seg 6.02e-4) as a *teacher* for the smaller witness is a curriculum option but risks the capacity-cliff already measured — flag as lower-EV.

---

## Area 6 — Rate / quantization / conditional-residual (the COUNTED bytes — binding sub-0.15 lever per 2026-06-30)

Per CURRENT-STATE (2026-06-30): lossless rate on the frontier is exhausted; **RATE is the binding sub-0.15 lever** → smaller representation / distortion-quant. For the witness, the counted bytes = the learned residual + any stored INR params. These give end-to-end learned entropy models and conditional/residual coding.

**6a. NVRC — Neural Video Representation Compression (Kwan et al.) — `arXiv:2409.07414` (NeurIPS 2024).**
- *Mechanism:* first **fully end-to-end** INR-video codec — neural representation **+ quantization model + entropy model jointly optimized under an RD objective**, plus a **hierarchical parameter-coding scheme** that codes the network/quantization/entropy params themselves (minimizes the entropy-model's own byte overhead). 23% over VVC on UVG.
- *Transferable mechanism:* replace our hand-picked quantization + generic compressor of the residual with a **co-trained entropy model** (add `−log p(quantized residual)` = the true R term to the objective) + **hierarchically code the entropy model's params** so its overhead doesn't eat the savings.
- *EV [HYPOTHESIS — derived]:* Δrate **−15% to −30%** of the counted residual bytes (the paper's coding gain over strong baselines) → direct S-impact via `25·Δbytes/37.5M`. train-time +10-20% (joint entropy model). Maturity: measured-elsewhere (UVG video); untested for our task-space residual.
- *Composition:* the rate half of the sub-0.15 path — pairs with "store only the ~8-dim lane-trajectory coords + minimal learned residual." The entropy model *is* the AR-coder we want over the residual.

**6b. RECOMBINER — Bayesian INR compression (He et al.) — `arXiv:2309.17182` · repo `github.com/cambridge-mlg/RECOMBINER` (ICLR 2024).**
- *Mechanism:* variational Bayesian INR that **avoids quantization** and **directly optimizes rate-distortion**; enriches the variational posterior via linear reparameterization of INR weights + **learnable positional encodings** + **patchwise hierarchical priors**. SOTA at **low bitrate** (our regime).
- *Transferable mechanism:* treat the witness weights/residual as a variational posterior compressed via relative-entropy coding → RD-optimal *without* a quantization grid; the learnable-PE + hierarchical-prior tricks are directly portable to our coord-INR.
- *EV [HYPOTHESIS — derived]:* Δrate **−10% to −25%** at low bitrate; especially strong where our residual is tiny (the low-bitrate regime it targets). train-time +15-30% (variational machinery); build cost non-trivial.
- *Composition:* alternative to 6a's quantized entropy model; run as a parallel rate arm.

**6c. Strümpler INR compression pipeline — `arXiv:2112.04267` (ECCV 2022).**
- *Mechanism:* the **practical** end-to-end INR compression recipe: post-training quantization → **quantization-aware retraining** → entropy coding, + MAML meta-init (see 5b). Battle-tested, lower-risk than 6a/6b.
- *EV [HYPOTHESIS — derived]:* Δrate **−10% to −20%** vs naive quant+generic-compress; **lowest integration risk** of the three. Use as the rate baseline to beat.

**Runner-up (area 6):** **MaskCRT** (`arXiv:2312.15829` · repo `github.com/NYCU-MAPL/MaskCRT`) + **Conditional Residual Coding** (Brand et al. `arXiv:2307.12864`) — learn a **soft per-pixel mask** that blends conditional coding (predict, store nothing) vs conditional-residual coding (store residual) *pixel-adaptively*. Transferable mechanism: the witness's deterministic generated partition is the "prediction"; a learned soft mask decides **per-region whether to spend residual bytes** — spend only where the texture-survival wall genuinely needs it, zero bytes elsewhere. Directly operationalizes "minimal learned residual." Also `arXiv:2305.02562` "Conditional and Residual Methods in Scalable Coding for Humans and Machines" (Cityscapes seg base) — the coding-for-machines framing that matches our indirect-RD objective.

---

## Area 7 — Steg-cost / UNIWARD-family (cheapest place to flip the CNN's perception)

We already name UNIWARD as LEVER-4. The untried delta is **using the cost model as a capacity/residual PLACEMENT PRIOR jointly with the SegNet's own saliency**, not as a standalone lever.

**7a. (S-)UNIWARD cost as a placement prior — Holub, Fridrich, Denemark, "Universal distortion function for steganography in an arbitrary domain," EURASIP J. Info. Security 2014. ⚠ journal paper, NO arXiv (flag); canonical + widely reimplemented.**
- *Mechanism:* UNIWARD assigns low embedding cost to **textured/high-variance regions** (directional wavelet residuals) where a detector CNN is least sensitive — the "square-root law / spread small errors in texture" discipline.
- *Transferable mechanism (the NEW composition):* build a placement map = **UNIWARD texture-cost ⊙ inverse-SegNet-margin-saliency**. Spend the witness's chroma/residual capacity where BOTH (i) the pixel is near an argmax flip (SegNet small-margin annulus) AND (ii) the change is cheap to hide in texture (UNIWARD low-cost). This is a joint "flip-the-argmax-cheaply" prior — the inverse-steganalysis dual of boundary loss.
- *EV [HYPOTHESIS — derived]:* not a standalone d_seg lever; a **capacity-routing prior** that could improve the KKT waterfill's Δd_seg-per-byte by concentrating changes where they flip the partition without triggering PoseNet/other-term regressions. Δd_seg **−5% to −15%** on the residual arm; Δrate favorable (fewer bytes for equal flip). Maturity: derived; untested.
- *Composition:* feeds the existing `boundary_routing.py` KKT capacity-router as the saliency weight; pairs with chroma (chroma is a d_seg lever and chroma texture is where UNIWARD cost is lowest).

**Runner-ups (area 7):** HILL cost (Li et al., ICIP 2014 — no arXiv) — high-pass + low-pass "spreading" cost, simpler than UNIWARD; adversarial-steg cost learners (ADV-EMB / min-max embedding, Tang et al. — verify before citing) — learn the cost against the *actual* frozen SegNet detector rather than a generic texture prior (the score-native version of 7a). Flag: these need per-paper verification before any adopt.

---

## Area 8 — Substrate & temporal-factor FACETS of the unified action (IN-TREE; context, not OSS-untried)

These two are **already built** (per git-log / MEMORY / coordinator relay) and are the shortest-train compute layer + the temporal/pose factor of the one unified action. They are **NOT untried OSS** — I include them because "shortest-train" is an objective and #201's synthesis needs the map. **All EVs here are `[in-tree — reported in git-log/MEMORY, NOT a measurement of this literature pass]`.** My literature contribution is only the OSS references that could *further* them.

### 8a. MLX / Metal substrate = the SHORTEST-TRAIN COMPUTE LAYER `[in-tree]`
- *What we ship:* MLX-first witness + frozen SegNet/PoseNet MLX port (FP32-exact parity #88/#89); custom Metal kernels — **fused-R operator** (`@mx.custom_function` + `mx.fast.metal_kernel`, bit-identical, a=−0.75) + **custom grouped-backward ~17× fast path** (`TAC_MLX_CUSTOM_GROUPED_BACKWARD=1`, exonerated bit-identical — this env is now the launcher DEFAULT per git-log HEAD `b46c79ac6`); `mx.compile` on the d_seg step; MPS as a valid **training-gradient** device (never a score authority); numpy-fp32 = deterministic bit-identical parity oracle.
- *EV `[in-tree]`:* the **~17× grouped-backward** is the dominant reported wall-clock lever (per git-log: forgetting the env "silently ground slow on the reference path" — a launch footgun now defaulted ON); the fused-R + `mx.compile` compound it. Determinism role: numpy-fp32 oracle = the authority that keeps every fast path bit-identical (deterministic-reproducibility non-negotiable).
- *OSS delta (my contribution):* essentially none untried here — this is a bespoke substrate. The only literature-adjacent note: `mx.fast.metal_kernel` + custom VJP is the MLX analog of Triton/CUTLASS fused kernels; no OSS INR-specific kernel is more transferable than what's already built. **Verify** the launch-gate throughput assertion (`custom_grouped_backward active=true` log line) per the launch-gate memo — that's the shortest-train guard, not a new lever.

### 8b. se(3) LIE ENGINE (`tac.lie`, #193) = the TEMPORAL/POSE FACTOR `[in-tree, SEALED]`
- *What we ship:* so3/se3 exp/log/Adjoint/J_r, dual-quaternion screw-blend, cumulative SE(3) B-spline ξ_ego(t), numpy-fp64→fp32→MLX 3-tier parity (49 tests SEALED); screw-warp reported REACH-GREEN through R (#190).
- *Role in the unified action:* the **screw advects the separatrices** (d_pose = twist ξ); **canonicalize-to-ground-frame removes the screw gauge → the residual code collapses** (the rate win). Pose rides ~free / dual-use with the per-class warp (per `project_gr_unified_action_...` + grok-confirmed stratified per-class warp).
- *EV `[in-tree]`:* pose ~free/dual-use (d_seg modulation via the stratified per-class warp); the ground-frame gauge-canonicalization is the reported **rate-collapse** mechanism (fewer residual bytes once the ego screw is factored out). Not re-measured here.
- *OSS delta (my contribution):* the reference literature for the continuous-time SE(3) B-spline is **Spline Fusion / cumulative-B-spline on SE(3)** (Lovegrove et al., "Spline Fusion: A continuous-time representation for visual-inertial fusion with application to rolling-shutter cameras," BMVC 2013 — ⚠ no arXiv, conference paper) and BARF-style pose-refinement (`arXiv:2104.06405`, already cited) — both are what `tac.lie` already instantiates, so no untried delta; cite them as the canonical grounding. For gauge-canonicalization → MDL rate collapse, the transferable framing is the standard "remove the group action / canonical frame before coding" (aligns with RECOMBINER's reparameterization 6b).

## GLOBAL TOP-8 (ranked by lowest-S × shortest-train, composition with our stack, integration cost, EV/uncertainty)

| # | Lever | arXiv / repo | Primary axis | EV [HYPOTHESIS] | Integration cost | Maturity |
|---|---|---|---|---|---|---|
| 1 | **Boundary loss (SDF contour integral)** | `1812.07032` / LIVIAETS/surface-loss | d_seg | Δd_seg −15..−35%; R-robust; SDF-native fit | LOW (loss term, reuse SDF) | measured-elsewhere |
| 2 | **CURVELET SCALING CURRICULUM** (coarse→fine `max-bank-freq`) | `2303.07418` FreeNeRF / `2104.06405` BARF | d_seg + train-time | Δd_seg −5..−15% **and** train −20..−40%; Δrate 0 | LOW (schedule on existing basis) | untested-derivable |
| 3 | **FINER activation** (finish the UNSWEPT activation sweep) | `2312.02434` / liuzhen0212/FINER | d_seg + train-time | Δd_seg −10..−25% @ equal param; train −15..−30% | LOW (activation swap) | measured-elsewhere |
| 4 | **NVRC end-to-end entropy model** (counted residual) | `2409.07414` | rate (binding) | Δrate −15..−30% of residual bytes | MED (co-trained entropy model) | measured-elsewhere |
| 5 | **Aurora / NAMO — Muon finisher upgrade** | Aurora (tilde-research/aurora-release, ⚠no arXiv) / **`2602.17080`** | train-time + capacity | Δd_seg −5..−20% via recovered capacity | LOW (drop-in Muon) | Aurora: HIGH-uncertainty; NAMO: arXiv |
| 6 | **clDice** (lane-marking connectivity, class-1) | `2003.07311` / jocpae/clDice | d_seg (lane orbit) | −20..−40% of lane component (−4..−8% total) | LOW (class-1 aux term) | measured-elsewhere |
| 7 | **Focal loss** (annulus gradient / slow-tail) | `1708.02002` | d_seg + train-time | keeps gradient on small-margin annulus; ~free | VERY LOW (drop-in) | measured-elsewhere |
| 8 | **Meta-init warm-start from same-rig comma10k** | `2112.04267` Strümpler / `2012.02189` Tancik | train-time (campaign) | train −40..−70% per-clip fit | MED-HIGH (meta-train pass) | measured-elsewhere |

**Just-missed (strong, second wave):** ⏱ **knee/critical-slowing early-stop** (`2103.02646` + kneedle/`kneed`) — if ranked by **ΔS-per-train-time alone this is #1** (near-free −30..−60% wall-clock); kept out of the top-8 because it lowers *train-time* not *S* directly · SGDR warm-restarts `1608.03983` (stage-transition re-treat) · RECOMBINER `2309.17182` (low-bitrate rate arm) · WIRE `2301.05187` (Gabor activation, curvelet-synergy) · muP `2203.03466` (campaign LR-transfer) · Topology/Betti loss `1906.05404`/`2407.04683` (lane topology) · MaskCRT `2312.15829` + CondResidual `2307.12864` (soft-mask residual allocation) · UNIWARD⊙margin placement prior · RMI `1910.12037` · Schedule-Free `2405.15682`.

---

## The two singles requested

- **Single best d_seg-loss lever → Boundary loss (Kervadec, `arXiv:1812.07032`, repo `github.com/LIVIAETS/surface-loss`).**
  Rationale: it is *literally a signed-distance-function integral over the boundary* — the **same level-set object our witness already computes** (eikonal/length on the SDF), so zero conceptual impedance; it directly targets the **codim-1 boundary annulus that IS d_seg**; it is designed for **highly-unbalanced thin structures** (the 19% lane-marking hard orbit); and it is **R-robust** (soft region integral, not a fragile edge-detector, so it survives the bicubic↑/uint8-STE/bilinear↓ round-trip). Deploy as a CE→boundary curriculum stage (α annealed 1→~0.01), with **clDice (`2003.07311`)** as the lane-connectivity companion and **focal (`1708.02002`)** as the free annulus-gradient warmup.

- **Single best faster-train lever → Muon-finisher upgrade: Aurora (repo `tilde-research/aurora-release`) with NAMO (`arXiv:2602.17080`) as the arXiv-backed fallback.**
  Rationale: it upgrades the *exact* optimizer that is "THE drop," at ~6% overhead, by **fixing Muon's dead-neuron problem — which for our capacity-limited tiny INR is not a side-effect but the binding trilemma constraint** (dead neurons = wasted capacity = the thing standing between us and adequate-d_seg-at-low-rate). Drop-in, lowest integration cost. **NO-FAKE flag:** Aurora is very new (2026-05), blog+repo only, no peer review, and designed for large transformer MLPs — **gate it behind a $0 CPU smoke** (measure dead-neuron fraction Muon vs Aurora on the witness trunk) before trusting the EV; if it disappoints, NAMO (`2602.17080`) is the peer-reviewable Muon+Adam alternative.
  **Two honest caveats on "single best," since shortest-train is an explicit objective:** (1) if the metric is **pure ΔS-per-wall-clock**, the winner is instead the **knee/critical-slowing early-stop (5d, `2103.02646`+kneedle)** — it buys −30..−60% wall-clock at ~0 d_seg cost, which no optimizer swap matches; I rank Aurora/NAMO as the best *quality-preserving finisher upgrade* and knee-stop as the best *near-free wall-clock cut* — **adopt both, they're orthogonal.** (2) for *campaign-wide* wall-clock across many A/B arms, **muP LR-transfer (`2203.03466`)** stacks on top by killing redundant LR sweeps.

---

## Composition / synergy notes with the current stack (for #201's synthesis)

- **Loss ⊕ basis ⊕ curriculum arrive at the annulus together:** the curvelet scaling curriculum (#2) unmasks the finest scales *late*; schedule that unmask epoch to coincide with turning on boundary loss (#1) + clDice (#6) + focal (#7). Capacity (fine curvelet scales) and gradient (boundary/topology loss) hit the boundary/dash long-tail simultaneously → compounding, not additive.
- **Activation ⊕ feature-basis are the same wavelet family:** WIRE/FINER (activation-side space-frequency locality) × curvelet basis (feature-side directional locality) — sweep them as ONE arm; don't double-count.
- **Optimizer ⊕ capacity:** Aurora/NAMO recover dead-neuron capacity — this is *multiplicative* with the trilemma resolution (more effective capacity at equal bytes = the whole witness thesis). Measure dead-neuron fraction as an observable.
- **Rate ⊕ residual placement:** NVRC/RECOMBINER (entropy model) × MaskCRT soft-mask × UNIWARD⊙margin prior all answer "spend counted bytes only where the flip needs it" from three sides (entropy / conditional-mask / steg-cost). The KKT waterfill in `boundary_routing.py` is the natural fusion point.
- **Meta-init multiplies everything:** warm-start (#8) cuts every A/B arm's cost → raises the ΔS-per-wall-clock of every other lever; highest-leverage infra bet if the campaign is long.
- **NO-FAKE compliance boundary (rate levers):** every learned/meta-init/video-derived weight that SHIPS is COUNTED (rule 118). The compliant pattern for #4/#6-rate/#8: keep the generic algorithm (entropy model forward, meta-init generator, deterministic curvelet/curriculum) in `inflate.py` (free); ship only the clip-specific sufficient statistic in `archive.zip`. Do not smuggle video-derived tables into "code."

## Suggested $0 next actions (measurement-first, no GPU)
1. **Finish the UNSWEPT activation sweep** on CPU/MLX: Gauss/step (have) vs FINER vs WIRE at fixed param — smallest build, directly tests the #3/area-3 hypothesis. `[advisory]`
2. **CPU smoke the curvelet scaling curriculum** (#2): static `max-bank-freq` vs linear coarse→fine ramp, same seed — measure d_seg + epochs-to-target. Free-in-inflate, so no rate risk.
3. **$0 dead-neuron smoke** (#5): Muon vs Aurora/NAMO dead-neuron fraction on the witness trunk — gates the faster-train pick before any GPU.
4. **Add boundary loss (#1) as a curriculum stage** and measure the annulus flip-rate delta on the cached GT argmax (`gt_n96.npz`) — CPU-feasible on n96.

All four are byte-close-adjacent and produce a *measured* d_seg/train-time row (not another interpretation) per the ANTI-SIGNAL-LOSS "measurement-first" rule. None asserts a score; pointer stays 0.19110 until an exact byte-closed row moves it.

---

## Verification Appendix (NO-FAKE — every citation web-verified this session 2026-07-01)

| Claim | arXiv | Repo | Venue | Verified |
|---|---|---|---|---|
| Boundary loss (Kervadec) | 1812.07032 | LIVIAETS/surface-loss | MIDL 2019 (best paper) | ✓ |
| clDice (Shit) | 2003.07311 | jocpae/clDice (+MONAI) | CVPR 2021 | ✓ |
| Region Mutual Information (Zhao) | 1910.12037 | ZJULearning/RMI | NeurIPS 2019 | ✓ |
| Focal loss (Lin) | 1708.02002 | — (widely impl.) | ICCV 2017 | ✓ (well-established) |
| Lovász-Softmax (Berman) | 1705.08790 | bermanmaxim/LovaszSoftmax | CVPR 2018 | ✓ |
| Active Boundary Loss (Wang) | 2102.02696 | wangchi95/active-boundary-loss | AAAI 2022 | ✓ |
| Topology-Preserving Seg (Hu) | 1906.05404 (+1910.01877) | HuXiaoling/TopoLoss | NeurIPS 2019 | ✓ |
| Efficient Betti matching | 2407.04683 | (see paper) | 2024 | ✓ |
| Spatial-aware persistence match | 2412.02076 | (see paper) | 2024 | ✓ |
| IGR eikonal (Gropp) | 2002.10099 | amosgropp/IGR | ICML 2020 | ✓ (we already ship eikonal) |
| FINER (Liu) | 2312.02434 | liuzhen0212/FINER | CVPR 2024 | ✓ |
| WIRE (Saragadam) | 2301.05187 | vishwa91/wire | CVPR 2023 | ✓ |
| Gauss "Beyond Periodicity" (Ramasinghe) | 2111.15135 | — | ECCV 2022 | ✓ (we have Gauss in-code) |
| Sampling-theory activations | 2402.05427 | (see paper) | 2024 | ✓ |
| SIREN (Sitzmann) | 2006.09661 | vsitzmann/siren | NeurIPS 2020 | ✓ |
| Fourier Features (Tancik) | 2006.10739 | tancik/fourier-feature-networks | NeurIPS 2020 | ✓ |
| Aurora (Tilde) | ⚠ NONE (blog+repo) | tilde-research/aurora-release | blog 2026-05 | ✓ repo; ⚠ no arXiv/peer-review |
| NAMO / Adam-improves-Muon (Schaeffer) | 2602.17080 | (see paper) | Feb 2026 | ✓ |
| muP / Tensor Programs V (Yang) | 2203.03466 (+u-µP 2407.17465) | microsoft/mup | NeurIPS 2021 | ✓ |
| Schedule-Free (Defazio) | 2405.15682 | facebookresearch/schedule_free | NeurIPS 2024 | ✓ |
| Sophia (Liu) | 2305.14342 | Liuhong99/Sophia | ICLR 2024 | ✓ |
| MD-Decoupling (magnitude-direction) | ⚠ 2606.25971 (verify exact id/EPFL) | (verify) | 2026 | partial — flag |
| FreeNeRF (Yang) | 2303.07418 | Jiawei-Yang/FreeNeRF | CVPR 2023 | ✓ |
| BARF (Lin) | 2104.06405 | chenhsuanlin/bundle-adjusting-NeRF | ICCV 2021 | ✓ |
| Strümpler INR compression | 2112.04267 | (EPFL CVLab project page) | ECCV 2022 | ✓ paper; repo via project page |
| Learned inits (Tancik) | 2012.02189 | tancik/learnit | CVPR 2021 | ✓ |
| SWA (Izmailov) | 1803.05407 | (in torch.optim.swa_utils) | UAI 2018 | ✓ (we already use EMA) |
| SGDR warm restarts (Loshchilov) | 1608.03983 | loshchil/SGDR | ICLR 2017 | ✓ |
| Critical slowing in RD (Agmon/Tishby) | 2103.02646 (+Part I RG 349786895) | — | 2021 | ✓ |
| Info-bottleneck dynamics (Tishby) | 1503.02406 | — | 2015 | ✓ |
| Kneedle knee detection (Satopää) | — (ICDCS-W 2011) | arvkevi/kneed (pypi `kneed`) | 2011 | ✓ paper+repo; no arXiv |
| Spline Fusion continuous-time SE(3) (Lovegrove) | ⚠ NONE (BMVC 2013) | — | BMVC 2013 | ✓ paper; grounds in-tree tac.lie #193 |
| NVRC (Kwan) | 2409.07414 | (see paper/OpenReview) | NeurIPS 2024 | ✓ |
| RECOMBINER (He) | 2309.17182 | cambridge-mlg/RECOMBINER | ICLR 2024 | ✓ |
| MaskCRT (Chen) | 2312.15829 | NYCU-MAPL/MaskCRT | TCSVT | ✓ |
| Conditional Residual Coding (Brand) | 2307.12864 | (see paper) | 2023 | ✓ |
| Scalable coding humans+machines | 2305.02562 | (see paper) | 2023 | ✓ |
| S-UNIWARD (Holub/Fridrich) | ⚠ NONE (journal) | (widely reimpl.) | EURASIP JIS 2014 | ✓ paper; ⚠ no arXiv |

**Uncertainty flags (NO-FAKE):** (1) **Aurora** — no arXiv/peer review, marketing claims ("100×") not credible at face value; the real, defensible claim is ~6% overhead + dead-neuron fix; gate behind a $0 smoke; NAMO `2602.17080` is the citeable fallback. (2) **MD-Decoupling** exact arXiv id (`2606.25971`?) and EPFL attribution NOT fully confirmed — verify before citing in a landing. (3) **UNIWARD / HILL** are journal papers (no arXiv) — real and canonical but cite the journal. (4) **All EV numbers are HYPOTHESES** derived from each paper's domain result, not measured on our SegNet/PoseNet — every one requires a byte-closed measured row before any adopt/kill per CLAUDE.md NO-FAKE + ANTI-SIGNAL-LOSS.
