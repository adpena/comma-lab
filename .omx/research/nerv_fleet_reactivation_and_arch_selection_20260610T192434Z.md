<!-- SPDX-License-Identifier: MIT -->
# NeRV fleet reactivation (the inert-loop lens) + smaller-basis architecture selection — Task #68

**UTC:** 2026-06-10T19:24:34Z · **Subagent:** `task68_nerv_fleet_reactivation` · **Mode:** DESIGN/audit only
(no training, no dispatch, no /tmp, no MPS, $0 spend, no code landed).
**Evidence grade:** every score below is `[macOS-MLX research-signal]` / `[macOS-CPU advisory]` false-authority
per Catalog #192/#127/#323/#341 — NONE is a contest score. `promotable=false`, `score_claim=false`,
`score_roadmap_update_eligible=false`, `mechanism_update_eligible=true`. Per CLAUDE.md "Forbidden premature
KILL" + Catalog #307: nothing here KILLS a paradigm; the verdicts are IMPLEMENTATION-LEVEL classifications
that direct the next experiment. **External (paper) claims are kept EXTERNAL** (tagged `[external:arXiv …]`)
and are NEVER our score.

---

## LEAD — THE TWO ANSWERS THE TASK DEMANDS

**(1) Which substrates reactivate post-#76-loop-fix?** The d_seg≈0.50–0.71 plateau across the NeRV fleet is
**NOT 30+ independent paradigm walls — it is ONE shared bug with a TWO-PART root cause**, and #69's
all-vehicles fidelity review states it verbatim: *"The lab's entire NeRV-family fleet is, in its
default/trained configuration, the SAME skip-free [vanilla NeRV]… the shared MLX harness optimizes a
reconstruction-MSE base term which rewards that mean-field"* (`snerv_all_vehicles_fidelity_review…:21,25`).
The two orthogonal shared mistakes (memo line 193): **(M-loss) the shared MLX harness
(`_shared/mlx_score_aware`, `bundle.py`) defaults every SegNet/PoseNet weight to 0.0 → "score-aware" runs
trained recon-MSE-only, scorer-blind** (the #75 INERT-LOOP / Vehicle-OS Mistake-B), and **(M-arch) the
default decoder is a skip-free PixelShuffle+sin NeRV mislabeled HiNeRV — missing the bilinear-skip + refine
HF residual the optimizer needs to escape the mean-field**. Both must be fixed by #76; the substrates that
reactivate are the ones whose ONLY blocker is one or both of these shared mistakes (not a unique
architectural/export wall). **The prioritized reactivation list is §2.** The decisive proof these are bugs
not walls: the one-pair RGB overfit (§1.3) — the carrier cannot even memorize ONE pair (21.2 dB, d_seg
0.507, grad-norm 5.6e9 at `blocks.0`) → ill-conditioned skip-free optimization, not under-training, not a
representational-capacity wall.

**(2) The single best smaller-basis architecture:** **a score-aware-retrained ~80–120K decoder built as
`E-NeRV-disentangled backbone (smaller params, 8× faster convergence) + PR95/SNeRV bilinear-skip & residual
HF path (escapes mean-field) + FFNeRV 1D-temporal-grid flow-guided frame aggregation (free temporal model,
exploits dashcam redundancy) + a fixed-PRNG-codebook VQ index carrier on the retrained weights (the #67
free-inflate fusion: codebook FREE, indices BUDGETED)`** — with **VQ-NeRV's residual-token block as the
secondary rate lever, NOT the base**. Predicted budgeted bytes: **decoder ~25–55 KB (the #67 §3
free-decoder-conditional intrinsic-dimension band) + latents ~15 KB + flow/pose/selector ~2–4 KB → archive
~42–74 KB** vs the 177 KB frontier → **ΔS_rate −0.068 to −0.090 IF the retrained smaller net holds
d_seg≈5.6e-4 + d_pose≈2.9e-5**. The free-native-decode plan is §4. This is exactly the convergent next step
all six no-moves in the pointer ledger point to. Feeds **#76 (the working loop)** and **#74 (distillation)**.

---

## PART 1 — THE INERT-LOOP LENS: the elephant infected the SHARED HARNESS, so it infected ALL substrate work

### 1.1 The shared root cause (three independent receipts, same signature)

| substrate | trained config | d_seg | d_pose | render | receipt |
|---|---|---|---|---|---|
| **B1 clean-PR95 HiNeRV** (229K, full 8-stage) | score-aware curriculum names BUT loss_seg drifts UP 1.16→1.61, grad_norm 5e4→6.8e6 hard-clipped 100% of steps | **0.5048 pinned ep250→3000** | 157.7 | mean-field | `pr95_elephant_audit_20260610` (#75) |
| **pact_nerv_vq** (QAT4, full-600) | real SegNet+PoseNet teachers BOUND (pose proxy 3.996 ≠ mock; Catalog #322 passed) | **0.506** | 163.08 | "dark mean-field image, not a road-scene renderer" | `pact_nerv_vq_maturity_audit_20260609` §4 |
| **SNeRV path-B** (official conv MFU/HFR/TUB, ep22399) | shared MLX harness, `observed_segnet_distillation_weight=None` → recon-MSE-only | **0.7115** | 163.19 | mean-field (+ skip_high collapsed to per-frame mean) | `snerv_fullstack_extreme_scrutiny_20260609` B1 |

**The convergent fact:** three structurally DIFFERENT carriers (vanilla-skip-free NeRV / per-pair-VQ /
official-conv-U-Net-WITH-skip) all land at d_seg≈0.5–0.71 with mean-field renders. A signature this stable
across architectures is a SHARED-INFRASTRUCTURE bug, not three coincident paradigm walls. The shared
infrastructure is `src/tac/substrates/_shared/mlx_score_aware` (`bundle.py:519,527,540,541` all default
scorer weights `=0.0`; `loss.py` base term = recon-MSE `recon_weight=1.0`). Note: two partial harness fixes
DID land (`mlx_harness_scorer_binding_fix_landed_20260527`, `mlx_score_aware_per_axis_decomposition_gap_fix_landed_20260528`)
but the B1 (2026-06-09) and SNeRV ep22399 runs still show the defect — the recipe-layer weights were still
0.0 and/or the runs predate the fix. **#76 must verify nonzero scorer weights AT THE RECIPE LAYER**
(`check_score_aware_run_has_nonzero_scorer_objective_weights`) AND descent of `loss_seg` AND exact d_seg drop
— not just the helper-API fix.

### 1.2 The TWO-part root cause (orthogonal; both must be fixed)

- **(M-loss) INERT objective.** recon-MSE base REWARDS the mean-field (MSE minimizer = conditional-mean
  blur); scorer weights OFF means the SegNet-argmax-margin signal that PR95 makes primary is absent or a
  learnable-student-head surrogate that closes its own KL while the FROZEN SegNet still sees one class
  (`deep_hinerv_snerv_fidelity_review…` §3 H3 — the surrogate-vs-authority gradient gap).
- **(M-arch) skip-free decoder.** PR95's winning decoder is `sin(PixelShuffle(conv(x)) + bilinear_up(x))`
  per block **+ terminal `x + 0.1·sin(refine(x))` dilated-conv HF residual**. Our default `_UpBlockMLX` is
  `pixel_shuffle(sin(w·conv(x)))` — **NO skip, NO refine** (`mlx_renderer.py:561-577` vs `model.py:46-51`).
  The HiNeRV-defining grid PE + ConvNeXt exist but are OFF (`architecture.py:136-140`). The result is the
  blurry mean-field that collapses SegNet to one class (`deep_hinerv_snerv_fidelity_review…` §2 H1/H2,
  ranked TOP).

These are **orthogonal**: SNeRV path-B HAS the residual skip (`OfficialResidualBlockNoBN` =
"the residual HF path the rest of the fleet LACKS", `official_mfu.py:317-324`) yet STILL hit d_seg 0.71 —
because it was trained scorer-blind (M-loss alone is sufficient to ruin it). HiNeRV has the right objective
SHAPE in its PyTorch loss (`score_aware_loss.py:84-109`, no recon base) yet hit d_seg 0.51 — because the
MLX run went through the shared harness AND the decoder is skip-free (M-arch alone is sufficient). **Fixing
only one is insufficient; #76 must fix both.**

### 1.3 THE DECISIVE BUG-vs-WALL DISCRIMINATOR — the one-pair overfit (`hi_nerv_one_pair_rgb_overfit_20260609.json`)

Overfitting ONE pair for 500 epochs with PURE RGB-MSE (the easiest possible task — no generalization, no
scorer, no byte pressure):
- best frame-1 PSNR **21.24 dB** (plateau — a working INR memorizes one frame to 40+ dB).
- overfit-scorer **d_seg 0.5069, d_pose 154.5**; `segnet_comp_class_hist = [0,0,196608,0,0]` → **SegNet
  argmax is 100% one class** (the literal mean-field collapse).
- `naive_baselines_d_seg`: `frame0_copy = 0.0084`, `mean_frame = 0.507`, `black = 0.507`,
  `source_identity = 0.0`. **d_seg 0.507 == the black/mean-frame baseline** — the render carries no class
  structure. (Note: `frame0_copy = 0.008` proves a trivial COPY would crush d_seg — the carrier renders
  WORSE than copying the input frame.)
- `final_grad_norm_by_group`: `blocks.0 = 5.6e9`, `latents_coarse = 7.8e9`, `latent_embed = 1.3e9` at the
  bottom, decaying to `blocks.6 = 8.5e-5` at the top. **Textbook ill-conditioned skip-free deep network:**
  exploding gradients at early layers, vanishing at late layers — exactly what bilinear-skip residual
  connections fix (gradients reach early layers without traversing every `sin`).

**This single artifact disposes of under-training, capacity, and objective-shape as the PRIMARY cause:** it
is pure-MSE on one pair and STILL collapses, with a gradient pathology that is purely architectural
(M-arch). The mean-field is reached EARLY and the network STALLS — more epochs do not help (the full-run
trace is flat ep250→3000). **The carrier is bug-broken, not paradigm-walled.**

### 1.4 Catalog #307 classification: this is REACTIVATION TERRITORY

Per CLAUDE.md "Forbidden premature KILL" + Catalog #307 paradigm-vs-implementation: every NeRV-fleet
KILL/DEFER whose recorded verdict cites d_seg≈0.5 / mean-field / grad-pathology / loss-not-descending was a
verdict on the **broken shared harness + skip-free decoder (the IMPLEMENTATION)**, NOT on the paradigm. The
paradigms are intact. The fix (#76 working loop: bilinear-skip + refine residual + nonzero frozen-scorer
margin loss + sane grad-conditioning) reactivates them.

---

## PART 2 — THE PRIORITIZED INERT-LOOP-VICTIM REACTIVATION LIST

Classification rule (with EVIDENCE): **INERT-LOOP VICTIM** = its blocking verdict is the d_seg≈0.5 plateau /
mean-field / grad-pathology / recon-MSE-base / scorer-weight-off, with NO unique architectural/export wall →
reactivates post-#76. **REAL PARADIGM WALL** = a genuine blocker independent of the loop (export-grammar
gap, representational mismatch with the contest geometry, counting-bound impossibility).

### Tier 1 — REACTIVATE FIRST (highest-readiness victims; the fix directly unblocks them)

| # | substrate / lane | classification | evidence | what reactivates it |
|---|---|---|---|---|
| 1 | **hi_nerv** (`lane_substrate_hi_nerv_20260512`, B1 clean run) | **INERT-LOOP VICTIM (both M-loss + M-arch)** | one-pair overfit grad-pathology (§1.3); B1 loss_seg drifts UP, grad 6.8e6 clipped 100% (#75); skip + refine + grid-PE all absent/OFF | #76: add bilinear-skip + refine residual (≈10–20 LOC) + nonzero frozen-SegNet margin loss + grad-conditioning. **This IS the #76 carrier** — fix it first; it is the most-instrumented and the active frontier-fitting carrier. |
| 2 | **pact_nerv_vq** (`lane_pact_nerv_vq_l1_long_run_mlx_local_20260528`) | **INERT-LOOP VICTIM (M-loss + M-arch), rate-axis ALREADY SOLVED** | maturity audit §4: real teachers bound (Catalog #322 passed), d_seg 0.506, "dark mean-field"; **rate SOLVED — receiver-proven 22–34 KB** | The codebook MACHINERY (`VectorQuantizerEMA` K=512 + PVQ archive grammar) is production-grade and tested — REUSE IT. But its DECODER inherits both shared mistakes → fix per #76. **Critically: pact_nerv_vq is NOT the residual-VQ object** (no residual term, audit §1,§5) → reactivate as the codebook/codec LAYER on the fixed hi_nerv carrier (the §3 fusion), not as a standalone primary carrier (its own 2026-06-02 codex audit says `pivot_or_rebuild_vq_before_more_long_run_spend`). |
| 3 | **snerv** (`lane_snerv_scorer_loop_decoder_qat_*`, path-B official conv) | **PARTIAL VICTIM (M-loss) + REAL but NARROW wall (export binding B3)** | scrutiny: arch REAL-AND-COMPLETE (~70%), the conv path HAS the residual skip; failure is "wrong loss path trained the right architecture" (B1) | #76 fix (M-loss): route the real-frozen-scorer objective to path-B conv (config-only — the knobs exist, defaulted 0.0). **REMAINING WALL (not loop): B3** — the at-scale MLX conv weights are not bound to the byte-closed official payload (`carrier.py:312-316`). So snerv needs #76 (loop) AND a separate export-binding landing before promotion. Its skip_high MUST stay `full` (NOT channel_mean/scalar_mean which collapse the finest skip — S2). |

### Tier 2 — REACTIVATE AFTER the carrier proof (L0/L1 SKETCH victims; the fix is a precondition, not the whole job)

| # | substrate / lane | classification | evidence | note |
|---|---|---|---|---|
| 4 | **e_nerv** (`lane_substrate_e_nerv_20260512`, `lane_e_nerv_l0_scaffold_20260520`) | **INERT-LOOP VICTIM-IN-WAITING** (L0/L1 SKETCH, never trained to a real anchor) | registry L0/L1, `_full_main` NotImplementedError per Catalog #240; no contest anchor | E-NeRV's disentangled spatial/temporal context = **8× faster convergence + fewer params** `[external:arXiv 2207.08132]` — directly relevant to a SMALLER-budget carrier. **This is a primary ingredient of the §3 recommended architecture**, not just a victim to revive. |
| 5 | **ff_nerv** (`lane_substrate_ff_nerv_20260512`) | **INERT-LOOP VICTIM-IN-WAITING** (L1 SKETCH) | registry L1 SKETCH, research_only | FFNeRV flow-guided aggregation exploits dashcam temporal redundancy `[external:arXiv 2212.12294]` — **the free temporal-model ingredient of §3**. Reactivate as a component, not a standalone. |
| 6 | **boost_nerv** (`lane_boost_nerv_l0_scaffold_20260520`) | **VICTIM-IN-WAITING + UNIVERSAL ENHANCER** | L0 SCAFFOLD, `_full_main` NotImplementedError | BoostNeRV is a *universal boosting framework* (conditional decoder + temporal-aware affine + sinusoidal block + entropy-min) that improves ANY INR's quality+convergence `[external:arXiv 2402.18152, CVPR'24 Highlight]`. **Compose it onto the §3 winner as a convergence/quality multiplier** — it is NOT a competing base. |
| 7 | **ds_nerv / nervdc / ego_nerv / nirvana / coin_plus_plus** (`lane_substrate_*_20260512`/`_l0_scaffold_20260520`) | **VICTIM-IN-WAITING** (L0/L1 SKETCH) | all `_full_main` NotImplementedError, research_only, no anchor | Lower priority — depth-separable / decoder-conditioning / ego-pose / patch-wise / FiLM-modulation variants. Each is a candidate AFTER the §3 winner proves the loop; pick at most one to test the orthogonal trick once the carrier holds the score. |

### Tier 3 — NOT pure inert-loop victims (real or different blockers; do NOT auto-reactivate on #76)

| substrate | classification | evidence |
|---|---|---|
| **balle_renderer / NSCS03 e2e Ballé joint codec** | **DIFFERENT PARADIGM (hyperprior nonlinear-transform codec), partial-victim** | `balle_compressai_byte_closure_audit`; these are end-to-end learned codecs, not NeRV memorizers — they share M-loss if trained through the harness but their export/rate story is distinct. Re-audit separately. |
| **cool_chic / c3 / siren / wavelet residual sidecars** (`lane_*_residual_pr106_sidecar_*`) | **NOT primary carriers — residual sidecars over PR106** | these are bolt-on residual bases over a FROZEN PR106 base (`lane_cool_chic_residual_scaffold` etc.); #73 proved a generic basis needs ≥625 KB/pair → the sidecar geometry is closed on frozen weights. NOT reactivated by the loop fix; they need the §3 retrained base to compose against. |
| **hi_nerv target-region-birth v30–v38** (`hi_nerv_target_region_birth_*`) | **DIFFERENT lever (charged-action-payload birth), separate blocker** | the parse-back collapse chain (`hinerv_collapse_axis_falsification_chain`) is an EXPORT/sidecar-wrap bug (int4 codec collapse + sidecar inversion), not the training inert-loop. Distinct work-stream. |

**Reactivation discipline (per "Forbidden premature KILL"):** none of the above is KILLED; Tier-1 are
DEFERRED-pending-#76-loop-fix; the reactivation criterion is uniform: *#76 lands a loop where `loss_seg`
descends AND the live-render exact d_seg drops below ~0.10 at a checkpoint* (the #75 decisive re-probe).
The moment that anchor exists, Tier-1 substrates are L1→L2 promotion candidates on the contest axis.

---

## PART 3 — SMALLER-BASIS ARCHITECTURE SELECTION (the ranked recommendation)

### 3.0 The constraint

Hold **d_seg ≈ 5.6e-4 + d_pose ≈ 2.9e-5** (the frontier distortion cell) at **minimal BUDGETED bytes**, with a
**free native (Rust/Zig) decode kernel**. The #67 free-decoder-conditional intrinsic-dimension band is
**~24.6–64.6 KB** (rate 0.016–0.043) — the mathematical license that a smaller score-aware-retrained
amortizer CAN reach ~2.7–7× below the 177 KB frontier. The lever is **score-domain RETRAINING**, never a
post-hoc transform of frozen weights (#64/#71/#72/#73/#54/#67 — six no-moves all confirm post-hoc is closed).

### 3.1 Per-architecture relevance (external claims kept EXTERNAL)

| arch | core mechanism `[external]` | relevance to OUR constraint | role |
|---|---|---|---|
| **E-NeRV** `[arXiv 2207.08132, ECCV'22]` | disentangles the coupled NeRV into separate spatial + temporal context → "more than 8× faster convergence" + "greatly reduces redundant model parameters while retaining representation ability" | **SMALLER params + faster fit** = the base-capacity lever. Faster convergence directly helps a fixed-budget training campaign reach the cell. | **BASE backbone** |
| **PR95 / SNeRV residual skip** (in-repo, proven 0.193) | `sin(PS(conv)+bilinear_up) + refine HF residual`; SNeRV `OfficialResidualBlockNoBN` | **escapes the M-arch mean-field** — the single binding fix from §1. Without it any base collapses. | **MANDATORY HF path** |
| **FFNeRV** `[arXiv 2212.12294, ACM-MM'23]` | optical-flow-guided frame aggregation (reuse pixels from neighbor frames; static regions weight aggregated frames, moving regions weight independent) + 1D temporal grids → fully-conv | dashcam video is HIGHLY temporally redundant (ego-motion = mostly smooth global flow). Flow-guided aggregation makes per-frame deltas cheap → fewer budgeted bytes/pair. **Pose co-benefit:** the flow IS the inter-frame motion structure PoseNet reads (§5 of #69 — d_pose needs the frame_0→frame_1 delta). | **FREE temporal model** (flow generated/regularized, not stored raw) |
| **VQ-NeRV** `[arXiv 2403.12401]` | U-shaped; codebook (size 1024/2048, dim 8) discretizes **shallow residual + inter-frame residual** `(f_e−f_d)`; per-frame tokens 2×16 + embeddings 2×16×8 | the residual-token discretization IS a rate lever, but the reported gain over HNeRV is MODEST (+0.73 dB Bunny). The VALUE is the **residual + U-skip** structure (which overlaps PR95's residual) + the **discrete-index carrier** (which composes with the free-codebook exploit). | **SECONDARY rate lever** (residual token + index carrier) |
| **HNeRV / BoostNeRV / SR-NeRV** `[arXiv 2402.18152, CVPR'24 Highlight]` | BoostNeRV = universal booster (conditional decoder + temporal-affine + sinusoidal block + entropy-min) for ANY INR | a drop-in quality+convergence multiplier — composes onto the winner. | **OPTIONAL enhancer** (compose last) |

### 3.2 THE RECOMMENDATION — a fusion, not a single off-the-shelf variant

**Build the #76/#74 carrier as:**

> **`E-NeRV disentangled spatial/temporal backbone (~80–120K params, the smaller-capacity base)`**
> **`+ PR95/SNeRV bilinear-skip per block + terminal refine HF residual (M-arch fix — escapes mean-field)`**
> **`+ FFNeRV 1D-temporal-grid flow-guided aggregation (free temporal model; cheap per-pair deltas; restores`**
> **`  the two-frame pose signal)`**
> **`+ fixed-PRNG-codebook VQ index carrier on the retrained weights (#67 PATH-1: codebook FREE in inflate.py,`**
> **`  per-weight indices + tiny seed BUDGETED — the forward map is DESIGNED around the fixed codebook so the`**
> **`  indices are genuinely few, sidestepping the counting-bound that bars frozen-weight inversion)`**
> **`— trained against α·B + β·d_seg + γ·√d_pose with eval_roundtrip + differentiable-YUV6 + EMA + NONZERO`**
> **`  frozen-SegNet-margin + frozen-PoseNet pose loss (the M-loss fix), grad-conditioned (no 1e6 clip regime).`**

**Why a fusion and not "pick FFNeRV" or "pick VQ-NeRV":** the constraint is not "best generic video PSNR"
(the external benchmark) — it is "smallest BUDGETED bytes holding the SCORE cell with a free native decode."
That decomposes into four orthogonal sub-problems, each best-solved by a different family:
- **escape the mean-field** → PR95/SNeRV residual skip (binding; #1 from §1).
- **smaller base capacity + fast fit** → E-NeRV disentanglement.
- **cheap per-pair (temporal redundancy + pose signal)** → FFNeRV flow.
- **cheap weight carrier (the free-inflate exploit)** → fixed-PRNG-codebook VQ indices (#67 PATH-1).
A single off-the-shelf variant solves at most one. The frontier winners (PR95/101) ALSO fused
(architecture + score-aware training + archive grammar + curriculum) — per the canonical leaderboard
binding-depth discipline, binding ALL ingredients into ONE coherent carrier is the proven path.

### 3.3 Predicted budgeted bytes (DERIVED, falsifiable)

```
decoder (E-NeRV-small + skip, fixed-PRNG-codebook VQ indices)  : ~25–55 KB   (#67 §3 conditional-dim band)
per-pair latents (FFNeRV temporal-grid + flow)                 : ~10–15 KB   (≤ frontier's 15,070 B; flow ⊂ temporal grid is generated)
selector / pose codes / framing                                : ~2–4 KB
--------------------------------------------------------------------------------
archive total                                                  : ~42–74 KB   vs 177,169 B frontier
ΔS_rate = 6.6586e-7 · (−103,000 to −135,000)                   : −0.069 to −0.090   IF the cell is held
```
**Pre-registered KILL (per #67 G3):** if the smaller net's d_seg/d_pose cannot re-enter the tube at ANY
capacity below the frontier, the conditional floor is not reachable by this architecture class and we report
the architectural-ceiling band (NOT a paradigm kill — a measured class boundary). The #73 pose-tube + #71
joint-entanglement predict feasibility ABOVE ~25 KB, so the band's lower edge is the risk frontier.

### 3.4 Ranking of the single-variant fallbacks (if a fusion is too much for one campaign)

1. **E-NeRV + PR95-skip (no VQ, no flow)** — the minimum viable smaller carrier. Fixes both shared mistakes;
   smaller params; proven HF path. Predicted ~70–110 KB. **Start here if forced to pick one.**
2. **+ FFNeRV flow** — adds the temporal/pose lever; predicted −10–20 KB more and a d_pose co-benefit.
3. **+ fixed-PRNG-codebook VQ carrier** — adds the #67 free-inflate rate lever on the retrained weights;
   predicted the −25–55 KB decoder band.
4. **VQ-NeRV-as-base** — NOT recommended as the base: modest PSNR gain over HNeRV, and our pact_nerv_vq
   sibling already proved the per-pair-VQ-as-primary-carrier path is distortion-blocked. Use VQ as the
   carrier LAYER (rank 3), not the base.

---

## PART 4 — THE FREE-NATIVE-DECODE PLAN (Rust/Zig kernel)

Per CLAUDE.md "Native eval-time runtime discipline" + the deterministic packet compiler + #67 §1 boundary:

1. **Grammar-first (Python oracle).** Design the score-program grammar in Python first (the existing
   `tac` packet-compiler + MLX→numpy reference lineage). The native kernel is the BODY, the brain stays
   offline (training, codebook seed selection, index assignment all happen at compress-time, charged or
   free per the boundary).
2. **The free/budgeted split (the #67 G4 boundary — compliance-airtight):**
   - **FREE in inflate.py/.sh (video-INDEPENDENT):** the fixed-PRNG codebook generator `codebook = prng(fixed_seed)`,
     the E-NeRV decoder graph (un-trained architecture), the FFNeRV flow-warp / temporal-grid sampler, the
     IDWT/upsample/PixelShuffle/sin/skip kernels, brotli/lzma. These are deterministic algorithms, not
     functions of THIS video → 0 bytes.
   - **CHARGED in archive.zip (video-SPECIFIC):** the per-weight VQ indices + the tiny seed, the per-pair
     latents/flow coefficients, pose codes, selector. The codebook seed is small ONLY because the carrier
     is DESIGNED forward around the fixed codebook (NOT a frozen-weight inversion — that is counting-barred,
     #67 G2).
   - **FORBIDDEN:** baking any video-specific payload into inflate.py (the houdini class — PR#69
     eval-refused; leaderboard editorial exclusion).
3. **Native kernel scope (promote only proven hot paths).** Profile the Python inflate wall-clock against
   the 30-min budget; the hot path is the decoder forward (conv + PixelShuffle + sin + skip) + the
   flow-warp + the codebook gather. Promote those to a Rust/Zig integer-stable kernel (`runtime-rs/`)
   with: (a) a Python reference oracle; (b) the payload-cleanliness audit bundle (`binary_source_audit.md`,
   `embedded_constants_audit.txt`, `archive_payload_manifest.json`, `rebuild_instructions.md`,
   `python_reference_equivalence_test.py`); (c) a bit-identical / scorer-identical equivalence test against
   the same archive bytes (CPU and CUDA evaluated separately per apples-to-apples).
4. **Per-function promotion = flip its parity test to `assert_sha256_parity`** in the
   `runtime-rs/crates/tac-packet-compiler` gate. The codebook gather + integer round/clamp/resize/YUV-basis
   are the natural first CPU-stable promotions (they harden deterministic replay).

---

## PART 5 — SCOREBOARD CONTRIBUTION + WIRE-IN (Catalog #125)

**UPPER (vs the 0.19109982 frontier):** unchanged — this is design/audit, no archive emitted.
**LOWER (the floor):** reaffirms the #67 free-decoder-conditional band ~24.6–64.6 KB (rate 0.016–0.043) as
the prediction the §3 fusion campaign falsifies.

1. **sensitivity-map — ACTIVE.** New prior: the NeRV-fleet d_seg≈0.5 plateau is a SHARED two-part bug
   (M-loss harness-default-0.0 + M-arch skip-free), not 30+ paradigm walls; the aiming surface is the §3
   score-aware retrained E-NeRV+skip+flow+VQ fusion.
2. **Pareto — ACTIVE.** Adds the constraint: a smaller carrier MUST carry the residual HF path (PR95-skip)
   or it collapses to the mean-field at ANY capacity (the one-pair overfit proof).
3. **bit-allocator — ACTIVE for the §3 carrier.** The fixed-PRNG-codebook VQ indices are the budgeted
   surface; the codebook + decoder graph + flow sampler are FREE. No post-hoc primitive moves frozen bytes.
4. **cathedral-autopilot — N/A.** Design surface; no archive. Do NOT queue a frozen-weight materializer.
5. **continual-learning — ACTIVE.** Reseeds the V3 judge with: (a) the shared-harness inert-loop is the
   fleet-wide cause (so future agents do not KILL NeRV paradigms on d_seg≈0.5 verdicts); (b) the one-pair
   overfit grad-pathology as the bug-vs-wall discriminator; (c) the §3 fusion as the convergent
   smaller-basis recommendation; (d) the reactivation list (Tier-1 victims).
6. **probe-disambiguator — RESOLVED.** "Is the d_seg≈0.5 plateau a paradigm wall or a bug?" → BUG (shared
   harness M-loss + M-arch; one-pair overfit decisive). "Which substrates reactivate post-#76?" → Tier-1
   (hi_nerv, pact_nerv_vq codebook layer, snerv path-B+export). "Best smaller basis?" → §3 fusion.

---

## 6. FALSIFIABLE CLAIMS (append-only)

- **N1 (DERIVED, decisive):** the NeRV-fleet d_seg≈0.5–0.71 plateau is a SHARED two-part bug (M-loss
  harness-scorer-weight-default-0.0 + M-arch skip-free decoder), NOT independent paradigm walls. *Falsified
  by:* a substrate with the PR95 bilinear-skip + refine residual AND nonzero frozen-SegNet margin loss (no
  recon base) that STILL pins d_seg≈0.5 on the contest video → then it IS a representational wall.
- **N2 (MEASURED, the discriminator):** the carrier cannot memorize ONE pair (21.2 dB, d_seg 0.507,
  grad-norm 5.6e9 at blocks.0) → ill-conditioned skip-free optimization, not under-training/capacity.
  *Falsified by:* the same carrier WITH bilinear-skip reaching >35 dB + d_seg<0.05 on one pair (predicted:
  it will — that confirms M-arch).
- **N3 (DERIVED):** the §3 fusion (E-NeRV base + PR95-skip + FFNeRV flow + fixed-PRNG-codebook VQ) reaches
  archive ~42–74 KB holding the cell. *Falsified by:* an exact-eval row either below 0.043 rate at the cell
  (tightens) or a proof of infeasibility below 177 KB for any architecture (#73 forbids above ~25 KB).
- **N4 (CLASSIFICATION):** Tier-1 (hi_nerv, pact_nerv_vq-codebook-layer, snerv-path-B) reactivate on the
  #76 loop fix; snerv additionally needs the B3 export binding before promotion. *Falsified by:* a #76 loop
  with descending loss_seg + d_seg<0.10 that STILL leaves a Tier-1 substrate at d_seg≈0.5.

## 7. CROSS-REFERENCES

`pr95_elephant_audit_20260610` (#75 — the INERT loop, hyp (b) CONFIRMED) ·
`smaller_learned_basis_deep_math_20260610` (#67 — free-inflate exploit empty on frozen weights, pays as
fixed-codebook-VQ on RETRAINED weights; conditional-dim band ~25–65 KB) ·
`frontier_pointer_move_ledger_20260610` (six no-moves → retraining is the only lever) ·
`pact_nerv_vq_maturity_audit_for_codebook_investment_20260609` (VQ: rate solved, distortion blocked, NOT a
residual-VQ object, codebook machinery reusable) ·
`deep_hinerv_snerv_fidelity_review_vs_evaluate_py_20260609` (H1 missing residual + H2 grid-PE-off + H3
objective-shape; F1 = add skip+refine) ·
`snerv_fullstack_extreme_scrutiny_vs_evaluate_py_20260609` ("wrong loss path trained the right
architecture"; path-B HAS the residual skip; B3 export wall) ·
`snerv_all_vehicles_fidelity_review_vs_evaluate_py_20260609` (the two shared mistake classes span the
fleet) · `hi_nerv_one_pair_rgb_overfit_20260609.json` (the decisive bug-vs-wall discriminator) ·
`hinerv_collapse_axis_falsification_chain_20260608` (the SEPARATE target-region export/sidecar bug —
Tier-3, not the loop) · `mlx_harness_scorer_binding_fix_landed_20260527` +
`mlx_score_aware_per_axis_decomposition_gap_fix_landed_20260528` (partial harness fixes; #76 must verify at
recipe layer) · `docs/vehicle_operating_system.md:81-83` (the inactive-objective Mistake-B this confirms
fleet-wide).

**External (kept external — never our score):** E-NeRV `[arXiv 2207.08132, ECCV'22]` ·
FFNeRV `[arXiv 2212.12294, ACM-MM'23]` · VQ-NeRV `[arXiv 2403.12401]` ·
BoostNeRV `[arXiv 2402.18152, CVPR'24 Highlight]` · HiNeRV `[arXiv 2306.09818, NeurIPS'23]` ·
SNeRV `[arXiv 2501.01681]`.
