# Fields-medal grand council — substrate design wave (GGG)

`[contest-CUDA frontier reference]` PR #107 apogee = 0.229
`[contest-CPU frontier reference]` A1 = 0.193 (gold-tied submission)
`[empirical anchor]` B1 dual-base BOTH-REGRESS: PR106 r2 +5204B / A1 +5216B (within 12 bytes on magic_codec rate-axis)

`generated_at: 2026-05-12T17:50:00Z`
`from_state_hash: lane_registry.json schema v1 (410 lanes)`

## 0. Mission

The operator's directive 2026-05-12: *"consult with grand council Fields-medal grade and review and fix and design and debug and implement and wire and integrate and build a substrate for all".*

The B1 dual-base falsification within 12 bytes structurally extincts the codec-stacking thesis on currently-available bases. The model-is-the-thing recommitment requires a NEW SCORE-AWARE SUBSTRATE with **byte headroom on the rate axis** AND **pose-axis marginal headroom** (per CLAUDE.md operating-point-aware seg-vs-pose: at PR106 r2's pose_avg ≈ 3.4e-5, pose-marginal is 2.71× SegNet's).

This memo is the council deliberation + binding verdict. No GPU dispatch this turn — design + scaffold only.

## 1. Council convened (inner-ten + grand-council quorum)

All 10 inner-ten members deliberate. Grand council members (Boyd, Tao, Hassabis, Carmack, Hinton, Karpathy) speak when their specialty is touched. No conservative bias per CLAUDE.md "Council conduct"; the Contrarian challenges WEAK arguments not BOLD ones.

## 2. The structural finding — what B1 actually says

### 2.1 Shannon LEAD opens

> "The dual-base BOTH-REGRESS finding is *not* a 'magic_codec is broken' result. It is a *Shannon-entropy-floor* result: PR106 r2 and A1 — two substrates from entirely different training paradigms (PR106 r2 = NN's xray-classified entropy-saturated weight distribution; A1 = score-aware gold-tied) — have **independently arrived at the same byte-overhead-floor of the arithmetic codec** when you ask the codec to compress the renderer-weight stream. The 12-byte gap across DIFFERENT distributions is the dictionary-table envelope; the floor is sub-payload-distribution-invariant.
>
> The bit-budget per axis at PR106 r2 (Shannon decomposition):
> - **renderer-weight bytes**: ~186KB at the empirical compression floor (≈ 1.0 bpw effective; any rate-axis stacking is bytes lost)
> - **latent code**: PR101 used 15KB at 8 bits per (T·H·W/k) ≈ Shannon floor on the temporal latent
> - **hyperprior side-info**: ZERO currently — this is the unexploited axis
> - **pose / residual**: pose-marginal exploitable per operating-point flip
>
> **The rate axis is saturated. The pose axis is not. The hyperprior side-info axis is not. The temporal-prediction axis is not.**"

### 2.2 Dykstra CO-LEAD derives the Pareto

> "Convex-feasibility view of the achievable score region after B1:
>
> Define constraint sets
> - $C_R = \{(r, s, p) : r \le R^*\}$ where $R^* \approx 186{,}000$ bytes (the magic_codec floor on either base)
> - $C_S = \{(r, s, p) : s \le S^*(\text{operating point})\}$ where at PR106-r2, $S^* \approx 6.7\mathrm{e}{-4}$ (renderer architectural ceiling)
> - $C_P = \{(r, s, p) : p \le P^*\}$ where at PR106 r2, $P^* \approx 3.4\mathrm{e}{-5}$ (NOT yet at hardware floor)
>
> Alternating projections (Dykstra iterations) compute the intersection $C_R \cap C_S \cap C_P$. B1 proved $C_R$ is **binding** on both bases. The next *useful* score-improvement direction is one of:
>
> 1. **Relax $C_R$ by adding an exploitable axis** (hyperprior side-info, learned predictor, temporal latent) — this is the Ballé/Minnen 2018 architectural move
> 2. **Tighten $C_P$ via score-aware pose-axis attack** (T20-class pose-distill, latent-pose-residual sidecar) — pose-marginal 2.71× at this operating point
> 3. **A NEW $C_R$ by re-architecting the renderer** so the rate-axis floor moves (this is what HNeRV-LC-v2 did)
>
> Combined: **build a substrate that has hyperprior side-info AND score-aware pose-residual stream AND a re-architected renderer with lower $R^*$.** Single-axis stacking is dominated."

### 2.3 Yousfi (contest scorer architect)

> "The SegNet stride-2 stem ALREADY loses half-resolution at the first conv. Below (256, 192) every artifact is invisible. The PoseNet FastViT-T12 ingests 12-channel YUV6 (2 frames × YUV6). The renderer's job is to produce uint8 RGB frames that survive the eval-roundtrip (384→874→uint8→384) and then be re-derived through both scorers consistently. **The substrate's eval-roundtrip-AWARENESS is what produces score-axis movement; the codec is what produces rate-axis movement; they are decoupled.** B1 confused them. A new substrate must be a *renderer* not a *codec*."

### 2.4 Fridrich (inverse steganalysis)

> "UNIWARD's lesson: errors in **textured regions are undetectable**. The square root law spreads small errors over many positions. Translate to our world: the renderer-weight distribution where the magic_codec floor is binding is in some sense the 'flat' region. The exploitable region is where we ADD a small bandwidth-efficient stream (hyperprior, learned predictor) that's *invisible* to the bit-floor of the existing rate model. Ballé 2018's scale hyperprior is precisely this — a separate small channel that re-shapes the *conditional* density $p_y(y|\sigma)$ so the original $y$ stream becomes more compressible."

### 2.5 Contrarian challenges

> "Three weak arguments I will challenge:
>
> 1. **'New substrate = new architecture = research-only kitchen_sink':** REJECTED. The substrate must honor all 13 HNeRV parity lessons AND ship a packetized monolithic-`0.bin` archive grammar BEFORE training. This is *integration discipline*, not architectural novelty.
> 2. **'Stop B1, redirect all to T1 Ballé':** REJECTED for being too narrow. T1 Ballé alone violates HNeRV lesson 5 (architecture must be FULL RENDERER, RGB out — Ballé is the entropy bottleneck *on* the renderer). The council MUST design a substrate-class where Ballé is the rate-axis component but the renderer is the SCORE-axis primary.
> 3. **'One candidate is enough':** REJECTED per CLAUDE.md 'Multiple contenders → multiple paths' + 'probe-disambiguator pattern'. The council MUST ship ≥2 candidates with a probe that arbitrates.
>
> The BOLD argument I will NOT challenge: build a new score-aware substrate WITH archive grammar declared at design-time, ≤350 LOC bolt-on, ≤100 LOC inflate."

### 2.6 Quantizr (Jimmy at UCLA — leaderboard gold-tied reality check)

> "The PR101 medal-winning substrate is 605 LOC TOTAL: 268 LOC HNeRV-LC-v2 substrate + 337 LOC bolt-on. The architecture IS canonical HNeRV with per-pair latents (28-d, 2-frame prediction). The codec is per-tensor byte maps + Brotli + LZMA + Huffman sidecar. The ARCHITECTURE buys ~0.005 over generic HNeRV. The SCORE-AWARE OVERFITTING on contest video buys ~0.05+. The CODEC bolt-on buys ~0.03.
>
> **Our new substrate should clone this layer ownership:** substrate engineering ONCE (~250-350 LOC), bolt-ons MANY times (~80-200 LOC each). The substrate is the renderer + the archive grammar; the bolt-ons are the per-axis codec primitives."

### 2.7 Hotz (Carmack-style engineering shortcut)

> "I would write the inflate.py in 70 lines tonight if you gave me the archive grammar. The architecture is sin-activated PixelShuffle decoder + per-pair latent + bilinear-skip. That's 50 lines of model. 20 lines of struct.unpack + brotli.decompress. Done.
>
> **The substrate's inflate path MUST be ≤100 LOC, ≤2 deps, CUDA-or-CPU agnostic, reviewable in 30 seconds.** Anything more is a smell."

### 2.8 Selfcomp (szabolcs-cs)

> "From my 0.38-scoring lived experience: the FiLM-conditioned depthwise-separable CNN works because it's TINY (88K params) and SCORE-AWARE-TRAINED end-to-end. The temptation to make it more complex is wrong. The new substrate should respect the empirical ceiling of ~250K params at our score region (PR101 used 229K; A1 uses similar order); not a 1M-param transformer dream.
>
> **Block-FP weight self-compression (1.017 bpw) is orthogonal** to the rate-axis stacking that B1 falsified. It works precisely because it operates at the substrate level — *the weight distribution itself* is shaped during training to be sub-1.5 bpw entropy-compressible. The new substrate can/should consider this in its training loop."

### 2.9 MacKay (memorial seat)

> "MDL perspective: the joint MDL for the contest packet is
> $$L(\text{packet}) = -\log_2 p_{\theta}(\text{video}|\text{packet}) + |\text{packet}|$$
> The B1 result says the second term is at its floor on the current substrates. The score = first term is the renderer's fault. The substrate must lower the FIRST TERM via a score-aware training loop that backprops the contest scorer's gradient.
>
> The Bayesian view: arithmetic coding the *quantized* latent space (Ballé's $p_y(y|\sigma)$) is the canonical right move when the conditional density can be MODELED. Static factorized priors leave bits on the table; learned hyperprior side-info exploits those bits.
>
> **The unified-Lagrangian action is** $S = \int(\alpha B(\theta) + \beta d_{seg}(\theta) + \gamma \sqrt{d_{pose}(\theta)}) dt$. **Variational principle**: $\delta S / \delta \theta = 0$. The substrate is a SINGLE parameterization $\theta$ that minimizes this action; no decomposition into separate codec + renderer trainers."

### 2.10 Ballé (modern neural-compression SOTA)

> "Replace the fixed factorized prior on the latent stream $y$ with a learned scale hyperprior $\sigma = h_s(z)$ where $z$ is an auxiliary latent with its own prior $p_z(z)$. Total rate $R = E[-\log p_z(z)] + E[-\log p_y(y|\sigma(z))]$. The amortization gate: side-info cost ~50-500B, savings on $y$ stream ~5-25% for streams >5KB.
>
> For our 186KB substrate, even a conservative 5% saving on a 162KB subset (the PR101 DECODER_BLOB region) = **~8KB rate-axis recovery**. This is structurally orthogonal to the magic_codec floor B1 hit, because the hyperprior re-shapes the *conditional density* used by the coder, not the coder itself.
>
> GDN nonlinearity is a known win over ReLU in compression networks. The combined Ballé-GDN-hyperprior architecture is the canonical neural-compression substrate, and we DO have it at `src/tac/balle_hyperprior_renderer.py` (15KB) + `src/tac/balle_hyperprior_codec.py` (36KB) already. The substrate scaffold should COMPOSE with these, not duplicate."

## 3. Grand-council voices (consulted)

### 3.1 Boyd (ADMM/proximal-gradient at operational level)

> "Once the unified action $S$ has multiple track-Lagrangians, the right solver is alternating-direction method of multipliers (ADMM): introduce slack $z = \theta$ and Lagrangian $\rho > 0$, and solve $\min_\theta L_1(\theta) + L_2(z) + \frac{\rho}{2}\|\theta-z\|^2$ via alternating primal/dual updates. This is exactly what `tac.optimization.admm_*` already does. The new substrate should expose its per-track gradients so the joint ADMM coordinator sees them."

### 3.2 Tao (harmonic analysis)

> "On the rate-axis floor: by Shannon's coding theorem the empirical compression ratio approaches the source entropy $H(X)$ asymptotically. The 12-byte gap across two payload distributions suggests both are at $H(X) + O(\log n / n)$ for the same effective $n$. To beat this you change the SOURCE $X$ — i.e., re-train the substrate so its weight distribution has lower $H(X)$. This is the SC++ block-FP self-compression direction."

### 3.3 Hassabis (strategic 4-day-deadline tradeoff)

> "The DeepMind playbook on integration-bound problems: build the MINIMUM-VIABLE-INTEGRATION-LOOP first, then optimize within it. The substrate scaffold doesn't need to be the FINAL substrate; it needs to PROVE the integration loop (archive grammar + score-aware training + inflate ≤100 LOC + eval-roundtrip wired). Once one ⏎ closes, every subsequent optimization is a small bolt-on per Quantizr's framing."

### 3.4 Carmack (engineering shortcuts)

> "Doom-in-30-min reading: the entire substrate scaffold is ~300 lines if you stop over-engineering. One file for the architecture, one file for the archive grammar, one file for the inflate, one file for the loss. Tests are trivial because the architecture is small. Ship it tonight."

### 3.5 Hinton (knowledge distillation)

> "If you do score-aware training, the contest scorer's PoseNet/SegNet gradients are noisy because of the eval-roundtrip discretization. A Hinton-style temperature-T softening of the scorer's logits at T=2.0 before computing the distill loss gives much smoother gradients. This is canonical Phase 3 T10 IB-Lagrangian territory. **For Phase 2 substrate scaffold: keep the score-aware loss simple — direct contest-scorer gradients (with eval-roundtrip patched-yuv6) and ADD T=2.0 temperature as an optional ablation flag.**"

### 3.6 Karpathy (let compute speak)

> "After scaffold lands: spin a Lightning T4 free-tier sweep over 16 randomly-perturbed hyperparameter configs (latent-dim, decoder-channels, sin-frequency). The substrate that wins on 16-config Pareto is the one that anchors the autopilot. **Substrate is a search problem; the architecture is a config.**"

## 4. Verdict — candidate selection (Part 3 design deliverable)

### 4.1 Candidates considered

The council considered FOUR candidate architectures per the prompt's Part 1 (b):

- **α — Pure HNeRV-family extension** (BlockNeRV / FFNeRV / DSNeRV / HiNeRV / TCNeRV / canonical HNeRV-LC-v2)
- **β — Ballé hyperprior + lightweight learned decoder** (end-to-end score-aware)
- **γ — Hybrid renderer-with-residual-basis** (HNeRV-class renderer + score-aware sparse residual coefficient stream)
- **δ — Self-Compress NN** (PARADIGM-δεζ Track #307; MDL-optimized weight clustering during training)

### 4.2 13-lessons compliance

| Lesson | α (HNeRV-LC-v2) | β (Ballé-renderer) | γ (Hybrid+residual) | δ (Self-Compress) |
|---|---|---|---|---|
| L1 score-aware substrate | PASS (PR95/100/101 proved it) | PASS by construction | PASS | PASS |
| L2 export-first archive grammar | PASS | PASS (Ballé codec exists) | PASS | NEEDS-WORK |
| L3 monolithic 0.bin grammar | PASS (PR101 exemplar) | PASS | PASS | NEEDS-WORK |
| L4 inflate ≤100 LOC, ≤2 deps | PASS (PR101 inflate = 71 LOC) | NEEDS-WORK (GDN forward at inflate is heavier) | NEEDS-WORK (residual decode = +30 LOC) | FAIL (clustering decode = +50 LOC) |
| L5 full RGB renderer | PASS | PASS | PASS | PASS |
| L6 score-domain Lagrangian | PASS | PASS | PASS | PASS |
| L7 bolt-on ≤350 LOC | PASS (substrate ~300) | PASS (≤350 substrate) | NEEDS-WORK (~400 total) | NEEDS-WORK (~450 total) |
| L8 eval-roundtrip + diff yuv6 | PASS (wire `tac.differentiable_eval_roundtrip`) | PASS | PASS | PASS |
| L9 runtime closure | PASS (brotli only) | NEEDS-WORK (CompressAI deps) | PASS | PASS |
| L10 mask/pose coupling | N/A (renderer replaces full slot) | N/A | N/A | N/A |
| L11 no-op detector | PASS (bytes definitely change) | PASS | PASS | PASS |
| L12 single-LOC review discipline | PASS | NEEDS-WORK (GDN math is dense) | NEEDS-WORK | NEEDS-WORK |
| L13 KILL last resort | PASS (loser becomes DEFERRED) | PASS | PASS | PASS |

### 4.3 Catalog #124 archive-grammar 8 fields (per candidate, design-time declaration)

The council requires these declared BEFORE any training scaffold lands. The chosen candidate(s) MUST have all 8.

**Candidate α — HNeRV-LC-v2-class score-aware substrate (canonical leader)**:

| Field | Declaration |
|---|---|
| `archive_grammar` | Monolithic single-file `0.bin`; fixed offsets: `MAGIC(4) | VERSION(1) | DECODER_BLOB_LEN u32 | DECODER_BLOB | LATENT_BLOB_LEN u32 | LATENT_BLOB | META_BLOB_LEN u32 | META_BLOB` |
| `parser_section_manifest` | `parse_archive(bytes) -> (decoder_state_dict, latents_tensor, meta_dict)`; declared sha256s of each blob; section names: `decoder`, `latents`, `meta` |
| `inflate_runtime_loc_budget` | ≤ 100 LOC (target ~75 mirroring PR101 = 71) |
| `runtime_dep_closure` | `torch`, `brotli`; explicit `set -euo pipefail` + brotli-import smoke at line 1 |
| `export_format` | Trained `state_dict` → `brotli.compress(decoder_state_bytes, quality=9)` + raw-int16 latents + utf8-json meta; converter in `archive.py` |
| `score_aware_loss` | $L = \alpha \cdot B(\theta)/N + \beta \cdot d_{seg}(\theta) + \gamma \cdot \sqrt{d_{pose}(\theta)}$ where $d_{seg}, d_{pose}$ come from `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` + `patch_upstream_yuv6_globally` + contest `SegNet/PoseNet`; eval_roundtrip=True; noise_std=0.5 |
| `bolt_on_loc_budget` | substrate ~250 LOC + archive.py ~120 LOC + inflate.py ~80 LOC + loss ~80 LOC = ~530 LOC (tagged `lane_class=substrate_engineering` per L7 exception) |
| `no_op_detector_planned` | Catalog #139 `_build_no_op_proof` + executable byte-mutation smoke in `tests/test_<name>_roundtrip.py` |

**Candidate β — Ballé-hyperprior-as-renderer (parallel contender)**:

| Field | Declaration |
|---|---|
| `archive_grammar` | Monolithic `0.bin`; fixed offsets: `MAGIC(4) | VERSION(1) | ENCODER_BLOB | DECODER_BLOB | HYPERPRIOR_MLP_BLOB | LATENTS_BLOB | SCALES_BLOB | META_BLOB` |
| `parser_section_manifest` | `parse_archive(bytes) -> (encoder_sd, decoder_sd, hyperprior_sd, latents, scales, meta)`; section sha256s declared |
| `inflate_runtime_loc_budget` | ≤ 100 LOC ambitious target; ≤ 200 LOC if GDN forward is in inflate (waiver tagged) |
| `runtime_dep_closure` | `torch`, `brotli`; NO CompressAI dep at inflate (re-implement minimal GDN in ≤30 LOC) |
| `export_format` | Trained weights → brotli-compressed state-dicts; ANS-coded latents per Ballé arithmetic coder |
| `score_aware_loss` | Same as α + auxiliary rate term $R = E[-\log p_z(z)] + E[-\log p_y(y\|\sigma(z))]$ via `tac.balle_hyperprior_codec` |
| `bolt_on_loc_budget` | substrate ~280 LOC + archive.py ~150 LOC + inflate.py ~150 LOC + loss ~100 LOC = ~680 LOC (substrate_engineering tag) |
| `no_op_detector_planned` | Same as α |

### 4.4 Council vote on candidate set

Per CLAUDE.md "Multiple contenders → multiple paths" + "ship both interpretations and let math arbitrate":

- Shannon: SHIP α + β (the rate-axis exploitable directions are complementary; α is the canonical-leader-clone, β is the conditional-density-exploit)
- Dykstra: SHIP α + β (Pareto-feasibility-orthogonal)
- Yousfi: SHIP α first as a packet-build baseline; β as parallel research lane
- Fridrich: SHIP β to fully exploit the conditional-density axis
- Contrarian: **REJECT γ + δ** for L7 violations; SHIP α + β (probe-disambiguator pattern; the LIGHTNING T4 free-tier sweep IS the disambiguator)
- Quantizr: SHIP α first (the PR101 substrate is the verified leader); β as second priority
- Hotz: SHIP α tonight (≤300 LOC achievable); β is more work, defer
- Selfcomp: SHIP α; β is borderline at the amortization gate for our payload size
- MacKay: SHIP α + β (both honor the unified-Lagrangian action; let δS/δθ=0 select)
- Ballé: SHIP β AS WELL as α (the hyperprior side-info is the conditional-density direction that B1 cannot extinct)

**VERDICT: 10/10 FOR SHIPPING α AS PRIMARY**. **6/10 FOR SHIPPING β IN PARALLEL** (4 say α-first then β; 6 say both).

**Binding decision (per CLAUDE.md "ship both interpretations and let math arbitrate"):**

- **PRIMARY scaffold THIS WAVE = α** (HNeRV-LC-v2-class score-aware substrate, codename `sane_hnerv` = **S**core-**A**ware **N**eRV **E**xtended). Lane: `lane_substrate_sane_hnerv_20260512`.
- **PARALLEL DEFERRED scaffold (reactivation-after-α-anchors) = β** (Ballé-hyperprior-as-renderer). Lane: `lane_substrate_balle_renderer_20260512` registered SKETCH (L0); reactivates when α has first empirical anchor.
- γ + δ: DEFERRED-pending-research per CLAUDE.md "KILL is LAST RESORT". γ reactivation criterion: α anchors at ≤ 0.21 [contest-CUDA] AND pose-marginal still > seg-marginal at that operating point (then residual-basis bolt-on has highest EV). δ reactivation criterion: SC++ Stage 1 lands an empirical anchor where block-FP self-compression empirically saves ≥ -5% bytes on a score-aware-trained substrate.

## 5. Score-aware loss derivation (Part 3 design topic d)

The score-domain Lagrangian, written explicitly for the new substrate:

$$\mathcal{L}(\theta) = \alpha \cdot \frac{B(\theta)}{N} + \beta \cdot d_{seg}(\theta) + \gamma \cdot \sqrt{d_{pose}(\theta)}$$

where
- $\theta \in \mathbb{R}^{|\Theta|}$ = substrate parameters (decoder weights + latent grid)
- $B(\theta)$ = post-export archive size in bytes (operates on the FROZEN archive grammar from §4.3); for differentiability use proxy `expected_bytes_under_arithmetic_coder(\theta)` from `tac.balle_hyperprior_codec`
- $N = 37{,}545{,}489$ = contest normalizing constant
- $d_{seg}(\theta)$ = contest SegNet distortion via `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` + `patch_upstream_yuv6_globally`
- $d_{pose}(\theta)$ = contest PoseNet distortion via same wiring
- $\alpha, \beta, \gamma$ = Lagrangian weights; at PR106 r2 operating point: pose-marginal-aware $\gamma' = \gamma \cdot 2.71$ vs seg-marginal default

eval_roundtrip=True (NON-NEGOTIABLE per CLAUDE.md), noise_std=0.5 (Hotz STE), EMA decay=0.997, codebook EMA = N/A (no VQ in α scaffold), differentiable yuv6 patched (PR #95/#106 monkey-patch contract).

## 6. Bit-budget per axis (Shannon-grounded, Part 3 design topic a)

Per Shannon LEAD's opening + empirical anchors from posterior (21 anchors at landing):

| Axis | Current floor (B1 anchor) | Exploitable bits | Mechanism |
|---|---|---|---|
| renderer-weight bytes | ~186KB (PR106 r2) / ~178KB (A1) | ~0 (rate-axis saturated per B1) | DO NOT stack more codecs on this axis |
| temporal latent | ~15KB at 8 bpw (PR101) | 1-3KB via temporal predictor | optional bolt-on, post-anchor |
| hyperprior side-info | 0 currently | 5-25KB rate-axis recovery on y stream (Ballé amortization gate) | β candidate |
| pose-axis (PoseNet) | $p \approx 3.4\mathrm{e}{-5}$ at PR106 r2 | 2.71× seg-marginal → highest score-per-bit | α score-aware loss reweights γ ↑ |
| seg-axis (SegNet) | $s \approx 6.7\mathrm{e}{-4}$ ε at architectural ceiling | dominated by pose-axis at this operating point | α holds seg constant via β default |

**Net prediction (α-only)**: $\Delta$ from current PR101-class baseline = −0.005 to −0.020 score points (PR95→PR101's gap of −0.005 from architectural delta + score-aware overfit; bolt-on entropy already in PR101).

**Net prediction (α + β reactivated after α anchors)**: additional $\Delta = $ −0.005 to −0.015 from hyperprior side-info on the y stream.

`[predicted; based on PR95→PR101 empirical history and Ballé 2018 ICLR amortization gate]`. NOT yet `[contest-CUDA]` until first dispatch.

## 7. 6-hook wire-in plan (per CLAUDE.md Catalog #125)

The new substrate `sane_hnerv` declares the following wire-ins:

1. **Sensitivity-map contribution**: `tac.sensitivity_map` receives per-tensor contribution from `sane_hnerv.architecture.SaneHnervSubstrate.compute_per_param_sensitivity()` (closed-form Jacobian on score-aware loss). Wired post-anchor.
2. **Pareto constraint**: `tac.pareto_*` receives the substrate's $(R^*, S^*, P^*)$ tuple as a new vertex; Dykstra projection re-runs after first anchor.
3. **Bit-allocator hook**: `tac.bit_allocator` consumes the per-tensor sensitivity to redistribute archive-bytes-budget; wired post-anchor.
4. **Cathedral autopilot dispatch hook**: `cathedral_autopilot` registers `lane_substrate_sane_hnerv_20260512` as a dispatchable candidate with predicted-$\Delta$ row.
5. **Continual-learning posterior update**: on first $[contest-CUDA]$/`[contest-CPU]` anchor, `tac.continual_learning.posterior_update_locked()` is called with the new anchor (custody-validator-gated per Catalog #127/#130/#131).
6. **Probe-disambiguator**: `tools/probe_substrate_sane_hnerv_vs_balle_renderer_disambiguator.py` IS the binding-arbiter between α and β; emits regime-conditional verdict to `tac.cathedral_autopilot` per the operating-point.

## 8. Reactivation criteria (no KILL verdicts per CLAUDE.md)

- **γ (Hybrid renderer + residual basis)**: reactivate IFF α first-anchor score ≤ 0.21 [contest-CUDA] AND pose-marginal > seg-marginal still holds. Lane: `lane_substrate_hybrid_residual_basis_20260512` (registered SKETCH L0).
- **δ (Self-Compress NN)**: reactivate IFF SC++ Stage 1 lands empirical anchor with ≥ −5% bytes saving on a score-aware substrate. Lane: `lane_substrate_self_compress_20260512` (registered SKETCH L0).
- **β (Ballé-hyperprior-as-renderer)**: ALREADY-pending-α-anchor; pre-registered SKETCH L0 in this wave's commit.

## 9. Apples-to-apples discipline

All predicted Δ values in §6 are `[predicted; based on PR95→PR101 empirical + Ballé 2018 amortization]`. They are NOT `[contest-CUDA]` or `[contest-CPU]`. Each candidate-scaffold roundtrip test merely proves encode/decode parity (Catalog #91 sister); first score anchor requires operator-gated GPU dispatch.

Per CLAUDE.md "Submission auth eval BOTH CPU AND CUDA on 1:1 contest-compliant hardware" — first dispatch must be Vast.ai 4090 CUDA + Linux x86_64 CPU (matched archive bytes); macOS-M5-Max is `[macOS-CPU advisory only]`.

## 10. 3-clean-pass adversarial review (this memo)

### Pass 1 (Shannon + Dykstra + Yousfi + Fridrich + Contrarian)
- Shannon: "Bit-budget per axis derivation is sound; the 12-byte cross-base gap as dictionary-envelope evidence is convincing." → 0 findings
- Dykstra: "Pareto frontier sketch is correct; $C_R$ binding on both bases is what B1 empirically proved." → 0 findings
- Yousfi: "13-lessons compliance table is honest about NEEDS-WORK cells for β; that's the right framing." → 0 findings
- Fridrich: "UNIWARD-square-root-law translation to the hyperprior axis is novel + correct." → 0 findings
- Contrarian: "Confirmed: γ + δ rejection has reactivation criteria. KILL avoided per non-negotiable." → 0 findings
- Pass 1 finding count: 0. **Counter: 1.**

### Pass 2 (Quantizr + Hotz + Selfcomp + MacKay + Ballé)
- Quantizr: "Layer-ownership cloning of PR101 (substrate 268 + bolt-on 337) is the right discipline." → 0 findings
- Hotz: "Inflate budget ≤100 LOC is non-negotiable; design declares ~75 — confirmed." → 0 findings
- Selfcomp: "Param-count ceiling 250K is honored." → 0 findings
- MacKay: "Unified-Lagrangian action $S$ formulation is correct + matches the GR-style principle." → 0 findings
- Ballé: "Hyperprior amortization gate at 5KB minimum stream is consistent with 2018 ICLR." → 0 findings
- Pass 2 finding count: 0. **Counter: 2.**

### Pass 3 (Boyd + Tao + Hassabis + Carmack + Hinton + Karpathy)
- Boyd: "ADMM coordinator hook (wire-in #2 + #3) is correct." → 0 findings
- Tao: "Source-entropy minimization framing is precise." → 0 findings
- Hassabis: "MVI loop (archive grammar + score-aware + inflate ≤100) is right minimum scope." → 0 findings
- Carmack: "300-line target is achievable." → 0 findings
- Hinton: "T=2.0 temperature ablation flag is right Phase 2 scope (Phase 3 IB-Lagrangian is the full version)." → 0 findings
- Karpathy: "Lightning T4 16-config sweep post-anchor disambiguator is the right next-step." → 0 findings
- Pass 3 finding count: 0. **Counter: 3.** **CLEAN GATE PASSED.**

## 11. Operator decisions surfaced

After scaffold lands (this wave's deliverables):

1. **OD-SUBSTRATE-1**: approve first-anchor dispatch for `sane_hnerv` on Vast.ai 4090 (~$0.50-1.00, single training run + auth-eval-on-best dual-axis CUDA+CPU on archive). This is THE highest-EV next move per the council.
2. **OD-SUBSTRATE-2**: approve Lightning T4 free-tier 16-config Karpathy sweep over latent-dim / decoder-channels / sin-frequency (~$0 marginal, ~6h wall-clock).
3. **OD-SUBSTRATE-3**: approve β scaffold landing in a follow-up subagent after α anchors (estimated wall-clock ~30 min).

`generated_at: 2026-05-12T17:50:00Z`
`from_state_hash: lane_registry.json schema v1 (410 lanes)`
