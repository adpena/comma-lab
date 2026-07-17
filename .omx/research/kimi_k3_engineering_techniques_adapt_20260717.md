# Kimi K3 engineering techniques — honest sourcing + adaptation ranking (2026-07-17)

**Operator directive (2026-07-17):** *"read the latest kimi 3 paper — beautiful engineering
techniques they used that we can learn from and adapt."* This memo is the read+design+rank arm —
**NO build in this arm**; any adopted lever lands later as a DSL `Lever` factory (triality).

**Pointer honesty first:** this is MEANS. The exact pointer (submittable 0.19108) is UNMOVED by
this memo.

---

## 0. Sourcing status — the NO-FAKE boundary (read this before citing any mechanism below)

**The Kimi K3 technical report DOES NOT EXIST as a public paper as of 2026-07-16/17.**
K3 launched 2026-07-16 (API + app). The only primary source is the official launch blog
(`https://www.kimi.com/blog/kimi-k3`), which names the five techniques in ONE paragraph each with
**no mechanisms, no formulas, no ablations**. The blog states verbatim: *"Further details on the
architecture, training, and evaluations will be released alongside the Kimi K3 technical report"*
(weights promised by 2026-07-27; report date unstated). The operator's screenshot text matches the
blog paragraph.

Therefore every K3-specific mechanism below is a **reconstruction from the name + published
lineage**, graded:

- **MEASURED-FROM-PAPER** — stated in a real published paper (K2 tech report arXiv:2507.20534;
  Muon, Keller Jordan et al. 2024; Gated Attention arXiv:2505.06708; DeepSeek-V3 aux-loss-free
  balancing; MLA, DeepSeek-V2 arXiv:2405.04434).
- **MEASURED-FROM-BLOG** — the one-sentence claim in the official K3 blog (a claim, not a mechanism).
- **INFERRED** — the most plausible mechanism given the name + the published lineage. Could be wrong.
- **SPECULATIVE** — no prior art found; a guess at the functional form.

**Standing follow-up:** when the K3 tech report lands (~2026-07-27+), re-grade every INFERRED/
SPECULATIVE row against the actual paper before any lever built from this memo is trusted.

Primary sources consulted:
- K3 blog: https://www.kimi.com/blog/kimi-k3 (fetched 2026-07-16; name-level only)
- K2 tech report: https://arxiv.org/abs/2507.20534 (MuonClip/QK-clip — published, detailed)
- Gated Attention (Qwen, NeurIPS 2025 oral): https://arxiv.org/abs/2505.06708
- DeepSeek-V3 aux-loss-free load balancing: https://openreview.net/forum?id=y1iU5czYpE
- Kimi Linear (KDA lineage): arXiv:2510.26692
- SiTU: **no published prior art found** (searched "Sigmoid Tanh Unit"/"SiTU" — nothing)

Our-stack grounding (all verified in-repo this session):
- Muon finisher: `src/tac/optimization/muon_finisher_mlx.py` — reuses `mlx.optimizers.Muon`
  (Newton-Schulz quintic (3.4445, −4.7750, 2.0315)), PR95-partition
  `MultiOptimizer([Muon(2-D hidden weights), AdamW(biases/code/out_sdf/out_tex/stem)])`;
  `muon_aspect_ratio_scale` imported by the levelset trainer. Muon = our measured d_seg finisher
  (−32%; `[[muonjump-segplateau-pivot-to-step-nonlinear-CURRENT-STATE]]`, tasks #164/#217/#269).
- Per-class-λ: #433 `aniso_perclass_lambda` (bulk/boundary Fisher-regime split; formulation wins,
  direction-neutral at n=1) + `witness_dsl/costate_agent_dsl.py::perclass_lambda_v8_schedule`.
- Activation arm: `step_basis` (stable survivor) / annealed-hosc (β 1→4; **fixed-β hosc DIVERGES
  via tanh(β·sin) saturation → vanishing grad**, DAG FEED 2026-06-25a) / `gauss` in-code UNSWEPT
  (#310).
- Costate organ #247: advisory routing/duty-to-measure ranking.
- Constants-are-poison (2026-07-15): derive LAWS, avoid sensitive hand-tuned constants.

---

## 1. Per-technique: mechanism · our-stack mapping · verdict

### 1.1 Per-Head Muon — **ADOPT-TEST** (the standout, confirmed)

**Mechanism.** MEASURED-FROM-BLOG: *"extends Muon by optimizing attention heads independently
for more adaptive learning at scale."* MEASURED-FROM-PAPER (Muon 2024): Muon orthogonalizes the
momentum of each 2-D weight matrix via Newton-Schulz, i.e. it replaces the update with
(approximately) UVᵀ from the momentum's SVD — a **per-matrix spectral normalization**. INFERRED
(high confidence): in a multi-head transformer the Q/K/V/O projections are stored as FUSED
matrices spanning all heads; vanilla Muon orthogonalizes the fused matrix, so the singular
spectrum is shared ACROSS heads — one dominant head's directions suppress the update energy
available to weak heads. Per-Head Muon = slice the fused projection into per-head blocks and run
Newton-Schulz **per block**, giving each head its own spectral normalization (and its own
aspect-ratio scale). Elegance: zero new hyperparameters; the partition is structural.

**Why it's plausible even without the paper:** it is the same move as K2's QK-clip (per-head
rescaling was already in MuonClip: QK-clip rescales per-head query/key weights) — Moonshot has a
demonstrated pattern of moving optimizer/stability interventions to per-head granularity.

**Our mapping.** We have no attention heads, but we have the exact structural analog: **fused
matrices containing semantically distinct groups whose gradient scales differ**:
- `film.weight` — the FiLM matrix maps the per-pair code to modulations for MANY hidden
  layers/channels; it is a concatenation of per-target-group "modulation heads." One NS over the
  whole matrix couples their spectra.
- `in_proj.weight` — Fourier/coordinate feature groups (frequency bands) fused into one matrix;
  low-freq vs high-freq groups have very different gradient scales (spectral bias is our measured
  enemy — lane-dash erasure = finest-scale loss).
- `hidden.<i>.weight` — already per-matrix (= already "per-layer Muon"); the sub-matrix split here
  would be per-class carrier groups IF v8 per-class carriers share a trunk matrix (v8 is design-stage).

This is the **optimizer-side dual of per-class-λ (#433)**: per-class-λ re-weights the LOSS per
regime; per-group Muon re-normalizes the UPDATE per weight-group. #433 found the formulation wins
but direction-neutral at n=1 — the optimizer-side dual is an untested orthogonal axis on a vehicle
where Muon demonstrably matters (−32%).

**Is tiny param count moot?** No — Muon's measured −32% on THIS vehicle proves update geometry is
load-bearing at a few-hundred-K params. The risk is the opposite: our matrices are small enough
that per-block NS on very skinny blocks (e.g. 2-row FiLM slices) degenerates (NS on a rank-2
block ≈ sign-ish update). So the partition must be at GROUP granularity (per target hidden-layer
modulation group; per frequency band), not per-row.

**The exact $0 A/B (build later as a DSL Lever, e.g. `muon_per_group`):**
- Lever: `--muon-group-partition {off, film_by_target_layer, inproj_by_freq_band, both}`
  (default `off` = bit-identical; registered in the activation ledger per the default-off queue rule).
- Implementation surface: a thin subclass/wrapper in `tac.optimization.muon_finisher_mlx` that,
  for a partitioned matrix, applies `_zeropower_via_newtonschulz5` per row-block (each block gets
  its own pre-normalization + aspect-ratio scale) and reassembles. ~50 LOC + tests. No new
  hyperparameter (partition is structural — honors constants-are-poison).
- A/B: the standard short local n600 config, Muon finishing stage only (`--muon-start-epoch`
  as sealed), arm-vs-control on the SAME seed; metric = d_seg descent slope in the Muon stage +
  final verdict d_seg (n600, chunked verdict batch). $0, local MLX, hours.
- Falsification threshold: no improvement in Muon-stage d_seg slope beyond seed noise ⇒ verdict
  scoped INSTANCE (this partition on this vehicle), per the verdict-scope ladder.

### 1.2 Quantile Balancing — **ADOPT-TEST** (as a design principle + one concrete site)

**Mechanism.** MEASURED-FROM-BLOG: *"derives expert allocation directly from router-score
quantiles, eliminating heuristic updates and a sensitive balancing hyperparameter."*
MEASURED-FROM-PAPER (lineage): DeepSeek-V3's aux-loss-free balancing adjusts a per-expert bias by
a fixed factor γ each step based on over/under-load — a heuristic update with a sensitive
hyperparameter (exactly what the K3 blog says it eliminates). Expert-Choice routing (Zhou et al.
2022) already assigns by per-expert top-k of scores = an implicit quantile threshold. INFERRED:
Quantile Balancing computes the empirical quantiles of the routing-score distribution and assigns
experts directly from which quantile bin a score falls in — balance holds BY CONSTRUCTION (equal
mass per bin), no γ, no bias state, no auxiliary loss. Elegance: replaces a tuned feedback
controller with an order statistic; the threshold is DERIVED from the live distribution.

**Our mapping — this is constants-are-poison made mechanical.** Sites where we carry absolute
thresholds that could become distribution-relative quantiles:
1. **Margin-annulus / capacity-routing (the concrete test site).** The codim-1 boundary annulus
   and KKT margin-saliency waterfill (`boundary_routing.py`, LEVER-4 family) select "boundary"
   pixels/regions via margin cutoffs and allocate capacity by saliency. Quantile form: annulus =
   top-q fraction of |margin|⁻¹-saliency pixels (q derived from the measured annulus mass, ~4.7%
   area carrying ~97% d_seg per #333 — i.e. q is MEASURED, not tuned); capacity bins = saliency
   quantiles with fixed mass per bin. Removes the absolute margin-threshold constant; the
   selection auto-adapts as training sharpens margins (an absolute threshold silently changes
   meaning as the margin distribution shifts — the quantile does not).
2. **Costate duty-to-measure ranking (#247)** — already rank-based (quantile-spirited); no change owed.
3. **Runtime alarms (term-domination ~40%, gnorm ~100×)** — deliberately crude tripwires; keep
   absolute (an alarm SHOULD have a hard floor; making it distribution-relative would let a slowly
   degrading run re-normalize its own alarm away).

**⚠ The measured confound this must respect (proactive recall):** we already shipped a
quantile-based guard — the spike-guard MEDIAN-freeze — and it deadlocked runs at ep103-114 because
the reference window updated from ACCEPTED batches only (confound hunt 2026-07-05; gate #398).
LAW for any quantile lever: quantiles are recomputed from ALL observations of the current window
(never accepted-only), with a re-arm path. This is the difference between "derived threshold"
and "self-referential freeze."

**The exact $0 A/B:** lever `--annulus-select {abs_margin, quantile}` (default abs = bit-identical)
on the margin-saliency routing path; same-seed A/B at n600; metrics = d_seg + the annulus-mass
telemetry row (already default-on) showing selection stability across epochs. Falsification: if
quantile selection churns the annulus membership epoch-to-epoch (>X% turnover measured, X derived
from the abs-margin arm's own turnover), the adaptive threshold is noise-amplifying — scope and stop.

### 1.3 MuonClip / QK-clip (K2-sourced, PUBLISHED) — **ADOPT-TEST (small)** — the sleeper

**Mechanism.** MEASURED-FROM-PAPER (arXiv:2507.20534): vanilla Muon at scale drove max attention
logits >1000 → loss spikes/divergence. QK-clip: after each Muon update, if a head's max attention
logit exceeds cap τ (=100), rescale that head's W_q and W_k by √(τ/max) each — **controlling the
pre-nonlinearity scale at the WEIGHT source, post-update**, rather than clipping gradients or
activations. Result: 15.5T tokens, zero loss spikes; the cap self-deactivates as logits decay into
range. Elegance: the intervention is in weight space (function-preserving up to the bilinear
form's scale), targeted per-head, and transient by construction.

**Our mapping.** We have a MEASURED instance of exactly this failure class: **fixed-β hosc
diverges** because tanh(β·sin(·)) saturates → vanishing gradient → AdamW random-walk → d_seg
rises (why the launch config must use step_basis or annealed-hosc). The current cure (anneal β
1→4 on a schedule) is a CONSTANT-SCHEDULE cure — constants-are-poison flags it. The QK-clip-shaped
cure: cap the PRE-ACTIVATION scale at the weight source — after each optimizer step, if a hidden
unit-group's max |pre-activation| (cheap to track on the training batch) exceeds cap τ, rescale
the incoming weight rows by τ/max. β can then be FIXED at its expressive value; saturation is
prevented at the source, and the clip self-deactivates as the network settles. Also note our
`gnorm_hijack` alarm (>100× clip) is DETECTION-only; this is the matching ACTUATOR pattern.

**The exact $0 A/B:** lever `--preact-clip-tau <τ|off>` (default off), arm = fixed-β hosc +
preact-clip vs control = annealed-hosc, same seed, short local run; metric = d_seg descent + the
saturation-fraction telemetry (fraction of units with |tanh arg| > 3). This simultaneously
resurrects the hosc arm (currently curriculum-worked-around, not solved) and de-constants the β
schedule. τ is one constant — derive it from the measured saturation knee of tanh (|x|≈3 ⇒
1−tanh²≈0.01), i.e. a LAW-derived constant, not a tuned one.

### 1.4 SiTU (Sigmoid Tanh Unit) — **WATCH** (queue as one arm of the owed activation sweep)

**Mechanism.** MEASURED-FROM-BLOG: name + *"enhances activation control"* only. **No published
prior art found** for "Sigmoid Tanh Unit"/"SiTU" (searched 2026-07-16). SPECULATIVE functional
form: the name suggests a self-gated bounded unit, most plausibly `SiTU(x) = σ(αx)·tanh(βx)`
(sigmoid-gated tanh; cf. SiLU = x·σ(x), Mish = x·tanh(softplus(x)), and the LSTM gate σ·tanh
product). Properties if so: bounded like tanh, but the σ gate keeps a gradient path alive when
tanh saturates, and the unit is asymmetric (kills negative lobe softly) — plausibly the
"activation control" claim.

**Our mapping.** Our activation arm is a named, UNSWEPT headroom item (#310: step/gauss/hosc;
best measured 0.004445 ≈ 4.4× above the ~0.001 need) and our one measured activation FAILURE is
tanh-saturation (hosc). A gated-tanh form attacks exactly that failure mode — same target as
§1.3's clip, from the architecture side. But with NO sourced form, building "SiTU" now would be
implementing a NAME (NO-FAKE: a technique named in a blog is a name, not a mechanism).

**Verdict:** WATCH. Add `sigmoid_gated_tanh` (our own honestly-named form, σ(αx)·tanh(βx),
trainable α,β per the step_basis trainable-slope precedent) as ONE arm of the already-owed #310
activation sweep — labeled OURS-inspired-by-name, not "SiTU" — and re-grade when the K3 report
publishes the real form. Zero new work owed before that sweep is scheduled anyway.

### 1.5 Gated MLA — **WATCH** (one transferable idea, aimed at v8 #359)

**Mechanism.** MEASURED-FROM-BLOG: *"improves attention selectivity."* MEASURED-FROM-PAPER
(likely lineage, Gated Attention arXiv:2505.06708, Qwen, NeurIPS 2025 oral): a head-specific,
input-dependent sigmoid gate σ(XW_θ) applied to the SDPA OUTPUT before the output projection;
adds non-linearity to the low-rank value path, input-dependent sparsity, eliminates attention
sinks, tolerates larger LR, improves stability. INFERRED: Gated MLA = that output gate composed
with DeepSeek-V2's Multi-head Latent Attention (MLA). We have no attention — MLA itself is N/A.

**Our mapping (the one real transfer).** The transferable IDEA is *input-dependent multiplicative
output-gating of parallel contributions before they merge*. Our v8 line (edge-centric per-class
carriers, `SPEC_v8_perclass_decomposition_20260708.md`) merges per-class carrier outputs via
merge→diff→correct reconciliation (#359). An input-dependent gate σ(g(coord, code)) on each
carrier's logit contribution before the merge = carrier selectivity, the exact analog of head
selectivity — and FiLM (already in the trunk) is the same gating family, so this is a natural
extension, not a bolt-on. **v8 is design-stage; do not build now.** Land as a named risk/cure row
in the v8 spec ("carrier interference at merge ⇒ gated merge, cf. arXiv:2505.06708") when v8 work
opens.

### 1.6 Stable LatentMoE — **N/A** (say it plainly)

MEASURED-FROM-BLOG: name + "16 of 896 experts" only. "Latent" mechanism unexplained; no paper.
We train a single tiny dense INR — no experts, no router, no token dimension. The only conceptual
echo (conditional computation per spatial region ≈ our Laguerre-cell/per-class carrier
decomposition #284) is already better served natively by v8. No honest transfer test exists.
Re-look only if the tech report reveals the "Stable Latent" part to be a training-stability
mechanism (e.g. routing in a learned latent space) with a general lesson.

---

## 2. Ranking (relevance × our-measured-need × $0-testability)

| # | Technique | Grade of source | Verdict | Why here |
|---|---|---|---|---|
| 1 | **Per-Head Muon → per-group NS partition** | blog name + published Muon; mechanism INFERRED | **ADOPT-TEST** | Muon is our measured −32% finisher; fused FiLM/in_proj groups are the exact structural analog; zero new constants; $0 same-seed A/B; optimizer-side dual of #433 |
| 2 | **Quantile Balancing → quantile-derived annulus/saliency selection** | blog name + DeepSeek-V3/Expert-Choice lineage; INFERRED | **ADOPT-TEST** | Directly serves constants-are-poison; concrete site (margin-annulus, #333-grounded q); must honor the #398 re-arm law (we already ate this confound once) |
| 3 | **QK-clip (K2, published)** → pre-activation source-clip for fixed-β hosc | **MEASURED-FROM-PAPER** | **ADOPT-TEST (small)** | The only fully-published mechanism in the set; attacks our one measured activation failure (hosc saturation); LAW-derived τ |
| 4 | SiTU → σ·tanh arm in the owed #310 activation sweep | blog name only; form SPECULATIVE | **WATCH** | No sourced form; fold as one honestly-named arm when the sweep runs; re-grade at report |
| 5 | Gated MLA → gated carrier-merge in v8 #359 | blog name + published Gated Attention; INFERRED | **WATCH** | v8 is design-stage; land as spec risk/cure row, not code |
| 6 | Stable LatentMoE | blog name only | **N/A** | No MoE, no router; no honest transfer |

**A-priori favorites confirmed:** Per-Head Muon and Quantile Balancing hold ranks 1–2, with the
qualifier that both mechanisms are INFERRED until the K3 report lands. **Overturn-grade addition:**
the K2-sourced QK-clip (not in the operator's five, but the published root of this whole lineage)
is arguably the highest evidence-per-effort item — it is the only one we can adapt from a real
paper today, and it targets a failure we have already measured on our own vehicle.

## 3. Triality + process notes

- Each ADOPT-TEST lever, when built, lands as a DSL `Lever` factory (`tac.witness_dsl`) with
  default-off + activation-ledger registration (the "off is a tracked queue" rule) — NOT a
  hand-added trainer flag. This memo is the design leg only.
- Nothing here touches the live run dir or launch chain. No code was changed in this arm.
- Every K3-specific claim above re-grades when `Kimi K3 technical report` publishes (watch from
  ~2026-07-27 with the weights release).
- Sources: kimi.com/blog/kimi-k3 · arXiv:2507.20534 (K2) · arXiv:2505.06708 (Gated Attention) ·
  arXiv:2510.26692 (Kimi Linear) · DeepSeek-V3 aux-loss-free (OpenReview y1iU5czYpE) ·
  Keller Jordan et al. 2024 (Muon).

**Pointer: submittable exact 0.19108 UNMOVED (this unit is means/design).**
