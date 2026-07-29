---
schema: pact.ddm_vae1_vae_corpus_harvest.v1
utc: 2026-07-29
research_only: true
execution_authority: false
score_claim: false
promotion_eligible: false
axis: "[research-only plus macOS-CPU advisory rate-only probe]"
local_custody_pointer: "0.1910828242 [contest-CPU] UNMOVED"
effective_frontier: "0.172141 [external official contest-CUDA]"
pointer_delta: 0
paid_spend_usd: 0
main_landing_review_required: true
---

# DDM VAE1 — VAE/VI corpus harvest against the post-burn token vehicle

## Verdict first

The VAE corpus changes campaign design, but it does not supply a current score row.

1. **MEASURED:** the bounded learned-prior row built in this pass, a fully counted static pooled
   categorical table on the exact endpoint token lattice, is **581,771 B** versus the same-object
   SMEVR control at **557,238 B**. It loses by **24,533 B** (`+4.4026%` of SMEVR), or
   `+0.0163355` in pure rate-score units. This falsifies only
   `STATIC_POOLED_MODE_DELTA_PREV1_COUNTED_CONFIG` at config SHA
   `4f86dd2101c7e6b992f797255917010297fcaab768fbff29f442ca5c8e6ffd62`; it does not kill neural,
   masked, spatial, or other first-order learned priors.
2. **DERIVED:** Ballé supplies the right row-8 discipline: optimize a deployed discrete
   code-length surrogate with every learned model/table/header bit charged, initialize the byte
   multiplier from the contest slope, and keep hard-coded bytes beside the surrogate. Generic
   Gaussian KL, arbitrary beta, KL warm-up, free bits, and delta-VAE are not row-8 defaults.
3. **DERIVED (category correction; current code verified):** row 7 is not a Concrete/Gumbel
   relaxation. Its loss is a
   deterministic temperature-scaled log-sum-exp path with no Gumbel draw or stochastic categorical
   posterior. Importing Gumbel temperature schedules or straight-through bias claims into that A/B
   would be a false transfer.
4. **DERIVED (category correction; current code verified):** the endpoint tokens are directly
   optimized variables, not outputs of an amortized inference network. E2's remaining gap is
   optimization, hard discretization, and coordination under the same decoder—not a classical
   amortization gap.
5. **DERIVED:** the 486 action-specific zero-flip observations are candidates for a four-gate
   semantic-activity ledger, not proof of posterior collapse and not certified free bytes. A token
   must clear statistical, receiver, scorer, and total-rate gates before #766 may reclaim it.
6. **DERIVED:** bits-back is not impossible for 600 records, because ANS chaining can amortize one
   clean seed. It is nevertheless `N-A` for current r7: there is no stochastic
   `p(z)p(x|z), q(z|x)` receiver, a delta posterior refunds zero bits, and all future model, CDF,
   seed, and header bytes must beat SMEVR together.

The local custody pointer remains **0.1910828242 `[contest-CPU]` UNMOVED**. The competitive
effective frontier is the distinct external official **0.172141 `[contest-CUDA]`** row. No VAE
result below is an exact contest score or a local frontier candidate.

## Current object and evidence boundary

The crosswalk uses the current completed solve-project endpoint, not the pre-token objects in the
2026-07-21 #614 pass:

| field | current custody |
|---|---|
| endpoint advisory distortion | full-n600 `d_seg=0.0038892195`; context only, not measured here |
| exact token object | `[600,24,32,4]`, `uint8` codes at 16 levels; raw token SHA-256 `c95441ac92499772705b8d8d9853564858aeb8c08de7ae5560a37562db7e2b02` |
| endpoint checkpoint | 14,963,203 B; SHA-256 `33776302e4fabf0127b01dc3551bafbd747daff71f76946883177fede24de0d3` |
| best same-object lossless token frame | SMEVR 557,238 B |
| current composed section estimate | 562,174 B; `OWED_EG1_INTEGRATION_NOT_AN_ARCHIVE_ROW` |
| whole-object conditional ceilings | 190,334 B / 157,294 B |
| post-burn decision chain | #766 structural waterfill/granularity re-race → row 7 tau-vs-CE → row 8 rate-in-loss |
| owned scorer slot | pb1; this arm ran no scorer or evaluator |

The literature labels below describe mechanisms. `MEASURED` is reserved for repository receipts or
paper-reported experiments as identified in the papers ledger; campaign transfers are
`DERIVED` or `DESIGN-INPUT` until measured on the exact current object.

## Ranked harvest

Precondition tags are executable routing, not abandon buckets:

- `MEASURABLE-NOW`: a $0 read-only/current-payload action may run without a scorer.
- `EVENT-GATED(X)`: do not start until named campaign event X exists.
- `BLOCKED(X)`: a required physical object or receiver is absent.
- `ALREADY-LIVE(X)`: consume X; do not rebuild it.
- `MAIN-REVIEW`: branch evidence is not campaign authority before landing review.

| rank | vein / corpus mechanism | disposition | corpus-to-current DIFF | falsifier and price | named consumer / precondition |
|---:|---|---|---|---|---|
| 1 | V2 counted static pooled mode-delta prev1 prior | **N-A** at `FORMULATION x STATIC_POOLED_MODE_DELTA_PREV1_COUNTED_CONFIG x config_sha256=4f86dd…ffd62` | VQ-VAE-2 motivates a learned prior; this pass physically races one pooled table with its zero-init, smoothing, CDF, traversal, range-state, base-tie, and compressor knobs machine-closed. It loses 581,771 B vs 557,238 B. | **Fired:** `B_frame >= B_SMEVR`; `+24,533 B`, $0, 4.18 s decode plus canonical re-encode. Neural/masked/spatial and other first-order priors remain open. | pb1 P5-v2 / #766; `MAIN-REVIEW`, measured receipt |
| 2 | V1 Ballé deployed-rate objective | **DESIGN-INPUT** | Row 8 already exists; VAE calculus adds a deployed discrete-CDF surrogate with fixed contest byte coefficient. A separate ceiling dual is legal only after deriving its whole-object byte budget. | Reject if surrogate ranks candidates opposite to their actual coder, or matched endpoint `100*(d_seg,new-d_seg,ref) + sqrt(10*d_pose,new)-sqrt(10*d_pose,ref) + 25*(B_new-B_ref)/N >= 0`. Training price only when row 8 opens; every model byte counts. | extension-window row 8; `EVENT-GATED(#766_AND_ROW7)` |
| 3 | V3 four-gate semantic-activity ledger | **DESIGN-INPUT** | Posterior active-units/MI diagnostics become hypotheses only. Current authority requires statistical → receiver/application → scorer → total-rate closure. | Pre-register cell-vs-`(cell,channel)` action, replacement quantum, and selector grain. Reject on unexpected parse-back, through-R cost above exact nonlinear break-even, or erased savings. Price: one token-statistics pass plus a pb1-governed full-n600 intervention batch. | #766 / wr1; stats `MEASURABLE-NOW`, score leg `EVENT-GATED(pb1)` |
| 4 | V5 per-instance hard-token refinement | **ALREADY-HELD (E2 solve-first / infinite encoder)** | SAVI and neural-compression inference work explain why encode-only iterative solving may improve a fixed decoder. Current direct variables mean there is no amortization gap to remove; VI adds gap labels and a same-parent hard-refinement gate. | Refute transfer if parsed hard local refinement has no cost-adjusted same-parent gain. Price is encode-only iteration; resulting code bytes still count. | E2 / eg1 E2 policy; `ALREADY-LIVE(E2)` |
| 5 | V4 Concrete/Gumbel schedule for current row 7 | **N-A** | The prompt's “Concrete-style” analogy is false at mechanism level: current row 7 has deterministic `tau*logsumexp(phi/tau)-phi_y`, no Gumbel noise or relaxed simplex. | Any claim of transferred Gumbel bias/schedule fails unless code contains the categorical sample and stochastic estimator it names. Price: $0 category guard; future selector costs live in row 13. | row-7 A/B / sc2 fold; `ALREADY-LIVE(DETERMINISTIC_ROW7)` |
| 6 | V6 guarded K control | **ALREADY-HELD (#319)** | IWAE K samples estimate a stochastic likelihood bound; IWAE/DReG estimator transfer is `N-A`. #319 K is exact best-of-K candidate emission, and only guarded K=1-vs-K>1 discipline is retained. | Reject K>1 if no cost-adjusted exact gain, duplicate/effective proposals collapse, or evaluator cost escapes budget. Approximate price is K× candidate evaluation. | #319; `EVENT-GATED(TERMINAL_BAND)`. STL/DReG mechanics stay with stl1 |
| 7 | V7 BB-ANS on current r7 | **N-A** at `FORMULATION x CURRENT_DETERMINISTIC_R7` | ANS chaining can share one chain-level initial state across 600 records, but its bit length may be material and r7 has no receiver-runnable stochastic latent p/q model. Delta q gives zero bits back. | `B_stream+B_p+B_q+B_initial_state+B_tables+B_header >= 557238`, support/CDF/order failure, token mismatch, noncanonical re-encode, or decode >30 min. Price: build two counted receiver models and canonical ANS chain; runtime and initial-state upper bounds required pre-build. | pb1 P5-v2 / r7 successor; `BLOCKED(COUNTED_PQ_UPPER_BOUND_BELOW_SMEVR)` |
| 8 | V2 higher-order/masked/hierarchical learned prior | **DESIGN-INPUT** | The narrow pooled-table failure exposes a physical hurdle, not a first-order/family negative. A successor must save at least 24,534 B relative to this frame before any added context-model bytes; equivalently >4.2171% of its current total. | Pre-code upper bound including model/CDF/header must be <557,238 B; physical same-object frame decides. Price: one $0 full-payload encode/decode plus counted model construction, bounded to the 30-min receiver limit. No third reflexive coder sweep. | pb1 P5-v2 / #766; `EVENT-GATED(PREFLIGHT_UPPER_BOUND)` |
| 9 | V2 FSQ-like product-scalar wire | **ALREADY-HELD (mechanism/wire only)** | The endpoint's fixed four-channel L16 rounding is an FSQ-like product-scalar analogue sharing the no-learned-codebook property. FSQ utilization/training results are not transferred. Rebuilding the wire is rediscovery. | Reopen only if a granularity re-race improves whole-object score after exact coder and map cost. Price: one governed granularity arm plus complete map/coder bytes. | #766 granularity; `ALREADY-LIVE(R7_TOKEN_OBJECT)` |
| 10 | V2 VQ-VAE-2 hierarchy / codebook-collapse cures | **DESIGN-INPUT** for a future vehicle/retrain | EMA, commitment, reinitialization, separate codebook LR, and hierarchy matter to a learned VQ codebook, which the current fixed lattice does not have. | A named future vehicle must beat the current d_seg/rate action after all hierarchy/prior/receiver bytes; utilization proxy is insufficient. Price: retraining plus a new receiver and all learned weights. | wr1 / #766 granularity rerace; `EVENT-GATED(CURRENT_CHAIN_FALSIFIED)` |
| 11 | V3 classical posterior-collapse cures | **N-A** current vehicle | KL warm-up, lagging-inference cures, skip connections, and MI floors target `q_phi(z|x)≈p(z)`. No q/p posterior exists here. #417 is a receiver-consumption bijection issue, not posterior collapse. | Reopen only with an actual stochastic encoder/prior and measured posterior-collapse symptom. Current price $0/N-A; future price is stochastic retraining plus counted model state. | future VAE vehicle; `BLOCKED(STOCHASTIC_POSTERIOR)` |
| 12 | V1 free bits, delta-VAE, aggressive encoder, arbitrary beta | **N-A** as row-8 defaults | Free bits/delta-VAE protect a minimum latent rate—the wrong default when rate is binding. Aggressive encoder needs an encoder; beta is unit-dependent. | Reject if total deployed bytes are not in the objective or if a minimum-rate floor increases the binding byte term. Current price $0/N-A; future price is a separate stochastic arm and all resulting bytes. | row 8; `EVENT-GATED(STOCHASTIC_VAE_ONLY)` |
| 13 | V4 future hard categorical selector with ST-Gumbel | **DESIGN-INPUT** as a separate arm | Concrete/ST-Gumbel is relevant only if a later vehicle introduces a sampled categorical routing decision. It must not overwrite the deterministic row-7 control. | K=1 deterministic control; charge selector/model bytes; reject on relaxation-only improvement that vanishes under hard parse-back. Price: one separate selector-training A/B plus counted selector/model bytes. | future selector/sc2; `EVENT-GATED(HARD_CATEGORICAL_DOF)` |

No row earns `ADOPT-AS-RACE-ROW`: the one raceable current-payload mechanism was built and lost.
That is the honest harvest outcome, not an empty result.

## V1 — Ballé-as-VAE: the exact row-8 transfer

Ballé-style learned compression uses a rate-distortion objective and replaces quantization by a
trainable relaxation, while the deployed entropy model prices quantized symbols. Hyperpriors add
side information that must itself be coded. The campaign transfer is therefore a counted discrete
rate model, not “add KL”:

`B_hat = (sum_i -log2 P_psi(q_i | c_i) + b_model + b_tables + b_side + b_header) / 8`,

where every `b_*` term is measured in **bits** before division by eight.

For a continuous CDF used to train a quantized symbol,

`P_psi(q | c) = F_psi(q + 1/2 | c) - F_psi(q - 1/2 | c)`,

with explicit edge-bin handling and exactly the contexts available to the shipped receiver.
The contest-normalized task objective is

`D_task = 100*d_seg + sqrt(10*d_pose)`,

and the default score-native row is

`L_row8,score = D_task + c_B*B_hat`,

with the **fixed**, never-updated contest coefficient

`c_B = 25 / 37,545,489 = 6.658589531221713e-7 score/B`
(`0.000681839568 score/KiB`).

An optional, separate ceiling controller may use

`L_row8,ceiling = L_row8,score + mu*(B_hat - B_budget)`, `mu >= 0`,

with projected `mu` updates only at stage boundaries. It is admissible only after deriving
`B_budget` as the applicable whole-object ceiling minus a measured non-token reserve. `mu` is not
`c_B`, and the fixed contest coefficient is never dual-updated. Both arms must log `B_hat` and
actual hard coder bytes. The parsed archive and exact evaluator remain authority. A surrogate that
improves while hard bytes or task score worsen is falsified.

Corpus mechanisms that do **not** become defaults:

- KL warm-up addresses posterior collapse, not a deterministic direct-token rate wall.
- Free bits and delta-VAE impose minimum information/rate and can spend bytes the contest does not
  reward.
- Aggressive inference-network updates require an amortized encoder, absent here.
- beta is not portable across loss units; the exact contest slope and measured waterfill supply the
  starting scale.

## V2 — learned discrete prior: measured physical race

The probe implements exactly one narrow VQ-VAE-2-lineage hypothesis:

`p(delta[t,h,w,c] | channel c, stored_global_mode[h,w,c], delta[t-1,h,w,c])`,

pooled across cells sharing channel/mode, with fixed `(2*count+1)` smoothing, a `2^15` integer-CDF
total, literal-zero previous residual at `t=0`, channel-major 32-bit range coding, zlib-9 model
compression, and an LZMA1-preset9-extreme mode-base side stream with lowest-symbol mode ties. It
fits positive normalized `uint16` categorical frequencies on the endpoint residual lattice, stores
all 16,384 frequencies (32,768 raw bytes) in the counted model section, and frames every section
with lengths and a semantic SHA-256. The receipt machine-closes the complete formulation config and
its SHA; source SHA closes the implementation. The decoder uses only counted model values and
already-decoded state. It proves exact parse-back and canonical re-encode; corruption, trailers,
truncation, and changed-but-valid model bytes fail closed.

| counted component | bytes |
|---|---:|
| header | 60 |
| mode base | 1,361 |
| compressed learned model | 21,868 |
| range-coded residual | 558,482 |
| **complete learned-prior frame** | **581,771** |
| same-object SMEVR control | **557,238** |
| **delta** | **+24,533** |

Receipt:
`.omx/research/ddm_vae1_ar_prior_probe_20260729/receipt.json`
(SHA-256 recorded in the review tracker). The frame SHA-256 is
`88fdc5d40537b69fcf334e20a6b9e601043653db68e84449a5118bb6d38ceb1d`.
Decode plus canonical re-encode took 4.181639 seconds on the recorded macOS/Python/NumPy runtime.
This is `[macOS-CPU advisory, rate-only]`; it is not a contest-runtime certification.

Replacing the endpoint SMEVR section in the current 562,174 B composed estimate would produce
586,707 B, still only a section estimate and farther from both ceilings. A stronger prior may
reopen only after a fully counted pre-code bound is below 557,238 B. VQ-VAE-2 hierarchy,
PixelCNN/Transformer depth, or “better entropy” is not a license to run another unbounded sweep.

## V3 — posterior collapse is not counted-but-inert

Classical posterior collapse is a distributional training failure:
`q_phi(z|x)` approaches `p(z)` and the decoder ignores the latent. Active units, aggregate-posterior
MI, KL allocation, lagging inference, and skip connections diagnose or cure that setting.

The current endpoint has deterministic optimized tokens, so those quantities have no literal q/p
referent. Catalog #417 instead requires not merely a named parser/receiver consumer, but runtime
proof that the section is decoded and actually applied; a read-and-discard parser is inert. These
can fail independently:

- a posterior may be statistically active yet its serialized section is receiver-inert;
- a counted section may be receiver-active while scorer-inert for a particular action;
- two tokens may each have marginal activity but be conditionally redundant.

Before measuring, the owner must specify whether the action unit is a spatial cell with all four
channels or one `(cell,channel)` coordinate, the replacement/quantization quantum, and selector
granularity. The correct ledger for each registered action is:

1. **statistical activity:** variance/entropy and conditional code-length gain
   `Delta ell_i = ell(z_i | C0) - ell(z_i | C1)` for legal contexts `C0 subset C1`;
2. **receiver activity:** removal/intervention is decoded, applied, and changes the parsed token or
   receiver output;
3. **scorer activity:** the exact intervention survives R and changes Seg/Pose on the frozen scorer;
4. **rate reclaim:** after charging assignment/map/header changes, the complete physical coder
   shrinks enough that
   `100*(d_seg,new-d_seg,ref) + sqrt(10*d_pose,new) - sqrt(10*d_pose,ref)`
   is less than `25*(B_ref-B_new)/37,545,489`.

The observed 486 zero-flip cells clear at most one action-specific scorer observation. They are not
certified free dimensions; S2 measured `nu_max_tolerable_q=0` for the tested snapping surface, and
GC6/R7 carry that result. Marginal MI or active-unit thresholds cannot promote a prune. Conditional
activity and whole-object bytes are binding.

## V4 — row 7 is not Concrete

The Concrete and Gumbel-Softmax estimators sample a relaxed categorical variable such as
`softmax((log alpha + gumbel)/tau)` and face estimator bias/variance as tau approaches zero.
Straight-through variants use a hard forward sample and a relaxed backward path.

Current row 7 instead compares CE against a deterministic smooth-max family:

`L_tau(phi,y) = tau*logsumexp(phi/tau) - phi_y`.

At tau 1 this matches the CE logit form; toward zero it approaches a max-margin objective. There is
no Gumbel variable, stochastic posterior, simplex sample, or straight-through categorical
estimator. Therefore:

- preserve the existing deterministic event-gated tau-vs-CE A/B;
- log smoothing gap, effective support/entropy of the deterministic softmax, and gradient
  concentration as diagnostics;
- do not import published Gumbel temperature schedules, low-temperature estimator claims, or
  Rao-Blackwellization into row 7;
- if a future hard selector exists, create a separate ST-Gumbel arm with hard parse-back and a
  deterministic control.

For that future-selector-only arm, the literature supplies brackets rather than a transferable
schedule. Concrete reports application-dependent fixed temperatures across its 2/4/8-way
experiments and treats roughly `tau≈2/3` as a starting region, not a law. Jang et al. tested an
exponential anneal floored at `0.5`, `tau=max(0.5, exp(-r*t))`. Straight-through Gumbel-Softmax is
hard in the forward pass and soft in the backward pass, hence biased; Gumbel-Rao
Rao-Blackwellizes that estimator to lower variance while preserving its expectation and therefore
its bias. None of those numeric choices transfers to the deterministic row-7 loss or to a future
16-way selector without its own hard-forward A/B.

## V5/V6 — inference gaps and K are mechanism-specific

Cremer decomposes VAE inference suboptimality into approximation and amortization gaps. SAVI and
iterative amortized inference initialize with an encoder and refine per instance. Neural image
compression work further separates amortization, discretization, and marginalization gaps while
keeping decode unchanged.

That theory corroborates the campaign's free-encoder/solve-first doctrine but changes its labels:
because the endpoint tokens are direct trainable variables, there is no amortized
`q_phi(z|x)` recognition map emitting them and therefore no endpoint amortization gap. E2 should
call the owed quantities **optimization gap**, **hard-discretization gap**, and **coordination
gap**. A same-parent hard-token local solve is admitted only if it improves the parsed objective
without changing the decoder lineage.

IWAE's K means K stochastic samples inside an importance-weighted likelihood bound. #319's K means
K emitted candidate programs followed by exact selection. These are not the same estimator. The
transfer is limited to a guarded experiment: K=1 control, K>1 only near a terminal uncertainty
band, record effective unique proposals and cost, stop without exact gain. DReG/STL gradient
mechanics remain outside this arm; no landed stl1 memo was present during recall.

## V7 — bits-back admission contract

Bits-back coding can attain a net latent-model length near the negative ELBO by decoding a latent
from `q(z|x)`, encoding the observation/latent under p, and recovering posterior bits. ANS permits
chaining records, so a 600-pair archive can in principle use one chain-level initial ANS state
rather than 600 independent states. That state is not assumed small or free: its quantified bit
length is counted. Hierarchical schemes still face initial-bit burden; Bit-Swap interleaves layers
to reduce it.

The current r7 token frame is an observed deterministic sequence, not a stochastic latent model.
Bits-back would need:

- a small receiver-runnable `p_theta(z)p_theta(x|z)` and `q_phi(z|x)`;
- integer CDFs, canonical symbol order, and ANS LIFO/reverse-order custody;
- support closure `q(z|x)>0 => p(z)*p(x|z)>0`, exact finite integer CDFs, and fixed record order;
- exact reconstruction of every int4 token followed by posterior-state restoration;
- counted p/q weights, tables, seed, syntax, and stream;
- a sequential/state-space formulation if it is to retain SMEVR's temporal dependence.

With delta q, posterior entropy is zero and no bits are refunded; the method reduces to ordinary
two-part coding. The family reopens only when quantified upper bounds for initial-state bits,
p/q weights, tables, runtime, and the complete
`B_stream+B_p+B_q+B_initial_state+B_tables+B_header` total are below 557,238 B. A delta-q or
uncounted-model probe would be a fake family test.

## Honest #614 DIFF

### What #614 got right and remains binding

- Rule 118 counts learned/video-derived decoder or prior state; free generic algorithms do not
  make learned weights free.
- A learned transform must beat the best same-object exact coder after decoder, latent, residual,
  table, and header bytes.
- Tiny-stream learned models face strict break-even ceilings; a proxy reconstruction or entropy
  gain is not enough.
- Existing VQ-VAE/Ballé/CompressAI/Cool-Chic/self-compression families were already broadly
  inventoried; this pass must not relabel that bibliography as new work.
- #558's DeepCABAC physical loss (`85,274 B` vs `83,838 B`, `+1,436 B`) remains a useful warning
  that learned coding gains disappear when model bytes close.

### What #614 could not contain because the current object did not yet exist

- the `[600,24,32,4]` fixed-L16 product-scalar endpoint and its exact token hash;
- the 557,238 B endpoint SMEVR control and 562,174 B composed estimate;
- a learned prior physically raced against that same object;
- the #766 → row 7 → row 8 endgame ordering;
- the distinction between deterministic row-7 temperature and Concrete/Gumbel sampling;
- the absence of an endpoint amortization gap and the E2 gap-label correction;
- the four-gate semantic-activity test for the 486 zero-flip observations;
- bits-back's counted p/q/seed/chaining admission contract.

### What is now stale

The old 83 KB/152 KB description-stream ceilings and VQ-decoder discussion remain valid for their
named objects but cannot price the current 557 KB token lattice. Old “learned prior may help”
language without a same-object physical row is sharpened by this pass's
`STATIC_POOLED_MODE_DELTA_PREV1_COUNTED_CONFIG` negative at config SHA
`4f86dd2101c7e6b992f797255917010297fcaab768fbff29f442ca5c8e6ffd62`.
Older claims that a generic VAE posterior cure, Concrete schedule, or amortization diagnosis applies
directly to the endpoint are mechanism-mismatched and must not route a launch.

## Consumer handoff

| consumer | exact handoff |
|---|---|
| #766 / wr1 | Add the four-gate semantic-activity ledger. Pre-register cell/channel action, replacement quantum, and selector grain. Treat 486 zero-flip cells as hypotheses. Charge selector/map bytes and use exact nonlinear whole-object break-even. |
| extension-window row 8 | Use deployed discrete-CDF code length plus counted model/table/header bits, hard-byte telemetry, and the fixed exact-score byte coefficient. A stage-boundary ceiling dual is a separate optional arm requiring a derived whole-object budget. Do not default to free bits, delta-VAE, beta, or KL warm-up. |
| row 7 / sc2 | Preserve the deterministic tau-vs-CE A/B. Explicitly mark `not_concrete_not_gumbel`; do not import stochastic-relaxation schedules. |
| E2 / eg1 | Preserve solve-first. Label endpoint debt optimization/discretization/coordination, not amortization. Gate same-parent hard refinement on parsed cost-adjusted gain. |
| #319 | Treat K as exact candidate emission, not IWAE. K=1 control; K>1 only in a terminal band with cost and unique-proposal accounting. |
| pb1 P5-v2 / r7 successor | Ingest the `STATIC_POOLED_MODE_DELTA_PREV1_COUNTED_CONFIG` negative at config SHA `4f86dd…ffd62`. Reopen a stronger prior or bits-back only with a fully counted upper bound below 557,238 B. |

## STORES CONSULTED

- `CLAUDE.md`, byte-identical `AGENTS.md`, `PROGRAM.md`, and
  `docs/operating_manual_craft_handoff.md`.
- `.omx/research/autoencoder_describe_crosswalk_20260721T232351Z.md` and its receipt (#614).
- `.omx/research/neural_selfcomp_sota_20260719_codex.md` (#558).
- VCM theory/coding/task-aware primitive memos (#152/#155 lineage).
- VQ-VAE council, maturity, and full-stack harvest memos; no VQ mechanism was rediscovered.
- `.omx/research/ddm_lv1_capstone_leverage_and_burn_20260728.md`.
- `.omx/research/ddm_gc6_from_endpoint_convocation_20260729.md` plus §7 amendment.
- `.omx/research/ddm_r7_token_coder_race_20260729.md`, receipt, and source implementation.
- current endpoint checkpoint and `tac.optimization.ddm_tr1_runtime` parse-back path.
- primary papers and official project/code pages recorded in
  `.omx/research/papers_checked_ddm_vae1_vae_corpus_20260729.md`.
- branch and worktree search for stl1/DReG output; none was landed or visible, so estimator detail
  was not duplicated.

## Authority boundary

This arm spent $0 and launched no training, n600 scorer, evaluator, provider, or paid job. The only
full-payload work was a lossless rate-only codec probe over an existing endpoint checkpoint. The
probe is resumable with immutable fitted-model and byte-frame stages; its evidence lives under
`.omx/research`. The ignored binary stage bytes were hash-verified, moved to the primary SSD tier,
and bound back into the repo-relative progress ledger through a fail-closed approved-root fallback;
`artifact_custody.json` records both original and cold-store paths, hashes, bytes, command, source
hashes, and cleanup. The bulky source checkpoint also remained on the connected SSD. No archive,
runtime, score, equation registry, task status, or canonical pointer was mutated.

MAIN must review the complete branch diff, the learned-prior frame/accounting, the exact
`STATIC_POOLED_MODE_DELTA_PREV1_COUNTED_CONFIG` plus config-SHA formulation scope
of its negative, and every consumer route before landing. Local pointer **0.1910828242
`[contest-CPU]` UNMOVED**; effective external official frontier **0.172141
`[contest-CUDA]`**; this arm moved neither.
