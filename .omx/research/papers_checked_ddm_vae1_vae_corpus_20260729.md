---
schema: pact.papers_checked.ddm_vae1_vae_corpus.v1
utc: 2026-07-29
research_only: true
score_claim: false
main_landing_review_required: true
---

# Papers checked — DDM VAE1 VAE/VI corpus

This is the anti-research ledger for the seven delegated veins. “Consumed claim” is the precise
mechanism used in the campaign memo; a title or family name alone was not treated as evidence.
Paper results are not Pact measurements.

## V1 — learned-compression/VAE rate objectives

| primary source | consumed claim | current disposition / duplicate guard |
|---|---|---|
| Ballé, Laparra, Simoncelli, [End-to-end Optimized Image Compression](https://arxiv.org/abs/1611.01704) | differentiable quantization relaxation plus entropy-model rate and distortion; deployed quantized symbols remain the coding object | `DESIGN-INPUT` for row 8; use deployed discrete probabilities and hard-byte telemetry |
| Ballé et al., [Variational Image Compression with a Scale Hyperprior](https://arxiv.org/abs/1802.01436) | hyperlatents improve conditional entropy modeling but hyper-stream side information itself carries rate | `DESIGN-INPUT`; every hyperprior/model/table bit counts under rule 118 |
| Alemi et al., [Fixing a Broken ELBO](https://proceedings.mlr.press/v80/alemi18a.html) | ELBO solutions span rate-distortion tradeoffs; beta/constraint choice determines operating point | `DESIGN-INPUT`; exact contest byte slope and budget dual replace arbitrary beta |
| Higgins et al., [beta-VAE](https://openreview.net/forum?id=Sy2fzU9gl) | beta changes relative pressure on reconstruction and latent rate/independence | `N-A` as a transferable numeric default; loss units differ |
| Bowman et al., [Generating Sentences from a Continuous Space](https://aclanthology.org/K16-1002/) | KL annealing is used against posterior collapse in a stochastic text VAE | `N-A` current deterministic tokens; do not route row 8 from analogy |
| Kingma et al., [Improved Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/abs/1606.04934) | “free bits” protects minimum per-group information by not rewarding KL below a threshold | `N-A` default when counted rate is binding |
| Razavi et al., [Preventing Posterior Collapse with delta-VAEs](https://arxiv.org/abs/1901.03416) | constrained posterior family enforces a positive minimum KL/rate | `N-A` current vehicle; direction can spend unwanted bytes |
| He et al., [Lagging Inference Networks and Posterior Collapse in Variational Autoencoders](https://arxiv.org/abs/1901.05534) | aggressive inference updates address an encoder lagging a decoder during VAE training | `N-A` current direct tokens; no amortized encoder |

## V2 — discrete bottlenecks and learned priors

| primary source | consumed claim | current disposition / duplicate guard |
|---|---|---|
| van den Oord, Vinyals, Kavukcuoglu, [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) | nearest codebook lookup, straight-through gradient, commitment/codebook losses, and an autoregressive prior over discrete indices | codebook mechanics `ALREADY-HELD`; prior idea physically reraced here |
| DeepMind, [official Sonnet VQ-VAE implementation](https://github.com/google-deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb) | EMA codebook update is an implementation lineage, not a free-rate mechanism | `ALREADY-HELD`; no code copied into the probe |
| Razavi, van den Oord, Vinyals, [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446) | hierarchical discrete latents and learned autoregressive priors improve generative modeling while introducing additional model/latent state | `DESIGN-INPUT` future retrain; all prior/hierarchy bytes count |
| Łańcucki et al., [Robust Training of Vector Quantized Bottleneck Models](https://arxiv.org/abs/2005.08520) | scale/initialization mismatch can destabilize VQ; separate/higher codebook learning rate and periodic data-dependent reinitialization are concrete remedies | `DESIGN-INPUT` only for a future learned-codebook retrain |
| Mentzer et al., [Finite Scalar Quantization](https://arxiv.org/abs/2309.15505) and [official code](https://github.com/google-research/google-research/tree/master/fsq) | product scalar levels avoid learned-codebook collapse and scale to many implicit codes | `ALREADY-HELD` at mechanism/wire level only: current four-channel fixed-L16 lattice is FSQ-like and shares no learned codebook; FSQ training/utilization results do not transfer |
| Fifty et al., [Restructuring Vector Quantization with the Rotation Trick](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2fefbb34af8008e81fb3f457fa5a2fc2-Abstract-Conference.html) | alternate gradient transport can improve learned VQ utilization | `DESIGN-INPUT` future retrain; no relevance to a fixed codebook-free endpoint |

## V3 — posterior collapse and activity

| primary source | consumed claim | current disposition / duplicate guard |
|---|---|---|
| Burda, Grosse, Salakhutdinov, [Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519) | includes the “active units” style posterior-mean variance diagnostic and the K-sample bound | activity threshold is scale/model-specific; current transfer is only a diagnostic analogy |
| Hoffman, Johnson, [ELBO Surgery](https://approximateinference.org/archives/2016/accepted/HoffmanJohnson2016.pdf) | expected per-example KL decomposes into aggregate-posterior mismatch and mutual information | motivates separating activity/MI from receiver/scorer authority |
| He et al., [Lagging Inference Networks](https://arxiv.org/abs/1901.05534) | posterior collapse can result from inference optimization lag, not only decoder power | `N-A` without q/p; prevents conflating current optimization debt with collapse |
| Dieng et al., [Avoiding Latent Variable Collapse With Generative Skip Models](https://proceedings.mlr.press/v89/dieng19a.html) | decoder architecture can force stronger latent dependence in a stochastic VAE | `N-A` current fixed decoder/direct tokens; future stochastic vehicle only |
| Kinoshita et al., [Controlling Posterior Collapse by an Inverse Lipschitz Constraint on the Decoder Network](https://proceedings.mlr.press/v202/kinoshita23a.html) | an inverse-Lipschitz decoder constraint provides a structural posterior-collapse control with theory and experiments | `N-A` current direct-token vehicle; `DESIGN-INPUT` only for a future stochastic VAE decoder |

## V4 — Concrete/Gumbel relaxations

| primary source | consumed claim | current disposition / duplicate guard |
|---|---|---|
| Maddison, Mnih, Teh, [The Concrete Distribution](https://arxiv.org/abs/1611.00712) | relaxed categorical sampling requires transformed random noise and temperature; paper experiments use application-dependent fixed temperatures over 2/4/8-way cases and treat roughly `tau≈2/3` as a starting region | mechanism/schedule guard: current row 7 has no such sample; no numeric transfer |
| Jang, Gu, Poole, [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) | differentiable categorical sample uses Gumbel perturbations; experiments test `tau=max(0.5,exp(-r*t))`; ST-GS is hard-forward/soft-backward and biased | `N-A` for current row 7; future hard selector must rerace its own schedule |
| Paulus, Maddison, Krause, [Rao-Blackwellizing the Straight-Through Gumbel-Softmax Gradient Estimator](https://arxiv.org/abs/2010.04838) | conditional Rao-Blackwellization lowers variance while retaining the ST estimator's expectation and bias | `N-A` without that stochastic estimator; do not import into deterministic smooth-max |

## V5 — amortization and iterative inference

| primary source | consumed claim | current disposition / duplicate guard |
|---|---|---|
| Cremer, Li, Duvenaud, [Inference Suboptimality in Variational Autoencoders](https://proceedings.mlr.press/v80/cremer18a.html) | separates approximation from amortization gap | category guard: current direct token variables have no amortization gap |
| Kim et al., [Semi-Amortized Variational Autoencoders](https://proceedings.mlr.press/v80/kim18e.html) | encoder initialization followed by differentiable per-instance variational refinement | `ALREADY-HELD` solve-first doctrine; same-parent hard refinement gate added |
| Marino, Yue, Mandt, [Iterative Amortized Inference](https://proceedings.mlr.press/v80/marino18a.html) | learned iterative updates can improve inference beyond one feed-forward pass | supporting theory, not a new decoder or byte row |
| Yang, Bamler, Mandt, [Improving Inference for Neural Image Compression](https://proceedings.neurips.cc/paper/2020/hash/066f182b787111ed4cb65ed437f0855b-Abstract.html) | compression-time iterative refinement can reduce amortization/discretization/marginalization gaps without changing decode | sharpens E2 labels and encode-only price; current endpoint lacks only the first of those by construction |

## V6 — multi-sample variational bounds

| primary source | consumed claim | current disposition / duplicate guard |
|---|---|---|
| Burda, Grosse, Salakhutdinov, [IWAE](https://arxiv.org/abs/1509.00519) | K stochastic p/q samples tighten an importance-weighted likelihood bound | not #319's exact candidate-emission K |
| Rainforth et al., [Tighter Variational Bounds Are Not Necessarily Better](https://proceedings.mlr.press/v80/rainforth18b.html) | increasing K can worsen inference-network gradient signal-to-noise even as the bound tightens | campaign transfer is only “K has a price”; no estimator implementation |
| Tucker et al., [Doubly Reparameterized Gradient Estimators for Monte Carlo Objectives](https://arxiv.org/abs/1810.04152) | DReG changes gradients for multi-sample variational objectives | ledgered but deferred to stl1; no duplicate design or code |

## V7 — bits-back and ANS

| primary source | consumed claim | current disposition / duplicate guard |
|---|---|---|
| Frey, Hinton, [Efficient Stochastic Source Coding and an Application to a Bayesian Network](https://doi.org/10.1093/comjnl/40.2_and_3.157) | original bits-back coding principle for latent-variable models | requires a genuine stochastic latent model, not merely deterministic tokens |
| Townsend, Bird, Barber, [Practical Lossless Compression with Latent Variables using Bits Back Coding](https://arxiv.org/abs/1901.04866) and [official code](https://github.com/bits-back/bits-back) | BB-ANS makes bits-back practical, supports chaining, and exposes initial-bit requirements | `N-A` current r7; future fully counted p/q race |
| Kingma, Abbeel, Ho, [Bit-Swap](https://proceedings.mlr.press/v97/kingma19a.html) | interleaving hierarchical decode/encode reduces the initial-bit burden relative to vanilla hierarchical BB-ANS | `DESIGN-INPUT` future hierarchy; no claim that seed cost vanishes |
| Flamich et al., [Relative Entropy Coding](https://proceedings.neurips.cc/paper/2020/hash/ba053350fe56ed93e64b3e769062b680-Abstract.html) | relative-entropy communication is a comparison family for sending samples under shared distributions | no current shared p/q custody; not a free prior |
| Townsend, Murray, [Lossless Compression with State Space Models using Bits Back Coding](https://arxiv.org/abs/2103.10150) | sequential/state-space bits-back is the relevant lineage when temporal dependence must be preserved | required reference for any successor that claims to retain SMEVR temporal context |

## Corpus-level disposition

- New current-payload measurement:
  `STATIC_POOLED_MODE_DELTA_PREV1_COUNTED_CONFIG` at config SHA `4f86dd…ffd62`, formulation
  falsified.
- New campaign design input: discrete hard-byte-calibrated row-8 objective and semantic-activity
  ledger.
- New category guards: row 7 is not Concrete; endpoint E2 debt is not amortization; #319 K is not
  IWAE K; #417 is not posterior collapse.
- Conditional future family: counted stronger prior or bits-back only after a pre-code upper bound
  below 557,238 B.
- No paper independently authorizes a training launch, scorer slot, or score claim.

The local pointer remains `0.1910828242 [contest-CPU] UNMOVED`; the distinct external official
effective frontier is `0.172141 [contest-CUDA]`. MAIN landing review is required.
