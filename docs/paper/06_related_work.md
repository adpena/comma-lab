# 6. Related Work

## 6.1 Neural video compression

End-to-end learned video codecs have advanced rapidly in recent years. Ballé et al. [2018] established the variational autoencoder framework for image compression. For video, DCVC [Li et al. 2021] and its successors [Li et al. 2023] use learned motion compensation and conditional coding to approach or exceed H.266/VVC on perceptual metrics. Cool-Chic [Ladune et al. 2023; Ladune et al. 2024] takes a different path: it overfits a very small decoder and compact latent representation to a single image or video, with the learned representation transmitted as the bitstream. This is encoder-side instance optimization rather than a universal feed-forward codec.

Our work differs from these in a fundamental way: we do not optimize for human perceptual quality. The distortion metrics are frozen neural networks (SegNet, PoseNet) with specific architectures and blind spots. A frame that looks terrible to a human can score well if it preserves the features these networks use. This inversion --- optimizing for machine perception rather than human perception --- changes the problem structure entirely.

C3 [Kim et al. 2024] is the closest recent work to our April 25 experimental direction: high-performance neural compression from a single image or video with emphasis on low decoder complexity. It is relevant because the decoder itself can be the archive, and because coordinate-conditioned representations are natural residual models. Our current C3 lane is only a prototype residual head on top of a scorer-aware renderer; it does not yet reproduce the paper's full coding system, entropy model, or bit allocation strategy.

The key distinction is that Cool-Chic and C3 target conventional reconstruction rate-distortion, while this challenge is task-aware compression against fixed SegNet and PoseNet scorers. We borrow their low-complexity overfitting principle, but every claim must be revalidated against eval-roundtrip scorer behavior and archive compliance.

Paper-faithfulness audit for the April 25 prototypes:

| Component | Cool-Chic/C3 papers | Current prototype | Risk |
|-----------|---------------------|-------------------|------|
| Instance optimization | Overfit representation to one image/video | Yes, through profile-specific renderer training | Needs smoke training evidence |
| Low-complexity decoder | Central design constraint | Yes, tiny shared decoder or small coordinate MLP | Parameter count is necessary but not sufficient |
| Entropy coding | Core rate term | Not implemented | Archive size may overstate paper-equivalent efficiency |
| Learned quantization / bit allocation | Core compression mechanism | Not implemented beyond existing FP4 roundtrip | Uniform FP4 may waste bits |
| Temporal sharing | Explicit in video setting | Shared decoder, no full temporal entropy model | May miss video-rate gains |
| Task-aware objective | Not the primary paper objective | Yes, our scorer losses define distortion | Paper PSNR/MS-SSIM claims do not transfer |
| Contest inflate path | Not applicable to papers | Not yet complete for prototypes | Blocks deployment claims |

## 6.2 Test-time optimization and adaptation

Test-time training (TTT) [Sun et al. 2020] updates model parameters at inference using a self-supervised auxiliary task. TENT [Wang et al. 2021] adapts batch normalization statistics to the test distribution. Both operate on the model; our TTO operates on the output pixels directly, treating the renderer output as an initialization for gradient-based refinement.

The closest analogy is 4D-Var data assimilation [Courtier et al. 1994], used in numerical weather prediction: a model state trajectory is optimized to fit observations at multiple time steps. In our case, the "observations" are the scorer outputs on original frames, the "model state" is the sequence of generated frames, and the "trajectory constraint" is PoseNet's requirement that consecutive frames be geometrically consistent.

Neural Radiance Fields [Mildenhall et al. 2020] and 3D Gaussian Splatting [Kerbl et al. 2023] also perform per-scene optimization, fitting a representation to observations. Our TTO is simpler --- we optimize pixels, not a learned representation --- but the principle of test-time fitting to a specific instance is shared.

## 6.3 Video coding for machines

The MPEG Video Coding for Machines (VCM) initiative [Duan et al. 2020] standardized as ISO/IEC 23888, addresses compression optimized for downstream analysis tasks rather than human viewing. The VCM framework includes pre-processing and post-processing modules around a standard codec, with task-specific optimization.

Neural Wrapping [Khan et al. 2025] adds learned pre- and post-processing around a standard codec, optimized for downstream task performance. Sandwiched Compression [Du et al. 2023] proposes a similar neural pre+post processor concept. Both target human perceptual metrics alongside task metrics; we optimize exclusively for the task.

Our CPU-lane postfilter (stage 3 in our results) is a concrete instance of Neural Wrapping: a convolutional network applied after H.265 decoding, trained to minimize scorer distortion. The ceiling we hit (auth 1.33) demonstrates the fundamental limitation of the wrapping approach: information destroyed by quantization cannot be recovered by any postfilter, regardless of capacity.

## 6.4 PR lineage and bolt-on engineering on the public leaderboard

The May 4 medal-band entries (gold PR #101 at 0.193, silver PR #103 at 0.195, bronze PR #102 at 0.195) share a common engineering pattern: each is a small, focused delta on PR #100's archive substrate (BradyMeighan, hnerv_lc_v2, 0.1954). Bit-level deconstruction of the archives reveals the lineage explicitly. PR #95 (AaronLeslie138, hnerv_muon) introduces the HNeRV decoder architecture; PR #98 adds a channel-postprocess delta; PR #100 adds the latent-correction sidecar that becomes the substrate for the medal-band; PR #101 wraps PR #100's bytes in a schema-driven decoder with split-Brotli streams and per-tensor byte-map permutations; PR #103 substitutes arithmetic coding for Brotli on the densest tensor blocks (8 tensors, ~290 bytes saved per the PR body); PR #102 is the most striking case — it ships PR #100's archive bytes verbatim and adjusts only the inference-time scale (0.0100 to 0.0095) plus per-frame channel nudges, no codec changes whatsoever.

The strategic implication is that at this score band the contest does not reward bespoke from-scratch codec design. Once one team makes its inflate and compress code public via a PR, every other team can read it, fork it, and start bolting on. Engineering velocity becomes the differentiator, not novel theory. The silver entry was 241 lines of code in 2 files; the bronze entry was decoder-side scalar adjustment with zero new compression machinery. Three teams reached medal-band scores by adding focused engineering deltas to one prior submission's archive layout.

A substrate-mismatch corollary follows. When we ported PR #101's split-Brotli + byte-maps codec onto the PR #106 substrate (different fine-tuned weights), the empirical saving was only 241 bytes — a roughly 33-fold shortfall versus the 7,963 bytes the same code achieved on PR #101's own substrate. Per-tensor byte-map permutations are tuned to a specific weight distribution; on a shifted substrate, the entropy structure that the byte-maps exploit no longer holds. Codec wins from one PR are not portable across substrates without retuning. We therefore expose an `auto_select_byte_maps` derivation in `tac.pr101_split_brotli_codec` that re-runs the per-tensor brotli search on the caller's actual weights, and we frame the four-way stack predictions in this paper as multiplicative on jointly trained weights rather than additive on borrowed substrates.

This lineage analysis informs both the methodology used here — every score citation in this paper carries an explicit substrate tag — and the broader question of how contest-scale neural-compression work should be evaluated. Reproducibility in this setting requires not just the code but also the exact weight checkpoint the codec was tuned against, because the codec's gain function is itself substrate-dependent.

## 6.5 Adversarial examples and gradient masking

Athalye et al. [2018] identified *obfuscated gradients* as a common failure mode in adversarial robustness research: defenses that appear robust to gradient-based attacks but are actually masking the gradients rather than increasing true robustness. They cataloged three types: shattered gradients (non-differentiable operations), stochastic gradients (randomized defenses), and vanishing/exploding gradients.

Our gradient bug (Section 3) is an unintentional instance of shattered gradients. The `@torch.no_grad` decorator on `rgb_to_yuv6` creates a non-differentiable barrier in an otherwise differentiable pipeline. The result matches the adversarial case precisely: the optimizer appears to make progress (PoseNet loss changes) but is actually blind to the true gradient direction, relying on incidental correlations through other loss terms.

Carlini and Wagner [2017] emphasized that evaluating adversarial defenses requires verifying that the optimization *actually works* --- checking gradient flow, confirming that the attack finds true local optima. The same discipline applies to any optimization through frozen networks: validate the gradient, not just the loss.

## 6.6 Steganalysis and steganographic security

The competition has a deep structural connection to steganalysis, first identified by Fridrich [2009]. In steganalysis, the goal is to detect whether an image has been modified (a message embedded). In our competition, the goal is to generate images that a detector (the scorer) cannot distinguish from originals. We are performing *inverse steganalysis*: embedding information (compressed representations) in a way that is undetectable by a specific analysis pipeline.

Fridrich's constrained optimization framework --- minimize the embedding payload subject to a detectability constraint --- maps directly to our formulation: minimize rate subject to scorer distortion constraints. The augmented Lagrangian method we use is standard in this literature.

Yousfi et al. [2020] extended Fridrich's framework to deep learning-based steganalysis, training detectors and embedders adversarially. The comma.ai challenge is a simplified version of this setup: we know the detector architecture (PoseNet + SegNet), and the detector is frozen. This asymmetry --- the defender (us) has complete white-box access to the detector --- is the opposite of real-world steganalysis, where the detector is unknown. It makes the problem easier in principle but introduces its own challenges, as our gradient bug demonstrates: white-box access is worthless if the gradients are broken.

## 6.7 The Yousfi-Fridrich floor

The standard Shannon rate-distortion bound `R_Sh(D)` quantifies the minimum bit-rate to transmit a source `X` such that a reconstruction `X̂` satisfies `E[d(X, X̂)] ≤ D` for some distortion measure `d`. In nearly all video-compression literature, `d` is a perceptual distortion (PSNR, MS-SSIM, LPIPS, or human-calibrated proxy). This contest is structurally different: the distortion is `d_task(X, X̂) = 100 · seg_dist(SegNet(X̂), SegNet(X)) + sqrt(10 · pose_dist(PoseNet(X̂), PoseNet(X)))` where SegNet and PoseNet are *frozen, public, fully-specified* neural networks. The relevant lower bound is therefore tighter than Shannon's `R_Sh(D)` for perceptual `d`.

Define `R_YF(D) = inf_{p(X̂|X)} I(X; X̂)  s.t.  E[d_task(X, X̂)] ≤ D` as the rate-distortion function under the contest's task-aware distortion. We name this bound the **Yousfi-Fridrich floor**, after Yassine Yousfi (the contest's scorer architect, formerly of Fridrich's DDE Lab at Binghamton) and Jessica Fridrich (whose constrained-optimization framework — minimize embedding payload subject to a detectability constraint — maps directly to the contest's structure). The naming honors the steganalysis lineage: UNIWARD's textured-region weighting, Fridrich's square-root law for distortion spread, and Yousfi et al.'s adversarial detector-embedder framework all anticipated that *adversarial compression against a known fixed detector* admits structurally tighter rate bounds than perceptual compression against an unknown human observer.

The YF-floor inequality `R_YF(D) ≤ R_Sh(D)` is strict whenever the scorer's preimage on its output space is non-trivial — i.e., whenever distinct reconstructions `X̂_1 ≠ X̂_2` produce identical scorer outputs `Φ(X̂_1) = Φ(X̂_2)`. For this contest, the inequality is strict because (a) SegNet's stride-2 EfficientNet-B2 stem coarsens spatial detail before any class-discriminative computation, making sub-(256, 192) artifacts invisible, and (b) PoseNet's YUV6 + FastViT-T12 attention coarsens chroma detail. Bytes spent reconstructing input regions inside the joint blind spot are wasted under YF but charged under Shannon.

The empirical leaderboard floor at the time of submission is approximately 0.31 (top three contestants in a tight band), against a realistic Shannon floor of ~0.155 derived from explicit per-component R(D) analysis. The YF floor lies strictly below 0.155. The gap between 0.31 and the YF floor reflects practical engineering and contest-deadline dynamics (Section 7.7) rather than a fundamental information limit.

The full derivation and bracket appear in the methodology addendum companion document. We claim no priority over the underlying steganalysis ideas, only on the explicit naming and contest-application of the floor concept and on the empirical observation that the contest's leaderboard band is wide above the YF floor. The scorer-Jacobian Karhunen-Loève (SJ-KL) basis primitive (the eigendecomposition of `F = 100·JᵀJ + 10·KᵀK`) is the natural residual-coding primitive that approaches the YF floor in practice.

## 6.8 Dual-axis evaluators and the CPU/CUDA distinction

A subtle methodological feature of this contest is that `upstream/evaluate.py`
produces *two* distinct authoritative score axes for the same archive bytes —
`--device cuda` and `--device cpu` — and the public leaderboard ranks by the
**CPU** axis, not CUDA. Most of the perceptual-codec literature treats device
choice as a runtime/throughput consideration rather than a scoring axis, but
in this contest the two axes are structurally different: `evaluate.py`
selects different decoder backends (`DaliVideoDataset` for CUDA, hardware
NVDEC; `AVVideoDataset` for CPU, libavcodec software decode), and the
SegNet/PoseNet kernels run with different FP32 reduction-tree shapes between
the two device backends.

We measured the gap empirically across the medal-band HNeRV cluster
(PR100/101/102/103/105) from public CI bot comments and found
`R_pose = pose_cuda / pose_cpu = 5.04 ± 0.10` and
`R_seg = seg_cuda / seg_cpu = 1.17 ± 0.01` — a near-constant 0.033
score-points gap, of which 70% sits in pose and 30% in seg. The asymmetry is
intrinsic to the score-aggregation shape: pose is regression
(MSE-quadratic in noise), seg is argmax classification (piecewise-constant
in logits). Section 4.8 covers the empirical computation; the OSS
documentation `docs/findings/cuda_cpu_auth_eval_split_20260508.md` covers
the mechanism analysis and the strategic-exploitation prescriptions
(floor-aware pose-Huber loss, leaderboard-aware Lagrangian via
`tac.score_geometry target_axis="cpu_leaderboard"`, calibrated
training-time noise injection at σ ≈ 1.7e-3).

The closest related work is the literature on FP32 reproducibility across
matmul backends (Hutter et al., NVIDIA reproducibility whitepaper),
non-determinism in deep-learning evaluation pipelines (Pham et al. on
training-stability variance), and the steganalysis literature on
detector-versus-detector consistency under input perturbations (Yousfi
et al.'s detector-detector calibration work). The novel observation here
is not that FP32 backends differ — that's well-known — but that **a
contest leaderboard's structural choice of CPU as the ranking axis,
combined with a regression head whose noise floor sits at the CPU axis's
precision boundary, produces a 5× pose ratio** that competitors who
optimize against the CUDA score will systematically over-spend bit-budget
on driving pose<sub>cuda</sub> below the CPU floor.
