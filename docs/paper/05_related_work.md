# 5. Related Work

## 5.1 Neural video compression

End-to-end learned video codecs have advanced rapidly in recent years. Ballé et al. [2018] established the variational autoencoder framework for image compression. For video, DCVC [Li et al. 2021] and its successors [Li et al. 2023] use learned motion compensation and conditional coding to approach or exceed H.266/VVC on perceptual metrics. Cool-Chic [Ladune et al. 2023] takes a different path: a tiny decoder (< 3K parameters) overfitted to each image, with the decoder weights transmitted as part of the bitstream --- a form of test-time optimization at the encoder side.

Our work differs from these in a fundamental way: we do not optimize for human perceptual quality. The distortion metrics are frozen neural networks (SegNet, PoseNet) with specific architectures and blind spots. A frame that looks terrible to a human can score well if it preserves the features these networks use. This inversion --- optimizing for machine perception rather than human perception --- changes the problem structure entirely.

C3 [Kim et al. 2024] applies neural compression to the comma.ai challenge specifically, using a NeRV-style [Chen et al. 2021] implicit neural representation. Our approach is closer to Cool-Chic in spirit (small model, test-time optimization) but replaces the autoencoder framework with a conditional generator that exploits the known structure of the distortion metrics.

## 5.2 Test-time optimization and adaptation

Test-time training (TTT) [Sun et al. 2020] updates model parameters at inference using a self-supervised auxiliary task. TENT [Wang et al. 2021] adapts batch normalization statistics to the test distribution. Both operate on the model; our TTO operates on the output pixels directly, treating the renderer output as an initialization for gradient-based refinement.

The closest analogy is 4D-Var data assimilation [Courtier et al. 1994], used in numerical weather prediction: a model state trajectory is optimized to fit observations at multiple time steps. In our case, the "observations" are the scorer outputs on original frames, the "model state" is the sequence of generated frames, and the "trajectory constraint" is PoseNet's requirement that consecutive frames be geometrically consistent.

Neural Radiance Fields [Mildenhall et al. 2020] and 3D Gaussian Splatting [Kerbl et al. 2023] also perform per-scene optimization, fitting a representation to observations. Our TTO is simpler --- we optimize pixels, not a learned representation --- but the principle of test-time fitting to a specific instance is shared.

## 5.3 Video coding for machines

The MPEG Video Coding for Machines (VCM) initiative [Duan et al. 2020] standardized as ISO/IEC 23888, addresses compression optimized for downstream analysis tasks rather than human viewing. The VCM framework includes pre-processing and post-processing modules around a standard codec, with task-specific optimization.

Neural Wrapping [Khan et al. 2025] adds learned pre- and post-processing around a standard codec, optimized for downstream task performance. Sandwiched Compression [Du et al. 2023] proposes a similar neural pre+post processor concept. Both target human perceptual metrics alongside task metrics; we optimize exclusively for the task.

Our CPU-lane postfilter (stage 3 in our results) is a concrete instance of Neural Wrapping: a convolutional network applied after H.265 decoding, trained to minimize scorer distortion. The ceiling we hit (auth 1.33) demonstrates the fundamental limitation of the wrapping approach: information destroyed by quantization cannot be recovered by any postfilter, regardless of capacity.

## 5.4 Adversarial examples and gradient masking

Athalye et al. [2018] identified *obfuscated gradients* as a common failure mode in adversarial robustness research: defenses that appear robust to gradient-based attacks but are actually masking the gradients rather than increasing true robustness. They cataloged three types: shattered gradients (non-differentiable operations), stochastic gradients (randomized defenses), and vanishing/exploding gradients.

Our gradient bug (Section 3) is an unintentional instance of shattered gradients. The `@torch.no_grad` decorator on `rgb_to_yuv6` creates a non-differentiable barrier in an otherwise differentiable pipeline. The result matches the adversarial case precisely: the optimizer appears to make progress (PoseNet loss changes) but is actually blind to the true gradient direction, relying on incidental correlations through other loss terms.

Carlini and Wagner [2017] emphasized that evaluating adversarial defenses requires verifying that the optimization *actually works* --- checking gradient flow, confirming that the attack finds true local optima. The same discipline applies to any optimization through frozen networks: validate the gradient, not just the loss.

## 5.5 Steganalysis and steganographic security

The competition has a deep structural connection to steganalysis, first identified by Fridrich [2009]. In steganalysis, the goal is to detect whether an image has been modified (a message embedded). In our competition, the goal is to generate images that a detector (the scorer) cannot distinguish from originals. We are performing *inverse steganalysis*: embedding information (compressed representations) in a way that is undetectable by a specific analysis pipeline.

Fridrich's constrained optimization framework --- minimize the embedding payload subject to a detectability constraint --- maps directly to our formulation: minimize rate subject to scorer distortion constraints. The augmented Lagrangian method we use is standard in this literature.

Yousfi et al. [2020] extended Fridrich's framework to deep learning-based steganalysis, training detectors and embedders adversarially. The comma.ai challenge is a simplified version of this setup: we know the detector architecture (PoseNet + SegNet), and the detector is frozen. This asymmetry --- the defender (us) has complete white-box access to the detector --- is the opposite of real-world steganalysis, where the detector is unknown. It makes the problem easier in principle but introduces its own challenges, as our gradient bug demonstrates: white-box access is worthless if the gradients are broken.
