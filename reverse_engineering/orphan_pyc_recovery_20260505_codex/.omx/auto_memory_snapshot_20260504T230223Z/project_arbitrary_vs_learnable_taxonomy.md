---
name: Arbitrary vs Fine-Tuned vs Learnable — Taxonomy of Knobs
description: Which parts of our pipeline are arbitrary (just pick something), which need empirical tuning, which should be learned end-to-end.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
User Q (2026-04-25): "what is arbitrary and what can be fine-tuned and what should be learnable and trained and such"

ARBITRARY (pick once, no need to tune):
- ZIP archive layout and filenames (validated by `submission_archive.py`)
- File magic bytes (ASYM, FP4A, MXLZ, I4LZ, TFC1, TPC1) — picked for unique disambiguation
- Tensor storage format (fp16 for poses, packed nibbles for FP4)
- Codec choice category (AV1 monochrome vs entropy coder vs PNG) — measured experimentally, AV1 wins clearly
- Video container format (.mkv for monochrome, .mp4 for color) — codec requirement
- Random seed (42 for reproducibility) — arbitrary as long as deterministic

FINE-TUNED (empirical sweep over a small grid, pick best by score):
- AV1 CRF value (sweep done: CRF=63 wins on rate alone, CRF=50 wins on PoseNet preservation)
- Per-block quantization size (32 weights/block — could sweep 16/32/64/128)
- Codebook entries (DEFAULT vs RESIDUAL — measured today, RESIDUAL 3.7x fewer dead weights for residual heads)
- Phase split (Phase 1 pretrain epochs vs Phase 2 scorer epochs — empirically 25/75 split)
- Gradient accumulation (2-8, depends on memory + batch size)
- Learning rate schedule (cosine vs step — cosine universally wins)
- KL distillation temperature (T=2.0 per Quantizr; could fine-tune T=1.5-3.0)
- Loss term weights (texture/L∞/markov/KL distill weights — sweep + pick)
- Mask noise probability during augmentation (currently 0.5; could be 0.3-0.7)
- Number of training epochs (depends on convergence curve)
- EMA decay (0.997 vs 0.999)

LEARNABLE (parameters of the system itself, learned end-to-end via backprop):
- Renderer weights (Conv/Linear/Embedding/FiLM)
- Per-class embeddings (5 classes × embed_dim)
- FiLM conditioning weights (pose → scale/shift modulation)
- Optimized poses (per-pair, optimized at compress time via TTO)
- Optimized embedding (per-class, optimized at compress time)
- Quantization scales when LSQ is enabled (per-tensor learned step size)

LEARNABLE BUT WE DO NOT YET (potential wins):
- Per-pair latent codes (Cool-Chic style — small learned vector per pair, compressed via int4+LZMA2)
- Boundary refinement filter (postfilter to snap mask edges to luminance gradient)
- Mask quantization codebook (currently AV1's standard codebook; could learn one tuned for our 5 classes)
- YUV6 chroma plane (currently derived from RGB; could output chroma directly)
- Inflate-time motion field (currently no warp; could learn a small warp model that runs at inflate)

ANALYTICALLY DETERMINED (no need to learn — derive from physics/geometry):
- FoE coordinates (256, 174) — function of camera focal length + principal point + driving direction
- Lane marking dimensions (MUTCD 3m × 15cm) — known by spec
- Camera matrix (K) — known from sensor calibration
- Pose dim 0 = scalar zoom (per rank-1 discovery, geometric necessity for forward motion)
- Pose dims 1-5 = pitch/yaw/roll/translation (small for highway driving, can be approximated as zero)
- Scoring formula: 100*seg + sqrt(10*pose) + 25*rate (contest spec)

THE BIG MISTAKE WE'VE BEEN MAKING:
We have been treating analytical knowledge as learnable — letting the renderer + PoseNet rediscover that ego-motion is mostly forward zoom. Per Hotz's verdict in the council deliberation:
> "The code is done. What's missing: it's not in the archive yet. Nobody wired RadialZoomWarp → compress.sh → inflate_renderer.py. The renderer is still consuming the 7KB poses.pt from old asymmetric warp. Pure laziness/diffusion of responsibility."

The 99.8% variance scalar is GEOMETRIC FACT, not a learnable parameter. Storing 6-D poses lets the system rediscover this every time at the cost of 4.6KB. Storing 1 scalar pose costs 2.4KB and bakes in the geometry.

CHARTER:
- Before training ANY new parameter, ask: "is this analytically derivable from the physics?"
- Before adding ANY hyperparameter, ask: "is this fine-tuneable with a 3-point sweep?"
- Before fixing ANY value at "arbitrary," ask: "would the wrong value break correctness or just performance?"
- Maximum effort goes to: TRAINING ONLY THINGS THAT MUST BE LEARNED. Everything else: derive, sweep, or pick.

CONCRETE NEXT MOVES BY CATEGORY:
- Arbitrary: nothing pending
- Fine-tuned: sweep KL distill weight (0.5/1.0/2.0), sweep mask aug prob (0.3/0.5/0.7), sweep CRF for SHIRAZ-trained renderer
- Learnable: train Cool-Chic at scale with FP4 robustness fix; train SHIRAZ-equivalent with mask augmentation
- Analytical: WIRE RADIAL ZOOM (Hotz's veto); IMPLEMENT LANE-MARK SPEED (zero archive cost); HARDCODE FoE (256, 174) instead of learning
