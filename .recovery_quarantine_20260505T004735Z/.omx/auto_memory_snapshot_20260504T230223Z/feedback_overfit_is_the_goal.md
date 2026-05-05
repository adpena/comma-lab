---
name: Overfitting to This Video IS the Goal — Not Generalization
description: The entire contest is about compressing ONE specific 60-second dashcam video. Memorization, per-frame optimization, and video-specific analysis are features, not bugs.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
The contest evaluates on a SINGLE 60-second video (0.mkv, 1200 frames, 600 pairs). There is no test set, no generalization requirement, no unseen data. **Overfitting to this specific video is the entire point.**

**Why this matters:**
- We can memorize every frame if the archive budget allows
- Per-pair latent codes are not "overfitting" — they're optimal compression
- The renderer should be tuned specifically for the content in this video (highway driving, specific lighting, specific road geometry)
- TTO per-pair optimization is the IDEAL approach — it literally finds the optimal output for each pair individually

**What this unlocks:**
- Deep analysis of the VIDEO itself: scene structure, lighting changes, ego-motion profile, class distributions per frame
- Deep analysis of the SCORERS: what specific features does PoseNet use? What textures does SegNet look for? Analyze their intermediate activations, gradients, sensitivity maps
- **Reverse-engineer the scorer binaries**: analyze PoseNet/SegNet weights, find null spaces, understand exactly what they measure and don't measure
- **Channel-level analysis**: what do the YUV6 channels look like? Which channels carry pose information? Which carry segmentation information?
- **Per-frame analysis**: which specific frames/pairs are hardest? What makes them hard? Craft specific solutions per pair.
- **Adversarial analysis**: what pixel perturbations fool PoseNet with minimum perceptual change? (This IS steganalysis — Fridrich's domain)

**How to apply:**
- Never add regularization, dropout, or generalization tricks to the renderer
- Never hold out validation data — train on everything
- Use per-pair conditioning (latent codes, pose vectors) aggressively
- Analyze the scorer networks' internal representations for this specific video
- Study the video content: when does the car turn? Where are the hard pairs? What's the scene structure?
