# Lever C viability smoke — PRE-REGISTRATION (task #62)

**Subagent:** `task62_lever_c_viability_smoke`. **Written BEFORE any measurement.** Authority of every
number this design will produce: `[macOS-MLX research-signal]` (the conv-decoder forward) +
`[local CPU-torch advisory]` (exact upstream PoseNet/SegNet `DistortionNet` on CPU, GT decoded via
`upstream/frame_utils.yuv420_to_rgb` ONLY, S recomputed from components — the rounded field lies).
**NOT** the contest 600-sample harness → non-promotable per the authority ladder. `$0` spend, no GPU,
**no paid dispatch from this smoke**, **NO MPS**. `promotable=false`, `score_claim=false`,
`ready_for_exact_eval_dispatch=false`.

**Frontier (pointer, not hardcoded):** `0.19109982` `[contest-CPU]`, 177,169 bytes (lane
`lane_pr110_payload_entropy_recode_20260610`). Secondary gate: sub-0.15.

---

## 1. What #57/#61 proved (the wall this smoke tests a way through)

- **#57:** the amortized coordinate-INR POSE carrier (frame0) works in isolation (d_pose 0.0036, 13 KB,
  20× better than naive low-res) but its RD curve is NON-monotone and ceilings at ~0.0036 (124× above
  the tube 2.9e-5). frame1 (the d_seg carrier) is the dominant remaining pose debt (12.14).
- **#61:** the frame1 dual (seg+pose) constraint is **ANTAGONISTIC at coordinate-INR capacity**: a
  pose-trained frame1 INR gets d_seg 0.733; a seg-trained frame1 (palette) gets d_pose 12.14. The
  cheapest frame1 holding BOTH terms is >400 KB raw-low-res → the score-native rate win is destroyed.
  The diagnosed cause: a smooth coordinate-MLP cannot represent the SHARP SegNet argmax boundaries
  (high-frequency) AND carry the pose luma simultaneously.
- The pre-registered next lever (both verdicts §5): a fresh-init **CONVOLUTIONAL** per-pair-latent
  frame1 decoder (HNeRV-class: conv + PixelShuffle upsample + sin, PR95 L18) — the structurally
  expressive carrier the coordinate-INR cannot be. The OPEN QUESTION: can it hold d_seg AND d_pose
  jointly below the 177 KB frontier byte budget?

## 2. The lever-C architecture (the ORIGINAL methodology, not an absorb-recode)

A small per-pair-latent conv decoder generating frame1 (and, in the joint variant, frame0 for pose):

- **per-pair latent** `z_p` (small, e.g. 16–32 dim) → linear → reshape to a tiny spatial seed
  (e.g. C×6×8) → **N conv-blocks**, each `Conv(in, out·4, 3×3) + PixelShuffle(2) + bilinear-skip + sin`
  (PR95 L18, the verified leaderboard decoder block) → upsample 6×8 → … → camera-res RGB (sigmoid·255).
- This is a SHARED decoder (the weights amortize across pairs) + per-pair latents (the rate is
  ~94% decoder weights / ~6% per-pair latents, PR95 L19 structure). Byte cost = int8+brotli of weights
  + latents (reusing the canonical `measure_carrier_bytes` accounting).

**The three ORIGINAL objective terms (no leaderboard entry uses this combination):**
1. **null-space-primary recon:** drive frame1's representation error INTO the seg-null subspace — the
   ~80.67% of pixels the SegNet argmax cannot see (#52 margin polytope free-budget) — by construction.
   Concretely: a recon-MSE anchor weighted by the per-pixel margin free-budget (large-margin/interior
   pixels are cheap to be wrong; small-margin/boundary pixels are protected). The error the decoder
   cannot avoid is steered to where SegNet's argmax does not flip.
2. **Jacobian-aimed pose:** weight the pose recon anchor by the MEASURED PoseNet pixel-Jacobian field
   (`posenet_jacobian_saliency`) so decoder capacity goes to PoseNet's 6 scored-dim sensitive pixels.
   The pose OBJECTIVE itself is the exact 6-dim PoseNet pose-MSE (in the loop, differentiable yuv6).
3. **argmax-polytope-constrained seg:** the seg OBJECTIVE is a soft d_seg surrogate — boundary-weighted
   cross-entropy/KL against the GT SegNet argmax+margin (the precomputed targets), with the
   margin-polytope as the constraint (keep frame1's SegNet argmax == GT at the boundary). The exact
   argmax-flip d_seg is RE-MEASURED on the frozen CPU SegNet (authority).

eval_roundtrip (uint8 STE) + differentiable rgb_to_yuv6 in the inner loop (non-negotiable). EMA shadow
is the inference checkpoint. The numpy-portable forward is the inflate-time reference (scorer-free).

**Composition (deferred to PASS):** the cheap seg-carrier/boundary-solver (#52/#55) as a correction
sidecar where the conv decoder's seg argmax is still wrong.

## 3. THE $0 VIABILITY GATE (pre-registered prediction + kill criterion)

**Smoke:** a short MLX/CPU-torch run (8 pairs, modest epochs on M5 Max) of the small conv decoder
trained JOINTLY seg+pose. Measure d_seg AND d_pose on the EXACT CPU-torch scorer (GT via
`yuv420_to_rgb`); byte-count decoder weights + per-pair latents (int8+brotli). Report the RD point
`{bytes, d_seg, d_pose, joint-hold Y/N}`.

**PRE-REGISTERED PREDICTION:** the small conv decoder, being structurally able to represent sharp
argmax boundaries (unlike the coordinate-MLP), CAN hold **d_seg < 0.01 AND d_pose < 0.01 JOINTLY at
< 120 KB** (decoder weights + per-pair latents, int8+brotli, 8-pair smoke extrapolated to the shared
decoder + per-pair latent split). The conv inductive bias breaks the #61 antagonism that the
coordinate-INR could not.

**KILL/DEFER CRITERION:** if the conv decoder CANNOT hold both terms below 0.01 at < 120 KB even in
smoke — i.e. holding both requires near-frontier (177 KB) decoder size, OR one term blocks the other
at any size — then the score-native carrier converges to full-frontier-size (the rate advantage is
gone). **FIRE → DEFER the score-native crux + pivot** (lever D contour-coding of the residual /
R1+R2+R3 lossless bank / AFSR-1 rate campaign). Record WHICH term blocked, at what byte cost.

**On PASS:** pre-register the full campaign per the long-burn default (timing smoke sec/epoch → 600-pair
full MLX run, resumable + harvested → byte-close → dual CPU+CUDA exact eval) with stop/continue
thresholds. Do NOT fire paid dispatch from this smoke; report the campaign as the next launchable unit.

## 4. Canonical-vs-unique decision per layer (CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD)

| Layer | Decision | Rationale |
|---|---|---|
| conv-block (Conv+PixelShuffle+bilinear-skip+sin) | ADOPT_CANONICAL (PR95 L18) | the verified leaderboard decoder block; this is the structurally-expressive family #61 named |
| per-pair latent + shared decoder split | ADOPT_CANONICAL (PR95 L19) | 94%/6% byte split is the canonical HNeRV-class rate structure |
| byte accounting (int8+brotli) | ADOPT_CANONICAL (`measure_carrier_bytes`) | same quant/brotli contract as #57; no substrate reason to fork |
| differentiable yuv6 + eval_roundtrip | ADOPT_CANONICAL (non-negotiable) | the PR95 inner-loop recipe; severing pose grad is the #8 forbidden pattern |
| **null-space-primary recon weight** | **FORK_PRINCIPLED (ORIGINAL)** | no leaderboard entry steers recon error into the seg-null by construction; this is the novel lever |
| **Jacobian-aimed pose recon weight** | **FORK_PRINCIPLED (ORIGINAL)** | reuses `posenet_jacobian_saliency` but as a frame1 capacity allocator — novel composition |
| GT decode | ADOPT_CANONICAL (`yuv420_to_rgb`) | the ONLY legal GT path (PyAV rgb24 manufactures phantom pose) |

## 5. NO-FAKE commitments (the SUPREME rule)
- Real training: a torch/MLX optimizer loop actually descends the joint loss; internal-consistency
  guard (elapsed ≥ epochs × min_sec) refuses a stub.
- d_seg/d_pose are the EXACT frozen-scorer measurements on the numpy-decoded frame (not a proxy; the
  fp32 train-loss "win" is NOT the verdict — the exact quantized measurement is).
- ≥15 behavior tests: the null-space projection actually confines error to the free subspace; the
  Jacobian weighting actually changes the gradient; a CONSTANT decoder FAILS (cannot hold either term);
  the conv decoder output actually varies per-pair and per-pixel; byte cost tracks capacity.
- $0, no paid dispatch from the smoke. NO MPS. Fail-closed on the eval gate (no $ to confirm a
  non-improvement).
