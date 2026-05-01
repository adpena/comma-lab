# writeup working notes

## current state — 2026-04-30 (3 days to deadline)

The writeup needs to reflect the post-AV1 era. We are no longer a tiny CNN post-filter pasted onto a codec — that arc ended around `1.73`. The current frontier is PFP16 A++ (`1.043987524793892` recomputed [contest-CUDA A++]) on top of the Lane G v3 renderer, and the live competitive context is Selfcomp 0.38 #2 / Quantizr 0.33 #1.

## live operating point

- contest-compliant best: **`1.04` [contest-CUDA A++]** (PFP16 A++, 686,635-byte archive, exact Tesla T4 auth eval)
- score authority: `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/eval/contest_auth_eval.json`
- stale provenance quarantine: `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/custody/custody_manifest.json`
- historical predecessor: **`1.05` [contest-CUDA]** (Lane G v3, 694KB archive, KL distill + pose TTO retry)
- fallback floor: `1.15` [contest-CUDA] (Lane A, baseline pose-TTO from baseline poses)
- nearby negative: Lane M-V2 = `1.84` [contest-CUDA] (radial-zoom rank-1 hypothesis; pose-pad asymmetry confirmed)
- nearby catastrophic: Lane GP v3 = `89.67` [Modal-T4-CPU] (Runge phenomenon at degree-10 polynomial)

## the headline arc has changed three times

1. **CNN post-filter on AV1** (4.06 → 1.73). The original story of dilation, QAT+EMA, and best-checkpoint int8 selection. Still publishable, still mathematically interesting (rank-1 Jacobian, dense mid-frequency CNN residual). But not the live frontier.
2. **Neural renderer that bypasses the codec** (1.73 → 1.05 → 1.04). Asymmetric warp + Lagrangian + dilated-h64. Pose TTO at compress time. KL distill weight=0.002 lands the gain that the v1/v2 attempts missed; PFP16 then makes the pose payload deterministic and smaller without changing the archive custody path.
3. **Selfcomp paradigm shift** (1.04 → ?). Live work. Five concrete shifts (grayscale-LUT mask, single-mask + 6-DOF affine duality, analytical pose, block-FP weight self-compression at 1.017 bpw, 94K-param SegMap arch). Eight Modal lanes in flight to validate each shift in isolation and then stack.

## strongest rigor angle for the writeup

The non-negotiable preflight catalog grew from 36 to **78 strict checks** in a single week. Every catastrophic measurement bug got a static check:

- `check_no_mps_fallback_default` — kills the MPS-vs-CUDA 23x PoseNet drift
- `check_anchor_masks_resolution` — kills the 48x64 vs 384x512 mask-resolution disaster
- `check_no_eval_roundtrip_false` — kills the 2-11x proxy-auth gap
- `check_no_scorer_load_at_inflate` — kills the strict-scorer-rule violation
- `check_remote_scripts_have_nvdec_probe` — kills the Vast.ai NVDEC roulette failure mode
- `preflight_arity` — kills the dead-flag wiring pattern (Council R3-1 catch)
- `check_no_brittle_six_line_waiver_lookback` — kills the same-line waiver scope violation

This is the engineering story under the score story: every mistake the lab made got a permanent guard-rail. The 0.90 → 1.04 → submitted-score arc only happens because the underlying measurement infrastructure stopped lying.

## strongest narrative angle

Three nested phase changes in one project:

1. CNN post-filter → neural renderer (give up the codec entirely)
2. Pixel-space TTO → conditioning-space TTO (give up pixel locality)
3. Hand-tuned bit allocation → block-FP self-compression (give up uniform precision)

Each phase abandons an assumption of the previous. At each transition the score makes a discrete jump (1.73 → 1.15 → 1.05 → 1.04). The Selfcomp paradigm is the next jump if any of MM/SA/SC++/SO land predicted ≤ 0.5.

## strongest competitive-intelligence angle (PRIVATE — do not publicize)

- Quantizr explicit admission: "sub 0.30 is possible just by sweeping conv dims" — they stopped optimizing.
- Selfcomp explicit admission: "underfit to segnet due to no architecture search" — implicit rate ceiling left on the table.
- Mask2mask author refused to publish compress script — they are signaling competitive secrecy too.

The writeup CAN reference public scores. The writeup MUST NOT publish our specific Lane W hard-pair weighting recipe, our Lane Ω Hessian-aware quantization schedule, or our specific Selfcomp-paradigm portfolio sequencing — those are the secret sauce. Public surfaces may headline PFP16 A++ = 1.04 only with the final deploy bundle custody path and the `contest_auth_eval.json` authority note.

## reusable visual story beats (still valid)

1. AV1 byte-layout repair: 97.45 → 2.20 (story unchanged)
2. Saliency map: only 7% of pixels matter for PoseNet (story unchanged, reusable for renderer story)
3. SVD rank-1 finding: Jacobian effective rank ~1, condition number 399 (still drives the renderer-architecture intuition)
4. 11.5x SegNet leverage at the operating point (still the central tradeoff lever)
5. NEW: CRF=50 dilated-h64 + matched poses 0.90 baseline — first reproducible-from-saved-artifacts contest-CUDA score
6. NEW: Lane G v3 1.05 contest-CUDA — KL distill weight=0.002 finally clears the proxy-auth gap
7. NEW: PFP16 A++ 1.04 contest-CUDA — deterministic fp16 pose payload drops archive bytes while `contest_auth_eval.json` stays authoritative

## what NOT to write yet

- Don't claim Selfcomp-paradigm scores until they land [contest-CUDA].
- Don't claim sub-0.5 stack predictions on any public surface — the EV math relies on additive stacking that we have not verified.
- Don't expose the specific Lane W / Lane Ω / Lane DARTS-S design details; they are the moat against Quantizr and Selfcomp.
- Don't publicize the Cloudflare site URL; the human will decide when to share.
