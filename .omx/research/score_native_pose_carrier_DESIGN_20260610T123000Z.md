# Score-native amortized pose carrier — DESIGN + pre-registration (task #57)

**Subagent:** `task57_pose_carrier`. **Authority:** `[local CPU-torch advisory]` (exact upstream
PoseNet/SegNet on CPU, GT via `frame_utils.yuv420_to_rgb` ONLY) + `[macOS-MLX research-signal]`
(carrier forward). `promotable=false`, `score_claim=false`. NO MPS. $0 unless paired eval fires.

## The problem (from #56 verdict)
The score-native carrier wins rate (72KB vs 177KB, −59%) by amortizing the SEG argmax into a 65KB
INR. But it has NO pose-carrying appearance section: the palette frame1 collapses pose (d_pose=12.66).
PoseNet reads BOTH frames (resize 384×512 → rgb_to_yuv6 → 12ch → MSE on first 6 of 12 pose dims).
SegNet reads ONLY frame1 (`x[:,-1]`). So frame0 is SegNet-invisible → free to carry pose motion.

## Pose-structure probe (4 pairs, exact CPU PoseNet) — the DECISIVE prior
| config | d_pose |
|---|---:|
| A identity (GT0,GT1) | 0.000000 |
| B frame0 lowres f4 / GT1 | 0.011572 |
| B frame0 lowres f8 / GT1 | 0.073626 |
| C both lowres f4 | 0.031925 |
| C both lowres f8 | 0.493041 |
| D frame1 lowres f4 / GT0 | 0.013051 |
| D frame1 lowres f8 / GT0 | 0.045286 |

**Reading:** pose is ~additive across the two frames' fidelity loss. NAIVE bilinear low-res caps
d_pose at ~0.01 (f4) — already sqrt(10·0.01)=0.32, 1.7× the whole frontier (0.191). Tube precision
(d_pose 2.9e-5) needs near-full-res frames — exactly the 177KB HNeRV decoder's job.

## Score budget (the binding math)
Frontier S = 100·5.6e-4 + sqrt(10·2.9e-5) + 25·177169/D = 0.056 + 0.017 + 0.118 = 0.191.
- IF seg term reaches frontier (5.6e-4 → 0.056) [bridge #2, not this task; bounds result]:
  to hit sub-0.15 the pose+rate budget is 0.094. At B=72KB (rate 0.048) → d_pose_max **2.1e-4**;
  at B=100KB (rate 0.067) → d_pose_max **7.5e-5**; at B=135KB (rate 0.090) → d_pose_max 1.7e-6.
- The seg generator's CURRENT d_seg=0.0068 → term 0.68, ALREADY 3.5× the frontier. The seg term is
  the FIRST binding constraint; the pose carrier is the SECOND.

## PRE-REGISTERED PREDICTION
A score-aware learned **amortized** luma carrier (an MLX INR conditioned on (pair,x,y) → camera-res
RGB, like the seg generator) can reach a LOWER d_pose at a given byte budget than naive bilinear
low-res, because it learns the high-frequency residual the upsampler misses. PREDICTION: at a ~30-60KB
amortized budget the carrier reaches d_pose < 0.005 (better than f4 bilinear's 0.013), and the RD
curve (d_pose vs carrier capacity) is monotone-decreasing.

## PRE-REGISTERED KILL CRITERION
IF no amortized operating point reaches d_pose below the level where the FULL S (with the seg term at
its honest current/achievable value) beats the frontier OR hits sub-0.15 — i.e. the luma needed for
pose precision costs more rate than it saves vs the 177KB frontier decoder — RECORD the finding (pose
is NOT cheaply amortizable to tube precision; the score-native rate win does not survive the pose
constraint at this seg-term level) + reactivate via lever C (jointly-trained smaller amortizer that
ALSO carries seg, i.e. a unified frame decoder = converging back toward HNeRV) OR a frame0-warp pose
carrier (store optical-flow/warp params, not pixels).

## Architecture
`AmortizedLumaCarrier` (MLX, mirrors `ScoreNativeSegGenerator`): Fourier(x,y) → in_proj → FiLM-per-pair
→ hidden → 3-channel RGB head (sigmoid·255). Per-pair mod code (mod_dim). Trained score-aware vs the
exact PoseNet via a differentiable rgb_to_yuv6 surrogate (the upstream is @no_grad/in-place) +
eval_roundtrip. Capacity knobs for the RD sweep: {hidden_dim, mod_dim, n_hidden, n_fourier, quant bits}.
numpy-portable forward (the portability contract). Byte cost = quantized weights + per-pair mod, brotli.

## Wire-in (Catalog #125): sensitivity-map (pose RD curve), Pareto (carrier {d_pose,B} surface),
bit-allocator (carrier byte section), continual-learning (the verdict), probe-disambiguator
("can an amortized carrier beat the low-res pose ceiling?"). cathedral-autopilot = the conditional paired-eval gate.
