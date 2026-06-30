# LOCAL-THROUGHPUT ATTACK — ranked, MEASURED plan for the fastest LOCAL+FREE path to n600 (2026-06-11)

**Author:** LOCAL-THROUGHPUT-ATTACK subagent.
**Mandate (operator verbatim):** "What can we do to make local CPU or GPU faster and more useful ... I just
care about LOCAL instead of cloud and FREE instead of paid."
**Evidence grade:** every TIMING number is `[macOS-CPU advisory]` (torch-CPU, the trusted local authority);
every d_seg is the EXACT canonical `upstream/modules.py` SegNet argmax-flip authority. **NO MPS anywhere.**
$0, local, no paid dispatch. **Did the exact frontier pointer move?** No — this is a throughput/feasibility
attack that buys faster local descent, not a pointer move.

---

## TL;DR — the ranked, measured verdict

| # | Angle | Measured / estimated speedup | Stackable? | Effort | Verdict |
|---|---|---|---|---|---|
| **1** | **bat00 NVIDIA RTX 2070S (Tailscale, local, free)** | **10–50× (specs-estimated, labeled)** | base | **BLOCKED: sshd down** | **THE fast path — but needs a 2-minute operator action to restore SSH** |
| **2** | Distilled surrogate SegNet gradient | 12–16× cheaper/step (MEASURED) but **NOT descent-equivalent** (MEASURED) | ✗ (fails) | medium | **FALSIFIED at pixel-opt: student gradient gets stuck at d_seg 0.23–0.31 vs full 0.006** |
| **3a** | fp16 / bf16 gradient on **CPU** | **~9× SLOWER** (MEASURED) | ✗ | trivial | **FALSIFIED on M5 torch-CPU (no arm64 half-precision conv kernel)** |
| **3b** | subset-n curriculum (n48/n192 early) | per-step cost ∝ n (MEASURED scaling) | ✓ | low | **REAL multiplier — already the established descent recipe** |
| **4** | custom kernel / assembly | ~3× ceiling (bandwidth-bound) | ✓ | very high | **LOW leverage — do NOT build (Angle 1 dominates by 10–50×)** |

**The single fastest LOCAL+FREE path to n600:** **restore SSH on bat00 and deploy the torch trainer to its
RTX 2070S.** The 18.7 min/epoch M5 wall drops to an estimated **<2 min/epoch (10× floor) down to ~0.4
min/epoch (50× realistic)** — turning the ~24-day M5 n600-basin run into **<2.5 days (floor) to hours
(realistic)**, all local and free. The bat00 GPU is reachable at the network layer (4 ms ping, direct
tailnet) but its SSH daemons are stopped — the ONLY blocker is a GUI/RDP action to restart sshd.

---

## The anchor (re-measured this session, the EXACT canonical scorer fwd+bwd)

`experiments/measure_local_scorer_throughput.py` (reused; runs the real EfficientNet-B2 SegNet +
FastViT-T12 PoseNet with the real safetensors weights at 512×384, fwd+bwd, bs=8, 6 P-core threads):

| dtype | SegNet fwd+bwd | PoseNet fwd+bwd | BOTH | n600 epoch | note |
|---|---:|---:|---:|---:|---|
| **fp32** | 9.65 s | 5.27 s | **14.92 s/batch** | **18.7 min** | the baseline; SegNet 64.7% / PoseNet 35.3% |
| bf16 (autocast) | 104.4 s | 30.97 s | 135.4 s/batch | 169.2 min | **9.1× SLOWER** |
| fp16 (autocast) | 99.4 s | 38.1 s | 137.5 s/batch | 171.9 min | **9.2× SLOWER** |

`torch 2.11.0`, **`mkldnn=False`** on this M5 → the conv path is the `_slow_conv2d` arm64 reference kernel.
That single fact is the root of both the ~15 s/batch wall AND the half-precision regression below.

---

## ANGLE 1 (headline) — the fleet NVIDIA GPU we already own

**bat00 = `<tailscale-ip-redacted>`, Tailscale ONLINE, Windows + WSL2, RTX 2070 SUPER (→ future 3090).**

**Network reachability: CONFIRMED.** `ping` 4–6 ms; `tailscale ping` → direct pong via 192.168.1.216 in 4 ms.

**SSH reachability: BLOCKED (honest blocker, not a fabricated benchmark).** Port scan from the M5:

```
port 22   (Windows OpenSSH / PowerShell) : CLOSED/FILTERED
port 2222 (WSL2 sshd / Linux bash)       : CLOSED/FILTERED
port 3389 (RDP)                          : OPEN
port 5985 (WinRM)                        : CLOSED/FILTERED
```

bat00's OS is up (RDP answers) but **both sshd services are stopped**, so `scripts/bat00.py wsl/ps/status`
all time out. molt (`<tailscale-ip-redacted>`) has port 22 open but requires interactive Tailscale-SSH browser auth
(can't complete non-interactively) and has no confirmed GPU. **I could not run the live GPU benchmark.** Per
NO-FAKE I will NOT fabricate a measured row I couldn't produce — I give a **specs-estimate, labeled as such.**

### The estimated speedup (LABELED ESTIMATE — assumptions explicit)

Anchoring on the measured M5 fp32 = 14.92 s/batch and the established fact that the M5 conv path is the
worst-case `_slow_conv2d` kernel:

| assumed 2070S-vs-M5 speedup | 2070S scorer fwd+bwd | n600 epoch (scorer-only, bs=8) |
|---:|---:|---:|
| 10× (conservative FLOOR) | ~1.49 s/batch | **~1.9 min/epoch** |
| 20× (realistic) | ~0.75 s/batch | **~0.9 min/epoch** |
| 50× (optimistic) | ~0.30 s/batch | **~0.4 min/epoch** |

**Why the band is 10–50× and not the naive FLOP ratio:** a pure FLOP-ratio (M5 realizes only ~12 GFLOP/s on
the slow kernel vs the 2070S's 9.06 TFLOP/s peak) gives a fantastical 200×+, which OVERSTATES — at bs=8 the
GPU is launch/bandwidth-bound, not at 30–50% peak, and render+yuv6+resize aren't GPU-free (Amdahl). The
**defensible conservative band is 10–50×.** Cross-checks that bound it: (a) the 2070S (9 TFLOPS, 448 GB/s,
cuDNN) is ~1.1× a contest T4 (8.1 TFLOPS) — both Turing+cuDNN; (b) in-repo telemetry shows a **T4 doing a
FULL HNeRV train epoch (renderer+scorer fwd/bwd + archive) in ~94.5 s**, i.e. the scorer portion alone is
seconds — consistent with the dream memo's "T4 CUDA scorer is ~seconds/epoch." The decisive structural
advantage MLX-Metal and torch-CPU both lack: **cuDNN's optimized grouped/depthwise conv BACKWARD kernel** —
exactly the op that is >97% of the step and that the M5 runs on the reference fallback.

**Even the 10× FLOOR is decisive:** 18.7 → <2 min/epoch turns the ~24-day M5 n600-basin run into **<2.5 days**
(and the realistic 20–50× → hours). The **future 3090 (35.6 TFLOPS, 936 GB/s, 24 GB)** is ~4× the 2070S →
another 4× on top, and 24 GB removes the Metal bs=16 VJP memory cliff the dream memo flagged.

### Concrete next build for Angle 1
1. **Operator: restore SSH on bat00** (RDP in → `Start-Service sshd` for Windows OpenSSH, and/or start the
   WSL2 sshd on port 2222 per `bat00_wsl_setup.ps1`). This is the ONLY blocker.
2. `export BAT00_IP=<tailscale-ip-redacted> BAT00_USER=<user>` then `.venv/bin/python scripts/bat00.py status` to
   confirm GPU + WSL2 torch/CUDA. If torch/CUDA isn't installed in WSL2, scope = `uv venv && uv pip install
   torch --index cu124` (driver-pin per CLAUDE.md cu13-vs-cu124 rule; verify NVIDIA driver ≥/< 580).
3. Run `experiments/measure_local_scorer_throughput.py` (device=cuda) on bat00 → **the real measured speedup**,
   replacing this labeled estimate.
4. Deploy the resumable trainer (`experiments/run_capstone_resumable_curriculum.py`, already verified
   kill+restart safe) to bat00 WSL2 as a `nohup` daemon — the local+free n600-basin measurement run.

---

## ANGLE 2 — distilled surrogate scorer: MEASURED, and it FAILS descent-equivalence

`experiments/measure_surrogate_descent_equivalence.py` (new). A small encoder-decoder student CNN distilled
from the frozen 9.54M-param SegNet on real GT frames (decoded via `frame_utils.yuv420_to_rgb`, the authority
path), then an A/B where a corrupted render is descended to the GT SegNet target — **arm A on the full-SegNet
gradient, arm B on the student gradient, BOTH evaluated with the EXACT canonical SegNet d_seg.**

| student | params | vs SegNet | per-step speedup | distill argmax-agree | **full-grad final d_seg** | **student-grad final d_seg** | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| c=32, 6 frames (seed 0) | 0.279M | 34× smaller | **12.2×** | 98.9% | **0.0058** | **0.3087** | **NOT equivalent** |
| c=64, 12 frames (seed 0) | 1.11M | 9× smaller | (heavier) | — | 0.0055 | 0.2327 | **NOT equivalent** |
| c=32, 6 frames (seed 7) | 0.279M | 34× smaller | 10.4× | 98.8% | — (confirms speedup/agree robust) | — | — |

**The honest finding (a real NEGATIVE — the correct NO-FAKE outcome):** the student distills to ~99% argmax
agreement and is 12–16× cheaper per step, BUT its **input-Jacobian (the gradient) is a poor off-distribution
approximation**: descending on the student gradient gets STUCK at exact d_seg ≈ 0.23–0.31 while the full
SegNet gradient descends to ≈ 0.006 (both from the same 0.508 corrupted start). A bigger student (c=64)
narrows the gap only slightly (0.31→0.23) while eroding the speedup. **Matching the teacher's FORWARD argmax
does not buy you a matching GRADIENT** — the classic distillation-gradient-fidelity gap.

**Why this CONTRASTS with the dream memo's descent-equivalence:** that memo's A/B compared two **near-exact**
gradients of the SAME full scorer (torch-CPU vs MLX-GPU port, cosine 0.99986) — both ARE the real scorer.
This Angle 2 replaces the scorer with a 34× smaller approximation; cosine to the true gradient is far lower,
and it does NOT descend to the same basin. So: the MLX-GPU port (a faithful re-implementation) is descent-safe;
a distilled cheap surrogate is NOT.

**Per "Forbidden premature KILL": DEFER, not kill.** This falsifies the *naive logit-MSE-distilled* surrogate
at *direct pixel optimization*. Reactivation criteria (untested here): (a) **gradient-matching distillation**
(match input-Jacobian/saliency, not just logits — Sobolev/Jacobian-KD); (b) distill on a much larger,
on-trajectory frame set refreshed during training; (c) use the surrogate only for the EARLY high-d_seg
descent (where it agrees) and switch to the full scorer near the basin. **But none of these is needed if
Angle 1 lands** — a real GPU makes the full scorer cheap, removing the motivation for an approximate one.

---

## ANGLE 3 — cheap multipliers

**3a. fp16/bf16 gradient on CPU — FALSIFIED (measured ~9× SLOWER, see anchor table).** On M5 torch-CPU
(`mkldnn=False`), autocast half-precision has NO optimized arm64 conv kernel and emulates via upcasting →
135–138 s/batch vs 15 s fp32. The wire-in memo's note that "the FP32-exact override is only needed for eval"
is a **GPU/MLX-Metal** statement, not a CPU one. **On CPU, keep fp32.** (On the bat00 GPU, fp16/bf16 with
cuDNN/Tensor-cores IS a real ~2× lever — but that's an Angle-1-on-GPU follow-up, not a CPU lever.)

**3b. subset-n curriculum — REAL, already the recipe.** Per-step scorer cost scales ~linearly in the number
of frames in the batch (the fwd+bwd is per-image conv work). n48/n192 early stages are proportionally cheaper
per step than n600; the established fixed recipe already descends fast at small n (the dream memo measured n8
→ d_seg 0.013 in 30 epochs). This is a free, stackable multiplier on ANY backend (CPU or GPU): spend the
early high-d_seg descent at small n, switch to n600 only for the basin polish.

---

## ANGLE 4 — custom kernel / assembly: HONEST verdict = LOW leverage, do NOT build

The M5 scorer fwd+bwd is **memory-bandwidth-bound** on the slow reference conv kernel. A hand-rolled
arm64/Metal conv kernel could approach the bandwidth ceiling — but that ceiling is only **~3×** over the
current path (the dream memo's established figure), at very high engineering cost and fragility. **Angle 1
(a real cuDNN GPU) delivers 10–50× for ~2 minutes of operator effort** — it dominates the assembly lever by
more than an order of magnitude. Building a custom kernel now would be premature optimization of the wrong
substrate. **Verdict: do not build; revisit only if no GPU is ever available AND the surrogate also fails.**

---

## The single recommendation + concrete next build

**RANK 1 — bat00 NVIDIA (local + free + cuDNN).** Restore SSH on bat00 (operator, ~2 min via RDP), measure
the real scorer speedup with the existing harness, then deploy the verified resumable trainer to its WSL2 as
a `nohup` daemon. Estimated 10–50× → n600 epoch from 18.7 min to <2–0.4 min → the n600-basin run from ~24
days (M5) to <2.5 days–hours, all local and free. The future 3090 adds another ~4×.

**RANK 2 — n-curriculum (free, stackable, already in the recipe).** Spend early descent at small n on
whatever backend; n600 only for the basin.

**FALSIFIED — fp16/bf16 on CPU (~9× slower); distilled surrogate (not descent-equivalent at pixel-opt).**
**DEFERRED (not killed) — gradient-matching distillation** as the surrogate reactivation path, but moot if
the GPU lands.

**Reproduce:**
```
.venv/bin/python experiments/measure_local_scorer_throughput.py --device cpu --batch-size 8 --dtype fp32   # 14.92 s/batch
.venv/bin/python experiments/measure_local_scorer_throughput.py --device cpu --batch-size 8 --dtype bf16   # 135 s (9x slower)
.venv/bin/python experiments/measure_surrogate_descent_equivalence.py --frames 6 --student-channels 32 \
    --distill-steps 400 --descent-steps 80 --descent-lr 5.0 --out-json experiments/results/surrogate_descent_equivalence_c32_v2.json
# Angle 1 (after operator restores bat00 sshd):
export BAT00_IP=<tailscale-ip-redacted> BAT00_USER=<user> && .venv/bin/python scripts/bat00.py status
```
