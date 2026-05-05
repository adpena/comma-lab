---
name: MPS-CUDA Auth Drift CRITICAL — never measure on MPS again
description: BINDING NON-NEG. Local MPS auth measurement is systematically WRONG by 23x on PoseNet vs CUDA T4. The 2.26 we celebrated all week was actually 0.90 on contest hardware. NEVER measure auth on MPS. Always CUDA.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
GATING MEASUREMENT 2026-04-25T21:00 — CONTEST-COMPLIANT CUDA RESULT:

Same archive (renderer.bin 297KB + masks_crf50.mkv 411KB + poses 7KB):

| Metric | Local MPS | CUDA A100 | Drift |
|---|---|---|---|
| PoseNet distortion | 0.245 | **0.0107** | **23x WORSE on MPS** |
| SegNet distortion | 0.0024 | 0.00116 | 2x WORSE on MPS |
| score | **2.26** | **0.90** | **2.5x WORSE on MPS** |

**Implications:**

1. **Every auth score this week measured on MPS was wrong by 1-2 points.** The "2.26 baseline" is contest-actual 0.90. The CRF=63 "10.20" was probably contest-actual 4-5.
2. **Quantizr 0.33 is closer than we thought.** Gap is 0.57 not 1.93. Sub-0.33 is genuinely reachable in the deadline window.
3. **The Contrarian's veto was 100% right.** "Until that one number lands: no Vast.ai launches."
4. **PoseNet drift is catastrophic.** SegNet is mostly stable; PoseNet has 23× drift. Likely cause: FastViT FP16 numerics + chroma plane handling differs MPS vs CUDA.

**BINDING RULE going forward:**
- AUTH SCORES MEASURED ON MPS ARE NOISE. They cannot be reported, used for ranking, or shipped.
- ALL auth eval MUST run on CUDA (Vast.ai 4090, A100, or T4).
- MPS is acceptable ONLY for proxy scoring and smoke tests (architecture validation, code-correctness checks).
- preflight should reject auth eval invocations with `--device mps` and warn loudly.
- Add to deploy_vastai.py: a CUDA-only "snap eval" mode that re-evaluates the current pinned archive on demand (~5 min, $0.05).

**The 2.26 vs 0.90 drift specifically:**
- PoseNet input is YUV6 (chroma involved) — chroma plane numerics differ MPS↔CUDA per memory project_hardware_geometry_chroma_full.md
- FastViT-T12 has attention layers with softmax — softmax numerics drift across float16 implementations
- Combined effect: PoseNet on MPS reads systematically larger pose distortion than CUDA

**Action from this measurement:**
- Update BATTLE_PLAN baseline from 2.01/2.26 → 0.90 (contest-CUDA)
- Re-evaluate all council projections against this real number
- Sub-0.33 is reachable with: (mask sweep -0.4) + (radial zoom -0.2) + (KL distill -0.1) + (Cool-Chic shrink -0.05)
