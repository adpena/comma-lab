# In-curriculum levers design — floor-chasing co-design with the base_ch=20 HNeRV substrate (2026-06-12)

**Author:** in-curriculum-lever DESIGN subagent (Layer 2 of the operator's three-layer stack: substrate / in-curriculum levers / post-hoc bolt-ons).
**Status:** DESIGN/SPEC ONLY. No GPU runs launched (Phase 1 base_ch=20 basin is live; a GPU run here would contend). Every lever is **gated on Phase 1 confirming a frontier** before Phase-2 deployment.
**Frontier (pointer, NOT hardcoded):** `.omx/state/canonical_frontier_pointer.json` → contest-CPU `0.19109982`, archive `177169 B`, lane `pr110_payload_entropy_recode`. **Frontier UNMOVED.** Target `T_floor = 0.11797` (rate-dominated, per the GOAL section; this memo treats it as a PREDICTION/derivation anchor, not a measured result).
**NO FAKE:** every quantified effect below is a **PREDICTION** with its first-principles basis named. A designed lever is a design, not a measured row. Nothing here asserts a score; deploy-gating + the dual CPU/CUDA exact-eval gate stand.

> Per CLAUDE.md "score-domain Lagrangian not weight-domain proxies" (HNeRV parity L6) + `eval_roundtrip` + EMA non-negotiables. The torch_vehicle loss loop ALREADY honors eval_roundtrip (bicubic↑874 → bilinear↓384 → uint8-STE round, `driver.py:389-395`) and EMA-after-every-step (`driver.py:445`); levers must NOT break those.

---

## 0. The decomposition that ranks every lever (the binding-constraint map)

The contest score is `S = 100·d_seg + sqrt(10·d_pose) + 25·|archive.zip|/37_545_489`.

At the **current frontier** (`0.19109982`, `177169 B`):
- **rate term** `= 25 · 177169 / 37_545_489 = 0.11796` — this is **61.7 %** of S and is **numerically equal to T_floor** (`0.11797`). That is not a coincidence: **T_floor is rate-dominated**; the measured floor is essentially "rate term at the best byte count we have, with d_seg and d_pose driven to their architectural floors."
- residual `S − rate = 0.19110 − 0.11796 = 0.07314` is the **distortion budget** still on the table: it splits into `100·d_seg` and `sqrt(10·d_pose)`.

**Consequence for ranking (the means/ends firewall applied to levers):**
1. **The rate term is the dominant lever AND the hardest floor.** Closing `0.19110 → 0.11797` requires the **distortion residual `0.07314` → ~0** *without paying it back in bytes*. So the highest-EV levers are the ones that **drive d_seg and d_pose to zero at constant-or-lower byte cost** — i.e. score-domain training (so we stop wasting bytes on reconstruction the scorer ignores) + a differentiable rate term (so the optimizer can trade reconstruction fidelity the scorer ignores for fewer bytes).
2. **Pose is structurally collapsible at near-zero byte (the Quantizr lever).** `sqrt(10·d_pose)` is the nonlinear term; at the frontier d_pose is already small but the **marginal value of pose near zero is high** (CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent": below `pose_avg ≈ 2.5e-4` the pose marginal *exceeds* SegNet's, derivative `5/sqrt(10·pose_avg) → ∞`). Pose-FiLM stores ~6 scalars/pair and **removes pose from the learning problem entirely**, freeing the whole decoder capacity for d_seg + rate.
3. **d_seg is argmax-flip, not L2.** The contest d_seg is the per-pixel **SegNet argmax-disagreement rate** (`upstream/modules.py`; `score_pair_components` confirms it). Training the decoder against an L2/reconstruction proxy spends bytes on pixels the scorer's argmax ignores. A **differentiable d_seg surrogate** + a **score-aware QAT** that protects the argmax boundary (not the L2 reconstruction) is how d_seg → its floor at lower byte.

So the levers are NOT independent knobs; they are a **co-designed system** that re-routes the entire byte/distortion budget toward the three score terms. The ranking in §6 follows directly from this map.

### What the substrate ALREADY has (do not reinvent — SEARCH-FIRST result)

The torch_vehicle PR95 8-stage curriculum (`src/tac/torch_vehicle/{driver,curriculum}.py`, vendored from PR95 — `build_curriculum` reads the live `StageConfig`) already carries:
- **C1a coder-aware entropy reg** = `cat_entropy_v2` (`src/tac/losses/cat_entropy_v2.py`), wired at `driver.py:409-415,424-429`. It penalizes the **per-weight Shannon entropy of the INT8-quantized weight histogram** (soft-histogram over `{-127..127}`, σ-annealed). **This is a per-tensor weight-entropy proxy, NOT the actual brotli/LZMA byte cost** — that is the gap lever #1 attacks.
- **σ noise schedule** (`cat_sigma`) + **QAT** (`spec.use_qat` → `self.v.apply_qat`/`restore_qat`, `driver.py:382-386`) — but QAT here is the **vendored fake-quant**, NOT score-aware.
- **Muon final-stage** + per-stage `seg_loss_fn` (the seg surrogate, swappable per stage) + EMA + eval_roundtrip.
- **split-by-head backward** (`driver.py:459-513`) — already routes SegNet grad on train-device (MPS, validated bit-identical on d_seg) and PoseNet grad on the **CPU authority** (zero MPS pose drift). This is the gradient-reachable score path the levers plug into.

The actual archive codec (the byte cost we must surrogate): vendored `codec.py` — **decoder = per-tensor symmetric INT8 → zigzag → brotli q=11** (one blob); **latents = per-dim minmax→uint8 → 1st-order temporal delta → zigzag uint16 → lo/hi byte-split → brotli**. `build_archive`/`parse_archive` = `self.v.*`, byte count = `len(archive)` (`driver.py:612-622`).

---
