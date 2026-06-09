# Frame1 JOINT SAFE CONE — landed 2026-06-09

**Subagent:** `frame1_joint_safe_cone_20260609` (task #35).
**Evidence grade:** `[macOS-CPU advisory]` / mechanism-only. No score claims; no
dispatch. $0 local, NO cloud, NO paid GPU.
**Frontier at landing** (orphan inventory `da62505aa`): contest-CPU **0.19198533**
(archive `b7106c9b…`, 178,493 B). The cone does not change the frontier; it is a
*budget surface* downstream actuators consume.

## What landed

The **missing frame1 methodology**: the per-pixel / per-region JOINT SAFE CONE of
frame1 — the frame BOTH contest scorers read (SegNet reads ONLY frame1 argmax;
PoseNet reads both frames via YUV6). frame0 already has a methodology (SegNet-blind
by construction; PR110 exploits it up to the PoseNet budget). frame1 had none:
every byte that touches frame1 (LF/HF carrier coefficients, source-state
quantization, residual sidecars, the masks-derived render) is constrained by BOTH
scorers simultaneously, and that joint budget was never computed.

The cone:

    safe(delta) = { SegNet source-class argmax margin survives delta
                    AND  J_pose . delta is small }

- **Core**: `src/tac/optimization/frame1_joint_safe_cone.py` — typed
  `Frame1ConeConfig` + `Frame1JointSafeCone` + `compute_frame1_joint_safe_cone`
  (real scorers) + `assemble_joint_cone` (the intersection) +
  `validate_cone_behaviorally` (the falsifiable proof).
- **CLI**: `tools/build_frame1_joint_safe_cone.py` — N pairs from
  `upstream/videos/0.mkv` → per-pair cone maps (.npz) on the SSD tier + summary
  JSON + behavioral validation.
- **Tests**: `src/tac/tests/test_frame1_joint_safe_cone.py` — 28 tests (23 unit
  on synthetic deterministic scorers + 5 real-scorer NO-FAKE proof tests).

### Per-pixel cone fields (on the SegNet 384×512 grid)

| Field | Meaning |
|---|---|
| `seg_margin` | SegNet top1−top2 logit margin on frame1 = distance to argmax flip |
| `seg_margin_budget` | `seg_margin_tol · margin / boundary_slope` (seg-safe half-cone) |
| `pose_jacobian_norm` | PoseNet **frame1-channel** pixel-Jacobian L2 (pose-null half-cone) |
| `pose_budget` | `pose_response_tol / J_pose` (pose-safe half-cone) |
| `joint_cone_radius` | `min(seg_margin_budget, pose_budget)` clamped — THE budget |
| `joint_sensitivity` | canonical P18/P19 `100·slope + (5/√(10·d_pose))·J_pose` |
| `fragile_cone_mask` | radius < ½ uint8 step = binding-constraint set (empty cone) |
| `seg_argmax_class` | SegNet source class id per pixel (region key) |

## REUSE (no-duplicative-code; orphan inventory 2026-06-09)

Both half-cones existed SEPARATELY and were reused, not rebuilt. The genuinely new
code is the **intersection logic + behavioral validation + frame1-only
perturbation harness**:

- **Seg-safe half** — `segnet_boundary_pixel_saliency` from
  `tac.substrates.z8_hierarchical_predictive_coding.joint_p18_p19_deadzone_rate_attack`
  (real SegNet forward; the top-2 margin is the budget source). The frame1-only
  constraint follows from `upstream/modules.py:108` (`SegNet.preprocess_input`
  slices `x[:, -1, ...]`).
- **Pose-null half** — `posenet_pixel_jacobian_norm` from the same z8 module (real
  differentiable PoseNet backward).
- **Coupled weight** — the P18/P19 formula from
  `tac.optimization.joint_p18_p19_waterfill` (`w_i = 100·|dL_seg| +
  (5/√(10·d_pose))·‖J_pose,i‖`). The cone radius is the INVERSE of this joint
  sensitivity (high sensitivity → small budget).

### NO-FAKE caution honored (MEMORY.md Slot RR)

The directive warned that a prior `apply_pose_axis_null_projection` was FAKE
(returned markers, applied ZERO perturbation). Before building, I **behaviorally
verified the reused pose-null surface is real**: the z8
`posenet_pixel_jacobian_norm` produces a non-constant Jacobian (std 0.0064, 86%
pose-null) — but ONLY after patching `rgb_to_yuv6`.

**Critical discovery:** the reused z8 `posenet_pixel_jacobian_norm` is NOT
gradient-reachable as-shipped — upstream `rgb_to_yuv6` is `@torch.no_grad()` /
in-place and **severs the pose gradient** (the CLAUDE.md differentiable-YUV6
non-negotiable), producing an all-zero Jacobian (= "everything pose-null" = a
silently FAKE-permissive cone). My core module:
1. requires the caller to patch via `patch_upstream_yuv6_globally()` /
   `load_differentiable_scorers` (which patches internally), and
2. **fails closed** (`Frame1JointSafeConeError`) if the measured Jacobian is
   identically zero — a non-reachable gradient can NEVER masquerade as a
   permissive cone. Unit test `test_posenet_jacobian_fails_closed_when_gradient_not_reachable`
   proves the guard fires on a graph-severing PoseNet stub.

## Cone summary stats (8 real pairs, $0 CPU; `[macOS-CPU advisory]`)

Artifacts: `/Volumes/VertigoDataTier/pact/frame1_joint_safe_cone_20260609T235339Z/`
(summary JSON + 8 per-pair .npz maps; deterministically rebuildable from the CLI).

- **Usable-budget fraction: 0.486** — ~49% of frame1 pixels have ≥ ½ uint8 step of
  joint perturbation budget (the bytes that touch them have real room).
- **Empty-cone (fragile) fraction: 0.514** — ~51% of frame1 pixels are the
  binding-constraint set (cone too small to safely move; the fragile boundary
  where no frame1-touching byte may move).
- **Pose-binds fraction: 0.731** — at the PR106 frontier operating point the
  POSE budget (not SegNet) is the binding half-cone for 73% of frame1 pixels. This
  is the CLAUDE.md marginal-value FLIP made concrete: `pose_ail_gain = 271.16 =
  100 × 2.71` (pose 2.71× SegNet marginal at d_pose ~ 3.4e-5).
- **Pose-null fraction: 0.798** — 80% of frame1 pixels are pose-null (free pose
  budget), but only 49% are *jointly* usable because the seg-boundary set removes
  the difference.

### Per-SegNet-class regions (mean over 8 pairs)

| class | ~n px | usable frac | mean usable radius | reading |
|---|---:|---:|---:|---|
| 0 | 44,801 | 0.159 | 1.00 | mostly fragile (boundary-dense class) |
| 1 | 1,200 | 0.032 | 0.74 | **smallest cone** — tiny fragile class, protect it |
| 2 | 96,872 | 0.725 | 1.53 | **largest free budget** — the road/sky bulk |
| 3 | 3,338 | 0.637 | 1.71 | high free budget |
| 4 | 50,395 | 0.317 | 1.16 | mixed |

The cone is semantically correct: large flat regions (class 2) carry most of the
free frame1 budget; small/boundary classes (1, 0) are the protect set.

## Behavioral validation — the falsifiable NO-FAKE proof (all 8 pairs)

Perturb frame1 INSIDE the cone (½ × radius) vs OUTSIDE (fragile pixels at 2× the
fragile threshold), measure exact DistortionNet d_seg / d_pose. **All 8 pairs
discriminate:**

| | INSIDE (mean Δ) | OUTSIDE (mean Δ) | discrimination |
|---|---:|---:|---:|
| d_seg | 0.000181 | 0.002073 | **13.1× median** |
| d_pose | 4.6e-6 | 0.008579 | **1817× median** |

A cone that does not discriminate is FAKE. This one moves the score 13× (seg) /
1817× (pose) more when perturbed OUTSIDE its budget than INSIDE — the budget is
real and binding.

## 6-hook wire-in (Catalog #125)

1. **sensitivity-map** — ACTIVE. `joint_sensitivity` IS a per-frame1-pixel
   sensitivity map (the P18/P19 coupling).
2. **Pareto constraint** — ACTIVE (advisory). The empty-cone (fragile) set is the
   frame1 feasibility boundary; perturbations must lie inside the per-pixel radius
   polytope.
3. **bit-allocator hook** — ACTIVE (PRIMARY). `joint_cone_radius` is the per-pixel
   frame1 quantization/perturbation budget the downstream waterfiller spends:
   large radius = coarse-quantizable free byte; zero radius = protect.
4. **cathedral autopilot dispatch** — N/A. Advisory budget surface; no archive
   bytes emitted by this lane (it informs lanes that do).
5. **continual-learning posterior** — N/A. No empirical anchor promoted
   (`[macOS-CPU advisory]`, non-promotable); the cone is recomputed per archive,
   not a static posterior.
6. **probe-disambiguator** — ACTIVE. `validate_cone_behaviorally` IS the
   regime-conditional probe (inside-stable vs outside-moves → `cone_discriminates`).

## Consumers (downstream wiring)

- **#46 LF rate-distortion waterfiller** (`lf_payload_rate_distortion.py`) — the
  cone radius is the per-frame1-pixel keep/coarsen budget the THE-LAW waterfill
  spends; pixels with large `joint_cone_radius` are where LF/HF carrier bytes can
  be coarsened for free.
- **rate-attack repair** (`repair_*` campaign) — protect the fragile set; spend
  the usable set.
- **PR110++ frame1-modes** — frame0 has PR110 perturbation modes; this cone is the
  budget for the symmetric frame1-mode family.
- **SNeRV LF quantization** — per-pixel frame1 quantization step bounded by the
  cone radius (no signal loss into the binding fragile set).
- **#47 invisibility basis** — this is its **frame1 nontrivial case**: the
  byte-space null basis (`null_space_exploiter`) is the linear invisibility basis;
  the frame1 cone is the pixel-domain, dual-scorer-coupled invisibility region.

## Single most important consumer wiring to do next

**Wire `joint_cone_radius` into `tac.optimization.lf_payload_rate_distortion`
(#46) as the per-frame1-pixel coarsening budget for its keep-component law.** The
LF waterfiller currently estimates section sensitivity from the
`scorer_spectral_sensitivity.v2` atlas (band×orientation cells); the frame1 cone
gives it the *spatial* per-pixel budget the band cells cannot resolve — exactly
the resolution at which frame1-touching LF/HF coefficients are allocated. That
turns "which section is sensitive" into "which frame1 pixel inside that section
has free budget", which is the missing spatial granularity for the 0.19199→lower
rate attack.

## Per-layer canonical-vs-unique decision (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| SegNet boundary saliency | ADOPT_CANONICAL (z8) | verified-real reuse; same scorer contract |
| PoseNet pixel Jacobian | ADOPT_CANONICAL (z8) + FORK_PRINCIPLED (frame1-channel only) | z8 sums both frames; the frame1 cone needs the frame1 input channel only |
| differentiable YUV6 | ADOPT_CANONICAL (`load_differentiable_scorers`) | the non-negotiable; reused not reforked |
| P18/P19 coupled weight | ADOPT_CANONICAL (formula) | the canonical joint sensitivity |
| cone radius (intersection) | FORK_PRINCIPLED | genuinely new: min of two budgets, the joint-safe region |
| behavioral validation | FORK_PRINCIPLED | genuinely new: the frame1-only perturbation falsification |

## Reproduce

```bash
PYTHONPATH=src:upstream .venv/bin/python tools/build_frame1_joint_safe_cone.py \
    --num-pairs 8 --save-maps
PYTHONPATH=src:upstream .venv/bin/python -m pytest \
    src/tac/tests/test_frame1_joint_safe_cone.py -q   # 28 passed
```
