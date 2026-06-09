# Evaluator Response Atlas + Waterfilling Compiler — canonical engine design

UTC 2026-06-09 · claude · operator directive (definitive): "build an all-pairs, all-evaluator-relevant
-dimensions atlas ... evaluator tomography engine ... then solve mathematically ... we have M5 Max
128GB + MLX + unlimited compression-time compute." This supersedes-by-extension the Order-0 pilot
(`evaluator_cell_tolerance.v1`) and `evaluator_response_surface_solve_plan_20260609.md`. North star:
lowest `S = 100·d_seg + √(10·d_pose) + 25·B/N` on 0.mkv via upstream evaluate.py. RGB fidelity is NOT
an objective — only the evaluator-equivalence class + bytes.

## Why not brute-force every raw pixel
Per evaluated pair ≈ 2×874×1164×3 ≈ 6.1M raw dims; ×600 pairs ≈ 3.7B coordinates. Finite-difference
per coordinate = billions–trillions of scorer evals = wrong basis. Instead query ALL dimensions via
**gradients / JVP / VJP / vmap + structured bases**, and rank with active-subspace + Morris-screening
+ Sobol. "All dimensions" = full-gradient maps + active subspaces + structured region/frequency/frame/
YUV bases + exact action atoms — NOT scalar finite differences (those only for blocked local maps).

## The object: evaluator tomography
Per source pair x_i=(frame0_i,frame1_i), action δ_i, response
  F_i(δ) = ( d_seg(x_i, x_i+δ), d_pose(x_i, x_i+δ), S_i(...) )
Atlas measures curves C_{i,b}(t)=F_i(t·b_i) for many bases b, plus gradient maps. The compiler then
either (a) finds the cheapest archive program whose output stays in the same evaluator cell, or (b)
accepts action a iff 100·Δd_seg + Δ√pose + 25·Δbytes/N < 0 (the evaluator-action waterfiller).

## Official asymmetry that shapes every family (verified upstream/modules.py)
- SegNet scores ONLY frame1 (last of pair), argmax of 5 classes, no normalization, [0,255].
- PoseNet scores BOTH frames via RGB→YUV6 + mean/std(255/2,255/4), first 6 of 12 dims, MSE.
⇒ frame0 is SegNet-invariant (pose-only); frame1 drives both. ~600/1200 frames are seg-free.

## Atlas families (each a typed sub-artifact; ALL over 600 pairs)
1. `evaluator_source_degradation_curves.v1` — blur / downsample / quantize / noise / chroma / luma.
   (Order-0 pilot done on pair 0; promote to all 600.)
2. `evaluator_frame_incidence_curves.v1` — frame0_only / frame1_only / both_same / both_opposite /
   frame_swap / duplicate0 / duplicate1. (Tests the SegNet/PoseNet asymmetry directly.)
3. `evaluator_frequency_sensitivity.v1` — DCT block bands / FFT radial bands / Laplacian-pyramid bands /
   Sobel-edge attenuation. (What frequencies does SegNet need? PoseNet? what is byte-waste?)
4. `evaluator_region_waterfill_curves.v1` — perturb per class region / class boundary band / interior /
   motion-dense vs sparse / hard L-set masks. (Where bytes must go spatially.)
5. `evaluator_gradient_atlas.v1` — the all-raw-dimension layer in ONE backward pass each:
   - SegNet: source class c_p = argmax SegNet(frame1)_p; margin m_p = logit_{c_p} − max_{j≠c_p} logit_j;
     g_seg = ∇_x Σ_p softplus(γ − m_p)  ⇒ per-pixel wall-distance t_wall(p,v) ≈ m_p / −⟨∇m_p, v⟩.
   - PoseNet: J_pose = ∂PoseNet(YUV6(f0,f1))/∂x; δd_pose ≈ δxᵀ JᵀJ δx; Hutchinson/VJP ⇒ pose saliency
     map + pose-null directions + pose-sensitive subspace.
6. `evaluator_action_atom_curves.v1` — REAL compiler atoms: HiNeRV backend δ, sidecar tile/RLE patch,
   frame0 pose-compensation, frame1 Seg-wall action, SNeRV LF/HF/MFU/HFR/TUB atom, semantic-cell
   primitive, codec choice. (The dims that actually build the archive.)
7. `evaluator_commutator_atlas.v1` — pairwise interactions among the top atoms (do they compose? KKT
   noncommutativity ordering).

## Canonical row schema (`evaluator_response_atlas_row.v1`)
{ pair_id, family, frame_incidence, parameter{...}, d_seg, d_pose, score_nonrate,
  seg_margin_p10/p50, pose_output_l2, boundary_flip_count, interior_flip_count, bytes_model,
  candidate_action_hint }  + axis tags ([macOS-CPU advisory] forward / [macOS-MLX research-signal] grad).

## MLX engine (saturate the M5 Max 128GB unified memory)
- Load all 600 pairs as resident MLX arrays (uint8 + float views); ~3.7 GB — fits easily.
- Cache source SegNet logits/argmax + PoseNet outputs ONCE.
- Compiled (`mx.compile`) perturbation kernels: blur / downsample / quantize / noise / freq masks /
  region masks. `mx.vmap` over (pair_id × level × frame_incidence). `mx.jvp`/`mx.vjp` for the local
  calculus (gradient atlas). Unified memory ⇒ no device copies; CPU/GPU streams via async eval.
- Memory telemetry (`mx.get_active_memory`/`get_peak_memory`/`set_memory_limit`/`clear_cache`) to pick
  atlas block size and saturate without OOM.
- FAITHFULNESS SPLIT: forward-only families use the EXACT torch contest scorer (CPU, faithful,
  promotable-adjacent). The gradient atlas + bulk vmap sweeps may use MLX scorer ports (fast,
  [macOS-MLX research-signal], NON-promotable) to FIND active subspaces/null spaces; key directions
  re-validated on the exact torch scorer. (MLX-portable-local-substrate-authority rule.)
- ALL long runs are nohup-detached daemons writing durable SSD artifacts (SIGURG-144 kills foreground
  AND background tool compute at ~3min; only detached daemons survive — re-confirmed 2026-06-09 twice).

## Sensitivity math (adapted high-dim tooling)
- **Active subspaces** (gradient-discovered dominant directions): the few directions along which d_seg/
  d_pose vary most ⇒ the evaluator-active subspace; its complement is the null space (free bytes).
- **Morris screening** (elementary effects): cheap ranking of region/frequency/YUV/action FACTORS as
  influential vs negligible, far fewer evals than full indices.
- **Sobol / variance decomposition**: for the top factors after Morris — variance attributable to each
  factor + interactions (the commutator atlas).

## Solve order (the final representation compiler)
1. NULL SPACE — perturbations with low Δd_seg AND low Δd_pose (free to spend zero bytes).
2. ACTIVE AXES — high score-sensitivity directions (bytes must protect these).
3. BYTE-CHEAP REPRESENTATIONS — masks / paths / semantic cells / low-rank bands / source-state atoms
   that span the active axes cheaply.
4. CANDIDATE ATOMS — each with (Δd_seg, Δd_pose, Δbytes) measured.
5. WATERFILL — admit atoms with ΔS < 0 (base-bound, anti-drift) via `tac.optimization.evaluator_action_waterfill`.
6. COMMUTATORS — decide which atoms compose (noncommutative ordering by ΔS/byte).
7. EMIT ARCHIVE PROGRAM — backend + selected atoms + inflate interpreter → full upstream evaluate.py →
   exact S → reseed the solver (meta-Lagrangian continual learning).

## Build order (incremental detached daemons; each lands a typed artifact)
- B0 (RUNNING): `segnet_margin_field.v1` over all 600 frames (deforestation map; forward-only, torch CPU).
- B1: promote `evaluator_cell_tolerance` → `evaluator_source_degradation_curves.v1` over all 600 pairs
  (+ chroma/luma/noise) + `evaluator_frame_incidence_curves.v1` (the SegNet/PoseNet asymmetry).
- B2: `evaluator_gradient_atlas.v1` — SegNet margin-gradient map (VJP) + PoseNet JᵀJ saliency
  (Hutchinson). Forward on torch; gradients on MLX scorer ports (research-signal) validated on torch.
- B3: `evaluator_frequency_sensitivity.v1` + `evaluator_region_waterfill_curves.v1`.
- B4: active-subspace + Morris + Sobol reduction → the evaluator-active basis + null space.
- B5: `evaluator_action_atom_curves.v1` + `evaluator_commutator_atlas.v1` → waterfill → archive program
  → full evaluate.py → reseed.

Goal: turn unlimited compression-time compute + MLX into a theoretical-floor search — the lowest-score
witness as the KKT/waterfilling solution of the exact evaluator response surface on the contest video.
This subsumes V1/V2/V3: any witness (NeRV/SNeRV/direct grammar) is scored by the same atlas; the
waterfilling picks the representation that spends the fewest bytes for the required margin + pose fidelity.
