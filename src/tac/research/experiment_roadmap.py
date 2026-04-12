"""Experiment roadmap for Yousfi's final push.

Structured catalog of every planned experiment across CPU and GPU lanes,
with dependencies, expected improvements, and current status. This file
is the single source of truth for experiment prioritization.

Usage::

    from tac.research.experiment_roadmap import ROADMAP, ready_experiments
    for exp in ready_experiments():
        print(f"{exp.priority} {exp.lane:3s} {exp.name}")
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Experiment:
    """A single planned experiment."""

    name: str
    lane: str  # "cpu" or "gpu"
    platform: str  # "local", "kaggle", "lightning", "paid_gpu"
    priority: int  # 1-5 (1=highest)
    dependencies: list[str] = field(default_factory=list)
    expected_improvement: str = ""
    estimated_runtime: str = ""
    eureka_source: str = ""
    status: str = "ready"  # "ready", "needs_work", "blocked", "pending_yousfi_review"
    blocking_issues: list[str] = field(default_factory=list)
    notes: str = ""

    # --- Execution details ---
    command: str = ""  # Exact command to run
    eval_command: str = ""  # How to evaluate results
    output_artifact: str = ""  # What the experiment produces

    # --- Platform requirements ---
    # min_vram_gb: 0 means CPU-only, 16 for P100/T4, 24 for A10G, 80 for A100/H100
    # needs_cuda: True if CUDA required (not just MPS-compatible)
    # needs_dali: True if NVIDIA DALI pipeline required
    platform_requirements: dict = field(default_factory=lambda: {
        "min_vram_gb": 0,
        "needs_cuda": False,
        "needs_dali": False,
    })

    # --- Smoke test config ---
    # Non-None if this experiment has a reduced-scale smoke variant.
    # frames: number of frames used in smoke (vs 1200 full).
    # steps: optimization steps in smoke (vs full).
    # purpose: what the smoke test validates.
    smoke_test_config: dict | None = None


def ready_experiments() -> list[Experiment]:
    """Return experiments with status 'ready', sorted by priority."""
    return sorted(
        [e for e in ROADMAP if e.status == "ready"],
        key=lambda e: e.priority,
    )


def gpu_experiments() -> list[Experiment]:
    """Return GPU-lane experiments sorted by priority."""
    return sorted(
        [e for e in ROADMAP if e.lane == "gpu"],
        key=lambda e: e.priority,
    )


def cpu_experiments() -> list[Experiment]:
    """Return CPU-lane experiments sorted by priority."""
    return sorted(
        [e for e in ROADMAP if e.lane == "cpu"],
        key=lambda e: e.priority,
    )


# ============================================================================
# THE ROADMAP
# ============================================================================

ROADMAP: list[Experiment] = [
    # ------------------------------------------------------------------
    # CPU Lane — Priority 1: CRF sweep
    # ------------------------------------------------------------------
    Experiment(
        name="crf_sweep_32_38",
        lane="cpu",
        platform="local",
        priority=1,
        dependencies=[
            "submissions/robust_current/postfilter_int8.pt",
            "ffmpeg with libx265",
        ],
        expected_improvement="0.05-0.15 score reduction (CRF 34 baseline -> optimal CRF)",
        estimated_runtime="~30 min (6 CRF values x 5 min encode+eval each)",
        eureka_source="yousfi_12_tricks: encoder parameter tuning",
        status="ready",
        command=(
            "for crf in 32 33 34 35 36 37 38; do\n"
            "  echo \"=== CRF $crf ===\"\n"
            "  ffmpeg -y -i input.y4m -c:v libx265 -preset medium "
            "-x265-params crf=$crf:keyint=-1 -pix_fmt yuv420p "
            "compressed_crf${crf}.mkv\n"
            "  # inflate + eval\n"
            "  INFLATE_POSTFILTER=1 python submissions/robust_current/inflate_postfilter.py "
            "compressed_crf${crf}.mkv inflated_crf${crf}/\n"
            "  python upstream/score.py inflated_crf${crf}/ "
            "| tee reports/raw/crf_sweep_${crf}.txt\n"
            "done"
        ),
        eval_command="grep 'total_score' reports/raw/crf_sweep_*.txt | sort -t= -k2 -n",
        output_artifact="reports/raw/crf_sweep_*.txt",
        notes=(
            "Step 1: Re-encode the ground truth video at each CRF.\n"
            "Step 2: Inflate with existing postfilter.\n"
            "Step 3: Run full authoritative eval.\n"
            "Lower CRF = higher quality + bigger file. Higher CRF = smaller file + worse quality.\n"
            "Sweet spot is where score = 100*seg + sqrt(10*pose) + 25*rate is minimized."
        ),
    ),

    # ------------------------------------------------------------------
    # CPU Lane — Priority 1: Quantization-directed rounding
    # ------------------------------------------------------------------
    Experiment(
        name="quantization_directed_rounding",
        lane="cpu",
        platform="local",
        priority=1,
        dependencies=[
            "submissions/robust_current/postfilter_int8.pt",
            "scorer models (PoseNet, SegNet) from upstream/models/",
        ],
        expected_improvement="0.02-0.05 score reduction (free at inflate time)",
        estimated_runtime="~15 min precompute on CPU, 0 extra inflate cost",
        eureka_source="eureka: quantization-directed rounding",
        status="ready",
        command=(
            "# Precompute at compress time:\n"
            "python -c \"\n"
            "from tac.precompute_corrections import compute_quantization_directions\n"
            "import torch\n"
            "frames = torch.load('inflated_frames.pt')  # (N, 3, H, W)\n"
            "dirs = compute_quantization_directions(\n"
            "    frames, posenet, segnet, device='cpu',\n"
            "    seg_weight=100.0, pose_weight=1.0,\n"
            ")\n"
            "torch.save(dirs, 'quant_directions.pt')\n"
            "\"\n"
            "# Apply at inflate time:\n"
            "python -c \"\n"
            "from tac.precompute_corrections import apply_quantization_directions\n"
            "import torch\n"
            "frames = torch.load('inflated_frames.pt')\n"
            "dirs = torch.load('quant_directions.pt')\n"
            "rounded = apply_quantization_directions(frames, dirs)\n"
            "\""
        ),
        eval_command="python upstream/score.py inflated_qround/ | tee reports/raw/qround_eval.txt",
        output_artifact="quant_directions.pt (stored in archive.zip)",
        notes=(
            "Scorer gradient sign at each pixel tells us whether to ceil or floor.\n"
            "Directions stored as compressed bitmask (~20KB in archive).\n"
            "Must be inside archive.zip per contest rules (affects rate).\n"
            "Integration: add to inflate_postfilter.py after postfilter, before final write."
        ),
    ),

    # ------------------------------------------------------------------
    # CPU Lane — Priority 2: Non-local means deblocking
    # ------------------------------------------------------------------
    Experiment(
        name="nonlocal_means_deblock",
        lane="cpu",
        platform="local",
        priority=2,
        dependencies=["opencv-python (cv2)"],
        expected_improvement="0.01-0.03 score reduction (cleaner postfilter input)",
        estimated_runtime="~2 min inflate overhead for 1200 frames",
        eureka_source="eureka: non-local means deblocking",
        status="ready",
        command=(
            "# Enable via environment variable:\n"
            "INFLATE_DEBLOCK=1 INFLATE_POSTFILTER=1 "
            "python submissions/robust_current/inflate_postfilter.py "
            "compressed.mkv inflated_deblock/"
        ),
        eval_command="python upstream/score.py inflated_deblock/ | tee reports/raw/deblock_eval.txt",
        output_artifact="inflated_deblock/ directory",
        notes=(
            "Zero archive cost (no stored parameters).\n"
            "Risk: adds ~2 min to inflate time. 10-min budget has ~7 min slack.\n"
            "Runs BEFORE postfilter. Parameters: h=10, template_window=7, search_window=21.\n"
            "If it helps, sweep h in {5, 10, 15, 20} to find optimal strength."
        ),
    ),

    # ------------------------------------------------------------------
    # GPU Lane — Priority 1: Coupled trajectory optimization
    # ------------------------------------------------------------------
    Experiment(
        name="coupled_trajectory_optimize",
        lane="gpu",
        platform="kaggle",
        priority=1,
        dependencies=[
            "scorer models (PoseNet, SegNet)",
            "mask data from existing archive",
            "pose targets from existing archive",
            "Kaggle P100 GPU (free tier)",
        ],
        expected_improvement="0.1-0.3 score reduction (joint optimization of all frames)",
        estimated_runtime="~45 min on P100 (1000 steps, 50ms/step, 1200 frames)",
        eureka_source="eureka: coupled trajectory optimization (4D-Var)",
        status="ready",
        platform_requirements={"min_vram_gb": 16, "needs_cuda": True, "needs_dali": False},
        smoke_test_config={"frames": 8, "steps": 100, "purpose": "viability on P100 hardware"},
        command=(
            "python -c \"\n"
            "from tac.constrained_gen import coupled_trajectory_optimize\n"
            "import torch\n"
            "masks = torch.load('masks.pt').to('cuda')  # (N, H, W)\n"
            "pose = torch.load('pose_targets.pt').to('cuda')  # (N-1, 6)\n"
            "frames = coupled_trajectory_optimize(\n"
            "    masks, pose, posenet, segnet,\n"
            "    num_steps=1000, lr=0.01,\n"
            "    seg_weight=100.0, pose_weight=10.0, compress_weight=1.0,\n"
            "    device='cuda', log_every=100,\n"
            ")\n"
            "torch.save(frames, 'coupled_frames.pt')\n"
            "\""
        ),
        eval_command=(
            "# Build archive from output frames:\n"
            "python experiments/build_archive.py coupled_frames.pt archive_coupled.zip\n"
            "# Inflate and eval:\n"
            "python submissions/robust_current/inflate_postfilter.py archive_coupled.zip inflated_coupled/\n"
            "python upstream/score.py inflated_coupled/"
        ),
        output_artifact="coupled_frames.pt",
        notes=(
            "Key insight: frames are NOT independent. PoseNet evaluates consecutive PAIRS.\n"
            "Frame t affects pair (t-1,t) AND pair (t,t+1). Joint optimization lets\n"
            "gradient flow through the entire coupled system.\n"
            "This is the highest-expected-value GPU experiment."
        ),
    ),

    # ------------------------------------------------------------------
    # GPU Lane — Priority 2: Newton/L-BFGS refinement
    # ------------------------------------------------------------------
    Experiment(
        name="newton_lbfgs_optimize",
        lane="gpu",
        platform="kaggle",
        priority=2,
        dependencies=[
            "scorer models (PoseNet, SegNet)",
            "coupled_trajectory output (or existing frames as warm start)",
            "Kaggle P100 GPU",
        ],
        expected_improvement="0.02-0.1 further reduction (second-order convergence)",
        estimated_runtime="~15 min on P100 (3 Newton steps x 20 evaluations each)",
        eureka_source="eureka: Newton/L-BFGS second-order optimization",
        status="ready",
        platform_requirements={"min_vram_gb": 16, "needs_cuda": True, "needs_dali": False},
        smoke_test_config={"frames": 8, "steps": 5, "purpose": "L-BFGS convergence check"},
        command=(
            "python -c \"\n"
            "from tac.constrained_gen import newton_step_optimize\n"
            "import torch\n"
            "frames = torch.load('coupled_frames.pt').to('cuda')  # warm start\n"
            "masks = torch.load('masks.pt').to('cuda')\n"
            "pose = torch.load('pose_targets.pt').to('cuda')\n"
            "refined = newton_step_optimize(\n"
            "    frames, posenet, segnet,\n"
            "    masks=masks, expected_pose=pose,\n"
            "    num_newton_steps=3, max_iter_per_step=20,\n"
            "    lr=1.0, history_size=10,\n"
            "    seg_weight=100.0, pose_weight=10.0, compress_weight=1.0,\n"
            "    device='cuda',\n"
            ")\n"
            "torch.save(refined, 'newton_frames.pt')\n"
            "\""
        ),
        eval_command="python upstream/score.py inflated_newton/",
        output_artifact="newton_frames.pt",
        notes=(
            "Run AFTER coupled_trajectory proves viability.\n"
            "L-BFGS converges in 1-3 steps where Adam needs 1000.\n"
            "Uses implicit Hessian approximation (rank-10 quasi-Newton).\n"
            "Line search with strong Wolfe conditions for guaranteed descent."
        ),
    ),

    # ------------------------------------------------------------------
    # CPU Lane — Priority 2: Trick stacking on existing 1.33 checkpoint
    # ------------------------------------------------------------------
    Experiment(
        name="trick_stack_existing_checkpoint",
        lane="cpu",
        platform="local",
        priority=2,
        dependencies=[
            "submissions/robust_current/postfilter_int8.pt",
            "scorer models",
            "opencv-python",
        ],
        expected_improvement="0.05-0.15 from stacking CRF + deblock + qround",
        estimated_runtime="~45 min (CRF best + deblock + quantization rounding)",
        eureka_source="trick stacking: combine independent improvements",
        status="ready",
        command=(
            "# Stack order: CRF best -> deblock -> postfilter -> qround\n"
            "# Step 1: Use best CRF from sweep\n"
            "# Step 2: INFLATE_DEBLOCK=1 INFLATE_POSTFILTER=1\n"
            "# Step 3: Apply quantization directions\n"
            "INFLATE_DEBLOCK=1 INFLATE_POSTFILTER=1 "
            "python submissions/robust_current/inflate_postfilter.py "
            "compressed_best_crf.mkv inflated_stacked/"
        ),
        eval_command="python upstream/score.py inflated_stacked/",
        output_artifact="inflated_stacked/",
        notes=(
            "YES, test trick stacking on existing 1.33 checkpoint.\n"
            "Reason: each trick is independent and additive.\n"
            "Stack order matters: CRF first (encoder), deblock second (pre-postfilter),\n"
            "postfilter third (learned), qround last (rounding).\n"
            "Expected: if CRF saves 0.05, deblock saves 0.02, qround saves 0.03,\n"
            "stacking gets ~0.08-0.10 (not fully additive due to interactions)."
        ),
    ),

    # ------------------------------------------------------------------
    # GPU Lane — Priority 3: Scorer-as-compressor
    # ------------------------------------------------------------------
    Experiment(
        name="scorer_as_compressor",
        lane="gpu",
        platform="kaggle",
        priority=3,
        dependencies=[
            "scorer models",
            "ground truth frames",
        ],
        expected_improvement="0.1-0.2 (store scorer outputs as sufficient statistic)",
        estimated_runtime="~20 min on P100",
        eureka_source="eureka: scorer as compressor",
        status="ready",
        platform_requirements={"min_vram_gb": 16, "needs_cuda": True, "needs_dali": False},
        command=(
            "python -c \"\n"
            "from tac.constrained_gen import scorer_as_compressor\n"
            "import torch\n"
            "frames = torch.load('gt_frames.pt')  # (N, H, W, 3)\n"
            "stats = scorer_as_compressor(frames, posenet, segnet, device='cuda', topk=2)\n"
            "torch.save(stats, 'scorer_sufficient_stats.pt')\n"
            "\""
        ),
        eval_command="python upstream/score.py inflated_scorer_compress/",
        output_artifact="scorer_sufficient_stats.pt",
        notes=(
            "The scorer networks already learned optimal compression for driving video.\n"
            "PoseNet: 2 frames -> 6 numbers. SegNet: 1 frame -> 96x128x5 logits.\n"
            "Total archive: ~15KB. At inflate time, find frames matching these outputs."
        ),
    ),

    # ------------------------------------------------------------------
    # GPU Lane — Priority 3: Alternating projections (Dykstra)
    # ------------------------------------------------------------------
    Experiment(
        name="alternating_projections",
        lane="gpu",
        platform="kaggle",
        priority=3,
        dependencies=["scorer models", "masks", "pose targets"],
        expected_improvement="0.05-0.15 (constraint satisfaction via ADMM-like)",
        estimated_runtime="~60 min on P100 (100 outer x 10 inner steps)",
        eureka_source="eureka: Dykstra alternating projections",
        status="pending_yousfi_review",
        platform_requirements={"min_vram_gb": 16, "needs_cuda": True, "needs_dali": False},
        smoke_test_config={"frames": 8, "steps": 10, "purpose": "ADMM convergence check"},
        command=(
            "python -c \"\n"
            "from tac.constrained_gen import alternating_projections_optimize\n"
            "frames = alternating_projections_optimize(\n"
            "    masks, pose, posenet, segnet,\n"
            "    num_outer_iterations=100, num_inner_steps=10,\n"
            "    lr=0.05, seg_weight=100.0, pose_weight=10.0,\n"
            "    device='cuda',\n"
            ")\n"
            "\""
        ),
        eval_command="python upstream/score.py inflated_altproj/",
        output_artifact="alternating_frames.pt",
        notes=(
            "Contrarian's 'maybe keep alive' item.\n"
            "Theoretically sound for convex constraints; our constraints are quasi-convex.\n"
            "Run only if coupled_trajectory leaves room for improvement."
        ),
    ),

    # ------------------------------------------------------------------
    # GPU Lane — Priority 4: Hamiltonian dynamics
    # ------------------------------------------------------------------
    Experiment(
        name="hamiltonian_dynamics",
        lane="gpu",
        platform="kaggle",
        priority=4,
        dependencies=["scorer models", "initial frames"],
        expected_improvement="unknown (exploratory, symplectic structure may help escape local minima)",
        estimated_runtime="~30 min on P100 (500 leapfrog steps)",
        eureka_source="eureka: Hamiltonian dynamics pixel optimizer",
        status="pending_yousfi_review",
        platform_requirements={"min_vram_gb": 16, "needs_cuda": True, "needs_dali": False},
        command=(
            "python -c \"\n"
            "from tac.contrib.hamiltonian_dynamics import HamiltonianPixelOptimizer\n"
            "opt = HamiltonianPixelOptimizer(cfg={'hamiltonian_steps': 500})\n"
            "frames, diag = opt.optimize_with_scorer(\n"
            "    init_frames, masks, posenet, segnet,\n"
            "    seg_weight=100.0, pose_weight=10.0,\n"
            ")\n"
            "\""
        ),
        eval_command="python upstream/score.py inflated_hamiltonian/",
        output_artifact="hamiltonian_frames.pt",
        notes=(
            "Contrarian's 'maybe keep alive' item.\n"
            "Symplectic integrator preserves energy exactly, preventing divergence.\n"
            "Nose-Hoover thermostat variant available for temperature control.\n"
            "Interesting for exploration but coupled_trajectory is likely better."
        ),
    ),

    # ------------------------------------------------------------------
    # GPU Lane — Priority 4: Variational frame generation
    # ------------------------------------------------------------------
    Experiment(
        name="variational_frame_generation",
        lane="gpu",
        platform="kaggle",
        priority=4,
        dependencies=["scorer models", "masks"],
        expected_improvement="0.05-0.1 (PDE-based smoothness + scorer matching)",
        estimated_runtime="~30 min on P100",
        eureka_source="eureka: Euler-Lagrange variational methods",
        status="pending_yousfi_review",
        platform_requirements={"min_vram_gb": 16, "needs_cuda": True, "needs_dali": False},
        command=(
            "python -c \"\n"
            "from tac.contrib.variational_gen import VariationalFrameGenerator\n"
            "gen = VariationalFrameGenerator()\n"
            "frames = gen.generate(masks, posenet, segnet, num_steps=500)\n"
            "\""
        ),
        eval_command="python upstream/score.py inflated_variational/",
        output_artifact="variational_frames.pt",
        notes="PDE smoothness priors may complement gradient-based methods.",
    ),

    # ------------------------------------------------------------------
    # GPU Lane — Priority 4: Lagrangian dual optimization
    # ------------------------------------------------------------------
    Experiment(
        name="lagrangian_dual",
        lane="gpu",
        platform="kaggle",
        priority=4,
        dependencies=["scorer models", "masks"],
        expected_improvement="0.05-0.15 (principled multi-objective balancing)",
        estimated_runtime="~45 min on P100",
        eureka_source="eureka: Lagrangian dual optimization",
        status="pending_yousfi_review",
        platform_requirements={"min_vram_gb": 16, "needs_cuda": True, "needs_dali": False},
        command=(
            "python -c \"\n"
            "from tac.contrib.variational_gen import LagrangianDualOptimizer\n"
            "opt = LagrangianDualOptimizer()\n"
            "frames = opt.optimize(masks, posenet, segnet, num_steps=500)\n"
            "\""
        ),
        eval_command="python upstream/score.py inflated_lagrangian/",
        output_artifact="lagrangian_frames.pt",
        notes=(
            "Learns Lagrange multipliers for each constraint dynamically.\n"
            "Could find better weight balance than static seg_weight/pose_weight."
        ),
    ),

    # ------------------------------------------------------------------
    # GPU Lane — Priority 4: Scorer manifold geodesic
    # ------------------------------------------------------------------
    Experiment(
        name="scorer_manifold_geodesic",
        lane="gpu",
        platform="kaggle",
        priority=4,
        dependencies=["scorer models", "initial frames"],
        expected_improvement="unknown (find iso-score frames with minimum rate)",
        estimated_runtime="~30 min on P100",
        eureka_source="eureka: scorer manifold differential geometry",
        status="pending_yousfi_review",
        platform_requirements={"min_vram_gb": 16, "needs_cuda": True, "needs_dali": False},
        command=(
            "python -c \"\n"
            "from tac.contrib.scorer_manifold import ScorerManifold\n"
            "manifold = ScorerManifold(posenet, segnet)\n"
            "# Project frames to iso-score manifold, then optimize rate\n"
            "\""
        ),
        eval_command="python upstream/score.py inflated_manifold/",
        output_artifact="manifold_frames.pt",
        notes=(
            "Differential geometry approach: find the frame on the iso-score surface\n"
            "that has minimum total variation (= minimum rate after H.265 encoding)."
        ),
    ),

    # ------------------------------------------------------------------
    # CPU Lane — Priority 3: Brightness shift exploit
    # ------------------------------------------------------------------
    Experiment(
        name="brightness_shift_exploit",
        lane="cpu",
        platform="local",
        priority=3,
        dependencies=["scorer models"],
        expected_improvement="0.005-0.02 (free: no archive cost)",
        estimated_runtime="~5 min grid search",
        eureka_source="yousfi_12_tricks: exploit PoseNet blind spots",
        status="ready",
        command=(
            "python -c \"\n"
            "from tac.scorer_exploits import apply_global_brightness_shift\n"
            "# Sweep shift in [-10, -5, 0, 5, 10] and eval each\n"
            "\""
        ),
        eval_command="python upstream/score.py inflated_bright/",
        output_artifact="optimal_brightness_shift value",
        notes="PoseNet has known blind spots for global brightness shifts.",
    ),

    # ------------------------------------------------------------------
    # CPU Lane — Priority 3: BatchNorm style matching
    # ------------------------------------------------------------------
    Experiment(
        name="batchnorm_style_matching",
        lane="cpu",
        platform="local",
        priority=3,
        dependencies=["scorer models"],
        expected_improvement="0.01-0.03 (match training distribution statistics)",
        estimated_runtime="~10 min",
        eureka_source="eureka: Gatys-style BN matching for scorer",
        status="ready",
        command=(
            "python -c \"\n"
            "from tac.scorer_exploits import extract_batchnorm_statistics, batchnorm_style_loss\n"
            "stats = extract_batchnorm_statistics(posenet)\n"
            "# Use as auxiliary loss during postfilter TTO\n"
            "\""
        ),
        eval_command="python upstream/score.py inflated_bn_match/",
        output_artifact="BN-matched frames",
        notes="Frames whose patch statistics match training distribution get lower distortion.",
    ),

    # ------------------------------------------------------------------
    # CPU Lane — Priority 3: Scorer null space projection
    # ------------------------------------------------------------------
    Experiment(
        name="scorer_null_space",
        lane="cpu",
        platform="local",
        priority=3,
        dependencies=["scorer models"],
        expected_improvement="0.01-0.05 (move in scorer-invisible directions for rate)",
        estimated_runtime="~15 min for Jacobian SVD",
        eureka_source="eureka: scorer null space exploitation",
        status="ready",
        command=(
            "python -c \"\n"
            "from tac.scorer_exploits import project_to_scorer_null_space\n"
            "# Project pixel perturbations into scorer null space\n"
            "# to reduce rate without affecting score\n"
            "\""
        ),
        eval_command="python upstream/score.py inflated_nullspace/",
        output_artifact="null_space_projected frames",
        notes="Jacobian SVD identifies directions invisible to scorer.",
    ),

    # ------------------------------------------------------------------
    # GPU Lane — Priority 5: Cross-disciplinary optimizers (exploration)
    # ------------------------------------------------------------------
    Experiment(
        name="cross_disciplinary_ensemble",
        lane="gpu",
        platform="kaggle",
        priority=5,
        dependencies=["scorer models", "initial frames"],
        expected_improvement="unknown (portfolio of physics-inspired optimizers)",
        estimated_runtime="~60 min on P100",
        eureka_source="eureka: cross-disciplinary optimization portfolio",
        status="pending_yousfi_review",
        platform_requirements={"min_vram_gb": 16, "needs_cuda": True, "needs_dali": False},
        smoke_test_config={"frames": 8, "steps": 10, "purpose": "verify all optimizers run"},
        blocking_issues=["17 optimizer variants, need to select top 3-5 for ensemble"],
        command=(
            "python -c \"\n"
            "from tac.contrib.cross_disciplinary_optimizers import ensemble_optimize\n"
            "frames = ensemble_optimize(init_frames, posenet, segnet, masks, device='cuda')\n"
            "\""
        ),
        eval_command="python upstream/score.py inflated_ensemble/",
        output_artifact="ensemble_frames.pt",
        notes=(
            "Includes: simulated annealing, HMC, Langevin, CMA-ES, DE, PSO,\n"
            "metadynamics, basin hopping, FWI, ensemble Kalman, 4D-Var, etc.\n"
            "Run smoke_test_all() first to verify all optimizers work."
        ),
    ),

    # ------------------------------------------------------------------
    # GPU Lane — Priority 5: Finance-inspired optimizers
    # ------------------------------------------------------------------
    Experiment(
        name="finance_optimizers",
        lane="gpu",
        platform="kaggle",
        priority=5,
        dependencies=["scorer models"],
        expected_improvement="unknown (novel optimization angles)",
        estimated_runtime="~30 min on P100",
        eureka_source="eureka: finance/HFT-inspired optimization",
        status="pending_yousfi_review",
        platform_requirements={"min_vram_gb": 16, "needs_cuda": True, "needs_dali": False},
        blocking_issues=["need to identify which finance analogies actually apply"],
        command=(
            "python -c \"\n"
            "from tac.contrib.finance_optimizers import yousfi_contrarian_picks\n"
            "picks = yousfi_contrarian_picks(device='cpu')\n"
            "# Run the recommended subset\n"
            "\""
        ),
        eval_command="python upstream/score.py inflated_finance/",
        output_artifact="finance_optimized_frames.pt",
        notes=(
            "Contrarian's 'maybe keep alive' item.\n"
            "Includes: Almgren-Chriss (optimal execution), Kelly criterion,\n"
            "implied volatility, Markowitz portfolio, pairs trading analogies."
        ),
    ),

    # ------------------------------------------------------------------
    # CPU Lane — Priority 3: Domain-specific solvers
    # ------------------------------------------------------------------
    Experiment(
        name="domain_solver_ego_motion",
        lane="cpu",
        platform="local",
        priority=3,
        dependencies=["masks"],
        expected_improvement="0.02-0.05 (ego-motion constrained flow for initialization)",
        estimated_runtime="~5 min on CPU",
        eureka_source="eureka: domain-specific solvers (ego-motion, homography)",
        status="ready",
        command=(
            "python -c \"\n"
            "from tac.contrib.domain_solvers import EgoMotionFlowSolver, RoadPlaneHomography\n"
            "solver = EgoMotionFlowSolver()\n"
            "frames = solver.generate(masks)\n"
            "# Use as initialization for coupled_trajectory_optimize\n"
            "\""
        ),
        eval_command="Use as warm start, then eval the downstream optimization",
        output_artifact="ego_motion_init_frames.pt",
        notes="Use as better initialization for GPU-lane optimizers instead of noise.",
    ),

    # ------------------------------------------------------------------
    # CPU Lane — Priority 4: Kalman smoother refinement
    # ------------------------------------------------------------------
    Experiment(
        name="kalman_smoother",
        lane="cpu",
        platform="local",
        priority=4,
        dependencies=["initial frames (from any source)"],
        expected_improvement="0.01-0.03 (optimal temporal smoothing)",
        estimated_runtime="~10 min on CPU",
        eureka_source="eureka: RTS Kalman smoother in PCA space",
        status="pending_yousfi_review",
        command=(
            "python -c \"\n"
            "from tac.contrib.domain_solvers import KalmanFrameSmoother\n"
            "smoother = KalmanFrameSmoother()\n"
            "smoothed = smoother.smooth(frames)\n"
            "\""
        ),
        eval_command="python upstream/score.py inflated_kalman/",
        output_artifact="kalman_smoothed_frames.pt",
        notes="Post-processing: apply after any frame generation for temporal coherence.",
    ),

    # ------------------------------------------------------------------
    # CPU Lane — Priority 2: Precomputed corrections (full pipeline)
    # ------------------------------------------------------------------
    Experiment(
        name="precomputed_corrections_full",
        lane="cpu",
        platform="local",
        priority=2,
        dependencies=["scorer models", "existing inflated frames"],
        expected_improvement="0.03-0.08 (combined corrections: gradients + null space + fragility)",
        estimated_runtime="~20 min on CPU",
        eureka_source="eureka: precomputed corrections pipeline",
        status="ready",
        command=(
            "python -c \"\n"
            "from tac.precompute_corrections import precompute_all_corrections, save_corrections\n"
            "corrections = precompute_all_corrections(\n"
            "    frames, posenet, segnet, device='cpu',\n"
            ")\n"
            "save_corrections(corrections, 'corrections.bin')\n"
            "\""
        ),
        eval_command="python upstream/score.py inflated_corrected/",
        output_artifact="corrections.bin (stored in archive.zip)",
        notes="Must fit in archive size budget. Includes gradient directions, null space, fragility maps.",
    ),
]
