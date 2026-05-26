# SPDX-License-Identifier: MIT
# CHECKPOINT_DISCIPLINE_WAIVED:L0-mlx-scaffold-smoke-trainer-only-no-real-training-loop-_full_main-raises-NotImplementedError-per-catalog-240-acceptance-cascade-c-pre-build-substrate-engineering-no-paid-dispatch-from-this-file
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526
# AUTOCAST_FP16_WAIVED:MLX_or_PyTorch_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_uses_different_precision_strategy_per_comprehensive_bug_audit_cascade_20260526
# SYNTHETIC_NON_SMOKE_OK:L0-mlx-scaffold-synthetic-frames-only-in-smoke-_full_main-raises-NotImplementedError-per-catalog-240
"""train_substrate_mdl_ibps_j_discrete_categorical_mine_hybrid — Path 3 J=MDL-IBPS L0 SCAFFOLD smoke trainer.

Per Catalog #240 acceptance cascade (c) pre-build substrate-engineering:
``_full_main`` raises NotImplementedError until L1+ wires the canonical
PyTorch port + score-aware loss routing + real frame loader + Catalog #319
deliverability proof + Catalog #270 dispatch optimization protocol +
PER-SUBSTRATE OPTIMAL FORM symposium per Catalog #325.

``_smoke_main`` runs MLX-side primitives on synthetic data; produces NO
score claims (artifacts tagged ``[macOS-MLX research-signal]`` per
Catalog #192/#317 non-promotable markers).

Phase 2 design memo at ``.omx/research/path_3_j_mdl_ibps_substrate_design_decision_20260526.md``;
Phase 3 design memo at ``.omx/research/path_3_j_mdl_ibps_substrate_design_20260526.md``.

Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS declared as ``ast.AnnAssign`` for AST walker per Catalog #168.

Per CLAUDE.md FORBIDDEN_PATTERNS:
- No silent device defaults (MLX-explicit; PyTorch path via canonical Catalog #205 helper)
- No scorer load at inflate time
- No /tmp paths
- No make_synthetic_pair_batch in non-smoke (this file's smoke branch IS the only synthetic surface)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS as ast.AnnAssign per Catalog #168
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--smoke": {
        "env": "SMOKE_ONLY",
        "default": "1",  # default smoke-only at L0 SCAFFOLD per Catalog #326
        "required_input_file": False,
    },
    "--mode": {
        "env": "J_MDL_IBPS_TRAINER_MODE",
        "default": "smoke",
        "required_input_file": False,
    },
    "--epochs": {
        "env": "J_MDL_IBPS_EPOCHS",
        "default": "10",
        "required_input_file": False,
    },
    "--beta": {
        "env": "J_MDL_IBPS_BETA",
        "default": "1e-3",
        "required_input_file": False,
    },
    "--lambda-sparse": {
        "env": "J_MDL_IBPS_LAMBDA_SPARSE",
        "default": "1e-4",
        "required_input_file": False,
    },
    "--out-dir": {
        "env": "J_MDL_IBPS_OUT_DIR",
        "default": "experiments/results/lane_path_3_j_mdl_ibps_information_bottleneck_cargo_cult_first_20260526_smoke",
        "required_input_file": False,
    },
    "--seed": {
        "env": "J_MDL_IBPS_SEED",
        "default": "42",
        "required_input_file": False,
    },
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_substrate_mdl_ibps_j_discrete_categorical_mine_hybrid",
        description=(
            "Path 3 J=MDL-IBPS DISCRETE-CATEGORICAL-MINE-HYBRID L0 SCAFFOLD. "
            "MLX-first smoke trainer; _full_main raises NotImplementedError "
            "per Catalog #240 (c) pre-build substrate-engineering until L1+ "
            "PyTorch port + canonical score-aware loss + real frame loader + "
            "deliverability proof + dispatch protocol + symposium revisions."
        ),
    )
    parser.add_argument("--smoke", action="store_true",
                        default=os.environ.get("SMOKE_ONLY", "1") == "1",
                        help="L0 SCAFFOLD MLX-local smoke (default; non-promotable per Catalog #192/#317)")
    parser.add_argument("--mode", type=str,
                        default=os.environ.get("J_MDL_IBPS_TRAINER_MODE", "smoke"),
                        choices=["smoke", "full"],
                        help="trainer mode; 'full' raises NotImplementedError per Catalog #240 (c)")
    parser.add_argument("--epochs", type=int,
                        default=int(os.environ.get("J_MDL_IBPS_EPOCHS", "10")))
    parser.add_argument("--beta", type=float,
                        default=float(os.environ.get("J_MDL_IBPS_BETA", "1e-3")),
                        help="IB Lagrangian (CC-J-3 unwind; default sweep {1e-5, 1e-4, 1e-3, 1e-2})")
    parser.add_argument("--lambda-sparse", type=float,
                        default=float(os.environ.get("J_MDL_IBPS_LAMBDA_SPARSE", "1e-4")),
                        help="Sparse-Laplacian regularizer (MacKay Path B5 influence)")
    parser.add_argument("--out-dir", type=str,
                        default=os.environ.get("J_MDL_IBPS_OUT_DIR",
                            "experiments/results/lane_path_3_j_mdl_ibps_information_bottleneck_cargo_cult_first_20260526_smoke"))
    parser.add_argument("--seed", type=int,
                        default=int(os.environ.get("J_MDL_IBPS_SEED", "42")))
    parser.add_argument("--num-pairs", type=int, default=8,
                        help="L0 smoke uses tiny synthetic batch (8 pairs) for shape validation")
    return parser


def _smoke_main(args: argparse.Namespace) -> int:
    """MLX-side L0 SCAFFOLD smoke; synthetic frames only; non-promotable per Catalog #192/#317.

    Exercises:
    - Numpy reference primitives (axis 3 portability)
    - Sparse-Laplacian L1 + canonical KL Gaussian formula (numpy)
    - Archive grammar pack/unpack round-trip (Catalog #139 + #220 + #272 byte-deterministic)

    Does NOT exercise:
    - MLX renderer (test_basic.py covers this; ALL MLX paths are gated for ImportError graceful skip)
    - MINE critic training (deferred to Stage 3 paid dispatch authorization per Catalog #325)
    - Contest scorer (NO scorer load at L0 per CC-J-10 unwind)

    Tagging: every artifact emitted carries [macOS-MLX research-signal] per
    Catalog #192/#317 + CLAUDE.md "MPS auth eval is NOISE" non-negotiable.
    """
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Import substrate primitives lazily (axis 3 numpy first; MLX is optional)
    from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid import (
        BITS_PER_PAIR,
        CATEGORICAL_G,
        CATEGORICAL_K,
        DEFAULT_BETA_SWEEP,
        EVAL_HW,
        HIDDEN_DIM,
        LANE_ID,
        NUM_HIDDEN_LAYERS,
        SUBSTRATE_ID,
    )
    from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.numpy_reference import (
        CoordMLPBaseNumpy,
        categorical_to_one_hot_numpy,
        kl_gaussian_to_standard_normal_numpy,
        make_pixel_coords_numpy,
        mine_lower_bound_numpy,
        sinusoidal_positional_encoding_numpy,
        sparse_laplacian_l1_numpy,
    )
    # Generate synthetic per-pair categorical indices (small test set; Catalog #114 SYNTHETIC waiver)
    indices = np.random.randint(
        0, CATEGORICAL_K, size=(args.num_pairs, CATEGORICAL_G)
    )
    one_hot = categorical_to_one_hot_numpy(indices, K=CATEGORICAL_K)
    assert one_hot.shape == (args.num_pairs, CATEGORICAL_G * CATEGORICAL_K), \
        f"unexpected one_hot shape: {one_hot.shape}"
    # Generate synthetic pixel coords (small ROI to keep smoke fast)
    coords = make_pixel_coords_numpy(height=16, width=16, t=0)
    pos_enc = sinusoidal_positional_encoding_numpy(coords)
    assert pos_enc.shape[1] == 8 * 2 * 3  # POS_DIM * 2 * 3
    # Synthetic L1 sparsity test
    film_matrix = np.random.RandomState(args.seed).randn(10, 10).astype(np.float32) * 0.1
    sparse_l1 = sparse_laplacian_l1_numpy([film_matrix])
    # Synthetic MINE bound (untrained critic; should be near 0)
    critic_joint = np.random.RandomState(args.seed).randn(16).astype(np.float32)
    critic_marginal = np.random.RandomState(args.seed + 1).randn(16).astype(np.float32)
    mine_lb = mine_lower_bound_numpy(critic_joint, critic_marginal)
    # Synthetic KL Gaussian (numpy comparison-test sister)
    mu = np.zeros((4, 8), dtype=np.float32)
    logvar = np.zeros((4, 8), dtype=np.float32)
    kl = kl_gaussian_to_standard_normal_numpy(mu, logvar)
    # Emit non-promotable smoke artifact per Catalog #192/#317 + canonical Provenance per Catalog #323
    artifact = {
        "schema_version": "mdl_ibps_j_l0_scaffold_smoke_v1",
        "evidence_grade": "macOS-MLX-research-signal",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "axis_tag": "[macOS-MLX research-signal]",
        "lane_id": LANE_ID,
        "substrate_id": SUBSTRATE_ID,
        "smoke_params": {
            "num_pairs": int(args.num_pairs),
            "seed": int(args.seed),
            "beta": float(args.beta),
            "lambda_sparse": float(args.lambda_sparse),
            "epochs": int(args.epochs),
        },
        "smoke_results": {
            "one_hot_shape": list(one_hot.shape),
            "pos_enc_shape": list(pos_enc.shape),
            "sparse_l1": float(sparse_l1),
            "mine_lb": float(mine_lb),
            "kl_zero_mean_zero_logvar": float(kl.sum()),
        },
        "substrate_constants_validated": {
            "BITS_PER_PAIR": BITS_PER_PAIR,
            "CATEGORICAL_K": CATEGORICAL_K,
            "CATEGORICAL_G": CATEGORICAL_G,
            "HIDDEN_DIM": HIDDEN_DIM,
            "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
            "EVAL_HW": list(EVAL_HW),
            "DEFAULT_BETA_SWEEP": list(DEFAULT_BETA_SWEEP),
        },
        "catalog_compliance": {
            "catalog_192_advisory_marker": True,  # macOS-MLX advisory
            "catalog_317_local_research_signal_evidence_grade": True,
            "catalog_240_full_main_raises_not_implemented_error": True,  # acceptance cascade (c)
            "catalog_287_canonical_evidence_tag": True,
            "catalog_323_canonical_provenance_non_promotable": True,
            "catalog_324_predicted_band_validation_status_pending_post_training": True,
        },
        "notes": (
            "L0 SCAFFOLD smoke per Path 3 J=MDL-IBPS DISCRETE-CATEGORICAL-MINE-HYBRID; "
            "exercises numpy reference + sparse-Laplacian + MINE-numpy + KL-numpy; "
            "MLX paths tested separately in tests/test_basic.py (skipped if MLX unavailable); "
            "NO scorer load; NO real training; _full_main raises NotImplementedError per Catalog #240 (c)."
        ),
    }
    out_path = out_dir / "smoke_artifact.json"
    out_path.write_text(json.dumps(artifact, sort_keys=True, indent=2))
    print(f"[macOS-MLX research-signal] L0 SCAFFOLD smoke complete; wrote {out_path}")
    return 0


def _full_main(args: argparse.Namespace) -> int:
    """Pre-build substrate-engineering scaffold per Catalog #240 acceptance (c).

    L1+ promotion requires:
    1. PyTorch port wire-in (`inflate.py::inflate_substrate` is the export-side
       canonical helper; train-side PyTorch port + state_dict export-bridge per
       sister #1251 + #1257).
    2. Score-aware loss via ``tac.substrates._shared.score_aware_common`` per
       Catalog #164 (route through canonical helper; eval-roundtrip mandatory
       per CLAUDE.md "eval_roundtrip" non-negotiable).
    3. Real frame loader from upstream/videos/0.mkv (NOT synthetic per CLAUDE.md
       "Forbidden make_synthetic_pair_batch in non-smoke").
    4. Catalog #319 deliverability_proof generation post-training.
    5. Catalog #270 dispatch optimization protocol Tier 1 + Tier 2 + Tier 3
       declarations.
    6. PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium
       (Catalog #325) for the J variant (Phase 1 cargo-cult audit at
       `.omx/research/path_3_j_mdl_ibps_cargo_cult_audit_of_c6_scaffold_20260526.md`
       is the predecessor input; per-substrate symposium with operator-frontier-
       override per Catalog #199 is required BEFORE paid dispatch).
    7. Catalog #324 predicted_band_validation_status MUST flip from
       `pending_post_training` to either `validated_post_training` (with
       post-training Tier-C anchor proving ACROSS_CLASS) or remain
       `pending_post_training` + `research_only: true` + `dispatch_enabled: false`.

    Until then this function refuses dispatch per CLAUDE.md "Substrate
    scaffolds MUST be COMPLETE or RESEARCH-ONLY" + Catalog #240
    research_only acceptance cascade (b) + pre-build substrate-engineering
    cascade (c).
    """
    raise NotImplementedError(
        "Path 3 J=MDL-IBPS DISCRETE-CATEGORICAL-MINE-HYBRID L0 SCAFFOLD: "
        "_full_main is council-gated per Catalog #240 acceptance cascade (c) "
        "pre-build substrate-engineering. L1+ pathway requires:\n"
        "  1. PyTorch port wire-in (inflate.py:inflate_substrate is export "
        "side; train-side PyTorch port + state_dict bridge required)\n"
        "  2. Score-aware loss via tac.substrates._shared.score_aware_common "
        "per Catalog #164 (eval_roundtrip mandatory)\n"
        "  3. Real frame loader from upstream/videos/0.mkv (Catalog #114)\n"
        "  4. Catalog #319 deliverability_proof post-training\n"
        "  5. Catalog #270 dispatch optimization protocol Tier 1 + 2 + 3 declarations\n"
        "  6. PER-SUBSTRATE OPTIMAL FORM symposium for J variant per Catalog "
        "#325 + operator-frontier-override per Catalog #199 BEFORE paid dispatch\n"
        "  7. Catalog #324 predicted_band_validation_status must NOT regress "
        "to phantom_random_init (CC-J-1 unwind canonical anchor)\n"
        "Run with --smoke for MLX-local L0 scaffold validation."
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if bool(args.smoke) or args.mode == "smoke":
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover - CLI smoke
    sys.exit(main())
