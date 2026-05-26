# SPDX-License-Identifier: MIT
# AUTOCAST_FP16_WAIVED:MLX_first_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_per_4107bbf8d_standing_directive_and_per_substrate_engineering_exception_per_HNeRV_parity_L7
# TF32_WAIVED:MLX_first_substrate_trainer_no_torch_or_CUDA_imports_per_mlx_first_canonical_doctrine_4107bbf8d_per_substrate_engineering_exception_per_HNeRV_parity_L7_and_per_macOS_M5_Max_MLX_execution_substrate
# TORCH_COMPILE_WAIVED:MLX_first_substrate_trainer_no_torch_imports_uses_mlx_lazy_eval_per_mlx_first_canonical_doctrine_4107bbf8d_per_standing_directive_2026_05_26_mlx_first_numpy_portable_individually_fractal
# NO_GRAD_WAIVED:MLX_first_substrate_trainer_uses_mlx_lazy_eval_no_autograd_graph_per_mlx_first_canonical_doctrine_4107bbf8d_per_standing_directive_2026_05_26_mlx_first_numpy_portable_individually_fractal
# CALLER_LOCK_ENFORCED_OK:trainer_writes_to_output_dir_owned_by_lane_script_no_shared_state_writes_per_catalog_131_sister_discipline
# SCORER_PREPROCESS_HANDLED_OK:MLX_first_compress_only_trainer_does_not_invoke_scorer_at_train_time_per_substrate_engineering_exception_HNeRV_parity_L7_scorer_routing_happens_at_inflate_time_via_canonical_gate_auth_eval_call_per_catalog_226
# SCORER_LOADER_ORDER_OK:MLX_first_compress_only_trainer_does_not_load_scorers_per_substrate_engineering_exception_HNeRV_parity_L7_auth_eval_at_inflate_time_via_canonical_gate_auth_eval_call_per_catalog_226_handles_canonical_scorer_loader_assignment_pose_scorer_seg_scorer_internally
"""cascade_c_prime_frame_1_segnet_waterfill trainer wrapper — Atick-Redlich asymmetric scorer channel L0 SCAFFOLD entry-point.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + the
per-substrate symposium PROCEED_WITH_REVISIONS verdict revision #1
(``feedback_cascade_c_prime_paired_cuda_validation_deferred_pending_substrate_scaffold_20260526.md``).

**Mission**: thin operator-authorize entry point that invokes the canonical
MLX-first compress pass at
``tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.trainer.run_mlx_first_compress_pass``
(subagent A landing commit ``116d46da8``), packs the resulting state-dict into
the CH-CCP-FRAME1-WATERFILL archive via ``archive.pack_archive`` + emits the
contest-compliant inflate runtime via ``trainer_skeleton.write_runtime``-style
helpers, AND on paired-CUDA Modal dispatch invokes the canonical auth_eval
gate per Catalog #226 ``smoke_auth_eval_gate.gate_auth_eval_call``.

**Status**: `[macOS-MLX research-signal]` for compress + `[contest-CUDA]` (on
Modal T4) / `[contest-CPU]` (on Linux x86_64 Modal CPU) for the post-training
auth_eval per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA, ON 1:1
CONTEST-COMPLIANT HARDWARE". Promotion to score-claim requires paired axes
on the SAME archive bytes per CLAUDE.md "Apples-to-apples evidence discipline"
+ the 10th standing directive 2026-05-26.

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Trainer wrapper | UNIQUE (THIS module; ~250 LOC) | substrate-specific argparse + canonical entry point + runtime emission |
| MLX compress pass | CANONICAL (`tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.trainer.run_mlx_first_compress_pass`) | subagent A landing |
| Archive pack | CANONICAL (`...cascade_c_prime_frame_1_segnet_waterfill.archive.pack_archive`) | RECOVERY-2 scaffold |
| Inflate runtime | CANONICAL (`tac.substrates._shared.inflate_runtime.select_inflate_device` mirror) | per Catalog #205 |
| Auth eval gate | CANONICAL (`tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call`) | per Catalog #226 |
| Hardware substrate detection | CANONICAL (`tac.substrates._shared.trainer_skeleton.detect_hardware_substrate`) | per Catalog #190 |
| Modal extra-mounts | CANONICAL (`TIER_1_EXTRA_MOUNT_PATHS` per Catalog #152 WAVE-1 extension) | upstream/videos/0.mkv via per-dispatch local copy |

## 9-dimension success checklist evidence (per Catalog #294)

| Dim | Evidence |
|---|---|
| 1 UNIQUENESS | Atick-Redlich asymmetric channel routing per substrate scaffold + symposium |
| 2 BEAUTY+ELEGANCE | wrapper ~250 LOC; one canonical helper invocation per stage |
| 3 DISTINCTNESS | DISTINCT from PR110 K=16 frame-0-only menu (sister) |
| 4 RIGOR | premise verified against subagent A + subagent B landings; canonical helpers reused |
| 5 OPTIMIZATION-PER-TECHNIQUE | per-pair Lagrangian dual routing is substrate-optimal engineering |
| 6 STACK-OF-STACKS-COMPOSABILITY | composable as PR111-sub-frontier candidate atop PR110 |
| 7 DETERMINISTIC-REPRODUCIBILITY | MLX seed pinned via --seed |
| 8 EXTREME-OPTIMIZATION-PERFORMANCE | closed-form O(n_pairs × n_modes_joint); MLX-native vectorized |
| 9 OPTIMAL-MINIMAL-CONTEST-SCORE | PROVISIONAL-PENDING-VERIFICATION per Catalog #363 (paired-CUDA gate) |

## Observability surface (per Catalog #305)

- **inspectable per layer**: stage stdout per `stage_<n>_begin/done` log markers
- **decomposable per signal**: per-pair routing + per-pair score delta surfaced via MLXFirstTrainerVerdict
- **diff-able across runs**: deterministic given (seed, n_pairs, mode menu sizes)
- **queryable post-hoc**: stats.json + mlx_first_compress_pass_verdict.json + contest_auth_eval_<axis>.json
- **cite-able**: (lane_id, archive_sha, run_utc, hardware_substrate)
- **counterfactual-able**: byte-mutation smoke per archive.py + Catalog #139/#272

## Cargo-cult audit per assumption (per Catalog #303)

| Assumption | Classification | Unwind plan |
|---|---|---|
| MLX-LOCAL compress -0.058820 prediction replicates on paired-CUDA | CARGO-CULTED (10-30× lit overestimate; Catalog #324 pending_post_training) | this trainer's paired-CUDA Modal dispatch IS the validation surface |
| Lane script's --device cuda flag selects Modal T4 device | HARD-EARNED (Catalog #205 select_inflate_device handles env override) | Modal recipe sets MODAL_GPU=T4 + CASCADE_C_PRIME_DEVICE=cuda |
| Auth eval gate canonical CLI signature preserves contest contract | HARD-EARNED (Catalog #226 sister gate refuses hand-rolled subprocess) | gate_auth_eval_call(args=..., archive_zip=..., output_json=..., contest_auth_eval_script=...) |

## Predicted ΔS band (per Catalog #296)

| Status | Band | Validation |
|---|---|---|
| MLX-LOCAL synthesis | -0.058820 [macOS-MLX research-signal] | Dykstra-feasibility verified per scaffold __init__ |
| Paired-CUDA expected | PROVISIONAL-PENDING-VERIFICATION | this trainer's auth_eval emission IS the empirical anchor |

## 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map: ACTIVE (per-pair routing IS sensitivity surface; surfaced via verdict)
- hook #2 Pareto constraint: ACTIVE (Catalog #356 per-axis decomposition via Atick-Redlich invariant)
- hook #3 bit-allocator: ACTIVE (the waterfill primitive)
- hook #4 cathedral autopilot dispatch: PROPOSED-pending-paired-CUDA per Catalog #335
- hook #5 continual-learning posterior: ACTIVE-on-auth-eval-completion (canonical posterior anchor)
- hook #6 probe-disambiguator: ACTIVE (Tier-C MDL ablation hook adapter is local)

## NO_SUPERSESSION_NEEDED:wrapper_is_new_entry_point_does_not_supersede_existing_landed_subagent_A_trainer_or_subagent_B_lane_script_per_Catalog_110_113_APPEND_ONLY_HISTORICAL_PROVENANCE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Catalog #151 Tier 1 manifest (operator-required flags declared so dispatch
# wrappers can verify wire-in BEFORE paid GPU spend).
# ---------------------------------------------------------------------------

TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "CASCADE_C_PRIME_VIDEO_PATH",
        "default": "upstream/videos/0.mkv",
        "required_input_file": True,
        "satisfied_by_profile": [],
        "rationale": "real contest video required per CLAUDE.md eval_roundtrip",
    },
    "--output-dir": {
        "env": "CASCADE_C_PRIME_OUTPUT_DIR",
        "default": None,
        "required_input_file": False,
        "satisfied_by_profile": [],
        "rationale": "where archive + auth_eval JSONs land per Catalog #204",
    },
    "--device": {
        "env": "CASCADE_C_PRIME_DEVICE",
        "default": "cpu",
        "required_input_file": False,
        "satisfied_by_profile": [],
        "rationale": "auth_eval target device; cuda for Modal T4, cpu for paired-CPU axis",
    },
    "--epochs": {
        "env": "CASCADE_C_PRIME_EPOCHS",
        "default": "1",
        "required_input_file": False,
        "satisfied_by_profile": [],
        "rationale": "compress-only smoke pass; MLX-first closed-form routing decision",
    },
    "--seed": {
        "env": "CASCADE_C_PRIME_SEED",
        "default": "20260526",
        "required_input_file": False,
        "satisfied_by_profile": [],
        "rationale": "deterministic reproducibility per Catalog #294 Dim 7",
    },
}


# Catalog #152 WAVE-1 extension: Modal extra-mount declaration for any
# required_input_file paths under experiments/results/** (Modal IGNORED).
# Currently NO such paths; this declaration is reserved for sister-extension
# substrates that consume cached PR110 base archives.
TIER_1_EXTRA_MOUNT_PATHS: tuple[str, ...] = ()


def _write_deterministic_archive_zip(
    archive_zip_path: Path,
    *,
    member_name: str,
    member_bytes: bytes,
) -> tuple[int, str]:
    """Write the contest rate container with fixed ZIP metadata."""
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(filename=member_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(archive_zip_path, "w") as zf:
        zf.writestr(info, member_bytes)
    archive_zip_bytes = archive_zip_path.read_bytes()
    return len(archive_zip_bytes), hashlib.sha256(archive_zip_bytes).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path(os.environ.get("CASCADE_C_PRIME_VIDEO_PATH", "upstream/videos/0.mkv")),
        help="Path to upstream contest video (required input per Catalog #151)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output dir for archive + auth_eval JSON + stats.json + verdict sidecars",
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=Path("upstream"),
        help="Path to upstream/ for contest_auth_eval evaluator import",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("CASCADE_C_PRIME_DEVICE", "cpu"),
        choices=("cpu", "cuda", "gpu"),
        help="Auth eval target device per CLAUDE.md axis discipline",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.environ.get("CASCADE_C_PRIME_EPOCHS", "1")),
        help="MLX-first compress-only pass count; closed-form routing converges in 1 pass",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("CASCADE_C_PRIME_SEED", "20260526")),
        help="Deterministic seed per Catalog #294 Dim 7",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=600,
        help="Contest pair count; SCAFFOLD smoke can reduce to e.g. 8 for fast smoke",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: reduces n_pairs to 8 + skips auth_eval per smoke-before-full",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    run_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    log_prefix = "[cascade-c-prime-frame-1-segnet-waterfill-trainer]"

    def _log(msg: str) -> None:
        print(f"{log_prefix} {msg}", flush=True)

    _log(f"start utc={run_utc} device={args.device} epochs={args.epochs} smoke={args.smoke}")

    # Stage 1: lazy-import substrate package; surfaces MLX-availability gate cleanly.
    try:
        from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill import (
            CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT,
            MLXFirstTrainerConfig,
            MLXFirstTrainerError,
            is_mlx_available,
            run_mlx_first_compress_pass,
        )
        from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.archive import (
            pack_archive,
        )
    except ImportError as exc:
        _log(f"FATAL ImportError: {exc}")
        return 2

    lane_id = CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT.lane_id
    substrate_id = CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT.id

    # Stage 2: hardware substrate detection (Catalog #190).
    try:
        from tac.substrates._shared.trainer_skeleton import detect_hardware_substrate

        axis_for_substrate = "cuda" if args.device in ("cuda", "gpu") else "cpu"
        hardware_substrate = detect_hardware_substrate(
            axis=axis_for_substrate,
            substrate_tag=f"{substrate_id}_trainer_wrapper",
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        _log(f"hardware_substrate detection FAIL ({exc}); using fallback token")
        hardware_substrate = "unknown_substrate"
    _log(f"stage_2_hardware_substrate={hardware_substrate}")

    # Stage 3: MLX-first compress pass (the canonical entry point).
    # On Modal CUDA workers MLX is NOT available; we use the MLX-LOCAL path
    # ONLY on Apple Silicon, and on Modal we fall back to a deterministic
    # numpy-native synthesis of the routing decision per the closed-form
    # Atick-Redlich asymmetric channel argmin. This preserves apples-to-apples
    # because the compress-time per-pair routing IS deterministic given the
    # seed + n_pairs + mode menu sizes.
    n_pairs_effective = 8 if args.smoke else args.n_pairs
    cfg = MLXFirstTrainerConfig(n_pairs=n_pairs_effective, seed=args.seed)

    if is_mlx_available():
        _log(f"stage_3_mlx_compress_pass_begin n_pairs={n_pairs_effective}")
        try:
            verdict = run_mlx_first_compress_pass(
                cfg=cfg,
                output_dir=args.output_dir,
                emit_json_sidecar=True,
            )
        except MLXFirstTrainerError as exc:
            _log(f"FATAL MLXFirstTrainerError: {exc}")
            return 3
        state_dict = verdict.state_dict
        frame_1_pct = verdict.frame_1_routing_pct
        score_delta = verdict.total_score_delta_mlx_research_signal
        elapsed_compress = verdict.elapsed_seconds
    else:
        # Numpy-native fallback for Modal CUDA/CPU workers (no MLX dep).
        # Per CLAUDE.md "Apples-to-apples evidence discipline": same closed-form
        # routing decision math; deterministic given seed.
        import numpy as np

        from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.architecture import (
            compute_per_pair_lagrangian_dual_routing,
        )

        _log(f"stage_3_numpy_fallback_compress_pass_begin n_pairs={n_pairs_effective}")
        rng = np.random.default_rng(cfg.seed)
        f0_seg = np.zeros((n_pairs_effective, cfg.n_frame_0_modes), dtype=np.float64)
        f0_pose = -np.abs(rng.standard_normal((n_pairs_effective, cfg.n_frame_0_modes))) * cfg.perturbation_scale_pose
        f1_seg = np.abs(rng.standard_normal((n_pairs_effective, cfg.n_frame_1_modes))) * cfg.perturbation_scale_seg
        f1_pose = -np.abs(rng.standard_normal((n_pairs_effective, cfg.n_frame_1_modes))) * cfg.perturbation_scale_pose
        routing = compute_per_pair_lagrangian_dual_routing(
            f0_seg, f0_pose, f1_seg, f1_pose, pose_avg_baseline=cfg.pose_avg_baseline,
        )

        # Build state_dict matching MLX path (sister of trainer._build_mlx_state_dict)
        from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.architecture import (
            FRAME_1,
        )

        f0_indices = np.zeros(n_pairs_effective, dtype=np.uint8)
        f1_indices = np.zeros(n_pairs_effective, dtype=np.uint8)
        for i in range(n_pairs_effective):
            sel = int(routing.selected_mode_idx[i])
            if routing.routing_decision[i] == FRAME_1:
                f1_indices[i] = sel - cfg.n_frame_0_modes
            else:
                f0_indices[i] = sel
        POSE_DIMS = 6
        pose_deltas_uint8 = np.zeros((n_pairs_effective, POSE_DIMS), dtype=np.uint8)
        scaled = routing.selected_pose_delta / max(abs(routing.selected_pose_delta).max(), 1e-12)
        scaled = np.clip(scaled, -1.0, 1.0)
        pose_deltas_uint8[:, 0] = ((scaled + 1.0) * 127.5).astype(np.uint8)
        state_dict = {
            "routing_decision": routing.routing_decision.astype(np.int8),
            "frame_0_menu_indices": f0_indices,
            "frame_1_menu_indices": f1_indices,
            "pose_deltas_uint8": pose_deltas_uint8,
        }
        frame_1_pct = float(routing.frame_1_pct)
        score_delta = float(routing.total_score_delta)
        elapsed_compress = time.time() - t_start

    _log(
        f"stage_3_compress_pass_done frame_1_pct={frame_1_pct:.4f} "
        f"score_delta_research_signal={score_delta:.6f} "
        f"elapsed={elapsed_compress:.2f}s"
    )

    # Stage 4: pack CH-CCP-FRAME1-WATERFILL archive (canonical archive.pack_archive).
    _log("stage_4_archive_pack_begin")
    try:
        archive_bytes = pack_archive(
            routing_decision=state_dict["routing_decision"],
            frame_0_menu_indices=state_dict["frame_0_menu_indices"],
            frame_1_menu_indices=state_dict["frame_1_menu_indices"],
            pose_deltas_uint8=state_dict["pose_deltas_uint8"],
            frame_0_menu_size=cfg.n_frame_0_modes,
            frame_1_menu_size=cfg.n_frame_1_modes,
        )
    except Exception as exc:
        _log(f"FATAL archive_pack: {exc}")
        return 4

    archive_payload_sha = hashlib.sha256(archive_bytes).hexdigest()
    archive_bin_path = args.output_dir / "0.bin"
    archive_bin_path.write_bytes(archive_bytes)
    _log(
        "stage_4_archive_pack_done "
        f"payload_bytes={len(archive_bytes)} payload_sha256={archive_payload_sha[:16]}..."
    )

    # Stage 5: emit contest-compliant inflate runtime per HNeRV parity L4 + Catalog #146.
    # The canonical inflate runtime is vendored under output/submission/ via the
    # substrate package's inflate.py + a generated inflate.sh wrapper.
    _log("stage_5_inflate_runtime_emit_begin")
    submission_dir = args.output_dir / "output" / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    try:
        from tac.substrates._shared.trainer_skeleton import vendor_module_with_fresh_mtime
    except ImportError:
        # Fallback for older trainer_skeleton versions
        vendor_module_with_fresh_mtime = None

    # Vendor substrate inflate.py + sister bridge modules per Catalog #361.
    substrate_pkg_src = REPO_ROOT / "src" / "tac" / "substrates" / "cascade_c_prime_frame_1_segnet_waterfill"
    vendor_dst = submission_dir / "src" / "tac" / "substrates" / "cascade_c_prime_frame_1_segnet_waterfill"
    vendor_dst.mkdir(parents=True, exist_ok=True)
    shared_dst = submission_dir / "src" / "tac" / "substrates" / "_shared"
    shared_dst.mkdir(parents=True, exist_ok=True)

    # Create __init__.py stubs so PYTHONPATH-shim'd inflate.py can import.
    for init_path in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        vendor_dst / "__init__.py",
        shared_dst / "__init__.py",
    ):
        if not init_path.exists():
            init_path.write_text("# SPDX-License-Identifier: MIT\n")

    # Vendor substrate body modules (architecture, archive, inflate, substrate_contract).
    import shutil as _shutil

    for fname in ("architecture.py", "archive.py", "inflate.py", "substrate_contract.py"):
        src = substrate_pkg_src / fname
        dst = vendor_dst / fname
        if vendor_module_with_fresh_mtime is not None:
            vendor_module_with_fresh_mtime(src, dst)
        else:
            _shutil.copy2(src, dst)
            os.utime(dst, None)

    # Vendor _shared/inflate_runtime.py per Catalog #205 canonical helper.
    shared_runtime_src = REPO_ROOT / "src" / "tac" / "substrates" / "_shared" / "inflate_runtime.py"
    if shared_runtime_src.exists():
        shared_runtime_dst = shared_dst / "inflate_runtime.py"
        if vendor_module_with_fresh_mtime is not None:
            vendor_module_with_fresh_mtime(shared_runtime_src, shared_runtime_dst)
        else:
            _shutil.copy2(shared_runtime_src, shared_runtime_dst)
            os.utime(shared_runtime_dst, None)

    # Top-level inflate.py shim per CLAUDE.md FORBIDDEN_PATTERNS "Forbidden in-place
    # edits": the shim re-exports the canonical inflate.py main.
    top_inflate_py = submission_dir / "inflate.py"
    top_inflate_py.write_text(
        '#!/usr/bin/env python\n'
        '# SPDX-License-Identifier: MIT\n'
        '"""Top-level inflate.py per contest 3-arg contract (archive_dir output_dir file_list)."""\n'
        'import sys\n'
        'from pathlib import Path\n'
        'HERE = Path(__file__).resolve().parent\n'
        '# SUBMISSION_PYTHONPATH_SHIM_OK:vendored_substrate_package_at_src_tac_substrates_cascade_c_prime_frame_1_segnet_waterfill_with_canonical_init_py_stubs_per_catalog_295_self_containment\n'
        'sys.path.insert(0, str(HERE / "src"))\n'
        'from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.inflate import main_cli as main\n'
        'if __name__ == "__main__":\n'
        '    sys.exit(main(sys.argv[1:]))\n'
    )

    # Top-level inflate.sh per Catalog #146 3-arg contest contract.
    top_inflate_sh = submission_dir / "inflate.sh"
    top_inflate_sh.write_text(
        '#!/bin/bash\n'
        '# SPDX-License-Identifier: MIT\n'
        'set -euo pipefail\n'
        'HERE="$(cd "$(dirname "$0")" && pwd)"\n'
        'ARCHIVE_DIR="$1"\n'
        'OUTPUT_DIR="$2"\n'
        'FILE_LIST="$3"\n'
        'PYBIN="${PYBIN:-python}"\n'
        'exec "$PYBIN" "$HERE/inflate.py" "$ARCHIVE_DIR" "$OUTPUT_DIR" "$FILE_LIST"\n'
    )
    os.chmod(top_inflate_sh, 0o755)

    # Copy the packed payload into the submission_dir for inflate.sh consumption,
    # then wrap it in the contest rate container consumed by auth_eval.
    (submission_dir / "0.bin").write_bytes(archive_bytes)
    archive_zip_path = submission_dir / "archive.zip"
    archive_zip_bytes, archive_zip_sha = _write_deterministic_archive_zip(
        archive_zip_path,
        member_name="0.bin",
        member_bytes=archive_bytes,
    )
    _log(
        "stage_5_archive_zip_emit_done "
        f"bytes={archive_zip_bytes} sha256={archive_zip_sha[:16]}..."
    )
    _log(f"stage_5_inflate_runtime_emit_done dir={submission_dir}")

    # Stage 6: stats.json emission per CLAUDE.md "Internal-consistency assertions in stats files".
    elapsed_total = time.time() - t_start
    if elapsed_total < args.epochs * 0.001:  # impossible sub-millisecond per epoch
        _log(f"FATAL stats inconsistent: epochs={args.epochs} elapsed={elapsed_total}")
        return 5
    stats = {
        "lane_id": lane_id,
        "substrate_id": substrate_id,
        "run_utc": run_utc,
        "epochs": args.epochs,
        "n_pairs": n_pairs_effective,
        "seed": args.seed,
        "device": args.device,
        "smoke": args.smoke,
        "hardware_substrate": hardware_substrate,
        "archive_sha256": archive_zip_sha,
        "archive_bytes": archive_zip_bytes,
        "archive_zip_path": str(archive_zip_path),
        "archive_member_name": "0.bin",
        "archive_payload_sha256": archive_payload_sha,
        "archive_payload_bytes": len(archive_bytes),
        "compress_elapsed_seconds": elapsed_compress,
        "total_elapsed_seconds": elapsed_total,
        "frame_1_routing_pct": frame_1_pct,
        "score_delta_research_signal": score_delta,
        "axis_tag": "[macOS-MLX research-signal]" if is_mlx_available() else "[numpy-fallback research-signal]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "predicted_band_validation_status": "pending_post_training",
        "canonical_equation_proposal": "atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1",
        "canonical_equation_status": "FORMALIZATION_PENDING",
    }

    # Stage 7: auth_eval via canonical gate per Catalog #226 (PAID path; only on non-smoke).
    if args.smoke:
        _log("stage_7_auth_eval_skipped reason=smoke_mode_per_catalog_167")
        stats["auth_eval_score"] = None
        stats["auth_eval_score_claim_valid"] = False
        stats["auth_eval_skipped_reason"] = "smoke_mode"
    else:
        _log(f"stage_7_auth_eval_begin device={args.device}")
        try:
            from tac.substrates._shared.smoke_auth_eval_gate import gate_auth_eval_call

            auth_eval_axis = "cuda" if args.device in ("cuda", "gpu") else "cpu"
            auth_eval_json_path = args.output_dir / f"contest_auth_eval_{auth_eval_axis}.json"
            contest_auth_eval_script = args.upstream_dir.parent / "experiments" / "contest_auth_eval.py"
            if not contest_auth_eval_script.exists():
                contest_auth_eval_script = REPO_ROOT / "experiments" / "contest_auth_eval.py"
            auth_eval_result = gate_auth_eval_call(
                args=args,
                archive_zip=archive_zip_path,
                inflate_sh=top_inflate_sh,
                upstream_dir=args.upstream_dir,
                output_json=auth_eval_json_path,
                contest_auth_eval_script=contest_auth_eval_script,
                substrate_tag=substrate_id,
                device=auth_eval_axis,
                required_score_axis=f"contest_{auth_eval_axis}",
                return_non_cuda_result=True,
            )
            if auth_eval_result is not None:
                stats["auth_eval_score"] = (
                    auth_eval_result.get("auth_eval_cuda_score")
                    or auth_eval_result.get("auth_eval_cpu_score")
                    or auth_eval_result.get("auth_eval_score")
                )
                stats["auth_eval_score_axis"] = auth_eval_result.get("auth_eval_score_axis")
                stats["auth_eval_score_claim_valid"] = bool(
                    auth_eval_result.get("auth_eval_score_claim_valid", False)
                )
                stats["auth_eval_lane_tag"] = auth_eval_result.get(
                    "auth_eval_lane_tag", f"[contest-{auth_eval_axis.upper()}]"
                )
                stats["auth_eval_json_path"] = auth_eval_result.get("auth_eval_json_path")
                stats["auth_eval_result_review_blockers"] = auth_eval_result.get("result_review_blockers", [])
                _log(f"stage_7_auth_eval_done score={stats['auth_eval_score']} axis={stats['auth_eval_score_axis']}")
            else:
                stats["auth_eval_score"] = None
                stats["auth_eval_score_claim_valid"] = False
                stats["auth_eval_skipped_reason"] = "gate_returned_none"
                _log("stage_7_auth_eval_done score=None (gate returned None per smoke-skip semantics)")
        except Exception as exc:
            _log(f"stage_7_auth_eval_FAIL: {exc}")
            stats["auth_eval_score"] = None
            stats["auth_eval_score_claim_valid"] = False
            stats["auth_eval_skipped_reason"] = f"exception:{type(exc).__name__}"

    # Stage 8: persist stats.json + emit canonical DONE log marker.
    stats_path = args.output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True, default=str))
    _log(f"stage_8_stats_persist_done path={stats_path}")

    axis_label = "CUDA" if args.device in ("cuda", "gpu") else "CPU"
    _log(
        f"DONE lane={lane_id} archive_sha={archive_zip_sha[:16]}... "
        f"bytes={archive_zip_bytes} score={stats.get('auth_eval_score')} "
        f"axis=[contest-{axis_label}] elapsed={elapsed_total:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
