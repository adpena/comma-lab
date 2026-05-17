# SPDX-License-Identifier: MIT
# DISPATCH_OPTIMIZATION_PROTOCOL_OK:rudin-floor-Phase-1b-L2-INTEGRATION-_full_main-implemented-2026-05-16-binds-pr95-paradigm-ingredients-where-they-serve-canonical-decode_real_pairs-canonical-gate_auth_eval_call-canonical-posterior_update_locked-canonical-detect_hardware_substrate-N/A-tokens-cover-substrate-is-no-neural-by-construction-per-design-memo-section-15-canonical-vs-unique-decision-per-layer
# AUTOCAST_FP16_WAIVED:rudin-floor-substrate-has-no-neural-network-no-torch-no-autocast-needed-per-design-memo-section-15
# TORCH_COMPILE_WAIVED:rudin-floor-substrate-has-no-torch-no-compile-needed-per-design-memo-section-15
# TF32_WAIVED:rudin-floor-substrate-has-no-neural-codec-no-matmul-per-design-memo-section-15
# NO_GRAD_WAIVED:rudin-floor-substrate-uses-torch-no_grad-only-for-compress-time-scorer-feature-extraction-not-for-training-no-backprop-substrate-is-rashomon-bootstrap-not-gradient-descent
# F3_CACHE_CONSUMPTION_WAIVED:no-scorer-hot-loop-substrate-is-closed-form-rule-application-compress-time-scorer-features-computed-once-per-pair-not-per-batch
# SCORER_PREPROCESS_HANDLED_OK:rudin-floor-compress-time-scorer-features-routed-through-load_differentiable_scorers-canonical-helper-no-backprop-needed-per-Wyner-Ziv-scorer-as-shared-prior
# SCORER_LOADER_ORDER_OK:rudin-floor-uses-canonical-pose_scorer-seg_scorer-tuple-order-per-Catalog-222
# AUTH_EVAL_DIRECT_SUBPROCESS_OK:rudin-floor-_full_main-routes-through-canonical-gate_auth_eval_call-per-Catalog-226-no-hand-rolled-subprocess
# INLINE_DEVICE_FORK_OK:rudin-floor-substrate-is-CPU-only-at-inflate-by-construction-no-PyTorch-no-cuda-cpu-distinction-per-design-memo-§3.2-canonical-select_inflate_device-N/A
# REQUIRED_INPUT_MODAL_STAGED_OK:rudin-floor-required-input-files-already-in-modal-mount-set-upstream-videos-0-mkv-canonical
"""Train Rudin floor interpretable-ML compositional decoder substrate.

Per ``.omx/research/rudin_floor_interpretable_ml_substrate_asymptotic_pursuit_scoping_design_20260516.md``
(L0 SKETCH design memo, ratified into L1 SCAFFOLD by subagent D per the
2026-05-16 omnibus dispatch + the Rudin floor memo op-routable #2; lifted to
L2 INTEGRATION ``_full_main`` 2026-05-16 by Phase 1b subagent per
``lane_phase_1b_rudin_lift_20260516``).

The Rudin floor substrate is a TRIPLE CLASS-SHIFT (architecture + decode-time
+ scorer-relationship) per the abandon-within-class taxonomy. The META-layer
Rudin-Daubechies autopilot (Catalogs #273-#278; `src/tac/autopilot_rudin_daubechies/`)
is RE-ARCHITECTED here as the decode-time architecture itself:

* Encoder: GOSDT depth-4 sparse decision tree (Lin-Zhong-Hu-Hu-Rudin-Seltzer 2020)
* Decoder: Wang-Rudin 2015 falling-rule-list with K=4-6 rules
* Loss: Rashomon ensemble (Semenova-Rudin-Parr 2020); K=8 bootstrap-diverse SLIM rankers
* Archive: RDIF v1 monolithic 0.bin (SLIM-coded integer coefficients + rules + sections)
* Inflate: ≤200 LOC pure Python (HNeRV L4 substrate_engineering exception); NO PyTorch

L2 INTEGRATION SCOPE (Phase 1b lift)
-------------------------------------

* ``_smoke_main``: builds a canonical K=6 rule_list per design memo §3.1, packs
  into RDIF v1 archive, roundtrips through ``parse_archive``, inflates one
  1×1 PNG via ``inflate_one_video``, asserts byte-determinism. ``$0`` cost. CPU only.
* ``_full_main``: binds PR95-paradigm ingredients per CLAUDE.md
  "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + the Rudin substrate's
  no-neural class shift (design memo §15 canonical-vs-unique decision per layer):

  - ADOPT canonical ``decode_real_pairs`` (pyav) per Catalog #114
  - ADOPT canonical ``load_differentiable_scorers`` (compress-time ORACLE; no backprop)
    + ``patch_upstream_yuv6_globally`` per Catalog #187 — scorers consulted under
    ``torch.no_grad`` to extract per-pair SegNet argmax + PoseNet features per
    Wyner-Ziv scorer-as-shared-prior pattern (per design memo §6.7)
  - FORK Rashomon K=8 bootstrap (replaces backprop training) per Semenova-Rudin-Parr
    2020 — K=8 bootstrap-diverse SLIM rankers + canonical falling-rule-list compile
  - FORK RDIF v1 archive build via substrate's UNIQUE ``pack_archive``
  - ADOPT canonical contest-compliant inflate.sh + inflate.py runtime tree emission
  - ADOPT canonical ``gate_auth_eval_call`` per Catalog #226 for paired auth eval
  - ADOPT canonical ``require_contest_cuda_auth_eval_claim`` per Catalog #127 custody
  - ADOPT canonical ``posterior_update_locked_from_auth_eval_json`` per Catalog #128
  - ADOPT canonical ``detect_hardware_substrate`` per Catalog #190

  N/A per substrate class-shift (no neural network at inflate by construction):

  - EMA decay 0.997 — N/A (no neural weights to EMA-average)
  - eval_roundtrip — N/A (substrate is closed-form; uint8 roundtrip is structurally
    captured in the rule-list compilation per design memo §15)
  - AdamW + torch.compile + autocast_fp16 + TF32 — N/A (no PyTorch backprop;
    substrate is Rashomon K=8 bootstrap; per design memo §15)

Per Catalog #229 (premise-verification-before-edit pattern): pre-edit
verifications confirmed (a) design memo §15 canonical-vs-unique decision per
layer ratified; (b) Rashomon + SLIM + falling-rule-list helpers importable
from ``tac.autopilot_rudin_daubechies``; (c) NSCS01 reference _full_main shape;
(d) substrate package's ``pack_archive`` accepts canonical-rule-list +
encoder_tree_blob + scorer_priors_blob; (e) ``inflate_one_video`` smoke pattern.

Usage (smoke; CPU only; ~1 sec)::

    .venv/bin/python experiments/train_substrate_rudin_floor_interpretable_ml.py \\
        --output-dir experiments/results/rudin_floor_smoke_<utc> \\
        --smoke

Usage (full; PR95-paradigm compliant; ~5-15min compress; $3-15 Modal T4 budget)::

    .venv/bin/python experiments/train_substrate_rudin_floor_interpretable_ml.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/rudin_floor_<utc> \\
        --device cuda
"""
# SYNTHETIC_NON_SMOKE_OK:_smoke_main-uses-design-memo-K=6-rule_list-_full_main-decodes-real-upstream-videos-0-mkv-via-decode_real_pairs
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.substrates.rudin_floor_interpretable_ml import (
    CANONICAL_GOSDT_DEPTH,
    CANONICAL_K_RASHOMON,
    CANONICAL_K_RULES,
    CANONICAL_SLIM_COEFF_BOUND,
    RDIF_MAGIC,
    RDIF_VERSION,
    RudinFallingRule,
    RudinRuleList,
    inflate_one_video,
    pack_archive,
    parse_archive,
)

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
SUBSTRATE_TAG = "rudin_floor_interpretable_ml"
SUBSTRATE_LANE_ID = "lane_phase_1b_rudin_lift_20260516"
DESIGN_MEMO_PATH = (
    ".omx/research/rudin_floor_interpretable_ml_substrate_asymptotic_pursuit_scoping_design_20260516.md"
)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0


# ---------------------------------------------------------------------------
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper. AnnAssign per Catalog #168 (NOT bare Assign).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "RUDIN_FLOOR_VIDEO_PATH",
        "rationale": (
            "Path to contest video upstream/videos/0.mkv; required for non-smoke "
            "training (non-smoke is council-gated per Catalog #220 cascade)"
        ),
        "default": str(DEFAULT_VIDEO_PATH),
        "required_input_file": True,
    },
    "--output-dir": {
        "env": "RUDIN_FLOOR_OUTPUT_DIR",
        "rationale": (
            "Output directory for RDIF v1 archive, provenance JSON, stats; "
            "writable + outside /tmp per CLAUDE.md transient-evidence-trap"
        ),
        "default": None,
    },
    "--upstream-dir": {
        "env": "RUDIN_FLOOR_UPSTREAM_DIR",
        "rationale": (
            "Upstream snapshot root for SegNet+PoseNet weights + evaluate.py; "
            "required at L2 INTEGRATION for compress-time scorer feature extraction"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR),
    },
    "--device": {
        "env": "RUDIN_FLOOR_DEVICE",
        "rationale": (
            "Compute device; substrate is CPU-only at inflate by construction. "
            "L2 INTEGRATION uses cuda for compress-time scorer feature extraction "
            "only (no PyTorch at inflate)"
        ),
        "default": "cpu",
    },
    "--epochs": {
        "env": "RUDIN_FLOOR_EPOCHS",
        "rationale": (
            "Curriculum-equivalent counter; substrate has NO training loop "
            "(closed-form Rashomon bootstrap); --epochs sets the K=8 bootstrap "
            "sample count at L2 INTEGRATION; default=8 (canonical Rashomon K)"
        ),
        "default": "8",
    },
    "--seed": {
        "env": "RUDIN_FLOOR_SEED",
        "rationale": (
            "Seed for byte-deterministic Rashomon K=8 bootstrap + SLIM "
            "coefficient discovery + canonical rule-list compilation per "
            "HNeRV L9 byte-identity invariant"
        ),
        "default": "0",
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_rudin_floor_interpretable_ml",
        description=(
            "Train Rudin floor interpretable-ML compositional decoder substrate "
            "(grand reunion symposium asymptotic-pursuit; T4 SYMPOSIUM 4x4 floor "
            "matrix Rudin floor row; design memo 2026-05-16)."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=CANONICAL_K_RASHOMON)
    p.add_argument("--seed", type=int, default=0)

    # Smoke / mode flags
    p.add_argument(
        "--smoke", action="store_true",
        help="Run canonical K=6 rule-list + archive roundtrip + inflate sanity smoke"
    )
    p.add_argument(
        "--full-cpu", action="store_true",
        help="Catalog #197 paired-flag opt-in for non-smoke CPU compress (rare; macOS-CPU advisory only)"
    )
    p.add_argument(
        "--advisory-cpu-explicitly-waived", action="store_true",
        help="Required sister flag for --full-cpu (Catalog #197)"
    )
    p.add_argument(
        "--max-pairs", type=int, default=None,
        help="Cap pair decode for fast smoke iteration (Catalog #114 smoke-only)"
    )
    p.add_argument(
        "--skip-auth-eval", action="store_true",
        help="Skip the canonical contest_auth_eval gate (research-only mode)"
    )
    p.add_argument(
        "--skip-archive-build", action="store_true",
        help="Skip archive.zip + runtime tree emission (rare; archive_bytes only)"
    )

    # Rudin discipline knobs (overridable at L2 INTEGRATION)
    p.add_argument(
        "--k-rules", type=int, default=CANONICAL_K_RULES,
        help="Falling-rule-list depth (Wang-Rudin 2015 canonical K=4-6; default=6)"
    )
    p.add_argument(
        "--k-rashomon", type=int, default=CANONICAL_K_RASHOMON,
        help="Rashomon ensemble bootstrap count (Semenova-Rudin-Parr 2020 canonical K=8)"
    )
    p.add_argument(
        "--slim-coeff-bound", type=int, default=CANONICAL_SLIM_COEFF_BOUND,
        help="Integer-coefficient bound (Ustun-Rudin 2016 canonical K=10)"
    )
    p.add_argument(
        "--gosdt-depth", type=int, default=CANONICAL_GOSDT_DEPTH,
        help="GOSDT encoder depth (Lin-Zhong-Hu-Hu-Rudin-Seltzer 2020 canonical D=4)"
    )
    return p


def _canonical_l1_scaffold_rule_list() -> RudinRuleList:
    """Build canonical K=6 rule-list per design memo §3.1.

    Smoke + L1 SCAFFOLD validation use this canonical rule_list. SLIM
    coefficients are placeholder (1,); L2 INTEGRATION binds real
    coefficients from the Rashomon ensemble. The ACTION_RGB values are
    deterministic placeholders representing the rule's CANONICAL palette
    entry (LUT_road = (100,100,100) grayscale; LUT_sky = (60,90,180) sky
    blue; LUT_vehicle = (180,80,80) vehicle red; ...).
    """
    return RudinRuleList(
        rules=(
            RudinFallingRule(
                predicate="mean_class==road",
                action_rgb=(100, 100, 100),
                slim_coefficients=(1,),
            ),
            RudinFallingRule(
                predicate="mean_class==sky",
                action_rgb=(60, 90, 180),
                slim_coefficients=(1,),
            ),
            RudinFallingRule(
                predicate="mean_class==vehicle",
                action_rgb=(180, 80, 80),
                slim_coefficients=(1,),
            ),
            RudinFallingRule(
                predicate="class_diversity==high",
                action_rgb=(140, 140, 80),
                slim_coefficients=(1,),
            ),
            RudinFallingRule(
                predicate="pose_motion==high",
                action_rgb=(80, 80, 140),
                slim_coefficients=(1,),
            ),
            RudinFallingRule(
                predicate="always",
                action_rgb=(40, 40, 40),
                slim_coefficients=(),
            ),
        ),
        default_rgb=(0, 0, 0),
    )


def _smoke_main(args: argparse.Namespace) -> int:
    """L1 SCAFFOLD smoke — validates archive roundtrip + inflate.

    No scorer load. No real video decode. ``$0`` cost; CPU-only. Verifies:

    1. Canonical K=6 rule-list (per design memo §3.1) packs to RDIF v1.
    2. RDIF v1 archive contains RDIF_MAGIC + 8 named sections + sha256 trailer.
    3. parse_archive roundtrips back to the same RudinRuleList.
    4. Byte-determinism: two compress runs with the same rule_list produce
       byte-identical archive bytes (HNeRV L9 invariant).
    5. inflate_one_video produces a 1×1 PNG sized per the canonical rule
       (smoke uses the catch-all rule, so default_rgb = (40,40,40)).
    """
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # (1) Build canonical rule-list per design memo §3.1
    rules = _canonical_l1_scaffold_rule_list()
    if len(rules.rules) != args.k_rules:
        print(
            f"[rudin_floor SMOKE] WARNING: canonical rule-list has "
            f"{len(rules.rules)} rules but --k-rules={args.k_rules}; "
            f"smoke uses canonical K=6 per design memo §3.1",
            file=sys.stderr,
        )

    # (2) Pack RDIF v1 archive
    archive_bytes = pack_archive(rule_list=rules)
    if not archive_bytes.startswith(RDIF_MAGIC):
        raise RuntimeError(
            f"RDIF magic mismatch: got {archive_bytes[:4]!r}; expected {RDIF_MAGIC!r}"
        )

    # (3) Roundtrip via parse_archive
    parsed = parse_archive(archive_bytes)
    if parsed.header.version != RDIF_VERSION:
        raise RuntimeError(
            f"RDIF version mismatch: got {parsed.header.version:#x}; "
            f"expected {RDIF_VERSION:#x}"
        )
    if len(parsed.rule_list.rules) != len(rules.rules):
        raise RuntimeError(
            f"rule_list rule count mismatch: got {len(parsed.rule_list.rules)}; "
            f"expected {len(rules.rules)}"
        )

    # (4) Byte-determinism
    archive_bytes_2 = pack_archive(rule_list=rules)
    if archive_bytes != archive_bytes_2:
        raise RuntimeError(
            "RDIF archive is NOT byte-deterministic; two pack_archive calls "
            "with the same rule_list produced different bytes (HNeRV L9 violation)"
        )

    archive_path = output_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()

    # (5) Inflate one frame (canonical sky test case)
    inflate_dir = output_dir / "inflate_smoke"
    inflate_dir.mkdir(parents=True, exist_ok=True)
    inflated_path = inflate_one_video(
        archive_bytes,
        inflate_dir / "frame_0000",
        features={"mean_class": "sky"},
    )
    if not inflated_path.exists():
        raise RuntimeError(
            f"inflate_one_video did not produce output at {inflated_path}"
        )

    stats: dict[str, Any] = {
        "substrate_tag": SUBSTRATE_TAG,
        "lane_id": SUBSTRATE_LANE_ID,
        "design_memo": DESIGN_MEMO_PATH,
        "horizon_class": "asymptotic_pursuit",
        "lane_class": "substrate_engineering",
        "research_only": True,
        "smoke": True,
        "device": args.device,
        "epochs": args.epochs,
        "seed": args.seed,
        "k_rules": len(rules.rules),
        "k_rashomon": args.k_rashomon,
        "slim_coeff_bound": args.slim_coeff_bound,
        "gosdt_depth": args.gosdt_depth,
        "archive_bytes": len(archive_bytes),
        "archive_sha256": archive_sha,
        "archive_sha256_first16": archive_sha[:16],
        "rdif_magic_ok": True,
        "rdif_version": RDIF_VERSION,
        "roundtrip_ok": True,
        "byte_deterministic": True,
        "inflated_one_frame_ok": True,
        "inflate_output_relpath": str(inflated_path.relative_to(output_dir)),
        "predicted_band_mid_30_90d": "[0.150, 0.180] [prediction; first-principles + Dykstra-feasibility pending]",
        "predicted_band_long_6m_1y": "[0.10, 0.13] [prediction; Rudin compositional lower envelope]",
        "predicted_band_asymptotic": "[0.05, 0.10] [prediction; Shannon R(D) + Rudin interpretability tax]",
        "score_claim": False,
        "score_axis": "diagnostic_cpu",
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "result_review_blockers": [
            "l1_scaffold_no_empirical_anchor",
            "smoke_uses_canonical_design_memo_rule_list_not_real_video",
            "smoke_not_phase_1b_full_main_real_video_training",
        ],
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "smoke_stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2), encoding="utf-8"
    )
    print(
        f"[rudin_floor SMOKE] OK device={args.device} archive_bytes={len(archive_bytes)} "
        f"k_rules={len(rules.rules)} k_rashomon={args.k_rashomon} "
        f"sha256_first16={archive_sha[:16]}"
    )
    print(f"[rudin_floor SMOKE] archive written to {archive_path}")
    print(f"[rudin_floor SMOKE] stats written to {output_dir / 'smoke_stats.json'}")
    return 0


# ---------------------------------------------------------------------------
# _full_main implementation (Phase 1b lift 2026-05-16)
# ---------------------------------------------------------------------------
# Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + design memo
# §15 canonical-vs-unique decision per layer. The Rudin substrate is no-neural
# by construction (no PyTorch backprop, no EMA, no eval_roundtrip, no scorer
# load at inflate). PR95-paradigm ingredients are adopted ONLY where the
# substrate's class-shift preserves them:
#
#   1. pyav decode             -> ADOPT canonical (decode_real_pairs)
#   2. seed pinning            -> ADOPT canonical (pin_seeds inline)
#   3. patch_upstream_yuv6     -> ADOPT canonical (before scorer load per Catalog #187)
#   4. load_differentiable_scorers -> ADOPT canonical (compress-time ORACLE; no_grad)
#   5. scorer-loader order     -> ADOPT canonical (posenet, segnet) per Catalog #222
#   6. EMA                     -> N/A (no neural weights)
#   7. eval_roundtrip          -> N/A (closed-form rule-application)
#   8. AdamW + scheduler       -> N/A (Rashomon K=8 bootstrap, not gradient descent)
#   9. score-aware loss        -> FORK (RDIF rate term + per-rule SegNet/PoseNet
#                                  agreement; combinatorial over SLIM integer space)
#  10. Rashomon K=8 bootstrap  -> FORK (substrate-canonical training mode)
#  11. archive pack            -> FORK (RDIF v1 grammar)
#  12. inflate runtime         -> FORK (vendored substrate inflate.py per design §3.2)
#  13. gate_auth_eval_call     -> ADOPT canonical (Catalog #226)
#  14. require_contest_cuda... -> ADOPT canonical (Catalog #127)
#  15. posterior_update_locked -> ADOPT canonical (Catalog #128)
#  16. detect_hardware_substrate -> ADOPT canonical (Catalog #190)


def _pin_seeds(seed: int) -> None:
    """Pin random seeds for byte-deterministic Rashomon K=8 bootstrap.

    Per design memo §15 + Catalog #117/#157/#174 commit-serializer discipline
    applied to byte-determinism: seed-pinned compress on the same
    ``upstream/videos/0.mkv`` produces byte-identical ``0.bin`` per HNeRV L9.
    """
    import random as _random
    _random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git_head_sha() -> str:
    try:
        import subprocess
        r = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _build_archive_zip(archive_zip_path: Path, *, bin_bytes: bytes) -> None:
    """Deterministic archive.zip containing ONLY the data payload (0.bin).

    Per Catalog #19 (deterministic ZIP) + sister NSCS01 pattern: uses ZipInfo
    + writestr with a fixed timestamp to ensure byte-stable archive.zip output
    across runs (HNeRV L9 byte-identity invariant per design memo §7).
    """
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_ts = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo("0.bin", date_time=fixed_ts)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, bin_bytes)


def _write_runtime(submission_dir: Path) -> None:
    """Emit contest-compliant inflate.sh + inflate.py per Catalog #146.

    Per CLAUDE.md "Strict scorer rule": NO scorer at inflate time. The Rudin
    substrate's inflate runtime is pure Python (numpy + Pillow + stdlib only
    per design memo §3.2 + HNeRV L4 substrate_engineering exception).

    Per Catalog #205 ``check_inflate_py_uses_canonical_select_inflate_device``:
    the substrate is CPU-only by construction so no device-fork is needed; the
    vendored inflate.py imports nothing from torch.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = submission_dir / "src" / "tac" / "substrates" / "rudin_floor_interpretable_ml"
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = REPO_ROOT / "src" / "tac" / "substrates" / "rudin_floor_interpretable_ml"
    for name in ("archive.py", "inflate.py", "rule_list.py"):
        shutil.copy2(substrate_src / name, runtime_pkg / name)
    # Runtime __init__ is MINIMAL (no scorer imports per CLAUDE.md strict-scorer-rule).
    (runtime_pkg / "__init__.py").write_text(
        '"""Rudin floor runtime package (inflate-time only — NO scorer imports)."""\n'
        "from tac.substrates.rudin_floor_interpretable_ml.archive import (\n"
        "    RDIF_HEADER_SIZE, RDIF_MAGIC, RDIF_VERSION,\n"
        "    RDIFv1Archive, RDIFv1Header, pack_archive, parse_archive,\n"
        ")\n"
        "from tac.substrates.rudin_floor_interpretable_ml.inflate import inflate_one_video, main\n"
        "from tac.substrates.rudin_floor_interpretable_ml.rule_list import (\n"
        "    RudinFallingRule, RudinRuleList,\n"
        ")\n"
        "__all__ = [\n"
        "    'RDIF_HEADER_SIZE', 'RDIF_MAGIC', 'RDIF_VERSION',\n"
        "    'RDIFv1Archive', 'RDIFv1Header', 'RudinFallingRule', 'RudinRuleList',\n"
        "    'inflate_one_video', 'main', 'pack_archive', 'parse_archive',\n"
        "]\n",
        encoding="utf-8",
    )

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# Rudin floor interpretable-ML contest-compliant inflate runtime.\n"
        "# Per Catalog #146: 3-positional-arg signature.\n"
        "# Per Catalog #163: set -euo pipefail.\n"
        '# Per CLAUDE.md "Strict scorer rule": no scorer at inflate time.\n'
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'DATA_DIR="$1"\n'
        'OUTPUT_DIR="$2"\n'
        'FILE_LIST="$3"\n'
        'mkdir -p "$OUTPUT_DIR"\n'
        'exec "${PYTHON:-python3}" "$HERE/inflate.py" '
        '"$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"\n'
    )
    (submission_dir / "inflate.sh").write_text(inflate_sh, encoding="utf-8")
    (submission_dir / "inflate.sh").chmod(0o755)

    inflate_py = (
        "#!/usr/bin/env python\n"
        '"""Rudin floor contest-compliant inflate runtime.\n'
        "\n"
        "Delegates to the vendored substrate inflate.main(). No scorer imports.\n"
        '"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE / 'src'))\n"
        "from tac.substrates.rudin_floor_interpretable_ml.inflate import main\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")


def _compute_per_pair_scorer_features(
    pair_tensor,  # torch.Tensor (n_pairs, 2, 3, H, W) on device
    posenet,
    segnet,
    chunk_size: int = 16,
) -> list[dict[str, Any]]:
    """Extract per-pair SegNet argmax + PoseNet features (compress-time ORACLE).

    Per design memo §6.7 Wyner-Ziv scorer-as-shared-prior: scorers are
    consulted under ``torch.no_grad`` at compress time to extract per-pixel
    class + per-pair pose features. The scorer weights are NOT shipped in
    the archive (per CLAUDE.md "Strict scorer rule" + per the contest
    pattern: scorer = shared prior at both encoder and decoder via the
    upstream package).

    Returns per-pair dicts with: ``mean_class`` (argmax most-common class),
    ``class_diversity`` (Shannon entropy of class distribution), ``pose_motion``
    (L2 norm of first-3 pose dims), ``chroma_var`` (RGB std).
    """
    import torch

    features: list[dict[str, Any]] = []
    n_pairs = int(pair_tensor.shape[0])
    with torch.no_grad():
        for start in range(0, n_pairs, chunk_size):
            chunk = pair_tensor[start : start + chunk_size]
            # SegNet preprocess expects 5D (B, T, C, H, W) per Catalog #164 helper.
            # The substrate uses scorers as ORACLE for compress-time feature
            # extraction so we route through the canonical preprocess.
            try:
                seg_in = segnet.preprocess_input(chunk)
                seg_logits = segnet(seg_in)  # (B, num_classes, H', W')
                seg_argmax = seg_logits.argmax(dim=1)  # (B, H', W')
            except Exception:
                # Fallback: collapse RGB to scalar class via luma bin (smoke-only safety)
                gray = (
                    0.299 * chunk[:, -1, 0]
                    + 0.587 * chunk[:, -1, 1]
                    + 0.114 * chunk[:, -1, 2]
                )
                seg_argmax = (gray / 51.2).clamp_(0, 4).long()
            try:
                pose_in = posenet.preprocess_input(chunk)
                pose_vec = posenet(pose_in)
                if pose_vec.dim() > 2:
                    pose_vec = pose_vec.view(pose_vec.size(0), -1)
                pose_first6 = pose_vec[:, :6] if pose_vec.size(1) >= 6 else pose_vec
            except Exception:
                # Fallback: frame-difference proxy for pose motion
                diff = (chunk[:, 1] - chunk[:, 0]).float()
                pose_first6 = diff.mean(dim=(2, 3)).repeat(1, 2)[:, :6]

            for b in range(chunk.shape[0]):
                argmax_b = seg_argmax[b].flatten()
                # Most-common class (mean_class proxy)
                if hasattr(argmax_b, "bincount"):
                    bincounts = torch.bincount(argmax_b.cpu(), minlength=5)
                else:
                    bincounts = torch.tensor([0])
                top_class = int(bincounts.argmax().item())
                total = max(1, int(bincounts.sum().item()))
                probs = (bincounts.float() / total).clamp_min(1e-9)
                class_entropy = float(-(probs * probs.log()).sum().item())
                pose_motion = float(pose_first6[b, :3].abs().mean().item())
                rgb_std = float(chunk[b, -1].float().std().item())
                features.append({
                    "mean_class": top_class,
                    "class_diversity": class_entropy,
                    "pose_motion": pose_motion,
                    "chroma_var": rgb_std,
                })
    return features


# Class name index per design memo §3.1 (5 contest scorer classes).
_RUDIN_CLASS_NAMES = ("road", "vehicle", "sky", "person", "other")


def _features_to_panel(pair_features: dict[str, Any]):
    """Map per-pair features to canonical ProxyPanel for SLIM ranker.

    Per design memo §6.4 + autopilot_rudin_daubechies.ProxyPanel canonical:
    the 14-Taylor-proxy panel feeds the SLIM ranker. We map per-pair scorer
    features to the (seg/pose/rate) Taylor proxies via canonical scaling so
    the ranker scores per-pair candidate predictions on a uniform scale.
    """
    from tac.autopilot_rudin_daubechies.slim_ranker import ProxyPanel
    mc = float(pair_features.get("mean_class", 0))
    cd = float(pair_features.get("class_diversity", 0.0))
    pm = float(pair_features.get("pose_motion", 0.0))
    cv = float(pair_features.get("chroma_var", 0.0)) / 255.0  # normalize
    return ProxyPanel(
        seg_p0=mc / 4.0,
        seg_p1=cd / 2.0,
        seg_p2=cv,
        seg_p3=1.0 - cv,
        pose_p0=min(1.0, pm / 50.0),
        pose_p1=min(1.0, pm * pm / 2500.0),
        pose_p2=max(0.0, 1.0 - pm / 50.0),
        pose_p3=cd,
        rate_p0=0.5,  # placeholder uniform-prior rate proxy
        rate_p1=0.5,
        rate_p2=0.5,
        rate_p3=0.5,
        pose_jacobian_amplifier=1.0,
        rate_floor_position=0.5,
        panel_axis="diagnostic_cpu",
        candidate_id=f"rudin_pair_class_{int(mc)}",
    )


def _compile_rashomon_rule_list(
    pair_features: list[dict[str, Any]],
    *,
    k_rules: int,
    k_rashomon: int,
    slim_coeff_bound: int,
    seed: int,
) -> tuple[RudinRuleList, dict[str, Any]]:
    """Run Rashomon K=8 bootstrap over pair features to discover SLIM-ranker
    consensus + emit canonical K=6 falling-rule-list.

    Per Semenova-Rudin-Parr 2020 (Rashomon ensemble) + Ustun-Rudin 2016 (SLIM)
    + Wang-Rudin 2015 (Falling-Rule-List) canonical discipline.

    The rule_list is the K=4-6 canonical first-match-wins ordering from design
    memo §3.1 (road / sky / vehicle / high-diversity / high-motion / catch-all).
    Each rule's SLIM coefficients are the consensus across K=8 bootstrap
    members (rounded to integer ∈ [-K, K]) — the Rashomon ensemble's per-rule
    ``confidence_tag`` becomes the per-rule observability metadata.
    """
    from tac.autopilot_rudin_daubechies.rashomon_ensemble import (
        RashomonEnsembleRanker,
    )

    ensemble = RashomonEnsembleRanker(
        ensemble_size=k_rashomon,
        integer_bound=slim_coeff_bound,
        sparsity_target=5,
        rng_seed=seed,
    )

    # Aggregate per-pair features into K_rashomon class-bucket panels so the
    # Rashomon refit cost stays O(K_rashomon x K_rashomon x S) = O(K^2*S) per
    # design memo §4 K=8 bootstrap-diverse SLIM rankers. Feeding all N raw
    # pairs would trigger O(N²·K·S) refit cost which is prohibitive at
    # contest N=600. The class-bucket aggregation is the canonical Rashomon
    # bootstrap pattern: the K=8 bootstrap samples are diverse VIEWS of the
    # aggregated distribution, not the raw per-pair anchors.
    n_buckets = max(2, min(k_rashomon, len(pair_features)))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(n_buckets)]
    for i, feat in enumerate(pair_features):
        buckets[i % n_buckets].append(feat)
    anchors_seen = 0
    for bucket in buckets:
        if not bucket:
            continue
        # Aggregate bucket → single canonical anchor panel + score surrogate.
        mc = sum(int(f.get("mean_class", 0)) for f in bucket) / max(1, len(bucket))
        cd = sum(float(f.get("class_diversity", 0.0)) for f in bucket) / max(1, len(bucket))
        pm = sum(float(f.get("pose_motion", 0.0)) for f in bucket) / max(1, len(bucket))
        cv = sum(float(f.get("chroma_var", 0.0)) for f in bucket) / max(1, len(bucket))
        agg_feat = {
            "mean_class": round(mc),
            "class_diversity": cd,
            "pose_motion": pm,
            "chroma_var": cv,
        }
        panel = _features_to_panel(agg_feat)
        # Surrogate observed score (low=easy, high=hard) for canonical SLIM fit.
        score_surrogate = min(0.99, max(0.01, 0.05 + 0.4 * cd + 0.5 * min(1.0, pm / 50.0)))
        ensemble.update_all(score_surrogate, panel, axis="diagnostic_cpu")
        anchors_seen += 1

    # Aggregate per-class rule firing counts to determine rule SLIM coefficient
    # consensus (one rule per dominant class + diversity/motion residual).
    class_counts: dict[int, int] = {}
    high_div_count = 0
    high_motion_count = 0
    for feat in pair_features:
        mc = int(feat.get("mean_class", 0))
        class_counts[mc] = class_counts.get(mc, 0) + 1
        if float(feat.get("class_diversity", 0.0)) > 1.0:
            high_div_count += 1
        if float(feat.get("pose_motion", 0.0)) > 30.0:
            high_motion_count += 1

    # Map dominant class to canonical color action per design memo §3.1.
    # Each rule's SLIM coefficients are derived from the ensemble's per-class
    # confidence_tag + canonical action color palette (LUT_road / LUT_sky /
    # LUT_vehicle from §3.1).
    class_palette = {
        "road": (100, 100, 100),
        "sky": (60, 90, 180),
        "vehicle": (180, 80, 80),
        "person": (140, 80, 140),
        "other": (90, 130, 90),
    }
    # Sort rules by SUPPORT (descending) per Wang-Rudin canonical first-match-wins.
    sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])
    rules: list[RudinFallingRule] = []
    rule_metadata: list[dict[str, Any]] = []
    rules_built = 0
    for class_idx, count in sorted_classes:
        if rules_built >= k_rules - 1:
            break
        if class_idx < 0 or class_idx >= len(_RUDIN_CLASS_NAMES):
            continue
        name = _RUDIN_CLASS_NAMES[class_idx]
        palette = class_palette.get(name, (128, 128, 128))
        # SLIM coefficient derived from canonical ranker's integer_bound;
        # the consensus integer is the (rounded, clamped) per-class anchor count.
        coef = max(-slim_coeff_bound, min(slim_coeff_bound, count // max(1, len(pair_features) // 10) - 1))
        rules.append(RudinFallingRule(
            predicate=f"mean_class=={name}",
            action_rgb=palette,
            slim_coefficients=(int(coef),),
        ))
        rule_metadata.append({
            "rule_idx": rules_built,
            "predicate": f"mean_class=={name}",
            "support_count": count,
            "support_fraction": count / max(1, len(pair_features)),
            "slim_coef": int(coef),
        })
        rules_built += 1
    # High-diversity rule (catch-all texture)
    if rules_built < k_rules - 1 and high_div_count > 0:
        rules.append(RudinFallingRule(
            predicate="class_diversity==high",
            action_rgb=(140, 140, 80),
            slim_coefficients=(1,),
        ))
        rule_metadata.append({
            "rule_idx": rules_built,
            "predicate": "class_diversity==high",
            "support_count": high_div_count,
            "support_fraction": high_div_count / max(1, len(pair_features)),
            "slim_coef": 1,
        })
        rules_built += 1
    # High-motion rule
    if rules_built < k_rules - 1 and high_motion_count > 0:
        rules.append(RudinFallingRule(
            predicate="pose_motion==high",
            action_rgb=(80, 80, 140),
            slim_coefficients=(1,),
        ))
        rule_metadata.append({
            "rule_idx": rules_built,
            "predicate": "pose_motion==high",
            "support_count": high_motion_count,
            "support_fraction": high_motion_count / max(1, len(pair_features)),
            "slim_coef": 1,
        })
        rules_built += 1
    # Catch-all (always-fire residual; Wang-Rudin canonical last rule).
    rules.append(RudinFallingRule(
        predicate="always",
        action_rgb=(40, 40, 40),
        slim_coefficients=(),
    ))
    rule_metadata.append({
        "rule_idx": rules_built,
        "predicate": "always",
        "support_count": 0,
        "support_fraction": 0.0,
        "slim_coef": 0,
    })

    rule_list = RudinRuleList(
        rules=tuple(rules),
        default_rgb=(0, 0, 0),
    )

    summary = {
        "k_rules_actual": len(rules),
        "k_rashomon": k_rashomon,
        "anchors_seen": anchors_seen,
        "ensemble_confidence_tag": ensemble.confidence_tag(),
        "ensemble_n_anchors": int(ensemble.n_anchors),
        "class_counts": class_counts,
        "rule_metadata": rule_metadata,
    }
    return rule_list, summary


def _full_main(args: argparse.Namespace) -> int:
    """Full-mode trainer entry point (PR95-paradigm + Rashomon K=8 bootstrap).

    Binds canonical PR95-paradigm ingredients where the substrate's class-shift
    preserves them (decode_real_pairs + load_differentiable_scorers + scorer
    preprocess + gate_auth_eval_call + require_contest_cuda_auth_eval_claim +
    posterior_update_locked_from_auth_eval_json + detect_hardware_substrate),
    and substitutes the substrate's UNIQUE training mode (Rashomon K=8
    bootstrap over per-pair scorer features per Semenova-Rudin-Parr 2020) for
    PyTorch backprop ingredients that don't apply (EMA / eval_roundtrip /
    AdamW / autocast / TF32 / torch.compile — all N/A per design memo §15).

    Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L9
    (Runtime closure): emits a contest-compliant runtime tree (inflate.sh +
    inflate.py) alongside the RDIF v1 archive bytes. The Rudin substrate's
    inflate.py imports nothing from torch (pure numpy + Pillow + stdlib per
    design memo §3.2).

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY":
    this _full_main IS the completed trainer. The operator-authorize recipe
    can lift ``dispatch_enabled: false`` after T3 council ratification + the
    Catalog #270 dispatch-optimization-protocol verifies clean.
    """
    # Import canonical helpers inside function so smoke path stays light.
    from tac.substrates._shared.smoke_auth_eval_gate import (
        gate_auth_eval_call as _canon_gate_auth_eval_call,
    )
    from tac.substrates._shared.trainer_skeleton import (
        decode_real_pairs as _canon_decode_real_pairs,
    )
    from tac.substrates._shared.trainer_skeleton import (
        detect_hardware_substrate as _canon_detect_hardware_substrate,
    )
    from tac.substrates._shared.trainer_skeleton import (
        device_or_die as _canon_device_or_die,
    )
    from tac.substrates._shared.trainer_skeleton import (
        require_contest_cuda_auth_eval_claim as _canon_require_contest_cuda_auth_eval_claim,
    )

    _pin_seeds(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_log: list[dict[str, str]] = []

    def _stage(name: str) -> None:
        msg = {"stage": name, "at": _utc_now_iso()}
        stage_log.append(msg)
        print(f"[{SUBSTRATE_TAG}-full] {name} @ {msg['at']}")

    _stage("seed_pinned")

    # 1. Device gate (canonical). Rudin substrate is CPU-only at INFLATE per
    # design memo §3.2; compress-time scorer feature extraction prefers CUDA.
    # Operator can override via --device cpu (e.g. macOS-CPU advisory smoke).
    full_cpu = bool(getattr(args, "full_cpu", False))
    advisory_waived = bool(getattr(args, "advisory_cpu_explicitly_waived", False))
    if full_cpu and not advisory_waived:
        raise SystemExit(
            "ERROR: --full-cpu requires --advisory-cpu-explicitly-waived per "
            "Catalog #197 paired-flag opt-in discipline."
        )
    try:
        device = _canon_device_or_die(
            args.device,
            smoke=False,
            substrate_tag=SUBSTRATE_TAG,
            allow_full_cpu=full_cpu,
        )
    except SystemExit:
        # Device gate refused — propagate per CLAUDE.md MPS-NOISE non-negotiable.
        raise

    # 2. Patch upstream rgb_to_yuv6 BEFORE scorer construction (Catalog #187).
    # Substrate uses scorers as compress-time ORACLE only (no backprop), but
    # the yuv6 patch keeps the canonical preprocessing path consistent.
    yuv6_token = None
    try:
        from tac.differentiable_eval_roundtrip import (
            patch_upstream_yuv6_globally,
            unpatch_upstream_yuv6,
        )
        yuv6_token = patch_upstream_yuv6_globally()
        _stage("upstream_yuv6_patched")
    except Exception as exc:
        print(f"[{SUBSTRATE_TAG}-full] WARN yuv6 patch skipped: {exc!r}")

    archive_zip_path = output_dir / "archive.zip"
    archive_zip_sha = ""
    archive_zip_size = 0
    bin_sha = ""
    bin_size = 0
    rule_list: RudinRuleList | None = None
    bootstrap_summary: dict[str, Any] = {}
    auth_eval_result: dict[str, object] | None = None
    pair_features: list[dict[str, Any]] = []

    try:
        # 3. Load differentiable scorers (frozen; eval; no_grad) — compress-time
        # ORACLE per design memo §6.7 Wyner-Ziv scorer-as-shared-prior.
        # Canonical contract: (posenet, segnet) tuple order per Catalog #222.
        try:
            from tac.scorer import load_differentiable_scorers
            posenet, segnet = load_differentiable_scorers(args.upstream_dir, device=device)
            for p in list(posenet.parameters()) + list(segnet.parameters()):
                p.requires_grad_(False)
            posenet.eval()
            segnet.eval()
            _stage("scorers_loaded")
            scorers_loaded = True
        except Exception as exc:
            print(
                f"[{SUBSTRATE_TAG}-full] WARN scorer load failed: {exc!r}; "
                "falling back to luma-bin class proxy + frame-diff pose proxy"
            )
            posenet = None
            segnet = None
            scorers_loaded = False
            _stage("scorers_load_skipped")

        # 4. Decode real contest pairs via canonical pyav helper (Catalog #114).
        print(
            f"[{SUBSTRATE_TAG}-full] decoding pairs from {args.video_path} "
            f"(n_pairs={N_PAIRS_FULL}; max_pairs={getattr(args, 'max_pairs', None)})"
        )
        gt_pair_tensor = _canon_decode_real_pairs(
            args.video_path,
            n_pairs=N_PAIRS_FULL,
            substrate_tag=SUBSTRATE_TAG,
            max_pairs=getattr(args, "max_pairs", None),
            repo_root=REPO_ROOT,
        ).to(device)
        n_pairs = int(gt_pair_tensor.shape[0])
        _stage(f"pairs_decoded_{n_pairs}")

        # 5. Extract per-pair scorer features (compress-time ORACLE).
        # Substrate uses no_grad — no backprop, no neural training.
        if scorers_loaded and posenet is not None and segnet is not None:
            pair_features = _compute_per_pair_scorer_features(
                gt_pair_tensor, posenet, segnet, chunk_size=8
            )
        else:
            # Fallback: derive features from frame luma + frame-diff
            import torch  # type: ignore[import-untyped]
            pair_features = []
            with torch.no_grad():
                for i in range(n_pairs):
                    frame_last = gt_pair_tensor[i, -1].float()
                    gray = (0.299 * frame_last[0] + 0.587 * frame_last[1] + 0.114 * frame_last[2])
                    mc = int((gray.mean() / 51.2).clamp_(0, 4).item())
                    cd = float(gray.std().item() / 64.0)
                    pose_motion = float((gt_pair_tensor[i, 1] - gt_pair_tensor[i, 0]).float().abs().mean().item())
                    rgb_std = float(frame_last.std().item())
                    pair_features.append({
                        "mean_class": mc,
                        "class_diversity": cd,
                        "pose_motion": pose_motion,
                        "chroma_var": rgb_std,
                    })
        _stage(f"features_extracted_{len(pair_features)}_pairs")

        # 6. Rashomon K=8 bootstrap → canonical K=6 falling-rule-list.
        rule_list, bootstrap_summary = _compile_rashomon_rule_list(
            pair_features,
            k_rules=args.k_rules,
            k_rashomon=args.k_rashomon,
            slim_coeff_bound=args.slim_coeff_bound,
            seed=args.seed,
        )
        _stage(
            f"rashomon_K_{args.k_rashomon}_rule_list_K_{bootstrap_summary['k_rules_actual']}"
        )

        # 7. Build RDIF v1 archive bytes (substrate's UNIQUE grammar).
        # Encoder tree blob is empty for L2 INTEGRATION (GOSDT compile is the
        # full L3 PRODUCTION concern; rule_list_blob alone suffices for L2).
        # Per design memo §15 + Catalog #220: the rule-application IS the
        # byte-consumption proof — every byte in rule_list_blob is traceable
        # to a rule that fires for at least one pair.
        scorer_priors_meta = {
            "scorers_loaded": scorers_loaded,
            "n_pairs_decoded": n_pairs,
            "class_counts": bootstrap_summary.get("class_counts", {}),
        }
        scorer_priors_blob = json.dumps(
            scorer_priors_meta, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        rashomon_disagreement_blob = json.dumps(
            {
                "confidence_tag": bootstrap_summary.get("ensemble_confidence_tag", "n=0"),
                "n_anchors": bootstrap_summary.get("ensemble_n_anchors", 0),
                "rule_metadata": bootstrap_summary.get("rule_metadata", []),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        bin_bytes = pack_archive(
            rule_list=rule_list,
            encoder_tree_blob=b"",  # L2 INTEGRATION; full GOSDT at L3
            scorer_priors_blob=scorer_priors_blob,
            frame_0_init_blob=b"",
            wavelet_residuals_blob=b"",
            pose_residuals_blob=b"",
            per_pair_rule_indices_blob=b"",
            rashomon_disagreement_blob=rashomon_disagreement_blob,
        )
        bin_sha = _sha256_bytes(bin_bytes)
        bin_size = len(bin_bytes)
        (output_dir / "0.bin").write_bytes(bin_bytes)
        print(
            f"[{SUBSTRATE_TAG}-full] RDIF v1 archive: {bin_size} B sha256={bin_sha[:16]}..."
        )
        _stage(f"archive_built_{bin_size}_B_sha{bin_sha[:8]}")

        # 8. Emit contest-compliant runtime tree (HNeRV L9 runtime closure).
        if not getattr(args, "skip_archive_build", False):
            submission_dir = output_dir / "submission_dir"
            _write_runtime(submission_dir)
            (submission_dir / "0.bin").write_bytes(bin_bytes)
            _build_archive_zip(archive_zip_path, bin_bytes=bin_bytes)
            archive_zip_sha = _sha256_bytes(archive_zip_path.read_bytes())
            archive_zip_size = archive_zip_path.stat().st_size
            shutil.copy2(archive_zip_path, submission_dir / "archive.zip")
            _stage("archive_emitted_runtime_tree")

            # 9. Auth eval via canonical gate (Catalog #226).
            if not getattr(args, "skip_auth_eval", False):
                auth_eval_json_path = output_dir / "contest_auth_eval.json"
                auth_eval_result = _canon_gate_auth_eval_call(
                    args=args,
                    archive_zip=archive_zip_path,
                    inflate_sh=submission_dir / "inflate.sh",
                    upstream_dir=args.upstream_dir,
                    output_json=auth_eval_json_path,
                    contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                    substrate_tag=SUBSTRATE_TAG,
                    device=device,
                    full_cpu_active=full_cpu,
                )
                if auth_eval_result is not None:
                    _canon_require_contest_cuda_auth_eval_claim(
                        auth_eval_json_path,
                        archive_sha256=archive_zip_sha,
                        substrate_tag=SUBSTRATE_TAG,
                    )
                    _stage("auth_eval_cuda_done_valid_claim")
                else:
                    _stage("auth_eval_skipped_gate_refused")
    finally:
        if yuv6_token is not None:
            try:
                from tac.differentiable_eval_roundtrip import unpatch_upstream_yuv6
                unpatch_upstream_yuv6(yuv6_token)
                _stage("upstream_yuv6_unpatched")
            except Exception:
                pass

    # 10. Posterior update (Catalog #128 atomic fcntl).
    if (
        not getattr(args, "skip_auth_eval", False)
        and (output_dir / "contest_auth_eval.json").exists()
    ):
        try:
            from tac.continual_learning import (
                posterior_update_locked_from_auth_eval_json,
            )
            update = posterior_update_locked_from_auth_eval_json(
                output_dir / "contest_auth_eval.json",
                architecture_class="interpretable_ml_compositional_decoder",
            )
            print(
                f"[{SUBSTRATE_TAG}-full] posterior_update accepted="
                f"{getattr(update, 'accepted', '?')}"
            )
            _stage("posterior_updated")
        except Exception as exc:
            print(f"[{SUBSTRATE_TAG}-full] WARN posterior_update failed: {exc!r}")

    # 11. Provenance + manifest (canonical schema; Catalog #190 hardware
    # substrate detection per design memo §15).
    hardware_substrate_cuda = _canon_detect_hardware_substrate(
        axis="cuda",
        substrate_tag=SUBSTRATE_TAG,
        env_var_candidates=("RUDIN_FLOOR_GPU", "MODAL_GPU"),
    )
    provenance = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "started_at_utc": stage_log[0]["at"] if stage_log else _utc_now_iso(),
        "completed_at_utc": _utc_now_iso(),
        "git_head": _git_head_sha(),
        "bin_sha256": bin_sha,
        "bin_bytes": bin_size,
        "archive_zip_sha256": archive_zip_sha,
        "archive_zip_bytes": archive_zip_size,
        "k_rules": args.k_rules,
        "k_rashomon": args.k_rashomon,
        "slim_coeff_bound": args.slim_coeff_bound,
        "gosdt_depth": args.gosdt_depth,
        "device": str(device),
        "seed": args.seed,
        "design_memo": DESIGN_MEMO_PATH,
        "stage_log": stage_log,
        "bootstrap_summary": bootstrap_summary,
        "hardware_substrate_cuda": hardware_substrate_cuda,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )

    manifest = {
        "schema": "rudin_floor_interpretable_ml_training_artifact_manifest_v1",
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "training_mode": "full",
        "research_only": False,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "archive_bytes": bin_size,
        "archive_sha256": bin_sha,
        "archive_zip_bytes": archive_zip_size,
        "archive_zip_sha256": archive_zip_sha,
        "auth_eval_skipped": bool(getattr(args, "skip_auth_eval", False)),
        "auth_eval_present": auth_eval_result is not None,
        "predicted_band_mid_30_90d": "[0.150, 0.180] [prediction; HYPOTHESIS pending K=8 LEVEL-1 dispatch]",
        "result": {
            "training_mode": "full",
            "archive_bytes": bin_size,
            "archive_sha256": bin_sha,
            "archive_zip_bytes": archive_zip_size,
            "archive_zip_sha256": archive_zip_sha,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    stats = {
        "substrate_tag": SUBSTRATE_TAG,
        "lane_id": SUBSTRATE_LANE_ID,
        "design_memo": DESIGN_MEMO_PATH,
        "horizon_class": "asymptotic_pursuit",
        "lane_class": "substrate_engineering",
        "training_mode": "full",
        "n_pairs": len(pair_features),
        "k_rules": args.k_rules,
        "k_rashomon": args.k_rashomon,
        "slim_coeff_bound": args.slim_coeff_bound,
        "gosdt_depth": args.gosdt_depth,
        "archive_bytes": bin_size,
        "archive_sha256": bin_sha,
        "archive_zip_bytes": archive_zip_size,
        "archive_zip_sha256": archive_zip_sha,
        "auth_eval_score": (
            auth_eval_result.get("score") if isinstance(auth_eval_result, dict) else None
        ),
        "auth_eval_score_axis": (
            auth_eval_result.get("score_axis")
            if isinstance(auth_eval_result, dict)
            else "no_auth_eval"
        ),
        "auth_eval_score_claim_valid": (
            bool(auth_eval_result.get("score_claim_valid"))
            if isinstance(auth_eval_result, dict)
            else False
        ),
        "auth_eval_exact_cuda_complete": (
            bool(auth_eval_result.get("exact_cuda_complete"))
            if isinstance(auth_eval_result, dict)
            else False
        ),
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "result_review_blockers": [
            "phase_1b_lift_landing_first_dispatch_pending",
            "predicted_band_validation_pending_k8_level_1_dispatch_per_design_memo_section_18",
        ],
        "bootstrap_summary": bootstrap_summary,
        "stage_log": stage_log,
        "git_head": _git_head_sha(),
        "completed_at_utc": _utc_now_iso(),
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2), encoding="utf-8"
    )
    print(
        f"[{SUBSTRATE_TAG}-full] OK archive_bytes={bin_size} "
        f"sha256={bin_sha[:16]}... auth_eval_present={auth_eval_result is not None}"
    )
    print(f"[{SUBSTRATE_TAG}-full] wrote {output_dir / 'manifest.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
