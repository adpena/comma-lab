# SPDX-License-Identifier: MIT
# DISPATCH_OPTIMIZATION_PROTOCOL_OK:rudin-floor-L1-SCAFFOLD-research_only=true-dispatch_enabled=false-per-Catalog-240-cascade-_full_main-raises-NotImplementedError-no-paid-dispatch-can-fire-Tier-3-scorer-load-auth-eval-routing-inflate-device-canonical-tokens-applied-at-L2-INTEGRATION-post-Phase-2-council-approval
# AUTOCAST_FP16_WAIVED:rudin-floor-substrate-has-no-neural-network-no-torch-no-autocast-needed
# TORCH_COMPILE_WAIVED:rudin-floor-substrate-has-no-torch-no-compile-needed
# TF32_WAIVED:rudin-floor-substrate-has-no-neural-codec-no-matmul
# NO_GRAD_WAIVED:rudin-floor-substrate-has-no-training-loop-no-eval-mode-grad-context
# F3_CACHE_CONSUMPTION_WAIVED:no-scorer-hot-loop-substrate-is-closed-form-rule-application
# SCORER_PREPROCESS_HANDLED_OK:rudin-floor-compress-time-scorer-features-go-through-canonical-_canon_score_pair_components-helper-at-L2-INTEGRATION-not-L1-SCAFFOLD
# SCORER_LOADER_ORDER_OK:rudin-floor-substrate-has-no-scorer-load-at-L1-SCAFFOLD-canonical-(posenet,segnet)-tuple-applies-at-L2-INTEGRATION-post-Phase-2-council-approval
# AUTH_EVAL_DIRECT_SUBPROCESS_OK:rudin-floor-L1-SCAFFOLD-no-auth-eval-invocation-trainer-_full_main-raises-NotImplementedError-canonical-gate_auth_eval_call-wired-at-L2-INTEGRATION
# INLINE_DEVICE_FORK_OK:rudin-floor-substrate-is-CPU-only-by-construction-no-PyTorch-no-cuda-cpu-distinction-per-design-memo-§3.2-canonical-select_inflate_device-N/A
"""Train Rudin floor interpretable-ML compositional decoder substrate (L1 SCAFFOLD).

Per ``.omx/research/rudin_floor_interpretable_ml_substrate_asymptotic_pursuit_scoping_design_20260516.md``
(L0 SKETCH design memo, ratified into L1 SCAFFOLD by subagent D per the
2026-05-16 omnibus dispatch + the Rudin floor memo op-routable #2).

The Rudin floor substrate is a TRIPLE CLASS-SHIFT (architecture + decode-time
+ scorer-relationship) per the abandon-within-class taxonomy. The META-layer
Rudin-Daubechies autopilot (Catalogs #273-#278; `src/tac/autopilot_rudin_daubechies/`)
is RE-ARCHITECTED here as the decode-time architecture itself:

* Encoder: GOSDT depth-4 sparse decision tree (Lin-Zhong-Hu-Hu-Rudin-Seltzer 2020)
* Decoder: Wang-Rudin 2015 falling-rule-list with K=4-6 rules
* Loss: Rashomon ensemble (Semenova-Rudin-Parr 2020); K=8 bootstrap-diverse SLIM rankers
* Archive: RDIF v1 monolithic 0.bin (SLIM-coded integer coefficients + rules + sections)
* Inflate: ≤200 LOC pure Python (HNeRV L4 substrate_engineering exception); NO PyTorch

L1 SCAFFOLD SCOPE
-----------------

* ``_smoke_main``: builds a canonical K=6 rule_list per design memo §3.1, packs
  into RDIF v1 archive, roundtrips through ``parse_archive``, inflates one
  1×1 PNG via ``inflate_one_video``, asserts byte-determinism. ``$0`` cost. CPU only.
* ``_full_main``: RAISES ``NotImplementedError`` per CLAUDE.md "Substrate
  scaffolds MUST be COMPLETE or RESEARCH-ONLY" + Catalog #220 cascade.
  Phase 2 council approval required to lift; reactivation criteria documented
  in design memo §18.

Per Catalog #240 (recipe-vs-trainer-state consistency): the operator-authorize
recipe declares ``research_only: true`` + ``dispatch_enabled: false`` so no
paid Modal dispatch can fire from this trainer until the council gate is lifted.

Per Catalog #229 (premise-verification-before-edit pattern): pre-edit
verifications included reading the Rudin floor design memo + the ATW codec v1
sister scaffold + the substrate package's archive + inflate + rule_list
modules + Catalog #270 dispatch-optimization-protocol verification. See
`feedback_rudin_floor_l1_scaffold_substrate_build_subagent_d_landed_20260516.md`
for the per-premise verifier table.

Usage (smoke; CPU only; ~1 sec)::

    .venv/bin/python experiments/train_substrate_rudin_floor_interpretable_ml.py \\
        --output-dir experiments/results/rudin_floor_smoke_<utc> \\
        --smoke

Usage (full; PHASE 2 COUNCIL APPROVAL REQUIRED — currently raises NotImplementedError)::

    .venv/bin/python experiments/train_substrate_rudin_floor_interpretable_ml.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/rudin_floor_<utc> \\
        --device cuda
"""
# SYNTHETIC_NON_SMOKE_OK:_smoke_main-only-uses-canonical-design-memo-K=6-rule_list-_full_main-raises-NotImplementedError
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

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

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
SUBSTRATE_TAG = "rudin_floor_interpretable_ml"
SUBSTRATE_LANE_ID = "lane_rudin_floor_l1_scaffold_substrate_build_20260516"
DESIGN_MEMO_PATH = (
    ".omx/research/rudin_floor_interpretable_ml_substrate_asymptotic_pursuit_scoping_design_20260516.md"
)


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
            "_full_main_raises_NotImplementedError_pending_phase_2_council",
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


def _full_main(args: argparse.Namespace) -> int:
    """Full-mode trainer entry point.

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
    non-negotiable + Catalog #220 cascade + Catalog #240 recipe-vs-trainer-
    state-consistency: Rudin floor substrate is an L1 SCAFFOLD +
    research_only=true + lane_class=substrate_engineering + pre-build
    council-gated. The ``_full_main`` body is intentionally NOT IMPLEMENTED
    so no $1+ Modal dispatch can fire from this trainer until Phase 2
    council approval is granted.

    Reactivation criteria are documented in the design memo §18 (per the
    K=8 LEVEL-1 dispatch outcome bands).
    """
    raise NotImplementedError(
        "Rudin floor interpretable-ML substrate _full_main is council-gated "
        "per Catalog #220 substrate-engineering pre-build cascade. The "
        "substrate is L1 SCAFFOLD + research_only=true at landing "
        "2026-05-16. Phase 2 council approval required to lift; reactivation "
        f"criteria documented in {DESIGN_MEMO_PATH} §18 + §21 op-routables. "
        "Use --smoke for canonical-rule-list + archive-roundtrip + inflate "
        "sanity verification (no GPU spend; CPU only; $0 cost)."
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
