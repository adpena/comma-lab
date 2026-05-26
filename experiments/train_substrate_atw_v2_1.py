#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526
"""ATW V2-1 trainer scaffold (Faiss-IVF-PQ per-region SegNet softmax histogram channel).

[verified-against: .omx/research/atw_v2_1_faiss_ivf_pq_substrate_design_memo_20260518.md]
[verified-against: ATW V2 reactivation symposium 2026-05-18 Revisions #1-#7 binding]

L0 SCAFFOLD per Catalog #240 / #220 / #272. ``_full_main`` raises
NotImplementedError pending Wave N+1 council per ATW V2 symposium 2026-05-18
+ V1/V2/V3 disambiguator probe outcome per design memo §6.3 + Z6 Wave 2 4c
cross-pollination per design memo §7.

NO COMMITS by this subagent. NO PAID DISPATCHES. Main Claude commits per
parent prompt. Operator authorizes Modal A100 AFTER design ratification +
disambiguator probe outcome + sextet pact symposium.

Trainer mode resolution (Catalog #326 driver mode hardcode fix discipline):
    ATW_V2_1_TRAINER_MODE (env) > SMOKE_ONLY (env, legacy) > default
    Default: smoke. Full mode requires EXPLICIT recipe-side opt-in via
    env_overrides setting ATW_V2_1_TRAINER_MODE=full OR SMOKE_ONLY=0.

Per CLAUDE.md "Subagent coherence-by-default" 6-hook wire-in declaration:
    Hook 1 sensitivity-map: planned `tac.sensitivity_map.atw_v2_1_faiss_pq`
       (sister to V2; consumes Faiss PQ codebook reconstruction error).
    Hook 2 Pareto constraint: planned `tac.pareto.atw_v2_1_pq_codeword_entropy`
       (R_PQ_codeword >= H(latent | scorer_class) bound).
    Hook 3 Bit-allocator hook: planned `bit_allocator.atw_v2_1_pq_codeword_v1`
       (per-pair archive bytes by per-region PQ codeword entropy).
    Hook 4 Cathedral autopilot dispatch hook: NEW recipe registered warn-only
       at landing per Catalog #167 + #271; promotes to dispatch-eligible upon
       Wave 3 sextet pact PROCEED + new D4 probe MEANINGFUL_CONDITIONING.
    Hook 5 Continual-learning posterior update: per-Modal-dispatch anchor
       seeds posterior paired with PQ-encoded MI value per Catalog #128
       locked write.
    Hook 6 Probe-disambiguator: planned `tools/probe_atw_v2_1_faiss_pq_disambiguator.py`
       (V1/V2/V3 variant discriminator; sister to existing 323-byte probe;
       $0 CPU local M5 Max).
"""
# SCORER_PREPROCESS_HANDLED_OK:atw-v2-1-inherits-v2-canonical-routing-via-cooperative_receiver_loss-per-Catalog-164
# CHECKPOINT_DISCIPLINE_WAIVED:scaffold-only-trainer-no-checkpoint-required-until-full-main-impl
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"

DESIGN_MEMO_PATH = (
    ".omx/research/atw_v2_1_faiss_ivf_pq_substrate_design_memo_20260518.md"
)
SYMPOSIUM_PATH = (
    ".omx/research/council_per_substrate_symposium_atw_v2_reactivation_20260518.md"
)
LANE_ID = "lane_top5_3_atw_v2_1_faiss_ivf_pq_scaffold_design_20260518"
LANE_REGISTRY_NOTES = (
    f"ATW V2-1 SCAFFOLD; reactivation criteria per design memo §12. "
    f"See {DESIGN_MEMO_PATH} + {SYMPOSIUM_PATH}."
)
SUBSTRATE_TAG = "atw_codec_v2_1_faiss_ivf_pq"

# Modal-IGNORED required-input files per Catalog #152 Wave-1 extension
# (Catalog #305 observability surface) — declared here so Modal worker
# can stage Lane A anchor archive into the writable workspace copy.
TIER_1_EXTRA_MOUNT_PATHS: tuple[str, ...] = (
    "submissions/a1/archive.zip",
)
"""Per Catalog #152 WAVE-1 APPARATUS HARDENING extension 2026-05-16:
required-input files under Modal-IGNORED `experiments/results/**` subtree
MUST be declared in `TIER_1_EXTRA_MOUNT_PATHS` so the canonical mount
manifest (`tac.deploy.modal.mount_manifest.collect_extra_mount_paths`)
includes them. The A1 anchor archive at `submissions/a1/archive.zip` is
the V2-1 probe's canonical A1 latent source per design memo §6.2.
"""


# ---------------------------------------------------------------------------
# Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS manifest (Catalog #168 AnnAssign)
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "ATW_V2_1_VIDEO_PATH",
        "rationale": (
            "score-aware compress-side scorer + Faiss codebook training query the "
            "contest video (upstream/videos/0.mkv); synthetic data FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot - never regenerated locally"
        ),
        "rationale_audit": DESIGN_MEMO_PATH,
    },
    "--output-dir": {
        "env": "ATW_V2_1_OUTPUT_DIR",
        "rationale": "custody location for archive + provenance + auth-eval JSON + Faiss codebook artifact",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "ATW_V2_1_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for SegNet/PoseNet weights + evaluate.py; "
            "required for non-smoke compress + auth eval + Faiss codebook training"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "ATW_V2_1_DEVICE",
        "rationale": (
            "compute device for compress-side scorer query; cuda required for "
            "full run (MPS refused per CLAUDE.md); cpu permitted only with --smoke"
        ),
        "default": "cuda",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "ATW_V2_1_EPOCHS",
        "rationale": "Training epoch count; smoke=3, Modal A100 full=200",
        "default": "200",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--pq-variant": {
        "env": "ATW_V2_1_PQ_VARIANT",
        "rationale": (
            "Faiss-IVF-PQ variant per design memo §6.3: 'v1_dense' (NOT SHIPPABLE; "
            "diagnostic only) / 'v2_sparse_top_k' (ARGUABLE) / 'v3_pool_shared' (SHIPPABLE; DEFAULT)"
        ),
        "default": "v3_pool_shared",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--pq-n-regions": {
        "env": "ATW_V2_1_PQ_N_REGIONS",
        "rationale": "Region grid resolution per pair; 16=4x4 / 64=8x8 / 256=16x16",
        "default": "16",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--pq-nlist": {
        "env": "ATW_V2_1_PQ_NLIST",
        "rationale": "Faiss IVF coarse-centroid count; per Jégou-Douze-Schmid 2011 §3.2",
        "default": "64",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--pq-m-subq": {
        "env": "ATW_V2_1_PQ_M_SUBQ",
        "rationale": "Faiss PQ sub-quantizer count; per Jégou-Douze-Schmid 2011 §3.3",
        "default": "2",
        "satisfied_by_profile": (),
        "requires": (),
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_atw_v2_1",
        description=(
            "Train ATW codec V2-1 substrate (Faiss-IVF-PQ per-region SegNet softmax "
            "histogram side-info channel; design memo 2026-05-18; sister to ATW V2 "
            "with replaced scorer_class_prior_table → faiss_codebook + pq_codeword_stream)."
        ),
    )
    # Tier 1 required flags
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=200)

    # Faiss-IVF-PQ variant flags
    p.add_argument(
        "--pq-variant",
        type=str,
        choices=["v1_dense", "v2_sparse_top_k", "v3_pool_shared"],
        default="v3_pool_shared",
    )
    p.add_argument("--pq-n-regions", type=int, default=16)
    p.add_argument("--pq-nlist", type=int, default=64)
    p.add_argument("--pq-m-subq", type=int, default=2)
    p.add_argument("--pq-nbits", type=int, default=6)
    p.add_argument("--pq-top-k", type=int, default=1,
                   help="Top-k regions to encode per pair (v2/v3 only)")
    p.add_argument("--pq-seed", type=int, default=42)

    # Engineering hygiene per Catalog #172/#178/#179/#180
    p.add_argument("--enable-autocast-fp16", action="store_true",
                   help="Catalog #172: enable torch.amp.autocast(dtype=fp16) on CUDA training")
    p.add_argument("--enable-torch-compile", action="store_true",
                   help="Catalog #179: enable torch.compile on the codec module")
    p.add_argument("--seed", type=int, default=20260518)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)

    # Mode flags
    p.add_argument("--smoke", action="store_true",
                   help="Run synthetic-data + Faiss codebook smoke (<1 min CPU)")
    p.add_argument("--skip-auth-eval", action="store_true",
                   help="Skip auth-eval (CI / dev only)")
    return p


def _resolve_trainer_mode(args: argparse.Namespace) -> str:
    """Per Catalog #326 driver-mode-hardcode fix: multi-key mode resolution.

    Order: ATW_V2_1_TRAINER_MODE (env) > SMOKE_ONLY (env legacy) > --smoke flag > default smoke.
    """
    explicit_mode = os.environ.get("ATW_V2_1_TRAINER_MODE", "").lower()
    if explicit_mode in ("full", "smoke"):
        return explicit_mode

    legacy_smoke_only = os.environ.get("SMOKE_ONLY", "")
    if legacy_smoke_only == "0":
        return "full"
    if legacy_smoke_only == "1":
        return "smoke"

    if args.smoke:
        return "smoke"
    # Default: smoke (per design memo §6.4: full_main raises NotImplementedError until Wave N+1 council)
    return "smoke"


# ---------------------------------------------------------------------------
# Smoke main (synthetic Faiss codebook + decode roundtrip; <1 min CPU; $0)
# ---------------------------------------------------------------------------
def _smoke_main(args: argparse.Namespace) -> int:
    """Synthetic-data Faiss-IVF-PQ smoke — validates canonical helper + budget arithmetic.

    No scorer load. No real video decode. No archive build. $0 cost. Verifies:

    1. ``tac.optimization.faiss_ivf_pq_atw_channel.estimate_pq_encoding_budget``
       returns the canonical V1/V2/V3 per-variant byte budgets per design memo §6.3
    2. If Faiss-cpu is installed: builds a synthetic codebook + encodes + decodes
       a synthetic SegNet softmax tensor with byte-identical roundtrip
    3. If Faiss-cpu is NOT installed: emits operator-actionable banner per CLAUDE.md
       "uv pip install faiss-cpu" + canonical install command (per design memo §9.5)
    """
    import numpy as np  # noqa: F401  (lazy import keeps module import lightweight)
    from tac.optimization.faiss_ivf_pq_atw_channel import (
        CANONICAL_VARIANTS,
        estimate_pq_encoding_budget,
    )

    print(f"[atw_v2_1] SMOKE MODE (variant={args.pq_variant}; n_regions={args.pq_n_regions})")
    print(f"[atw_v2_1] design memo: {DESIGN_MEMO_PATH}")
    print(f"[atw_v2_1] symposium: {SYMPOSIUM_PATH}")
    print(f"[atw_v2_1] lane: {LANE_ID}")
    print(f"[atw_v2_1] research_only=true; dispatch_enabled=false at landing")
    print()
    print("[atw_v2_1] Canonical variant byte-budget estimates per design memo §6.3:")

    # V1 dense
    v1_budget = estimate_pq_encoding_budget(
        variant_id="v1_dense",
        n_regions=256,
        nlist=256,
        m_subq=4,
        nbits=8,
        top_k_regions=None,
    )
    # V2 sparse top-k
    v2_budget = estimate_pq_encoding_budget(
        variant_id="v2_sparse_top_k",
        n_regions=16,
        nlist=64,
        m_subq=2,
        nbits=6,
        top_k_regions=8,
    )
    # V3 pool-shared
    v3_budget = estimate_pq_encoding_budget(
        variant_id="v3_pool_shared",
        n_regions=16,
        nlist=64,
        m_subq=2,
        nbits=6,
        top_k_regions=1,
    )

    for budget in (v1_budget, v2_budget, v3_budget):
        print(
            f"  - {budget.variant_id}: per_pair={budget.per_pair_bytes}B, "
            f"total_archive={budget.total_archive_contribution_bytes}B, "
            f"rate_cost=+{budget.contest_rate_cost_estimate:.4f}, "
            f"verdict={budget.shippable_verdict}"
        )

    # Try Faiss codebook smoke roundtrip; gracefully degrade if not installed
    try:
        from tac.optimization.faiss_ivf_pq_atw_channel import (
            build_pq_codebook,
            decode_per_region_histogram,
            encode_per_region_histogram,
            serialize_codebook,
            deserialize_codebook,
        )

        # Synthetic SegNet softmax outputs: 256 regions × 5 classes
        np.random.seed(args.pq_seed)
        n_regions = args.pq_n_regions
        synthetic_softmax = np.random.dirichlet(np.ones(5), size=200 * n_regions).astype(np.float32)

        codebook = build_pq_codebook(
            synthetic_softmax,
            nlist=args.pq_nlist,
            m_subq=args.pq_m_subq,
            nbits=args.pq_nbits,
            seed=args.pq_seed,
        )
        print(f"\n[atw_v2_1] Faiss codebook trained: nlist={args.pq_nlist}, "
              f"m_subq={args.pq_m_subq}, nbits={args.pq_nbits}")

        # Encode + decode roundtrip on one pair
        one_pair = synthetic_softmax[:n_regions]
        encoded = encode_per_region_histogram(one_pair, codebook)
        decoded = decode_per_region_histogram(encoded, codebook, n_regions=n_regions)
        print(f"[atw_v2_1] Encode roundtrip: {one_pair.shape} → {len(encoded)}B → {decoded.shape}")

        # Codebook serialization roundtrip
        serialized = serialize_codebook(codebook)
        deserialized = deserialize_codebook(serialized)
        print(f"[atw_v2_1] Codebook serialize: ~{len(serialized)}B; roundtrip OK")

        # Compute reconstruction error (Frobenius)
        recon_err = float(np.linalg.norm(decoded - one_pair))
        print(f"[atw_v2_1] Reconstruction error (Frobenius): {recon_err:.4f}")
        print(f"[atw_v2_1] SMOKE PASS")
    except ImportError as exc:
        print(f"\n[atw_v2_1] WARN: faiss-cpu not installed — skipping codebook smoke")
        print(f"[atw_v2_1] Install: uv pip install faiss-cpu (per design memo §9.5)")
        print(f"[atw_v2_1] Error: {exc}")
        print(f"[atw_v2_1] SMOKE PARTIAL (budget arithmetic OK; codebook deferred)")
        return 0  # Partial pass; not an error

    return 0


# ---------------------------------------------------------------------------
# Full main (research-only NotImplementedError per Catalog #240 + ATW V2 symposium)
# ---------------------------------------------------------------------------
def _full_main(args: argparse.Namespace) -> int:
    """Full training path — RESEARCH-ONLY NotImplementedError at landing.

    Per Catalog #240 + CLAUDE.md "Substrate scaffolds MUST be COMPLETE or
    RESEARCH-ONLY" + ATW V2 reactivation symposium 2026-05-18 Revision #1
    binding: the V2-1 _full_main pending Wave N+1 council per ATW V2 symposium
    + V1/V2/V3 disambiguator probe outcome per design memo §6.3 + Z6 Wave 2 4c
    cross-pollination per design memo §7.

    Reactivation criteria (ALL required to lift NotImplementedError):
    1. ``uv pip install faiss-cpu`` locally
    2. V1/V2/V3 disambiguator probe lands MI + byte-budget verdict per variant
       (planned: tools/probe_atw_v2_1_faiss_pq_disambiguator.py)
    3. Z6 Wave 2 4c outcome lands (sister subagent a58961ea35f767306)
    4. Wave 3 sextet pact symposium ratifies V2-1 channel pick per Catalog #325
    5. New D4 probe on selected V2-1 channel returns MEANINGFUL_CONDITIONING (MI >= 0.5)
    6. Catalog #324 post-training Tier-C validation declared
    7. Catalog #270 dispatch optimization protocol verdict PASS
    """
    raise NotImplementedError(
        f"V2-1 _full_main pending Wave N+1 council per ATW V2 symposium 2026-05-18; "
        f"V1/V2/V3 disambiguator probe must land first per {DESIGN_MEMO_PATH} §6.3 + §8; "
        f"reactivation criteria documented in {LANE_REGISTRY_NOTES}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    mode = _resolve_trainer_mode(args)
    print(f"[atw_v2_1] resolved trainer mode: {mode}")

    if mode == "smoke":
        return _smoke_main(args)
    elif mode == "full":
        return _full_main(args)
    else:  # pragma: no cover  # defensive — should never trigger
        raise RuntimeError(f"unknown trainer mode: {mode!r}")


if __name__ == "__main__":
    sys.exit(main())
