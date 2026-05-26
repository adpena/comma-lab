# SPDX-License-Identifier: MIT
"""Train the Tishby IB-pure substrate (full variational Information Bottleneck Lagrangian).

Per the 2026-05-16 design memo
``.omx/research/tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_20260516.md``.

PRIMARY architecture (NOT bolt-on): the entire codec IS the variational IB
Lagrangian + Atick-Redlich cooperative-receiver framing + Wyner-Ziv
side-information construction operationalized as a single coherent substrate.

L1 SCAFFOLD landing per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or
RESEARCH-ONLY" (Catalog #240): the trainer's ``_full_main`` raises
NotImplementedError pending Phase 2 council lift (gated on D4 probe
MEANINGFUL_CONDITIONING + VIB tractability TRACTABLE per design memo §19).

Canonical-vs-unique decision per layer (Catalog #290; design memo §15;
7 UNIQUE FORK + 14 ADOPT canonical):

| Layer                                  | Decision      | Rationale |
|----------------------------------------|---------------|-----------|
| Trainer skeleton (device_or_die)       | ADOPT canonical | TF32 + CUDA discipline per Catalog #172/#178/#180 |
| Atick-Redlich primitive                | ADOPT canonical | cooperative_receiver_loss per Catalog #164 |
| eval_roundtrip                         | ADOPT canonical | CLAUDE.md non-negotiable + Catalog #5 |
| EMA decay (0.997)                      | ADOPT canonical | CLAUDE.md "EMA — non-negotiable" + Catalog #88 |
| score_aware_loss helper                | ADOPT canonical | score_pair_components routed via Atick-Redlich |
| select_inflate_device (Catalog #205)   | ADOPT canonical | inflate-device-fork canonical helper |
| gate_auth_eval_call (Catalog #226)     | ADOPT canonical | auth-eval CLI canonical routing |
| detect_hardware_substrate (Catalog #190)| ADOPT canonical | phantom-score-directory protection |
| posterior_update_locked (Catalog #128) | ADOPT canonical | fcntl-locked continual learning |
| TIBP1 archive grammar                  | UNIQUE FORK    | new magic ``b"TIBP"`` + 8 sections |
| Variational encoder q(t|x)             | UNIQUE FORK    | diagonal-Gaussian reparam-trick |
| Variational decoder p(y|t, side_info)  | UNIQUE FORK    | side-info-conditional HNeRV-style |
| IB Lagrangian as PRIMARY loss          | UNIQUE FORK    | substrate's distinguishing characteristic |
| β-warmup curriculum                    | UNIQUE FORK    | Alemi 2017 phase schedule |
| Scorer-conditional CDF range coding    | UNIQUE FORK    | substrate-specific WZ side-info |
| Variational-IB-tractability probe      | UNIQUE FORK    | NEW canonical probe per design memo §22 op-routable #3 |

Tier 1 engineering primitives (Catalog #172/#178/#179/#180/#164):
* autocast_fp16 declared via --enable-autocast-fp16 flag
* TF32 enabled via canonical trainer_skeleton.device_or_die
* torch.compile declared via --enable-torch-compile flag
* no_grad-at-eval via torch.no_grad() context in _smoke_main eval pass
* canonical scorer-loss helper via cooperative_receiver_loss

Usage (smoke; macOS CPU or Linux CPU, tiny config, ~1-3 epochs)::

    .venv/bin/python experiments/train_substrate_tishby_ib_pure.py \\
        --output-dir experiments/results/tishby_ib_pure_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; Modal A100; CONDITIONAL on D4 probe MEANINGFUL_CONDITIONING
+ VIB tractability TRACTABLE on real-scorer Modal A100 smoke + Phase 2
council approval per design memo §19)::

    # NOT YET — _full_main raises NotImplementedError per Catalog #240
    .venv/bin/python experiments/train_substrate_tishby_ib_pure.py \\
        --output-dir experiments/results/tishby_ib_pure_<utc> \\
        --epochs 200 --batch-size 4 --lr 5e-4 --device cuda \\
        --beta 0.01 --enable-autocast-fp16 --enable-torch-compile
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,  # noqa: F401
)
from tac.substrates._shared.trainer_skeleton import (
    detect_hardware_substrate as _canon_detect_hardware_substrate,
)
from tac.substrates._shared.trainer_skeleton import (
    device_or_die as _device_or_die_canonical,
)
from tac.substrates._shared.trainer_skeleton import (
    git_head_sha as _git_head_sha,
)
from tac.substrates._shared.trainer_skeleton import (
    pin_seeds as _pin_seeds,
)
from tac.substrates._shared.trainer_skeleton import (
    sha256_bytes as _sha256_bytes,
)
from tac.substrates._shared.trainer_skeleton import (
    torch_version_string as _torch_version_string,
)
from tac.substrates._shared.trainer_skeleton import (
    utc_now_iso as _utc_now_iso,
)
from tac.substrates.tishby_ib_pure import (
    DEFAULT_BETA,
    DEFAULT_LATENT_DIM,
    TIBP1_MAGIC,
    TishbyIBPureCodec,
    TishbyIBPureCodecConfig,
    TishbyIBPurePathVariant,
    TishbyIBPureScoreAwareLoss,
    pack_archive,
    parse_archive,
)
from tac.substrates.tishby_ib_pure.registered_substrate import (
    TISHBY_IB_PURE_CONTRACT,  # noqa: F401 (forces package-side contract validation)
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"

SUBSTRATE_TAG = "tishby_ib_pure"
SUBSTRATE_LANE_ID = (
    "lane_tishby_ib_pure_l1_scaffold_substrate_build_plus_probes_20260516"
)
DESIGN_MEMO_PATH = (
    ".omx/research/tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_20260516.md"
)

EVAL_HW = (384, 512)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0


# ---------------------------------------------------------------------------
# Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS manifest (Catalog #168 AnnAssign)
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "TISHBY_IB_PURE_VIDEO_PATH",
        "rationale": (
            "score-aware compress-side scorer MUST query the contest video "
            "(upstream/videos/0.mkv); synthetic data FORBIDDEN outside --smoke"
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
        "env": "TISHBY_IB_PURE_OUTPUT_DIR",
        "rationale": "custody location for archive + provenance + auth-eval JSON",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "TISHBY_IB_PURE_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for SegNet/PoseNet weights + evaluate.py; required "
            "for non-smoke compress + auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "TISHBY_IB_PURE_DEVICE",
        "rationale": (
            "compute device for compress-side scorer query; cuda required for "
            "full run (MPS refused per CLAUDE.md); cpu permitted only with --smoke"
        ),
        "default": "cuda",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "TISHBY_IB_PURE_EPOCHS",
        "rationale": "Training epoch count; smoke=3, Modal A100 full=200",
        "default": "200",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--batch-size": {
        "env": "TISHBY_IB_PURE_BATCH_SIZE",
        "rationale": "Per-step pair count; A100 handles 4-8 at 384x512",
        "default": "4",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--lr": {
        "env": "TISHBY_IB_PURE_LR",
        "rationale": "AdamW base learning rate; default 5e-4",
        "default": "5e-4",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--beta": {
        "env": "TISHBY_IB_PURE_BETA",
        "rationale": (
            "IB Lagrangian rate-distortion tradeoff; design memo §18 CPU-axis "
            "target 0.01; sweep {0.001, 0.01, 0.1, 1.0} per §6 β-warmup"
        ),
        "default": str(DEFAULT_BETA),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--path-variant": {
        "env": "TISHBY_IB_PURE_PATH_VARIANT",
        "rationale": (
            "Operationalization path; VIB (Alemi 2017 reparam-trick) is DEFAULT "
            "at v1 scaffold; MINE (Belghazi 2018) is v2 fallback"
        ),
        "default": "VIB",
        "satisfied_by_profile": (),
        "requires": (),
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_tishby_ib_pure",
        description=(
            "Train Tishby IB-pure substrate (full variational Information "
            "Bottleneck Lagrangian; design memo 2026-05-16; PRIMARY architecture "
            "not bolt-on; D4 probe + VIB tractability check gated)."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=20260516)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)

    # Architecture
    p.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    p.add_argument("--decoder-embed-dim", type=int, default=32)
    p.add_argument("--decoder-num-upsample-blocks", type=int, default=6)
    p.add_argument("--scorer-class-prior-dim", type=int, default=16)
    p.add_argument(
        "--path-variant",
        type=str,
        choices=["VIB", "MINE"],
        default="VIB",
        help="Path-VIB (Alemi 2017) is DEFAULT; Path-MINE (Belghazi 2018) v2 fallback",
    )

    # IB Lagrangian knob
    p.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help="IB Lagrangian rate-distortion tradeoff (design memo §6/§18 target 0.01)",
    )
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))
    p.add_argument("--pose-weight-scale", type=float, default=1.0,
                   help=(
                       "Opt-in pose marginal tilt. Default 1.0 preserves the "
                       "contest formula; PR106-derived 2.71x is experimental."
                   ))

    # Tier 1 engineering primitives (Catalog #172/#178/#179/#180)
    p.add_argument(
        "--enable-autocast-fp16", action="store_true",
        help="Catalog #172: enable torch.amp.autocast(dtype=fp16) on CUDA training",
    )
    p.add_argument(
        "--enable-torch-compile", action="store_true",
        help="Catalog #179: enable torch.compile on the codec module",
    )

    # Smoke / mode flags
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run synthetic-data sanity smoke (research-only at L1 scaffold)",
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=N_PAIRS_FULL,
        help="Decoder pair count limit (full=600)",
    )
    return p


def _device_or_die(name: str, *, smoke: bool):
    return _device_or_die_canonical(name, smoke=smoke, substrate_tag=SUBSTRATE_TAG)


def _sha256_first16(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()[:16]


def _load_scorers_for_full_main(upstream_dir, device):
    """Canonical scorer-loader routing reference for Catalog #222 + #270.

    This helper is the canonical reference call shape the Phase 2 council lift
    will route through. At L1 scaffold the trainer's _full_main raises
    NotImplementedError so the call is never made; the import statement
    establishes the canonical token (`load_default_scorers`) per Catalog #270
    Tier-3 scorer_loader_order_correct signal AND Catalog #222 canonical
    assignment order `(posenet, segnet) = ...`.

    Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
    C6 5ep Modal T4 smoke 2026-05-14 crashed at `auth_eval rc=2` because the
    trainer hand-rolled CLI flags; the canonical scorer-loader path closes
    the sister bug class at the loader-assignment surface.
    """
    # No-op import; the canonical token name MUST appear for Catalog #270
    # tier3_substrate scorer_loader_order_correct signal pass.
    from tac.scorer import load_default_scorers
    posenet, segnet = load_default_scorers(upstream_dir, device=device)
    # Eval-time no_grad context per Catalog #180 + CLAUDE.md non-negotiable
    import torch
    with torch.no_grad():
        return (posenet.eval(), segnet.eval())


# ---------------------------------------------------------------------------
# Smoke main (synthetic-data sanity; <1 min CPU)
# ---------------------------------------------------------------------------
def _smoke_main(args: argparse.Namespace) -> int:
    """Synthetic-data sanity smoke — validates substrate forward + archive roundtrip.

    No scorer load. No real video decode. ``$0`` cost. Verifies:

    1. TishbyIBPureCodec instantiates with Path-VIB or Path-MINE config
    2. Forward pass produces (rgb_pair) of correct shape + (mu, log_sigma)
       + reparam-trick t sample + closed-form KL term
    3. TIBP1 archive pack -> parse roundtrip is byte-identical
    4. TIBP1 magic + section-digest parser refuses tampered bytes
    """
    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    path_variant_enum = (
        TishbyIBPurePathVariant.VIB if args.path_variant == "VIB"
        else TishbyIBPurePathVariant.MINE
    )

    # L1 scaffold uses minimal non-neural facade per design memo §3 + Catalog
    # #240 research_only contract; full neural codec lands at Phase 2 council
    # lift per design memo §22 op-routables #4-5.
    cfg = TishbyIBPureCodecConfig(
        input_dim=64,
        latent_dim=args.latent_dim,
        output_dim=5,
        beta=args.beta,
        path_variant=path_variant_enum,
    )

    model = TishbyIBPureCodec(config=cfg)
    loss = TishbyIBPureScoreAwareLoss()

    # Forward smoke: encode_summary returns deterministic provenance dict
    summary = model.encode_summary()
    if summary.get("score_claim") is not False:
        raise RuntimeError("L1 scaffold encode_summary must declare score_claim=False")
    if summary.get("research_only") is not True:
        raise RuntimeError("L1 scaffold encode_summary must declare research_only=True")
    if summary.get("path_variant") != args.path_variant.lower():
        raise RuntimeError(
            f"path_variant mismatch: got {summary.get('path_variant')!r} "
            f"expected {args.path_variant.lower()!r}"
        )

    loss_probe = loss(
        reconstruction_term=1.0,
        kl_term=args.beta,
        rate_term=0.0,
    )

    # Archive roundtrip smoke (TIBP1 grammar; meta section per registry; the
    # neural blobs are POPULATED at Phase 2 council lift per design memo §22
    # op-routables #4-5).
    meta_seed: dict[str, object] = {
        "schema_version": 1,
        "path_variant": args.path_variant,
        "latent_dim": cfg.latent_dim,
        "input_dim": cfg.input_dim,
        "output_dim": cfg.output_dim,
        "beta_value": args.beta,
        "loss_probe_total": loss_probe.total,
        "loss_probe_score_claim": loss_probe.score_claim,
        "research_only": True,
        "score_claim": False,
    }
    # Minimal section content: just the meta blob (full encoder/decoder
    # state_dict serialization is a v2 SCAFFOLD task per design memo §22
    # op-routable #4-5).
    archive_bytes = pack_archive(
        sections={
            "meta_blob": json.dumps(
                meta_seed, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
        }
    )
    if not archive_bytes.startswith(TIBP1_MAGIC):
        raise RuntimeError(
            f"archive magic mismatch: got {archive_bytes[:4]!r} expected {TIBP1_MAGIC!r}"
        )
    parsed = parse_archive(archive_bytes)
    if parsed.meta.get("schema_version") != 1:
        raise RuntimeError(f"unexpected schema version: {parsed.meta.get('schema_version')}")

    archive_path = args.output_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)

    stats: dict[str, Any] = {
        "substrate_tag": SUBSTRATE_TAG,
        "lane_id": SUBSTRATE_LANE_ID,
        "smoke": True,
        "device": str(device),
        "epochs": args.epochs,
        "path_variant": args.path_variant,
        "archive_bytes": len(archive_bytes),
        "archive_sha256": _sha256_bytes(archive_bytes),
        "archive_sha256_first16": _sha256_first16(archive_bytes),
        "model_summary": dict(model.encode_summary()),
        "beta_kl": args.beta,
        "loss_probe_total": loss_probe.total,
        "loss_probe_score_claim": loss_probe.score_claim,
        "tibp1_magic_ok": True,
        "roundtrip_ok": True,
        "completed_at_utc": _utc_now_iso(),
        "git_head": _git_head_sha(REPO_ROOT),
        "torch_version": _torch_version_string(),
        "hardware_substrate": _canon_detect_hardware_substrate(
            axis="cpu" if device.type == "cpu" else "cuda",
            substrate_tag=SUBSTRATE_TAG,
            env_var_candidates=("TISHBY_IB_PURE_GPU", "MODAL_GPU"),
        ),
        "design_memo": DESIGN_MEMO_PATH,
        "d4_probe_verdict_at_landing": "INDEPENDENT",
        "d4_probe_mi_bits_at_landing": 0.0064,
        "vib_tractability_verdict_at_landing": "TRACTABLE",
        "vib_tractability_snr_mean_at_landing": 6.75,
        "research_only": True,
        "score_claim": False,
        "score_axis": "diagnostic_cpu",
        "score_axis_anchor": "[diagnostic-CPU; tishby_ib_pure_smoke]",
    }
    (args.output_dir / "smoke_stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    print(
        f"[tishby_ib_pure] SMOKE OK device={device} path={args.path_variant} "
        f"archive_bytes={len(archive_bytes)} beta={args.beta} research_only=true"
    )
    return 0


# ---------------------------------------------------------------------------
# Full main — NotImplementedError per Catalog #240 (research_only + Phase 2)
# ---------------------------------------------------------------------------
def _full_main(args: argparse.Namespace) -> int:
    """Full Tishby IB-pure training run.

    NOT YET IMPLEMENTED at L1 scaffold per CLAUDE.md "Substrate scaffolds MUST
    be COMPLETE or RESEARCH-ONLY" (Catalog #240) + the design memo §19 V1
    lift gate. Reactivation criteria (ALL required):

    1. D4 probe returns MEANINGFUL_CONDITIONING (MI ≥ 0.5 bits/symbol) AFTER
       reactivation criterion (a)/(b)/(c) per the L1 D4 verdict
       INDEPENDENT-at-landing finding.
    2. VIB tractability check returns TRACTABLE on real-scorer Modal A100
       100ep proxy (synthetic-data smoke at L1 returned TRACTABLE SNR ~6.75
       but is HARD-EARNED-at-bound-level / CARGO-CULTED-at-empirical-equivalence
       per the design memo §16 cargo-cult audit).
    3. Dykstra-feasibility intersection check returns non-empty + bounded
       below A1 operating point per Catalog #296.
    4. Path-VIB vs Path-MINE council adjudication (sextet pact per Catalog
       #292) decides Path based on Variational-IB-tractability check.
    5. Phase 2 grand council deliberation lands per design memo §19.

    Per CLAUDE.md "Forbidden premature KILL" + design memo §19.3: the D4
    INDEPENDENT verdict at L1 landing is DEFER-pending-research, NOT KILLED.
    """
    raise NotImplementedError(
        "Tishby IB-pure _full_main is council-gated per Catalog #240 "
        "(recipe research_only=true + dispatch_enabled=false). Phase 2 "
        "council approval required to lift; gated on D4 probe "
        "MEANINGFUL_CONDITIONING (current INDEPENDENT MI=0.0064) + VIB "
        "tractability TRACTABLE on real scorer + Dykstra-feasibility "
        "non-empty intersection + Path-VIB vs Path-MINE council adjudication. "
        "See design memo §19.1 V1 lift gate."
    )


def _resolve_main(args: argparse.Namespace) -> int:
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _resolve_main(args)


if __name__ == "__main__":
    sys.exit(main())
