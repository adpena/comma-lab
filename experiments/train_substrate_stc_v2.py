# SPDX-License-Identifier: MIT
"""STC clean-source v2 substrate trainer (encode-only pass, NO training loop).

Per ``.omx/research/batched_reactivation_lane17_imp_stc_apogee_int4_full_stack_design_20260516.md``
Section 2 + ``.omx/research/resurrection_audit_20260516.md`` Tier 1 #2. The
$0.20 Modal T4 CUDA disambiguator probe re-runs the STC clean-source codec
on CUDA-derived SegNet argmax masks to resolve the UNDETERMINED-pending-CUDA
verdict (the original 50x kill was MPS-PROXY per CLAUDE.md "MPS auth eval
is NOISE").

UNIQUE-AND-COMPLETE-PER-METHOD architecture: one coherent compress-only pass
binding CUDA-derived SegNet argmax + Filler & Pevny 2010 STC boundary codec
+ STC v2 archive grammar (single-file ``0.bin`` with STCB blob + Lane A
renderer.bin + Lane A poses bundled) + canonical auth-eval routing. NO
TRAINING LOOP. Closed-form codec.

Canonical-vs-unique decision per layer (per the 2026-05-16 design memo
Section 2.2.8):

| Layer                                      | Decision        | Rationale |
|--------------------------------------------|-----------------|-----------|
| Codec primitives (Filler & Pevny 2010 STC) | UNIQUE FORK     | STC is substrate-class-shift |
| Mask source (CUDA SegNet argmax)           | ADOPT canonical | the ORIGINAL kill came from NOT adopting CUDA |
| Archive grammar (STC v2 ``0.bin``)         | UNIQUE FORK     | new wire format per Catalog #146 |
| Inflate runtime                            | UNIQUE FORK     | substrate-package scaffold; submission-time uses Lane A's inflate.sh |
| Score-aware loss                           | FORK (N/A)      | STC is a codec; no gradient path |
| eval_roundtrip in loop                     | FORK (N/A)      | no training loop |
| EMA shadow weights                         | FORK (N/A)      | no learnable weights |
| autocast_fp16                              | FORK (N/A)      | no training loop; SegNet forward fp32 |
| TF32                                       | FORK (N/A)      | no neural training (canonical device_or_die still routes through trainer_skeleton, which enables TF32 for the SegNet forward as a downstream benefit) |
| torch.compile                              | FORK (N/A)      | no training loop |
| no_grad at eval                            | ADOPT canonical | scorer.extract_gt_masks wraps in torch.no_grad |
| Canonical scorer-loss helper               | FORK (N/A)      | no loss; codec is lossless |
| load_differentiable_scorers                | ADOPT canonical | compress-side SegNet query (Catalog #226 sister discipline) |
| gate_auth_eval_call (Catalog #226)         | ADOPT canonical | universal auth-eval routing |
| detect_hardware_substrate (Catalog #190)   | ADOPT canonical | phantom-score-directory protection |
| device_or_die (canonical)                  | ADOPT canonical | MPS-fallback-trap protection |
| trainer_skeleton (pin_seeds, etc.)         | ADOPT canonical | universal coordination |
| SubstrateContract decoration               | ADOPT canonical | META layer Catalog #241/#242 |
| select_inflate_device (Catalog #205)       | ADOPT canonical | substrate inflate.py uses canonical helper |

Tier 1 engineering waivers (per Catalog #270 dispatch optimization protocol):
"""
# AUTOCAST_FP16_WAIVED:no-training-loop-stc-is-a-codec
# TORCH_COMPILE_WAIVED:no-training-loop-stc-is-a-codec
# TF32_WAIVED:no-neural-training-canonical-helper-still-enables-tf32-for-segnet-forward
# SCORER_PREPROCESS_HANDLED_OK:stc-v2-uses-segnet-via-tac-scorer-extract-gt-masks-which-wraps-segnet.preprocess_input-canonical-routing-internally
# F3_CACHE_CONSUMPTION_WAIVED:no-scorer-hot-loop-single-segnet-forward-pass-over-the-gt-video
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from tac.substrate_registry import SubstrateContract, register_substrate
from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
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

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
DEFAULT_ANCHOR_ARCHIVE = (
    REPO_ROOT / "experiments" / "results" / "lane_a_landed" / "archive_lane_a.zip"
)
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"

# Lane A inflate.sh — STC v2 auth eval routes through this (per design memo
# Section 2.2.6) because Lane A's inflate already dispatches masks.stcb.
LANE_A_INFLATE_SH = REPO_ROOT / "submissions" / "robust_current" / "inflate.sh"

CONTEST_NORMALIZER = 37_545_489.0


# ---------------------------------------------------------------------------
# Catalog #152 modal-mounted-input discipline (WAVE-1 APPARATUS HARDENING
# 2026-05-16 extension; bug-class anchor STC v2 smoke fc-01KRSB76H04HM4958V2HX2JZZ4
# rc=25 / 1.6s because the Lane A anchor archive lives under
# experiments/results/lane_a_landed/ which the canonical Modal mount manifest
# IGNORES via DEFAULT_RESULTS_IGNORE=("results/**",); the file existed on the
# operator workstation so the local Catalog #152 validator passed, but the
# Modal worker never received the file and crashed at exit 25). The canonical
# fix is to declare the path in TIER_1_EXTRA_MOUNT_PATHS so the Modal mount
# manifest stages it explicitly via mount_manifest.collect_extra_mount_paths.
# ---------------------------------------------------------------------------
TIER_1_EXTRA_MOUNT_PATHS: tuple[str, ...] = (
    str(DEFAULT_ANCHOR_ARCHIVE.relative_to(REPO_ROOT)),
)


# ---------------------------------------------------------------------------
# Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS manifest
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "STC_V2_VIDEO_PATH",
        "rationale": (
            "compress-time SegNet runs on the contest video "
            "(upstream/videos/0.mkv); synthetic data FORBIDDEN per CLAUDE.md "
            "Catalog #114"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot - never regenerated locally"
        ),
        "rationale_audit": (
            ".omx/research/batched_reactivation_lane17_imp_stc_apogee_int4_full_stack_design_20260516.md"
        ),
    },
    "--anchor-archive": {
        "env": "STC_V2_ANCHOR_ARCHIVE",
        "rationale": (
            "Lane A archive (~677KB) supplies renderer.bin + optimized_poses.pt "
            "that STC v2 bundles into its swap-archive; the only varying byte "
            "channel is the masks slot (AV1 monochrome -> STCB)"
        ),
        "default": str(DEFAULT_ANCHOR_ARCHIVE.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": (
            "Lane A landed archive — pulled from experiments/results/lane_a_landed/"
        ),
    },
    "--output-dir": {
        "env": "STC_V2_OUTPUT_DIR",
        "rationale": "custody location for archive + provenance + auth-eval JSON",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "STC_V2_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for SegNet/PoseNet weights + evaluate.py + modules.py"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "STC_V2_DEVICE",
        "rationale": (
            "compute device for compress-side SegNet forward; cuda required "
            "for full run (MPS-PROXY is what the ORIGINAL kill was — the v2 "
            "smoke EXISTS to resolve the CUDA-vs-MPS disambiguator); cpu "
            "permitted only with --smoke for local validation"
        ),
        "default": "cuda",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "STC_V2_EPOCHS",
        "rationale": (
            "STC v2 has no training loop; trainer skeleton + dispatch infra "
            "expect --epochs; we accept any positive value and run ONE "
            "compress pass"
        ),
        "default": "1",
        "satisfied_by_profile": (),
        "requires": (),
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_stc_v2",
        description=(
            "STC clean-source v2 substrate compress pass (Filler & Pevny 2010 "
            "STC + STC v2 archive grammar). The $0.20 Modal T4 CUDA "
            "disambiguator probe per the 2026-05-16 batched-reactivation memo."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--anchor-archive", type=Path, default=DEFAULT_ANCHOR_ARCHIVE)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=20260516)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-auth-eval", action="store_true")
    p.add_argument("--skip-archive-build", action="store_true")
    p.add_argument(
        "--boundary-fraction",
        type=float,
        default=0.05,
        help=(
            "STC boundary detection target fraction per the design memo "
            "Section 2.2.3 default (0.05)."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="SegNet forward batch size at compress time.",
    )
    return p


def _device_or_die(name: str, *, smoke: bool):
    return _device_or_die_canonical(name, smoke=smoke, substrate_tag="stc_v2")


def _decode_gt_video(gt_video: Path):
    """Decode contest video to list of (H, W, 3) uint8 tensors via pyav.

    Inlined (not via canonical decode_real_pairs) because STC v2 needs the
    FULL frame sequence for SegNet, not pair-decoded tensors.
    """
    import av as _av
    import torch as _torch

    container = _av.open(str(gt_video))
    stream = container.streams.video[0]
    frames: list[Any] = []
    for frame in container.decode(stream):
        rgb = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
        frames.append(_torch.from_numpy(rgb))
    container.close()
    return frames


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _audit_jsonl_append(record: dict[str, Any]) -> None:
    """Append a JSONL row to the per-substrate audit log (observability surface).

    Per the 2026-05-16 design memo Section 2.2.12 the substrate emits a per-
    config record so an operator can audit byte counts + SegNet device + sha
    across multiple smoke runs. Path: ``.omx/state/lane_stc_clean_source_v2_audit.jsonl``.
    Best-effort: silently no-op if the parent dir is missing (CI envs).
    """
    try:
        target = REPO_ROOT / ".omx" / "state" / "lane_stc_clean_source_v2_audit.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Smoke main (CPU; synthetic small masks; validates STC v2 grammar)
# ---------------------------------------------------------------------------
def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU smoke validates STC v2 codec + archive grammar + inflate."""
    import numpy as np
    import torch

    from tac.substrates.stc_v2 import (
        build_stc_v2_archive_bytes,
        encode_stc_v2_masks,
        inflate_one_video,
        parse_stc_v2_archive,
    )

    _pin_seeds(args.seed)
    _ = _device_or_die(args.device, smoke=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build a tiny mask volume with class-boundary structure
    rng = np.random.default_rng(args.seed)
    h, w = 24, 32
    n = 4
    masks_np = rng.integers(0, 5, size=(n, h, w), dtype=np.int64)
    masks_np[:, :, : w // 2] = 0
    masks_np[:, :, w // 2 :] = 2
    masks = torch.from_numpy(masks_np)

    stcb_path = args.output_dir / "masks.stcb"
    encode_stc_v2_masks(masks, stcb_path, boundary_fraction=args.boundary_fraction)
    stcb_bytes = stcb_path.read_bytes()
    print(f"[stc-v2-smoke] STCB bytes: {len(stcb_bytes)}")

    bin_bytes = build_stc_v2_archive_bytes(
        stcb_blob=stcb_bytes,
        renderer_bin_blob=b"\xaa" * 1024,
        poses_pt_blob=b"\xbb" * 256,
        num_pairs=n,
    )
    (args.output_dir / "0.bin").write_bytes(bin_bytes)
    archive = parse_stc_v2_archive(bin_bytes)
    assert archive.num_pairs == n
    print(f"[stc-v2-smoke] STC v2 archive bytes: {len(bin_bytes)}")

    # Inflate roundtrip
    out_stcb = inflate_one_video(bin_bytes, args.output_dir / "inflate_smoke" / "0")
    assert out_stcb.read_bytes() == stcb_bytes
    print("[stc-v2-smoke] inflate roundtrip OK")
    print("[stc-v2-smoke] OK")
    return 0


# ---------------------------------------------------------------------------
# Full main (CUDA-required; CUDA SegNet argmax + STC encode + swap archive)
# ---------------------------------------------------------------------------
def _full_main(args: argparse.Namespace) -> int:
    """Compress-only pass: GT video -> CUDA SegNet argmax -> STC v2 archive."""
    import tempfile

    import torch

    from tac.scorer import extract_gt_masks, load_differentiable_scorers
    from tac.submission_archive import safe_extract_zip
    from tac.substrates.stc_v2 import (
        build_stc_v2_archive_bytes,
        encode_stc_v2_masks,
    )

    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_log: list[dict[str, Any]] = []
    train_started_at = time.time()

    def _stage(name: str) -> None:
        stage_log.append({"stage": name, "at": _utc_now_iso()})

    _stage("seed_pinned")

    # 1. Load Lane A anchor archive -> renderer.bin + poses + (legacy) masks.mkv
    if not args.anchor_archive.exists():
        raise FileNotFoundError(f"anchor archive missing: {args.anchor_archive}")
    anchor_bytes_total = args.anchor_archive.stat().st_size
    print(f"[stc-v2-full] anchor archive: {args.anchor_archive} ({anchor_bytes_total} B)")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        safe_extract_zip(args.anchor_archive, td_path)
        renderer_bin_path = td_path / "renderer.bin"
        if not renderer_bin_path.exists():
            raise FileNotFoundError(
                f"anchor archive missing renderer.bin: {args.anchor_archive}"
            )
        renderer_bin_blob = renderer_bin_path.read_bytes()
        # Lane A may have either name; prefer optimized_poses.pt
        poses_blob: bytes | None = None
        for poses_name in ("optimized_poses.pt", "poses.pt"):
            p = td_path / poses_name
            if p.exists():
                poses_blob = p.read_bytes()
                break
        if poses_blob is None:
            raise FileNotFoundError(
                f"anchor archive missing poses (optimized_poses.pt OR poses.pt): "
                f"{args.anchor_archive}"
            )
        legacy_masks_mkv = td_path / "masks.mkv"
        legacy_av1_bytes = (
            legacy_masks_mkv.stat().st_size if legacy_masks_mkv.exists() else -1
        )
        _stage(f"anchor_extracted_renderer_{len(renderer_bin_blob)}_poses_{len(poses_blob)}_av1_{legacy_av1_bytes}")

        # 2. Decode GT video (full frame sequence for SegNet)
        print(f"[stc-v2-full] decoding GT video {args.video_path} ...")
        gt_frames = _decode_gt_video(args.video_path)
        print(f"[stc-v2-full] decoded {len(gt_frames)} frames")
        _stage(f"gt_video_decoded_{len(gt_frames)}_frames")

        # 3. Load scorers (CUDA per CLAUDE.md "MPS auth eval is NOISE")
        print(f"[stc-v2-full] loading frozen SegNet/PoseNet on {device}")
        # load_differentiable_scorers returns (posenet, segnet) per CLAUDE.md
        # canonical scorer-loader assignment order (Catalog #222)
        posenet, segnet = load_differentiable_scorers(args.upstream_dir, device=device)
        segnet.eval()
        _stage("scorers_loaded")

        # 4. CUDA SegNet argmax — this is the substrate's distinguishing step
        print(
            f"[stc-v2-full] running CUDA SegNet argmax "
            f"(batch_size={args.batch_size})"
        )
        with torch.no_grad():
            masks = extract_gt_masks(
                gt_frames, segnet, device, batch_size=args.batch_size
            )
        del segnet, posenet
        if device.type == "cuda":
            torch.cuda.empty_cache()
        masks = masks.cpu().contiguous()
        # Number of contest pairs = num_frames // 2 (every odd frame is the pair's f1)
        n_pairs = int(masks.shape[0]) // 2
        # PV-1 observability: hash the argmax tensor
        mask_source_sha = _sha256_bytes(masks.numpy().tobytes())
        print(
            f"[stc-v2-full] argmax masks shape={tuple(masks.shape)} "
            f"sha256={mask_source_sha}"
        )
        _stage(f"segnet_argmax_done_n_pairs_{n_pairs}_sha_{mask_source_sha[:12]}")

        # 5. STC encode (Filler & Pevny 2010 boundary codec)
        stcb_path = args.output_dir / "masks.stcb"
        t0 = time.monotonic()
        stcb_size = encode_stc_v2_masks(
            masks, stcb_path, boundary_fraction=args.boundary_fraction
        )
        encode_wall = time.monotonic() - t0
        print(
            f"[stc-v2-full] STC encoded {stcb_size} B "
            f"(boundary_fraction={args.boundary_fraction}) in {encode_wall:.1f}s"
        )
        _stage(f"stc_encoded_{stcb_size}_bytes_wall_{encode_wall:.1f}s")

        # 6. Disambiguator decision (per design memo Section 2.3 decision tree)
        if stcb_size < 200_000:
            verdict = "REACTIVATED"
        elif stcb_size < 1_000_000:
            verdict = "COMPETITIVE_PIVOT_TO_AV1_PLUS_STC_RESIDUAL"
        elif stcb_size < 5_000_000:
            verdict = "RESEARCH_ONLY"
        else:
            verdict = "HARD_EARNED_FALSIFICATION"
        print(
            f"[stc-v2-full] DISAMBIGUATOR VERDICT: {verdict} "
            f"(stcb_bytes={stcb_size} vs AV1 anchor masks {legacy_av1_bytes} B)"
        )
        _audit_jsonl_append({
            "lane_id": "lane_stc_clean_source_v2_substrate_build_20260516",
            "ts_utc": _utc_now_iso(),
            "boundary_fraction": float(args.boundary_fraction),
            "stc_bytes_raw": int(stcb_size),
            "av1_bytes_baseline": int(legacy_av1_bytes),
            "encode_wall_clock_seconds": float(encode_wall),
            "mask_source_device": str(device),
            "mask_source_sha256": mask_source_sha,
            "n_pairs": int(n_pairs),
            "disambiguator_verdict": verdict,
        })

        # 7. STC v2 archive
        bin_bytes = build_stc_v2_archive_bytes(
            stcb_blob=stcb_path.read_bytes(),
            renderer_bin_blob=renderer_bin_blob,
            poses_pt_blob=poses_blob,
            num_pairs=n_pairs,
        )
        (args.output_dir / "0.bin").write_bytes(bin_bytes)
        payload_0bin_sha = _sha256_bytes(bin_bytes)
        payload_0bin_bytes = len(bin_bytes)
        print(
            f"[stc-v2-full] wrote 0.bin ({payload_0bin_bytes} B, "
            f"sha256={payload_0bin_sha})"
        )
        _stage(f"payload_0bin_built_{payload_0bin_bytes}")

        # 8. SWAP-ARCHIVE: Lane A renderer + poses + STC masks bundled as zip.
        # Auth-eval routes through Lane A's existing inflate.sh per the design
        # memo Section 2.2.6 (Lane A already supports masks.stcb dispatch).
        archive_zip_path = args.output_dir / "archive.zip"
        archive_zip_sha: str | None = None
        archive_zip_bytes: int | None = None
        if not args.skip_archive_build:
            fixed_ts = (2026, 1, 1, 0, 0, 0)
            with zipfile.ZipFile(
                archive_zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9
            ) as zout:
                for name, data in (
                    ("renderer.bin", renderer_bin_blob),
                    ("masks.stcb", stcb_path.read_bytes()),
                    ("optimized_poses.pt", poses_blob),
                ):
                    zi = zipfile.ZipInfo(name, date_time=fixed_ts)
                    zi.compress_type = zipfile.ZIP_DEFLATED
                    zout.writestr(zi, data)
            archive_zip_sha = _sha256_file(archive_zip_path)
            archive_zip_bytes = archive_zip_path.stat().st_size
            rate_delta = (
                25 * (archive_zip_bytes - anchor_bytes_total) / CONTEST_NORMALIZER
            )
            print(
                f"[stc-v2-full] wrote {archive_zip_path} "
                f"({archive_zip_bytes} B, sha256={archive_zip_sha})"
            )
            print(
                f"[stc-v2-full] rate_delta vs Lane A anchor: "
                f"{archive_zip_bytes - anchor_bytes_total:+d} B "
                f"({rate_delta:+.4f} score)"
            )

        # 9. Auth eval (Catalog #226 canonical helper). The auth-eval inflate
        # is Lane A's submissions/robust_current/inflate.sh which dispatches
        # masks.stcb natively per the legacy lane-stc-clean script.
        auth_eval_result_path: Path | None = None
        contest_cuda_score: float | None = None
        if (
            not args.skip_auth_eval
            and archive_zip_path.is_file()
            and LANE_A_INFLATE_SH.is_file()
        ):
            auth_eval_result_path = args.output_dir / "contest_auth_eval_cuda.json"
            auth_result = _canon_gate_auth_eval_call(
                args=args,
                archive_zip=archive_zip_path,
                inflate_sh=LANE_A_INFLATE_SH,
                upstream_dir=args.upstream_dir,
                output_json=auth_eval_result_path,
                contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                substrate_tag="stc_v2",
                device=device,
            )
            if auth_result is not None:
                contest_cuda_score = auth_result["auth_eval_cuda_score"]
                print(
                    f"[stc-v2-full] [contest-CUDA] score = {contest_cuda_score} "
                    f"(archive_sha256={archive_zip_sha})"
                )
            _stage("auth_eval_cuda_done")

        train_elapsed_sec = time.time() - train_started_at

        # 10. Posterior update (Catalog #128)
        if (
            contest_cuda_score is not None
            and archive_zip_sha is not None
            and archive_zip_bytes is not None
        ):
            try:
                from tac.continual_learning import (
                    ContestResult,
                    posterior_update_locked,
                )

                detected_substrate = _canon_detect_hardware_substrate(
                    axis="cuda",
                    substrate_tag="stc_v2",
                    provenance_path=args.output_dir / "provenance.json",
                    env_var_candidates=("STC_V2_GPU", "MODAL_GPU"),
                )
                result = ContestResult(
                    axis="cuda",
                    hardware_substrate=detected_substrate,
                    architecture_class=(
                        "lane_stc_clean_source_v2_substrate_build_20260516"
                    ),
                    score_value=contest_cuda_score,
                    evidence_tag="[contest-CUDA]",
                    archive_sha256=archive_zip_sha,
                    archive_bytes=archive_zip_bytes,
                    notes=(
                        f"stc_v2 cuda disambiguator first-anchor "
                        f"(stcb_bytes={stcb_size}, verdict={verdict}, "
                        f"av1_baseline={legacy_av1_bytes})"
                    ),
                    observed_at_utc=_utc_now_iso(),
                )
                update = posterior_update_locked(result)
                print(
                    f"[stc-v2-full] posterior_update: accepted={update.accepted} "
                    f"reason={update.reason!r}"
                )
            except Exception as exc:
                print(f"[stc-v2-full] posterior_update failed: {exc}", file=sys.stderr)

        # 11. Provenance
        provenance = {
            "schema": "stc_v2_provenance_v1",
            "generated_at": _utc_now_iso(),
            "from_state_hash": "regen_per_session",
            "git_head": _git_head_sha(),
            "trainer": "experiments/train_substrate_stc_v2.py",
            "lane_id": "lane_stc_clean_source_v2_substrate_build_20260516",
            "args": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            },
            "pytorch_version": _torch_version_string(),
            "device": str(device),
            "n_pairs": n_pairs,
            "stcb_bytes": int(stcb_size),
            "av1_baseline_bytes": int(legacy_av1_bytes),
            "anchor_archive_bytes": int(anchor_bytes_total),
            "archive_sha256": archive_zip_sha,
            "archive_bytes": archive_zip_bytes,
            "archive_zip_path": (
                str(archive_zip_path) if archive_zip_path.is_file() else None
            ),
            "payload_0bin_sha256": payload_0bin_sha,
            "payload_0bin_bytes": payload_0bin_bytes,
            "boundary_fraction": float(args.boundary_fraction),
            "mask_source_sha256": mask_source_sha,
            "encode_wall_clock_seconds": float(encode_wall),
            "disambiguator_verdict": verdict,
            "auth_eval_cuda_score": contest_cuda_score,
            "auth_eval_json_path": (
                str(auth_eval_result_path) if auth_eval_result_path else None
            ),
            "stage_log": stage_log,
            "custody_status": "ci-rebuildable",
            "score_claim": contest_cuda_score is not None,
            "score_axis_tag": (
                "[contest-CUDA]" if contest_cuda_score is not None else None
            ),
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "train_elapsed_sec": float(train_elapsed_sec),
        }
        (args.output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        return 0


# ---------------------------------------------------------------------------
# META layer SubstrateContract (Catalog #241/#242)
# ---------------------------------------------------------------------------

STC_V2_SUBSTRATE_CONTRACT = SubstrateContract(
    id="stc_v2",
    lane_id="lane_stc_clean_source_v2_substrate_build_20260516",
    target_modes=("contest_one_video_replay", "research_substrate"),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/batched_reactivation_lane17_imp_stc_apogee_int4_full_stack_design_20260516.md"
    ),
    archive_grammar=(
        "STC v2 monolithic single-file 0.bin: 38-byte header (magic=STC2, "
        "version=1, output_height=874, output_width=1164, num_pairs, "
        "stcb_len, renderer_bin_len, poses_pt_len) + concatenated payload "
        "(STCB blob + Lane A renderer.bin + Lane A optimized_poses.pt). "
        "NO scorer weights. NO PyTorch at inflate. Per CLAUDE.md HNeRV "
        "parity discipline lesson 3 (monolithic single-file with fixed "
        "offsets declared in source) + Filler & Pevny 2010 STC encoding "
        "of CUDA SegNet argmax masks."
    ),
    parser_section_manifest={
        "header": "STC2_v1_magic_version_output_dimensions_num_pairs_lengths",
        "stcb_blob": "stcb_v1_arith_coded_boundary_gaps_classes_exception_streams",
        "renderer_bin_blob": "lane_a_fp4_brotli_renderer_weights",
        "poses_pt_blob": "lane_a_optimized_poses_pt_torch_pickle",
    },
    inflate_runtime_loc_budget=200,  # substrate inflate is 100 LOC; budget 200 for headroom
    runtime_dep_closure=("numpy>=1.24", "torch>=2.5"),
    export_format="custom",
    score_aware_loss="custom",
    bolt_on_loc_budget=900,  # trainer is ~500 LOC; substrate_engineering exception
    no_op_detector_planned=True,
    archive_bytes_added=(
        "Predicted band: PVP varies by SegNet device. MPS-PROXY anchor was "
        "~21 MB (50.5x AV1 421 KB). CUDA disambiguator question: does CUDA "
        "argmax structure reduce STCB to <1 MB? Per design memo Section 2.3 "
        "decision tree: <200 KB REACTIVATED / 200 KB-1 MB COMPETITIVE / "
        "1 MB-5 MB RESEARCH_ONLY / >5 MB HARD-EARNED FALSIFICATION."
    ),
    score_improvement_mechanism_status="RESEARCH_ONLY",
    runtime_overlay_consumed=False,
    recipe_smoke_only=False,
    recipe_research_only=True,  # UNDETERMINED-pending-CUDA until smoke verdict
    recipe_min_smoke_gpu="T4",
    recipe_min_vram_gb=8,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="independent_substrate",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency=None,
    cost_band_epochs=1,
    cost_band_gpu_key="T4",
    cost_band_platform_key="modal",
    cost_band_p50_usd=0.20,  # per design memo Section 2.2.11 cheapest possible
    hook_sensitivity_contribution="not_applicable_with_rationale",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="not_applicable_with_rationale",
    hook_autopilot_ranker_class_shift_token="stc_clean_source_v2",
    hook_continual_learning_anchor_kind="cuda_only",
    hook_probe_disambiguator=None,
    catalog_compliance_declarations=(
        "catalog_124_archive_grammar_8_fields_declared",
        "catalog_146_3arg_inflate_sh_contract",
        "catalog_151_tier1_required_flags_declared",
        "catalog_163_remote_lane_sentinel_used",
        "catalog_164_scorer_preprocess_input_called",
        "catalog_167_smoke_before_full_routed",
        "catalog_170_min_vram_gb_declared",
        "catalog_171_video_input_strategy_declared",
        "catalog_173_canary_status_declared",
        "catalog_181_pyav_decode_strategy_declared",
        "catalog_182_target_modes_declared",
        "catalog_205_select_inflate_device_used",
        "catalog_215_min_smoke_gpu_consistent",
        "catalog_220_operational_mechanism_declared",
        "catalog_226_gate_auth_eval_call_used",
        "catalog_244_modal_nvml_env_block_auto_emitted",
        "catalog_290_canonical_vs_unique_decision_per_layer_documented",
    ),
    hook_not_applicable_rationale={
        "hook_sensitivity_contribution": (
            "STC v2 is a lossless codec; per-byte sensitivity is captured by "
            "hook_pareto_constraint=rate_distortion_v1 (the per-frame boundary "
            "fraction IS the sensitivity knob)"
        ),
        "hook_bit_allocator_class": (
            "the Filler & Pevny 2010 syndrome-trellis arithmetic-coded "
            "boundary/exception streams ARE the bit allocator (closed-form "
            "per per-frame mode + arith CDF); no per-tensor uniform / "
            "per-channel-lsq / ibps_kkt allocator applies"
        ),
        "hook_probe_disambiguator": (
            "the substrate IS its own probe-disambiguator: stcb_size vs the "
            "design-memo decision tree (200 KB / 1 MB / 5 MB thresholds) "
            "resolves the UNDETERMINED-pending-CUDA verdict on a single "
            "$0.20 smoke run"
        ),
    },
)


@register_substrate(STC_V2_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
