#!/usr/bin/env python3
"""r3_ep250_authority_trace.v1 — localize the FIRST surface where a HiNeRV
checkpoint becomes evaluator-bad (operator directive 2026-06-09).

THE QUESTION (after B1-R3 ep250 scored d_seg=0.505 / d_pose=151.5 on the int8
archive surface): is the model GENUINELY bad (alive latents -> non-fixed-but-wrong
frames), or did the export/inflate BRIDGE corrupt a good model? The proxy showed
pose "improving" while exact d_pose exploded -- that contradiction MUST be
localized before any carrier fork or more training.

THE METHOD (controlled ablation): re-export the SAME ema_shadow checkpoint state
through the SAME MLX->numpy export + SAME inflate + SAME evaluate.py, changing ONLY
the decoder codec int8_mixed -> fp16_brotli_legacy. The single changed variable
isolates int8-quantization damage:

  Surface 1  int8_mixed archive  -> inflate -> evaluate.py   (= the B2 ep250 result)
  Surface 2  fp16 archive        -> inflate -> evaluate.py   (this tool builds it)

Verdict routing:
  fp16 GOOD (d_seg<<0.5, d_pose sane)         -> int8 quant IS the bug; fix the codec
                                                 (QAT / per-channel int8 / keep fp16).
  fp16 ALSO bad (~int8)                       -> quant EXONERATED; the fault is the
                                                 MLX model itself OR MLX->numpy export
                                                 parity. Next surface = live-MLX score
                                                 (or a fp32-numpy parity probe), then
                                                 carrier fork only if live-MLX confirms
                                                 the model is bad.

REUSES (no reimplementation -> no new apples-to-apples bug): the harvester's
``export_backend_only_archive_from_checkpoint`` and the B2 bridge
``tools/run_hi_nerv_backend_only_b2_exact_eval.py`` (which itself reuses
experiments/contest_auth_eval.py -> upstream/evaluate.py). The ONLY new code here
is orchestration + the localization verdict.

[macOS-CPU advisory] -- NOT a contest score (promotion/score-claim stay False).

Run detached (the fp16 inflate+evaluate is ~10 min, foreground tools die at
SIGURG-144):
  nohup .venv/bin/python tools/run_hi_nerv_authority_trace.py \\
      --checkpoint-meta <run>/checkpoints/epoch000249_*.meta.json \\
      --int8-eval-json <run>/hi_nerv_backend_only_ep250_exact_eval.json \\
      --out-dir <run> --tag r3_ep250 </dev/null >/dev/null 2>&1 & disown
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tools"))

from watch_and_harvest_b1_checkpoint import (  # noqa: E402
    _checkpoint_ref_from_meta,
    _load_checkpoint_meta,
    export_backend_only_archive_from_checkpoint,
)

_B2_BRIDGE = _REPO_ROOT / "tools" / "run_hi_nerv_backend_only_b2_exact_eval.py"
# "good" thresholds (well below the 0.50 flat-collapse seg + sane pose).
_SEG_GOOD = 0.30
_POSE_SANE = 1.0
_FP16_HARD_BYTE_CEILING = 2_000_000  # fp16 is ~2x int8; rate is irrelevant for the trace.


def _eval_surface_from_json(path: Path) -> dict[str, Any]:
    """Extract (d_seg, d_pose, bytes, score, sha) from a B2/contest_auth_eval JSON."""
    ev = json.loads(path.read_text())
    b2r = ev.get("b2", {}).get("b2_result", {}) if isinstance(ev.get("b2"), dict) else {}
    export = ev.get("export", {}) if isinstance(ev.get("export"), dict) else {}

    def _first(*cs: Any) -> Any:
        for c in cs:
            if c is not None:
                return c
        return None

    d_seg = _first(ev.get("avg_segnet_dist"), b2r.get("avg_segnet_dist"))
    d_pose = _first(ev.get("avg_posenet_dist"), b2r.get("avg_posenet_dist"))
    bytes_ = _first(export.get("archive_bytes"), b2r.get("archive_size_bytes"), ev.get("archive_bytes"))
    score = _first(ev.get("final_score"), b2r.get("canonical_score"), ev.get("canonical_score"))
    sha = _first(export.get("archive_sha256"), b2r.get("archive_sha256"), ev.get("archive_sha256"))
    if d_seg is None or d_pose is None:
        raise ValueError(f"{path}: missing avg_segnet_dist/avg_posenet_dist")
    return {
        "d_seg": float(d_seg),
        "d_pose": float(d_pose),
        "archive_bytes": int(bytes_) if bytes_ is not None else None,
        "score": float(score) if score is not None else None,
        "archive_sha256": str(sha) if sha is not None else None,
    }


def _localize(int8: dict[str, Any], fp16: dict[str, Any]) -> dict[str, Any]:
    """Decide whether int8 quantization is the first-bad surface."""
    fp16_seg_good = fp16["d_seg"] < _SEG_GOOD
    fp16_pose_good = fp16["d_pose"] < _POSE_SANE
    fp16_good = fp16_seg_good and fp16_pose_good
    # "materially better" = fp16 cuts the bad axis by >=2x (partial quant fault).
    seg_improved = fp16["d_seg"] <= int8["d_seg"] * 0.5
    pose_improved = fp16["d_pose"] <= int8["d_pose"] * 0.5

    if fp16_good:
        first_bad_surface = "int8_quantization"
        verdict = "INT8_QUANT_IS_THE_BUG"
        route = "adaptive_waterfilling_quantization"
        reason = (
            "fp16 re-export of the SAME checkpoint scores good "
            f"(d_seg={fp16['d_seg']:.4f}<{_SEG_GOOD}, d_pose={fp16['d_pose']:.3f}<{_POSE_SANE}) "
            f"while int8 scored d_seg={int8['d_seg']:.4f}/d_pose={int8['d_pose']:.3f}. "
            "Do NOT retrain. fp16 is only the CONTROL (proves quant is the culprit); the "
            "OPTIMAL fix is ADAPTIVE / sensitivity-weighted quantization (waterfill codec "
            "bits by evaluator sensitivity: high precision on the pose-Y + seg-boundary "
            "tensors per the B2 atlas, low/int8 elsewhere -- PR95 L21-L32 + QAT). Uniform "
            "fp16 is the rate-expensive fallback, not the target."
        )
    elif seg_improved or pose_improved:
        first_bad_surface = "int8_quantization_partial"
        verdict = "INT8_QUANT_PARTIAL_PLUS_UPSTREAM"
        route = "fix_decoder_codec_then_carrier"
        reason = (
            f"fp16 materially improves one axis (seg {int8['d_seg']:.4f}->{fp16['d_seg']:.4f}, "
            f"pose {int8['d_pose']:.3f}->{fp16['d_pose']:.3f}) but is still not 'good'. "
            "int8 quant is PART of the fault; an upstream (model/parity) fault remains."
        )
    else:
        first_bad_surface = "upstream_model_or_mlx_numpy_parity"
        verdict = "QUANT_EXONERATED_MODEL_OR_PARITY"
        route = "live_mlx_surface_then_fork_carrier"
        reason = (
            f"fp16 ~ int8 (seg {fp16['d_seg']:.4f}~{int8['d_seg']:.4f}, "
            f"pose {fp16['d_pose']:.3f}~{int8['d_pose']:.3f}). Quantization EXONERATED. "
            "Next: score the live-MLX render (or fp32-numpy parity probe). If live-MLX "
            "is ALSO bad -> the model/carrier failed (fork carrier); if good -> MLX->numpy "
            "export parity bug."
        )
    return {
        "first_bad_surface": first_bad_surface,
        "verdict": verdict,
        "route": route,
        "reason": reason,
        "fp16_seg_good": fp16_seg_good,
        "fp16_pose_good": fp16_pose_good,
        "seg_thresholds": {"good_below": _SEG_GOOD, "sane_pose_below": _POSE_SANE},
        "auto_kill": False,  # never auto-kill (Forbidden premature KILL).
    }


def run_authority_trace(
    *,
    checkpoint_meta: Path,
    int8_eval_json: Path,
    out_dir: Path,
    tag: str,
    device: str = "cpu",
    b2_bridge: Path = _B2_BRIDGE,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Surface 1: int8 archive (the B2 ep250 result; already on disk).
    int8 = _eval_surface_from_json(int8_eval_json)

    # Surface 2: fp16 re-export of the SAME checkpoint -> SAME bridge.
    meta = _load_checkpoint_meta(checkpoint_meta)
    if meta is None:
        raise ValueError(f"could not load checkpoint meta {checkpoint_meta}")
    ref = _checkpoint_ref_from_meta(checkpoint_meta, meta)
    if ref is None:
        raise ValueError(f"could not resolve CheckpointRef from {checkpoint_meta}")

    fp16_export_dir = out_dir / f"authority_trace_fp16_export_ep{ref.global_epoch}"
    export = export_backend_only_archive_from_checkpoint(
        ref,
        fp16_export_dir,
        decoder_codec="fp16_brotli_legacy",
        hard_byte_ceiling=_FP16_HARD_BYTE_CEILING,
        use_ema=True,
    )

    # NOTE: the B2 bridge's result-row flag is `--out-row` (NOT `--json-out`; the
    # latter is the flag the bridge passes to the INTERNAL contest_auth_eval call,
    # not its own argparse). Verified against the bridge's argparse usage.
    fp16_eval_json = out_dir / f"authority_trace_fp16_eval_ep{ref.global_epoch}.json"
    cmd = [
        sys.executable,
        str(b2_bridge),
        "--archive",
        str(export.archive_path),
        "--device",
        device,
        "--out-row",
        str(fp16_eval_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT))
    fp16: dict[str, Any]
    if proc.returncode == 0 and fp16_eval_json.is_file():
        fp16 = _eval_surface_from_json(fp16_eval_json)
        fp16["archive_bytes"] = export.archive_bytes
        fp16["archive_sha256"] = export.archive_sha256
        bridge_ok = True
    else:
        # Bridge mechanically failed on the fp16 archive -> that itself is the
        # first-bad surface (export/inflate/eval bug, not the model).
        fp16 = {"d_seg": None, "d_pose": None, "archive_bytes": export.archive_bytes,
                "score": None, "archive_sha256": export.archive_sha256,
                "bridge_stderr_tail": proc.stderr[-2000:]}
        bridge_ok = False

    if bridge_ok:
        localization = _localize(int8, fp16)
    else:
        localization = {
            "first_bad_surface": "fp16_inflate_or_evaluate_bridge",
            "verdict": "FP16_BRIDGE_FAILED",
            "route": "patch_b2_bridge",
            "reason": "fp16 archive failed inflate/evaluate mechanically; patch the bridge before any model verdict.",
            "auto_kill": False,
        }

    trace = {
        "schema": "r3_ep250_authority_trace.v1",
        "tag": tag,
        "axis_tag": "[macOS-CPU advisory]",
        "authoritative": False,
        "promotion_eligible": False,
        "score_claim": False,
        "checkpoint": {
            "meta_path": str(checkpoint_meta),
            "global_epoch": ref.global_epoch,
            "role": ref.role,
            "ema_state_path": str(ref.ema_state_path),
            "selection_metric_value": ref.selection_metric_value,
        },
        "surfaces": {
            "s1_int8_archive": {**int8, "codec": "int8_mixed", "source": str(int8_eval_json)},
            "s2_fp16_archive": {**fp16, "codec": "fp16_brotli_legacy", "source": str(fp16_eval_json)},
        },
        "controlled_variable": "decoder_codec (int8_mixed -> fp16_brotli_legacy); ALL else identical",
        "localization": localization,
    }
    trace_path = out_dir / f"{tag}_authority_trace.v1.json"
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    trace["trace_path"] = str(trace_path)
    return trace


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-meta", required=True, type=Path)
    p.add_argument("--int8-eval-json", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--tag", default="r3_ep250")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--b2-bridge", type=Path, default=_B2_BRIDGE)
    args = p.parse_args(argv)

    trace = run_authority_trace(
        checkpoint_meta=args.checkpoint_meta,
        int8_eval_json=args.int8_eval_json,
        out_dir=args.out_dir,
        tag=args.tag,
        device=args.device,
        b2_bridge=args.b2_bridge,
    )
    s1 = trace["surfaces"]["s1_int8_archive"]
    s2 = trace["surfaces"]["s2_fp16_archive"]
    loc = trace["localization"]
    print("=== r3 authority trace: int8 vs fp16 (same checkpoint, same bridge) ===")
    print(f"  S1 int8  d_seg={s1['d_seg']:.4f} d_pose={s1['d_pose']:.3f} bytes={s1.get('archive_bytes')}")
    if s2.get("d_seg") is not None:
        print(f"  S2 fp16  d_seg={s2['d_seg']:.4f} d_pose={s2['d_pose']:.3f} bytes={s2.get('archive_bytes')}")
    else:
        print(f"  S2 fp16  BRIDGE FAILED (bytes={s2.get('archive_bytes')})")
    print(f"  FIRST-BAD SURFACE: {loc['first_bad_surface']}")
    print(f"  VERDICT: {loc['verdict']}  ROUTE: {loc['route']}")
    print(f"  {loc['reason']}")
    print(f"  wrote: {trace['trace_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
