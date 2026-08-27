#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Shadow-vs-live byte-close comparator (ARM-C; SPEC_v10 §13.5 terminal measurement gate).

Given a PER-STAGE resume checkpoint (``levelset_resume_stage*.npz`` — carries BOTH the EMA
shadow ``emaP__*`` and the live weights ``liveP__*``) plus its paired deploy stage checkpoint
(``levelset_ckpt_stage*.npz`` — the byte-close cfg custody template, EMA weights + ``__cfg_*``
/ ``__bank_*`` / ``__render_hw`` block), this tool:

  1. MATERIALIZES two deploy-format npz files into ``--out-dir``:
       ``levelset_witness_ema_mlx.npz``  (template cfg + emaP weights)
       ``levelset_witness_live_mlx.npz`` (template cfg + liveP weights)
     — SAME cfg custody, ONLY the weight values differ.
  2. VERIFIES NO ALIASING (NO-FAKE): the ema/live param key sets match the template's; the two
     weight sets genuinely DIFFER (max-abs delta > 0); the template's stored weights match the
     resume npz's EMA shadow (they are the same object saved twice); the two materialized npz
     files have different sha256.
  3. Byte-closes + realized-scores BOTH arms through the REAL decode harness
     (``tools/levelset_byte_close_and_eval.select_best_weights_arm`` -> the shipped inflate +
     numpy-fp32 oracle + frozen CPU-torch scorer parity) and emits PAIRED d_seg/d_pose/rate/S
     rows + the ranked selection.

AUTHORITY: ``[macOS advisory] NON-PROMOTABLE`` — realized parity on inflated frames is the
gate; only ``upstream/evaluate.py`` on contest hardware is a score. ``score_claim=false``,
``promotable=false`` are stamped in the report. This is the empirical decider for the
SPEC_v10 §13.3 EMA-calibration question ("ship the winner").

Usage (smoke, n<=96 pairs):
    .venv/bin/python tools/compare_shadow_vs_live_byte_close.py \
        --resume-npz experiments/results/<run>/levelset_resume_stageTau_muon_ep1000.npz \
        --stage-ckpt experiments/results/<run>/levelset_ckpt_stageTau_muon_ep1000.npz \
        --out-dir experiments/results/shadow_vs_live_<tag> \
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n96.npz --max-pairs 96

``--stage-ckpt`` defaults to the sibling ``levelset_ckpt_<same stage tag>.npz``. A run dir
that ALREADY holds both deploy npz files can skip materialization via ``--ckpt-dir``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "tools"))

_AXIS = "[macOS advisory] NON-PROMOTABLE"

EMA_PREFIX = "emaP__"
LIVE_PREFIX = "liveP__"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True,  # subprocess-no-check-OK: git-sha provenance capture; except arm records 'unknown'
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def extract_arm_params(resume_npz: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """(ema_params, live_params, meta) from a per-stage resume npz. Fail-closed on absence."""
    z = np.load(resume_npz, allow_pickle=False)
    ema: dict[str, np.ndarray] = {}
    live: dict[str, np.ndarray] = {}
    for k in z.files:
        if k.startswith(EMA_PREFIX):
            ema[k[len(EMA_PREFIX):]] = np.asarray(z[k], dtype=np.float32)
        elif k.startswith(LIVE_PREFIX):
            live[k[len(LIVE_PREFIX):]] = np.asarray(z[k], dtype=np.float32)
    if not ema or not live:
        raise SystemExit(
            f"{resume_npz} lacks {EMA_PREFIX}*/{LIVE_PREFIX}* keys — not a per-stage resume "
            "checkpoint (NO-FAKE: refusing to fabricate an arm).")
    if set(ema) != set(live):
        raise SystemExit(
            f"EMA/live param key sets differ in {resume_npz}: "
            f"only-ema={sorted(set(ema) - set(live))[:5]} only-live={sorted(set(live) - set(ema))[:5]}")
    meta = {"resume_epoch": int(z["__resume_epoch"]) if "__resume_epoch" in z.files else None}
    return ema, live, meta


def materialize_arm_npz(template_ckpt: Path, params: dict[str, np.ndarray], out_path: Path) -> dict:
    """Write a deploy-format npz: the template's cfg block + the given weight values.

    The template's param keys define the deploy surface; every template param MUST be present
    in ``params`` (same architecture) — fail-closed otherwise."""
    t = np.load(template_ckpt, allow_pickle=False)
    out: dict[str, np.ndarray] = {}
    missing = []
    for k in t.files:
        if k.startswith("__"):
            out[k] = t[k]                      # cfg custody: byte-identical to the template
        elif k in params:
            out[k] = params[k]
        else:
            missing.append(k)
    if missing:
        raise SystemExit(
            f"template {template_ckpt} has params absent from the resume npz arms: {missing[:8]} "
            "(architecture mismatch — refusing to mix weight sets).")
    extra = [k for k in params if k not in set(t.files)]
    if extra:
        raise SystemExit(
            f"resume npz carries params the template lacks: {extra[:8]} (architecture mismatch).")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    return {"path": str(out_path), "sha256": _sha256(out_path), "n_params": len(params)}


def anti_aliasing_report(ema: dict, live: dict, template_ckpt: Path) -> dict:
    """Prove the two arms are genuinely distinct weight sets (and the template == EMA)."""
    max_delta = 0.0
    for k in ema:
        d = float(np.max(np.abs(ema[k] - live[k]))) if ema[k].size else 0.0
        max_delta = max(max_delta, d)
    t = np.load(template_ckpt, allow_pickle=False)
    tmpl_vs_ema = 0.0
    for k in t.files:
        if k.startswith("__") or k not in ema:
            continue
        d = float(np.max(np.abs(np.asarray(t[k], np.float32) - ema[k])))
        tmpl_vs_ema = max(tmpl_vs_ema, d)
    return {
        "ema_vs_live_max_abs_delta": max_delta,
        "arms_distinct": bool(max_delta > 0.0),
        "template_vs_ema_max_abs_delta": tmpl_vs_ema,
        "template_is_ema_shadow": bool(tmpl_vs_ema == 0.0),
        "note": ("arms_distinct MUST be true (else the comparator would score one weight set "
                 "twice — the aliasing fake this block refuses); template_is_ema_shadow is "
                 "expected true (the deploy stage ckpt IS the EMA shadow), a nonzero delta is "
                 "LOUD but non-fatal (e.g. a later intra-stage save)"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--resume-npz", type=Path,
                     help="per-stage resume npz carrying emaP__*/liveP__* weight sets")
    src.add_argument("--ckpt-dir", type=Path,
                     help="run dir ALREADY holding levelset_witness_{ema,live}_mlx.npz (skip "
                          "materialization)")
    ap.add_argument("--stage-ckpt", type=Path, default=None,
                    help="deploy-format stage ckpt (cfg custody template); default: the sibling "
                         "levelset_ckpt_<stage>.npz inferred from --resume-npz")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="output dir for the materialized arm npz files + the paired report "
                         "(durable, NOT /tmp)")
    ap.add_argument("--gt-cache", type=str, required=True,
                    help="GT npz for realized parity (e.g. experiments/results/mlx_fleet_gt_cache/"
                         "gt_n96.npz)")
    ap.add_argument("--max-pairs", type=int, default=96,
                    help="cap inflate+parity pairs (default 96 = the smoke scope; pass 600 for "
                         "the full verdict)")
    ap.add_argument("--keep-packet", action="store_true")
    # self-orient decode overrides (NOT persisted by the trainer; defaults == trainer defaults,
    # mirroring the byte-close tool's own --so-* flags).
    ap.add_argument("--so-freq-across", type=float, default=32.0)
    ap.add_argument("--so-freq-along", type=float, default=4.0)
    ap.add_argument("--so-tau", type=float, default=4.0)
    ap.add_argument("--so-iters", type=int, default=4)
    args = ap.parse_args(argv)

    if "/tmp/" in str(args.out_dir) or str(args.out_dir).startswith("/tmp"):
        raise SystemExit("--out-dir must be durable (never /tmp) per CLAUDE.md")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    materialized: dict = {}
    aliasing: dict = {}
    meta: dict = {}
    if args.resume_npz is not None:
        resume_npz = args.resume_npz.resolve()
        stage_ckpt = args.stage_ckpt
        if stage_ckpt is None:
            inferred = resume_npz.name.replace("levelset_resume_", "levelset_ckpt_")
            stage_ckpt = resume_npz.parent / inferred
        if not stage_ckpt.exists():
            raise SystemExit(f"stage ckpt template {stage_ckpt} not found (pass --stage-ckpt)")
        ema, live, meta = extract_arm_params(resume_npz)
        aliasing = anti_aliasing_report(ema, live, stage_ckpt)
        if not aliasing["arms_distinct"]:
            raise SystemExit(
                "EMA and live weight sets are IDENTICAL (max-abs delta 0) — refusing to emit a "
                "fake two-arm comparison (aliasing guard).")
        materialized = {
            "ema": materialize_arm_npz(stage_ckpt, ema, out_dir / "levelset_witness_ema_mlx.npz"),
            "live": materialize_arm_npz(stage_ckpt, live, out_dir / "levelset_witness_live_mlx.npz"),
            "template": {"path": str(stage_ckpt), "sha256": _sha256(stage_ckpt)},
            "resume_npz": {"path": str(resume_npz), "sha256": _sha256(resume_npz)},
        }
        if materialized["ema"]["sha256"] == materialized["live"]["sha256"]:
            raise SystemExit("materialized arm npz files are byte-identical — aliasing guard fired")
        ckpt_dir = out_dir
    else:
        ckpt_dir = args.ckpt_dir.resolve()
        for fn in ("levelset_witness_ema_mlx.npz", "levelset_witness_live_mlx.npz"):
            if not (ckpt_dir / fn).exists():
                raise SystemExit(f"--ckpt-dir mode needs {fn} present in {ckpt_dir}")
        s_ema = _sha256(ckpt_dir / "levelset_witness_ema_mlx.npz")
        s_live = _sha256(ckpt_dir / "levelset_witness_live_mlx.npz")
        if s_ema == s_live:
            raise SystemExit("ema and live npz are byte-identical — aliasing guard fired")
        aliasing = {"arms_distinct": True, "ema_sha256": s_ema, "live_sha256": s_live,
                    "note": "pre-existing deploy npz pair (byte-distinct verified)"}

    # ---- byte-close + realized-score BOTH arms through the real decode harness -------------
    import levelset_byte_close_and_eval as bce  # tools/ import (sys.path above)

    winner_report, arm_reports = bce.select_best_weights_arm(
        ckpt_dir, arms=["ema", "live"],
        max_pairs=int(args.max_pairs),
        fold_pose_sidecar=False, pose_sidecar_path=None,
        gt_cache=str(args.gt_cache), keep_packet=bool(args.keep_packet), packet_dir=None,
        skip_parity=False,
        so_overrides={"freq_across": args.so_freq_across, "freq_along": args.so_freq_along,
                      "tau": args.so_tau, "iters": args.so_iters},
    )

    paired_rows = {
        arm: {
            "weights_arm": arm,
            **{k: winner_report["arm_selection"]["per_arm"][arm].get(k)
               for k in winner_report["arm_selection"]["per_arm"][arm]},
            "axis": _AXIS, "score_claim": False, "promotable": False,
        } for arm in winner_report["arm_selection"]["per_arm"]
    }
    report = {
        "schema": "shadow_vs_live_byte_close.v1",
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        "axis": _AXIS,
        "score_claim": False,
        "promotable": False,
        "verdict_scope": ("paired byte-closed realized rows on the SAME cfg custody; "
                         f"max_pairs={int(args.max_pairs)} "
                         f"({'SMOKE subset — NOT n600 evidence' if int(args.max_pairs) < 600 else 'n600'})"),
        "resume_meta": meta,
        "materialized": materialized,
        "anti_aliasing": aliasing,
        "paired_rows": paired_rows,
        "arm_selection": winner_report["arm_selection"],
        "note": ("SPEC_v10 §13.5 shadow-vs-live gate: decides the §13.3 EMA-calibration question "
                 "empirically (ship the winner). Advisory realized parity; the exact-eval row "
                 "(upstream/evaluate.py, contest CPU/CUDA) remains the ONLY score authority."),
    }
    rpt_path = out_dir / "shadow_vs_live_report.json"
    rpt_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"report": str(rpt_path),
                      "winner": winner_report["arm_selection"]["winner"],
                      "per_arm": {a: {kk: vv for kk, vv in r.items()
                                      if kk in ("d_seg_realized_on_inflated",
                                                "d_pose_realized_on_inflated",
                                                "implied_S_advisory", "archive_zip_bytes",
                                                "pairs_scored")}
                                  for a, r in winner_report["arm_selection"]["per_arm"].items()},
                      "axis": _AXIS}, default=str), flush=True)
    # full per-arm reports alongside (queryable post-hoc; max-observability)
    for arm, rep in arm_reports.items():
        (out_dir / f"byte_close_report_{arm}.json").write_text(json.dumps(rep, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
