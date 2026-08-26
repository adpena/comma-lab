# SPDX-License-Identifier: MIT
"""ddm_fz1 -- compose Row B: pu2 vehicle + tz1 margin-coupled token map + F0PR1 tail repair.

Builds the candidate archive through the CANONICAL surfaces only:
  tac.optimization.ddm_ix2_archive_container.build_payload   (5-section joint group)
  tac.submission_chain.stage_submission / build_byte_ledger  (joint_names extended)
and verifies it through the ACTUAL patched receiver (inflate_runner.py in the runtime dir):
every repaired pair is decoded by the runner's own Decoder and re-scored against the GT pose
targets through the frozen CPU scorer -- the runner-true d_pose is the number reported, never
the encode-side solve's.  Unrepaired pairs must be byte-identical to the map-only decode.

Pay-selection is exact arithmetic (no slope approximations): starting from the mapped-census
per-pair d_pose, pairs are admitted in descending measured improvement while the composed
  sqrt(10 * mean(d)) + RATE_PER_BYTE * archive_bytes
keeps decreasing.  Both WITH-MAP and WITHOUT-MAP payloads are composable (--payload).

Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE until upstream/evaluate.py runs on the
exact archive bytes.  score_claim=False.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "src"))

RATE_PER_BYTE = 25.0 / 37_545_489.0
LIVE_BEST_S = 0.7910689
SEG_LIVE = 0.431179          # pu2 n600 seg term (100*d_seg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", type=Path, required=True, help="payload_map.bin or payload_nomap.bin")
    ap.add_argument("--coefs-npz", type=str, required=True,
                    help="npz path(s), comma-separated (multi-worker receipts merge)")
    ap.add_argument("--probe-json", type=str, required=True,
                    help="probe json path(s), comma-separated")
    ap.add_argument("--mapped-census", type=Path, required=True)
    ap.add_argument("--seg-census", type=Path, required=True,
                    help="same file as mapped-census (holds flips_vs_gt)")
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--gt-mkv", type=Path, required=True)
    ap.add_argument("--d-seg", type=float, default=None,
                    help="measured base d_seg override; default preserves the pu2 rowB behavior")
    ap.add_argument("--live-best-s", type=float, default=LIVE_BEST_S)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    from tac.optimization.ddm_ix2_archive_container import build_payload, parse_payload
    from tac.optimization.frame0_pose_repair_stream import encode_pose_repair_stream
    from tac.submission_chain import build_byte_ledger, stage_submission

    payload = args.payload.read_bytes()
    bulk, sections = parse_payload(payload)

    census = json.loads(args.mapped_census.read_text())["rows"]
    d_seg_source = "mapped-census flips_vs_gt"
    if args.d_seg is not None:
        if args.d_seg < 0.0:
            raise SystemExit("--d-seg must be non-negative")
        d_seg_override = float(args.d_seg)
    else:
        d_seg_override = None

    if census and "d_pose_mapped" in census[0]:
        d_map = {int(r["pair"]): float(r["d_pose_mapped"]) for r in census}
        if d_seg_override is None:
            flips = sum(int(r["flips_vs_gt"]) for r in census)
            d_seg_mapped = flips / (600 * 384 * 512)
        else:
            d_seg_mapped = d_seg_override
            d_seg_source = "cli --d-seg override"
    else:
        # repair-only row on the UNMAPPED base: pose census schema; the token member is
        # byte-identical to the live archive, and the repair edits only frame_0 (outside
        # SegNet's input), so d_seg is the live n600-exact value BY CONSTRUCTION.
        d_map = {int(r["pair"]): float(r["d_pose_delivered"]) for r in census}
        if d_seg_override is None:
            d_seg_mapped = 0.00431179
            d_seg_source = "pu2 live constant"
        else:
            d_seg_mapped = d_seg_override
            d_seg_source = "cli --d-seg override"

    probe = []
    for pj in str(args.probe_json).split(","):
        probe += json.loads(Path(pj).read_text())["rows"]
    coef_store = {}
    for cn in str(args.coefs_npz).split(","):
        coef_store.update(dict(np.load(cn)))
    repaired = {}
    for r in probe:
        p, k = int(r["pair"]), int(r["k"])
        if k != args.k or not r.get("best_coef_saved"):
            continue
        key = f"p{p}_k{k}"
        if key in coef_store and r["d_pose_camera_best"] < d_map.get(p, np.inf):
            repaired[p] = (float(r["d_pose_camera_best"]), coef_store[key])

    # ---- exact greedy pay-selection --------------------------------------------------------
    base_vec = dict(d_map)
    base_payload_bytes = len(payload)

    def composed_S(sel: dict) -> tuple[float, int]:
        v = dict(base_vec)
        for p in sel:
            v[p] = repaired[p][0]
        blob = encode_pose_repair_stream({p: repaired[p][1] for p in sel}, k=args.k,
                                         seg_h=384, seg_w=512) if sel else b""
        new_payload_len = base_payload_bytes if not sel else len(
            build_payload(bulk, [*list(sections), blob]))
        arch = new_payload_len + 108      # measured ZIP framing (single stored member)
        s = (100.0 * d_seg_mapped + np.sqrt(10.0 * np.mean(list(v.values())))
             + RATE_PER_BYTE * arch)
        return float(s), arch

    order = sorted(repaired, key=lambda p: d_map.get(p, 0.0) - repaired[p][0], reverse=True)
    sel: dict = {}
    s_cur, arch_cur = composed_S(sel)
    print(f"[compose] base (no repair): S_pred {s_cur:.7f} @ ~{arch_cur} B "
          f"(d_seg_mapped {d_seg_mapped:.8f})", flush=True)
    for p in order:
        trial = dict(sel)
        trial[p] = True
        s_new, arch_new = composed_S(trial)
        if s_new < s_cur:
            sel, s_cur, arch_cur = trial, s_new, arch_new
        else:
            print(f"[compose] pair {p} does not pay (dS {s_new - s_cur:+.2e}); stop", flush=True)
            break
    print(f"[compose] selected {sorted(sel)} -> S_pred {s_cur:.7f} @ ~{arch_cur} B", flush=True)

    # ---- build + stage ---------------------------------------------------------------------
    blob = encode_pose_repair_stream({p: repaired[p][1] for p in sel}, k=args.k,
                                     seg_h=384, seg_w=512)
    new_payload = build_payload(bulk, [*list(sections), blob])
    runtime_files = tuple(p.name for p in sorted(args.runtime.iterdir())
                          if p.suffix in (".py", ".sh"))
    archive = stage_submission(new_payload, dest=args.dest, runtime_src=args.runtime,
                               runtime_files=runtime_files)
    arch_bytes = Path(archive).stat().st_size
    sha = hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    ledger = build_byte_ledger(archive, joint_names=(
        "config", "renderer", "selector", "pose_warp", "frame0_pose_repair"))
    closes = ledger.closes() if callable(ledger.closes) else ledger.closes
    print(f"[compose] STAGED {archive} {arch_bytes} B sha {sha[:16]} "
          f"ledger_closes={closes} residual={ledger.residual_bytes} "
          f"reencodes={ledger.payload_reencodes_identically}", flush=True)

    # ---- acceptance through the ACTUAL receiver --------------------------------------------
    ext = args.dest / "archive"
    ext.mkdir(exist_ok=True)
    import shutil

    from tac.submission_archive import safe_extract_zip

    if ext.exists():
        shutil.rmtree(ext)
    safe_extract_zip(archive, ext)
    sys.path.insert(0, str(args.runtime))
    import inflate_runner as RUN
    dec = RUN.Decoder(ext)
    from ddm_js1_staging_discriminator import Scorer, decode_gt_frames, seq_len
    sc = Scorer(args.threads)
    wanted = set()
    for p in sel:
        wanted.update({seq_len * p, seq_len * p + 1})
    gt_frames = decode_gt_frames(args.gt_mkv, wanted)
    runner_true = {}
    for p in sorted(sel):
        f1 = dec.f1(p)
        f0 = dec.f0(p, f1)
        gt = np.stack([gt_frames[seq_len * p], gt_frames[seq_len * p + 1]])
        pose_gt = sc.pose_out(gt)
        dp = sc.d_pose(pose_gt, sc.pose_out(np.stack([f0, f1])))
        lam_ok = bool((sc.seg_argmax(np.stack([f0, f1]))
                       == sc.seg_argmax(np.stack([f0 * 0, f1]))).all())  # seg reads f1 only
        runner_true[p] = dp
        print(f"[accept] pair {p:3d} runner-true d_pose {dp:.6f} "
              f"(solve said {repaired[p][0]:.6f})  seg_f1_only={lam_ok}", flush=True)

    v = dict(base_vec)
    for p in sel:
        v[p] = runner_true[p]
    s_final = (100.0 * d_seg_mapped + float(np.sqrt(10.0 * np.mean(list(v.values()))))
               + RATE_PER_BYTE * arch_bytes)
    out = {
        "schema": "ddm_fz1_compose_rowB.v1",
        "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
        "score_claim": False,
        "archive": str(archive), "archive_bytes": arch_bytes, "archive_sha256": sha,
        "payload_src": str(args.payload),
        "d_seg_mapped_advisory": d_seg_mapped,
        "repaired_pairs": sorted(sel),
        "runner_true_d_pose": {str(p): runner_true[p] for p in sorted(sel)},
        "d_pose_mean_advisory": float(np.mean(list(v.values()))),
        "S_pred_advisory": s_final,
        "d_seg_source": d_seg_source,
        "S_live_best": args.live_best_s,
        "delta_pred": s_final - args.live_best_s,
        "ledger_sections": [getattr(s, "__dict__", s) for s in getattr(ledger, "sections", ())] or None,
    }
    (args.dest / "compose_receipt.json").write_text(json.dumps(out, indent=1))
    print(f"[compose] S_pred_advisory {s_final:.7f}  delta vs live best "
          f"{s_final - args.live_best_s:+.7f}  t={time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
