#!/usr/bin/env python3
"""Regenerate the sealed jd1 #366 ticket (6564914a...) against the TP1 boundary WINNER.

The sealed ticket chains from the OLD full_birth_lane_on endpoint (ep946, 130-epoch
window). Per the ticket's own recursion contract + the seg-uncapped/pose-proper-order
steer ([[seg-uncapped-pose-proper-time-order-20260805]]), the boundary fire swaps the
resume checkpoint to the adjudicated WINNER endpoint and shifts the horizon by the SAME
window length; the pose engagement stays predicate-driven (post_knee, start-epoch 0 —
nothing to shift). EMA decay is RE-DERIVED via the trainer's own derive_ema_decay for
the new geometry (LawRef ema_decay_run_geometry_v1: the LAW transfers, never the pinned
constant). jd1's decay law — VERIFIED by inverting the sealed value — is U =
resume_epoch x parent_steps_per_epoch (parent-chain accumulated updates at the resume
point, the warm-start law): the script first REPRODUCES the sealed 0.9999436... from
the old ep946 resume as a self-check (mismatch = geometry drift: STOP), then applies
the same law to the winner epoch to produce the NEW decay.

Emits: regenerated argv (launch.sh-style) + a ticket JSON next to the winner run dir.
$0, no launch — the governed launch fires separately under the standing GO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

SEALED_TICKET = REPO / ".omx/research/ddm_jd1_20260805/JD1_TICKET.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--winner-dir", type=Path, required=True,
                    help="the adjudicated TP1 winner run dir (full_birth_lane_on_w4 | _w4m)")
    ap.add_argument("--winner-ckpt", type=Path, default=None,
                    help="override the resume checkpoint (e.g. an INTERIOR-min intra ckpt "
                         "when the adjudicated best state is not the window endpoint — "
                         "never-weaker-state m40); default = <winner-dir>/checkpoints/"
                         "stage_seg_trunk_tau_final.npz")
    ap.add_argument("--out-ticket", type=Path, default=None)
    args = ap.parse_args()

    import numpy as np

    from experiments.train_tr1_partition_renderer_mlx import derive_ema_decay

    t = json.loads(SEALED_TICKET.read_text())
    argv = list(t["argv"])

    def get(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    def put(flag: str, value: str) -> None:
        argv[argv.index(flag) + 1] = value

    # Sealed geometry (re-derived from the ticket itself, never assumed).
    # jd1's decay law (verified by inverting the sealed value): the EMA decay is
    # derived from the PARENT CHAIN's accumulated updates at the RESUME POINT —
    # U = resume_epoch * parent_steps_per_epoch — so the inherited shadow's
    # effective horizon matches its actual history (the warm-start law [[m72]]),
    # NOT from the child window length.
    old_resume = Path(get("--resume-from"))
    z = np.load(old_resume, allow_pickle=False)
    old_ep = int(z["meta::epoch"][0])
    window = int(get("--epochs")) - old_ep
    num_pairs = int(get("--num-pairs"))
    PARENT_BATCH_PAIRS = 8  # the tp1 chain's --batch-pairs (w3/w4 launchers), provenance: launch_w4_staged.sh
    parent_steps = max(1, num_pairs // PARENT_BATCH_PAIRS)
    sealed_decay = float(get("--ema-decay"))
    check, _ = derive_ema_decay(old_ep * parent_steps)
    if abs(check - sealed_decay) > 1e-9:
        raise SystemExit(
            f"GEOMETRY DRIFT: derive_ema_decay({old_ep}x{parent_steps}) = {check!r} "
            f"!= sealed {sealed_decay!r} — the sealed decay law is not what this script "
            "reconstructed; re-derive by hand, do not fire.")

    # Winner state (endpoint by default; --winner-ckpt overrides for interior-min states).
    winner_ckpt = args.winner_ckpt or (args.winner_dir / "checkpoints" / "stage_seg_trunk_tau_final.npz")
    if not winner_ckpt.exists():
        raise SystemExit(f"winner checkpoint missing: {winner_ckpt}")
    zw = np.load(winner_ckpt, allow_pickle=False)
    win_ep = int(zw["meta::epoch"][0])

    out_dir = f"/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/tr1_joint_pose_finish_from_{args.winner_dir.name}"
    new_decay, decay_prov = derive_ema_decay(win_ep * parent_steps)
    put("--resume-from", str(winner_ckpt))
    put("--epochs", str(win_ep + window))
    put("--ema-decay", repr(new_decay))
    put("--out-dir", out_dir)

    # SEG-HOLD FLOOR CALIBRATION CURE (recursive pass 2026-08-05, MEASURED defect):
    # the sealed floor source `checkpoint_tail_ep_loss` latches the PARENT tail ep_loss,
    # but the parent arm may run a DIFFERENT loss form than the child (w4m margin-ON tail
    # ep_loss 1.598 vs margin-OFF 0.480 at MATCHED d_seg ~0.0039 => the margin weighting
    # inflates the loss SCALE 3.33x without changing quality). A cross-form floor is ~3.3x
    # too loose: pose descent could triple seg before the hinge resists. Cure (no invented
    # constants): force ONE seg-only calibration epoch (start_epoch = resume+2) and latch
    # the floor from the child's OWN measured seg-only epoch loss under ITS OWN loss form.
    put("--jd1-pose-finish-start-epoch", str(win_ep + 2))
    put("--jd1-seg-hold-floor-source", "last_pre_pose_epoch_loss")

    regen = dict(t)
    regen["argv"] = argv
    regen["child_resume_from"] = str(winner_ckpt)
    regen["child_out_dir"] = out_dir
    regen["regenerated_from"] = {
        "sealed_ticket_hash": t.get("ticket_hash"),
        "winner_dir": str(args.winner_dir), "winner_epoch": win_ep,
        "winner_ckpt": str(winner_ckpt),
        "winner_ckpt_is_override": args.winner_ckpt is not None,
        "old_resume_epoch": old_ep, "window_epochs": window,
        "ema_decay_rederivation": {
            "law": "decay derived from PARENT-CHAIN accumulated updates at the resume "
                   "point (U = resume_epoch * parent_steps_per_epoch), the warm-start "
                   "law — verified by reproducing the SEALED value from the OLD resume "
                   "epoch before applying the same law to the winner",
            "sealed_value_reproduced": sealed_decay, "sealed_check": check,
            "new_value": new_decay, "provenance": decay_prov,
        },
        "seg_hold_floor_calibration_cure": {
            "defect": "cross-form floor scale: parent margin-ON tail ep_loss 1.598 vs "
                      "margin-OFF 0.480 at matched d_seg ~0.0039 (3.33x) — sealed "
                      "checkpoint_tail_ep_loss floor would be ~3.3x too loose for a "
                      "margin-OFF child (MEASURED, recursive pass 2026-08-05)",
            "cure": "one seg-only calibration epoch (start_epoch=resume+2) + floor from "
                    "last_pre_pose_epoch_loss (child's OWN loss form, no invented constants)",
            "flags_changed_from_sealed": ["--jd1-pose-finish-start-epoch",
                                          "--jd1-seg-hold-floor-source"],
        },
        "steer": "seg-uncapped-pose-proper-time-order-20260805 (operator 2026-08-05)",
    }
    body = json.dumps(regen, indent=1, sort_keys=True)
    regen["ticket_hash"] = hashlib.sha256(body.encode()).hexdigest()
    out_ticket = args.out_ticket or (args.winner_dir.parent /
                                     f"jd1_ticket_regenerated_from_{args.winner_dir.name}.json")
    out_ticket.write_text(json.dumps(regen, indent=1, sort_keys=True))
    print(json.dumps({"winner_epoch": win_ep, "epochs": win_ep + window,
                      "window": window, "ema_decay": new_decay,
                      "out_dir": out_dir, "ticket": str(out_ticket),
                      "ticket_hash": regen["ticket_hash"]}, indent=1))
    print("\nLAUNCH ARGV (governed launch consumes the ticket; do not hand-run):")
    print(" ".join(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
