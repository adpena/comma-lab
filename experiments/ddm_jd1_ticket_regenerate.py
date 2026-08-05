#!/usr/bin/env python3
"""Regenerate sealed JD1/JD3/JD4 continuation tickets.

The regenerator is a value-custody factory: every emitted ticket is rebuilt from
the final argv, validates inherited recursive metadata, and refuses stale child
out-dir reuse before writing.  It emits tickets only; governed launch remains a
separate MAIN action.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tac.witness_dsl.scope_laws import (  # noqa: E402
    jd1_tail_average_scope_law_refs,
    jd3_default_scope_law_refs,
    ticket_payload_hash,
    validate_ticket_scope_laws,
)

TRAINER_SCRIPT = "experiments/train_tr1_partition_renderer_mlx.py"
VENV_PYTHON = str(REPO / ".venv/bin/python")
SEALED_TICKET = REPO / ".omx/research/ddm_jd1_20260805/JD1_TICKET.json"
JD1_V2_RUN_DIR = Path(
    "/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/"
    "tr1_joint_pose_finish_from_full_birth_lane_on_w4m"
)
JD3_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_jd3_20260805")
JD3_FULL_TICKET = Path("/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/jd3_ticket_v3_full_entry_cont.json")
JD4_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_jd4_20260805")
JD4_ENDPOINT_CKPT = (
    JD3_ROOT / "full_v3_endpoint_ep1405_snapshot" / "stage_joint_pose_finish_final.npz"
)
JD3_START_CANDIDATES = {
    "entry_ep1336": "checkpoints/stage_joint_pose_finish_entry.npz",
    "refuse_final_ep1354": "checkpoints/stage_joint_pose_finish_final.npz",
}
PARENT_BATCH_PAIRS = 8  # TP1 chain provenance: launch_w4_staged.sh.


def argv_get(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def argv_put(argv: list[str], flag: str, value: str) -> None:
    argv[argv.index(flag) + 1] = str(value)


def argv_ensure_value(argv: list[str], flag: str, value: str) -> None:
    if flag in argv:
        argv_put(argv, flag, str(value))
    else:
        argv.extend([flag, str(value)])


def argv_ensure_flag(argv: list[str], flag: str) -> None:
    if flag not in argv:
        argv.append(flag)


def argv_drop_flag(argv: list[str], flag: str) -> None:
    while flag in argv:
        i = argv.index(flag)
        del argv[i:i + 1]


def ensure_repo_python_argv(argv: list[str]) -> list[str]:
    """Return argv with the repo venv Python as argv[0]."""
    if not argv:
        raise SystemExit("ticket argv is empty")
    out = list(argv)
    first = Path(out[0]).name
    if first.startswith("python"):
        out[0] = VENV_PYTHON
        if len(out) < 2 or not out[1].endswith("train_tr1_partition_renderer_mlx.py"):
            raise SystemExit("python argv ticket must put the TR1 trainer script at argv[1]")
        return out
    if not out[0].endswith("train_tr1_partition_renderer_mlx.py"):
        raise SystemExit(f"ticket argv[0] is not the TR1 trainer or Python: {out[0]}")
    return [VENV_PYTHON, *out]


def argv_value_map(argv: list[str]) -> dict[str, str | bool]:
    values: dict[str, str | bool] = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            i += 1
            continue
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            values[tok] = argv[i + 1]
            i += 2
        else:
            values[tok] = True
            i += 1
    return values


def rebuild_lever_overrides_from_argv(levers: list[dict[str, Any]], argv: list[str]) -> list[dict[str, Any]]:
    final = argv_value_map(argv)
    rebuilt = copy.deepcopy(levers)
    for lever in rebuilt:
        overrides = lever.get("overrides")
        if not isinstance(overrides, dict):
            continue
        for flag in list(overrides):
            if flag not in final:
                raise SystemExit(f"lever override {flag} declared but final argv lacks it")
            overrides[flag] = final[flag]
    return rebuilt


def validate_lever_overrides_match_argv(ticket: dict[str, Any]) -> None:
    final = argv_value_map([str(x) for x in ticket.get("argv", [])])
    mismatches: list[str] = []
    for lever in ticket.get("levers", []):
        overrides = lever.get("overrides") if isinstance(lever, dict) else None
        if not isinstance(overrides, dict):
            continue
        for flag, declared in overrides.items():
            if flag not in final:
                mismatches.append(f"{lever.get('name', '<unnamed>')} {flag}: absent from argv")
                continue
            if declared != final[flag]:
                mismatches.append(
                    f"{lever.get('name', '<unnamed>')} {flag}: declared {declared!r} "
                    f"!= argv {final[flag]!r}"
                )
    if mismatches:
        joined = "; ".join(mismatches[:8])
        raise SystemExit(f"declared-vs-argv mismatch in emitted ticket: {joined}")


def _merged_scope_laws(*groups: Any) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for group in groups:
        for row in group or []:
            name = str(row.get("name", ""))
            if not name:
                raise SystemExit("scope law row is missing name")
            by_name[name] = dict(row)
    rows = list(by_name.values())
    if rows:
        validate_ticket_scope_laws(rows)
    return rows


def sync_ticket_scope_laws_from_argv(regen: dict[str, Any], argv: list[str]) -> None:
    scope_laws = list(regen.get("scope_laws") or [])
    argv_values = argv_value_map(argv)
    if argv_values.get("--jd1-ema-mode") == "plateau_tail_average":
        scope_laws = _merged_scope_laws(scope_laws, jd1_tail_average_scope_law_refs())
    if scope_laws:
        regen["scope_laws"] = _merged_scope_laws(scope_laws)
    else:
        regen.pop("scope_laws", None)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def validate_recursive_resume_template(
    ticket: dict[str, Any],
    *,
    child_out_dir: Path,
    declared_new_out_dir: Path | None = None,
) -> None:
    loop = ticket.get("recursive_encode_pass_loop")
    if not isinstance(loop, dict):
        return
    policy = loop.get("continue_policy")
    if not isinstance(policy, dict):
        return
    template = policy.get("next_resume_from_template")
    if not template:
        return
    path = Path(str(template))
    roots = [child_out_dir]
    if declared_new_out_dir is not None:
        roots.append(declared_new_out_dir)
    if not any(_is_under(path, root) for root in roots):
        raise SystemExit(
            "recursive_encode_pass_loop.next_resume_from_template must resolve under "
            f"child_out_dir or declared new out-dir: {path}"
        )


def set_recursive_resume_template(ticket: dict[str, Any], *, child_out_dir: Path) -> None:
    loop = ticket.setdefault("recursive_encode_pass_loop", {})
    policy = loop.setdefault("continue_policy", {})
    policy["next_resume_from_template"] = str(
        child_out_dir / "checkpoints" / "stage_joint_pose_finish_final.npz"
    )
    policy["next_ticket_rule"] = (
        "regenerate from the actual completed child endpoint into a fresh child out-dir; "
        "never inherit an ancestor lane template"
    )
    loop["launch_now"] = False
    loop["reason_not_launched"] = "MAIN fires after endpoint probe adjudication; jd4 owns no scorer slot"


def refuse_child_out_dir_checkpoint_reuse(child_out_dir: Path) -> None:
    ckpt_dir = child_out_dir / "checkpoints"
    if ckpt_dir.exists() and any(ckpt_dir.glob("*.npz")):
        raise SystemExit(
            f"child_out_dir already contains checkpoints from another run: {child_out_dir}"
        )


def read_checkpoint_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"checkpoint missing: {path}")
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["meta::json"]).decode()) if "meta::json" in z else {}
    tail = meta.get("telemetry_tail") if isinstance(meta, dict) else []
    tail_epochs = [
        int(row["epoch"])
        for row in (tail or [])
        if isinstance(row, dict) and row.get("epoch") is not None
    ]
    return {
        "epoch": int(z["meta::epoch"][0]),
        "meta": meta,
        "last_tail_epoch": max(tail_epochs) if tail_epochs else None,
    }


def checkpoint_stage_ema_u(meta: dict[str, Any]) -> int | None:
    jd1 = meta.get("jd1_pose_finish") if isinstance(meta, dict) else None
    if not isinstance(jd1, dict):
        return None
    prov = str(jd1.get("active_ema_decay_provenance", ""))
    match = re.search(r"\bU=(\d+)\b", prov)
    return int(match.group(1)) if match else None


def maybe_force_window_reanchor(argv: list[str], *, parent_u: int | None, new_u: int) -> bool:
    if parent_u is None or int(parent_u) != int(new_u):
        argv_ensure_flag(argv, "--jd1-force-ema-reanchor-on-resume")
        return True
    argv_drop_flag(argv, "--jd1-force-ema-reanchor-on-resume")
    return False


def finalize_ticket(
    regen: dict[str, Any],
    *,
    argv: list[str],
    child_out_dir: Path,
    child_resume_from: Path,
    out_ticket: Path,
) -> dict[str, Any]:
    final_argv = ensure_repo_python_argv(argv)
    refuse_child_out_dir_checkpoint_reuse(child_out_dir)
    regen["argv"] = final_argv
    regen["child_resume_from"] = str(child_resume_from)
    regen["child_out_dir"] = str(child_out_dir)
    regen["launch_now"] = False
    set_recursive_resume_template(regen, child_out_dir=child_out_dir)
    validate_recursive_resume_template(regen, child_out_dir=child_out_dir)
    regen["levers"] = rebuild_lever_overrides_from_argv(regen.get("levers", []), final_argv)
    validate_lever_overrides_match_argv(regen)
    sync_ticket_scope_laws_from_argv(regen, final_argv)
    regen.pop("ticket_hash", None)
    regen["ticket_hash"] = ticket_payload_hash(regen)
    out_ticket.parent.mkdir(parents=True, exist_ok=True)
    out_ticket.write_text(json.dumps(regen, indent=1, sort_keys=True) + "\n")
    return regen


def sealed_jd1_geometry(argv: list[str], derive_ema_decay) -> tuple[int, int, float, float]:
    old_resume = Path(argv_get(argv, "--resume-from"))
    z = np.load(old_resume, allow_pickle=False)
    old_ep = int(z["meta::epoch"][0])
    window = int(argv_get(argv, "--epochs")) - old_ep
    num_pairs = int(argv_get(argv, "--num-pairs"))
    parent_steps = max(1, num_pairs // PARENT_BATCH_PAIRS)
    sealed_decay = float(argv_get(argv, "--ema-decay"))
    check, _ = derive_ema_decay(old_ep * parent_steps)
    if abs(check - sealed_decay) > 1e-9:
        raise SystemExit(
            f"GEOMETRY DRIFT: derive_ema_decay({old_ep}x{parent_steps}) = {check!r} "
            f"!= sealed {sealed_decay!r}; re-derive by hand, do not fire."
        )
    return old_ep, window, sealed_decay, check


def emit_jd4_continuation(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.train_tr1_partition_renderer_mlx import derive_jd1_stage_ema_decay

    base_ticket = Path(args.base_ticket or JD3_FULL_TICKET)
    t = json.loads(base_ticket.read_text())
    argv = [str(x) for x in t["argv"]]
    resume_ckpt = Path(args.winner_ckpt or JD4_ENDPOINT_CKPT)
    ckpt = read_checkpoint_meta(resume_ckpt)
    source_epoch = int(ckpt["last_tail_epoch"] if ckpt["last_tail_epoch"] is not None else ckpt["epoch"] - 1)
    resume_start_epoch = source_epoch + 1
    window_epochs = int(args.window_epochs)
    if window_epochs <= 0:
        raise SystemExit("--window-epochs must be > 0")
    num_pairs = int(argv_get(argv, "--num-pairs"))
    batch_pairs = int(argv_get(argv, "--batch-pairs"))
    steps_per_epoch = max(1, num_pairs // max(1, batch_pairs))
    new_u = window_epochs * steps_per_epoch
    derived_decay, derived_prov = derive_jd1_stage_ema_decay(window_epochs, steps_per_epoch)
    parent_u = checkpoint_stage_ema_u(ckpt["meta"])
    force_reanchor = maybe_force_window_reanchor(argv, parent_u=parent_u, new_u=new_u)
    child_out_dir = JD4_ROOT / f"tr1_jd4_cont_ep{resume_start_epoch}"
    epoch_limit = int(args.epochs) if args.epochs is not None else resume_start_epoch + window_epochs
    wall_minutes = int(math.ceil((55.0 * window_epochs / 60.0) * 1.5))

    argv_ensure_value(argv, "--resume-from", str(resume_ckpt))
    argv_ensure_value(argv, "--out-dir", str(child_out_dir))
    argv_ensure_value(argv, "--epochs", str(epoch_limit))
    argv_ensure_value(argv, "--max-wall-minutes", str(wall_minutes))
    argv_ensure_value(argv, "--jd1-ema-stage-scope", "window")
    argv_ensure_value(argv, "--jd1-live-gate-telemetry", "on")
    argv_drop_flag(argv, "--full-confirm")

    regen = copy.deepcopy(t)
    regen["regenerated_from"] = {
        "source_ticket": str(base_ticket),
        "source_ticket_hash": t.get("ticket_hash"),
        "resume_checkpoint": str(resume_ckpt),
        "resume_checkpoint_epoch_field": int(ckpt["epoch"]),
        "resume_checkpoint_last_tail_epoch": ckpt["last_tail_epoch"],
        "resume_start_epoch": int(resume_start_epoch),
        "window_epochs": int(window_epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "new_window_u": int(new_u),
        "parent_stage_ema_u": parent_u,
        "force_ema_reanchor_on_resume": bool(force_reanchor),
        "derived_stage_ema_decay": float(derived_decay),
        "derived_stage_ema_decay_provenance": derived_prov,
        "wall_cap_source": "MEASURED 55 s/epoch x window_epochs x 1.5 safety",
        "ticket_version": "jd4_continuation",
        "launch_order": "launch_now=false; MAIN fires after n600 endpoint probe completion",
        "score_claim": False,
    }
    out_ticket = args.out_ticket or (JD4_ROOT / "jd4_ticket_cont_ep1406.json")
    return finalize_ticket(
        regen,
        argv=argv,
        child_out_dir=child_out_dir,
        child_resume_from=resume_ckpt,
        out_ticket=out_ticket,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--winner-dir", type=Path, default=None,
                    help="the adjudicated TP1 winner run dir (full_birth_lane_on_w4 | _w4m)")
    ap.add_argument("--winner-ckpt", type=Path, default=None,
                    help="override the resume checkpoint")
    ap.add_argument("--out-ticket", type=Path, default=None)
    ap.add_argument("--base-ticket", type=Path, default=None,
                    help="base ticket to inherit for JD4 continuation; default is fired full-v3 ticket")
    ap.add_argument("--v3", action="store_true",
                    help="emit jd3 v3 bounded re-smoke ticket against the jd1 v2 endpoint family")
    ap.add_argument("--jd4-continuation", action="store_true",
                    help="emit the jd4 120-epoch continuation ticket from the full-v3 endpoint")
    ap.add_argument("--start-candidate", choices=sorted(JD3_START_CANDIDATES),
                    default="entry_ep1336",
                    help="jd3 v3 resume candidate to reseal")
    ap.add_argument("--smoke-epochs", type=int, default=8,
                    help="bounded jd3 v3 smoke length")
    ap.add_argument("--window-epochs", type=int, default=120,
                    help="jd4 continuation window length")
    ap.add_argument("--epochs", type=int, default=None,
                    help="explicit exclusive epoch limit for jd4; default resume_start+window")
    args = ap.parse_args()

    if args.jd4_continuation:
        regen = emit_jd4_continuation(args)
        print(json.dumps({
            "ticket": str(args.out_ticket or (JD4_ROOT / "jd4_ticket_cont_ep1406.json")),
            "ticket_hash": regen["ticket_hash"],
            "resume_from": regen["child_resume_from"],
            "out_dir": regen["child_out_dir"],
            "epochs": argv_get(regen["argv"], "--epochs"),
            "max_wall_minutes": argv_get(regen["argv"], "--max-wall-minutes"),
            "force_ema_reanchor_on_resume": "--jd1-force-ema-reanchor-on-resume" in regen["argv"],
        }, indent=1))
        print("\nLAUNCH ARGV (governed launch consumes the ticket; MAIN fires later):")
        print(" ".join(regen["argv"]))
        return 0

    from experiments.train_tr1_partition_renderer_mlx import derive_ema_decay

    t = json.loads(SEALED_TICKET.read_text())
    argv = [str(x) for x in t["argv"]]
    old_ep, window, sealed_decay, check = sealed_jd1_geometry(argv, derive_ema_decay)
    num_pairs = int(argv_get(argv, "--num-pairs"))
    parent_steps = max(1, num_pairs // PARENT_BATCH_PAIRS)

    if args.v3:
        if int(args.smoke_epochs) <= 0:
            raise SystemExit("--smoke-epochs must be > 0")
        base_dir = args.winner_dir or JD1_V2_RUN_DIR
        winner_ckpt = args.winner_ckpt or (base_dir / JD3_START_CANDIDATES[args.start_candidate])
        ckpt = read_checkpoint_meta(winner_ckpt)
        win_ep = int(ckpt["epoch"])
        parent_cfg = ckpt["meta"].get("cfg") if isinstance(ckpt["meta"], dict) else {}
        parent_decay = float(parent_cfg["ema_decay"]) if "ema_decay" in parent_cfg else sealed_decay
        run_name = f"tr1_jd3_v3_smoke_{args.start_candidate}"
        child_out_dir = JD3_ROOT / run_name
        epochs = win_ep + 1 + int(args.smoke_epochs)
        max_wall_minutes = int(math.ceil(max(12.0, (112.0 * int(args.smoke_epochs) / 60.0) * 1.5)))

        argv_ensure_value(argv, "--resume-from", str(winner_ckpt))
        argv_ensure_value(argv, "--epochs", str(epochs))
        argv_ensure_value(argv, "--ema-decay", repr(parent_decay))
        argv_ensure_value(argv, "--out-dir", str(child_out_dir))
        argv_ensure_value(argv, "--max-wall-minutes", str(max_wall_minutes))
        argv_ensure_value(argv, "--jd1-pose-finish-start-epoch", "0")
        argv_ensure_value(argv, "--jd1-seg-hold-floor-source", "last_pre_pose_epoch_loss")
        argv_ensure_value(argv, "--jd1-seg-hold-space", "realized")
        argv_ensure_value(argv, "--jd1-realized-hold-margin", "0.0")
        argv_ensure_value(argv, "--jd1-realized-hold-pose-retreat", "0.0")
        argv_ensure_value(argv, "--jd1-realized-hold-max-retreats", "0")
        argv_ensure_value(argv, "--jd1-ema-stage-scope", "window")
        argv_ensure_value(argv, "--jd1-live-gate-telemetry", "on")
        argv_drop_flag(argv, "--full-confirm")

        regen = copy.deepcopy(t)
        regen["regenerated_from"] = {
            "sealed_ticket_hash": t.get("ticket_hash"),
            "start_candidate": args.start_candidate,
            "winner_dir": str(base_dir),
            "winner_epoch": win_ep,
            "winner_ckpt": str(winner_ckpt),
            "old_resume_epoch": old_ep,
            "smoke_epochs": int(args.smoke_epochs),
            "epochs_exclusive": epochs,
            "max_wall_minutes": max_wall_minutes,
            "parent_ema_decay_preserved_until_jd3_reanchor": parent_decay,
            "stage_ema_reanchor": {
                "flag": "--jd1-ema-stage-scope window",
                "law": "trainer derives active_ema_decay from remaining jd3 stage window",
            },
            "ticket_version": "jd3_v3",
            "steer": "ddm_jd3 2026-08-05 #366 reroute",
        }
        regen["scope_laws"] = jd3_default_scope_law_refs()
        out_ticket = args.out_ticket or (
            JD1_V2_RUN_DIR.parent / f"jd3_ticket_v3_{args.start_candidate}.json")
        regen = finalize_ticket(
            regen,
            argv=argv,
            child_out_dir=child_out_dir,
            child_resume_from=winner_ckpt,
            out_ticket=out_ticket,
        )
        print(json.dumps({"candidate": args.start_candidate, "resume_epoch": win_ep,
                          "epochs": epochs, "smoke_epochs": int(args.smoke_epochs),
                          "ema_decay": parent_decay, "out_dir": str(child_out_dir),
                          "max_wall_minutes": max_wall_minutes,
                          "ticket": str(out_ticket),
                          "ticket_hash": regen["ticket_hash"]}, indent=1))
        print("\nLAUNCH ARGV (bounded jd3 smoke; use tools/launch_detached_process.py):")
        print(" ".join(regen["argv"]))
        return 0

    winner_dir = args.winner_dir
    if winner_dir is None:
        raise SystemExit("--winner-dir is required unless --v3 or --jd4-continuation is set")

    winner_ckpt = args.winner_ckpt or (winner_dir / "checkpoints" / "stage_seg_trunk_tau_final.npz")
    ckpt = read_checkpoint_meta(winner_ckpt)
    win_ep = int(ckpt["epoch"])
    child_out_dir = (
        Path("/Volumes/VertigoDataTier/pact/ddm_jd1_20260805")
        / f"tr1_joint_pose_finish_from_{winner_dir.name}"
    )
    new_decay, decay_prov = derive_ema_decay(win_ep * parent_steps)
    argv_ensure_value(argv, "--resume-from", str(winner_ckpt))
    argv_ensure_value(argv, "--epochs", str(win_ep + window))
    argv_ensure_value(argv, "--ema-decay", repr(new_decay))
    argv_ensure_value(argv, "--out-dir", str(child_out_dir))
    argv_ensure_value(argv, "--jd1-pose-finish-start-epoch", str(win_ep + 2))
    argv_ensure_value(argv, "--jd1-seg-hold-floor-source", "last_pre_pose_epoch_loss")

    regen = copy.deepcopy(t)
    regen["regenerated_from"] = {
        "sealed_ticket_hash": t.get("ticket_hash"),
        "winner_dir": str(winner_dir),
        "winner_epoch": win_ep,
        "winner_ckpt": str(winner_ckpt),
        "winner_ckpt_is_override": args.winner_ckpt is not None,
        "old_resume_epoch": old_ep,
        "window_epochs": window,
        "ema_decay_rederivation": {
            "law": "decay derived from parent-chain accumulated updates at the resume point",
            "sealed_value_reproduced": sealed_decay,
            "sealed_check": check,
            "new_value": new_decay,
            "provenance": decay_prov,
        },
        "seg_hold_floor_calibration_cure": {
            "cure": "one seg-only calibration epoch plus last_pre_pose_epoch_loss",
            "flags_changed_from_sealed": [
                "--jd1-pose-finish-start-epoch",
                "--jd1-seg-hold-floor-source",
            ],
        },
        "steer": "seg-uncapped-pose-proper-time-order-20260805",
    }
    out_ticket = args.out_ticket or (
        winner_dir.parent / f"jd1_ticket_regenerated_from_{winner_dir.name}.json")
    regen = finalize_ticket(
        regen,
        argv=argv,
        child_out_dir=child_out_dir,
        child_resume_from=winner_ckpt,
        out_ticket=out_ticket,
    )
    print(json.dumps({"winner_epoch": win_ep, "epochs": win_ep + window,
                      "window": window, "ema_decay": new_decay,
                      "out_dir": str(child_out_dir), "ticket": str(out_ticket),
                      "ticket_hash": regen["ticket_hash"]}, indent=1))
    print("\nLAUNCH ARGV (governed launch consumes the ticket; do not hand-run):")
    print(" ".join(regen["argv"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
