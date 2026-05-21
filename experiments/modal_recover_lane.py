# SPDX-License-Identifier: MIT
"""Recover artifacts from a spawned Modal lane training run.

Companion to `experiments/modal_train_lane.py`. When a lane is dispatched
via `.spawn()` (the only way to survive local CLI disconnect), the function
runs detached and the call_id is saved to
`experiments/results/lane_<label>_modal/modal_call_id.txt`.

Use this script to:
1. Poll the function call status
2. Once complete, download artifacts to local
3. Surface any auth score with device-aware advisory/CUDA labeling

Usage:
    .venv/bin/python experiments/modal_recover_lane.py --label lane_omega_hessian
    .venv/bin/python experiments/modal_recover_lane.py --call-id fc-abc123...
    .venv/bin/python experiments/modal_recover_lane.py --all  # poll every lane_*_modal/

Reference: project_modal_pipeline_trusted_lane_g_v3_1_04_20260429
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.deploy.modal.training_claims import append_modal_training_terminal_claim
from tac.deploy.modal.training_cost import append_modal_training_cost_anchor
from tac.deploy.modal.auth_eval import (
    ModalArtifactWriteError,
    materialize_modal_artifacts,
)
from tac.deploy.modal.harvest_outcomes import append_terminal_call_id_ledger_event


def label_from_modal_result_dir(dirname: str) -> str:
    """Return the exact label encoded by `lane_<label>_modal`.

    Do not use `str.replace`, because labels themselves can start with
    `lane_` (for example `lane_g_v3_...`). Removing every occurrence silently
    points recovery at a different sentinel directory.
    """
    prefix = "lane_"
    suffix = "_modal"
    if not dirname.startswith(prefix) or not dirname.endswith(suffix):
        raise ValueError(f"not a Modal lane result directory: {dirname}")
    return dirname[len(prefix):-len(suffix)]


def label_from_modal_result(result: dict) -> str | None:
    """Recover the original Modal dispatch label from returned artifacts.

    Direct ``--call-id`` recovery does not have the local sentinel path, so the
    label must come from the provider result itself. Prefer the modal log name
    because it is written by ``modal_train_lane.py`` from the original dispatch
    label; fall back to structured provenance and then broad artifact prefixes.
    """

    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    for key in artifacts:
        name = Path(str(key)).name
        match = re.fullmatch(r"modal_lane_(.+)\.log", name)
        if match:
            return match.group(1)
    for key, data in artifacts.items():
        if not str(key).endswith("provenance.json"):
            continue
        try:
            payload = json.loads(bytes(data).decode("utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            label = str(payload.get("dispatch_instance_job_id") or "").strip()
            if label:
                return label
    for key in artifacts:
        match = re.match(r"results/([^/]+)/", str(key))
        if match:
            return match.group(1)
    for key in artifacts:
        match = re.match(r"([^/]+)_results/", str(key))
        if match:
            return match.group(1)
    return None


def _first_present(payload: dict, keys: tuple[str, ...]):
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def auth_score_summary_lines(payload: dict, *, label: str, source: str) -> list[str] | None:
    """Format a recovered auth-eval payload without promoting CPU/MPS scores."""
    score = _first_present(payload, ("score_recomputed_from_components", "score", "final_score"))
    if score is None:
        return None

    provenance = payload.get("provenance")
    device = None
    if isinstance(provenance, dict):
        device = provenance.get("device")
    if device is None:
        device = payload.get("device")
    device_label = str(device or "unknown").lower()
    is_cuda = device_label == "cuda"

    if is_cuda:
        header = (
            "\n=== CUDA AUTH SCORE "
            f"(UNADJUDICATED, NON-PROMOTABLE): {score} (label={label}) ==="
        )
    else:
        header = (
            "\n=== ADVISORY AUTH SCORE "
            f"(NON-PROMOTABLE, device={device_label}): {score} (label={label}) ==="
        )

    lines = [
        header,
        f"  source:  {source}",
    ]
    if not is_cuda:
        lines.append(
            "  CUDA required: rerun the exact archive via contest_auth_eval.py "
            "--device cuda before promotion, ranking, retirement, or stack claims."
        )
    else:
        lines.append(
            "  Adjudication required: verify archive SHA, runtime tree SHA, sample "
            "count, exact evaluator schema, component recomputation, and terminal "
            "dispatch claim before promotion, ranking, retirement, or stack claims."
        )

    pose = _first_present(payload, ("avg_posenet_dist", "pose", "pose_dist", "posenet_dist"))
    seg = _first_present(payload, ("avg_segnet_dist", "seg", "seg_dist", "segnet_dist"))
    rate = _first_present(payload, ("rate", "archive_rate"))
    lines.extend([
        f"  PoseNet: {pose}",
        f"  SegNet:  {seg}",
        f"  Rate:    {rate}",
    ])
    return lines


def print_still_running_guidance(call_id: str, label: str | None = None) -> None:
    """Print commands supported by current Modal CLI versions."""
    if label:
        recover = f".venv/bin/python experiments/modal_recover_lane.py --label {label}"
    else:
        recover = f".venv/bin/python experiments/modal_recover_lane.py --call-id {call_id}"
    print("  STILL RUNNING — re-run recovery later:")
    print(f"    {recover}")
    print("  To stream logs with Modal 1.4+:")
    print("    .venv/bin/modal app list")
    print("    .venv/bin/modal app logs <app-id>")


def _append_cost_band_anchor_if_requested(
    *,
    out_dir: Path,
    result: dict,
) -> dict | None:
    metadata_path = out_dir / "modal_metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            return None
        manifest = append_modal_training_cost_anchor(
            out_dir=out_dir,
            metadata=metadata,
            result=result,
        )
    except Exception as exc:
        manifest = {
            "schema": "modal_training_cost_anchor_append_v1",
            "appended": False,
            "reason": f"append_failed:{type(exc).__name__}:{exc}",
            "score_claim": False,
            "promotion_eligible": False,
        }
        try:
            (out_dir / "cost_band_anchor_append_error.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
    return manifest


def _append_terminal_claim_if_requested(
    *,
    out_dir: Path,
    result: dict | None,
    status: str | None = None,
) -> dict | None:
    metadata_path = out_dir / "modal_metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            return None
        manifest = append_modal_training_terminal_claim(
            repo_root=REPO_ROOT,
            out_dir=out_dir,
            metadata=metadata,
            result=result,
            status=status,
            agent="codex:modal_recover_lane",
        )
    except Exception as exc:
        manifest = {
            "schema": "modal_training_terminal_claim_v1",
            "appended": False,
            "reason": f"append_failed:{type(exc).__name__}:{exc}",
            "score_claim": False,
            "promotion_eligible": False,
        }
        try:
            (out_dir / "modal_training_terminal_claim_error.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
    return manifest


def _print_terminal_claim_summary(manifest: dict | None) -> None:
    if manifest is None:
        return
    if manifest.get("appended"):
        print(
            "  terminal claim appended: "
            f"status={manifest.get('status')} lane_id={manifest.get('lane_id')}"
        )
    else:
        print(
            "  terminal claim skipped: "
            f"{manifest.get('reason', 'unknown')}"
        )


def modal_terminal_ledger_metadata(
    *,
    out_dir: Path,
    call_id: str,
    label: str | None,
) -> dict:
    """Build terminal Modal call-ledger metadata from recovered dispatch metadata."""

    fallback_label = label or call_id
    metadata = {
        "call_id": call_id,
        "lane_id": fallback_label,
        "label": fallback_label,
        "platform": "modal",
    }
    metadata_path = out_dir / "modal_metadata.json"
    if not metadata_path.is_file():
        return metadata
    try:
        recovered = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return metadata
    if not isinstance(recovered, dict):
        return metadata
    lane_id = str(recovered.get("lane_id") or "").strip()
    recovered_label = str(recovered.get("label") or "").strip()
    recovered_call_id = str(recovered.get("call_id") or "").strip()
    if lane_id:
        metadata["lane_id"] = lane_id
    if recovered_label:
        metadata["label"] = recovered_label
    if recovered_call_id:
        metadata["call_id"] = recovered_call_id
    return metadata


def recover_one(label: str | None, call_id: str | None) -> int:
    import modal

    if label and not call_id:
        sentinel = REPO_ROOT / "experiments" / "results" / f"lane_{label}_modal" / "modal_call_id.txt"
        if not sentinel.exists():
            print(f"FATAL: no call_id sentinel at {sentinel}", file=sys.stderr)
            return 2
        call_id = sentinel.read_text().strip()
    if not call_id:
        print("FATAL: must provide --label or --call-id", file=sys.stderr)
        return 2

    print(f"=== Polling Modal call_id={call_id} ===")
    fc = modal.FunctionCall.from_id(call_id)

    # `get(timeout=0)` returns immediately if done; raises TimeoutError if not.
    try:
        result = fc.get(timeout=0)
    except TimeoutError:
        print_still_running_guidance(call_id, label=label)
        return 0
    except modal.exception.OutputExpiredError:
        append_terminal_call_id_ledger_event(
            repo_root=REPO_ROOT,
            metadata={
                "call_id": call_id,
                "lane_id": label or call_id,
                "label": label or call_id,
                "platform": "modal",
            },
            harvested={"status": "expired", "crash_kind": "RESULT_CACHE_EXPIRED"},
            terminal_claim=None,
            agent="codex:modal_recover_lane",
        )
        if label:
            expired_out_dir = REPO_ROOT / "experiments" / "results" / f"lane_{label}_modal"
            _print_terminal_claim_summary(
                _append_terminal_claim_if_requested(
                    out_dir=expired_out_dir,
                    result=None,
                    status="failed_modal_training_result_cache_expired",
                )
            )
        print("  OUTPUT EXPIRED (Modal expires output after 7 days). Lane logs may still be available.", file=sys.stderr)
        return 3

    if not isinstance(result, dict):
        append_terminal_call_id_ledger_event(
            repo_root=REPO_ROOT,
            metadata={
                "call_id": call_id,
                "lane_id": label or call_id,
                "label": label or call_id,
                "platform": "modal",
            },
            harvested={
                "status": "error_unexpected_result_type",
                "error_type": type(result).__name__,
            },
            terminal_claim=None,
            agent="codex:modal_recover_lane",
        )
        print(f"  unexpected result type: {type(result)}", file=sys.stderr)
        return 4

    # Save artifacts
    if not label:
        # Derive label from the result itself, else isolate by call_id.
        label = label_from_modal_result(result) or call_id
    out_dir = REPO_ROOT / "experiments" / "results" / f"lane_{label}_modal"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = result.get("artifacts", {})
    try:
        saved_artifacts = (
            materialize_modal_artifacts(out_dir=out_dir, artifacts=artifacts)
            if isinstance(artifacts, dict)
            else []
        )
    except ModalArtifactWriteError as exc:
        failure = {
            "schema_version": "modal_lane_recover_summary_v1",
            "status": "invalid_artifacts",
            "call_id": call_id,
            "label": label,
            "artifact_write_errors": exc.errors,
            "score_claim": False,
            "promotion_eligible": False,
        }
        (out_dir / "modal_recover_summary.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _print_terminal_claim_summary(
            _append_terminal_claim_if_requested(
                out_dir=out_dir,
                result=failure,
                status="failed_modal_training_invalid_artifacts",
            )
        )
        append_terminal_call_id_ledger_event(
            repo_root=REPO_ROOT,
            metadata=modal_terminal_ledger_metadata(
                out_dir=out_dir,
                call_id=call_id,
                label=label,
            ),
            harvested=failure,
            terminal_claim=None,
            agent="codex:modal_recover_lane",
        )
        print(
            "  FATAL: Modal returned unsafe or malformed artifacts; "
            f"wrote fail-closed summary to {out_dir / 'modal_recover_summary.json'}",
            file=sys.stderr,
        )
        return 5
    n_saved = len(saved_artifacts)
    print(f"  saved {n_saved} artifacts → {out_dir}")

    rc = result.get("returncode")
    timed_out = result.get("timed_out", False)
    elapsed = result.get("elapsed_seconds")
    print(f"  returncode={rc}  timed_out={timed_out}  elapsed={elapsed:.0f}s" if elapsed else f"  returncode={rc}")
    cost_anchor_manifest = _append_cost_band_anchor_if_requested(
        out_dir=out_dir,
        result=result,
    )
    if cost_anchor_manifest is not None:
        if cost_anchor_manifest.get("appended"):
            print(
                "  cost-band anchor appended: "
                f"estimated_cost_usd={cost_anchor_manifest.get('estimated_cost_usd')} "
                f"gpu={cost_anchor_manifest.get('gpu')}"
            )
        else:
            print(
                "  cost-band anchor skipped: "
                f"{cost_anchor_manifest.get('reason', 'unknown')}"
            )
    _print_terminal_claim_summary(
        _append_terminal_claim_if_requested(
            out_dir=out_dir,
            result=result,
        )
    )
    append_terminal_call_id_ledger_event(
        repo_root=REPO_ROOT,
        metadata=modal_terminal_ledger_metadata(
            out_dir=out_dir,
            call_id=call_id,
            label=label,
        ),
        harvested={key: value for key, value in result.items() if key != "artifacts"},
        terminal_claim=None,
        agent="codex:modal_recover_lane",
    )

    # Extract score from artifacts
    score_found = False
    for path, data_bytes in result.get("artifacts", {}).items():
        if score_found:
            break
        if path.endswith(".json"):
            try:
                d = json.loads(data_bytes.decode())
                if isinstance(d, dict):
                    summary = auth_score_summary_lines(d, label=label, source=path)
                    if summary is not None:
                        print("\n".join(summary))
                        score_found = True
            except Exception:
                pass
        elif path.endswith(".log"):
            try:
                text = data_bytes.decode(errors="ignore")
                m = re.search(r"RESULT_JSON:\s*(\{[^\n]+\})", text)
                if m:
                    d = json.loads(m.group(1))
                    summary = auth_score_summary_lines(
                        d, label=label, source=f"{path} (RESULT_JSON line)"
                    )
                    if summary is not None:
                        print("\n".join(summary))
                        score_found = True
            except Exception:
                pass

    if not score_found:
        print(f"  WARNING: no auth score in artifacts. Check {out_dir}/ for run.log + auth_eval.log.")
    if rc != 0:
        return rc
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", help="Lane label (read call_id from sentinel)")
    p.add_argument("--call-id", help="Modal function call ID directly")
    p.add_argument("--all", action="store_true",
                   help="Poll every lane_*_modal/ directory with a sentinel")
    args = p.parse_args()

    if args.all:
        results_dir = REPO_ROOT / "experiments" / "results"
        labels = []
        for d in sorted(results_dir.glob("lane_*_modal")):
            sentinel = d / "modal_call_id.txt"
            if sentinel.exists():
                labels.append(label_from_modal_result_dir(d.name))
        print(f"Polling {len(labels)} lanes: {labels}")
        rcs = [recover_one(label=lbl, call_id=None) for lbl in labels]
        return max(rcs) if rcs else 0
    return recover_one(args.label, args.call_id)


if __name__ == "__main__":
    sys.exit(main())
