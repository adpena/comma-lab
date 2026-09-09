"""Build the per-charter Catalog #416 historical exemption ledger and census.

This reads only committed repository evidence plus the predecessor census.  It
does not edit historical charters.  Every row is bound to the exact charter and
terminal-evidence bytes and must predate the gate landing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LAW_COMMIT = "8038f9e77d90f7a1c7ad9373e27079e73d801c89"

# These charters were not born through the Codex queue, or (ra2) ended rc=1.
# Their owning result memo is therefore the durable terminal evidence instead
# of a queue-captured final message.  The mapping is explicit so a similarly
# prefixed but scientifically different memo can never be chosen by glob order.
MEMO_EVIDENCE = {
    "ddm_bs4x_stage0_cure_and_stage_fire_20260826.md": "ddm_bs4x_stage0_cure_and_stage_fire_20260826.md",
    "ddm_bs4y_stage_1_4_executor_20260827.md": "ddm_bs4y_stage_1_4_execution_20260826.md",
    "ddm_cl3_hpac_smaller_prior_and_seed_selection_20260905.md": "ddm_cl3_hpac_smaller_prior_and_seed_selection_20260905.md",
    "ddm_dg2_diagonal_reentry_20260823.md": "ddm_dg2_diagonal_reentry_20260823.md",
    "ddm_fe1_per_pair_frame_embedding_realized_search_20260908.md": "ddm_fe1_per_pair_frame_embedding_realized_search_20260908.md",
    "ddm_fs2_carrier_resolve_on_changed_pairs_20260904.md": "ddm_fs2_carrier_resolve_on_changed_pairs_20260904.md",
    "ddm_ft1_shipped_renderer_aligned_finetune_20260903.md": "ddm_ft1_shipped_renderer_aligned_finetune_20260903.md",
    "ddm_fx2_probability_model_axis_all_sections_20260818.md": "ddm_fx2_model_axis_all_sections_20260818.md",
    "ddm_hc2_wrong_half_flip_location_decomposition_20260905.md": "ddm_hc2_wrong_half_flip_location_decomposition_20260905.md",
    "ddm_js3_learned_implicit_conditioning_20260812.md": "ddm_js3_learned_implicit_conditioning_20260812.md",
    "ddm_js4_pose_null_projected_conditioning_20260812.md": "ddm_js4_pose_null_projected_conditioning_20260812.md",
    "ddm_js5_projector_distilled_conditioning_20260812.md": "ddm_js5_projector_distilled_conditioning_20260812.md",
    "ddm_msr1_manufactured_seg_reduction_20260823.md": "ddm_msr1_manufactured_seg_reduction_20260823.md",
    "ddm_pc1_pose_carrier_efficiency_basis_precision_and_generated_basis_20260905.md": "ddm_pc1_pose_carrier_efficiency_20260905.md",
    "ddm_pc2_carrier_rice_k_width_rank_cut_and_scales_20260908.md": "ddm_pc2_carrier_rice_k_width_rank_cut_and_scales_20260908.md",
    "ddm_pd1_pr_archive_decomposition_priors_20260818.md": "ddm_pd1_pr_archive_decomposition_priors_20260818.md",
    "ddm_qbt2b_r7_lane_constrained_margin_20260828.md": "ddm_qbt2b_r7_lane_constrained_margin_20260828.md",
    "ddm_ra2_carrier_rank_pose_calibration.md": "ddm_ra2_charter_stale_family_closed_and_lossless_axis_20260817.md",
    "ddm_se1_shipping_axis_survival_resolve_20260812.md": "ddm_se1_shipping_axis_survival_resolve_20260812.md",
    "ddm_sj1_multipass_token_predistortion_to_convergence_20260905.md": "ddm_sj1_t4_token_predistortion_pass4_20260909_pointer_move_35_20260909.md",
    "ddm_tac1_two_axis_composition_20260823.md": "ddm_tac1_two_axis_composition_20260823.md",
    "ddm_tba1_token_bit_attribution_20260823.md": "ddm_tba1_token_bit_attribution_20260823.md",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit_and_time(path: Path) -> tuple[str, str]:
    rel = str(path.resolve().relative_to(REPO))
    output = subprocess.check_output(
        ["git", "log", "-1", "--format=%H%x00%cI", "--", rel],
        cwd=REPO,
        text=True,
    ).strip()
    commit, written_at = output.split("\0", 1)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"no durable Git evidence for {rel}")
    return commit, written_at


def queue_latest() -> dict[str, dict]:
    rows = []
    for line in (REPO / ".omx/state/codex_arm_queue.jsonl").read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("name") and row.get("prompt_path"):
            rows.append(row)
    return {row["prompt_path"]: row for row in rows}


def latest_pre_cutoff_final(name: str, cutoff: datetime) -> Path | None:
    candidates = sorted(
        (REPO / ".omx/research/arm_final_messages").glob(f"{name}*.md"), reverse=True
    )
    for path in candidates:
        try:
            _, committed_at = git_commit_and_time(path)
        except (subprocess.CalledProcessError, ValueError):
            continue
        if datetime.fromisoformat(committed_at) < cutoff:
            return path
    return None


def load_queue_module():
    path = REPO / "tools/codex_arm_queue.py"
    spec = importlib.util.spec_from_file_location("ddm_sw1_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--before-census",
        type=Path,
        default=REPO / ".omx/research/ddm_pm2_20260909/screening_census.json",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO
        / ".omx/research/ddm_pm2_20260909/screening_law_exemptions_20260910.jsonl",
    )
    parser.add_argument(
        "--after-census",
        type=Path,
        default=REPO / ".omx/research/ddm_sw1_20260910/screening_census_after.json",
    )
    args = parser.parse_args()
    for output in (args.ledger, args.after_census):
        output.resolve().relative_to(REPO.resolve())

    law_commit_time = subprocess.check_output(
        ["git", "show", "-s", "--format=%cI", LAW_COMMIT], cwd=REPO, text=True
    ).strip()
    cutoff = datetime.fromisoformat(law_commit_time)
    # The linter pins the UTC spelling, not the local-offset spelling.
    from datetime import UTC

    cutoff_utc = cutoff.astimezone(UTC).isoformat()

    before = json.loads(args.before_census.read_text())
    findings = [row for row in before["rows"] if row.get("findings")]
    if len(findings) != before["violating_charters"] or len(findings) != 93:
        raise ValueError("predecessor warning denominator changed; re-adjudicate instead of guessing")
    latest = queue_latest()
    rows = []
    for finding in findings:
        charter = REPO / finding["path"]
        evidence: Path | None = None
        evidence_kind = ""
        queue_row = latest.get(finding["path"])
        terminal_receipt = None
        if queue_row:
            name = queue_row["name"]
            done = REPO / ".omx/tmp/codex_runs" / f"{name}.done"
            if done.exists():
                terminal_receipt = done.read_text(errors="replace").strip().splitlines()[-1]
            if terminal_receipt and terminal_receipt.startswith("rc=0"):
                evidence = latest_pre_cutoff_final(name, cutoff)
                evidence_kind = "queue_final_message"
        override = MEMO_EVIDENCE.get(charter.name)
        if override is not None:
            evidence = REPO / ".omx/research" / override
            evidence_kind = "owning_memo"
        if evidence is None or not evidence.is_file():
            raise ValueError(f"no pre-law terminal evidence for {finding['path']}")
        evidence_commit, finished_at = git_commit_and_time(evidence)
        if datetime.fromisoformat(finished_at) >= cutoff:
            raise ValueError(f"terminal evidence is not pre-law: {evidence}")
        rows.append(
            {
                "schema": "ddm_pm2_screening_law_exemption.v1",
                "classification": "historical_finished_before_law",
                "classification_plain": "historical — FINISHED before the law, exempt by date",
                "charter_path": finding["path"],
                "charter_sha256": sha256(charter),
                "original_finding": finding["findings"][0],
                "evidence_kind": evidence_kind,
                "evidence_path": str(evidence.relative_to(REPO)),
                "evidence_sha256": sha256(evidence),
                "evidence_commit": evidence_commit,
                "finished_at_utc": datetime.fromisoformat(finished_at).astimezone(UTC).isoformat(),
                "terminal_receipt": terminal_receipt,
                "law_cutoff_utc": cutoff_utc,
                "law_landing_commit": LAW_COMMIT,
                "reason": "Exact terminal evidence predates Catalog #416; no future spawn may inherit this historical charter without changing its hash.",
            }
        )
    if len({row["charter_path"] for row in rows}) != 93:
        raise ValueError("exemption paths are not one-to-one")
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )

    queue = load_queue_module()
    after_rows = []
    charters = sorted((REPO / ".omx/research/charters").glob("*.md"))
    for charter in charters:
        result = queue.lint_charter_screening_law(str(charter))
        after_rows.append(
            {
                "path": str(charter.relative_to(REPO)),
                "sha256": sha256(charter),
                "findings": result,
            }
        )
    after = {
        "axis": "[macOS-CPU static apparatus / scorer-free]",
        "charters_scanned": len(after_rows),
        "violating_charters": sum(bool(row["findings"]) for row in after_rows),
        "historical_exemptions": len(rows),
        "classification_counts": {"historical_finished_before_law": len(rows)},
        "strict_default": queue._screening_law_strict(),
        "rows": after_rows,
    }
    args.after_census.parent.mkdir(parents=True, exist_ok=True)
    args.after_census.write_text(json.dumps(after, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: after[key] for key in after if key != "rows"}, indent=2))
    return 0 if after["violating_charters"] == 0 and after["strict_default"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
