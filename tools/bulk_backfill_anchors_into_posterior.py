#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bulk back-fill orphaned authoritative anchors into the continual-learning posterior.

Per XX's posterior validation finding 2026-05-11
(`.omx/research/continual_learning_posterior_validation_20260511.md`): the
continual-learning posterior at ``.omx/state/continual_learning_posterior.json``
contains 6 anchors but custody-OK authoritative anchors exist for ~22 unique
contest-eval results. The orphan gap is a tooling-state artifact rather than a
custody failure: each orphan candidate has its custody intact in the per-result
``contest_auth_eval.json`` file; the gap is that no agent has called
``posterior_update_locked`` on those rows yet.

This tool walks ``experiments/results/{modal_auth_eval,modal_auth_eval_cpu}/``
(plus operator-supplied additional roots), discovers every
``contest_auth_eval.json`` artifact, builds a typed
:class:`tac.continual_learning.ContestResult` from each, validates custody per
Catalog #127 (``ContestResult.validate_custody_verdict``), cross-references with
the existing posterior to identify orphans (i.e. anchors with custody-VALID
state that have not been posterior-updated), and:

- in default ``--dry-run`` mode: prints a structured table + JSONL of every
  would-add anchor with per-anchor SHA + substrate-class + axis + custody
  verdict + canonical_score; produces zero posterior changes;
- with ``--commit`` flag: invokes
  ``tac.continual_learning.posterior_update_locked`` per validated orphan
  inside the canonical fcntl lock per Catalog #128
  (``check_continual_learning_writes_use_lock``), then writes a structured
  JSONL audit log of every back-fill action.

Per CLAUDE.md "Continual-learning posterior — non-negotiable" + Catalog #127 +
Catalog #128: the tool NEVER writes to the posterior outside the canonical
lock; idempotent re-run is safe (duplicate-sha refusal is built in to
``posterior_update_locked``); custody-refused anchors are surfaced as
``custody_refused`` rows with the typed ``refused_class`` taxonomy from
:class:`tac.continual_learning.CustodyVerdict` so the operator can audit
which axes are not 1:1 contest-compliant (for example, Modal CPU instances
do NOT satisfy the ``[contest-CPU]`` GHA-only substrate requirement —
those rows are correctly REFUSED and surfaced as such).

Per CLAUDE.md "Forbidden score claims": this tool NEVER claims a score; every
row carries its own evidence_tag (``[contest-CUDA]`` or ``[contest-CPU]``)
inherited from the source ``contest_auth_eval.json``.

Per CLAUDE.md FORBIDDEN /tmp paths: scratch goes to
``experiments/results/bulk_backfill_<UTC>/`` (when ``--commit`` is used) or
to the operator-specified ``--audit-log-path``.

Per CLAUDE.md "Subagent coherence-by-default" 6-hook wire-in: the tool feeds
the continual-learning posterior_update hook (#5); the orphan-table output
feeds the sensitivity-map (#1) and Pareto-frontier (#2) updates by enabling
per-architecture-class drift posterior estimation; the verdict format
plugs into the cathedral autopilot dispatch hook (#4) by enriching the
per-class drift profile the autopilot's CandidateRow predictor consumes.

CLAUDE.md compliance tags:
- ``operator_gate_non_negotiable_at_every_dispatch``
- ``no_score_claim_only_predicted_band``
- ``custody_validator_aware``
- ``continual_learning_posterior_aware``
- ``halt_and_ask_default_on``
- ``no_tmp_paths``
- ``research_only=false``
- ``lane_class=substrate_engineering``

Cross-references
----------------
- ``feedback_phase1_cheap_config_dashboard_posterior_validation_landed_20260511.md``
- ``.omx/research/continual_learning_posterior_validation_20260511.md``
- :mod:`tac.continual_learning`
- :func:`tac.continual_learning.posterior_update_locked`
- :func:`tac.continual_learning.posterior_update_locked_from_auth_eval_json`
- :func:`tac.continual_learning.contest_result_from_auth_eval_payload`
- ``tools/cathedral_autopilot_autonomous_loop.py``
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.continual_learning import (  # noqa: E402  (after path bootstrap)
    ContestResult,
    contest_result_from_auth_eval_payload,
    DEFAULT_POSTERIOR_PATH,
    load_posterior,
    posterior_update_locked,
)


SCHEMA = "tac_bulk_backfill_anchors_into_posterior_v1"
DEFAULT_SEARCH_ROOTS = (
    "experiments/results/modal_auth_eval",
    "experiments/results/modal_auth_eval_cpu",
)


# ── Architecture-class inference ──────────────────────────────────────────────


# Tokens mapped to canonical architecture_class labels. Labels are kept in
# sync with the labels XX surfaced in the posterior-validation ledger so the
# back-fill produces architecture_class values the dashboard already
# recognizes.
#
# Order matters: the LONGEST/most-specific token wins. For example,
# ``pr106_latent_sidecar_r2_pr101_grammar`` is preferred over
# ``pr106_latent_sidecar_r2`` which is preferred over ``pr106``.
_ARCH_CLASS_TOKENS: tuple[tuple[str, str], ...] = (
    # PR106-family lateral leaps (highest specificity first).
    (
        "pr106_latent_sidecar_r2_pr101_grammar",
        "lane_pr106_latent_sidecar_r2_pr101_grammar",
    ),
    ("pr106_latent_sidecar_r2", "lane_pr106_latent_sidecar_r2"),
    ("pr106_latent_sidecar", "lane_pr106_latent_sidecar_r1"),
    # L2 sparse-aware family.
    (
        "lane_c3_residual_pr106_sidecar_l2_sparse_aware_dispatch",
        "lane_c3_residual_pr106_sidecar_l2_sparse_aware_dispatch",
    ),
    # L1 family extensions.
    (
        "lane_c3_residual_pr106_sidecar",
        "lane_c3_residual_pr106_sidecar_dispatch_ready",
    ),
    (
        "lane_wavelet_residual_pr106_sidecar",
        "lane_wavelet_residual_pr106_sidecar_dispatch_ready",
    ),
    (
        "lane_cool_chic_residual_pr106_sidecar",
        "lane_cool_chic_residual_pr106_sidecar_dispatch_ready",
    ),
    (
        "lane_coord_mlp_residual_pr106_sidecar",
        "lane_coord_mlp_residual_pr106_sidecar_dispatch_ready",
    ),
    (
        "lane_siren_residual_pr106_sidecar",
        "lane_siren_residual_pr106_sidecar_dispatch_ready",
    ),
    # PR103-on-PR106 cross-substrate.
    ("pr103_pr106", "pr103_on_pr106"),
    # PR101 / PR103 lossy-coarsening + arithmetic-coding families.
    ("pr101_a5", "pr101_lossy_coarsening"),
    ("pr101_bias_refine", "pr101_lossy_coarsening"),
    ("pr101_kaggle_proxy", "pr101_lossy_coarsening"),
    ("pr101", "pr101_lossy_coarsening"),
    ("pr103_global_combo", "pr103_arithmetic_coding"),
    ("pr103_histogram", "pr103_arithmetic_coding"),
    ("pr103_source_same_runtime", "pr103_arithmetic_coding"),
    ("pr103_ac_hidden_gem", "pr103_arithmetic_coding"),
    ("pr103", "pr103_arithmetic_coding"),
    # A1 HNeRV-FT-microcodec.
    ("a1_dual_eval", "hnerv_ft_microcodec"),
    ("a1_dual_cuda", "hnerv_ft_microcodec"),
    ("a1_modal", "hnerv_ft_microcodec"),
    ("a1_", "hnerv_ft_microcodec"),
    # PR106 quantization-only sweep.
    ("pr106_q10", "pr106_quantization_sweep"),
)


def infer_architecture_class(result_dir_name: str) -> str:
    """Infer the canonical ``architecture_class`` label from the result-dir name.

    Per the posterior-validation ledger, architecture_class labels follow XX's
    canonical taxonomy (``hnerv_ft_microcodec`` /
    ``lane_pr106_latent_sidecar_r2`` / ``pr101_lossy_coarsening`` etc.). When no
    mapping matches the directory token, returns
    ``f"unknown__{result_dir_name}"`` so the operator can review and add a
    new mapping.
    """
    name_lower = result_dir_name.lower()
    for token, arch_class in _ARCH_CLASS_TOKENS:
        if token.lower() in name_lower:
            return arch_class
    return f"unknown__{result_dir_name}"


# ── Discovered-anchor record ──────────────────────────────────────────────────


@dataclass
class DiscoveredAnchor:
    """One auth-eval JSON discovered on disk + its derived custody verdict."""

    auth_eval_path: Path
    architecture_class: str
    contest_result: Optional[ContestResult]
    parse_error: Optional[str]
    custody_accepted: bool
    custody_refused_class: Optional[str]
    custody_reason: str
    in_posterior_already: bool
    is_promotable_orphan: bool

    @property
    def axis(self) -> str:
        return self.contest_result.axis if self.contest_result else ""

    @property
    def evidence_tag(self) -> str:
        return self.contest_result.evidence_tag if self.contest_result else ""

    @property
    def archive_sha256(self) -> str:
        return self.contest_result.archive_sha256 if self.contest_result else ""

    @property
    def archive_bytes(self) -> int:
        return self.contest_result.archive_bytes if self.contest_result else 0

    @property
    def score_value(self) -> float:
        return self.contest_result.score_value if self.contest_result else float("nan")

    @property
    def hardware_substrate(self) -> str:
        return self.contest_result.hardware_substrate if self.contest_result else ""


# ── Discovery ─────────────────────────────────────────────────────────────────


def discover_auth_eval_paths(
    repo_root: Path,
    search_roots: Iterable[str] = DEFAULT_SEARCH_ROOTS,
) -> list[Path]:
    """Walk every search-root for ``contest_auth_eval.json`` files.

    Excludes ``contest_auth_eval.adjudicated.json`` siblings (those are
    derived adjudicator output, not the canonical authoritative artifact).
    """
    out: list[Path] = []
    for raw in search_roots:
        root = (repo_root / raw).resolve()
        if not root.exists():
            continue
        for path in sorted(root.rglob("contest_auth_eval.json")):
            if path.name == "contest_auth_eval.json":
                out.append(path)
    return out


def existing_posterior_anchor_keys(
    posterior_path: Optional[Path] = None,
) -> set[tuple[str, str]]:
    """Return the set of ``(archive_sha256, axis)`` pairs already in posterior."""
    p = posterior_path or DEFAULT_POSTERIOR_PATH
    if not p.exists():
        return set()
    posterior = load_posterior(p)
    out: set[tuple[str, str]] = set()
    for h in posterior.accepted_anchor_history:
        sha = h.get("archive_sha256")
        axis = h.get("axis")
        if isinstance(sha, str) and isinstance(axis, str) and sha and axis:
            out.add((sha, axis))
    return out


def discover_anchors(
    repo_root: Path,
    search_roots: Iterable[str] = DEFAULT_SEARCH_ROOTS,
    posterior_path: Optional[Path] = None,
) -> list[DiscoveredAnchor]:
    """Discover every auth-eval artifact + classify its posterior status."""
    paths = discover_auth_eval_paths(repo_root, search_roots)
    existing_keys = existing_posterior_anchor_keys(posterior_path)
    anchors: list[DiscoveredAnchor] = []
    for path in paths:
        anchors.append(_classify_one(path, existing_keys))
    return anchors


def _classify_one(
    auth_eval_path: Path,
    existing_keys: set[tuple[str, str]],
) -> DiscoveredAnchor:
    """Build a :class:`DiscoveredAnchor` for one file (no posterior writes)."""
    arch_class = infer_architecture_class(auth_eval_path.parent.name)

    # Parse the auth-eval JSON; handle corrupt/missing fields gracefully.
    try:
        payload = json.loads(auth_eval_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return DiscoveredAnchor(
            auth_eval_path=auth_eval_path,
            architecture_class=arch_class,
            contest_result=None,
            parse_error=f"JSONDecodeError: {exc}",
            custody_accepted=False,
            custody_refused_class="missing_metadata",
            custody_reason=f"could not parse {auth_eval_path}: {exc}",
            in_posterior_already=False,
            is_promotable_orphan=False,
        )

    try:
        result = contest_result_from_auth_eval_payload(
            payload,
            architecture_class=arch_class,
            source_path=auth_eval_path,
            notes=f"bulk-backfill discovery from {auth_eval_path}",
        )
    except (TypeError, ValueError, KeyError) as exc:
        return DiscoveredAnchor(
            auth_eval_path=auth_eval_path,
            architecture_class=arch_class,
            contest_result=None,
            parse_error=f"{type(exc).__name__}: {exc}",
            custody_accepted=False,
            custody_refused_class="missing_metadata",
            custody_reason=f"could not build ContestResult: {exc}",
            in_posterior_already=False,
            is_promotable_orphan=False,
        )

    verdict = result.validate_custody_verdict()
    in_posterior = (result.archive_sha256, result.axis) in existing_keys
    is_orphan = verdict.accepted and not in_posterior

    return DiscoveredAnchor(
        auth_eval_path=auth_eval_path,
        architecture_class=arch_class,
        contest_result=result,
        parse_error=None,
        custody_accepted=verdict.accepted,
        custody_refused_class=verdict.refused_class,
        custody_reason=verdict.reason,
        in_posterior_already=in_posterior,
        is_promotable_orphan=is_orphan,
    )


# ── Reporting ────────────────────────────────────────────────────────────────


def summarize_anchors(anchors: list[DiscoveredAnchor]) -> dict[str, Any]:
    """Build a summary dict of the discovery + classification."""
    total = len(anchors)
    parse_errors = sum(1 for a in anchors if a.parse_error)
    custody_accepted = sum(1 for a in anchors if a.custody_accepted)
    custody_refused = sum(1 for a in anchors if not a.custody_accepted)
    in_posterior = sum(1 for a in anchors if a.in_posterior_already)
    promotable_orphans = sum(1 for a in anchors if a.is_promotable_orphan)
    refused_by_class: dict[str, int] = {}
    for a in anchors:
        if not a.custody_accepted and a.custody_refused_class:
            refused_by_class[a.custody_refused_class] = (
                refused_by_class.get(a.custody_refused_class, 0) + 1
            )
    by_axis: dict[str, int] = {}
    for a in anchors:
        if a.is_promotable_orphan:
            by_axis[a.axis] = by_axis.get(a.axis, 0) + 1
    by_arch_class: dict[str, int] = {}
    for a in anchors:
        if a.is_promotable_orphan:
            by_arch_class[a.architecture_class] = (
                by_arch_class.get(a.architecture_class, 0) + 1
            )
    return {
        "total_artifacts_discovered": total,
        "parse_errors": parse_errors,
        "custody_accepted": custody_accepted,
        "custody_refused": custody_refused,
        "custody_refused_by_class": refused_by_class,
        "already_in_posterior": in_posterior,
        "promotable_orphans": promotable_orphans,
        "promotable_orphans_by_axis": by_axis,
        "promotable_orphans_by_architecture_class": by_arch_class,
    }


def anchor_to_jsonl_row(anchor: DiscoveredAnchor) -> dict[str, Any]:
    """Serialize one anchor into a JSON-safe dict for the audit log."""
    return {
        "auth_eval_path": str(anchor.auth_eval_path),
        "architecture_class": anchor.architecture_class,
        "axis": anchor.axis,
        "evidence_tag": anchor.evidence_tag,
        "archive_sha256": anchor.archive_sha256,
        "archive_bytes": anchor.archive_bytes,
        "score_value": anchor.score_value,
        "hardware_substrate": anchor.hardware_substrate,
        "custody_accepted": anchor.custody_accepted,
        "custody_refused_class": anchor.custody_refused_class,
        "custody_reason": anchor.custody_reason,
        "in_posterior_already": anchor.in_posterior_already,
        "is_promotable_orphan": anchor.is_promotable_orphan,
        "parse_error": anchor.parse_error,
    }


def render_dry_run_table(anchors: list[DiscoveredAnchor]) -> str:
    """Render a fixed-width table of all anchors for human review."""
    cols = (
        "axis",
        "arch_class",
        "sha16",
        "score",
        "tag",
        "status",
        "result_dir",
    )
    rows: list[tuple[str, ...]] = [cols]
    for a in anchors:
        if a.in_posterior_already:
            status = "in_posterior"
        elif a.is_promotable_orphan:
            status = "ORPHAN_PROMOTABLE"
        elif a.parse_error:
            status = f"parse_error:{a.parse_error[:24]}"
        else:
            status = f"refused:{a.custody_refused_class or 'unknown'}"
        rows.append(
            (
                a.axis or "?",
                (a.architecture_class[:42] + "...") if len(a.architecture_class) > 45 else a.architecture_class,
                (a.archive_sha256[:16] if a.archive_sha256 else "?"),
                f"{a.score_value:.5f}" if a.contest_result is not None else "?",
                a.evidence_tag or "?",
                status,
                a.auth_eval_path.parent.name[:50],
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]
    out: list[str] = []
    for r in rows:
        out.append("  ".join(r[i].ljust(widths[i]) for i in range(len(cols))))
    return "\n".join(out)


# ── Commit (transactional back-fill) ─────────────────────────────────────────


def commit_orphans_to_posterior(
    anchors: list[DiscoveredAnchor],
    *,
    audit_log_path: Path,
    posterior_path: Optional[Path] = None,
    lock_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Transactionally back-fill every promotable-orphan anchor.

    Per CLAUDE.md "Continual-learning posterior — non-negotiable" + Catalog
    #128: writes go through ``posterior_update_locked`` exclusively (fcntl
    LOCK_EX on the canonical lock-path); idempotent re-run is safe.

    Writes one JSONL row per attempted back-fill to ``audit_log_path`` so the
    operator has a forensic record.
    """
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    refused = 0
    skipped_already_in_posterior = 0
    skipped_custody_refused = 0
    rows: list[dict[str, Any]] = []
    started_at = _dt.datetime.now(_dt.UTC).isoformat()

    for anchor in anchors:
        base_row = anchor_to_jsonl_row(anchor)
        if anchor.in_posterior_already:
            skipped_already_in_posterior += 1
            base_row["action"] = "skipped_already_in_posterior"
            rows.append(base_row)
            continue
        if not anchor.custody_accepted or anchor.contest_result is None:
            skipped_custody_refused += 1
            base_row["action"] = "skipped_custody_refused"
            rows.append(base_row)
            continue
        update = posterior_update_locked(
            anchor.contest_result,
            posterior_path=posterior_path,
            lock_path=lock_path,
        )
        base_row["action"] = "posterior_update_locked_called"
        base_row["posterior_update_accepted"] = update.accepted
        base_row["posterior_update_refusal_reason"] = update.refusal_reason
        base_row["posterior_n_anchors_after"] = update.posterior_n_anchors_after
        if update.accepted:
            accepted += 1
        else:
            refused += 1
        rows.append(base_row)

    completed_at = _dt.datetime.now(_dt.UTC).isoformat()
    with audit_log_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "schema": SCHEMA,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "audit_log_path": str(audit_log_path),
        "accepted": accepted,
        "refused": refused,
        "skipped_already_in_posterior": skipped_already_in_posterior,
        "skipped_custody_refused": skipped_custody_refused,
        "rows": rows,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repo root (default: detected from this script's location).",
    )
    parser.add_argument(
        "--search-root",
        action="append",
        default=None,
        help=(
            "Additional search root for contest_auth_eval.json files (relative "
            "to --repo-root). May be passed multiple times. Default: "
            f"{list(DEFAULT_SEARCH_ROOTS)}."
        ),
    )
    parser.add_argument(
        "--posterior-path",
        type=Path,
        default=None,
        help="Path to continual_learning_posterior.json (default: canonical).",
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=None,
        help="Path to posterior fcntl lock (default: canonical).",
    )
    parser.add_argument(
        "--audit-log-path",
        type=Path,
        default=None,
        help=(
            "Path to JSONL audit log (required when --commit). Operator must "
            "supply an EXPLICIT path; /tmp paths are FORBIDDEN per CLAUDE.md."
        ),
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help=(
            "When set, invoke posterior_update_locked per validated orphan. "
            "Default OFF — dry-run produces a structured table + JSONL summary "
            "without modifying the posterior."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help=(
            "Optional path to write a structured summary JSON of the run "
            "(works in both dry-run and commit modes)."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-anchor table; only print the summary block.",
    )
    args = parser.parse_args(argv)

    search_roots = tuple(args.search_root) if args.search_root else DEFAULT_SEARCH_ROOTS

    anchors = discover_anchors(
        args.repo_root,
        search_roots=search_roots,
        posterior_path=args.posterior_path,
    )
    summary = summarize_anchors(anchors)

    if args.commit:
        if args.audit_log_path is None:
            print(
                "bulk_backfill_anchors_into_posterior: --commit requires "
                "--audit-log-path (per CLAUDE.md no-/tmp-path)",
                file=sys.stderr,
            )
            return 2
        path_str = str(args.audit_log_path)
        if path_str.startswith("/tmp/") or path_str.startswith("/var/tmp/"):
            print(
                "bulk_backfill_anchors_into_posterior: --audit-log-path must "
                "NOT live under /tmp; pick a durable evidence path "
                "(e.g. experiments/results/bulk_backfill_<UTC>/audit.jsonl).",
                file=sys.stderr,
            )
            return 2
        commit_result = commit_orphans_to_posterior(
            anchors,
            audit_log_path=args.audit_log_path,
            posterior_path=args.posterior_path,
            lock_path=args.lock_path,
        )
        if not args.quiet:
            print(render_dry_run_table(anchors))
            print()
        print("=== bulk back-fill commit summary ===")
        for k in (
            "accepted",
            "refused",
            "skipped_already_in_posterior",
            "skipped_custody_refused",
            "audit_log_path",
        ):
            print(f"  {k}: {commit_result[k]}")
        if args.summary_json is not None:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "mode": "commit",
                        "search_roots": list(search_roots),
                        "discovery_summary": summary,
                        "commit_result": {
                            k: v for k, v in commit_result.items() if k != "rows"
                        },
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return 0

    # Dry-run path.
    if not args.quiet:
        print(render_dry_run_table(anchors))
        print()
    print("=== bulk back-fill dry-run summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "mode": "dry_run",
                    "search_roots": list(search_roots),
                    "discovery_summary": summary,
                    "anchors": [anchor_to_jsonl_row(a) for a in anchors],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
