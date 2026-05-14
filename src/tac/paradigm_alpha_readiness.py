# SPDX-License-Identifier: MIT
"""Deterministic readiness audit for Paradigm-alpha mask codec candidates.

The April 30 mask-overhaul audit is now stale in important places: wavelet and
VQ-style mask codec modules exist in this checkout, but runtime/archive
integration is still uneven. This module keeps that distinction mechanical.

The audit is local-only planning evidence. It never unlocks exact-eval dispatch
and never makes a score claim.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.hidden_gems import all_hidden_gems
from tac.repo_io import read_json, sha256_file

SCHEMA_VERSION = 1
AUDIT_NAME = "paradigm_alpha_mask_overhaul_readiness"
DISPATCH_BLOCKER_AUDIT_ONLY = "paradigm_alpha_readiness_not_exact_eval_dispatch_evidence"


@dataclass(frozen=True)
class PathGroup:
    """One named group of repo-relative paths used by a candidate."""

    label: str
    paths: tuple[str, ...]
    required_for_core: bool = False


@dataclass(frozen=True)
class TextRequirement:
    """A source-string requirement that proves a runtime hook is present."""

    label: str
    path: str
    needle: str
    blocker: str


@dataclass(frozen=True)
class AlphaCandidateDefinition:
    """Static definition of one alpha mask-overhaul candidate."""

    key: str
    title: str
    component: str
    path_groups: tuple[PathGroup, ...]
    text_requirements: tuple[TextRequirement, ...]
    hidden_gem_keys: tuple[str, ...]
    next_patch: str


@dataclass(frozen=True)
class PathProbe:
    """Stable probe for one repo-relative file or directory."""

    path: str
    exists: bool
    kind: str
    bytes: int | None
    sha256: str | None


@dataclass(frozen=True)
class TextProbe:
    """Stable probe for a required source-string hook."""

    label: str
    path: str
    needle: str
    found: bool
    blocker: str


@dataclass(frozen=True)
class AlphaCandidateReadiness:
    """Readiness row for one Paradigm-alpha candidate."""

    key: str
    title: str
    component: str
    hidden_gem_keys: tuple[str, ...]
    hidden_gem_registry_status: tuple[str, ...]
    components_on_disk: tuple[str, ...]
    missing_core_components: tuple[str, ...]
    readiness_status: str
    eligible_for_local_patch: bool
    ready_for_exact_eval_dispatch: bool
    local_blockers: tuple[str, ...]
    dispatch_blockers: tuple[str, ...]
    path_groups: dict[str, tuple[PathProbe, ...]]
    text_requirements: tuple[TextProbe, ...]
    empirical_summary: dict[str, Any]
    next_patch: str


_ALPHA_REAL_REPORT = "reports/paradigm_alpha_real_archive.json"


_CANDIDATES: tuple[AlphaCandidateDefinition, ...] = (
    AlphaCandidateDefinition(
        key="alpha1_nerv",
        title="Lane 12 NeRV coordinate-MLP mask codec",
        component="nerv_mask_codec",
        path_groups=(
            PathGroup("codec_module", ("src/tac/nerv_mask_codec.py",), True),
            PathGroup("trainer", ("experiments/train_nerv_mask.py",), True),
            PathGroup(
                "tests",
                (
                    "src/tac/tests/test_nerv_mask_codec.py",
                    "src/tac/tests/test_lane12_nerv_dependency_closure.py",
                    "src/tac/tests/test_preflight_nerv_codec_discipline.py",
                ),
                True,
            ),
            PathGroup("runtime_inflate", ("submissions/robust_current/inflate_renderer.py",), True),
            PathGroup("remote_dispatch", ("scripts/remote_lane_nerv.sh",), True),
            PathGroup("empirical_report", ("reports/lane_12_nerv_real_archive.json",), True),
            PathGroup("l2_clearance", (".omx/state/lane12_nerv_l2_clearance.json",), False),
        ),
        text_requirements=(
            TextRequirement(
                label="runtime_accepts_nrv1",
                path="submissions/robust_current/inflate_renderer.py",
                needle="NRV1",
                blocker="runtime_missing_nrv1_loader",
            ),
        ),
        hidden_gem_keys=("nerv_mask_l2_readiness",),
        next_patch=(
            "Produce a geometry-passing Lane 12 candidate and write a real "
            "lane12_nerv_l2_clearance packet before any retraining or exact eval."
        ),
    ),
    AlphaCandidateDefinition(
        key="alpha2_wavelet",
        title="Wavelet mask codec",
        component="wavelet_mask_codec",
        path_groups=(
            PathGroup("codec_module", ("src/tac/wavelet_mask_codec.py",), True),
            PathGroup("tests", ("src/tac/tests/test_wavelet_mask_codec.py",), True),
            PathGroup("real_archive_evaluator", ("experiments/paradigm_alpha_real_archive_eval.py",), True),
            PathGroup("empirical_report", (_ALPHA_REAL_REPORT,), False),
            PathGroup("runtime_inflate_target", ("submissions/robust_current/inflate_renderer.py",), True),
        ),
        text_requirements=(
            TextRequirement(
                label="runtime_accepts_wmc1",
                path="submissions/robust_current/inflate_renderer.py",
                needle="WMC1",
                blocker="runtime_missing_wmc1_loader",
            ),
        ),
        hidden_gem_keys=("wavelet_residual_basis_gate",),
        next_patch=(
            "Do not wire runtime first: replace the byte-negative real-archive "
            "profile with a byte-positive codec configuration or residual-only "
            "manifest, then add WMC1 runtime consumption."
        ),
    ),
    AlphaCandidateDefinition(
        key="alpha3_vqvae",
        title="VQ-VAE patch-codebook mask codec",
        component="vqvae_mask_codec",
        path_groups=(
            PathGroup("codec_module", ("src/tac/vqvae_mask_codec.py",), True),
            PathGroup("tests", ("src/tac/tests/test_vqvae_mask_codec.py",), True),
            PathGroup("real_archive_evaluator", ("experiments/paradigm_alpha_real_archive_eval.py",), True),
            PathGroup("empirical_report", (_ALPHA_REAL_REPORT,), False),
            PathGroup("runtime_inflate_target", ("submissions/robust_current/inflate_renderer.py",), True),
        ),
        text_requirements=(
            TextRequirement(
                label="runtime_accepts_vqm1",
                path="submissions/robust_current/inflate_renderer.py",
                needle="VQM1",
                blocker="runtime_missing_vqm1_loader",
            ),
        ),
        hidden_gem_keys=(),
        next_patch=(
            "Generate a real-archive VQM1 empirical row with codebook bytes, "
            "token-stream bytes, and decoded-mask disagreement before runtime wiring."
        ),
    ),
    AlphaCandidateDefinition(
        key="alpha4_grayscale_lut",
        title="Selfcomp grayscale-LUT mask stream",
        component="mask_grayscale_lut",
        path_groups=(
            PathGroup("codec_module", ("src/tac/mask_grayscale_lut.py",), True),
            PathGroup("tests", ("src/tac/tests/test_mask_grayscale_lut.py",), True),
            PathGroup("archive_builder", ("experiments/build_lane_mm_archive.py",), True),
            PathGroup("runtime_inflate", ("submissions/robust_current/inflate_renderer_grayscale.py",), True),
            PathGroup("shell_dispatch", ("submissions/robust_current/inflate.sh",), True),
            PathGroup("remote_dispatch", ("scripts/remote_lane_mm_grayscale_lut.sh",), True),
            PathGroup("preflight_guard", ("src/tac/preflight.py",), True),
            PathGroup("empirical_report", (_ALPHA_REAL_REPORT,), False),
        ),
        text_requirements=(
            TextRequirement(
                label="inflate_dispatches_renderer_grayscale",
                path="submissions/robust_current/inflate.sh",
                needle="PYTHON_INFLATE=\"renderer_grayscale\"",
                blocker="inflate_missing_renderer_grayscale_dispatch",
            ),
            TextRequirement(
                label="preflight_checks_grayscale_lut_consistency",
                path="src/tac/preflight.py",
                needle="check_segmap_grayscale_lut_consistency",
                blocker="preflight_missing_grayscale_lut_consistency",
            ),
        ),
        hidden_gem_keys=(),
        next_patch=(
            "Refresh alpha4 through a current real-archive empirical row and an "
            "exact CUDA archive candidate that proves train/inflate grayscale "
            "parity; previous encoder-only evidence is not sufficient."
        ),
    ),
)


def audit_paradigm_alpha(
    *,
    repo_root: Path | str,
    candidates: Iterable[AlphaCandidateDefinition] | None = None,
) -> tuple[AlphaCandidateReadiness, ...]:
    """Audit alpha candidates against the current checkout."""

    root = Path(repo_root)
    rows = tuple(_CANDIDATES if candidates is None else candidates)
    hidden_status = {entry.key: entry.status for entry in all_hidden_gems()}
    return tuple(_audit_candidate(root, row, hidden_status) for row in rows)


def readiness_payload(
    *,
    repo_root: Path | str,
    candidates: Iterable[AlphaCandidateDefinition] | None = None,
) -> dict[str, Any]:
    """Return a JSON-stable alpha readiness report."""

    rows = audit_paradigm_alpha(repo_root=repo_root, candidates=candidates)
    status_counts = Counter(row.readiness_status for row in rows)
    return {
        "audit": AUDIT_NAME,
        "schema_version": SCHEMA_VERSION,
        "entries": [alpha_readiness_to_dict(row) for row in rows],
        "summary": {
            "candidate_count": len(rows),
            "eligible_for_local_patch_count": sum(row.eligible_for_local_patch for row in rows),
            "missing_core_component_count": sum(len(row.missing_core_components) for row in rows),
            "readiness_status_counts": dict(sorted(status_counts.items())),
            "ready_for_exact_eval_dispatch_count": sum(row.ready_for_exact_eval_dispatch for row in rows),
        },
    }


def alpha_readiness_to_dict(row: AlphaCandidateReadiness) -> dict[str, Any]:
    """Convert one alpha readiness row to a deterministic JSON mapping."""

    return {
        "component": row.component,
        "components_on_disk": list(row.components_on_disk),
        "dispatch_blockers": list(row.dispatch_blockers),
        "eligible_for_local_patch": row.eligible_for_local_patch,
        "empirical_summary": row.empirical_summary,
        "hidden_gem_keys": list(row.hidden_gem_keys),
        "hidden_gem_registry_status": list(row.hidden_gem_registry_status),
        "key": row.key,
        "local_blockers": list(row.local_blockers),
        "missing_core_components": list(row.missing_core_components),
        "next_patch": row.next_patch,
        "path_groups": {
            key: [path_probe_to_dict(probe) for probe in probes]
            for key, probes in sorted(row.path_groups.items())
        },
        "readiness_status": row.readiness_status,
        "ready_for_exact_eval_dispatch": row.ready_for_exact_eval_dispatch,
        "text_requirements": [text_probe_to_dict(probe) for probe in row.text_requirements],
        "title": row.title,
    }


def path_probe_to_dict(row: PathProbe) -> dict[str, Any]:
    """Convert a path probe to a deterministic JSON mapping."""

    return {
        "bytes": row.bytes,
        "exists": row.exists,
        "kind": row.kind,
        "path": row.path,
        "sha256": row.sha256,
    }


def text_probe_to_dict(row: TextProbe) -> dict[str, Any]:
    """Convert a source-string probe to a deterministic JSON mapping."""

    return {
        "blocker": row.blocker,
        "found": row.found,
        "label": row.label,
        "needle": row.needle,
        "path": row.path,
    }


def render_markdown(rows: Iterable[AlphaCandidateReadiness]) -> str:
    """Render alpha readiness rows as deterministic markdown."""

    audited = tuple(rows)
    lines = [
        "# Paradigm-Alpha Mask Readiness",
        "",
        "This audit checks local code, tests, runtime hooks, and empirical rows. "
        "It never unlocks exact-eval dispatch.",
        "",
        "| candidate | on-disk components | readiness | local blockers | dispatch blockers | next patch |",
        "|---|---|---|---|---|---|",
    ]
    if not audited:
        lines.append("| _none_ | _none_ | _none_ | _none_ | _none_ | _none_ |")
    for row in audited:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{_markdown_cell(row.key)}`",
                    _markdown_list(row.components_on_disk),
                    f"`{_markdown_cell(row.readiness_status)}`",
                    _markdown_list(row.local_blockers),
                    _markdown_list(row.dispatch_blockers),
                    _markdown_cell(row.next_patch),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _audit_candidate(
    root: Path,
    candidate: AlphaCandidateDefinition,
    hidden_status: dict[str, str],
) -> AlphaCandidateReadiness:
    path_groups = {
        group.label: tuple(_probe_path(root, path) for path in group.paths)
        for group in candidate.path_groups
    }
    components_on_disk = tuple(
        group.label
        for group in candidate.path_groups
        if all(probe.exists for probe in path_groups[group.label])
    )
    missing_core = tuple(
        group.label
        for group in candidate.path_groups
        if group.required_for_core and not all(probe.exists for probe in path_groups[group.label])
    )
    text_probes = tuple(_probe_text(root, req) for req in candidate.text_requirements)
    text_blockers = tuple(probe.blocker for probe in text_probes if not probe.found)
    empirical_summary, empirical_blockers = _empirical_summary(root, candidate.key)
    static_blockers = _static_local_blockers(root, candidate.key)
    local_blockers = tuple(dict.fromkeys((*missing_core, *text_blockers, *empirical_blockers, *static_blockers)))
    dispatch_blockers = tuple(
        dict.fromkeys((DISPATCH_BLOCKER_AUDIT_ONLY, *_dispatch_blockers(root, candidate.key)))
    )
    readiness_status = _readiness_status(
        missing_core=missing_core,
        text_blockers=text_blockers,
        empirical_blockers=empirical_blockers,
        static_blockers=static_blockers,
    )
    return AlphaCandidateReadiness(
        key=candidate.key,
        title=candidate.title,
        component=candidate.component,
        hidden_gem_keys=candidate.hidden_gem_keys,
        hidden_gem_registry_status=tuple(
            f"{key}:{hidden_status.get(key, 'missing_registry_entry')}"
            for key in candidate.hidden_gem_keys
        ),
        components_on_disk=components_on_disk,
        missing_core_components=missing_core,
        readiness_status=readiness_status,
        eligible_for_local_patch=not missing_core,
        ready_for_exact_eval_dispatch=False,
        local_blockers=local_blockers,
        dispatch_blockers=dispatch_blockers,
        path_groups=path_groups,
        text_requirements=text_probes,
        empirical_summary=empirical_summary,
        next_patch=candidate.next_patch,
    )


def _probe_path(root: Path, relpath: str) -> PathProbe:
    path = root / relpath
    if path.is_file():
        return PathProbe(
            path=relpath,
            exists=True,
            kind="file",
            bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
    if path.is_dir():
        return PathProbe(path=relpath, exists=True, kind="dir", bytes=None, sha256=None)
    if path.exists():
        return PathProbe(path=relpath, exists=True, kind="other", bytes=None, sha256=None)
    return PathProbe(path=relpath, exists=False, kind="missing", bytes=None, sha256=None)


def _probe_text(root: Path, req: TextRequirement) -> TextProbe:
    path = root / req.path
    found = False
    if path.is_file():
        found = req.needle in path.read_text(encoding="utf-8", errors="replace")
    return TextProbe(
        label=req.label,
        path=req.path,
        needle=req.needle,
        found=found,
        blocker=req.blocker,
    )


def _empirical_summary(root: Path, key: str) -> tuple[dict[str, Any], tuple[str, ...]]:
    if key == "alpha1_nerv":
        return _nerv_empirical_summary(root)
    if key in {"alpha2_wavelet", "alpha3_vqvae", "alpha4_grayscale_lut"}:
        report_key = {
            "alpha2_wavelet": "alpha2_wavelet",
            "alpha3_vqvae": "alpha3_vqvae",
            "alpha4_grayscale_lut": "alpha4_grayscale_lut",
        }[key]
        return _alpha_real_empirical_summary(root, report_key)
    return {}, ()


def _nerv_empirical_summary(root: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    path = root / "reports/lane_12_nerv_real_archive.json"
    if not path.is_file():
        return {}, ("missing_nerv_empirical_report",)
    data = _read_json_mapping(path)
    baseline = _safe_int(data.get("baseline_av1_bytes"))
    fp16 = _safe_int(data.get("nerv_fp16_bytes"))
    int8 = _safe_int(data.get("nerv_int8_bytes"))
    best = min(value for value in (fp16, int8) if value is not None) if fp16 or int8 else None
    summary = {
        "baseline_av1_bytes": baseline,
        "best_payload_bytes": best,
        "final_argmax_disagreement_vs_av1_source": data.get(
            "final_argmax_disagreement_vs_av1_source"
        ),
        "report_path": "reports/lane_12_nerv_real_archive.json",
    }
    blockers: list[str] = []
    if baseline is not None and best is not None and best >= baseline:
        blockers.append("nerv_empirical_bytes_not_better_than_baseline")
    return summary, tuple(blockers)


def _alpha_real_empirical_summary(root: Path, report_key: str) -> tuple[dict[str, Any], tuple[str, ...]]:
    path = root / _ALPHA_REAL_REPORT
    if not path.is_file():
        return {}, ("missing_paradigm_alpha_real_archive_report",)
    data = _read_json_mapping(path)
    candidates = data.get("candidates")
    if not isinstance(candidates, dict) or report_key not in candidates:
        return {
            "report_path": _ALPHA_REAL_REPORT,
            "available_candidates": sorted(candidates) if isinstance(candidates, dict) else [],
        }, (f"missing_real_archive_empirical_row_{report_key}",)
    row = candidates[report_key]
    if not isinstance(row, dict):
        return {"report_path": _ALPHA_REAL_REPORT}, (f"invalid_real_archive_empirical_row_{report_key}",)
    baseline = _safe_int(data.get("baseline_av1_bytes"))
    encoded = _safe_int(row.get("encoded_bytes"))
    summary = {
        "argmax_disagreement_vs_source": row.get("argmax_disagreement_vs_source"),
        "argmax_agreement_vs_source": row.get("argmax_agreement_vs_source"),
        "baseline_av1_bytes": baseline,
        "encoded_bytes": encoded,
        "pct_savings_vs_av1": row.get("pct_savings_vs_av1"),
        "report_path": _ALPHA_REAL_REPORT,
    }
    blockers: list[str] = []
    if baseline is not None and encoded is not None and encoded >= baseline:
        blockers.append(f"empirical_bytes_not_better_than_baseline_{report_key}")
    return summary, tuple(blockers)


def _static_local_blockers(root: Path, key: str) -> tuple[str, ...]:
    blockers: list[str] = []
    if key == "alpha1_nerv" and not (root / ".omx/state/lane12_nerv_l2_clearance.json").is_file():
        blockers.append("missing_lane12_l2_clearance_packet")
    return tuple(blockers)


def _dispatch_blockers(root: Path, key: str) -> tuple[str, ...]:
    if key == "alpha1_nerv":
        blockers = ["requires_exact_cuda_auth_eval_after_l2_clearance"]
        if not (root / ".omx/state/lane12_nerv_l2_clearance.json").is_file():
            blockers.append("missing_lane12_l2_clearance_packet")
        return tuple(blockers)
    if key == "alpha2_wavelet":
        return (
            "requires_wmc1_runtime_loader_and_archive_manifest",
            "requires_byte_positive_real_archive_profile",
        )
    if key == "alpha3_vqvae":
        return (
            "requires_vqm1_runtime_loader_and_archive_manifest",
            "requires_real_archive_empirical_profile",
        )
    if key == "alpha4_grayscale_lut":
        return (
            "requires_current_exact_cuda_archive_evidence",
            "requires_train_inflate_grayscale_lut_parity",
        )
    return ()


def _readiness_status(
    *,
    missing_core: tuple[str, ...],
    text_blockers: tuple[str, ...],
    empirical_blockers: tuple[str, ...],
    static_blockers: tuple[str, ...],
) -> str:
    if missing_core:
        return "blocked_missing_core_component"
    if text_blockers:
        return "blocked_missing_runtime_integration"
    if empirical_blockers:
        return "blocked_empirical_evidence"
    if static_blockers:
        return "blocked_scientific_gate"
    return "ready_for_local_patch"


def _read_json_mapping(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if not isinstance(data, dict):
        return {}
    return data


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _markdown_cell(value: str) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _markdown_list(values: Iterable[str]) -> str:
    rows = tuple(values)
    if not rows:
        return "_none_"
    return "<br>".join(f"`{_markdown_cell(row)}`" for row in rows)


__all__ = [
    "AUDIT_NAME",
    "DISPATCH_BLOCKER_AUDIT_ONLY",
    "SCHEMA_VERSION",
    "AlphaCandidateDefinition",
    "AlphaCandidateReadiness",
    "PathGroup",
    "PathProbe",
    "TextProbe",
    "TextRequirement",
    "alpha_readiness_to_dict",
    "audit_paradigm_alpha",
    "path_probe_to_dict",
    "readiness_payload",
    "render_markdown",
    "text_probe_to_dict",
]
