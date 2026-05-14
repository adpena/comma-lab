#!/usr/bin/env python3
"""Promote a verified dual-axis frontier archive into tracked custody.

Closes A1 PR Council Round 1 finding F2 (CRITICAL — frontier archive lives
under ``experiments/results/`` which is ``.gitignore``d, so the canonical
"frontier" bytes are not in the git history and would be lost on a clean
checkout / new machine).

Per CLAUDE.md non-negotiables:

- "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
  HARDWARE" — promotion REQUIRES paired ``[contest-CUDA]`` AND
  ``[contest-CPU]`` evidence, both on 1:1 contest-compliant hardware
  (Linux x86_64 GHA / Modal / Vast / Lightning CPU; NVIDIA T4/4090/A100/etc).
- "Apples-to-apples evidence discipline" — both eval JSONs must reference
  the SAME archive sha256.
- "MPS auth eval is NOISE" — refuses MPS-graded evidence.
- "macOS-CPU advisory only" — refuses ``[macOS-CPU advisory]`` /
  ``[macOS-CPU calibrated]`` graded evidence (Catalog #192).
- "Public Disclosure Hygiene" — copies into ``submissions_frontier/<label>/``
  (or operator-specified target) which IS git-tracked.
- "Subagent commits MUST use serializer" — emits a recommended commit
  command using ``tools/subagent_commit_serializer.py`` with
  ``--expected-content-sha256`` per Catalog #157+#174. Does NOT auto-commit;
  the operator runs the printed command.

The tool is local custody machinery. It does NOT mutate the source archive,
launch any dispatch, or claim any score. It writes:

- ``<target_dir>/archive.zip``           — copy of source archive bytes
- ``<target_dir>/inflate.sh``            — copy of source inflate.sh
- ``<target_dir>/inflate.py``            — copy of source inflate.py (if present)
- ``<target_dir>/integrity_manifest.json`` — paired-axis evidence + sha256s
- ``<target_dir>/promotion_provenance.json`` — operator + utc + verification log

Refusal classes (each surfaces a typed blocker):

- ``only_one_axis_verified``      — one of CPU / CUDA evidence missing
- ``archive_sha256_mismatch``     — CPU and CUDA evidence reference different archives
- ``hardware_not_contest_compliant`` — substrate not in the CLAUDE.md 1:1 set
- ``macos_advisory_grade``        — ``[macOS-CPU advisory]`` evidence rejected
- ``mps_evidence``                — ``[MPS-PROXY]`` / ``[MPS-research-signal]`` rejected
- ``advisory_grade``              — any other ``[advisory only]`` / ``[byte-anchor]`` tag
- ``axis_score_diff_above_noise`` — CPU vs CUDA score diff outside known band (advisory)
- ``target_dir_not_under_repo``   — refuses absolute targets outside the repo
- ``target_path_collision``       — target dir already populated and ``--overwrite`` not set
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.continual_learning import (  # noqa: E402  (after sys.path mutation)
    AUTHORITATIVE_TAGS,
    NON_PROMOTABLE_TAGS,
    TAG_AXIS_REQUIREMENT,
    TAG_HARDWARE_REQUIREMENT,
    ContestResult,
)


PROMOTION_SCHEMA_VERSION = "tac_frontier_promotion_manifest_v1"


# ── Refusal taxonomy ──────────────────────────────────────────────────────


REFUSAL_CLASSES = (
    "only_one_axis_verified",
    "archive_sha256_mismatch",
    "archive_path_missing",
    "inflate_sh_missing",
    "eval_json_missing",
    "eval_json_unparseable",
    "evidence_tag_missing",
    "macos_advisory_grade",
    "mps_evidence",
    "advisory_grade",
    "hardware_not_contest_compliant",
    "axis_score_diff_above_noise",
    "target_dir_not_under_repo",
    "target_path_collision",
    "score_value_missing",
    "archive_sha256_field_missing",
)


class FrontierPromotionRefused(Exception):
    """Raised when promotion is refused. ``refusal_class`` carries the typed reason."""

    def __init__(self, refusal_class: str, message: str) -> None:
        super().__init__(message)
        self.refusal_class = refusal_class


@dataclass
class AxisEvidence:
    """One side of the dual-axis paired evidence."""

    axis: str  # "cpu" or "cuda"
    eval_json_path: Path
    archive_sha256: str
    score_value: float
    evidence_tag: str
    hardware_substrate: str
    n_samples: int | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def as_contest_result(
        self,
        *,
        archive_bytes: int,
        architecture_class: str,
    ) -> ContestResult:
        return ContestResult(
            axis=self.axis,
            hardware_substrate=self.hardware_substrate,
            architecture_class=architecture_class,
            score_value=self.score_value,
            evidence_tag=self.evidence_tag,
            archive_sha256=self.archive_sha256,
            archive_bytes=archive_bytes,
        )


# ── Evidence parsing ──────────────────────────────────────────────────────


def _read_eval_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FrontierPromotionRefused(
            "eval_json_missing",
            f"eval JSON not found: {path}",
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FrontierPromotionRefused(
            "eval_json_unparseable",
            f"eval JSON unparseable at {path}: {exc}",
        ) from exc


def _normalize_evidence_tag(payload: Mapping[str, Any]) -> str:
    """Return the evidence tag from one of the known field names.

    Accepts ``lane_tag`` / ``evidence_tag`` / ``score_axis_tag``. Returns
    empty string if no recognized field is present.
    """
    for key in ("lane_tag", "evidence_tag", "score_axis_tag"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_hardware_substrate(payload: Mapping[str, Any], axis: str) -> str:
    """Map common eval-JSON fields onto the canonical hardware_substrate string."""
    # Direct field if present.
    for key in ("hardware_substrate", "evidence_substrate"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Provenance-style nested dict.
    prov = payload.get("provenance")
    if isinstance(prov, Mapping):
        for key in ("hardware_substrate",):
            value = prov.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    # Heuristic mapping from ``hardware`` / ``device`` strings.
    hardware = str(payload.get("hardware") or "").lower()
    device = str(payload.get("device") or "").lower()

    if axis == "cuda":
        # GPU model heuristics.
        if "t4" in hardware:
            return "linux_x86_64_t4"
        if "4090" in hardware:
            return "linux_x86_64_4090"
        if "a100" in hardware:
            return "linux_x86_64_a100"
        if "h100" in hardware:
            return "linux_x86_64_h100"
        if "a10g" in hardware or "a10" in hardware:
            return "linux_x86_64_a10g"
        if "l40s" in hardware or "l40" in hardware:
            return "linux_x86_64_l40s"
        # Keep raw; will fail the 1:1 check below.
        return hardware or device or ""

    # CPU axis heuristics.
    if "github" in hardware or "gha" in hardware or "ubuntu" in hardware:
        return "linux_x86_64_gha_cpu"
    if "modal" in hardware:
        return "linux_x86_64_modal_cpu"
    if "vast" in hardware:
        return "linux_x86_64_vast_cpu"
    if "lightning" in hardware:
        return "linux_x86_64_lightning_cpu"
    if "darwin" in hardware or "mac" in hardware or "apple" in hardware:
        return "macos_arm64"
    if "x86_64" in hardware or "x86_64" in device:
        return "linux_x86_64_cpu"
    return hardware or device or ""


def _extract_score_value(payload: Mapping[str, Any]) -> float | None:
    """Return the canonical score from a contest_auth_eval JSON payload."""
    for key in (
        "canonical_score_recomputed",
        "score_recomputed_from_components",
        "canonical_score",
        "score_value",
        "score",
    ):
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_archive_sha(payload: Mapping[str, Any]) -> str:
    for key in ("archive_sha256", "archive_zip_sha256"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def parse_axis_evidence(eval_json_path: Path, axis: str) -> AxisEvidence:
    """Parse a contest_auth_eval JSON and return the typed AxisEvidence row.

    Refuses early on missing/unparseable files. ``axis`` MUST be "cpu" or
    "cuda" (the caller already knows which side this evidence is for).
    """
    if axis not in ("cpu", "cuda"):
        raise ValueError(f"axis must be 'cpu' or 'cuda', got {axis!r}")

    payload = _read_eval_json(eval_json_path)
    if not isinstance(payload, Mapping):
        raise FrontierPromotionRefused(
            "eval_json_unparseable",
            f"eval JSON at {eval_json_path} did not parse to a mapping",
        )

    tag = _normalize_evidence_tag(payload)
    if not tag:
        raise FrontierPromotionRefused(
            "evidence_tag_missing",
            f"eval JSON at {eval_json_path} has no lane_tag/evidence_tag field",
        )

    score = _extract_score_value(payload)
    if score is None:
        raise FrontierPromotionRefused(
            "score_value_missing",
            f"eval JSON at {eval_json_path} has no canonical_score* / score_value field",
        )

    sha = _extract_archive_sha(payload)
    if not sha:
        raise FrontierPromotionRefused(
            "archive_sha256_field_missing",
            f"eval JSON at {eval_json_path} has no archive_sha256 field",
        )

    hardware = _normalize_hardware_substrate(payload, axis)

    n_samples_raw = payload.get("n_samples")
    n_samples = int(n_samples_raw) if isinstance(n_samples_raw, int) else None

    return AxisEvidence(
        axis=axis,
        eval_json_path=eval_json_path,
        archive_sha256=sha,
        score_value=float(score),
        evidence_tag=tag,
        hardware_substrate=hardware,
        n_samples=n_samples,
        raw_payload=dict(payload),
    )


# ── Refusal validators ────────────────────────────────────────────────────


def _classify_evidence_grade(tag: str) -> str | None:
    """Return a refusal class for non-promotable tags, or None if promotable."""
    if not tag:
        return "evidence_tag_missing"
    # Catalog #192 + CLAUDE.md "MPS auth eval is NOISE" + macOS-CPU advisory.
    if "macOS" in tag or tag.lower().startswith("[macos"):
        return "macos_advisory_grade"
    if "MPS" in tag.upper():
        return "mps_evidence"
    if tag in NON_PROMOTABLE_TAGS:
        return "advisory_grade"
    if tag not in AUTHORITATIVE_TAGS:
        return "advisory_grade"
    return None


def _verify_axis_compliance(evidence: AxisEvidence) -> list[str]:
    """Return blockers for one axis. Empty list means promotion-eligible."""
    blockers: list[str] = []

    grade_refusal = _classify_evidence_grade(evidence.evidence_tag)
    if grade_refusal is not None:
        blockers.append(f"{grade_refusal}:{evidence.evidence_tag!r}")
        return blockers  # don't bother with hardware checks if grade is bad

    required_axis = TAG_AXIS_REQUIREMENT.get(evidence.evidence_tag)
    if required_axis is None:
        blockers.append(
            f"evidence_tag_unknown_axis_requirement:{evidence.evidence_tag!r}"
        )
        return blockers
    if required_axis != evidence.axis:
        blockers.append(
            f"axis_mismatch:tag={evidence.evidence_tag!r} requires {required_axis!r}, "
            f"got {evidence.axis!r}"
        )

    allowed = TAG_HARDWARE_REQUIREMENT.get(evidence.evidence_tag, frozenset())
    if evidence.hardware_substrate not in allowed:
        blockers.append(
            f"hardware_not_contest_compliant:{evidence.hardware_substrate!r} "
            f"not in {sorted(allowed)} for {evidence.evidence_tag!r}"
        )

    return blockers


def _verify_axis_pair(
    cpu: AxisEvidence,
    cuda: AxisEvidence,
    *,
    cpu_cuda_noise_floor: float = 0.05,
) -> list[str]:
    """Return blockers for the (CPU, CUDA) pair (apples-to-apples discipline)."""
    blockers: list[str] = []
    if cpu.archive_sha256 != cuda.archive_sha256:
        blockers.append(
            f"archive_sha256_mismatch:cpu={cpu.archive_sha256[:12]!r} "
            f"cuda={cuda.archive_sha256[:12]!r}"
        )

    diff = abs(cpu.score_value - cuda.score_value)
    if diff > cpu_cuda_noise_floor:
        blockers.append(
            f"axis_score_diff_above_noise:|cpu={cpu.score_value:.6f} - "
            f"cuda={cuda.score_value:.6f}|={diff:.6f} > "
            f"noise_floor={cpu_cuda_noise_floor:.6f} (advisory; F6 cross-machine "
            "variance probe should run before treating as method negative)"
        )
    return blockers


# ── Filesystem helpers ────────────────────────────────────────────────────


def _resolve_under_repo(p: Path, repo_root: Path) -> Path:
    """Resolve ``p`` and refuse if it falls outside the repo root."""
    resolved = p if p.is_absolute() else (repo_root / p)
    resolved = resolved.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise FrontierPromotionRefused(
            "target_dir_not_under_repo",
            f"target dir {resolved} is not under repo root {repo_root}",
        ) from exc
    return resolved


def _sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


# ── Top-level promotion ───────────────────────────────────────────────────


def build_integrity_manifest(
    *,
    archive_path: Path,
    archive_sha256: str,
    archive_bytes: int,
    cpu: AxisEvidence,
    cuda: AxisEvidence,
    label: str,
    architecture_class: str,
    inflate_sh_path: Path,
    inflate_py_path: Path | None,
    target_dir: Path,
    operator: str | None,
    notes: str | None,
    cpu_cuda_noise_floor: float,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build the integrity manifest dict for the dual-axis paired evidence."""
    now = datetime.now(timezone.utc).isoformat()
    target_dir_for_manifest: str
    try:
        target_dir_for_manifest = str(
            target_dir.resolve().relative_to(repo_root.resolve())
        )
    except ValueError:
        # target_dir lives outside the resolved repo_root (e.g., test fixtures
        # under tmp_path with a non-default repo_root). Persist absolute.
        target_dir_for_manifest = str(target_dir.resolve())
    manifest = {
        "schema": PROMOTION_SCHEMA_VERSION,
        "label": label,
        "architecture_class": architecture_class,
        "promoted_at_utc": now,
        "promoted_by": operator or "<unspecified>",
        "target_dir": target_dir_for_manifest,
        "archive": {
            "source_path": str(archive_path),
            "sha256": archive_sha256,
            "bytes": archive_bytes,
        },
        "inflate_sh_source_path": str(inflate_sh_path),
        "inflate_py_source_path": (
            str(inflate_py_path) if inflate_py_path is not None else None
        ),
        "axes": {
            "cpu": {
                "eval_json_path": str(cpu.eval_json_path),
                "score_value": cpu.score_value,
                "evidence_tag": cpu.evidence_tag,
                "hardware_substrate": cpu.hardware_substrate,
                "archive_sha256": cpu.archive_sha256,
                "n_samples": cpu.n_samples,
            },
            "cuda": {
                "eval_json_path": str(cuda.eval_json_path),
                "score_value": cuda.score_value,
                "evidence_tag": cuda.evidence_tag,
                "hardware_substrate": cuda.hardware_substrate,
                "archive_sha256": cuda.archive_sha256,
                "n_samples": cuda.n_samples,
            },
        },
        "cpu_cuda_score_diff": abs(cpu.score_value - cuda.score_value),
        "cpu_cuda_noise_floor_used": cpu_cuda_noise_floor,
        "claude_md_compliance_tags": [
            "submission_auth_eval_dual_axis_1to1_contest_compliant",
            "apples_to_apples_paired_archive_sha256",
            "no_mps_authoritative",
            "no_macos_cpu_authoritative",
            "no_score_claim_advanced_by_this_artifact",
            "no_tmp_paths",
            "f2_council_round_1_critical_remediation",
        ],
        "score_claim": False,
        "promotion_eligible": True,
        "ready_for_exact_eval_dispatch": False,
        "notes": notes or "",
    }
    return manifest


def promote_archive(
    *,
    archive_path: Path,
    inflate_sh_path: Path,
    contest_cpu_eval_path: Path,
    contest_cuda_eval_path: Path,
    label: str,
    target_dir: Path,
    architecture_class: str | None = None,
    inflate_py_path: Path | None = None,
    operator: str | None = None,
    notes: str | None = None,
    overwrite: bool = False,
    cpu_cuda_noise_floor: float = 0.05,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Promote a verified dual-axis frontier archive into ``target_dir``.

    Returns the integrity manifest dict on success. Raises
    :class:`FrontierPromotionRefused` with a typed ``refusal_class`` on any
    contract violation.
    """
    # 0. Resolve target dir; refuse outside-repo.
    target_resolved = _resolve_under_repo(target_dir, repo_root)
    if target_resolved.exists() and not overwrite:
        # only refuse if the target dir has files we'd overwrite.
        existing = list(target_resolved.iterdir()) if target_resolved.is_dir() else [target_resolved]
        if existing:
            raise FrontierPromotionRefused(
                "target_path_collision",
                f"target dir {target_resolved} is non-empty; pass --overwrite to replace",
            )

    # 1. Source-file existence checks.
    if not archive_path.is_file():
        raise FrontierPromotionRefused(
            "archive_path_missing",
            f"archive path not found: {archive_path}",
        )
    if not inflate_sh_path.is_file():
        raise FrontierPromotionRefused(
            "inflate_sh_missing",
            f"inflate.sh not found: {inflate_sh_path}",
        )

    # 2. Parse both eval JSONs; refuses early on missing axis.
    cpu = parse_axis_evidence(contest_cpu_eval_path, axis="cpu")
    cuda = parse_axis_evidence(contest_cuda_eval_path, axis="cuda")

    # 3. Per-axis compliance checks.
    cpu_blockers = _verify_axis_compliance(cpu)
    cuda_blockers = _verify_axis_compliance(cuda)
    pair_blockers = _verify_axis_pair(
        cpu, cuda, cpu_cuda_noise_floor=cpu_cuda_noise_floor
    )

    all_blockers = cpu_blockers + cuda_blockers + pair_blockers

    # Map the FIRST refusal back to its typed class for the exception.
    if all_blockers:
        first = all_blockers[0]
        # Pull a refusal class prefix from the blocker string.
        cls = next((c for c in REFUSAL_CLASSES if first.startswith(c)), "advisory_grade")
        raise FrontierPromotionRefused(
            cls,
            f"promotion refused; blockers: {all_blockers}",
        )

    # 4. Compute archive sha256 and verify it matches BOTH eval JSONs.
    archive_sha256, archive_bytes = _sha256_file(archive_path)
    if archive_sha256 != cpu.archive_sha256 or archive_sha256 != cuda.archive_sha256:
        raise FrontierPromotionRefused(
            "archive_sha256_mismatch",
            f"archive bytes sha256={archive_sha256[:12]} differs from "
            f"cpu_eval_sha={cpu.archive_sha256[:12]} or "
            f"cuda_eval_sha={cuda.archive_sha256[:12]}",
        )

    # 5. Materialize target dir and copy bytes.
    target_resolved.mkdir(parents=True, exist_ok=True)
    target_archive = target_resolved / "archive.zip"
    target_inflate_sh = target_resolved / "inflate.sh"
    shutil.copyfile(archive_path, target_archive)
    shutil.copyfile(inflate_sh_path, target_inflate_sh)
    if inflate_py_path is not None and inflate_py_path.is_file():
        shutil.copyfile(inflate_py_path, target_resolved / "inflate.py")

    arch_class = architecture_class or label

    # 6. Write integrity manifest.
    manifest = build_integrity_manifest(
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        cpu=cpu,
        cuda=cuda,
        label=label,
        architecture_class=arch_class,
        inflate_sh_path=inflate_sh_path,
        inflate_py_path=inflate_py_path,
        target_dir=target_resolved,
        operator=operator,
        notes=notes,
        cpu_cuda_noise_floor=cpu_cuda_noise_floor,
        repo_root=repo_root,
    )

    manifest_path = target_resolved / "integrity_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # 7. Provenance log (separate from manifest so future re-promotions
    # append rather than overwrite custody history).
    provenance_path = target_resolved / "promotion_provenance.json"
    if provenance_path.exists():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            provenance = {"attempts": []}
    else:
        provenance = {"attempts": []}
    provenance.setdefault("attempts", []).append(
        {
            "promoted_at_utc": manifest["promoted_at_utc"],
            "promoted_by": manifest["promoted_by"],
            "archive_sha256": archive_sha256,
            "cpu_score": cpu.score_value,
            "cuda_score": cuda.score_value,
            "cpu_evidence_tag": cpu.evidence_tag,
            "cuda_evidence_tag": cuda.evidence_tag,
            "cpu_hardware": cpu.hardware_substrate,
            "cuda_hardware": cuda.hardware_substrate,
            "label": label,
        }
    )
    provenance["schema"] = "tac_frontier_promotion_provenance_v1"
    provenance["last_promoted_at_utc"] = manifest["promoted_at_utc"]
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return manifest


def render_serializer_command(
    target_dir: Path,
    label: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Render the canonical serializer commit command for the operator.

    Per Catalog #117 + #157 + #174, every subagent commit MUST go through
    ``tools/subagent_commit_serializer.py`` with ``--expected-content-sha256``.
    """
    rel = (
        target_dir.relative_to(repo_root)
        if target_dir.is_absolute()
        else target_dir
    )
    files = [
        f"{rel}/archive.zip",
        f"{rel}/inflate.sh",
        f"{rel}/integrity_manifest.json",
        f"{rel}/promotion_provenance.json",
    ]
    inflate_py = target_dir / "inflate.py"
    if inflate_py.is_file():
        files.insert(1, f"{rel}/inflate.py")
    sha_args = []
    for f in files:
        full = repo_root / f if not Path(f).is_absolute() else Path(f)
        if full.is_file():
            digest, _ = _sha256_file(full)
            sha_args.append(f"--expected-content-sha256 '{f}={digest}'")
    sha_clause = " ".join(sha_args)
    return (
        ".venv/bin/python tools/subagent_commit_serializer.py "
        f"--message 'frontier promotion: {label} (F2 fix; dual-axis verified)' "
        f"--reason 'frontier_promotion_{label}' "
        f"--files {' '.join(files)} "
        f"{sha_clause}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Promote a verified dual-axis frontier archive into tracked custody. "
            "Closes A1 PR Council Round 1 finding F2 (CRITICAL)."
        )
    )
    p.add_argument(
        "--archive-path",
        required=True,
        type=Path,
        help="Source archive.zip (typically under experiments/results/...)",
    )
    p.add_argument(
        "--inflate-sh",
        required=True,
        type=Path,
        help="Source inflate.sh that pairs with the archive",
    )
    p.add_argument(
        "--inflate-py",
        type=Path,
        default=None,
        help="Source inflate.py (optional; copied if present)",
    )
    p.add_argument(
        "--contest-cpu-eval",
        required=True,
        type=Path,
        help="Path to the contest_auth_eval CPU JSON (Linux x86_64; 1:1 contest CI)",
    )
    p.add_argument(
        "--contest-cuda-eval",
        required=True,
        type=Path,
        help="Path to the contest_auth_eval CUDA JSON (T4 / 4090 / A100)",
    )
    p.add_argument(
        "--label",
        required=True,
        help="Slug for the promoted artifact (e.g. 'a1_dual_axis_20260513')",
    )
    p.add_argument(
        "--target-dir",
        required=True,
        type=Path,
        help="Tracked target dir (e.g. submissions_frontier/<slug>/)",
    )
    p.add_argument(
        "--architecture-class",
        default=None,
        help="Architecture class (defaults to --label)",
    )
    p.add_argument(
        "--operator",
        default=None,
        help="Operator handle (recorded in provenance log)",
    )
    p.add_argument(
        "--notes",
        default=None,
        help="Free-form notes for the integrity manifest",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing non-empty target dir",
    )
    p.add_argument(
        "--cpu-cuda-noise-floor",
        type=float,
        default=0.05,
        help=(
            "Tolerance band for the |cpu_score - cuda_score| advisory blocker. "
            "Defaults to 0.05 (covers PR102/PR107 observed +0.033 pose-driven gap)."
        ),
    )
    p.add_argument(
        "--print-commit-command",
        action="store_true",
        help="Print the canonical serializer commit command after success",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        manifest = promote_archive(
            archive_path=args.archive_path,
            inflate_sh_path=args.inflate_sh,
            inflate_py_path=args.inflate_py,
            contest_cpu_eval_path=args.contest_cpu_eval,
            contest_cuda_eval_path=args.contest_cuda_eval,
            label=args.label,
            target_dir=args.target_dir,
            architecture_class=args.architecture_class,
            operator=args.operator,
            notes=args.notes,
            overwrite=args.overwrite,
            cpu_cuda_noise_floor=args.cpu_cuda_noise_floor,
        )
    except FrontierPromotionRefused as exc:
        print(f"REFUSED [{exc.refusal_class}]: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"REFUSED [missing_file]: {exc}", file=sys.stderr)
        return 2

    print(f"PROMOTED: {manifest['label']}")
    print(f"  archive.sha256: {manifest['archive']['sha256']}")
    print(f"  archive.bytes:  {manifest['archive']['bytes']}")
    print(f"  cpu_score:      {manifest['axes']['cpu']['score_value']}")
    print(f"  cuda_score:     {manifest['axes']['cuda']['score_value']}")
    print(f"  target_dir:     {manifest['target_dir']}")

    if args.print_commit_command:
        print()
        print("Canonical commit command (Catalog #117+#157+#174):")
        print()
        print(render_serializer_command(args.target_dir, args.label))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
