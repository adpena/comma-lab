#!/usr/bin/env python3
"""Audit the active semantic-label contract and selected live callsites."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac import camera  # noqa: E402
from tac.mask_grayscale_lut import CLASS_TO_GRAY  # noqa: E402
from tac.semantic_label_contract import (  # noqa: E402
    CONTEST_SEGNET_CLASS_NAME_TUPLE,
    CONTEST_SEGNET_CLASS_NAMES,
    NUM_CONTEST_SEGNET_CLASSES,
    SELFCOMP_CLASS_TO_GRAY,
    validate_contest_class_table,
)
from tac.semantic_quantization import CLASS_NAMES, DEFAULT_CLASS_BITS  # noqa: E402


CORE_REL_PATHS: tuple[str, ...] = (
    "src/tac/semantic_label_contract.py",
    "src/tac/semantic_quantization.py",
    "src/tac/mask_grayscale_lut.py",
    "src/tac/camera.py",
    "src/tac/mask_prior.py",
    "src/tac/mini_scorer.py",
    "src/tac/depth_motion.py",
    "experiments/diagnose_nerv_geometry.py",
    "experiments/constrained_gen_research/inflate.py",
)

ADVISORY_REL_PATHS: tuple[str, ...] = (
    "src/tac/constrained_gen.py",
    "submissions/robust_current/inflate_renderer.py",
    "src/tac/dp_sims_renderer.py",
    "experiments/plan_c067_micro_mask_reencode.py",
    "scripts/remote_lane_ps_per_class_segnet.sh",
    "scripts/remote_lane_ps_v2_learnable_class_weights.sh",
    "experiments/train_distill.py",
    "src/tac/learnable_class_weights.py",
    "src/tac/experiments/train_renderer.py",
    "experiments/optimize_poses.py",
    "scripts/remote_lane_sq_semantic_quantization.sh",
    "src/tac/network_codec.py",
    "src/tac/mae_mask_aug.py",
    "src/tac/learnable_class_weights.py",
    "submissions/robust_current/compress.sh",
    "submissions/robust_current/sky_degrade.py",
    "src/tac/visualization/comparison_video.py",
    "src/tac/viz/analysis_panels.py",
)

STALE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "legacy class order",
        re.compile(r"\broad\s*,\s*lane\s*,\s*(?:bg|background)\s*,\s*vehicle\s*,\s*sky\b", re.IGNORECASE),
    ),
    (
        "legacy class order",
        re.compile(r"\broad\s*,\s*lane\s*,\s*vehicle\s*,\s*sky\s*,\s*background\b", re.IGNORECASE),
    ),
    ("class 0 background", re.compile(r"\bclass\s*0\s*[:=]\s*background\b", re.IGNORECASE)),
    ("class 0 background", re.compile(r"\b0\s*[:=]\s*[\"']?background[\"']?\b", re.IGNORECASE)),
    ("class 0 background", re.compile(r"\b0\s*\(\s*background\s*\)", re.IGNORECASE)),
    ("class 2 vehicle", re.compile(r"\bclass\s*2\s*(?:[:=]|\bis\b|\bas\b)\s*[\"']?(?:vehicle|car|movable)", re.IGNORECASE)),
    (
        "class 2 background",
        re.compile(r"\bclass\s*2\s*(?:[:=]|\bis\b|\bas\b)\s*[\"']?(?:background|vegetation)", re.IGNORECASE),
    ),
    ("class 3 sky", re.compile(r"\bclass\s*3\s*(?:[:=]|\bis\b|\bas\b)\s*[\"']?sky", re.IGNORECASE)),
    ("class 4 sky", re.compile(r"\bclass\s*4\s*(?:[:=]|\bis\b|\bas\b)\s*[\"']?sky", re.IGNORECASE)),
    ("class 4 vehicle", re.compile(r"\bclass\s*4\s*(?:[:=]|\bis\b|\bas\b)\s*[\"']?(?:vehicle|movable)", re.IGNORECASE)),
    ("class 4 background", re.compile(r"\bclass\s*4\s*(?:[:=]|\bis\b|\bas\b)\s*[\"']?background", re.IGNORECASE)),
    ("index 4 sky", re.compile(r"(?<![#\w])4\s*[:=]\s*[\"']?sky[\"']?\b", re.IGNORECASE)),
    ("index 4 vehicle", re.compile(r"(?<![#\w])4\s*[:=]\s*[\"']?(?:vehicle|movable)[\"']?\b", re.IGNORECASE)),
    ("index 4 background", re.compile(r"(?<![#\w])4\s*[:=]\s*[\"']?background[\"']?\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class AuditFinding:
    path: str
    line: int
    severity: str
    pattern: str
    text: str


@dataclass(frozen=True)
class AuditResult:
    contract_ok: bool
    blocking_findings: tuple[AuditFinding, ...]
    advisory_findings: tuple[AuditFinding, ...]

    @property
    def findings(self) -> tuple[AuditFinding, ...]:
        return self.blocking_findings + self.advisory_findings

    @property
    def ok(self) -> bool:
        return self.contract_ok and not self.blocking_findings


def _line_findings(path: Path, rel_path: str, severity: str) -> list[AuditFinding]:
    text = path.read_text(encoding="utf-8")
    findings: list[AuditFinding] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern_name, pattern in STALE_PATTERNS:
            if pattern_name.startswith("index ") and re.search(r"\bclass\s*4\b", line, re.IGNORECASE):
                continue
            if pattern.search(line):
                findings.append(
                    AuditFinding(
                        path=rel_path,
                        line=line_no,
                        severity=severity,
                        pattern=pattern_name,
                        text=line.strip(),
                    )
                )
    return findings


def audit_semantic_label_contract(
    repo_root: Path = REPO_ROOT,
    rel_paths: tuple[str, ...] = CORE_REL_PATHS,
    advisory_rel_paths: tuple[str, ...] = ADVISORY_REL_PATHS,
) -> AuditResult:
    validate_contest_class_table()
    contract_ok = (
        camera.NUM_CLASSES == NUM_CONTEST_SEGNET_CLASSES
        and camera.SEGNET_CLASS_NAMES == CONTEST_SEGNET_CLASS_NAME_TUPLE
        and CLASS_TO_GRAY == SELFCOMP_CLASS_TO_GRAY
        and CLASS_NAMES == CONTEST_SEGNET_CLASS_NAMES
        and DEFAULT_CLASS_BITS[0] == 8
        and DEFAULT_CLASS_BITS[1] == 8
    )

    blocking_findings: list[AuditFinding] = []
    for rel_path in tuple(dict.fromkeys(rel_paths)):
        path = repo_root / rel_path
        if not path.exists():
            blocking_findings.append(
                AuditFinding(
                    path=rel_path,
                    line=0,
                    severity="blocking",
                    pattern="missing",
                    text="curated semantic-label audit path is missing",
                )
            )
            continue
        blocking_findings.extend(_line_findings(path, rel_path, "blocking"))

    advisory_findings: list[AuditFinding] = []
    for rel_path in tuple(dict.fromkeys(advisory_rel_paths)):
        if rel_path in rel_paths:
            continue
        path = repo_root / rel_path
        if not path.exists():
            advisory_findings.append(
                AuditFinding(
                    path=rel_path,
                    line=0,
                    severity="advisory",
                    pattern="missing",
                    text="curated semantic-label advisory path is missing",
                )
            )
            continue
        advisory_findings.extend(_line_findings(path, rel_path, "advisory"))

    return AuditResult(
        contract_ok=contract_ok,
        blocking_findings=tuple(blocking_findings),
        advisory_findings=tuple(advisory_findings),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--no-advisory",
        action="store_true",
        help="skip advisory scans of risky runtime/provenance surfaces",
    )
    parser.add_argument(
        "--fail-on-advisory",
        action="store_true",
        help="return nonzero when advisory findings are present",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_semantic_label_contract(
        repo_root=args.repo_root,
        advisory_rel_paths=() if args.no_advisory else ADVISORY_REL_PATHS,
    )
    if args.format == "json":
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "contract_ok": result.contract_ok,
                    "blocking_findings": [asdict(finding) for finding in result.blocking_findings],
                    "advisory_findings": [asdict(finding) for finding in result.advisory_findings],
                    "findings": [asdict(finding) for finding in result.findings],
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif result.ok:
        if result.advisory_findings:
            print(f"semantic label contract audit: OK ({len(result.advisory_findings)} advisory findings)")
            for finding in result.advisory_findings:
                loc = f"{finding.path}:{finding.line}" if finding.line else finding.path
                print(f"- advisory {loc}: stale pattern {finding.pattern!r}: {finding.text}")
        else:
            print("semantic label contract audit: OK")
    else:
        print("semantic label contract audit: FAILED")
        if not result.contract_ok:
            print("- live contract exports disagree")
        for finding in result.blocking_findings:
            loc = f"{finding.path}:{finding.line}" if finding.line else finding.path
            print(f"- blocking {loc}: stale pattern {finding.pattern!r}: {finding.text}")
        for finding in result.advisory_findings:
            loc = f"{finding.path}:{finding.line}" if finding.line else finding.path
            print(f"- advisory {loc}: stale pattern {finding.pattern!r}: {finding.text}")
    return 0 if result.ok and (not args.fail_on_advisory or not result.advisory_findings) else 2


if __name__ == "__main__":
    raise SystemExit(main())
