# SPDX-License-Identifier: MIT
"""Index public PR writeups into device-labeled mechanism evidence.

This module is intentionally evidence-only. It parses local public-PR mirrors
and emits score rows, mechanism tags, and source file references without
promoting any row to contest evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

CONTEST_N_BYTES = 37_545_489
INDEX_SCHEMA = "public_pr_mechanism_index.v1"

DEFAULT_FILENAMES = frozenset({"pr_body.md", "README.md", "report.txt"})
DEFAULT_NAME_PATTERNS = ("writeup",)
PR_RE = re.compile(r"public_pr(?P<pr>\d+)_", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s)>\]]+")
DEVICE_RE = re.compile(r"^\s*device:\s*([^\n]+)", re.IGNORECASE | re.MULTILINE)
GPU_REQUIREMENT_RE = re.compile(
    r"requires?\s+gpu\s+for\s+evaluation.*?\n\s*(yes|no)\b",
    re.IGNORECASE | re.DOTALL,
)
POSE_RE = re.compile(r"Average PoseNet Distortion:\s*([0-9.,]+)", re.IGNORECASE)
SEG_RE = re.compile(r"Average SegNet Distortion:\s*([0-9.,]+)", re.IGNORECASE)
SIZE_RE = re.compile(r"Submission file size:\s*([0-9.,]+)\s*bytes?", re.IGNORECASE)
FINAL_RE = re.compile(r"Final score:.*?=\s*([0-9.]+)", re.IGNORECASE | re.DOTALL)

MECHANISM_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "hnerv_renderer": (
        r"\bHNeRV\b",
        r"hnerv",
    ),
    "nerv_family": (
        r"\bNeRV\b",
        r"SingleNeRV",
        r"neural_inflate",
    ),
    "rgb_renderer": (
        r"RGB",
        r"renderer",
        r"token.?RGB",
        r"frames?",
    ),
    "score_aware_training": (
        r"SegNet",
        r"PoseNet",
        r"score",
        r"distortion",
        r"DALI",
        r"PyAV",
    ),
    "eval_roundtrip_or_loader": (
        r"eval",
        r"roundtrip",
        r"DALI",
        r"PyAV",
        r"cu128",
    ),
    "quantization_aware_training": (
        r"\bQAT\b",
        r"quantization",
        r"INT8",
        r"fp16",
        r"uint8",
    ),
    "muon_or_optimizer_curriculum": (
        r"Muon",
        r"curriculum",
        r"lambda",
        r"sigma",
        r"fine.?tune",
        r"SWA",
    ),
    "latent_sidecar_or_correction": (
        r"sidecar",
        r"latent correction",
        r"correction sidecar",
        r"per.?pair",
        r"delta",
    ),
    "arithmetic_entropy_codec": (
        r"arithmetic",
        r"range cod",
        r"constriction",
        r"Huffman",
        r"ANS",
        r"entropy",
        r"HPAC",
        r"PPMd",
        r"brotli",
        r"LZMA",
    ),
    "video_codec_baseline": (
        r"\bAV1\b",
        r"\bSVT",
        r"\bH265\b",
        r"\bHEVC\b",
        r"\bCRF\b",
        r"\bGOP\b",
    ),
    "roi_foveation_or_spatial_priority": (
        r"\bROI\b",
        r"fove",
        r"QP.?map",
        r"boundary",
        r"hard.?pixel",
        r"unsharp",
        r"lanczos",
    ),
    "pose_or_qpose_codec": (
        r"\bpose\b",
        r"qpose",
        r"PoseNet",
    ),
    "mask_action_or_seg_codec": (
        r"\bmask\b",
        r"seg.?action",
        r"SegNet",
        r"class",
    ),
    "token_or_categorical_codec": (
        r"token",
        r"categorical",
        r"class",
        r"probabil",
    ),
    "channel_bias_or_yshift": (
        r"yshift",
        r"channel",
        r"bias",
        r"red channel",
        r"YUV",
    ),
    "compressai_balle_hyperprior": (
        r"CompressAI",
        r"Ball[ée]",
        r"hyperprior",
    ),
    "coolchic_c3_wavelet_residual": (
        r"Cool.?Chic",
        r"\bC3\b",
        r"wavelet",
        r"residual",
    ),
}


def recompute_score(*, pose: float, seg: float, archive_bytes: int) -> float:
    """Return the contest formula from rounded component fields."""

    return 100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * archive_bytes / CONTEST_N_BYTES


def build_public_pr_mechanism_index(
    roots: Iterable[str | Path],
    *,
    min_pr: int = 0,
    max_pr: int | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build an evidence-only public-PR mechanism index from local files."""

    repo = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for path in _iter_source_files([Path(root) for root in roots], min_pr=min_pr, max_pr=max_pr):
        text = path.read_text(encoding="utf-8", errors="replace")
        pr = _pr_from_path(path)
        if pr is None:
            continue
        text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        dedupe_key = (pr, path.name, text_hash)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        records.append(_file_record(path, text, pr=pr, repo_root=repo, text_sha256=text_hash))
    records.sort(key=lambda row: (row["pr"], row["relative_path"]))
    return {
        "schema": INDEX_SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "local_public_pr_text_index",
        "roots": [_repo_rel(Path(root), repo) for root in roots],
        "min_pr": min_pr,
        "max_pr": max_pr,
        "file_count": len(records),
        "records": records,
        "summary_by_pr": _summarize_by_pr(records),
    }


def render_markdown_summary(index: Mapping[str, Any]) -> str:
    """Render a compact human-facing summary of an index payload."""

    lines = [
        "# Public PR mechanism index",
        "",
        "This is an evidence-only local text index. It is not a score claim.",
        "",
        f"- schema: `{index.get('schema')}`",
        f"- files indexed: `{index.get('file_count')}`",
        f"- promotion eligible: `{index.get('promotion_eligible')}`",
        "",
        "## PR summary",
        "",
        "| PR | files | devices | eval rows | mechanisms | best rounded-score row |",
        "| --- | ---: | --- | ---: | --- | --- |",
    ]
    for item in index.get("summary_by_pr", []):
        devices = ", ".join(f"{k}:{v}" for k, v in item.get("devices", {}).items()) or "-"
        mechanisms = ", ".join(item.get("mechanisms", [])[:10]) or "-"
        best = item.get("best_eval_row")
        if best:
            best_text = (
                f"{best.get('score_or_recomputed'):.9f} "
                f"({best.get('device') or 'unknown'}, {best.get('source')})"
            )
        else:
            best_text = "-"
        lines.append(
            f"| {item['pr']} | {item['file_count']} | {devices} | {item['eval_row_count']} | "
            f"{mechanisms} | {best_text} |"
        )
    lines.extend(
        [
            "",
            "## Evidence files with mechanism hits",
            "",
            "| PR | file | mechanisms |",
            "| --- | --- | --- |",
        ]
    )
    for record in index.get("records", []):
        mechanisms = ", ".join(hit["family"] for hit in record.get("mechanism_hits", []))
        if mechanisms:
            lines.append(f"| {record['pr']} | `{record['relative_path']}` | {mechanisms} |")
    lines.append("")
    return "\n".join(lines)


def write_index_outputs(
    index: Mapping[str, Any],
    *,
    json_out: str | Path | None = None,
    md_out: str | Path | None = None,
) -> None:
    """Write deterministic JSON and/or Markdown outputs."""

    if json_out is not None:
        path = Path(json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if md_out is not None:
        path = Path(md_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown_summary(index), encoding="utf-8")


def _iter_source_files(roots: Sequence[Path], *, min_pr: int, max_pr: int | None) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or not _is_source_file(path):
                continue
            pr = _pr_from_path(path)
            if pr is None or pr < min_pr or (max_pr is not None and pr > max_pr):
                continue
            files.append(path)
    return sorted(files)


def _is_source_file(path: Path) -> bool:
    if path.name in DEFAULT_FILENAMES:
        return True
    lower_name = path.name.lower()
    return any(pattern in lower_name for pattern in DEFAULT_NAME_PATTERNS) and path.suffix.lower() in {
        ".md",
        ".txt",
    }


def _file_record(path: Path, text: str, *, pr: int, repo_root: Path, text_sha256: str) -> dict[str, Any]:
    return {
        "pr": pr,
        "relative_path": _repo_rel(path, repo_root),
        "kind": _kind(path),
        "text_sha256": text_sha256,
        "line_count": text.count("\n") + (0 if text.endswith("\n") or not text else 1),
        "eval_rows": _extract_eval_rows(text, source=_repo_rel(path, repo_root)),
        "mechanism_hits": _extract_mechanism_hits(text),
        "archive_urls": _extract_archive_urls(text),
    }


def _extract_eval_rows(text: str, *, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    file_device = _infer_file_device(text)
    # Most public comments/report files contain one table. If more appear in a
    # file, split at result headers and parse each result-shaped chunk.
    chunks = re.split(r"(?=^\s*=== Evaluation|^# upload|^Eval Results)", text, flags=re.MULTILINE)
    for chunk in chunks:
        pose = _first_float(POSE_RE, chunk)
        seg = _first_float(SEG_RE, chunk)
        size = _first_int(SIZE_RE, chunk)
        final_score = _first_float(FINAL_RE, chunk)
        if pose is None and seg is None and size is None and final_score is None:
            continue
        device_match = DEVICE_RE.search(chunk)
        device = device_match.group(1).strip() if device_match else file_device
        row: dict[str, Any] = {
            "source": source,
            "device": device,
            "pose": pose,
            "seg": seg,
            "archive_bytes": size,
            "printed_score": final_score,
        }
        if pose is not None and seg is not None and size is not None:
            row["recomputed_score_from_rounded_components"] = recompute_score(
                pose=pose,
                seg=seg,
                archive_bytes=size,
            )
        row["score_or_recomputed"] = row.get("recomputed_score_from_rounded_components", final_score)
        rows.append(row)
    return rows


def _extract_mechanism_hits(text: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    lines = text.splitlines()
    for family, patterns in MECHANISM_PATTERNS.items():
        compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        family_hits: list[dict[str, Any]] = []
        for line_no, line in enumerate(lines, start=1):
            if any(regex.search(line) for regex in compiled):
                family_hits.append({"line": line_no, "text": _compact(line)})
                if len(family_hits) == 3:
                    break
        if family_hits:
            hits.append({"family": family, "evidence": family_hits})
    return hits


def _extract_archive_urls(text: str) -> list[str]:
    urls = []
    for url in URL_RE.findall(text):
        lower = url.lower()
        if "archive" in lower or lower.endswith(".zip"):
            urls.append(url.rstrip(".,"))
    return sorted(set(urls))


def _infer_file_device(text: str) -> str | None:
    match = DEVICE_RE.search(text)
    if match:
        return match.group(1).strip()
    gpu_match = GPU_REQUIREMENT_RE.search(text)
    if not gpu_match:
        return None
    return "gpu_required" if gpu_match.group(1).lower() == "yes" else "cpu_capable"


def _summarize_by_pr(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["pr"])].append(record)
    summaries: list[dict[str, Any]] = []
    for pr in sorted(grouped):
        pr_records = grouped[pr]
        mechanisms = sorted({hit["family"] for record in pr_records for hit in record.get("mechanism_hits", [])})
        eval_rows = [row for record in pr_records for row in record.get("eval_rows", [])]
        devices = Counter(str(row.get("device") or "unknown") for row in eval_rows)
        score_rows = [row for row in eval_rows if isinstance(row.get("score_or_recomputed"), int | float)]
        best = min(score_rows, key=lambda row: float(row["score_or_recomputed"])) if score_rows else None
        summaries.append(
            {
                "pr": pr,
                "file_count": len(pr_records),
                "eval_row_count": len(eval_rows),
                "devices": dict(sorted(devices.items())),
                "mechanisms": mechanisms,
                "best_eval_row": best,
            }
        )
    return summaries


def _pr_from_path(path: Path) -> int | None:
    match = PR_RE.search(path.as_posix())
    return int(match.group("pr")) if match else None


def _kind(path: Path) -> str:
    if path.name == "pr_body.md":
        return "pr_body"
    if path.name == "report.txt":
        return "report"
    if path.name == "README.md":
        return "readme"
    return "writeup"


def _first_float(regex: re.Pattern[str], text: str) -> float | None:
    match = regex.search(text)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def _first_int(regex: re.Pattern[str], text: str) -> int | None:
    match = regex.search(text)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def _repo_rel(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _compact(line: str, *, max_len: int = 220) -> str:
    text = " ".join(line.strip().split())
    return text if len(text) <= max_len else text[: max_len - 3] + "..."
