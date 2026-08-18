#!/usr/bin/env python3
"""AU1 measurement-integrity audit surfaces.

This tool is intentionally read-only with respect to scorer/evaluator paths.  It
turns AU1's recalled audit requirements into queryable JSONL receipts:

* consumption-side receiver decode-read coverage, seeded from the gk2/#847
  inventory and checked against live source text;
* a corrections index over research memos;
* a headline-vs-body numeric refutation detector over memos, task rows, and
  canonical task ledger git-log subjects;
* a registry-first row for the #840/ddm_cf1 naming collision.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_ROOT = REPO_ROOT / ".omx" / "research"
TASK_LEDGER = REPO_ROOT / ".omx" / "state" / "canonical_task_status.jsonl"
DEFAULT_OUTDIR = RESEARCH_ROOT / "ddm_au1_20260805"

NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?(?:e[-+]?\d+)?%?"
)
REFUTATION_PHRASES = (
    "corrected",
    "corrects",
    "correction",
    "wrong",
    "refuted",
    "superseded",
    "stale",
    "withdrawn",
)

# --- the scope-word pass (ddm_hd1 2026-08-18, from ddm_na9 F5) -------------------------
#
# A closure is only as good as the OBJECT it names. Four measured instances of a memo
# measuring a narrow object and naming a wide one; the fourth sat on a live -560 B win for
# a day. The detector asks na9's one-line question of every closure sentence: does this
# sentence carry the scope of what was actually measured?
#
# A flag is NOT a claim that the sentence is wrong. It is "this closure names a wide object
# and states no scope" -- cheap to read, cheap to dismiss. The summary reports both sides,
# because a detector that fires on everything is as uninformative as one that never fires.

CLOSURE_WORDS = (
    "closed",
    "exhausted",
    "dead",
    "retired",
    "saturated",
    "settled",
)

#: Idioms of OUR OWN vocabulary that contain a closure word but assert no closure. Measured,
#: not guessed: a first pass without these emitted 3,062 rows of which 382 were "byte-closed"
#: (our standard phrase for a verified archive) and 257 "closed-form" (a kind of maths). A
#: detector that fires on the house dialect is a detector nobody reads.
CLOSURE_FALSE_FRIENDS = (
    "byte-closed",
    "byte closed",
    "closed-form",
    "closed form",
    "closed-loop",
    "closed loop",
    "receiver-closed",
    "fail-closed",
    "fail closed",
    "deadline",
    "deadlock",
    "dead code",
    "dead flag",
    "dead-flag",
)

#: The object nouns that make a closure a FAMILY-level claim rather than an instance one.
#: Deliberately narrow. "all"/"any"/"every"/"whole" were measured as pure noise (1,588 of the
#: first pass's 3,062 rows) because they appear in ordinary prose; they are dropped.
WIDE_OBJECT_WORDS = (
    "axis",
    "family",
    "families",
    "class",
    "paradigm",
    "entire",
)

#: A closure word and a wide object noun in the same sentence but forty words apart are
#: usually two unrelated clauses. Require them to be near each other.
SCOPE_PROXIMITY_CHARS = 60

#: A sentence that carries any of these is stating what it held fixed, i.e. it is already
#: scoped. These are the HONEST form -- the denominator that proves the gauge can pass.
SCOPE_QUALIFIERS = (
    "given",
    "at fixed",
    "holding",
    "held constant",
    "held fixed",
    "instance",
    "formulation",
    "scope",
    "scoped",
    "swap",
    "only",
    "assuming",
    "under the",
    "at this",
    "on this",
    "for this",
    "measured on",
    "narrowed",
    "at the conditional level",
    "within",
    "restricted",
    "sub-",
    "verdict_scope",
)


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def json_dump_row(row: Mapping[str, object]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json_dump_row(row))
            fh.write("\n")
            count += 1
    return count


def number_literals(text: str) -> list[str]:
    return [match.group(0) for match in NUMBER_RE.finditer(text)]


def normalize_number(value: str) -> str:
    return value.replace(",", "").rstrip("%")


def first_phrase(text: str) -> str | None:
    lowered = text.lower()
    for phrase in REFUTATION_PHRASES:
        if phrase in lowered:
            return phrase
    return None


def unique_numbers(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        norm = normalize_number(value)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(value)
    return out


def correction_rows_for_text(source: str, text: str, *, radius: int = 2) -> list[dict[str, object]]:
    """Return refutation-adjacent numeric rows for one text blob."""

    lines = text.splitlines()
    rows: list[dict[str, object]] = []
    for idx, line in enumerate(lines):
        phrase = first_phrase(line)
        if phrase is None:
            continue
        start = max(0, idx - radius)
        end = min(len(lines), idx + radius + 1)
        window = "\n".join(lines[start:end])
        nums = unique_numbers(number_literals(window))
        if len(nums) < 2:
            continue
        rows.append(
            {
                "source": source,
                "line": idx + 1,
                "phrase": phrase,
                "refuted_value": nums[0],
                "corrected_value": nums[1],
                "numeric_literals": nums,
                "window_start": start + 1,
                "window_end": end,
            }
        )
    return rows


def iter_markdown_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.md")):
        if ".git" in path.parts:
            continue
        yield path


def corrections_index(research_root: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows: list[dict[str, object]] = []
    scanned = 0
    matched_files: set[str] = set()
    for path in iter_markdown_files(research_root):
        scanned += 1
        source = repo_rel(path)
        file_rows = correction_rows_for_text(source, read_text(path))
        if file_rows:
            matched_files.add(source)
            rows.extend(file_rows)
    return rows, {
        "memos_scanned": scanned,
        "memos_matched": len(matched_files),
        "rows_emitted": len(rows),
    }


def split_title_body(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip(), "\n".join(lines[idx + 1 :])
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            return stripped, "\n".join(lines[idx + 1 :])
    return "", ""


def _lowered_hits(text: str, vocabulary: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    return [word for word in vocabulary if word in lowered]


def _word_spans(text: str, vocabulary: tuple[str, ...]) -> list[tuple[str, int]]:
    """Whole-word occurrences with their offsets. Substring matching over-fires badly here."""
    lowered = text.lower()
    found: list[tuple[str, int]] = []
    for word in vocabulary:
        for match in re.finditer(rf"\b{re.escape(word)}\b", lowered):
            found.append((word, match.start()))
    return found


def scope_word_rows(*, source: str, text: str) -> tuple[list[dict[str, object]], int]:
    """Flag closure sentences that name a WIDE object without stating their scope.

    Returns ``(rows, scoped_count)``. ``scoped_count`` is the honest denominator: closure
    sentences that DID carry a qualifier, i.e. the ones the gauge deliberately let through.
    Without it a reader cannot tell a clean corpus from a broken detector.
    """
    rows: list[dict[str, object]] = []
    scoped = 0
    for idx, line in enumerate(text.splitlines()):
        stripped = line.strip()
        # Skip table rules and blockquotes: a quoted or ruled line is not this memo asserting.
        if not stripped or stripped.startswith(("|---", "> ", "#")):
            continue
        lowered = stripped.lower()
        # Strip our own idioms before looking, so the house dialect cannot trip the gauge.
        probe = lowered
        for friend in CLOSURE_FALSE_FRIENDS:
            probe = probe.replace(friend, " ")
        closure_spans = _word_spans(probe, CLOSURE_WORDS)
        if not closure_spans:
            continue
        wide_spans = _word_spans(probe, WIDE_OBJECT_WORDS)
        if not wide_spans:
            continue
        near = [
            (closure, wide)
            for closure, c_at in closure_spans
            for wide, w_at in wide_spans
            if abs(c_at - w_at) <= SCOPE_PROXIMITY_CHARS
        ]
        if not near:
            continue
        qualifiers = _lowered_hits(stripped, SCOPE_QUALIFIERS)
        if qualifiers:
            scoped += 1
            continue
        rows.append(
            {
                "source": source,
                "line": idx + 1,
                "closure_words": sorted({closure for closure, _ in near}),
                "wide_object_words": sorted({wide for _, wide in near}),
                "sentence": stripped[:400],
                "question": (
                    "which object did this unit MEASURE, and is that the object this sentence NAMES?"
                ),
                "advisory_only": True,
            }
        )
    return rows, scoped


def scope_vs_object_detector(research_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run the scope-word pass over the memo corpus. Warn-only, denominators reported."""
    rows: list[dict[str, object]] = []
    scanned = 0
    scoped_total = 0
    flagged_sources: set[str] = set()
    for path in iter_markdown_files(research_root):
        scanned += 1
        source = repo_rel(path)
        found, scoped = scope_word_rows(source=source, text=read_text(path))
        scoped_total += scoped
        if found:
            rows.extend(found)
            flagged_sources.add(source)

    # Positive control: the na9 F5 instance #4 sentence must be detectable in the shape the
    # detector looks for. A gauge whose known-positive is invisible is not a gauge.
    control_rows, _ = scope_word_rows(
        source="<positive_control>", text="The coder axis is CLOSED on hv1."
    )
    # Negative control: the SAME claim, correctly scoped, must NOT fire.
    negative_rows, negative_scoped = scope_word_rows(
        source="<negative_control>",
        text="The coder-swap axis is CLOSED given fixed probabilities.",
    )

    summary = {
        "memos_scanned": scanned,
        "rows_emitted": len(rows),
        "sources_flagged": len(flagged_sources),
        "closure_sentences_already_scoped": scoped_total,
        "positive_control_caught": bool(control_rows),
        "negative_control_caught": bool(negative_rows),
        "negative_control_recognised_as_scoped": negative_scoped == 1,
        "advisory_only": True,
    }
    return rows, summary


def body_refutation_different_number(
    *,
    source: str,
    source_kind: str,
    title: str,
    body: str,
) -> list[dict[str, object]]:
    title_nums = unique_numbers(number_literals(title))
    if not title_nums:
        return []
    title_norms = {normalize_number(value) for value in title_nums}
    rows: list[dict[str, object]] = []
    lines = body.splitlines()
    for idx, line in enumerate(lines):
        phrase = first_phrase(line)
        if phrase is None:
            continue
        start = max(0, idx - 2)
        end = min(len(lines), idx + 3)
        window = "\n".join(lines[start:end])
        window_nums = unique_numbers(number_literals(window))
        body_nums = [value for value in window_nums if normalize_number(value) not in title_norms]
        if not body_nums:
            continue
        rows.append(
            {
                "source": source,
                "source_kind": source_kind,
                "title": title,
                "title_numeric_literals": title_nums,
                "phrase": phrase,
                "line": idx + 1,
                "body_different_numeric_literals": body_nums,
                "body_numeric_literals": window_nums,
            }
        )
    return rows


def task_ledger_rows(path: Path) -> Iterator[tuple[int, Mapping[str, object]]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield line_no, payload


def ledger_title_body(row: Mapping[str, object]) -> tuple[str, str]:
    title_parts: list[str] = []
    for key in ("id", "task_id", "task", "title", "subject", "name", "lane", "status"):
        value = row.get(key)
        if value is not None:
            title_parts.append(str(value))
    body_parts: list[str] = []
    for key in ("description", "notes", "event_notes", "summary", "receipts", "evidence"):
        value = row.get(key)
        if value is not None:
            body_parts.append(str(value))
    return " | ".join(title_parts), "\n".join(body_parts)


def git_log_subjects(pathspec: str) -> list[tuple[str, str]]:
    result = subprocess.run(
        ["git", "log", "--oneline", "--", pathspec],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows: list[tuple[str, str]] = []
    if result.returncode != 0:
        return rows
    for line in result.stdout.splitlines():
        if " " not in line:
            continue
        sha, subject = line.split(" ", 1)
        rows.append((sha, subject))
    return rows


def split_subject_on_refutation(subject: str) -> tuple[str, str] | None:
    lowered = subject.lower()
    positions = [lowered.find(phrase) for phrase in REFUTATION_PHRASES if phrase in lowered]
    positions = [pos for pos in positions if pos >= 0]
    if not positions:
        return None
    pos = min(positions)
    return subject[:pos].strip(), subject[pos:].strip()


def headline_body_detector(research_root: Path, ledger_path: Path) -> tuple[list[dict[str, object]], dict[str, int | bool]]:
    rows: list[dict[str, object]] = []
    memos_scanned = 0
    task_rows_scanned = 0
    git_subjects_scanned = 0

    for path in iter_markdown_files(research_root):
        memos_scanned += 1
        title, body = split_title_body(read_text(path))
        rows.extend(
            body_refutation_different_number(
                source=repo_rel(path),
                source_kind="research_memo",
                title=title,
                body=body,
            )
        )

    for line_no, row in task_ledger_rows(ledger_path):
        task_rows_scanned += 1
        title, body = ledger_title_body(row)
        rows.extend(
            body_refutation_different_number(
                source=f"{repo_rel(ledger_path)}:{line_no}",
                source_kind="task_ledger_row",
                title=title,
                body=body,
            )
        )

    for sha, subject in git_log_subjects(".omx/state/canonical_task_status.jsonl"):
        git_subjects_scanned += 1
        split = split_subject_on_refutation(subject)
        if split is None:
            continue
        title, body = split
        rows.extend(
            body_refutation_different_number(
                source=f"git_log:.omx/state/canonical_task_status.jsonl:{sha}",
                source_kind="task_ledger_git_subject",
                title=title,
                body=body,
            )
        )

    control_933 = any(
        row.get("source_kind") == "task_ledger_git_subject"
        and "23,655" in json.dumps(row, sort_keys=True)
        and "24,605" in json.dumps(row, sort_keys=True)
        for row in rows
    )
    control_931_caught = any(
        "#931" in json.dumps(row, sort_keys=True) or " 931 " in json.dumps(row, sort_keys=True)
        for row in rows
    )
    return rows, {
        "memos_scanned": memos_scanned,
        "task_rows_scanned": task_rows_scanned,
        "git_subjects_scanned": git_subjects_scanned,
        "rows_emitted": len(rows),
        "positive_control_933_caught": control_933,
        "negative_control_931_caught": control_931_caught,
    }


@dataclass(frozen=True)
class DecodeReadSeed:
    read_id: str
    label: str
    classification: str
    governing_status: str
    source_files: tuple[str, ...]
    needles: tuple[str, ...]
    recall_source: str
    positive_control: str = ""


DECODE_READ_SEED: tuple[DecodeReadSeed, ...] = (
    DecodeReadSeed(
        "token_quant_levels_default_16",
        "TR1/R7 token_quant_levels default 16",
        "NONE",
        "unladdered governance knob; #933 producer-side L=14 byte win queued for scorer",
        ("src/tac/optimization/ddm_tr1_runtime.py", "experiments/ddm_r7_token_coder.py"),
        ("token_quant_levels", "smevr"),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
        "#933_L16",
    ),
    DecodeReadSeed(
        "token_range_pm1_clamp",
        "TR1 token field clip to +/-1.0",
        "NONE",
        "unladdered governance knob; #933 measured 33.3 percent token mass pinned",
        ("src/tac/optimization/ddm_tr1_runtime.py",),
        ("np.clip(combined, np.float32(-1), np.float32(1))",),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
        "#933_pm1_clamp",
    ),
    DecodeReadSeed(
        "window_solve_default_off",
        "receiver window_solve default OFF",
        "NONE",
        "unladdered governance knob with measured local benefit in docstring",
        ("src/tac/optimization/ddm_tr1_runtime.py",),
        ("window_solve: bool = False", "window_solve"),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
    DecodeReadSeed(
        "st_grid_support",
        "PFS1 ST_GRID support knots",
        "NONE",
        "unladdered generic/fitted support choice; gk2 says GuardedConstant destination after measurement",
        ("src/tac/optimization/pfs1_warp_receiver.py", "src/tac/optimization/ddm_ix2_archive_container.py"),
        ("ST_GRID", "st_grid"),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
    DecodeReadSeed(
        "ix2_lzma_filters",
        "IX2 LZMA filter tuple",
        "NONE",
        "unladdered entropy-coder race constant",
        ("src/tac/optimization/ddm_ix2_archive_container.py",),
        ("_LZMA_FILTERS", "lzma.FILTER_LZMA1"),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
    DecodeReadSeed(
        "default_beta_mags",
        "v4d DEFAULT_BETA_MAGS fallback",
        "NONE",
        "unladdered beta magnitude table fallback",
        ("experiments/inflate_runner_v4d.py",),
        ("DEFAULT_BETA_MAGS",),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
    DecodeReadSeed(
        "token_codec_smevr_default",
        "R7 token codec default/pin to smevr",
        "NONE",
        "unladdered coder selection default",
        ("experiments/ddm_r7_token_coder.py", "src/tac/optimization/ddm_tr1_runtime.py"),
        ('DEFAULT_CODEC: Final = "smevr"', 'codec="smevr"'),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
    DecodeReadSeed(
        "smevr_rescale_at",
        "R7 SMEVR arithmetic rescale threshold",
        "NONE",
        "unladdered arithmetic-coder state threshold",
        ("experiments/ddm_r7_token_coder.py",),
        ("_RESCALE_AT",),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
    DecodeReadSeed(
        "ix2_code_block_coder_race",
        "IX2 code_block selects shortest coded block",
        "swept",
        "coder race explicitly swept/implemented in IX2 container",
        ("src/tac/optimization/ddm_ix2_archive_container.py",),
        ("def code_block", "brotli.compress", "lzma.compress"),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
    DecodeReadSeed(
        "ix2_exact_table_encoding",
        "IX2 exact-table precision encoding",
        "swept",
        "precision ladder encoded by exact-table apparatus",
        ("src/tac/optimization/ddm_ix2_archive_container.py",),
        ("def encode_exact_table", "pose_dim0_offset"),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
    DecodeReadSeed(
        "tr1_brotli_q11",
        "TR1 raw frame Brotli q11 block",
        "swept",
        "known lossless coder surface, not a live unguarded knob",
        ("src/tac/optimization/ddm_tr1_runtime.py",),
        ("brotli.compress(raw, quality=11)",),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
    DecodeReadSeed(
        "ix2_brotli_q11_lgwin24",
        "IX2 Brotli q11 lgwin24 candidate",
        "swept",
        "candidate inside code_block shortest-block race",
        ("src/tac/optimization/ddm_ix2_archive_container.py",),
        ("quality=11, lgwin=24",),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
    DecodeReadSeed(
        "warp_bicubic_round_clip",
        "PFS1 warp uint8 round/clip",
        "levered",
        "receiver-realization operator is the measured/levered R surface",
        ("src/tac/optimization/pfs1_warp_receiver.py",),
        ("np.clip(np.round(frame_f)",),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
    DecodeReadSeed(
        "tr1_camera_round_clip",
        "TR1 camera uint8 round/clip",
        "levered",
        "receiver-realization operator is the measured/levered R surface",
        ("src/tac/optimization/ddm_tr1_runtime.py",),
        ("np.clip(np.rint(up)",),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
    DecodeReadSeed(
        "manifest_frame0_policy",
        "v4d manifest frame0_policy read",
        "flagged",
        "manifest-read genus flagged by TR1/v4d map as live named key",
        ("experiments/inflate_runner_v4d.py",),
        ('manifest["frame0_policy"]',),
        ".omx/research/ddm_bs3_full_scope_wrong_projection_genus_20260803.md",
    ),
    DecodeReadSeed(
        "manifest_pose_dim0_offset",
        "v4d manifest pose_dim0_offset read",
        "levered",
        "counted/fitted live manifest value",
        ("experiments/inflate_runner_v4d.py", "src/tac/optimization/ddm_ix2_archive_container.py"),
        ("pose_dim0_offset",),
        ".omx/research/ddm_bs3_full_scope_wrong_projection_genus_20260803.md",
    ),
    DecodeReadSeed(
        "manifest_rs_beta_mags",
        "v4d manifest rs_beta_mags read",
        "levered",
        "counted/fitted beta magnitude value",
        ("experiments/inflate_runner_v4d.py", "src/tac/optimization/ddm_ix2_archive_container.py"),
        ("rs_beta_mags",),
        ".omx/research/ddm_bs3_full_scope_wrong_projection_genus_20260803.md",
    ),
    DecodeReadSeed(
        "manifest_st_grid",
        "v4d manifest st_grid read",
        "levered",
        "counted/fitted or generic migrated ST grid value",
        ("experiments/inflate_runner_v4d.py", "src/tac/optimization/ddm_ix2_archive_container.py"),
        ("st_grid",),
        ".omx/research/ddm_bs3_full_scope_wrong_projection_genus_20260803.md",
    ),
    DecodeReadSeed(
        "manifest_tr1_metadata",
        "v4d manifest tr1_metadata read",
        "flagged",
        "live manifest read flagged by TR1/v4d map; not a governance knob in gk2 denominator",
        ("experiments/inflate_runner_v4d.py", "src/tac/optimization/ddm_ix2_archive_container.py"),
        ("tr1_metadata",),
        ".omx/research/ddm_bs3_full_scope_wrong_projection_genus_20260803.md",
    ),
    DecodeReadSeed(
        "ix2_section_count_decode",
        "IX2 joint section-count decode",
        "swept",
        "container parse-back surface with deterministic section count",
        ("src/tac/optimization/ddm_ix2_archive_container.py",),
        ("unpack_container", "count"),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
    DecodeReadSeed(
        "ix2_residual_nibble_unpack",
        "IX2 residual nibble unpack shape",
        "swept",
        "lossless parse-back surface",
        ("src/tac/optimization/ddm_ix2_archive_container.py",),
        ("_unpack_nibbles", "residual"),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
    DecodeReadSeed(
        "ix2_base_nibble_unpack",
        "IX2 base nibble unpack shape",
        "swept",
        "lossless parse-back surface",
        ("src/tac/optimization/ddm_ix2_archive_container.py",),
        ("_unpack_nibbles", "base"),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
    DecodeReadSeed(
        "r7_decode_smevr_reference",
        "R7 SMEVR decode reference/oracle",
        "swept",
        "byte/array oracle retained for parse-back parity",
        ("experiments/ddm_r7_token_coder.py",),
        ("def _decode_smevr_reference", "def _decode_smevr"),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
    DecodeReadSeed(
        "r7_encode_smevr_reference",
        "R7 SMEVR encode reference/oracle",
        "swept",
        "byte oracle retained for parse-back parity",
        ("experiments/ddm_r7_token_coder.py",),
        ("def _encode_smevr_reference", "def _encode_smevr"),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
    DecodeReadSeed(
        "tr1_selector_token_quant_levels_read",
        "TR1 selector token_quant_levels decode read",
        "flagged",
        "same receiver-read family as #933; flagged by gk2 as requiring GC once measured",
        ("src/tac/optimization/ddm_tr1_runtime.py",),
        ('selector["token_quant_levels"]',),
        ".omx/research/ddm_gk2_decode_read_coverage_20260804.md",
    ),
)


def verify_seed_site(seed: DecodeReadSeed) -> tuple[bool, list[str]]:
    missing: list[str] = []
    combined = ""
    for rel in seed.source_files:
        path = REPO_ROOT / rel
        if not path.exists():
            missing.append(f"{rel}:missing-file")
            continue
        combined += read_text(path)
    for needle in seed.needles:
        if needle not in combined:
            missing.append(needle)
    return not missing, missing


def decode_read_coverage_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in DECODE_READ_SEED:
        verified, missing = verify_seed_site(seed)
        rows.append(
            {
                "read_id": seed.read_id,
                "label": seed.label,
                "classification": seed.classification,
                "governing_status": seed.governing_status,
                "source_files": list(seed.source_files),
                "needles": list(seed.needles),
                "source_site_verified": verified,
                "missing_needles": missing,
                "recall_source": seed.recall_source,
                "positive_control": seed.positive_control,
            }
        )
    by_class: dict[str, int] = {}
    for row in rows:
        key = str(row["classification"])
        by_class[key] = by_class.get(key, 0) + 1
    summary = {
        "reads_enumerated": len(rows),
        "reads_classified": len(rows),
        "by_classification": by_class,
        "source_sites_verified": sum(1 for row in rows if row["source_site_verified"]),
        "positive_control_933_pm1": any(
            row["positive_control"] == "#933_pm1_clamp" and row["classification"] == "NONE"
            for row in rows
        ),
        "positive_control_933_l16": any(
            row["positive_control"] == "#933_L16" and row["classification"] == "NONE"
            for row in rows
        ),
    }
    return rows, summary


def git_tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return []
    return result.stdout.splitlines()


def registry_first_rows(research_root: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    tracked = git_tracked_files()
    repo_codex = [path for path in tracked if Path(path).name.startswith("codex_findings_") and path.endswith(".md")]
    research_codex = [path for path in repo_codex if path.startswith(".omx/research/")]
    local_research_codex = sorted(path.as_posix() for path in research_root.glob("codex_findings_*.md"))
    rows = [
        {
            "leg": "registry_first_extension",
            "task": "#840",
            "collision_key": "ddm_cf1",
            "original_referent": ".omx/research/ddm_cf1_coarse_framing_audit_20260731.md",
            "original_referent_meaning": "2026-07-31 coarse-framing audit arm; task #840 unswept 91 percent extension",
            "new_referent": ".omx/research/ddm_cf1_20260805/CF1_CROSSWALK_RECEIPT.md",
            "new_referent_meaning": "2026-08-05 conformal-anomaly crosswalk arm; not task #840",
            "collision_recorded": True,
            "tracked_codex_findings_repo_wide": len(repo_codex),
            "tracked_codex_findings_under_research": len(research_codex),
            "workspace_codex_findings_under_research": len(local_research_codex),
            "folded_into_leg2": "partial: AU1 scans research memos for correction/headline genus; general coarse-framing classifier remains separate",
        }
    ]
    return rows, {
        "tracked_codex_findings_repo_wide": len(repo_codex),
        "tracked_codex_findings_under_research": len(research_codex),
        "workspace_codex_findings_under_research": len(local_research_codex),
        "rows_emitted": len(rows),
    }


TEXT_EXTENSIONS = {
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".rst",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
}


def current_history_counts() -> tuple[int | None, int | None]:
    wc_result = subprocess.run(
        ["zsh", "-lc", "git log --oneline | wc -l"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rev_result = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    wc_count: int | None = None
    rev_count: int | None = None
    if wc_result.returncode == 0:
        try:
            wc_count = int(wc_result.stdout.strip())
        except ValueError:
            wc_count = None
    if rev_result.returncode == 0:
        try:
            rev_count = int(rev_result.stdout.strip())
        except ValueError:
            rev_count = None
    return wc_count, rev_count


def exact_string_consumers(needle: str) -> list[str]:
    consumers: list[str] = []
    for rel in git_tracked_files():
        path = REPO_ROOT / rel
        if path.suffix not in TEXT_EXTENSIONS or not path.exists():
            continue
        try:
            text = read_text(path)
        except OSError:
            continue
        if needle in text:
            consumers.append(rel)
    return consumers


def if_budget_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    wc_count, rev_count = current_history_counts()
    phase_residual = "src/tac/boundary_math/phase_residual_carrier.py"
    dash_phase = "src/tac/boundary_math/dash_phase_carrier.py"
    phase_hook = "tools/levelset_byte_close_and_eval.py"
    rows = [
        {
            "task": "#885",
            "check": "git history count symptom",
            "current_git_log_oneline_wc_l": wc_count,
            "current_git_rev_list_count": rev_count,
            "exact_consuming_sites": exact_string_consumers("git log --oneline | wc -l"),
            "verdict": (
                "current checkout did not reproduce the 50-vs-true symptom; exact command "
                "consumers are flagged for any future history-denominator claim"
            ),
        },
        {
            "task": "#867",
            "check": "#425 phase carrier disambiguation",
            "referent": "#425 phase carrier",
            "module_referents": [
                {
                    "path": phase_residual,
                    "role": "raster tie-field residual store-side carrier",
                    "exists": (REPO_ROOT / phase_residual).exists(),
                },
                {
                    "path": dash_phase,
                    "role": "curve-domain per-dash store-side carrier",
                    "exists": (REPO_ROOT / dash_phase).exists(),
                },
            ],
            "hook_path": phase_hook,
            "hook_mentions_both_flags": (
                (REPO_ROOT / phase_hook).exists()
                and "--phase-carrier" in read_text(REPO_ROOT / phase_hook)
                and "--dash-phase-carrier" in read_text(REPO_ROOT / phase_hook)
            ),
            "retag": "Use phase_residual_carrier_425 or dash_phase_carrier_425; avoid naked '#425 phase carrier'.",
        },
    ]
    return rows, {
        "rows_emitted": len(rows),
        "task_885_current_git_log_oneline_wc_l": wc_count,
        "task_885_current_git_rev_list_count": rev_count,
        "task_867_module_referents": 2,
    }


def build_summary(
    *,
    decode_summary: Mapping[str, object],
    corrections_summary: Mapping[str, object],
    headline_summary: Mapping[str, object],
    registry_summary: Mapping[str, object],
    if_budget_summary: Mapping[str, object],
    scope_summary: Mapping[str, object],
) -> dict[str, object]:
    return {
        "audit": "AU1 measurement-integrity audit",
        "scorer_evaluate_launch": "not_run",
        "decode_read_coverage": dict(decode_summary),
        "corrections_index": dict(corrections_summary),
        "headline_vs_body": dict(headline_summary),
        "registry_first_extension": dict(registry_summary),
        "if_budget_checks": dict(if_budget_summary),
        "scope_vs_object": dict(scope_summary),
    }


def run(outdir: Path) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)

    decode_rows, decode_summary = decode_read_coverage_rows()
    corrections_rows, corrections_summary = corrections_index(RESEARCH_ROOT)
    headline_rows, headline_summary = headline_body_detector(RESEARCH_ROOT, TASK_LEDGER)
    registry_rows, registry_summary = registry_first_rows(RESEARCH_ROOT)
    if_budget_check_rows, if_budget_summary = if_budget_rows()
    scope_rows, scope_summary = scope_vs_object_detector(RESEARCH_ROOT)

    paths = {
        "decode_read_coverage": outdir / "au1_decode_read_coverage.jsonl",
        "corrections_index": outdir / "au1_corrections_index.jsonl",
        "headline_vs_body": outdir / "au1_headline_vs_body.jsonl",
        "registry_first_extension": outdir / "au1_registry_first_extension.jsonl",
        "if_budget_checks": outdir / "au1_if_budget_checks.jsonl",
        "scope_vs_object": outdir / "au1_scope_vs_object.jsonl",
        "summary": outdir / "au1_summary.json",
    }
    write_jsonl(paths["decode_read_coverage"], decode_rows)
    write_jsonl(paths["corrections_index"], corrections_rows)
    write_jsonl(paths["headline_vs_body"], headline_rows)
    write_jsonl(paths["registry_first_extension"], registry_rows)
    write_jsonl(paths["if_budget_checks"], if_budget_check_rows)
    write_jsonl(paths["scope_vs_object"], scope_rows)
    summary = build_summary(
        decode_summary=decode_summary,
        corrections_summary=corrections_summary,
        headline_summary=headline_summary,
        registry_summary=registry_summary,
        if_budget_summary=if_budget_summary,
        scope_summary=scope_summary,
    )
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--summary-json", action="store_true")
    args = parser.parse_args(argv)

    summary = run(args.outdir)
    if args.summary_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
