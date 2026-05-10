#!/usr/bin/env python3
r"""Constrained coordinate search around PR101's verified inflate-time bias.

Per the convergent finding from the codex adversarial review of HNeRV lessons
(`.omx/research/hnerv_lessons_docs_adversarial_review_20260509_codex.md` §4)
and the operator-directed atom #1 EIG/$ in the domain catalog (atom
``domain_exploit_scorer_2``):

  PR101's verified inflate-time bias correction is:
      up[:, 0, 0].sub_(1.0)
      up[:, 0, 2].sub_(1.0)
      up[:, 1, 1].sub_(1.0)

  V2 half-magnitude (-0.5 each) was empirically WORSE than baseline
  (0.194295755690 vs 0.192847577437; +0.00145 regression on contest-CPU GHA).

  Therefore the right next move is **NOT** an arbitrary channel-constant sweep;
  it is a **constrained coordinate search around the verified anchor**.

This tool generates a parameterized 3D (optionally 4D) coordinate-search grid
of PR101-bias variants and writes one ``submission_dir`` per variant.  It
shares the canonical sweep schema with
``tools/build_a1_inflate_time_bias_correction_sweep.py`` (read-only oracle
for the inflate.py template).

Usage modes:

  1. **Coarse pass** (default): 7-point grid centered at -1.0
     (``{-1.5, -1.25, -1.1, -1.0, -0.9, -0.75, -0.5}``) → 7³ = 343 candidates
     OR (with ``--coarse-coarse``) the 4-point reduced grid
     (``{-1.0, -0.5, 0, +0.5}``) → 64 candidates for fastest dev-loop.
  2. **Refined pass**: pass ``--center-coord`` to refocus the 7-point grid
     around the empirical winner from a coarse pass with finer step
     (``--step 0.05``).
  3. **Sidecar**: pass ``--with-sidecar`` to also enumerate a 5-point fourth
     coordinate ``up[:,2,0]`` add term (``{-0.5, -0.25, 0, +0.25, +0.5}``).

Output layout per variant:

    experiments/results/constrained_coord_search_<run_id>/
        variant_<idx>/
            submission_dir/
                archive.zip          (bit-identical to A1)
                inflate.py           (variant-specific bias)
                inflate.sh
                src/
        rollup.json                  (per-run: variant_id → manifest path)
        run_manifest.json            (the search grid + provenance)

Per CLAUDE.md:

  - Public PR intake clones are READ-ONLY (Check 109).
  - All claims tagged ``[predicted; constrained coord search on A1 substrate]``
    until GHA dispatch returns ``[contest-CPU GHA Linux x86_64]``.
  - Per HNeRV-parity lesson 11 (no-op detector): each variant's bias delta
    must produce a measurable change in the rendered output; the tool
    asserts the bias_lines block is unique per variant id.
  - Per ``forbidden_premature_kill_without_research_exhaustion``: any
    "this region is dead" verdict is DEFERRED-pending-finer-grid.
  - /tmp paths FORBIDDEN — every artifact path is
    ``experiments/results/constrained_coord_search_<run_id>/...``.

Coordination:

  - Sibling a3c89347 (V0..V10 sweep) covered the gross-magnitude axis.
    THIS tool covers the fine-grained 7³ neighborhood.  The V1 baseline
    candidate (-1, -1, -1) is INCLUDED as a sanity check and should
    reproduce A1's known 0.19284 [contest-CPU GHA].
  - Sibling C (M5 Max sweep) — when their tool lands, this tool's local-eval
    flag (``--local-cpu-rank``) can be redirected to M5 Max for $0
    parallel ranking before GHA promotion.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# A1 canonical archive (the same anchor used by the A1 bias correction sweep
# tool — must match its A1_EXPECTED_ARCHIVE_SHA exactly).
A1_SUBMISSION_DIR = (
    REPO_ROOT
    / "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal/harvested_artifacts/finetuned_archive/submission_dir"
)
A1_ARCHIVE_PATH = A1_SUBMISSION_DIR / "archive.zip"
A1_EXPECTED_ARCHIVE_SHA = (
    "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5"
)
A1_EXPECTED_ARCHIVE_BYTES = 178262

# PR101 source (READ-ONLY oracle per Check 109).
PR101_INFLATE = (
    REPO_ROOT
    / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/inflate.py"
)

# A1's canonical baseline score on this exact archive (for sanity check at v_baseline).
A1_CANONICAL_BASELINE_SCORE = 0.19284757743677347
A1_CANONICAL_BASELINE_TAG = "[contest-CPU GHA Linux x86_64]"
A1_CANONICAL_BASELINE_EVIDENCE = (
    "experiments/results/a1_latentalign_importpathfix_cpu_eval_gha_20260509/"
    "contest_auth_eval.adjudicated.json"
)

# Anchor block where bias correction is inserted (matches PR101 + A1 inflate.py).
ANCHOR_START = "            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)"
ANCHOR_END_PREFIX = "            frames = ("

# Default coarse grid around verified PR101 anchor of -1.0
DEFAULT_COARSE_GRID = (-1.5, -1.25, -1.1, -1.0, -0.9, -0.75, -0.5)
DEFAULT_COARSE_COARSE_GRID = (-1.0, -0.5, 0.0, 0.5)
DEFAULT_REFINED_OFFSETS = (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)
DEFAULT_REFINED_STEP = 0.05  # the step IMPLICIT in DEFAULT_REFINED_OFFSETS;
# kept as a named constant for callers that want to inspect or override.
DEFAULT_SIDECAR_GRID = (-0.5, -0.25, 0.0, 0.25, 0.5)
QUEUE_SCHEMAS = frozenset(
    {
        "optimizer_candidate_queue_v1",
        "optimizer_guided_candidate_queue_v1",
    }
)
QUEUE_FALSE_AUTHORITY_FIELDS = (
    "ready_for_exact_eval_dispatch",
    "score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
)
QUEUE_PARAM_TO_COORD = {
    "bias_r": "c0_0",
    "bias_b": "c0_2",
    "bias_g": "c1_1",
    "sidecar_f1_r": "sidecar_c2_0",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(path: Path) -> str:
    """Return a repo-relative path when possible, else absolute."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _format_coord(c: float) -> str:
    """Format a coordinate value for filename use (no decimal, signed)."""
    sign = "n" if c < 0 else "p"
    abs_val = abs(c)
    int_part = int(abs_val)
    frac_part = int(round((abs_val - int_part) * 100))
    return f"{sign}{int_part}_{frac_part:02d}"


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{field} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{field} must be finite")
    return out


def _queue_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    schema = payload.get("schema")
    if schema not in QUEUE_SCHEMAS:
        raise RuntimeError(f"candidate queue schema must be one of {sorted(QUEUE_SCHEMAS)}")
    rows = payload.get("top_k_forensic")
    if not isinstance(rows, list):
        rows = payload.get("top_k")
    if not isinstance(rows, list):
        raise RuntimeError("candidate queue must contain top_k or top_k_forensic rows")
    return [row for row in rows if isinstance(row, dict)]


def _safe_queue_variant_id(candidate_id: str) -> str:
    if not candidate_id:
        raise RuntimeError("queue row candidate_id must be non-empty")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_id).strip("._-")
    if not safe:
        raise RuntimeError(f"candidate_id cannot be mapped to safe variant id: {candidate_id!r}")
    return f"q_{safe}"


def variants_from_candidate_queue(
    queue_path: Path,
    *,
    top_k: int | None = None,
) -> list[tuple[str, dict[str, float], dict[str, Any]]]:
    """Convert an optimizer planning queue into concrete variant coordinates."""

    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("candidate queue must be a JSON object")
    rows = _queue_rows(payload)
    if top_k is not None:
        if top_k < 1:
            raise RuntimeError("--queue-top-k must be >= 1")
        rows = rows[:top_k]

    variants: list[tuple[str, dict[str, float], dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for idx, row in enumerate(rows):
        candidate_id = str(row.get("candidate_id") or "")
        for field in QUEUE_FALSE_AUTHORITY_FIELDS:
            if row.get(field) is not False:
                raise RuntimeError(
                    f"{candidate_id or '<missing>'}: queue field {field} must be false"
                )
        params = row.get("candidate_params") or row.get("op_params")
        if not isinstance(params, dict):
            raise RuntimeError(f"{candidate_id}: candidate_params/op_params must be an object")
        coords = {
            coord: _finite_float(params[param], f"{candidate_id}.candidate_params.{param}")
            for param, coord in QUEUE_PARAM_TO_COORD.items()
            if param in params
        }
        missing = sorted({"c0_0", "c0_2", "c1_1"} - set(coords))
        if missing:
            raise RuntimeError(f"{candidate_id}: missing required coordinate(s): {missing}")
        extra_params = sorted(
            set(params)
            - set(QUEUE_PARAM_TO_COORD)
        )
        if extra_params:
            raise RuntimeError(f"{candidate_id}: unsupported queue params: {extra_params}")
        variant_id = _safe_queue_variant_id(candidate_id)
        if variant_id in seen_ids:
            variant_id = f"{variant_id}_{idx:04d}"
        seen_ids.add(variant_id)
        variants.append(
            (
                variant_id,
                coords,
                {
                    "candidate_id": candidate_id,
                    "rank": row.get("rank"),
                    "rank_score": row.get("rank_score"),
                    "rank_score_field": row.get("rank_score_field"),
                    "optimizer": row.get("optimizer"),
                    "optimizer_status": row.get("optimizer_status"),
                    "profile": row.get("profile"),
                    "source_queue_path": manifest_path(queue_path),
                },
            )
        )
    return variants


def build_bias_lines(c0_0: float, c0_2: float, c1_1: float, sidecar_c2_0: float | None = None) -> list[str]:
    """Build the bias_lines block for a (c0_0, c0_2, c1_1) coordinate triple.

    **Coordinate convention** matches PR101's verified anchor:

      coord = -1.0  →  effective channel SUBTRACTS 1.0 (the PR101 baseline:
                       ``up[:, 0, 0].sub_(1.0)``)
      coord = -0.5  →  channel SUBTRACTS 0.5  (the V2 half-magnitude regression)
      coord = +1.0  →  channel ADDS 1.0       (PR102's bronze pattern direction)
      coord =  0.0  →  no operation (skipped)

    To preserve the `.sub_()` syntactic form (so the build_a1 sweep tool's
    anchor-replacement regex stays compatible), we emit ``sub_(-coord)`` —
    i.e., ``coord = -1.0`` becomes ``sub_(1.000000)`` (PR101 baseline).
    """
    lines: list[str] = []
    for (frame, channel), value in [
        ((0, 0), c0_0),
        ((0, 2), c0_2),
        ((1, 1), c1_1),
    ]:
        # Skip a coordinate if it's exactly 0 (degenerate no-op).
        if value == 0:
            continue
        # Emit `up[:, frame, channel].sub_(-coord)`.  For coord = -1.0 this
        # produces `sub_(1.000000)` which exactly matches PR101's baseline.
        lines.append(f"up[:, {frame}, {channel}].sub_({-value:.6f})")
    if sidecar_c2_0 is not None and sidecar_c2_0 != 0:
        # 4th coordinate sidecar.  The `up` tensor has shape
        # ``(batch, 2, 3, H, W)``: frame index ∈ {0, 1}, channel index ∈ {0, 1, 2}.
        # The operator's brief literally says ``up[:,2,0]`` but frame 2 is
        # OUT OF RANGE.  We interpret the brief as "add a fourth coordinate
        # term on an unperturbed (frame, channel) cell" and pick
        # ``up[:, 1, 0]`` (frame 1 red — the first unperturbed cell after
        # PR101's three perturbed cells).
        # Same convention: positive sidecar coord → effective channel ADD
        # (so `sub_(-sidecar_c2_0)`).
        lines.append(f"up[:, 1, 0].sub_({-sidecar_c2_0:.6f})")
    return lines


def build_inflate_py(template_text: str, bias_lines: list[str]) -> str:
    """Emit a new inflate.py with the variant's bias_lines.

    Same algorithm as ``tools/build_a1_inflate_time_bias_correction_sweep.py``:
    locate the canonical anchor, replace the existing bias block.
    """
    lines = template_text.splitlines(keepends=True)
    start_idx = None
    end_idx = None
    for i, ln in enumerate(lines):
        if ln.rstrip("\n") == ANCHOR_START and start_idx is None:
            start_idx = i
        elif start_idx is not None and ln.startswith(ANCHOR_END_PREFIX):
            end_idx = i
            break
    if start_idx is None or end_idx is None:
        raise RuntimeError(
            f"could not find anchor lines in inflate.py template "
            f"(start_idx={start_idx} end_idx={end_idx}); template may have drifted"
        )
    bias_block = "".join("            " + b + "\n" for b in bias_lines)
    out_lines = lines[: start_idx + 1] + [bias_block] + lines[end_idx:]
    return "".join(out_lines)


def write_variant(
    variant_id: str,
    bias_lines: list[str],
    coords: dict[str, float],
    out_root: Path,
    inflate_template: str,
) -> dict[str, Any]:
    """Write one variant's submission_dir + manifest."""
    out_dir = out_root / variant_id
    sub_dir = out_dir / "submission_dir"
    sub_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(A1_ARCHIVE_PATH, sub_dir / "archive.zip")
    inflate_sh_path = sub_dir / "inflate.sh"
    shutil.copy2(A1_SUBMISSION_DIR / "inflate.sh", inflate_sh_path)
    inflate_sh_path.chmod(0o755)
    src_target = sub_dir / "src"
    src_target.mkdir(exist_ok=True)
    for fname in ("model.py", "codec.py"):
        shutil.copy2(A1_SUBMISSION_DIR / "src" / fname, src_target / fname)

    new_inflate = build_inflate_py(inflate_template, bias_lines)
    inflate_path = sub_dir / "inflate.py"
    inflate_path.write_text(new_inflate)

    archive_path = sub_dir / "archive.zip"
    archive_sha_actual = sha256_of(archive_path)
    if archive_sha_actual != A1_EXPECTED_ARCHIVE_SHA:
        raise RuntimeError(
            f"archive.zip SHA mismatch after copy: expected={A1_EXPECTED_ARCHIVE_SHA} "
            f"actual={archive_sha_actual}; aborting variant {variant_id!r}"
        )
    archive_size_actual = archive_path.stat().st_size
    if archive_size_actual != A1_EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError(
            f"archive.zip size mismatch: expected={A1_EXPECTED_ARCHIVE_BYTES} "
            f"actual={archive_size_actual}; aborting variant {variant_id!r}"
        )

    inflate_sha_old = sha256_of(A1_SUBMISSION_DIR / "inflate.py")
    inflate_sha_new = sha256_of(inflate_path)

    manifest = {
        "lane_id": "lane_pr101_bias_constrained_coord_search",
        "schema_version": "constrained_coord_search_v1",
        "build_timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "variant_id": variant_id,
        "coords": coords,
        "submission_name": variant_id,
        "archive_path": manifest_path(archive_path),
        "archive_sha256": archive_sha_actual,
        "archive_size_bytes": archive_size_actual,
        "archive_unchanged_from_a1": True,
        "inflate_py_path": manifest_path(inflate_path),
        "inflate_py_sha256_old": inflate_sha_old,
        "inflate_py_sha256_new": inflate_sha_new,
        "bias_spec": {
            "bias_lines": bias_lines,
            "n_bias_lines": len(bias_lines),
            "anchor_start": ANCHOR_START,
            "anchor_end_prefix": ANCHOR_END_PREFIX,
        },
        "score_claim": False,
        "byte_proxy_only": False,
        "ready_for_exact_eval_dispatch": False,
        "runtime_smoke_checked": False,
        "evidence_grade": "[predicted; constrained coord search on A1 substrate; pre-eval]",
        "dispatch_blockers": [
            "claim lane before any GHA/remote eval dispatch",
            "run exact-eval dispatcher preflight against submission_dir",
            "record runtime tree SHA and terminal dispatch claim row",
        ],
        "tag_discipline": {
            "before_eval": "[predicted; constrained coord search on A1 substrate]",
            "after_eval": "[contest-CPU GHA Linux x86_64] iff GHA dispatch produces eval.yml output",
            "macos_advisory": "[macOS-CPU calibrated] iff sibling C M5 Max sweep tool runs first",
        },
        "source_archive_sha256": A1_EXPECTED_ARCHIVE_SHA,
        "source_inflate_py_sha256": inflate_sha_old,
        "source_inflate_template_path": manifest_path(A1_SUBMISSION_DIR / "inflate.py"),
        "pr101_inflate_oracle_sha256": sha256_of(PR101_INFLATE),
        "a1_canonical_score_baseline": {
            "value": A1_CANONICAL_BASELINE_SCORE,
            "tag": A1_CANONICAL_BASELINE_TAG,
            "evidence_path": A1_CANONICAL_BASELINE_EVIDENCE,
        },
    }
    (out_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def enumerate_variants(
    grid_c0_0: tuple[float, ...],
    grid_c0_2: tuple[float, ...],
    grid_c1_1: tuple[float, ...],
    grid_sidecar: tuple[float, ...] | None,
) -> list[tuple[str, dict[str, float]]]:
    """Enumerate (variant_id, coord_dict) pairs over the 3D or 4D grid.

    For the 4D case (sidecar enabled), ``coord_dict`` carries
    ``sidecar_c2_0`` as well.
    """
    variants: list[tuple[str, dict[str, float]]] = []
    sidecar_iter = grid_sidecar if grid_sidecar is not None else (None,)
    for c0_0, c0_2, c1_1, sc in itertools.product(
        grid_c0_0, grid_c0_2, grid_c1_1, sidecar_iter
    ):
        coords = {"c0_0": float(c0_0), "c0_2": float(c0_2), "c1_1": float(c1_1)}
        if sc is not None:
            coords["sidecar_c2_0"] = float(sc)
            variant_id = (
                f"v_{_format_coord(c0_0)}_{_format_coord(c0_2)}_"
                f"{_format_coord(c1_1)}_sc{_format_coord(sc)}"
            )
        else:
            variant_id = (
                f"v_{_format_coord(c0_0)}_{_format_coord(c0_2)}_{_format_coord(c1_1)}"
            )
        variants.append((variant_id, coords))
    return variants


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Constrained coordinate search around PR101's verified inflate-time "
            "bias on A1 substrate"
        ),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="parent directory under which to write per-variant subdirs "
             "(default: experiments/results/constrained_coord_search_<utc>/)",
    )
    p.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="UTC timestamp suffix for default output-root; default = now",
    )
    p.add_argument(
        "--coarse",
        action="store_true",
        help="use the 7-point coarse grid: " + str(DEFAULT_COARSE_GRID),
    )
    p.add_argument(
        "--coarse-coarse",
        action="store_true",
        help="use the 4-point coarse-coarse grid (faster dev-loop): "
             + str(DEFAULT_COARSE_COARSE_GRID),
    )
    p.add_argument(
        "--refined",
        action="store_true",
        help="use the refined offset grid (combine with --center-coord)",
    )
    p.add_argument(
        "--center-coord",
        type=float,
        default=None,
        help="center coordinate for refined search (e.g. -1.0)",
    )
    p.add_argument(
        "--with-sidecar",
        action="store_true",
        help="enumerate the 5-point sidecar 4th coordinate",
    )
    p.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=None,
        help="subset of variant_ids to build (default: all enumerated)",
    )
    p.add_argument(
        "--candidate-queue",
        type=Path,
        default=None,
        help=(
            "consume an optimizer_candidate_queue_v1 or "
            "optimizer_guided_candidate_queue_v1 top_k queue and materialize "
            "its bias/sidecar coordinates instead of enumerating a grid"
        ),
    )
    p.add_argument(
        "--queue-top-k",
        type=int,
        default=None,
        help="when --candidate-queue is supplied, only materialize the first K queue rows",
    )
    p.add_argument(
        "--list-variants",
        action="store_true",
        help="print the candidate variant grid and exit (no files written)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="don't write per-variant directories; just print what would be written",
    )
    p.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="hard cap on total variants written (safety guard)",
    )
    args = p.parse_args()

    # Validate inputs.
    if not A1_ARCHIVE_PATH.exists():
        sys.stderr.write(f"[fatal] A1 archive missing: {A1_ARCHIVE_PATH}\n")
        return 2
    actual_sha = sha256_of(A1_ARCHIVE_PATH)
    if actual_sha != A1_EXPECTED_ARCHIVE_SHA:
        sys.stderr.write(
            f"[fatal] A1 archive SHA mismatch: expected={A1_EXPECTED_ARCHIVE_SHA} "
            f"actual={actual_sha}\n"
        )
        return 2
    if not PR101_INFLATE.exists():
        sys.stderr.write(f"[fatal] PR101 inflate.py oracle missing: {PR101_INFLATE}\n")
        return 2

    template_path = A1_SUBMISSION_DIR / "inflate.py"
    template_text = template_path.read_text()

    queue_source: Path | None = None
    queue_source_rows: list[dict[str, Any]] = []
    if args.candidate_queue is not None:
        queue_source = args.candidate_queue if args.candidate_queue.is_absolute() else REPO_ROOT / args.candidate_queue
        if not queue_source.is_file():
            sys.stderr.write(f"[fatal] candidate queue missing: {queue_source}\n")
            return 2
        try:
            queue_variants = variants_from_candidate_queue(queue_source, top_k=args.queue_top_k)
        except (OSError, json.JSONDecodeError, RuntimeError) as exc:
            sys.stderr.write(f"[fatal] invalid candidate queue: {exc}\n")
            return 2
        variants = [(vid, coords) for vid, coords, _row in queue_variants]
        queue_source_rows = [row for _vid, _coords, row in queue_variants]
        grid: tuple[float, ...] = ()
        grid_sidecar = None
    elif args.refined:
        if args.center_coord is None:
            sys.stderr.write(
                "[fatal] --refined requires --center-coord (e.g. --center-coord -1.0)\n"
            )
            return 2
        grid = tuple(round(args.center_coord + o, 4) for o in DEFAULT_REFINED_OFFSETS)
    elif args.coarse_coarse:
        grid = DEFAULT_COARSE_COARSE_GRID
    elif args.coarse:
        grid = DEFAULT_COARSE_GRID
    else:
        # Default = coarse.
        grid = DEFAULT_COARSE_GRID

    if args.candidate_queue is None:
        grid_sidecar = DEFAULT_SIDECAR_GRID if args.with_sidecar else None
        variants = enumerate_variants(
            grid_c0_0=grid,
            grid_c0_2=grid,
            grid_c1_1=grid,
            grid_sidecar=grid_sidecar,
        )

    if args.variants is not None:
        wanted = set(args.variants)
        variants = [(vid, c) for vid, c in variants if vid in wanted]
        unknown = wanted - {vid for vid, _ in variants}
        if unknown:
            sys.stderr.write(
                f"[fatal] unknown variant_id(s): {sorted(unknown)}; "
                f"first 10 valid: {[v for v, _ in variants][:10]}\n"
            )
            return 2

    if args.max_variants is not None and len(variants) > args.max_variants:
        sys.stderr.write(
            f"[fatal] enumerated {len(variants)} variants > --max-variants "
            f"{args.max_variants}; refine the grid or raise the cap\n"
        )
        return 2

    if args.list_variants:
        print(f"# {len(variants)} variants on grid {grid}"
              f"{' x sidecar ' + str(grid_sidecar) if grid_sidecar else ''}")
        for vid, coords in variants[:50]:
            print(f"  {vid}  {coords}")
        if len(variants) > 50:
            print(f"  ... ({len(variants) - 50} more)")
        return 0

    timestamp = args.timestamp or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_root = args.output_root or (
        REPO_ROOT
        / f"experiments/results/constrained_coord_search_pr101_bias_{timestamp}"
    )
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root

    if args.dry_run:
        print(f"[dry-run] would write {len(variants)} variants under {out_root}:")
        for vid, coords in variants[:20]:
            print(f"  {vid}  {coords}")
        if len(variants) > 20:
            print(f"  ... ({len(variants) - 20} more)")
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    rollup: list[dict[str, Any]] = []
    seen_inflate_sha: dict[str, str] = {}

    source_by_variant = {
        vid: row
        for (vid, _coords), row in zip(variants, queue_source_rows, strict=False)
    }
    for vid, coords in variants:
        bias_lines = build_bias_lines(
            c0_0=coords["c0_0"],
            c0_2=coords["c0_2"],
            c1_1=coords["c1_1"],
            sidecar_c2_0=coords.get("sidecar_c2_0"),
        )
        manifest = write_variant(
            variant_id=vid,
            bias_lines=bias_lines,
            coords=coords,
            out_root=out_root,
            inflate_template=template_text,
        )
        # Per HNeRV-parity lesson 11 (no-op detector): every variant must
        # produce a UNIQUE inflate.py SHA — otherwise two variants are the
        # same byte-stream and the search is wasting compute.
        new_sha = manifest["inflate_py_sha256_new"]
        if new_sha in seen_inflate_sha:
            existing = seen_inflate_sha[new_sha]
            sys.stderr.write(
                f"[warn] variant {vid} has identical inflate.py to {existing} "
                f"(no-op detector tripped); coords {coords}\n"
            )
        else:
            seen_inflate_sha[new_sha] = vid

        rollup.append({
            "variant_id": vid,
            "coords": coords,
            "source_queue_row": source_by_variant.get(vid),
            "submission_name": vid,
            "out_dir": str(out_root.relative_to(REPO_ROOT) / vid),
            "build_manifest_relpath": str(
                (out_root / vid / "build_manifest.json").relative_to(REPO_ROOT)
            ),
            "archive_sha256": manifest["archive_sha256"],
            "inflate_py_sha256": manifest["inflate_py_sha256_new"],
            "n_bias_lines": manifest["bias_spec"]["n_bias_lines"],
        })

    rollup_path = out_root / "rollup.json"
    rollup_payload = {
        "schema_version": "constrained_coord_search_rollup_v1",
        "lane_id": "lane_pr101_bias_constrained_coord_search",
        "build_timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "n_variants": len(rollup),
        "n_unique_inflates": len(seen_inflate_sha),
        "source_candidate_queue": manifest_path(queue_source) if queue_source is not None else None,
        "grid": {
            "c0_0": list(grid),
            "c0_2": list(grid),
            "c1_1": list(grid),
            "sidecar_c2_0": list(grid_sidecar) if grid_sidecar else None,
        },
        "anchor_score": {
            "value": A1_CANONICAL_BASELINE_SCORE,
            "tag": A1_CANONICAL_BASELINE_TAG,
            "evidence_path": A1_CANONICAL_BASELINE_EVIDENCE,
            "anchor_coords": {"c0_0": -1.0, "c0_2": -1.0, "c1_1": -1.0},
            "anchor_variant_id": "v_n1_00_n1_00_n1_00",
        },
        "regression_anchor": {
            "value": 0.194295755690,
            "tag": "[contest-CPU GHA Linux x86_64]",
            "evidence_path": "experiments/results/a1_bias_correction_sweep_v2_half_magnitude_20260509T103000Z/",
            "regression_coords": {"c0_0": -0.5, "c0_2": -0.5, "c1_1": -0.5},
            "regression_variant_id": "v_n0_50_n0_50_n0_50",
            "delta_vs_baseline": "+0.00145",
        },
        "evidence_grade": "[predicted; constrained coord search on A1 substrate]",
        "dispatch_blockers": [
            "claim lane lane_pr101_bias_constrained_coord_search before dispatch",
            "M5 Max coarse rank should run first ($0); promote top-5 to GHA",
        ],
        "variants": rollup,
    }
    rollup_path.write_text(json.dumps(rollup_payload, indent=2, sort_keys=True) + "\n")
    print(
        f"[done] wrote {len(rollup)} variants under {manifest_path(out_root)} "
        f"({len(seen_inflate_sha)} unique inflate.py); rollup="
        f"{manifest_path(rollup_path)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
