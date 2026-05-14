#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Static AST-based inflate.py op-cost xray.

WHEN TO USE: when looking for cheap byte-level wins inside an inflate.py
runtime. The PR101 → PR103 medal delta was THREE lines of
``up[:, channel].sub_(1.0)`` per-channel bias corrections. Without an
op-by-op xray, those tiny structural diffs blend into the wider 71-LOC
inflate.py file.

WHAT IT REVEALS: a per-line catalog of every ``torch.*``, ``F.*``,
``tensor.method(...)`` call. For each op:
  - line number
  - op name (e.g. ``F.interpolate``, ``Tensor.sub_``, ``Tensor.permute``)
  - approximate cost class (``cheap`` ~constant; ``per-frame`` ~O(H*W);
    ``per-batch`` ~O(B*H*W); ``decoder-forward`` ~O(B * decoder params))
  - whether the op is a per-channel slice mutation (the PR101→PR103
    medal-delta pattern)

Operationally this answers: "what one-liners can I add to/remove from
inflate.py to gain bytes or save score?" Without this, a reviewer has to
mentally ledger 71 LOC × 10 candidate inflate.py files.

NOT a score claim. Tagged ``[diagnostic: inflate op-cost xray]``.

Output:
  experiments/results/xray_inflate_op_cost_profiler_<timestamp>/
    op_catalog.json
    op_catalog.md
    rebuild_command.txt

Usage:
  .venv/bin/python tools/xray_inflate_op_cost_profiler.py \
      --inflate-py experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/inflate.py \
      [--inflate-py ...]   # multi-file comparison
      [--label pr101]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "xray_inflate_op_cost_profiler_v1"
TOOL = "tools/xray_inflate_op_cost_profiler.py"

# Cost-class heuristic. Conservative — actual runtime depends on tensor
# shapes which we cannot resolve statically. These are PROXY classes.
COST_CLASS_PRIORS: dict[str, str] = {
    # Decoder forward pass — dominates inflate runtime
    "decoder.__call__": "decoder-forward",
    "Tensor.__call__": "decoder-forward",
    # Per-frame ops on (B*2, 3, H, W) tensors
    "F.interpolate": "per-frame",
    "F.grid_sample": "per-frame",
    "F.conv2d": "per-frame",
    "F.relu": "per-frame",
    "F.sigmoid": "per-frame",
    "Tensor.clamp": "per-frame",
    "Tensor.clamp_": "per-frame",
    "Tensor.permute": "per-frame",
    "Tensor.contiguous": "per-frame",
    "Tensor.round": "per-frame",
    "Tensor.round_": "per-frame",
    "Tensor.to": "per-frame",
    "Tensor.cpu": "per-frame",
    "Tensor.numpy": "per-frame",
    "Tensor.reshape": "cheap",  # view in most cases
    "Tensor.view": "cheap",
    # Per-channel slice mutations — cheap but score-relevant
    "Tensor.sub_": "per-channel-mutation",
    "Tensor.add_": "per-channel-mutation",
    "Tensor.mul_": "per-channel-mutation",
    "Tensor.div_": "per-channel-mutation",
    # I/O
    "open": "io",
    "Tensor.tobytes": "io",
    "ndarray.tobytes": "io",
    # Setup ops (one-shot)
    "torch.device": "cheap",
    "torch.cuda.is_available": "cheap",
    "torch.inference_mode": "cheap",
    "torch.no_grad": "cheap",
    "Tensor.eval": "cheap",
    "Tensor.load_state_dict": "cheap",
}

# Names of identifiers known to be torch tensors in inflate context. Used
# to attribute method calls without true type resolution.
TENSOR_BIND_NAMES: set[str] = {
    "decoded", "flat", "up", "frames", "latents", "decoder_sd",
    "archive_bytes", "x", "y", "out", "tensor", "img", "rgb",
}

PER_CHANNEL_MUTATING_OPS = {"sub_", "add_", "mul_", "div_", "copy_"}


def _qualified_name(node: ast.Call) -> str:
    """Best-effort dotted-name for the called expression."""
    func = node.func
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def _classify_op(qname: str) -> tuple[str, str]:
    """Return (op_kind, cost_class) for a qualified call name.

    op_kind is the canonicalized form keyed into COST_CLASS_PRIORS.
    cost_class is one of: decoder-forward, per-frame, per-channel-mutation,
    cheap, io, unknown.
    """
    # Direct hit
    if qname in COST_CLASS_PRIORS:
        return qname, COST_CLASS_PRIORS[qname]
    # F.* heuristic
    if qname.startswith("F."):
        return qname, COST_CLASS_PRIORS.get(qname, "per-frame")
    # torch.* heuristic
    if qname.startswith("torch."):
        return qname, COST_CLASS_PRIORS.get(qname, "cheap")
    # Tensor.method on a tensor binding
    parts = qname.split(".")
    if len(parts) >= 2 and parts[0] in TENSOR_BIND_NAMES:
        method = parts[-1]
        canonical = f"Tensor.{method}"
        if canonical in COST_CLASS_PRIORS:
            return canonical, COST_CLASS_PRIORS[canonical]
        if method in PER_CHANNEL_MUTATING_OPS:
            return canonical, "per-channel-mutation"
    return qname, "unknown"


def _is_subscript_lhs_call(node: ast.Call) -> bool:
    """True if call is on a Subscript (e.g. ``up[:,0,0].sub_(1.0)``)."""
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Subscript):
        return True
    return False


def profile_inflate_py(inflate_path: Path, *, label: str | None = None) -> dict:
    """Catalog every torch.* / F.* / tensor-method call in inflate.py."""
    inflate_path = Path(inflate_path)
    src = inflate_path.read_text()
    sha = hashlib.sha256(src.encode()).hexdigest()
    line_count = len(src.splitlines())

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {
            "label": label or inflate_path.stem,
            "inflate_path": str(inflate_path),
            "inflate_sha256": sha,
            "line_count": line_count,
            "parse_error": str(e),
            "ops": [],
            "by_cost_class": {},
            "per_channel_mutations": [],
        }

    ops: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        qname = _qualified_name(node)
        if not qname:
            continue
        op_kind, cost_class = _classify_op(qname)
        is_subscript = _is_subscript_lhs_call(node)
        if is_subscript and qname.split(".")[-1] in PER_CHANNEL_MUTATING_OPS:
            cost_class = "per-channel-mutation"
        ops.append({
            "line": node.lineno,
            "qualified_name": qname,
            "op_kind": op_kind,
            "cost_class": cost_class,
            "is_subscript_lhs": is_subscript,
        })

    ops.sort(key=lambda o: (o["line"], o["qualified_name"]))

    cost_counter = Counter(o["cost_class"] for o in ops)
    per_channel = [
        o for o in ops if o["cost_class"] == "per-channel-mutation"
    ]

    return {
        "label": label or inflate_path.stem,
        "inflate_path": str(inflate_path),
        "inflate_sha256": sha,
        "line_count": line_count,
        "op_count": len(ops),
        "ops": ops,
        "by_cost_class": dict(cost_counter),
        "per_channel_mutations": per_channel,
        "per_channel_mutation_count": len(per_channel),
    }


def render_markdown(report: dict, regen_header: str) -> str:
    lines = [regen_header, ""]
    lines.append("# Inflate op-cost xray")
    lines.append("")
    lines.append(
        f"_Schema_: `{report['schema_version']}` · _Generated_: "
        f"`{report['generated_at_utc']}`"
    )
    lines.append("")
    lines.append("## Per-file summary")
    lines.append("")
    lines.append(
        "| label | LOC | ops | decoder-fwd | per-frame | per-channel-mut | cheap | io | unknown |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for f in report["files"]:
        bcc = f["by_cost_class"]
        lines.append(
            f"| `{f['label']}` | {f['line_count']} | {f['op_count']} | "
            f"{bcc.get('decoder-forward', 0)} | "
            f"{bcc.get('per-frame', 0)} | "
            f"{bcc.get('per-channel-mutation', 0)} | "
            f"{bcc.get('cheap', 0)} | "
            f"{bcc.get('io', 0)} | "
            f"{bcc.get('unknown', 0)} |"
        )
    lines.append("")
    for f in report["files"]:
        lines.append(f"## `{f['label']}` — per-channel mutations (medal-delta candidates)")
        lines.append("")
        if not f["per_channel_mutations"]:
            lines.append("_(none)_")
            lines.append("")
            continue
        lines.append("| line | op | qualified_name |")
        lines.append("|---:|---|---|")
        for o in f["per_channel_mutations"]:
            lines.append(
                f"| {o['line']} | `{o['op_kind']}` | "
                f"`{o['qualified_name']}` |"
            )
        lines.append("")
        lines.append(f"## `{f['label']}` — full op catalog")
        lines.append("")
        lines.append("| line | op | cost_class |")
        lines.append("|---:|---|---|")
        for o in f["ops"]:
            lines.append(
                f"| {o['line']} | `{o['qualified_name']}` | "
                f"`{o['cost_class']}` |"
            )
        lines.append("")
    lines.append(
        "_Tag_: `[diagnostic: inflate op-cost xray]`. Cost classes are "
        "static heuristics (no shape inference). The `per-channel-mutation` "
        "rows are the PR101→PR103 medal-delta pattern: tiny one-liners "
        "with disproportionate score impact. Diagnostic only."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Static op-cost xray of inflate.py runtimes. Diagnostic only."
    )
    parser.add_argument("--inflate-py", action="append", required=True,
                        help="Path to inflate.py (repeat for multi-file comparison)")
    parser.add_argument("--label", action="append", default=None,
                        help="Per-file label (repeat in same order)")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.inflate_py]
    labels = args.label or [None] * len(paths)
    if len(labels) != len(paths):
        print(
            f"ERROR: --label count ({len(labels)}) != --inflate-py count "
            f"({len(paths)})",
            file=sys.stderr,
        )
        return 2

    for p in paths:
        if not p.exists():
            print(f"ERROR: inflate.py not found: {p}", file=sys.stderr)
            return 2

    files = [profile_inflate_py(p, label=lab) for p, lab in zip(paths, labels)]

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or (
        REPO_ROOT
        / "experiments"
        / "results"
        / f"xray_inflate_op_cost_profiler_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    state_hash = hashlib.sha256(
        "|".join(f["inflate_sha256"] for f in files).encode()
    ).hexdigest()[:16]

    report = {
        "schema_version": SCHEMA,
        "tool": TOOL,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "from_state_hash": state_hash,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "diagnostic_only",
        "files": files,
    }
    out_json = out_dir / "op_catalog.json"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True))

    regen = (
        f"<!-- generated_at: {report['generated_at_utc']}, "
        f"from_state_hash: {report['from_state_hash']} -->"
    )
    out_md = out_dir / "op_catalog.md"
    out_md.write_text(render_markdown(report, regen))

    parts = [".venv/bin/python tools/xray_inflate_op_cost_profiler.py"]
    parts.extend(f"--inflate-py {p}" for p in args.inflate_py)
    if args.label:
        parts.extend(f"--label {lab}" for lab in args.label)
    (out_dir / "rebuild_command.txt").write_text(" \\\n  ".join(parts) + "\n")

    print(f"[xray-op-cost] wrote {out_json}")
    print(f"[xray-op-cost] wrote {out_md}")
    total_per_chan = sum(f["per_channel_mutation_count"] for f in files)
    print(
        f"[xray-op-cost] {len(files)} files | "
        f"{sum(f['op_count'] for f in files)} ops | "
        f"{total_per_chan} per-channel mutations"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
