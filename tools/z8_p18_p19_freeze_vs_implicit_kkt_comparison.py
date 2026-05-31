#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the Z8 P18/P19 freeze-vs-implicit-KKT allocator comparison.

The implementation lives in TAC. This CLI is only the experiment actuator:
parse arguments, call the reusable comparison API, attach provenance, and write
the advisory result. It deliberately imports no other ``tools.*`` module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "upstream"
DEFAULT_VIDEO = str(UPSTREAM / "videos" / "0.mkv")
FRONTIER_CPU_ANCHOR = 0.192  # HISTORICAL_SCORE_LITERAL_OK:gap_banner_only_read_from_pointer_at_runtime

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.substrates.z8_hierarchical_predictive_coding.allocator_comparison import (  # noqa: E402
    run_allocator_comparison,
    write_allocator_comparison_result,
)


def _parse_csv_floats(text: str) -> list[float]:
    values = [float(part) for part in str(text).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def _frontier_cpu_anchor() -> float:
    ptr = REPO_ROOT / ".omx" / "state" / "canonical_frontier_pointer.json"
    try:
        if ptr.is_file():
            data = json.loads(ptr.read_text(encoding="utf-8"))
            cpu = data.get("our_local_frontier_contest_cpu") or {}
            value = cpu.get("score") if isinstance(cpu, dict) else None
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
    except Exception:  # pragma: no cover - stale operator pointer must not crash advisory CLI
        pass
    return FRONTIER_CPU_ANCHOR


def _write_provenance_sidecar(result_path: Path) -> dict:
    from tac.provenance import (
        build_provenance_for_macos_cpu_advisory,
        provenance_to_dict,
    )

    result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    try:
        source_path = str(result_path.relative_to(REPO_ROOT))
    except ValueError:
        source_path = str(result_path)
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=result_sha,
        source_path=source_path,
    )
    sidecar = {
        "schema": "z8_p18_p19_allocator_comparison_result_provenance.v1",
        "result_path": result_path.as_posix(),
        "result_sha256": result_sha,
        "provenance": provenance_to_dict(provenance),
    }
    sidecar_path = result_path.with_name("result_provenance.json")
    sidecar["provenance_path"] = sidecar_path.as_posix()
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--num-pairs", type=int, default=600)
    parser.add_argument("--eval-h", type=int, default=96)
    parser.add_argument("--eval-w", type=int, default=128)
    parser.add_argument(
        "--freeze-keep-fractions",
        type=_parse_csv_floats,
        default=_parse_csv_floats("0.5,0.25,0.1,0.05"),
        help="comma-separated keep fractions for the freeze allocator",
    )
    parser.add_argument(
        "--implicit-kkt-budget-fractions",
        type=_parse_csv_floats,
        default=_parse_csv_floats("0.002,0.004,0.008,0.012,0.02,0.03,0.045,0.06"),
        help="comma-separated budget fractions for the implicit-KKT allocator",
    )
    parser.add_argument("--pose-null-fraction", type=float, default=0.05)
    parser.add_argument("--seg-tau", type=float, default=1.0)
    parser.add_argument(
        "--noise-band",
        type=float,
        default=0.01,
        help="|S delta| <= noise-band is classified TIE_WITHIN_NOISE",
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            REPO_ROOT
            / "experiments"
            / "results"
            / f"z8_freeze_vs_implicit_kkt_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    result = run_allocator_comparison(
        video_path=str(args.video),
        num_pairs=int(args.num_pairs),
        eval_h=int(args.eval_h),
        eval_w=int(args.eval_w),
        freeze_keep_fractions=args.freeze_keep_fractions,
        implicit_kkt_budget_fractions=args.implicit_kkt_budget_fractions,
        pose_null_fraction=float(args.pose_null_fraction),
        seg_tau=float(args.seg_tau),
        noise_band=float(args.noise_band),
        out_dir=out_dir,
        repo_root=REPO_ROOT,
        upstream_dir=UPSTREAM,
        frontier_cpu_anchor=_frontier_cpu_anchor(),
        emit=lambda message: print(message, flush=True),
    )
    manifest = write_allocator_comparison_result(result, out_dir)
    result_path = Path(manifest["result_path"])
    _write_provenance_sidecar(result_path)
    verdict = result.get("verdict", {})
    print(
        "[z8-cmp] VERDICT: "
        f"winner={verdict.get('winner')} "
        f"headline_delta={verdict.get('headline_implicit_kkt_minus_freeze_S')}",
        flush=True,
    )
    print(f"[z8-cmp] -> {result_path}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
