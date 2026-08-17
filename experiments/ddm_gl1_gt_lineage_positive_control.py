#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""ddm_gl1 — the POSITIVE CONTROL for the GT decode-lineage guard.

A detector that has never been shown to fire is not a detector.  This script plants wrong-lineage
reads against REAL registered artifacts and proves the guard refuses them, and proves it passes
the correct ones.  Every control asserts BOTH directions; a guard that refuses everything is as
useless as one that refuses nothing.

Control 5 is the crown jewel: it reconstructs ``ddm_pi2``'s exact historical defect -- a DALI
seg cache read alongside a fresh PyAV runtime decode -- and shows :func:`assert_single_lineage`
refusing the pair.  That configuration shipped for months and no per-read check could see it,
because each half was individually sane and only the SPAN was wrong.

Exemplars are SELECTED FROM THE REGISTRY at runtime, never hardcoded, so the control keeps
working when retained run directories move.

Exit code 0 = every control behaved as specified.  Exit code 1 = the guard is broken.

Axis: ``[macOS-CPU advisory]``.  This script makes no score claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.gt_lineage import (
    AUTHORITY_LINEAGE,
    DALI_NVDEC,
    PYAV_YUV420_TO_RGB,
    GtLineageMismatch,
    GtLineageSplit,
    GtLineageUnknown,
    GtSource,
    assert_gt_lineage,
    assert_single_lineage,
    basename_lineage_collisions,
    load_registry,
    population_split_report,
)


def _pick(registry, lineage: str) -> tuple[str, str]:
    """Return (path, sha256) of a registered artifact of the given lineage that exists on disk."""
    for entry in sorted(registry.values(), key=lambda e: e.bytes):
        if entry.lineage != lineage:
            continue
        for p in entry.known_paths:
            if Path(p).is_file():
                return p, entry.sha256
    raise SystemExit(f"positive control cannot run: no on-disk registered artifact of {lineage}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    reg = load_registry()
    dali_path, dali_sha = _pick(reg, DALI_NVDEC)
    av_path, av_sha = _pick(reg, PYAV_YUV420_TO_RGB)

    results: list[dict[str, object]] = []

    def record(name: str, expected: str, fn) -> None:
        try:
            fn()
            observed, detail = "PASS", ""
        except (GtLineageMismatch, GtLineageUnknown, GtLineageSplit) as exc:
            observed, detail = type(exc).__name__, str(exc).splitlines()[0][:200]
        ok = observed == expected
        results.append(
            {
                "control": name,
                "expected": expected,
                "observed": observed,
                "ok": ok,
                "detail": detail,
            }
        )
        mark = "ok  " if ok else "FAIL"
        print(f"[{mark}] {name}\n        expected={expected} observed={observed}")
        if detail:
            print(f"        {detail}")

    print("=" * 100)
    print("ddm_gl1 GT decode-lineage guard — POSITIVE CONTROL")
    print(f"  DALI exemplar : {dali_path}\n                  sha256 {dali_sha}")
    print(f"  PyAV exemplar : {av_path}\n                  sha256 {av_sha}")
    print("=" * 100)

    # 1. The guard PASSES a correct read.  Without this, "refuses everything" would look like success.
    record(
        "1 correct read: DALI artifact, instrument requires DALI",
        "PASS",
        lambda: assert_gt_lineage(dali_path, required=DALI_NVDEC, instrument="gl1_control"),
    )

    # 2. Planted wrong-lineage read, one direction.
    record(
        "2 PLANTED wrong lineage: DALI artifact, instrument requires PyAV",
        "GtLineageMismatch",
        lambda: assert_gt_lineage(dali_path, required=PYAV_YUV420_TO_RGB, instrument="gl1_control"),
    )

    # 3. Planted wrong-lineage read, the direction that actually costs score: an instrument that
    #    means to track the contest-CUDA authority row silently reading an AV-lineage cache.
    record(
        "3 PLANTED wrong lineage: PyAV artifact, instrument requires AUTHORITY (DALI)",
        "GtLineageMismatch",
        lambda: assert_gt_lineage(av_path, required=AUTHORITY_LINEAGE, instrument="gl1_control"),
    )

    # 4. Fail-closed on UNRECORDED lineage.  An unknown must never read as a pass -- that state is
    #    indistinguishable from the one pi2 found, where the cache happened to be right by luck.
    unregistered = REPO_ROOT / "pyproject.toml"  # a real file that is not a registered GT artifact
    record(
        "4 fail-closed: unregistered file, lineage unknown",
        "GtLineageUnknown",
        lambda: assert_gt_lineage(unregistered, required=DALI_NVDEC, instrument="gl1_control"),
    )

    # 5. THE ddm_pi2 DEFECT, reconstructed: a DALI seg cache read alongside a fresh PyAV decode.
    record(
        "5 SPAN detector: DALI seg cache + fresh PyAV runtime decode (the pi2 instrument)",
        "GtLineageSplit",
        lambda: assert_single_lineage(
            [GtSource.file(dali_path), GtSource.runtime_decode("frame_utils.yuv420_to_rgb")],
            instrument="reconstructed_pi2_instrument",
        ),
    )

    # 6. The span detector must NOT fire on a coherent single-lineage instrument.
    record(
        "6 SPAN detector negative: DALI seg cache + DALI runtime decode",
        "PASS",
        lambda: assert_single_lineage(
            [GtSource.file(dali_path), GtSource.runtime_decode("DaliVideoDataset")],
            instrument="coherent_instrument",
        ),
    )

    # 7. Content-addressing is load-bearing: two files with the SAME basename resolve to DIFFERENT
    #    lineages, so any name-keyed rule (including pi2 §0.3, which says "keep using the cached
    #    gt_argmax_n600.npy") can be satisfied by the wrong bytes.
    collisions = basename_lineage_collisions(registry=reg)
    print("\n[info] filenames that resolve to MORE THAN ONE lineage (why the registry is keyed by sha256):")
    for n, sides in sorted(collisions.items()):
        print(f"        {n}")
        for lin, shas in sorted(sides.items()):
            print(f"            {lin:<22} {', '.join(s[:12] for s in shas)}")
    if not collisions:
        print("        (none found in this registry)")

    report = population_split_report()
    print("\n[info] population gauge (does not zero out when this module's cure is applied):")
    print(f"        registered artifacts               : {report['registered_artifacts']}")
    print(f"        distinct resolved lineages present : {report['distinct_resolved_lineage_count']} "
          f"{report['distinct_resolved_lineages_present']}")
    print(f"        population is single-lineage       : {report['population_is_single_lineage']}")
    print(f"        lineage tally                      : {json.dumps(report['lineage_tally'])}")

    failed = [r for r in results if not r["ok"]]
    print("\n" + "=" * 100)
    print(f"CONTROLS: {len(results) - len(failed)}/{len(results)} behaved as specified")
    print("=" * 100)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "schema": "ddm_gl1_positive_control_v1",
                    "axis": "[macOS-CPU advisory]",
                    "score_claim": False,
                    "dali_exemplar": {"path": dali_path, "sha256": dali_sha},
                    "pyav_exemplar": {"path": av_path, "sha256": av_sha},
                    "controls": results,
                    "basename_lineage_collisions": collisions,
                    "population_gauge": report,
                    "all_controls_passed": not failed,
                },
                indent=2,
            )
        )
        print(f"wrote {args.json_out}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
