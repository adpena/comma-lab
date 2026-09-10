#!/usr/bin/env python3
"""MAIN's separate committed, one-shot first-measurement authorization producer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(1, str(REPO))  # tac.decode_wall_clock imports experiments.contest_auth_eval for the T4 digest
from tac.candidate_seal import (  # noqa: E402
    PrefireRefusal,
    _pf_write_new,
    build_first_measurement_authorization,
    write_prefire_refusal,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("intent", "lane-id", "instance-job-id", "output-dir", "cost-preflight", "out"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args(argv)
    try:
        document = build_first_measurement_authorization(intent_path=Path(args.intent), lane_id=args.lane_id,
            instance_job_id=args.instance_job_id, output_dir=Path(args.output_dir), cost_path=Path(args.cost_preflight))
        _pf_write_new(Path(args.out), document)
    except (PrefireRefusal, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        refusal = exc if isinstance(exc, PrefireRefusal) else PrefireRefusal("FIRST_MEASUREMENT_AUTHORIZATION_REFUSED", str(exc))
        write_prefire_refusal(refusal, paths=(Path(args.intent), Path(args.out)), output_dir=Path(args.output_dir))
        return 3
    print(f"AUTHORIZATION CREATED: {args.out}; MAIN must commit these exact bytes before dispatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
