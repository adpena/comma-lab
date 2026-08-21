"""Point the whole proven ddm_jg5 toolchain at a DIFFERENT candidate, without editing it.

`ddm_jg5_pose_resolve_on_edited_renders.py` is the machinery that produced the
fifteenth pointer move: cold Gauss-Newton on the frame-0 carrier lattice, the
derived-materiality refine, the matched-batch-shape baseline, the joint KEEP/DROP
waterfill, and the byte-close. All of it is candidate-agnostic in substance and
candidate-pinned in form: three module constants name jg5's own candidate, and
`load_candidate_instrument` takes them as **keyword defaults**.

Python binds default arguments at DEFINITION time, so reassigning
`jg5.CANDIDATE_RUNTIME` does nothing - the already-bound default wins. That is the
trap this shim exists to avoid: a successor who reassigns the constants gets a run
that silently solves against jg5's candidate and reports it as theirs.

So the shim rebinds the FUNCTION, not the constants, and changes nothing else:
every jg5 subcommand then runs unmodified against the named runtime, archive sha
and raw decode. Identity is still enforced inside jg5 - the archive sha is checked
against `--archive-sha256`, and `--expect-raw-sha256` still pins the decode
([[a_delta_without_its_baseline_is_unanchored_and_baselines_move_20260803]]).

Usage mirrors jg5 exactly after the three pins:

    ddm_fs2_jg5_on_candidate.py \
        --runtime <candidate_runtime_dir> \
        --archive-sha256 <sha> \
        --raw <candidate 0.raw> \
        -- gn --out <dir> --shard-index 0 --shard-count 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument(
        "jg5_argv",
        nargs=argparse.REMAINDER,
        help="jg5 subcommand and its flags, after a bare --",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rest = list(args.jg5_argv)
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        raise SystemExit("no jg5 subcommand given after --")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ddm_jg5_pose_resolve_on_edited_renders as jg5

    original = jg5.load_candidate_instrument
    runtime = args.runtime
    archive_sha = args.archive_sha256
    raw = args.raw

    def rebound(
        *,
        runtime: Path = runtime,
        expect_archive_sha256: str = archive_sha,
        raw_path: Path = raw,
        expect_raw_sha256: str | None = None,
    ):
        return original(
            runtime=runtime,
            expect_archive_sha256=expect_archive_sha256,
            raw_path=raw_path,
            expect_raw_sha256=expect_raw_sha256,
        )

    jg5.load_candidate_instrument = rebound
    print(
        f"[fs2] jg5 repinned: runtime={runtime} archive={archive_sha[:16]}... raw={raw}",
        flush=True,
    )
    return int(jg5.main(rest))


if __name__ == "__main__":
    raise SystemExit(main())
