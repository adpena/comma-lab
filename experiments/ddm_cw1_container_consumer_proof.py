# SPDX-License-Identifier: MIT
"""ddm_cw1 -- the F3 CONSUMER PROOF: drive ddm_up3's real container search through the
canonical optimizer and prove the winner is byte-identical to the shipped pointer archive.

WHY THIS EXISTS.  A canonical surface with no consumer is an orphan, and the campaign has
the receipt for that (#936: the cure for kill-signal was itself killed by non-wiring).  So
:mod:`tac.win_families.container_optimizer` is not landed on its own say-so.  This driver
runs it against the actual ``ddm_up3`` build path, on the actual shipped body, and checks
two byte-level facts:

1. **Identity control** -- the shipped codes under the shipped container must reproduce
   the ddm_to1 base archive sha exactly.  Without that, a smaller number from the same
   builder is a different object, not a saving.
2. **Winner identity** -- the canonical search over the same declared space must select a
   config whose bytes reproduce the shipped thirteenth-move pointer archive
   ``7ce46fd7...`` at 176,420 B.

The canonical optimizer measures ``len(archive.zip)`` directly, where ``ddm_up3``'s own
loop minimised the carrier STREAM length.  On this body the two agree, because every other
section is fixed -- but the stream is a proxy and the archive is the counted quantity, so
the canonical form measures the thing that is actually scored.

``ddm_up3``'s file is READ, never modified.  The per-config drive is done by rebinding
``up3.CONTAINER_OPTIONS`` in-process for the duration of one compile and restoring it,
which exercises up3's own code path unchanged.

AXIS ``[macOS-CPU exact byte measurement]``.  No score is produced or implied.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_up3_carrier_splice as up3

from tac.win_families import container_optimizer as co

#: The ddm_to1 base the splice starts from (ddm_up2's own recorded pointer body).
BASE_ARCHIVE_SHA256 = "50e561454b23026d3870f056747e848a49bd5f2b1e23930155d1281aeee91927"
#: The thirteenth-move pointer archive ddm_up3 shipped.
THIRTEENTH_MOVE_SHA256 = "7ce46fd7a845d5987903a0d85a56581961eb7716a55c38a7361e3b5ecae94b5f"
POINTER_ARCHIVE_BYTES = 176_420
DEFAULT_SOLVED = Path("/Volumes/APDataStore/pact/ddm_up2/retained/solved_codes.npy")


@contextlib.contextmanager
def _single_option(config: co.ContainerConfig):
    """Drive up3's own loop with exactly one declared option, then restore."""
    saved = up3.CONTAINER_OPTIONS
    up3.CONTAINER_OPTIONS = (
        (config.interleave, config.brotli_quality, config.brotli_lgwin),
    )
    try:
        yield
    finally:
        up3.CONTAINER_OPTIONS = saved


def _compiler(body, codes: np.ndarray, runtime_dir: Path):
    def compile_fn(config: co.ContainerConfig) -> bytes:
        with _single_option(config):
            built = up3.build_archive(
                body, codes, runtime_dir=runtime_dir, container_search=True, verify=False
            )
        return built["archive_bytes"]

    return compile_fn


def _parser(runtime_dir: Path):
    def parse_back_fn(blob: bytes) -> np.ndarray:
        recovered, _info = up3.parse_back_codes(blob, runtime_dir=runtime_dir)
        return np.asarray(recovered, dtype=np.int64)

    return parse_back_fn


def run(runtime_dir: Path, solved_path: Path) -> dict[str, Any]:
    body = up3.parse_shipped_body(runtime_dir)
    shipped_config = co.ContainerConfig(
        interleave=bool(body.ck2_carrier),
        brotli_quality=up3.BROTLI_QUALITY,
        brotli_lgwin=up3.BROTLI_LGWIN,
        label="shipped",
    )

    # (1) Identity control on the SHIPPED codes.
    identity = co.identity_control(
        compile_fn=_compiler(body, body.codes, runtime_dir),
        shipped_config=shipped_config,
        expected_sha256=BASE_ARCHIVE_SHA256,
    )

    # (2) Determinism control.
    determinism = co.determinism_control(
        compile_fn=_compiler(body, body.codes, runtime_dir), config=shipped_config
    )

    # (3) The real search on the SOLVED codes, over the same declared space up3 used.
    solved = np.load(solved_path).astype(np.int32)
    space = co.ContainerSpace(
        co.UP3_DECLARED_OPTIONS, incumbent_index=0, name="up3_declared_container_space"
    )
    result = co.search_container_space(
        space,
        compile_fn=_compiler(body, solved, runtime_dir),
        parse_back_fn=_parser(runtime_dir),
        expected_payload=solved.astype(np.int64),
        payload_equal=np.array_equal,
    )

    winner_matches = (
        result.winner.archive_sha256 == THIRTEENTH_MOVE_SHA256
        and result.winner.archive_bytes == POINTER_ARCHIVE_BYTES
    )
    return {
        "consumer": "ddm_cw1_container_consumer_proof",
        "axis": "[macOS-CPU exact byte measurement]",
        "score_claim": False,
        "promotable": False,
        "runtime_dir": str(runtime_dir),
        "identity_control": identity.to_json(),
        "determinism_control": determinism.to_json(),
        "search": result.to_json(),
        "winner_reproduces_thirteenth_move": winner_matches,
        "expected_winner_sha256": THIRTEENTH_MOVE_SHA256,
        "all_controls_pass": bool(
            identity.passed and determinism.passed and winner_matches
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=up3.DEFAULT_RUNTIME)
    parser.add_argument("--solved", type=Path, default=DEFAULT_SOLVED)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args.runtime_dir, args.solved)
    print(
        f"identity={report['identity_control']['passed']} "
        f"determinism={report['determinism_control']['passed']} "
        f"winner_sha_match={report['winner_reproduces_thirteenth_move']} "
        f"delta_bytes={report['search']['delta_archive_bytes']}",
        flush=True,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"wrote {args.out}", flush=True)
    return 0 if report["all_controls_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
