#!/usr/bin/env python3
"""ddm_wc2c - re-check the split native receiver against the Python reference.

The native runtime discipline requires every native lowering to ship with a
Python reference oracle and an equivalence test that can be re-run against the
retained receipts rather than trusted from a prior PASS marker.  This is that
test for the ``ddm_wc2c`` split path.

FOUR CHECKS, each a refusal on its own:

1. **Receipt identity.**  The retained full-field split-native run reproduces the
   jg5 [contest-CUDA T4] receipt's ``corrected_quantized_logit_sha256``,
   ``corrected_cdf_input_sha256``, ``decoded_token_sha256`` and RC64
   ``decoder_bit_position``.
2. **Scalar-twin equality.**  The ``-DF26_FORCE_SCALAR=1`` build -- which uses no
   intrinsics at all -- produces the same four values as the dispatched build.
   This is what makes the ISA gating legal: if a SIMD lane ever disagreed with
   the portable twin, this check fails before any speed is quoted.
3. **Token-field byte equality.**  The retained token payloads of the two builds
   are compared byte for byte, not merely by their own reported digest.
4. **Thread-count independence.**  Runs at different ``F26_HPAC_THREADS`` values
   agree, which is the observable form of "no cross-thread reduction exists".

A missing receipt is reported as MISSING and fails the run.  It is never treated
as a pass: a check that silently skips is a check that reads green on an empty
denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

RETAINED = Path("/Volumes/APDataStore/pact/ddm_wc2/retained")

JG5_ANCHORS = {
    "corrected_quantized_logit_sha256": (
        "8269fe1aad031620b18051ad784d877bc9e6e9a4a71e775e78681955c4eec4dd"
    ),
    "corrected_cdf_input_sha256": (
        "370a5e2a85ccbb1e598c84333cc851f0a8c352091fde272160826b4b04e46000"
    ),
    "decoded_token_sha256": (
        "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
    ),
    "decoder_bit_position": 910_837,
}
IDENTITY_KEYS = tuple(JG5_ANCHORS)


class EquivalenceFailure(RuntimeError):
    """A retained receipt is missing, malformed, or disagrees."""


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EquivalenceFailure(f"MISSING retained receipt: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise EquivalenceFailure(f"MISSING retained payload: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_checks(retained: Path) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    dispatched = _load(retained / "wc2c_split_native_n600_t4.json")
    scalar = _load(retained / "wc2c_scalar_twin_n600_t4.json")
    python_row = _load(retained / "wc2c_python_baseline_n600.json")

    for row, label in ((dispatched, "dispatched"), (scalar, "scalar_twin")):
        if not row.get("full_field"):
            raise EquivalenceFailure(
                f"{label} receipt is a prefix run; identity requires the full field"
            )
        for key in IDENTITY_KEYS:
            results.append(
                {
                    "check": "receipt_identity",
                    "build": label,
                    "key": key,
                    "expected": JG5_ANCHORS[key],
                    "measured": row.get(key),
                    "pass": row.get(key) == JG5_ANCHORS[key],
                }
            )

    for key in IDENTITY_KEYS:
        results.append(
            {
                "check": "scalar_twin_equality",
                "key": key,
                "expected": dispatched.get(key),
                "measured": scalar.get(key),
                "pass": dispatched.get(key) == scalar.get(key),
            }
        )

    # The Python reference oracle: the shipping decoder, run locally on the same
    # archive.  It is the object the native path must not change.
    for key in (
        "prefix_corrected_quantized_logit_sha256",
        "prefix_corrected_cdf_input_sha256",
        "prefix_decoded_token_sha256",
    ):
        target = key.removeprefix("prefix_")
        results.append(
            {
                "check": "python_reference_equality",
                "key": target,
                "expected": python_row.get(key),
                "measured": dispatched.get(target),
                "pass": python_row.get(key) == dispatched.get(target),
            }
        )
    results.append(
        {
            "check": "python_reference_equality",
            "key": "decoder_bit_position",
            "expected": python_row.get("decoder_bit_position"),
            "measured": dispatched.get("decoder_bit_position"),
            "pass": python_row.get("decoder_bit_position")
            == dispatched.get("decoder_bit_position"),
        }
    )

    dispatched_tokens = _sha256_file(retained / "tokens_split_native_n600.u8")
    scalar_tokens = _sha256_file(retained / "tokens_scalar_twin_n600.u8")
    results.append(
        {
            "check": "token_field_byte_equality",
            "key": "tokens_u8_sha256",
            "expected": dispatched_tokens,
            "measured": scalar_tokens,
            "pass": dispatched_tokens == scalar_tokens,
        }
    )
    results.append(
        {
            "check": "token_field_byte_equality",
            "key": "tokens_vs_receipt_anchor",
            "expected": JG5_ANCHORS["decoded_token_sha256"],
            "measured": dispatched_tokens,
            "pass": dispatched_tokens == JG5_ANCHORS["decoded_token_sha256"],
        }
    )

    thread_rows = sorted(retained.glob("wc2c_thread_independence_t*.json"))
    if not thread_rows:
        results.append(
            {
                "check": "thread_count_independence",
                "key": "receipts_present",
                "expected": ">=2 thread rows",
                "measured": 0,
                "pass": False,
            }
        )
    else:
        reference = _load(thread_rows[0])
        for row_path in thread_rows[1:]:
            row = _load(row_path)
            for key in IDENTITY_KEYS:
                results.append(
                    {
                        "check": "thread_count_independence",
                        "build": row_path.name,
                        "key": key,
                        "expected": reference.get(key),
                        "measured": row.get(key),
                        "pass": reference.get(key) == row.get(key),
                    }
                )

    failed = [item for item in results if not item["pass"]]
    return {
        "schema": "ddm_wc2c_equivalence_test.v1",
        "retained": str(retained),
        "checks": results,
        "checks_run": len(results),
        "checks_failed": len(failed),
        "verdict": "PASS" if not failed else "REFUSE",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained", type=Path, default=RETAINED)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        report = run_checks(args.retained)
    except EquivalenceFailure as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for item in report["checks"]:
        if not item["pass"]:
            print(
                f"FAIL {item['check']}/{item.get('build', '-')}/{item['key']}: "
                f"expected {item['expected']!r} measured {item['measured']!r}"
            )
    print(f"{report['verdict']}: {report['checks_run'] - report['checks_failed']}"
          f"/{report['checks_run']} checks passed")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
