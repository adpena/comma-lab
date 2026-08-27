#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""Fail-closed tombstone for the retired v10 measurement/cleanup tool."""

from __future__ import annotations

from typing import NoReturn

TOMBSTONE_STATUS = "RETIRED_UNSAFE_CLEANUP_CERTIFICATE_FAIL_CLOSED"
REFUSAL = (
    "tools/measure_v10_power_diagram_generator_byteclose.py is retired: the historical "
    "implementation's cleanup certificate could replay an existing output path and cleanup flag. "
    "Use read-only blocked-evidence tools; execution, resume, certification, and cleanup are refused."
)


class RetiredV10MeasurementToolError(RuntimeError):
    """Raised for every attempted use of the retired tool."""


def refuse(*_args: object, **_kwargs: object) -> NoReturn:
    raise RetiredV10MeasurementToolError(REFUSAL)


def run_measurement(*args: object, **kwargs: object) -> NoReturn:
    return refuse(*args, **kwargs)


def prepare_extraction_scratch(*args: object, **kwargs: object) -> NoReturn:
    return refuse(*args, **kwargs)


def certify_feature_cache(*args: object, **kwargs: object) -> NoReturn:
    return refuse(*args, **kwargs)


def cleanup_certified_scratch(*args: object, **kwargs: object) -> NoReturn:
    return refuse(*args, **kwargs)


def main(*args: object, **kwargs: object) -> NoReturn:
    return refuse(*args, **kwargs)


if __name__ == "__main__":
    main()
