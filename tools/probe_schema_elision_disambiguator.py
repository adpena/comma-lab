#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Schema-elision probe-disambiguator (V1 / V2 / V2+V3).

For a target state-dict, encode under V1 (PR98 CD1 compact), V2 (PR100
schema-driven), and V2+V3 (V2 composed with PR105 size-sort reorder),
then brotli-compress each and report bytes-per-variant.

V1 and V2 are MUTUALLY EXCLUSIVE (both elide the same per-tensor
metadata region). V2+V3 stacks (V3 reorders the schema BEFORE V2
encodes, so brotli sees largest-entropy bodies first).

This is the probe-disambiguator design from
``.omx/research/schema_elision_design_pr98_pr100_pr105_20260512.md`` §5
— CPU-only, $0 GPU spend.

Per CLAUDE.md "Forbidden /tmp paths" — the output path is ALWAYS under
``.omx/research/`` (or a user-supplied repo-relative path), NEVER
``/tmp/``.

Per CLAUDE.md "Forbidden score claims" — the output JSON carries
``score_claim=false``, ``promotion_eligible=false``,
``ready_for_exact_eval_dispatch=false`` per Catalog #100.

Per CLAUDE.md "Apples-to-apples evidence discipline" — the probe MUST
run on the exact INT8 tensors that will ship in the archive (NOT
training-time tensors, NOT proxy archive bytes).

Usage
=====

::

    .venv/bin/python tools/probe_schema_elision_disambiguator.py \
        --state-dict-pt experiments/results/<archive>/state_dict.pt \
        --output .omx/research/probe_schema_elision_results_<utc>.json

If ``--state-dict-pt`` is omitted, the probe runs on a synthetic
fixture so the CLI shape can be smoke-tested without external state.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.packet_compiler import (  # noqa: E402
    encode_cd1_compact,
    encode_schema_driven,
    pack_state_schema_size_sorted,
)


def _load_state_dict_int8_tensors(
    pt_path: Path,
) -> dict[str, tuple[np.ndarray, float, tuple[int, ...]]]:
    """Load a state-dict and extract int8 tensors with shapes + scales.

    Returns a mapping ``name -> (int8_body, scale, shape)``. Scales
    default to 1.0 if not present in the file (probe focuses on bytes,
    not numerical accuracy).
    """
    import torch  # WEIGHTS_ONLY_FALSE_OK: weights_only=True is the safe path.

    sd_raw = torch.load(pt_path, weights_only=True, map_location="cpu")
    out: dict[str, tuple[np.ndarray, float, tuple[int, ...]]] = {}
    for name, val in sd_raw.items():
        if not isinstance(val, torch.Tensor):
            continue
        np_arr = val.detach().cpu().numpy()
        if np_arr.dtype != np.int8:
            continue
        shape = tuple(int(d) for d in np_arr.shape)
        # The probe only needs the bytes; scale defaults to 1.0.
        out[name] = (np_arr.reshape(-1), 1.0, shape)
    return out


def _synthetic_fixture() -> dict[str, tuple[np.ndarray, float, tuple[int, ...]]]:
    """Synthetic HNeRV-layout state-dict for CLI smoke testing."""
    rng = np.random.default_rng(0)
    schema = [
        ("blocks.0.weight", (36, 28, 3, 3)),
        ("blocks.0.bias", (36,)),
        ("blocks.1.weight", (36, 36, 3, 3)),
        ("blocks.1.bias", (36,)),
        ("blocks.2.weight", (45, 36, 3, 3)),
        ("blocks.2.bias", (45,)),
        ("skips.2.weight", (27, 36, 1, 1)),
        ("skips.2.bias", (27,)),
        ("rgb.weight", (3, 18, 3, 3)),
        ("rgb.bias", (3,)),
    ]
    out: dict[str, tuple[np.ndarray, float, tuple[int, ...]]] = {}
    for name, shape in schema:
        n_el = int(np.prod(shape))
        body = np.clip(
            np.round(rng.normal(0, 30, size=n_el)), -127, 127
        ).astype(np.int8)
        out[name] = (body, float(rng.uniform(0.01, 0.1)), shape)
    return out


def _try_brotli_bytes(data: bytes) -> int | None:
    try:
        import brotli  # type: ignore
    except ImportError:
        return None
    return len(brotli.compress(data, quality=11))


def probe(
    tensors_with_meta: dict[str, tuple[np.ndarray, float, tuple[int, ...]]],
) -> dict[str, object]:
    """Encode the state-dict under V1, V2, and V2+V3 and report bytes per variant.

    For each variant:
    - Encode + measure raw byte count (V2 has 2 streams; V1+V2+V3 also).
    - Brotli-compress the bodies (per PR98/PR100 design) + report bytes.
    """
    schema = [
        (name, meta[2]) for name, meta in tensors_with_meta.items()
    ]
    tensors = [
        (meta[0], meta[1]) for meta in tensors_with_meta.values()
    ]

    # V1 — PR98 CD1 (interleaved scale+body, then brotli the whole thing).
    v1_bytes_raw = encode_cd1_compact(tensors, scale_bits=16)
    v1_brotli = _try_brotli_bytes(v1_bytes_raw)

    # V2 — PR100 schema-driven (separate body + scales streams).
    v2_payload = encode_schema_driven(tensors)
    v2_body_brotli = _try_brotli_bytes(v2_payload.body_blob)
    # Scales are raw fp16 in PR100's design (no brotli) — count raw.
    v2_scales_raw = len(v2_payload.scales_blob)
    v2_total_brotli = (
        (v2_body_brotli + v2_scales_raw) if v2_body_brotli is not None else None
    )
    v2_raw_total = len(v2_payload.body_blob) + len(v2_payload.scales_blob)

    # V3 — PR105 size-sort applied to schema, then re-encode under V2.
    sorted_entries = pack_state_schema_size_sorted(schema)
    # Build a tensors-in-v3-order list keyed by the sorted schema.
    sorted_names = [e.name for e in sorted_entries]
    sorted_tensors = [
        (tensors_with_meta[n][0], tensors_with_meta[n][1])
        for n in sorted_names
    ]
    v3_payload = encode_schema_driven(sorted_tensors)
    v3_body_brotli = _try_brotli_bytes(v3_payload.body_blob)
    v3_scales_raw = len(v3_payload.scales_blob)
    v2_v3_total_brotli = (
        (v3_body_brotli + v3_scales_raw)
        if v3_body_brotli is not None
        else None
    )
    v2_v3_raw_total = (
        len(v3_payload.body_blob) + len(v3_payload.scales_blob)
    )

    # Compute verdict (post-brotli total bytes per variant).
    by_variant_bytes: dict[str, int | None] = {
        "v1": v1_brotli,
        "v2": v2_total_brotli,
        "v2_v3": v2_v3_total_brotli,
    }
    if all(v is not None for v in by_variant_bytes.values()):
        verdict_variant = min(
            by_variant_bytes.items(),
            key=lambda kv: kv[1] if kv[1] is not None else 1 << 62,
        )[0]
    else:
        verdict_variant = None

    return {
        "schema_version": "schema_elision_probe_v1",
        "evidence_grade": "[byte-anchor; non-authoritative]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "n_tensors_scanned": len(tensors_with_meta),
        "v1_raw_bytes": len(v1_bytes_raw),
        "v1_brotli_bytes": v1_brotli,
        "v2_raw_bytes": v2_raw_total,
        "v2_body_blob_raw_bytes": len(v2_payload.body_blob),
        "v2_body_blob_brotli_bytes": v2_body_brotli,
        "v2_scales_blob_bytes": v2_scales_raw,
        "v2_total_after_compression_bytes": v2_total_brotli,
        "v2_v3_raw_bytes": v2_v3_raw_total,
        "v2_v3_body_blob_raw_bytes": len(v3_payload.body_blob),
        "v2_v3_body_blob_brotli_bytes": v3_body_brotli,
        "v2_v3_scales_blob_bytes": v3_scales_raw,
        "v2_v3_total_after_compression_bytes": v2_v3_total_brotli,
        "delta_v1_minus_v2_bytes": (
            (v1_brotli - v2_total_brotli)
            if v1_brotli is not None and v2_total_brotli is not None
            else None
        ),
        "delta_v2_v3_minus_v2_bytes": (
            (v2_v3_total_brotli - v2_total_brotli)
            if v2_v3_total_brotli is not None and v2_total_brotli is not None
            else None
        ),
        "verdict_min_bytes_variant": verdict_variant,
        "cuda_eval_worth_testing": False,
        "notes": (
            "V1 and V2 are MUTUALLY EXCLUSIVE — they target the same "
            "metadata region. V2+V3 stacks (V3 size-sorts the schema "
            "before V2 encodes). Empirical Δ vs design-memo prediction "
            "(~840 B saved vs PR95 self-describing baseline) requires "
            "a same-fixture PR95-baseline measurement which is OUT OF "
            "SCOPE for this probe; this probe measures V1 vs V2 vs V2+V3 "
            "relative to each other."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Schema-elision V1/V2/V2+V3 probe-disambiguator. Sweeps all "
            "3 variants and reports bytes-per-variant. CPU-only, $0 GPU. "
            "Output JSON has score_claim=false per Catalog #100."
        )
    )
    parser.add_argument(
        "--state-dict-pt",
        type=Path,
        default=None,
        help=(
            "Path to a .pt state-dict whose int8 tensors will be probed. "
            "If omitted, runs on a synthetic HNeRV-layout fixture."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to write the probe results JSON. "
            "Default: .omx/research/probe_schema_elision_results_<utc>.json"
        ),
    )
    args = parser.parse_args(argv)

    if args.state_dict_pt is not None:
        if not args.state_dict_pt.is_file():
            print(
                f"error: state-dict file not found: {args.state_dict_pt}",
                file=sys.stderr,
            )
            return 2
        tensors_with_meta = _load_state_dict_int8_tensors(args.state_dict_pt)
        state_dict_sha = hashlib.sha256(
            args.state_dict_pt.read_bytes()
        ).hexdigest()
        source = str(args.state_dict_pt)
    else:
        tensors_with_meta = _synthetic_fixture()
        state_dict_sha = "synthetic-fixture-no-sha"
        source = "synthetic"

    if not tensors_with_meta:
        print(
            "error: no int8 tensors found in state-dict; "
            "schema-elision probe requires int8 weights",
            file=sys.stderr,
        )
        return 3

    result = probe(tensors_with_meta)
    result["state_dict_source"] = source
    result["state_dict_sha256"] = state_dict_sha
    result["generated_at_utc"] = _dt.datetime.now(_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    out_path = args.output
    if out_path is None:
        utc_stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = (
            REPO_ROOT
            / ".omx"
            / "research"
            / f"probe_schema_elision_results_{utc_stamp}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"schema-elision probe-disambiguator: wrote {out_path} "
        f"(n_tensors={result['n_tensors_scanned']}, "
        f"verdict={result['verdict_min_bytes_variant']!r}, "
        f"v1_brotli={result['v1_brotli_bytes']}, "
        f"v2_total={result['v2_total_after_compression_bytes']}, "
        f"v2_v3_total={result['v2_v3_total_after_compression_bytes']})",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
