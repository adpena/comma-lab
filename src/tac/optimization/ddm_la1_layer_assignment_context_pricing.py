# SPDX-License-Identifier: MIT
"""Real-coder pricing for the DDM LA1 seven-home C1 partition.

LA1 keeps EV2's rate law exact: a rate belongs to one whole counted stream,
never to a scorer cell.  The seven inputs are the six exact outer C1 homes
plus J2's separately measured Lane-seed home.  Re-homing changes ownership
and lossless framing only; it does not infer a distortion delta.

The explicit arms use a uniform self-delimiting envelope around identity,
Brotli-Q11, and stdlib raw-LZMA1.  The context arms reuse the landed G4 and
Bellard/KT byte coders, whose frames have the same 46-byte accounting header.
All arms must decode exactly and reproduce deterministically.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import lzma
import os
import platform
import struct
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from tac.optimization.arith_selfcomp_rate_coders import (
    byte_context_frame_accounting,
    decode_bellard_class_mixing,
    decode_brotli_q11,
    decode_g4_decoder_context,
    encode_bellard_class_mixing,
    encode_brotli_q11,
    encode_g4_decoder_context,
)

SCHEMA: Final = "ddm_la1_layer_assignment_context_pricing.v1"
CONFIG_SCHEMA: Final = "ddm_la1_layer_assignment_context_pricing_config.v1"
RACE_SCHEMA: Final = "ddm_la1_residual_context_race.v1"
ASSIGNMENT_SCHEMA: Final = "ddm_la1_layer_assignment.v1"
LANE_ID: Final = "lane_ddm_la1_layer_assignment_context_pricing_20260725"
POINTER: Final = "0.1910828242 [contest-CPU]"
LP1_BYTES: Final = 134_211
POST_CC3_BYTES: Final = 130_789
CONTEXT_FALSIFIER_FRACTION: Final = 0.01
REPO: Final = Path(__file__).resolve().parents[3]

_EXPLICIT_MAGIC: Final = b"LA1E1"
_EXPLICIT_PREFIX: Final = struct.Struct(">5sBQ32s")
_EXPLICIT_CODECS: Final = {
    "RAW_EXPLICIT": 0,
    "BROTLI_Q11": 1,
    "RAW_LZMA1": 2,
}
_EXPLICIT_IDS: Final = {value: key for key, value in _EXPLICIT_CODECS.items()}
LA1_CODECS: Final = (
    "RAW_EXPLICIT",
    "BROTLI_Q11",
    "RAW_LZMA1",
    "G4_FREE_DECODER_CONTEXT",
    "BELLARD_KT_MIXER",
)
_LZMA_FILTERS: Final = [
    {
        "id": lzma.FILTER_LZMA1,
        "dict_size": 1 << 20,
        "lc": 3,
        "lp": 0,
        "pb": 2,
    }
]
_LAYERS: Final = (
    "L1_program",
    "L2_chart",
    "L3_RGB_realization",
    "L4_scorer_feature",
)


class LA1PricingError(ValueError):
    """A source, stream-home, coder, or accounting invariant failed."""


@dataclass(frozen=True, slots=True)
class StreamPolicy:
    member: str | None
    current_home_bytes: int
    raw_payload_bytes: int
    lp1_type: str
    deepest_layer: str
    survival_authority: str


_POLICIES: Final = {
    "manifest": StreamPolicy(
        "manifest.json",
        3_345,
        3_302,
        "PROGRAM",
        "L1_program",
        "exact receiver manifest and parse-back custody",
    ),
    "v15_predictor_zip_outer": StreamPolicy(
        "predictor.zip",
        100_099,
        100_056,
        "CONTEXT",
        "L2_chart",
        "predictor chart and grammar state consumed by the receiver",
    ),
    "g1_movable_worldsheet_outer": StreamPolicy(
        "predict/movable_polygon_worldsheet.g1s",
        29_878,
        29_810,
        "CONTEXT",
        "L2_chart",
        "G1 worldsheet chart state consumed across all 600 pairs",
    ),
    "receiver_realization_profile": StreamPolicy(
        "render/receiver_realization.ddrp",
        85,
        23,
        "PROGRAM",
        "L1_program",
        "video-derived realization profile consumed by generic receiver code",
    ),
    "solved_template_outer": StreamPolicy(
        "render/scorer_solved_templates.ddst",
        151,
        86,
        "FIBER",
        "L4_scorer_feature",
        "shared scorer-solved template survives through exact R to L5",
    ),
    "central_directory": StreamPolicy(
        None,
        383,
        383,
        "PROGRAM",
        "L1_program",
        "exact current ZIP framing; generic rebuild remains E5-pending",
    ),
    "lane_seed": StreamPolicy(
        "predict/lane_periodic_programs.ddlp",
        270,
        90,
        "CONTEXT",
        "L2_chart",
        "J2 measured 270-byte home contains 90 video-derived Lane bytes",
    ),
}

_LP1_NAMES: Final = {
    "manifest": "manifest",
    "v15_predictor_zip_outer": "v15_predictor_zip_outer_home",
    "g1_movable_worldsheet_outer": "g1_movable_worldsheet_outer_home",
    "receiver_realization_profile": "receiver_realization_profile",
    "solved_template_outer": "solved_template_outer_home",
    "central_directory": "central_directory_and_eocd",
    "lane_seed": "lane_program_seed",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic JSON bytes used by every LA1 receipt."""

    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bound_sources(
    config: Mapping[str, Any],
) -> tuple[dict[str, bytes], dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    source_rows = config.get("sources")
    if not isinstance(source_rows, Sequence) or isinstance(source_rows, (str, bytes)):
        raise LA1PricingError("config.sources must be a sequence")
    blobs: dict[str, bytes] = {}
    objects: dict[str, Mapping[str, Any]] = {}
    custody: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows):
        if not isinstance(row, Mapping) or set(row) != {
            "id",
            "path",
            "sha256",
            "bytes",
            "schema",
        }:
            raise LA1PricingError(f"config.sources[{index}] keys differ")
        source_id = row["id"]
        if not isinstance(source_id, str) or not source_id or source_id in blobs:
            raise LA1PricingError("source ids must be nonempty and unique")
        relative = Path(row["path"])
        if relative.is_absolute():
            raise LA1PricingError("source paths must be repository-relative")
        path = (REPO / relative).resolve()
        if not path.is_relative_to(REPO) or not path.is_file() or path.is_symlink():
            raise LA1PricingError(f"bound source is absent or unsafe: {relative}")
        payload = path.read_bytes()
        if len(payload) != row["bytes"] or sha256_bytes(payload) != row["sha256"]:
            raise LA1PricingError(f"bound source bytes/SHA-256 differ: {relative}")
        blobs[source_id] = payload
        schema = row["schema"]
        if schema is not None:
            value = json.loads(payload)
            if not isinstance(value, Mapping) or value.get("schema") != schema:
                raise LA1PricingError(f"bound source schema differs: {relative}")
            objects[source_id] = value
        custody.append(
            {
                "id": source_id,
                "path": relative.as_posix(),
                "bytes": len(payload),
                "sha256": row["sha256"],
                "schema": schema,
            }
        )
    required = {"c1_archive", "lp1", "ev2", "cc3", "g4", "ga1"}
    if set(blobs) != required:
        raise LA1PricingError("source ids differ from the sealed LA1 set")
    return blobs, objects, custody


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema": CONFIG_SCHEMA,
        "lane_id": LANE_ID,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "main_landing_review_required": True,
        "context_falsifier_fraction": CONTEXT_FALSIFIER_FRACTION,
        "lp1_accounting_bytes": LP1_BYTES,
        "post_cc3_coordinated_bytes": POST_CC3_BYTES,
        "seeded_archive_bytes": LP1_BYTES,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise LA1PricingError(f"config {key} must equal {expected!r}")
    seeded_sha = config.get("seeded_archive_sha256")
    if not isinstance(seeded_sha, str) or len(seeded_sha) != 64:
        raise LA1PricingError("config seeded_archive_sha256 is malformed")


def _validate_upstream(
    objects: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    lp1 = objects["lp1"]
    ev2 = objects["ev2"]
    cc3 = objects["cc3"]
    g4 = objects["g4"]
    ga1 = objects["ga1"]
    for label, value in objects.items():
        if value.get("score_claim") is not False or value.get("pointer_moved") is not False:
            raise LA1PricingError(f"{label} false-authority fields differ")
    if lp1["c1_corrected_waterfill"]["corrected_measured_allocated_bytes"] != LP1_BYTES:
        raise LA1PricingError("LP1 allocation differs from 134211 bytes")
    if ev2["lp1_coarse_lawful_partition"]["counted_bytes"] != LP1_BYTES:
        raise LA1PricingError("EV2 coarse partition differs from 134211 bytes")
    if cc3["corrected_lp1"]["post_integration_corrected_total_bytes"] != POST_CC3_BYTES:
        raise LA1PricingError("CC3 coordinated total differs from 130789 bytes")
    free_context = g4["summary"]["free_context"]
    if free_context["aggregate_spatial_pixel_prior"]["context_payload_bytes"] != 0 or free_context[
        "real_coder_scope"
    ] != (
        "exact innovation bits only; generic traversal/context reset is free; "
        "archive section/container overhead excluded"
    ):
        raise LA1PricingError("G4 free-context authority differs")
    upper = ga1["current_c1_typed_mass_upper_bound"]
    if upper["disposition"] != "DOMINATED_INSTANCE_CURRENT_LP1_COMPOSITION":
        raise LA1PricingError("GA1 gauge disposition differs")
    lp1_rows = {
        row["stream"]: row for row in lp1["c1_corrected_waterfill"]["rows"] if row["counted_in_corrected_total"]
    }
    ev2_rows = {row["stream"]: row for row in ev2["lp1_coarse_lawful_partition"]["rows"]}
    for stream, policy in _POLICIES.items():
        lp1_name = _LP1_NAMES[stream]
        if (
            lp1_rows[lp1_name]["corrected_allocated_bytes"] != policy.current_home_bytes
            or ev2_rows[lp1_name]["counted_bytes"] != policy.current_home_bytes
        ):
            raise LA1PricingError(f"LP1/EV2 home differs for {stream}")
    return {"lp1_rows": lp1_rows, "ev2_rows": ev2_rows}


def _explicit_body(raw: bytes, codec: str) -> bytes:
    if codec == "RAW_EXPLICIT":
        return raw
    if codec == "BROTLI_Q11":
        return encode_brotli_q11(raw)
    if codec == "RAW_LZMA1":
        return lzma.compress(raw, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS)
    raise LA1PricingError(f"unknown explicit codec {codec}")


def _encode_explicit(raw: bytes, codec: str) -> bytes:
    body = _explicit_body(raw, codec)
    return (
        _EXPLICIT_PREFIX.pack(
            _EXPLICIT_MAGIC,
            _EXPLICIT_CODECS[codec],
            len(raw),
            hashlib.sha256(raw).digest(),
        )
        + body
    )


def _decode_explicit(frame: bytes) -> bytes:
    if len(frame) < _EXPLICIT_PREFIX.size:
        raise LA1PricingError("explicit frame is truncated")
    magic, codec_id, raw_length, digest = _EXPLICIT_PREFIX.unpack_from(frame)
    codec = _EXPLICIT_IDS.get(codec_id)
    if magic != _EXPLICIT_MAGIC or codec is None:
        raise LA1PricingError("explicit frame header differs")
    body = frame[_EXPLICIT_PREFIX.size :]
    try:
        if codec == "RAW_EXPLICIT":
            raw = body
        elif codec == "BROTLI_Q11":
            raw = decode_brotli_q11(body)
        else:
            decoder = lzma.LZMADecompressor(
                format=lzma.FORMAT_RAW,
                filters=_LZMA_FILTERS,
            )
            raw = decoder.decompress(body)
            if not decoder.eof or decoder.unused_data:
                raise LA1PricingError("raw-LZMA1 frame is truncated or has trailing bytes")
    except (lzma.LZMAError, ValueError) as exc:
        raise LA1PricingError(f"{codec} decode failed") from exc
    if len(raw) != raw_length or hashlib.sha256(raw).digest() != digest or _encode_explicit(raw, codec) != frame:
        raise LA1PricingError("explicit frame parse-back or canonicality differs")
    return raw


def encode_la1_frame(raw: bytes, codec: str) -> bytes:
    """Encode one payload with a named LA1 race arm."""

    if not raw:
        raise LA1PricingError("LA1 payload is empty")
    if codec in _EXPLICIT_CODECS:
        return _encode_explicit(raw, codec)
    if codec == "G4_FREE_DECODER_CONTEXT":
        return encode_g4_decoder_context(raw)
    if codec == "BELLARD_KT_MIXER":
        return encode_bellard_class_mixing(raw)
    raise LA1PricingError(f"unknown LA1 codec {codec}")


def decode_la1_frame(frame: bytes, codec: str) -> bytes:
    """Decode and canonically re-encode one named LA1 frame."""

    if codec in _EXPLICIT_CODECS:
        raw = _decode_explicit(frame)
    elif codec == "G4_FREE_DECODER_CONTEXT":
        raw = decode_g4_decoder_context(frame)
    elif codec == "BELLARD_KT_MIXER":
        raw = decode_bellard_class_mixing(frame)
    else:
        raise LA1PricingError(f"unknown LA1 codec {codec}")
    if encode_la1_frame(raw, codec) != frame:
        raise LA1PricingError(f"{codec} frame is not canonical")
    return raw


def race_payload(
    stream: str,
    raw: bytes,
    *,
    current_home_bytes: int,
) -> dict[str, Any]:
    """Run the exact LA1 coder race on an arbitrary whole-stream payload."""

    if not stream or not raw or current_home_bytes < 0:
        raise LA1PricingError("race stream identity/payload is invalid")
    arms: list[dict[str, Any]] = []
    for codec in _EXPLICIT_CODECS:
        frame = encode_la1_frame(raw, codec)
        if decode_la1_frame(frame, codec) != raw:
            raise LA1PricingError(f"{stream} {codec} parse-back differs")
        arms.append(
            {
                "codec": codec,
                "ownership": "RESIDUAL",
                "framed_bytes": len(frame),
                "header_bytes": _EXPLICIT_PREFIX.size,
                "model_parameter_bytes": 0,
                "coded_payload_bytes": len(frame) - _EXPLICIT_PREFIX.size,
                "frame_sha256": sha256_bytes(frame),
                "parseback_exact": True,
                "generic_decoder_code_bytes": 0,
            }
        )
    context_arms = (
        "G4_FREE_DECODER_CONTEXT",
        "BELLARD_KT_MIXER",
    )
    for codec in context_arms:
        frame = encode_la1_frame(raw, codec)
        if decode_la1_frame(frame, codec) != raw:
            raise LA1PricingError(f"{stream} {codec} parse-back differs")
        arms.append(
            {
                "codec": codec,
                "ownership": "CONTEXT",
                **byte_context_frame_accounting(frame),
                "frame_sha256": sha256_bytes(frame),
                "parseback_exact": True,
                "generic_decoder_code_bytes": 0,
            }
        )
    explicit = min(
        (row for row in arms if row["ownership"] == "RESIDUAL"),
        key=lambda row: (row["framed_bytes"], row["codec"]),
    )
    context = min(
        (row for row in arms if row["ownership"] == "CONTEXT"),
        key=lambda row: (row["framed_bytes"], row["codec"]),
    )
    selected = min((explicit, context), key=lambda row: (row["framed_bytes"], row["codec"]))
    winner = selected["ownership"]
    return {
        "schema": RACE_SCHEMA,
        "stream": stream,
        "current_home_bytes": current_home_bytes,
        "raw_semantic_payload_bytes": len(raw),
        "raw_semantic_payload_sha256": sha256_bytes(raw),
        "explicit_stream_bytes": explicit["framed_bytes"],
        "explicit_winning_codec": explicit["codec"],
        "context_model_bytes": context["framed_bytes"],
        "context_winning_codec": context["codec"],
        "winner": winner,
        "delta_B_context_minus_explicit": (context["framed_bytes"] - explicit["framed_bytes"]),
        "selected_framed_bytes": selected["framed_bytes"],
        "selected_codec": selected["codec"],
        "selected_delta_vs_current_home_bytes": (selected["framed_bytes"] - current_home_bytes),
        "parseback_exact_all_arms": all(row["parseback_exact"] for row in arms),
        "arms": arms,
        "verdict_scope": (
            "INSTANCE: this exact SHA-bound whole stream only; no cell allocation, "
            "coder-family closure, or distortion inference"
        ),
    }


def race_stream(
    stream: str,
    raw: bytes,
    *,
    current_home_bytes: int,
) -> dict[str, Any]:
    """Run the exact explicit-versus-context race on one governed LA1 home."""

    if stream not in _POLICIES:
        raise LA1PricingError("race stream identity/payload is invalid")
    return race_payload(stream, raw, current_home_bytes=current_home_bytes)


def _extract_streams(base_archive: bytes, lane_payload: bytes) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(base_archive), "r") as archive:
            infos = archive.infolist()
            names = [row.filename for row in infos]
            expected_names = [
                policy.member
                for policy in _POLICIES.values()
                if policy.member is not None and policy.member != "predict/lane_periodic_programs.ddlp"
            ]
            if names != expected_names:
                raise LA1PricingError("C1 outer member order differs")
            boundaries = [row.header_offset for row in infos[1:]] + [archive.start_dir]
            result: dict[str, bytes] = {}
            by_member = {
                policy.member: (stream, policy) for stream, policy in _POLICIES.items() if policy.member is not None
            }
            for info, stop in zip(infos, boundaries, strict=True):
                stream, policy = by_member[info.filename]
                if stop - info.header_offset != policy.current_home_bytes:
                    raise LA1PricingError(f"C1 exact home bytes differ for {stream}")
                result[stream] = archive.read(info)
            result["central_directory"] = base_archive[archive.start_dir :]
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise LA1PricingError("C1 archive is malformed") from exc
    result["lane_seed"] = lane_payload
    for stream, policy in _POLICIES.items():
        if len(result[stream]) != policy.raw_payload_bytes:
            raise LA1PricingError(f"semantic payload bytes differ for {stream}")
    return result


def _layer_candidates(policy: StreamPolicy, race: Mapping[str, Any]) -> list[dict[str, Any]]:
    deepest_index = _LAYERS.index(policy.deepest_layer)
    return [
        {
            "layer_home": layer,
            "candidate_status": "MEASURED_SAME_PAYLOAD_REAL_CODER_PRICE",
            "framed_bytes": race["selected_framed_bytes"],
            "winning_codec": race["selected_codec"],
            "same_payload_price_reused": index != deepest_index,
            "distortion_delta": 0,
            "distortion_authority": (
                "DERIVED_ONLY_FROM_EXACT_LOSSLESS_STREAM_PARSEBACK; RECEIVER_INTEGRATION_REMAINS_E5_PENDING"
            ),
        }
        for index, layer in enumerate(_LAYERS[: deepest_index + 1])
    ] + [
        {
            "layer_home": layer,
            "candidate_status": (
                "NULL_NO_SAME_OBJECT_DEEPER_REPRESENTATION"
                if layer != "L5_verdict"
                else "REFUSED_NO_SCORER_WEIGHTS_OR_GROUND_TRUTH_TABLE_IN_PAYLOAD"
            ),
            "framed_bytes": None,
            "winning_codec": None,
            "same_payload_price_reused": False,
            "distortion_delta": None,
            "distortion_authority": "UNMEASURED",
        }
        for layer in (*_LAYERS[deepest_index + 1 :], "L5_verdict")
    ]


def build_receipt(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    base_archive: bytes,
    seeded_archive: bytes,
) -> dict[str, Any]:
    """Build one complete LA1 receipt from exact source and seeded archives."""

    _validate_config(config)
    blobs, objects, custody = _load_bound_sources(config)
    upstream = _validate_upstream(objects)
    if len(base_archive) != 133_941 or base_archive != blobs["c1_archive"]:
        raise LA1PricingError("C1 archive argument differs from bound custody")
    if (
        len(seeded_archive) != config["seeded_archive_bytes"]
        or sha256_bytes(seeded_archive) != config["seeded_archive_sha256"]
    ):
        raise LA1PricingError("J2 seeded archive identity differs")
    try:
        with zipfile.ZipFile(io.BytesIO(seeded_archive), "r") as archive:
            lane_payload = archive.read("predict/lane_periodic_programs.ddlp")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise LA1PricingError("J2 seeded archive lacks the Lane payload") from exc
    streams = _extract_streams(base_archive, lane_payload)
    races = [
        race_stream(
            stream,
            streams[stream],
            current_home_bytes=policy.current_home_bytes,
        )
        for stream, policy in _POLICIES.items()
    ]
    race_by_stream = {row["stream"]: row for row in races}
    assignments: list[dict[str, Any]] = []
    lp1_rows = upstream["lp1_rows"]
    for stream, policy in _POLICIES.items():
        race = race_by_stream[stream]
        lp1_row = lp1_rows[_LP1_NAMES[stream]]
        assignments.append(
            {
                "schema": ASSIGNMENT_SCHEMA,
                "stream": stream,
                "source_accounting_home_bytes": policy.current_home_bytes,
                "semantic_payload_bytes": len(streams[stream]),
                "prior_lp1_type": policy.lp1_type,
                "deepest_admissible_layer": policy.deepest_layer,
                "survival_authority": policy.survival_authority,
                "lp1_source_typed_home": lp1_row["typed_home"],
                "residual_context_winner": race["winner"],
                "winning_codec": race["selected_codec"],
                "rehomed_counted_bytes": race["selected_framed_bytes"],
                "measured_delta_B": race["selected_delta_vs_current_home_bytes"],
                "candidate_home_prices": _layer_candidates(policy, race),
                "payload_cleanliness": {
                    "generic_interpreter_bytes_counted": 0,
                    "video_derived_payload_fully_counted": True,
                    "hash_or_table_hidden_in_code": False,
                    "receiver_consumption_bijection": ("EXACT_STREAM_PARSEBACK_PROVEN; E5_EXPORT_CONSUMER_PENDING"),
                    "derivable_manifest_or_container_bytes_removed": 0,
                    "disposition": (
                        "COUNT_CONSERVATIVELY_UNTIL_E5_GENERIC_REBUILD_PROVES DERIVATION_AND_EXACT_RECEIVER_PARSEBACK"
                    ),
                },
                "verdict_scope": (
                    "STREAM: exact EV2 whole-stream home; layer ownership does not "
                    "allocate bytes to cells or assert a contest score"
                ),
            }
        )
    if sum(row["source_accounting_home_bytes"] for row in assignments) != LP1_BYTES:
        raise LA1PricingError("seven-home source accounting does not conserve LP1 bytes")
    la1_bytes = sum(row["rehomed_counted_bytes"] for row in assignments)
    context_mass = sum(
        row["source_accounting_home_bytes"] for row in assignments if row["residual_context_winner"] == "CONTEXT"
    )
    context_fraction = context_mass / LP1_BYTES
    context_falsifier = context_fraction < CONTEXT_FALSIFIER_FRACTION
    coordinated = min(POST_CC3_BYTES, la1_bytes)
    ga1 = objects["ga1"]
    return {
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "lane_id": LANE_ID,
        "tasks": [669],
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "pointer_moved": False,
        "main_landing_review_required": True,
        "evidence_axis": "[macOS-CPU local lossless-byte advisory]",
        "verdict": (
            "CONTEXT_PRICED_OUT_INSTANCE_TYPED_MASS_LT_1_PERCENT;"
            "SEVEN_HOME_LA1_ALTERNATIVE_BEATS_POST_CC3_COORDINATED_ACCOUNTING;"
            "E5_RECEIVER_INTEGRATION_PENDING"
            if context_falsifier and la1_bytes < POST_CC3_BYTES
            else "LA1_MEASURED_WITH_SCOPED_DISPOSITION"
        ),
        "verdict_scope": (
            "INSTANCE: exact SHA-bound C1/EV2 seven-home accounting object and "
            "current real-coder implementations. The LA1 and CC3 totals are "
            "overlapping alternatives, not additive. No receiver-closed E5 archive, "
            "scorer replay, contest score, promotion, or pointer movement."
        ),
        "config_path": config_path.relative_to(REPO).as_posix(),
        "config_sha256": sha256_bytes(canonical_json_bytes(config)),
        "source_custody": custody,
        "coder_custody": {
            "uniform_frame_header_bytes": _EXPLICIT_PREFIX.size,
            "uniform_frame_reason": (
                "all explicit and context candidates pay raw length, raw SHA-256, "
                "and codec identity before ownership comparison"
            ),
            "python": platform.python_version(),
            "brotli_distribution": importlib.metadata.version("Brotli"),
            "raw_lzma1": {
                "format": "FORMAT_RAW",
                "filter": "FILTER_LZMA1",
                "dict_size": 1 << 20,
                "lc": 3,
                "lp": 0,
                "pb": 2,
            },
            "mixed_coder_source": ("src/tac/optimization/arith_selfcomp_rate_coders.py"),
            "mixed_coder_source_sha256": sha256_file(REPO / "src/tac/optimization/arith_selfcomp_rate_coders.py"),
            "la1_compiler_source_sha256": sha256_file(Path(__file__)),
        },
        "seeded_archive": {
            "bytes": len(seeded_archive),
            "sha256": sha256_bytes(seeded_archive),
            "lane_semantic_payload_bytes": len(lane_payload),
            "lane_semantic_payload_sha256": sha256_bytes(lane_payload),
            "source_accounting_delta_bytes": len(seeded_archive) - len(base_archive),
        },
        "residual_vs_context": {
            "schema": "ddm_la1_residual_context_table.v1",
            "rows": races,
            "context_winning_source_mass_bytes": context_mass,
            "total_typed_source_mass_bytes": LP1_BYTES,
            "context_winning_typed_mass_fraction": context_fraction,
            "falsifier_threshold_fraction": CONTEXT_FALSIFIER_FRACTION,
            "falsifier_fired": context_falsifier,
            "disposition": (
                "CONTEXT_PRICED_OUT_INSTANCE_CURRENT_SEVEN_HOME_GEOMETRY"
                if context_falsifier
                else "CONTEXT_RETAINS_INSTANCE_MASS"
            ),
            "reopener": (
                "a scorer-recursive same-object stream geometry where decoder-derived "
                "context wins at least 1% of typed source mass after uniform framing"
            ),
        },
        "layer_assignment": {
            "schema": "ddm_la1_layer_assignment_table.v1",
            "rows": assignments,
            "source_home_bytes": LP1_BYTES,
            "rehomed_real_coder_bytes": la1_bytes,
            "measured_delta_vs_lp1_bytes": la1_bytes - LP1_BYTES,
            "rate_is_stream_level": True,
            "distortion_is_cell_level": True,
            "per_cell_rate_allocation_performed": False,
        },
        "coordination": {
            "lp1_source_bytes": LP1_BYTES,
            "la1_rehomed_alternative_bytes": la1_bytes,
            "la1_delta_vs_lp1_bytes": la1_bytes - LP1_BYTES,
            "post_cc3_coordinated_incumbent_bytes": POST_CC3_BYTES,
            "cc3_delta_vs_lp1_bytes": POST_CC3_BYTES - LP1_BYTES,
            "composition_law": "MIN_OF_OVERLAPPING_ALTERNATIVES_NEVER_SUM_DELTAS",
            "composed_best_case_bytes": coordinated,
            "net_composed_delta_vs_130789_bytes": coordinated - POST_CC3_BYTES,
            "current_receiver_closed_accounting_bytes": POST_CC3_BYTES,
            "prospective_e5_bytes": coordinated,
            "prospective_only": True,
            "e5_disposition": (
                "QUEUE_SELECTED_FRAMES_FOR_RECEIVER_CLOSED_EXPORT_AND_EXACT PAYLOAD_CONSUMPTION_BIJECTION"
            ),
        },
        "ga1_non_reopening": {
            "disposition": ga1["current_c1_typed_mass_upper_bound"]["disposition"],
            "gauge_rerun_performed": False,
            "named_reopener": [row["required_closure"] for row in ga1["curve_admission"]["blockers"]],
        },
        "stores_consulted": [
            "CLAUDE.md",
            "AGENTS.md",
            "docs/operating_manual_craft_handoff.md",
            "LP1 134211-byte typed allocation",
            "EV2 exact seven-home stream partition and stream/cell join refusal",
            "CC3 130789-byte coordinated accounting",
            "G4 decoder-derived context measurement and implementation",
            "CC2/CC3 Bellard-KT mixed-coder implementation",
            "GA1 dominated gauge bound and named reopener",
            "operator inbox through 2026-07-24T23:09:25Z",
        ],
        "triality": {
            "dsl": ".omx/research/configs/ddm_la1_layer_assignment_context_pricing_20260725.json",
            "dag": ".omx/research/ddm_la1_layer_assignment_context_pricing_DAG_FEED_20260725.md",
            "equations": ".omx/research/ddm_la1_layer_assignment_context_pricing_canonical_equations_20260725.md",
        },
    }


def materialize(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Reconstruct the sealed J2 seed home and atomically write the LA1 receipt."""

    config_path = config_path.resolve()
    if not config_path.is_relative_to(REPO):
        raise LA1PricingError("config path must stay inside repository")
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(REPO):
        raise LA1PricingError("output directory must stay inside repository")
    config = json.loads(config_path.read_bytes())
    if not isinstance(config, Mapping):
        raise LA1PricingError("config root must be an object")
    _validate_config(config)
    blobs, _, _ = _load_bound_sources(config)
    base_archive = blobs["c1_archive"]
    # Imported lazily: the existing J2 lift carries SciPy through its wider
    # optimizer module, while LA1's reusable coder core does not require it.
    from tac.optimization.direct_description_joint_descent import lift_v15_archive

    seeded_archive = lift_v15_archive(base_archive).lane_seed_archive()
    result = build_receipt(
        config,
        config_path=config_path,
        base_archive=base_archive,
        seeded_archive=seeded_archive,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "receipt.json"
    payload = canonical_json_bytes(result)
    if target.exists():
        if target.read_bytes() != payload:
            raise LA1PricingError("existing LA1 receipt differs; refusing overwrite")
        return result
    temporary = output_dir / f".{target.name}.{os.getpid()}.tmp"
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    return result


__all__ = [
    "ASSIGNMENT_SCHEMA",
    "CONFIG_SCHEMA",
    "LA1_CODECS",
    "LANE_ID",
    "RACE_SCHEMA",
    "SCHEMA",
    "LA1PricingError",
    "build_receipt",
    "canonical_json_bytes",
    "decode_la1_frame",
    "encode_la1_frame",
    "materialize",
    "race_payload",
    "race_stream",
]
