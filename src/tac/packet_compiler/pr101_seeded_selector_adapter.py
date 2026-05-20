# SPDX-License-Identifier: MIT
"""PR101/FEC6 seeded selector adapter feasibility probe.

[verified-against: src/tac/packet_compiler/pr101_fec6_packetir.py]
[verified-against: src/tac/packet_compiler/pr101_fec7_selector.py]
[verified-against: .omx/research/null_seed_candidate_spec_lowering_20260520T231100Z_codex.md]

The fec6 null-byte probe identified the selector payload as scorer-null at the
master-gradient surface. That does not imply the selector bytes can be deleted:
the runtime parser still needs a valid selector stream. This module tests the
smallest honest adapter class: ship archive-charged seed bytes, derive a
selector-code prior deterministically, and store residual overrides for any
pair whose code differs from the real selector.

It is a feasibility/profiling surface only. It never rewrites archives and
never claims score movement.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from tac.authority_contract import apply_false_authority_contract
from tac.packet_compiler.pr101_fec6_packetir import (
    FEC6_FIXED_K16_MODE_IDS,
    PR101_FEC6_DEFAULT_MEMBER_NAME,
    parse_pr101_fec6_packetir_member,
    read_single_stored_fec6_member_archive,
    sha256_hex,
)
from tac.procedural_codebook_generator import derive_codebook_from_seed

SEEDED_SELECTOR_SCHEMA = "pr101_seeded_selector_adapter_profile_v1"
SEEDED_SELECTOR_MAGIC = b"F6SD"
SEED_SEARCH_DOMAIN = b"tac.pr101_seeded_selector_adapter.seed_search.v1"

GeneratorKind = Literal["xorshift", "lcg", "pcg64"]
PredictionMode = Literal["constant", "seed_mod16", "histogram_shuffle"]
ResidualEncoding = Literal["bitmask_nibble_codes", "u16_index_nibble_codes", "none"]


class PR101SeededSelectorAdapterError(ValueError):
    """Raised when seeded-selector adapter inputs are malformed."""


@dataclass(frozen=True)
class ResidualEncodingResult:
    """Encoded residual overrides plus diagnostics."""

    payload: bytes
    encoding: ResidualEncoding
    mismatch_count: int
    mismatch_fraction: float


@dataclass(frozen=True)
class SeededSelectorCandidate:
    """One byte-closed seeded-selector candidate."""

    candidate_id: str
    prediction_mode: PredictionMode
    payload: bytes
    predicted_codes: tuple[int, ...]
    reconstructed_codes: tuple[int, ...]
    seed: bytes
    generator_kind: str
    residual_encoding: ResidualEncodingResult
    header_bytes: int
    model_bytes: int

    @property
    def payload_bytes(self) -> int:
        return len(self.payload)

    @property
    def reconstructs_selector(self) -> bool:
        return self.predicted_codes == self.reconstructed_codes


def encode_residual_overrides(
    target_codes: Sequence[int],
    predicted_codes: Sequence[int],
    *,
    alphabet_size: int = 16,
) -> ResidualEncodingResult:
    """Encode target-vs-predicted selector differences with charged bytes."""

    target = _validate_codes(target_codes, alphabet_size=alphabet_size)
    predicted = _validate_codes(predicted_codes, alphabet_size=alphabet_size)
    if len(target) != len(predicted):
        raise PR101SeededSelectorAdapterError("target/predicted code lengths differ")
    mismatches = [
        (idx, code)
        for idx, (code, pred) in enumerate(zip(target, predicted, strict=True))
        if code != pred
    ]
    if not mismatches:
        return ResidualEncodingResult(
            payload=b"",
            encoding="none",
            mismatch_count=0,
            mismatch_fraction=0.0,
        )

    mask_payload = _encode_bitmask_nibble_payload(len(target), mismatches)
    index_payload = _encode_u16_index_nibble_payload(mismatches)
    if len(mask_payload) <= len(index_payload):
        payload = mask_payload
        encoding: ResidualEncoding = "bitmask_nibble_codes"
    else:
        payload = index_payload
        encoding = "u16_index_nibble_codes"
    return ResidualEncodingResult(
        payload=payload,
        encoding=encoding,
        mismatch_count=len(mismatches),
        mismatch_fraction=len(mismatches) / len(target),
    )


def decode_residual_overrides(
    predicted_codes: Sequence[int],
    residual_payload: bytes,
    *,
    encoding: ResidualEncoding,
    mismatch_count: int,
    alphabet_size: int = 16,
) -> tuple[int, ...]:
    """Apply residual bytes emitted by :func:`encode_residual_overrides`."""

    predicted = list(_validate_codes(predicted_codes, alphabet_size=alphabet_size))
    if encoding == "none":
        if residual_payload or mismatch_count != 0:
            raise PR101SeededSelectorAdapterError("none residual carries bytes/count")
        return tuple(predicted)
    if encoding == "bitmask_nibble_codes":
        mask_bytes = (len(predicted) + 7) // 8
        if len(residual_payload) < mask_bytes:
            raise PR101SeededSelectorAdapterError("bitmask residual truncated")
        mask = residual_payload[:mask_bytes]
        positions = [
            idx
            for idx in range(len(predicted))
            if (mask[idx // 8] >> (7 - (idx % 8))) & 1
        ]
        codes = _unpack_nibbles(residual_payload[mask_bytes:], len(positions))
    elif encoding == "u16_index_nibble_codes":
        index_bytes = mismatch_count * 2
        if len(residual_payload) < index_bytes:
            raise PR101SeededSelectorAdapterError("u16 residual truncated")
        positions = [
            int.from_bytes(residual_payload[2 * i : 2 * i + 2], "little")
            for i in range(mismatch_count)
        ]
        codes = _unpack_nibbles(residual_payload[index_bytes:], mismatch_count)
    else:
        raise PR101SeededSelectorAdapterError(f"unknown residual encoding: {encoding}")
    if len(positions) != mismatch_count:
        raise PR101SeededSelectorAdapterError("residual mismatch_count disagreement")
    for idx, code in zip(positions, codes, strict=True):
        if idx < 0 or idx >= len(predicted):
            raise PR101SeededSelectorAdapterError("residual index out of range")
        if code < 0 or code >= alphabet_size:
            raise PR101SeededSelectorAdapterError("residual code out of range")
        predicted[idx] = int(code)
    return tuple(predicted)


def build_seeded_selector_candidate(
    target_codes: Sequence[int],
    *,
    prediction_mode: PredictionMode,
    seed: bytes = b"",
    generator_kind: GeneratorKind = "pcg64",
    constant_code: int = 0,
    histogram_counts: Sequence[int] | None = None,
    alphabet_size: int = 16,
) -> SeededSelectorCandidate:
    """Build one byte-closed seeded-selector candidate payload."""

    target = _validate_codes(target_codes, alphabet_size=alphabet_size)
    if len(target) > 0xFFFF:
        raise PR101SeededSelectorAdapterError("selector code count exceeds u16")
    if prediction_mode == "constant":
        if not 0 <= int(constant_code) < alphabet_size:
            raise PR101SeededSelectorAdapterError("constant_code outside alphabet")
        predicted = tuple(int(constant_code) for _ in target)
        model_payload = bytes([int(constant_code)])
    elif prediction_mode == "seed_mod16":
        seed = _require_seed(seed)
        predicted = _seed_mod16_codes(
            seed,
            n_pairs=len(target),
            generator_kind=generator_kind,
            alphabet_size=alphabet_size,
        )
        model_payload = b""
    elif prediction_mode == "histogram_shuffle":
        seed = _require_seed(seed)
        counts = (
            tuple(int(Counter(target).get(code, 0)) for code in range(alphabet_size))
            if histogram_counts is None
            else tuple(int(value) for value in histogram_counts)
        )
        if len(counts) != alphabet_size or sum(counts) != len(target) or any(c < 0 for c in counts):
            raise PR101SeededSelectorAdapterError(
                "histogram_counts must have alphabet_size non-negative values summing to n_pairs"
            )
        predicted = _histogram_shuffle_codes(
            counts,
            seed,
            generator_kind=generator_kind,
        )
        model_payload = b"".join(int(count).to_bytes(2, "little") for count in counts)
    else:
        raise PR101SeededSelectorAdapterError(f"unknown prediction_mode: {prediction_mode}")

    residual = encode_residual_overrides(
        target,
        predicted,
        alphabet_size=alphabet_size,
    )
    reconstructed = decode_residual_overrides(
        predicted,
        residual.payload,
        encoding=residual.encoding,
        mismatch_count=residual.mismatch_count,
        alphabet_size=alphabet_size,
    )
    if reconstructed != target:
        raise PR101SeededSelectorAdapterError("internal residual reconstruction failed")
    payload = _encode_seeded_selector_payload(
        prediction_mode=prediction_mode,
        n_pairs=len(target),
        alphabet_size=alphabet_size,
        generator_kind=generator_kind,
        seed=seed,
        model_payload=model_payload,
        residual=residual,
    )
    return SeededSelectorCandidate(
        candidate_id=_candidate_id(
            prediction_mode=prediction_mode,
            seed=seed,
            generator_kind=generator_kind,
            constant_code=constant_code,
            model_payload=model_payload,
            residual=residual,
        ),
        prediction_mode=prediction_mode,
        payload=payload,
        predicted_codes=predicted,
        reconstructed_codes=reconstructed,
        seed=seed,
        generator_kind=generator_kind,
        residual_encoding=residual,
        header_bytes=_header_bytes(seed),
        model_bytes=len(model_payload),
    )


def profile_seeded_selector_adapter(
    target_codes: Sequence[int],
    *,
    fec6_selector_payload_bytes: int,
    seed_lengths: Sequence[int] = (1, 2, 4, 8, 16, 32),
    search_seeds_per_length: int = 256,
    generator_kind: GeneratorKind = "pcg64",
    target_saving_bytes: int = 1,
    alphabet_size: int = 16,
) -> dict[str, Any]:
    """Profile constant, seed-mod16, and histogram-shuffle adapter candidates."""

    target = _validate_codes(target_codes, alphabet_size=alphabet_size)
    if search_seeds_per_length <= 0:
        raise PR101SeededSelectorAdapterError("search_seeds_per_length must be positive")
    seed_lengths = tuple(int(length) for length in seed_lengths)
    if any(length <= 0 for length in seed_lengths):
        raise PR101SeededSelectorAdapterError("seed_lengths must all be positive")

    candidates: list[SeededSelectorCandidate] = []
    for code, _count in Counter(target).most_common():
        candidates.append(
            build_seeded_selector_candidate(
                target,
                prediction_mode="constant",
                constant_code=int(code),
                generator_kind=generator_kind,
                alphabet_size=alphabet_size,
            )
        )
    for seed_len in seed_lengths:
        best_seed_mod: SeededSelectorCandidate | None = None
        best_hist: SeededSelectorCandidate | None = None
        for ordinal in range(int(search_seeds_per_length)):
            seed = deterministic_seed(seed_len, ordinal)
            seed_mod = build_seeded_selector_candidate(
                target,
                prediction_mode="seed_mod16",
                seed=seed,
                generator_kind=generator_kind,
                alphabet_size=alphabet_size,
            )
            hist = build_seeded_selector_candidate(
                target,
                prediction_mode="histogram_shuffle",
                seed=seed,
                generator_kind=generator_kind,
                alphabet_size=alphabet_size,
            )
            best_seed_mod = _min_candidate(best_seed_mod, seed_mod)
            best_hist = _min_candidate(best_hist, hist)
        if best_seed_mod is not None:
            candidates.append(best_seed_mod)
        if best_hist is not None:
            candidates.append(best_hist)

    records = [
        candidate_record(
            candidate,
            fec6_selector_payload_bytes=fec6_selector_payload_bytes,
            target_saving_bytes=target_saving_bytes,
        )
        for candidate in candidates
    ]
    records.sort(
        key=lambda row: (
            int(row["payload_bytes"]),
            int(row["mismatch_count"]),
            str(row["candidate_id"]),
        )
    )
    best = records[0]
    blocked = int(best["saving_vs_fec6_selector_bytes"]) < int(target_saving_bytes)
    order_stats = selector_order_entropy_stats(target, context_mods=(2, 4, 8, 16, 25, 50, 100))
    profile = {
        "schema": SEEDED_SELECTOR_SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_provider_dispatch": False,
        "dispatch_attempted": False,
        "selector_code_count": len(target),
        "alphabet_size": alphabet_size,
        "generator_kind": generator_kind,
        "seed_lengths": list(seed_lengths),
        "search_seeds_per_length": int(search_seeds_per_length),
        "fec6_selector_payload_bytes": int(fec6_selector_payload_bytes),
        "target_saving_bytes": int(target_saving_bytes),
        "selector_histogram": {
            str(code): int(count)
            for code, count in sorted(Counter(target).items())
        },
        "order_entropy": order_stats,
        "candidate_count": len(records),
        "best_candidate": best,
        "charged_candidates": records,
        "can_meet_target_with_seeded_selector_adapter": not blocked,
        "explicit_blocker": {
            "blocked": bool(blocked),
            "reason": (
                "Seed-derived selector priors still require enough residual "
                "overrides that the charged adapter payload does not beat the "
                "current FEC6 selector payload."
            ),
            "reactivation_criteria": (
                "Reopen with a predictor whose residual override payload plus "
                "all charged model/seed bytes is smaller than the current FEC6 "
                "selector payload, then materialize a runtime adapter and run "
                "runtime-consumption/no-op plus exact CPU/CUDA eval."
            ),
        },
    }
    return apply_false_authority_contract(
        profile,
        preserve_dispatch_ready=False,
        reason="seeded_selector_adapter_profile_requires_materialized_runtime_and_exact_eval",
    )


def profile_archive_seeded_selector_adapter(
    archive_bytes: bytes,
    *,
    fec6_selector_payload_bytes: int | None = None,
    expected_member_name: str = PR101_FEC6_DEFAULT_MEMBER_NAME,
    seed_lengths: Sequence[int] = (1, 2, 4, 8, 16, 32),
    search_seeds_per_length: int = 256,
    generator_kind: GeneratorKind = "pcg64",
    target_saving_bytes: int = 1,
) -> dict[str, Any]:
    """Parse a PR101/FEC6 archive and profile seeded selector candidates."""

    member = read_single_stored_fec6_member_archive(
        archive_bytes,
        expected_member_name=expected_member_name,
    )
    packet = parse_pr101_fec6_packetir_member(member.payload)
    profile = profile_seeded_selector_adapter(
        packet.selector_codes,
        fec6_selector_payload_bytes=(
            len(packet.selector_fec6_payload)
            if fec6_selector_payload_bytes is None
            else int(fec6_selector_payload_bytes)
        ),
        seed_lengths=seed_lengths,
        search_seeds_per_length=search_seeds_per_length,
        generator_kind=generator_kind,
        target_saving_bytes=target_saving_bytes,
    )
    profile.update(
        {
            "archive_sha256": sha256_hex(archive_bytes),
            "member_name": member.name,
            "member_payload_bytes": len(member.payload),
            "member_payload_sha256": sha256_hex(member.payload),
            "fec6_selector_payload_sha256": sha256_hex(packet.selector_fec6_payload),
            "fec6_selector_index_bytes": len(packet.selector_bitstream),
            "fec6_selector_code_bits_total": packet.selector_code_bits_total,
        }
    )
    return profile


def candidate_record(
    candidate: SeededSelectorCandidate,
    *,
    fec6_selector_payload_bytes: int,
    target_saving_bytes: int,
) -> dict[str, Any]:
    """Return an operator-readable record for one seeded-selector candidate."""

    saving = int(fec6_selector_payload_bytes) - candidate.payload_bytes
    return {
        "candidate_id": candidate.candidate_id,
        "prediction_mode": candidate.prediction_mode,
        "payload_bytes": candidate.payload_bytes,
        "payload_sha256": sha256_hex(candidate.payload),
        "saving_vs_fec6_selector_bytes": saving,
        "meets_target_saving": saving >= int(target_saving_bytes),
        "seed_bytes": len(candidate.seed),
        "seed_sha256": sha256_hex(candidate.seed),
        "generator_kind": candidate.generator_kind,
        "header_bytes": candidate.header_bytes,
        "model_bytes": candidate.model_bytes,
        "residual_encoding": candidate.residual_encoding.encoding,
        "residual_payload_bytes": len(candidate.residual_encoding.payload),
        "mismatch_count": candidate.residual_encoding.mismatch_count,
        "mismatch_fraction": candidate.residual_encoding.mismatch_fraction,
        "reconstructs_selector_codes": candidate.reconstructed_codes
        == candidate.predicted_codes
        if candidate.residual_encoding.mismatch_count == 0
        else True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def selector_order_entropy_stats(
    codes: Sequence[int],
    *,
    context_mods: Sequence[int] = (2, 4, 8, 16, 25, 50, 100),
    alphabet_size: int = 16,
) -> dict[str, Any]:
    """Return order-aware entropy lower bounds for the selector sequence.

    Global entropy ignores order. The contest selector is ordered by pair
    index, so pair-index contexts and first-order transitions can lower the
    theoretical stream floor. These are lower bounds only: their model tables
    must be charged before they become byte-closed candidates.
    """

    target = _validate_codes(codes, alphabet_size=alphabet_size)
    runs = _run_lengths(target)
    transition_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for prev, nxt in zip(target, target[1:], strict=False):
        transition_counts[int(prev)][int(nxt)] += 1
    pairmod = []
    for mod in context_mods:
        mod_i = int(mod)
        if mod_i <= 0:
            raise PR101SeededSelectorAdapterError("context_mods must be positive")
        buckets: dict[int, Counter[int]] = defaultdict(Counter)
        for idx, code in enumerate(target):
            buckets[idx % mod_i][int(code)] += 1
        floor = sum(_entropy_floor_bytes(counter) for counter in buckets.values())
        pairmod.append(
            {
                "context_mod": mod_i,
                "zero_model_entropy_floor_bytes": int(floor),
                "charged_model_counts_bytes_u16": mod_i * alphabet_size * 2,
                "byte_closed_floor_with_u16_counts_bytes": int(
                    floor + mod_i * alphabet_size * 2
                ),
            }
        )
    pairmod.sort(
        key=lambda row: (
            int(row["zero_model_entropy_floor_bytes"]),
            int(row["charged_model_counts_bytes_u16"]),
        )
    )
    transition_floor = 1 + sum(
        _entropy_floor_bytes(counter) for counter in transition_counts.values()
    )
    transition_model_bytes = alphabet_size * alphabet_size * 2
    return {
        "global_entropy_floor_bytes": _entropy_floor_bytes(Counter(target)),
        "first_order_transition_entropy_floor_bytes_plus_first_symbol": int(
            transition_floor
        ),
        "first_order_transition_model_counts_bytes_u16": transition_model_bytes,
        "first_order_transition_byte_closed_floor_with_u16_counts_bytes": int(
            transition_floor + transition_model_bytes
        ),
        "run_count": len(runs),
        "run_mean_length": sum(runs) / len(runs),
        "run_max_length": max(runs),
        "pairmod_context_lower_bounds": pairmod,
        "order_lower_bound_note": (
            "Zero-model floors are diagnostic only; byte-closed candidates must "
            "charge the context/transition model bytes or derive them from an "
            "allowed runtime prior."
        ),
    }


def deterministic_seed(seed_len: int, ordinal: int) -> bytes:
    """Return deterministic search seed ``ordinal`` with ``seed_len`` bytes."""

    if seed_len <= 0:
        raise PR101SeededSelectorAdapterError("seed_len must be positive")
    if ordinal < 0:
        raise PR101SeededSelectorAdapterError("ordinal must be non-negative")
    out = bytearray()
    counter = 0
    while len(out) < seed_len:
        h = hashlib.sha256()
        h.update(SEED_SEARCH_DOMAIN)
        h.update(int(seed_len).to_bytes(4, "little"))
        h.update(int(ordinal).to_bytes(8, "little"))
        h.update(int(counter).to_bytes(4, "little"))
        out.extend(h.digest())
        counter += 1
    return bytes(out[:seed_len])


def render_seeded_selector_profile_markdown(profile: Mapping[str, Any]) -> str:
    """Render a compact profile memo."""

    best = _require_mapping(profile.get("best_candidate"), "best_candidate")
    blocker = _require_mapping(profile.get("explicit_blocker"), "explicit_blocker")
    lines = [
        "# PR101 Seeded Selector Adapter Profile",
        "",
        f"- Schema: `{profile.get('schema')}`",
        f"- Score claim: `{str(profile.get('score_claim')).lower()}`",
        f"- Ready for exact eval dispatch: `{str(profile.get('ready_for_exact_eval_dispatch')).lower()}`",
        f"- Selector codes: `{profile.get('selector_code_count')}`",
        f"- FEC6 selector payload bytes: `{profile.get('fec6_selector_payload_bytes')}`",
        f"- Candidate count: `{profile.get('candidate_count')}`",
        f"- Best candidate: `{best.get('candidate_id')}`",
        f"- Best candidate bytes: `{best.get('payload_bytes')}`",
        f"- Best saving vs FEC6 selector: `{best.get('saving_vs_fec6_selector_bytes')}`",
        f"- Can meet target: `{str(profile.get('can_meet_target_with_seeded_selector_adapter')).lower()}`",
        "",
        "## Order Entropy",
        "",
    ]
    order = profile.get("order_entropy")
    if isinstance(order, Mapping):
        lines.extend(
            [
                f"- Global entropy floor bytes: `{order.get('global_entropy_floor_bytes')}`",
                "- First-order transition floor bytes plus first symbol: "
                f"`{order.get('first_order_transition_entropy_floor_bytes_plus_first_symbol')}`",
                "- First-order transition byte-closed floor with u16 counts: "
                f"`{order.get('first_order_transition_byte_closed_floor_with_u16_counts_bytes')}`",
                f"- Run count: `{order.get('run_count')}`",
                f"- Run mean length: `{float(order.get('run_mean_length', 0.0)):.3f}`",
                f"- Run max length: `{order.get('run_max_length')}`",
                "",
                "| context_mod | zero-model floor bytes | u16 model bytes | byte-closed floor bytes |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for row in order.get("pairmod_context_lower_bounds", [])[:8]:
            rec = _require_mapping(row, "pairmod_context_lower_bound")
            lines.append(
                "| {mod} | {floor} | {model} | {closed} |".format(
                    mod=rec.get("context_mod"),
                    floor=rec.get("zero_model_entropy_floor_bytes"),
                    model=rec.get("charged_model_counts_bytes_u16"),
                    closed=rec.get("byte_closed_floor_with_u16_counts_bytes"),
                )
            )
        lines.extend(["", str(order.get("order_lower_bound_note")), ""])
    lines.extend(
        [
        "## Charged Candidates",
        "",
        "| candidate | mode | bytes | saving | seed bytes | model bytes | residual bytes | mismatches |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in profile.get("charged_candidates", [])[:32]:
        rec = _require_mapping(row, "charged_candidate")
        lines.append(
            "| {candidate} | {mode} | {bytes_} | {saving} | {seed} | {model} | {residual} | {mismatch} |".format(
                candidate=rec.get("candidate_id"),
                mode=rec.get("prediction_mode"),
                bytes_=rec.get("payload_bytes"),
                saving=rec.get("saving_vs_fec6_selector_bytes"),
                seed=rec.get("seed_bytes"),
                model=rec.get("model_bytes"),
                residual=rec.get("residual_payload_bytes"),
                mismatch=rec.get("mismatch_count"),
            )
        )
    lines.extend(
        [
            "",
            "## Blocker",
            "",
            f"- blocked: `{str(blocker.get('blocked')).lower()}`",
            f"- reason: {blocker.get('reason')}",
            f"- reactivation_criteria: {blocker.get('reactivation_criteria')}",
            "",
        ]
    )
    return "\n".join(lines)


def _encode_seeded_selector_payload(
    *,
    prediction_mode: PredictionMode,
    n_pairs: int,
    alphabet_size: int,
    generator_kind: str,
    seed: bytes,
    model_payload: bytes,
    residual: ResidualEncodingResult,
) -> bytes:
    if len(seed) > 255:
        raise PR101SeededSelectorAdapterError("seed length must fit in u8")
    if len(model_payload) > 0xFFFF:
        raise PR101SeededSelectorAdapterError("model payload length must fit in u16")
    if len(residual.payload) > 0xFFFF:
        raise PR101SeededSelectorAdapterError("residual payload length must fit in u16")
    return b"".join(
        (
            SEEDED_SELECTOR_MAGIC,
            bytes(
                [
                    _prediction_mode_code(prediction_mode),
                    _generator_kind_code(generator_kind),
                ]
            ),
            int(n_pairs).to_bytes(2, "little"),
            bytes([int(alphabet_size), len(seed)]),
            len(model_payload).to_bytes(2, "little"),
            len(residual.payload).to_bytes(2, "little"),
            int(residual.mismatch_count).to_bytes(2, "little"),
            bytes([_residual_encoding_code(residual.encoding)]),
            seed,
            model_payload,
            residual.payload,
        )
    )


def _header_bytes(seed: bytes) -> int:
    return 4 + 1 + 1 + 2 + 1 + 1 + 2 + 2 + 2 + 1 + len(seed)


def _seed_mod16_codes(
    seed: bytes,
    *,
    n_pairs: int,
    generator_kind: str,
    alphabet_size: int,
) -> tuple[int, ...]:
    raw = derive_codebook_from_seed(
        seed,
        output_shape=(n_pairs,),
        dtype=np.uint8,
        generator_kind=generator_kind,  # type: ignore[arg-type]
    )
    return tuple(int(value) % alphabet_size for value in raw.tolist())


def _histogram_shuffle_codes(
    counts: Sequence[int],
    seed: bytes,
    *,
    generator_kind: str,
) -> tuple[int, ...]:
    codes: list[int] = []
    for code, count in enumerate(counts):
        codes.extend([int(code)] * int(count))
    if not codes:
        return ()
    keys = derive_codebook_from_seed(
        seed,
        output_shape=(len(codes),),
        dtype=np.uint32,
        generator_kind=generator_kind,  # type: ignore[arg-type]
    )
    order = sorted(range(len(codes)), key=lambda idx: (int(keys[idx]), idx))
    return tuple(codes[idx] for idx in order)


def _entropy_floor_bytes(counts: Mapping[int, int] | Counter[int]) -> int:
    total = sum(int(count) for count in counts.values())
    if total <= 0:
        return 0
    bits = 0.0
    for count_raw in counts.values():
        count = int(count_raw)
        if count <= 0:
            continue
        p = count / total
        bits -= count * math.log2(p)
    return math.ceil(bits / 8.0)


def _run_lengths(codes: Sequence[int]) -> list[int]:
    if not codes:
        return []
    runs: list[int] = []
    prev = int(codes[0])
    length = 1
    for code_raw in codes[1:]:
        code = int(code_raw)
        if code == prev:
            length += 1
        else:
            runs.append(length)
            prev = code
            length = 1
    runs.append(length)
    return runs


def _encode_bitmask_nibble_payload(
    n_pairs: int, mismatches: Sequence[tuple[int, int]]
) -> bytes:
    mask = bytearray((n_pairs + 7) // 8)
    codes: list[int] = []
    for idx, code in mismatches:
        mask[idx // 8] |= 1 << (7 - (idx % 8))
        codes.append(int(code))
    return bytes(mask) + _pack_nibbles(codes)


def _encode_u16_index_nibble_payload(
    mismatches: Sequence[tuple[int, int]]
) -> bytes:
    return b"".join(int(idx).to_bytes(2, "little") for idx, _code in mismatches) + _pack_nibbles(
        [code for _idx, code in mismatches]
    )


def _pack_nibbles(codes: Sequence[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(codes), 2):
        hi = int(codes[i]) & 0x0F
        lo = int(codes[i + 1]) & 0x0F if i + 1 < len(codes) else 0
        out.append((hi << 4) | lo)
    return bytes(out)


def _unpack_nibbles(payload: bytes, count: int) -> tuple[int, ...]:
    codes: list[int] = []
    for byte in payload:
        codes.append((byte >> 4) & 0x0F)
        if len(codes) == count:
            return tuple(codes)
        codes.append(byte & 0x0F)
        if len(codes) == count:
            return tuple(codes)
    if len(codes) < count:
        raise PR101SeededSelectorAdapterError("nibble code residual truncated")
    return tuple(codes[:count])


def _candidate_id(
    *,
    prediction_mode: PredictionMode,
    seed: bytes,
    generator_kind: str,
    constant_code: int,
    model_payload: bytes,
    residual: ResidualEncodingResult,
) -> str:
    digest = hashlib.sha256(seed + model_payload + residual.payload).hexdigest()[:12]
    if prediction_mode == "constant":
        detail = f"code{constant_code}"
    else:
        detail = f"{generator_kind}_seed{len(seed)}"
    return f"seeded_selector::{prediction_mode}::{detail}::{digest}"


def _min_candidate(
    current: SeededSelectorCandidate | None,
    new: SeededSelectorCandidate,
) -> SeededSelectorCandidate:
    if current is None:
        return new
    key_new = (new.payload_bytes, new.residual_encoding.mismatch_count, new.candidate_id)
    key_current = (
        current.payload_bytes,
        current.residual_encoding.mismatch_count,
        current.candidate_id,
    )
    return new if key_new < key_current else current


def _validate_codes(
    codes: Sequence[int], *, alphabet_size: int
) -> tuple[int, ...]:
    out = tuple(int(code) for code in codes)
    if not out:
        raise PR101SeededSelectorAdapterError("selector code stream is empty")
    if min(out) < 0 or max(out) >= alphabet_size:
        raise PR101SeededSelectorAdapterError("selector code outside alphabet")
    return out


def _require_seed(seed: bytes) -> bytes:
    out = bytes(seed)
    if not out:
        raise PR101SeededSelectorAdapterError("seed is required for this mode")
    if len(out) > 255:
        raise PR101SeededSelectorAdapterError("seed must fit in u8 length field")
    return out


def _prediction_mode_code(mode: PredictionMode) -> int:
    return {"constant": 1, "seed_mod16": 2, "histogram_shuffle": 3}[mode]


def _generator_kind_code(kind: str) -> int:
    return {"xorshift": 1, "lcg": 2, "pcg64": 3}[kind]


def _residual_encoding_code(encoding: ResidualEncoding) -> int:
    return {"none": 0, "bitmask_nibble_codes": 1, "u16_index_nibble_codes": 2}[
        encoding
    ]


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PR101SeededSelectorAdapterError(f"{name} must be a mapping")
    return value


__all__ = [
    "SEEDED_SELECTOR_MAGIC",
    "SEEDED_SELECTOR_SCHEMA",
    "PR101SeededSelectorAdapterError",
    "ResidualEncodingResult",
    "SeededSelectorCandidate",
    "build_seeded_selector_candidate",
    "candidate_record",
    "decode_residual_overrides",
    "deterministic_seed",
    "encode_residual_overrides",
    "profile_archive_seeded_selector_adapter",
    "profile_seeded_selector_adapter",
    "render_seeded_selector_profile_markdown",
    "selector_order_entropy_stats",
]
