# SPDX-License-Identifier: MIT
"""Tests for slot 6 grain-aware routing in tac.master_gradient_per_byte_consumer
+ tac.cathedral_consumers.per_byte_sensitivity_consumer (v1.1).

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" + Catalog #318 + codex op7
finding 2026-05-19 (raw-archive-byte gradients are entropy-cascade-smeared).
Slot 6 + slot 10 grain-awareness wave.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.master_gradient_per_byte_consumer import (
    GRAIN_DIAGNOSTIC,
    GRAIN_POST_ARITHMETIC_DECOMPRESS,
    GRAIN_POST_BROTLI_DECOMPRESS,
    GRAIN_POST_DECOMPRESS_GENERIC,
    GRAIN_RAW_ARCHIVE_BYTE,
    GRAIN_RAW_ZIP_INNER_MEMBER_PAYLOAD,
    POST_DECOMPRESS_GRAINS,
    RAW_BYTE_GRAINS,
    PerByteSensitivityPayload,
    available_grains_for_archive,
    load_per_byte_sensitivity_for_archive,
    summarize_payload,
)


# ─────────────────────────────────────────────────────────────────────────
# Canonical grain taxonomy constants
# ─────────────────────────────────────────────────────────────────────────


def test_grain_constants_match_slot_15_canonical() -> None:
    """The post-brotli grain MUST match slot 15's MUTATION_GRAIN_POST_BROTLI_DECOMPRESS."""
    assert GRAIN_POST_BROTLI_DECOMPRESS == "post_brotli_decompress_decoder_weight_bytes"


def test_grain_constants_raw_archive_byte_canonical() -> None:
    """The raw-archive-byte grain matches MasterGradient.gradient_byte_domain default."""
    assert GRAIN_RAW_ARCHIVE_BYTE == "scored_archive_bytes"


def test_grain_constants_zip_inner_member_payload() -> None:
    """The zip-inner-member-payload grain matches live PR101 ledger anchors."""
    assert GRAIN_RAW_ZIP_INNER_MEMBER_PAYLOAD == "zip_inner_member_payload"


def test_post_decompress_grains_tuple_membership() -> None:
    """The 3 canonical post-decompress grains are in POST_DECOMPRESS_GRAINS."""
    assert GRAIN_POST_BROTLI_DECOMPRESS in POST_DECOMPRESS_GRAINS
    assert GRAIN_POST_ARITHMETIC_DECOMPRESS in POST_DECOMPRESS_GRAINS
    assert GRAIN_POST_DECOMPRESS_GENERIC in POST_DECOMPRESS_GRAINS


def test_raw_byte_grains_tuple_membership() -> None:
    """The 2 canonical raw-byte grains are in RAW_BYTE_GRAINS."""
    assert GRAIN_RAW_ARCHIVE_BYTE in RAW_BYTE_GRAINS
    assert GRAIN_RAW_ZIP_INNER_MEMBER_PAYLOAD in RAW_BYTE_GRAINS


def test_grain_classes_disjoint() -> None:
    """POST_DECOMPRESS_GRAINS and RAW_BYTE_GRAINS are disjoint."""
    assert set(POST_DECOMPRESS_GRAINS).isdisjoint(set(RAW_BYTE_GRAINS))


def test_grain_diagnostic_not_in_either_class() -> None:
    """GRAIN_DIAGNOSTIC is canonical 'unknown' — fail-conservative cascade_risk=True."""
    assert GRAIN_DIAGNOSTIC not in POST_DECOMPRESS_GRAINS
    assert GRAIN_DIAGNOSTIC not in RAW_BYTE_GRAINS


# ─────────────────────────────────────────────────────────────────────────
# PerByteSensitivityPayload invariant (cascade_smearing_risk vs grain)
# ─────────────────────────────────────────────────────────────────────────


def _make_payload(
    *,
    grain: str = GRAIN_RAW_ARCHIVE_BYTE,
    cascade: bool = True,
) -> PerByteSensitivityPayload:
    return PerByteSensitivityPayload(
        archive_sha256="a" * 64,
        gradient_array_path="/tmp/x.npy",
        n_bytes=100,
        measurement_axis="[contest-CPU]",
        measurement_hardware="linux_x86_64_t4",
        measurement_method="autograd_per_parameter",
        measurement_utc="2026-05-19T00:00:00Z",
        gradient_byte_domain=grain,
        cascade_smearing_risk=cascade,
    )


def test_payload_post_decompress_grain_with_cascade_false_accepted() -> None:
    p = _make_payload(grain=GRAIN_POST_BROTLI_DECOMPRESS, cascade=False)
    assert p.cascade_smearing_risk is False
    assert p.gradient_byte_domain == GRAIN_POST_BROTLI_DECOMPRESS


def test_payload_raw_byte_grain_with_cascade_true_accepted() -> None:
    p = _make_payload(grain=GRAIN_RAW_ARCHIVE_BYTE, cascade=True)
    assert p.cascade_smearing_risk is True


def test_payload_post_decompress_with_cascade_true_rejected() -> None:
    """Invariant violation: post-decompress grain MUST have cascade_risk=False."""
    with pytest.raises(ValueError, match="cascade_smearing_risk"):
        _make_payload(grain=GRAIN_POST_BROTLI_DECOMPRESS, cascade=True)


def test_payload_raw_byte_with_cascade_false_rejected() -> None:
    """Invariant violation: raw-byte grain MUST have cascade_risk=True."""
    with pytest.raises(ValueError, match="cascade_smearing_risk"):
        _make_payload(grain=GRAIN_RAW_ARCHIVE_BYTE, cascade=False)


def test_payload_diagnostic_grain_cascade_true_accepted() -> None:
    """Diagnostic / unknown grain default-conservatively cascade_risk=True."""
    p = _make_payload(grain=GRAIN_DIAGNOSTIC, cascade=True)
    assert p.cascade_smearing_risk is True


def test_payload_invalid_grain_string_raises() -> None:
    """Empty grain string is rejected at construction."""
    with pytest.raises(ValueError, match="gradient_byte_domain"):
        _make_payload(grain="", cascade=True)


def test_payload_invalid_cascade_type_raises() -> None:
    """Non-bool cascade_smearing_risk is rejected at construction."""
    with pytest.raises(TypeError, match="cascade_smearing_risk"):
        PerByteSensitivityPayload(
            archive_sha256="a" * 64,
            gradient_array_path="/tmp/x.npy",
            n_bytes=100,
            measurement_axis="[contest-CPU]",
            measurement_hardware="linux_x86_64_t4",
            measurement_method="autograd",
            measurement_utc="2026-05-19T00:00:00Z",
            gradient_byte_domain=GRAIN_RAW_ARCHIVE_BYTE,
            cascade_smearing_risk="true",  # type: ignore[arg-type]
        )


# ─────────────────────────────────────────────────────────────────────────
# summarize_payload grain-aware fields
# ─────────────────────────────────────────────────────────────────────────


def test_summarize_payload_includes_grain_fields() -> None:
    p = _make_payload(grain=GRAIN_POST_BROTLI_DECOMPRESS, cascade=False)
    s = summarize_payload(p)
    assert s["gradient_byte_domain"] == GRAIN_POST_BROTLI_DECOMPRESS
    assert s["cascade_smearing_risk"] is False


# ─────────────────────────────────────────────────────────────────────────
# load_per_byte_sensitivity_for_archive prefer_grain cascade
# ─────────────────────────────────────────────────────────────────────────


def _write_anchor(
    tmp_path: Path,
    *,
    archive_sha: str,
    n_bytes: int = 50,
    grain: str = GRAIN_RAW_ARCHIVE_BYTE,
    utc: str = "2026-05-19T03:00:00Z",
    ledger_name: str = "master_gradient_anchors.jsonl",
) -> Path:
    """Append an anchor row to the per-archive ledger."""
    ledger = tmp_path / ledger_name
    npy_path = tmp_path / f"grad_{archive_sha[:8]}_{grain[:8]}.npy"
    arr = np.random.RandomState(hash(grain) & 0xFFFF).randn(n_bytes, 3).astype(np.float32)
    np.save(npy_path, arr)
    row = {
        "archive_sha256": archive_sha,
        "gradient_array_path": str(npy_path),
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "gradient_byte_domain": grain,
        "n_bytes": n_bytes,
        "operating_point": {"d_seg": 0.001, "d_pose": 0.002, "rate": 0.005, "score": 0.34},
        "measurement_axis": "[contest-CPU]",
        "measurement_hardware": "linux_x86_64_t4",
        "measurement_method": "autograd",
        "measurement_utc": utc,
        "written_at_utc": utc,
        "schema_version": "master_gradient_anchor_v1",
    }
    with ledger.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return ledger


def test_prefer_grain_post_decompress_picks_post_when_both_exist(tmp_path: Path) -> None:
    """When BOTH grains exist, prefer_grain=post_decompress picks post-decompress."""
    archive_sha = "b" * 64
    ledger = _write_anchor(
        tmp_path,
        archive_sha=archive_sha,
        grain=GRAIN_RAW_ZIP_INNER_MEMBER_PAYLOAD,
        utc="2026-05-19T05:00:00Z",  # newer (would win latest-by-utc)
    )
    _write_anchor(
        tmp_path,
        archive_sha=archive_sha,
        grain=GRAIN_POST_BROTLI_DECOMPRESS,
        utc="2026-05-19T03:00:00Z",  # older
        ledger_name="master_gradient_anchors.jsonl",
    )

    payload = load_per_byte_sensitivity_for_archive(
        archive_sha, path=ledger, prefer_grain="post_decompress"
    )
    assert payload is not None
    # Post-decompress wins despite being older.
    assert payload.gradient_byte_domain == GRAIN_POST_BROTLI_DECOMPRESS
    assert payload.cascade_smearing_risk is False


def test_prefer_grain_post_decompress_falls_back_to_raw(tmp_path: Path) -> None:
    """When only raw-byte exists, prefer_grain=post_decompress falls back."""
    archive_sha = "c" * 64
    ledger = _write_anchor(
        tmp_path, archive_sha=archive_sha, grain=GRAIN_RAW_ARCHIVE_BYTE
    )

    payload = load_per_byte_sensitivity_for_archive(
        archive_sha,
        path=ledger,
        prefer_grain="post_decompress",
        fallback_to_raw_byte=True,
    )
    assert payload is not None
    assert payload.gradient_byte_domain == GRAIN_RAW_ARCHIVE_BYTE
    assert payload.cascade_smearing_risk is True


def test_prefer_grain_post_decompress_no_fallback_returns_none(tmp_path: Path) -> None:
    """With fallback_to_raw_byte=False and only raw-byte present, returns None."""
    archive_sha = "d" * 64
    ledger = _write_anchor(
        tmp_path, archive_sha=archive_sha, grain=GRAIN_RAW_ARCHIVE_BYTE
    )

    payload = load_per_byte_sensitivity_for_archive(
        archive_sha,
        path=ledger,
        prefer_grain="post_decompress",
        fallback_to_raw_byte=False,
    )
    assert payload is None


def test_prefer_grain_raw_byte_picks_raw_when_both_exist(tmp_path: Path) -> None:
    """prefer_grain=raw_byte picks raw-byte even when post-decompress is newer."""
    archive_sha = "e" * 64
    ledger = _write_anchor(
        tmp_path,
        archive_sha=archive_sha,
        grain=GRAIN_RAW_ARCHIVE_BYTE,
        utc="2026-05-19T01:00:00Z",
    )
    _write_anchor(
        tmp_path,
        archive_sha=archive_sha,
        grain=GRAIN_POST_BROTLI_DECOMPRESS,
        utc="2026-05-19T05:00:00Z",
    )

    payload = load_per_byte_sensitivity_for_archive(
        archive_sha, path=ledger, prefer_grain="raw_byte"
    )
    assert payload is not None
    assert payload.gradient_byte_domain == GRAIN_RAW_ARCHIVE_BYTE


def test_prefer_grain_any_picks_latest_regardless(tmp_path: Path) -> None:
    """prefer_grain=any picks the chronologically newest anchor."""
    archive_sha = "f" * 64
    ledger = _write_anchor(
        tmp_path,
        archive_sha=archive_sha,
        grain=GRAIN_RAW_ARCHIVE_BYTE,
        utc="2026-05-19T01:00:00Z",
    )
    _write_anchor(
        tmp_path,
        archive_sha=archive_sha,
        grain=GRAIN_POST_BROTLI_DECOMPRESS,
        utc="2026-05-19T05:00:00Z",
    )

    payload = load_per_byte_sensitivity_for_archive(
        archive_sha, path=ledger, prefer_grain="any"
    )
    assert payload is not None
    # Post-decompress is newest so latest-by-utc picks it.
    assert payload.gradient_byte_domain == GRAIN_POST_BROTLI_DECOMPRESS


def test_prefer_grain_invalid_value_rejected(tmp_path: Path) -> None:
    """prefer_grain must be one of {post_decompress, raw_byte, any}."""
    archive_sha = "a" * 64
    ledger = _write_anchor(tmp_path, archive_sha=archive_sha)
    with pytest.raises(ValueError, match="prefer_grain"):
        load_per_byte_sensitivity_for_archive(
            archive_sha, path=ledger, prefer_grain="invalid_grain"
        )


# ─────────────────────────────────────────────────────────────────────────
# available_grains_for_archive inventory
# ─────────────────────────────────────────────────────────────────────────


def test_available_grains_inventory_empty_for_missing_archive(tmp_path: Path) -> None:
    inv = available_grains_for_archive("0" * 64, path=tmp_path / "missing.jsonl")
    assert inv == {"post_decompress": [], "raw_byte": [], "other": []}


def test_available_grains_inventory_classifies_correctly(tmp_path: Path) -> None:
    archive_sha = "1" * 64
    ledger = _write_anchor(
        tmp_path, archive_sha=archive_sha, grain=GRAIN_RAW_ARCHIVE_BYTE
    )
    _write_anchor(
        tmp_path, archive_sha=archive_sha, grain=GRAIN_POST_BROTLI_DECOMPRESS
    )
    _write_anchor(
        tmp_path, archive_sha=archive_sha, grain=GRAIN_DIAGNOSTIC
    )

    inv = available_grains_for_archive(archive_sha, path=ledger)
    assert GRAIN_RAW_ARCHIVE_BYTE in inv["raw_byte"]
    assert GRAIN_POST_BROTLI_DECOMPRESS in inv["post_decompress"]
    assert GRAIN_DIAGNOSTIC in inv["other"]


def test_available_grains_empty_archive_sha_returns_empty() -> None:
    inv = available_grains_for_archive("")
    assert inv == {"post_decompress": [], "raw_byte": [], "other": []}


# ─────────────────────────────────────────────────────────────────────────
# Cathedral consumer v1.1 grain-aware verdict
# ─────────────────────────────────────────────────────────────────────────


def test_cathedral_consumer_v1_1_advertises_probe_disambiguator_hook() -> None:
    from tac.cathedral.consumer_contract import HookNumber
    from tac.cathedral_consumers import per_byte_sensitivity_consumer as c

    assert HookNumber.PROBE_DISAMBIGUATOR in c.CONSUMER_HOOK_NUMBERS
    assert c.CONSUMER_VERSION == "1.1"


def test_cathedral_consumer_verdict_carries_grain_routing_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consumer verdict's notes block exposes grain_used + cascade_risk + reason."""
    from tac import master_gradient_per_byte_consumer as helper_mod
    from tac.cathedral_consumers import per_byte_sensitivity_consumer as c

    archive_sha = "2" * 64
    ledger = _write_anchor(
        tmp_path,
        archive_sha=archive_sha,
        grain=GRAIN_POST_BROTLI_DECOMPRESS,
        n_bytes=80,
    )

    original_load = helper_mod.load_per_byte_sensitivity_for_archive
    original_inv = helper_mod.available_grains_for_archive

    def patched_load(*args, **kwargs):
        kwargs.setdefault("path", ledger)
        return original_load(*args, **kwargs)

    def patched_inv(*args, **kwargs):
        kwargs.setdefault("path", ledger)
        return original_inv(*args, **kwargs)

    monkeypatch.setattr(
        helper_mod, "load_per_byte_sensitivity_for_archive", patched_load
    )
    monkeypatch.setattr(
        helper_mod, "available_grains_for_archive", patched_inv
    )

    verdict = c.consume_candidate({"archive_sha256": archive_sha})
    notes = verdict["notes"]["per_byte_sensitivity"]
    assert notes["grain_used"] == GRAIN_POST_BROTLI_DECOMPRESS
    assert notes["cascade_smearing_risk"] is False
    assert "post_decompress" in notes["grain_routing_reason"]
    # Per Catalog #287/#323: observability-only.
    assert verdict["predicted_delta_adjustment"] == 0.0
    assert verdict["promotable"] is False


def test_cathedral_consumer_verdict_warns_when_only_raw_byte_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When only raw-byte exists, verdict notes carry CASCADE_SMEARING_RISK warning."""
    from tac import master_gradient_per_byte_consumer as helper_mod
    from tac.cathedral_consumers import per_byte_sensitivity_consumer as c

    archive_sha = "3" * 64
    ledger = _write_anchor(
        tmp_path,
        archive_sha=archive_sha,
        grain=GRAIN_RAW_ZIP_INNER_MEMBER_PAYLOAD,
        n_bytes=80,
    )

    original_load = helper_mod.load_per_byte_sensitivity_for_archive
    original_inv = helper_mod.available_grains_for_archive
    monkeypatch.setattr(
        helper_mod,
        "load_per_byte_sensitivity_for_archive",
        lambda *a, **kw: original_load(*a, **{**kw, "path": ledger}),
    )
    monkeypatch.setattr(
        helper_mod,
        "available_grains_for_archive",
        lambda *a, **kw: original_inv(*a, **{**kw, "path": ledger}),
    )

    verdict = c.consume_candidate({"archive_sha256": archive_sha})
    notes = verdict["notes"]["per_byte_sensitivity"]
    assert notes["grain_used"] == GRAIN_RAW_ZIP_INNER_MEMBER_PAYLOAD
    assert notes["cascade_smearing_risk"] is True
    assert "CASCADE_SMEARING_RISK" in notes["grain_routing_reason"]
    # Per Catalog #287/#323: observability-only even when grain is suboptimal.
    assert verdict["predicted_delta_adjustment"] == 0.0


def test_cathedral_consumer_rationale_string_embeds_grain_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One-line rationale embeds grain_used + cascade_smearing_risk for audit."""
    from tac import master_gradient_per_byte_consumer as helper_mod
    from tac.cathedral_consumers import per_byte_sensitivity_consumer as c

    archive_sha = "4" * 64
    ledger = _write_anchor(
        tmp_path,
        archive_sha=archive_sha,
        grain=GRAIN_POST_ARITHMETIC_DECOMPRESS,
        n_bytes=80,
    )

    original_load = helper_mod.load_per_byte_sensitivity_for_archive
    original_inv = helper_mod.available_grains_for_archive
    monkeypatch.setattr(
        helper_mod,
        "load_per_byte_sensitivity_for_archive",
        lambda *a, **kw: original_load(*a, **{**kw, "path": ledger}),
    )
    monkeypatch.setattr(
        helper_mod,
        "available_grains_for_archive",
        lambda *a, **kw: original_inv(*a, **{**kw, "path": ledger}),
    )

    verdict = c.consume_candidate({"archive_sha256": archive_sha})
    assert "grain_used=" in verdict["rationale"]
    assert "cascade_smearing_risk=" in verdict["rationale"]


def test_cathedral_consumer_grain_inventory_in_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """grain_inventory dict surfaces post_decompress / raw_byte / other lists."""
    from tac import master_gradient_per_byte_consumer as helper_mod
    from tac.cathedral_consumers import per_byte_sensitivity_consumer as c

    archive_sha = "5" * 64
    ledger = _write_anchor(
        tmp_path, archive_sha=archive_sha, grain=GRAIN_RAW_ARCHIVE_BYTE
    )
    _write_anchor(
        tmp_path, archive_sha=archive_sha, grain=GRAIN_POST_BROTLI_DECOMPRESS
    )

    original_load = helper_mod.load_per_byte_sensitivity_for_archive
    original_inv = helper_mod.available_grains_for_archive
    monkeypatch.setattr(
        helper_mod,
        "load_per_byte_sensitivity_for_archive",
        lambda *a, **kw: original_load(*a, **{**kw, "path": ledger}),
    )
    monkeypatch.setattr(
        helper_mod,
        "available_grains_for_archive",
        lambda *a, **kw: original_inv(*a, **{**kw, "path": ledger}),
    )

    verdict = c.consume_candidate({"archive_sha256": archive_sha})
    inv = verdict["notes"]["per_byte_sensitivity"]["grain_inventory"]
    assert GRAIN_POST_BROTLI_DECOMPRESS in inv["post_decompress"]
    assert GRAIN_RAW_ARCHIVE_BYTE in inv["raw_byte"]
