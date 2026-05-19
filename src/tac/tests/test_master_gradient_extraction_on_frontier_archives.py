# SPDX-License-Identifier: MIT
"""Frontier-archive master-gradient extraction regression guards.

[verified-against: tools/extract_master_gradient.py + src/tac/master_gradient.py canonical helpers]
[verified-against: .omx/state/master_gradient_anchors.jsonl ledger persisted via append_anchor_locked]
[verified-against: parser_extension_wave_20260519 anchors emitted for A1+PR101_lc_v2+PR106_format0d+PR107_apogee]

Per the MASTER-GRADIENT-PARSER-EXTENSION wave (task #887) + Cable D D2:
verify the producer side of the producer->consumer loop closes for the 4
highest-EV frontier archives (A1 / PR101_lc_v2 / PR106_format0d /
PR107_apogee). The slot 6 per-byte sensitivity cathedral consumer
(``src/tac/cathedral_consumers/per_byte_sensitivity_consumer/``) READS the
anchors this wave WRITES.

This file is integration-grade (touches the live ledger + npy sidecars on
disk). It is INTENTIONALLY skip-friendly: tests skip if the live archive
file is missing OR if the master-gradient ledger does not yet have the
expected anchor (so a clean clone without the parser_extension_wave_20260519
extractions does not falsely fail).

Per CLAUDE.md "Apples-to-apples evidence discipline": each anchor is
``[macOS-CPU advisory]`` axis-tagged (non-promotable proxy signal per
Catalog #1 + Catalog #192 + Catalog #317). The anchors are gradient signals
(sensitivity-map producer hook #1), NOT contest score claims. Promotion to
``[contest-CPU]`` or ``[contest-CUDA]`` requires paired Linux x86_64 +
NVIDIA T4/A100/4090 re-extraction per Catalog #317.

Per Catalog #327 raw-byte-authority discipline: this test asserts the
anchor's typed Provenance + grammar-aware projection contract, NEVER the
forbidden raw byte-flip / byte-modification authority pattern.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LEDGER_PATH = REPO_ROOT / ".omx/state/master_gradient_anchors.jsonl"

# Canonical frontier archive paths (per the parser_extension_wave_20260519
# extraction). Sister regressions in tac.frontier_scan + the inventory memo
# will surface deletions.
LIVE_A1_ARCHIVE = REPO_ROOT / "submissions" / "a1" / "archive.zip"
LIVE_PR101_LC_V2_ARCHIVE = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
    / "archive.zip"
)
LIVE_PR106_FORMAT0D_ARCHIVE = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "pr106_format0d_latent_score_table_materialized_20260515_codex"
    / "sidecar_archive.zip"
)
LIVE_PR107_APOGEE_ARCHIVE = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "public_pr_archive_release_view"
    / "public_pr107_intake_20260505_auto"
    / "archive.zip"
)

# Canonical SHAs pinned at parser_extension_wave_20260519 emission time. If a
# frontier archive's sha changes, the SHA-pin regression caught here is the
# canonical signal that a fresh re-extraction is needed (per CLAUDE.md
# "Apples-to-apples evidence discipline").
A1_FRONTIER_SHA = "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5"
PR101_LC_V2_FEC6_FRONTIER_SHA = (
    "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
)
PR106_FORMAT0D_FRONTIER_SHA = (
    "9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4"
)
PR107_APOGEE_FRONTIER_SHA = (
    "7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb"
)

# Wave call_id prefix; ANY anchor with this call_id_prefix is an artifact of
# this lane's extraction. The test scopes to the wave's anchors so a future
# re-extraction wave with a different call_id_prefix does not falsely fail.
WAVE_CALL_ID_PREFIX = "parser_extension_wave_20260519"


def _load_ledger_rows() -> list[Mapping[str, object]]:
    if not LEDGER_PATH.exists():
        return []
    rows: list[Mapping[str, object]] = []
    for line in LEDGER_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _wave_anchors_for_sha(archive_sha: str) -> list[Mapping[str, object]]:
    rows = _load_ledger_rows()
    return [
        r for r in rows
        if r.get("archive_sha256") == archive_sha
        and isinstance(r.get("measurement_call_id"), str)
        and r["measurement_call_id"].startswith(WAVE_CALL_ID_PREFIX)
    ]


def _require_wave_anchor(archive_sha: str) -> Mapping[str, object]:
    """Skip if the wave anchor is missing; otherwise return the latest row."""
    anchors = _wave_anchors_for_sha(archive_sha)
    if not anchors:
        pytest.skip(
            f"parser_extension_wave_20260519 anchor missing for archive {archive_sha[:16]}...; "
            "run tools/extract_master_gradient.py to populate"
        )
    # Latest wins on tie (anchors are append-only per Catalog #327)
    return sorted(
        anchors, key=lambda r: str(r.get("written_at_utc", ""))
    )[-1]


# --------------------------------------------------------------------------- #
# Wave-emission regression guards                                              #
# --------------------------------------------------------------------------- #


class TestWaveAnchorEmission:
    """Each frontier archive must have a parser_extension_wave_20260519 anchor."""

    def test_a1_wave_anchor_present(self):
        anchor = _require_wave_anchor(A1_FRONTIER_SHA)
        assert anchor["measurement_axis"].startswith("[")
        assert anchor["measurement_call_id"] == f"{WAVE_CALL_ID_PREFIX}_a1"

    def test_pr101_lc_v2_wave_anchor_present(self):
        anchor = _require_wave_anchor(PR101_LC_V2_FEC6_FRONTIER_SHA)
        assert anchor["measurement_axis"].startswith("[")
        assert anchor["measurement_call_id"] == f"{WAVE_CALL_ID_PREFIX}_pr101_lc_v2"

    def test_pr106_format0d_wave_anchor_present(self):
        anchor = _require_wave_anchor(PR106_FORMAT0D_FRONTIER_SHA)
        assert anchor["measurement_axis"].startswith("[")
        assert anchor["measurement_call_id"] == f"{WAVE_CALL_ID_PREFIX}_pr106_format0d"

    def test_pr107_apogee_wave_anchor_present(self):
        anchor = _require_wave_anchor(PR107_APOGEE_FRONTIER_SHA)
        assert anchor["measurement_axis"].startswith("[")
        assert anchor["measurement_call_id"] == f"{WAVE_CALL_ID_PREFIX}_pr107_apogee"


# --------------------------------------------------------------------------- #
# Per-anchor Provenance + non-promotability invariants                          #
# --------------------------------------------------------------------------- #


def _all_wave_anchors() -> Iterable[Mapping[str, object]]:
    return [
        a for a in _load_ledger_rows()
        if isinstance(a.get("measurement_call_id"), str)
        and a["measurement_call_id"].startswith(WAVE_CALL_ID_PREFIX)
    ]


class TestWaveAnchorProvenance:
    """Each wave anchor must carry typed Provenance per Catalog #323 / #327."""

    def test_every_wave_anchor_has_required_schema_fields(self):
        anchors = list(_all_wave_anchors())
        if not anchors:
            pytest.skip("no parser_extension_wave_20260519 anchors in ledger")
        required = {
            "archive_sha256",
            "operating_point",
            "gradient_array_path",
            "n_bytes",
            "measurement_method",
            "measurement_axis",
            "measurement_hardware",
            "measurement_utc",
            "schema_version",
        }
        for anchor in anchors:
            missing = required - set(anchor.keys())
            assert not missing, f"anchor {anchor.get('measurement_call_id')} missing {missing}"

    def test_every_wave_anchor_axis_is_macos_advisory_per_catalog_192(self):
        anchors = list(_all_wave_anchors())
        if not anchors:
            pytest.skip("no parser_extension_wave_20260519 anchors in ledger")
        # Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192 + Catalog #317:
        # local CPU extraction on Apple Silicon is non-promotable proxy signal.
        for anchor in anchors:
            assert anchor["measurement_axis"] == "[macOS-CPU advisory]", (
                f"anchor {anchor.get('measurement_call_id')} axis={anchor['measurement_axis']} "
                "must be [macOS-CPU advisory] (per Catalog #192); promotion to "
                "[contest-CPU] / [contest-CUDA] requires paired Linux x86_64 + NVIDIA "
                "re-extraction per Catalog #317"
            )

    def test_every_wave_anchor_hardware_marks_macos(self):
        # Per detect_hardware_substrate canonical: extraction on Darwin ARM64
        # auto-derives the hardware tag containing "macos" / "darwin".
        anchors = list(_all_wave_anchors())
        if not anchors:
            pytest.skip("no parser_extension_wave_20260519 anchors in ledger")
        for anchor in anchors:
            hw = str(anchor.get("measurement_hardware", "")).lower()
            assert "macos" in hw or "darwin" in hw, (
                f"anchor {anchor.get('measurement_call_id')} hardware={hw!r}; "
                "expected macos/darwin substrate tag for local CPU extraction"
            )


# --------------------------------------------------------------------------- #
# Per-byte sensitivity payload shape invariants (slot 6 consumer contract)    #
# --------------------------------------------------------------------------- #


class TestWaveAnchorSensitivityPayloadShape:
    """The per-byte sensitivity .npy MUST match the n_bytes claim.

    This is the canonical schema the slot 6 per-byte sensitivity consumer
    reads. A shape mismatch breaks the producer->consumer loop.
    """

    @pytest.mark.parametrize(
        "archive_sha",
        [
            A1_FRONTIER_SHA,
            PR101_LC_V2_FEC6_FRONTIER_SHA,
            PR106_FORMAT0D_FRONTIER_SHA,
            PR107_APOGEE_FRONTIER_SHA,
        ],
    )
    def test_sidecar_npy_shape_matches_n_bytes(self, archive_sha):
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        anchor = _require_wave_anchor(archive_sha)
        npy_path = Path(str(anchor["gradient_array_path"]))
        if not npy_path.is_absolute():
            npy_path = REPO_ROOT / npy_path
        if not npy_path.exists():
            pytest.skip(f"sidecar npy missing: {npy_path}")
        arr = np.load(npy_path)
        assert arr.shape == (anchor["n_bytes"], 3), (
            f"shape {arr.shape} != ({anchor['n_bytes']}, 3); slot 6 consumer "
            "contract REQUIRES (N_bytes, 3) [seg, pose, rate_bytes_delta] columns"
        )
        # Finite-value guard (no NaN / Inf entries)
        assert np.isfinite(arr).all(), (
            "per-byte sensitivity array has NaN/Inf entries; would corrupt "
            "downstream sensitivity-map + Pareto + bit-allocator hooks"
        )

    @pytest.mark.parametrize(
        "archive_sha",
        [
            A1_FRONTIER_SHA,
            PR101_LC_V2_FEC6_FRONTIER_SHA,
            PR106_FORMAT0D_FRONTIER_SHA,
            PR107_APOGEE_FRONTIER_SHA,
        ],
    )
    def test_sidecar_non_degenerate_nonzero_byte_fraction(self, archive_sha):
        """At least 50% of bytes must carry non-zero sensitivity signal.

        Empirical baseline from parser_extension_wave_20260519: each frontier
        archive shows ~90% non-zero bytes (decoder + latent regions; sidecar
        is zero-gradient v1 per the projector contract). A degenerate
        all-zero or <50% non-zero tensor signals a corrupted extraction.
        """
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        anchor = _require_wave_anchor(archive_sha)
        npy_path = Path(str(anchor["gradient_array_path"]))
        if not npy_path.is_absolute():
            npy_path = REPO_ROOT / npy_path
        if not npy_path.exists():
            pytest.skip(f"sidecar npy missing: {npy_path}")
        arr = np.load(npy_path)
        per_byte = np.abs(arr).sum(axis=1)
        nz_frac = float((per_byte > 0).sum()) / arr.shape[0]
        assert nz_frac >= 0.50, (
            f"sensitivity payload non-zero fraction {nz_frac:.3f} < 0.50; "
            "degenerate extraction signals broken decoder Jacobian projection"
        )


# --------------------------------------------------------------------------- #
# Operating-point sanity guards                                               #
# --------------------------------------------------------------------------- #


class TestWaveAnchorOperatingPoint:
    """Anchor operating point must be finite + non-degenerate.

    Per src/tac/master_gradient.py OperatingPoint __post_init__: d_pose > 0
    is required because the PoseNet marginal 5/sqrt(10*d_pose) is undefined
    at d_pose=0. The wave's 8-pair averaged extraction always produces
    d_pose > 0 on the canonical frontier archives.
    """

    @pytest.mark.parametrize(
        "archive_sha",
        [
            A1_FRONTIER_SHA,
            PR101_LC_V2_FEC6_FRONTIER_SHA,
            PR106_FORMAT0D_FRONTIER_SHA,
            PR107_APOGEE_FRONTIER_SHA,
        ],
    )
    def test_operating_point_finite_and_non_degenerate(self, archive_sha):
        anchor = _require_wave_anchor(archive_sha)
        op = anchor["operating_point"]
        assert isinstance(op, dict)
        for key in ("d_seg", "d_pose", "rate", "score"):
            assert key in op, f"operating_point missing {key}"
            assert isinstance(op[key], (int, float))
        assert op["d_seg"] >= 0
        assert op["d_pose"] > 0  # canonical non-degenerate requirement
        assert op["rate"] >= 0
        assert op["score"] > 0


# --------------------------------------------------------------------------- #
# Grammar projection contract (Catalog #327 raw-byte-authority discipline)    #
# --------------------------------------------------------------------------- #


class TestWaveAnchorMeasurementMethod:
    """Each anchor's measurement_method must name the canonical projector.

    Per CLAUDE.md "Catalog #318 raw byte authority NOT landed": only typed
    grammar-aware projectors (autograd_per_parameter_projected_*) are
    authorized. Raw bit-flip / finite-difference methods are forbidden by
    the gate at the source level (and would never produce a contest-faithful
    Jacobian against a ZIP + entropy-coded packet).
    """

    @pytest.mark.parametrize(
        "archive_sha,expected_method_pfx",
        [
            (
                A1_FRONTIER_SHA,
                "autograd_per_parameter_projected_fec6_int8_fp16_jacobian",
            ),
            (
                PR101_LC_V2_FEC6_FRONTIER_SHA,
                "autograd_per_parameter_projected_fec6_int8_fp16_jacobian",
            ),
            (
                PR106_FORMAT0D_FRONTIER_SHA,
                "autograd_per_parameter_projected_pr106_format0d_primary_packed_hnerv_decoder_jacobian_sidecar_zero_grad_v1",
            ),
            (
                PR107_APOGEE_FRONTIER_SHA,
                "autograd_per_parameter_projected_pr107_apogee_cd1_decoder_jacobian_camera_offset_roundtrip_latents_zero_grad_v1",
            ),
        ],
    )
    def test_method_names_canonical_projector(self, archive_sha, expected_method_pfx):
        anchor = _require_wave_anchor(archive_sha)
        assert anchor["measurement_method"] == expected_method_pfx, (
            f"measurement_method={anchor['measurement_method']!r} != "
            f"{expected_method_pfx!r}; this would indicate a non-canonical "
            "Jacobian extraction (Catalog #318 raw byte authority forbidden)"
        )


# --------------------------------------------------------------------------- #
# Live-archive existence guards (regression detection)                         #
# --------------------------------------------------------------------------- #


class TestLiveArchiveExistence:
    """Pin the frontier archive paths so a future move/deletion surfaces here."""

    @pytest.mark.parametrize(
        "label,path,expected_sha",
        [
            ("A1", LIVE_A1_ARCHIVE, A1_FRONTIER_SHA),
            ("PR101_lc_v2_fec6", LIVE_PR101_LC_V2_ARCHIVE, PR101_LC_V2_FEC6_FRONTIER_SHA),
            ("PR106_format0d", LIVE_PR106_FORMAT0D_ARCHIVE, PR106_FORMAT0D_FRONTIER_SHA),
            ("PR107_apogee", LIVE_PR107_APOGEE_ARCHIVE, PR107_APOGEE_FRONTIER_SHA),
        ],
    )
    def test_frontier_archive_path_and_sha_pinned(self, label, path, expected_sha):
        if not path.exists():
            pytest.skip(f"{label} frontier archive missing at {path}; re-extraction needed")
        import hashlib
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == expected_sha, (
            f"{label} archive sha changed: observed={observed[:16]}... "
            f"expected={expected_sha[:16]}...; re-run "
            f"tools/extract_master_gradient.py to refresh the wave anchor"
        )


# --------------------------------------------------------------------------- #
# Slot 6 producer->consumer contract regression guard                          #
# --------------------------------------------------------------------------- #


class TestSlot6ConsumerContractCompat:
    """The wave anchor JSONL schema must match what slot 6 consumer reads.

    Per src/tac/master_gradient_per_byte_consumer.py + slot 6's
    src/tac/cathedral_consumers/per_byte_sensitivity_consumer/__init__.py:
    the consumer reads canonical fields produced by append_anchor_locked.
    This regression guard pins the schema so a future producer-side
    refactor that drops a consumer-required field is caught here.
    """

    def test_wave_anchors_carry_slot_6_consumer_required_fields(self):
        anchors = list(_all_wave_anchors())
        if not anchors:
            pytest.skip("no parser_extension_wave_20260519 anchors in ledger")
        slot_6_required = {
            "archive_sha256",  # consumer keys per archive
            "gradient_array_path",  # consumer loads sensitivity tensor
            "n_bytes",  # consumer shape-validates tensor
            "operating_point",  # consumer derives marginal coefficients
            "measurement_axis",  # consumer respects axis labelling per Catalog #127
            "measurement_method",  # consumer routes per Jacobian flavor
        }
        for anchor in anchors:
            missing = slot_6_required - set(anchor.keys())
            assert not missing, (
                f"anchor {anchor.get('measurement_call_id')} missing slot-6-required "
                f"fields {missing}; would break per-byte-sensitivity consumer loop"
            )
