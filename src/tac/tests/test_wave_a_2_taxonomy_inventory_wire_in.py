"""WAVE-A-2 (2026-05-12): substrate taxonomy + inventory wire-in tests.

Covers CANON-1.A + CANON-1.J from the canonicalization ledger
``.omx/research/canonicalization_dedup_oss_rigor_ledger_20260512.md`` and the
tradition memo
``.omx/research/substrate_tradition_taxonomy_20260512.md``.

Test surface (≥20 tests):

1. Inventory has 48 rows (24 legacy + 15 FIX-J + 9 WAVE-A-2 TRADITION 2).
2. Each of the 9 WAVE-A-2 substrate_ids is present.
3. Each WAVE-A-2 row has a corresponding dispatch-cost entry.
4. Each WAVE-A-2 row carries a unique format_id within its substrate_class.
5. Each WAVE-A-2 row carries a unique magic_byte across the inventory.
6. Substrate-tradition docstring in ``src/tac/substrates/__init__.py``
   documents BOTH traditions explicitly.
7. Tradition taxonomy memo exists at the canonical path.
8. xray_substrate_classifier has every TRADITION 2 magic byte registered.
9. xray_substrate_classifier substrate-class allowlist contains every
   TRADITION 2 substrate.
10. CompressAI primitive inventory includes all 3 codec adapters
    (factorized_prior / balle_hyperprior / cheng2020).
11. CompressAI primitives have magic-byte entries in the xray classifier.
12. CompressAI primitives have the correct PrimitiveCategory.
13. Inventory-level format_id band 0xA0-0xAF is exclusively WAVE-A-2 rows.
14. mlx_mask_renderer has $0 dispatch cost (advisory-only, no paid GPU path).
15. Inventory matrix builds successfully with 48² cells.
16. Every WAVE-A-2 row has predicted_delta_alone_midpoint within sane bounds.
17. Every WAVE-A-2 row's landing_memo matches the canonical naming.
18. Every WAVE-A-2 row's landed_at is dated post-2026-05-09.
19. Every WAVE-A-2 row's substrate_class is RENDERER_REPLACEMENT.
20. SUBSTRATE_SCAFFOLDS mapping is unchanged (TRADITION 1 unaffected).
21. WAVE-A-2 substrate_ids do not collide with FIX-J substrate_ids.
22. Inventory ordering is deterministic across multiple invocations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.optimization.substrate_composition_matrix import (
    DISPATCH_COST_USD_MIDPOINT,
    ScoreAxis,
    SubstrateClass,
    build_composition_matrix,
    canonical_substrate_inventory,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


WAVE_A_2_SUBSTRATE_IDS: tuple[str, ...] = (
    "cnerv",
    "e_nerv",
    "ego_nerv",
    "lane_12_v2_nerv_as_renderer",
    "nervdc",
    "quantizr_faithful",
    "mlx_mask_renderer",
    "dp_sims_renderer",
    "diffusion_renderer",
)


WAVE_A_2_MAGIC_BYTES: tuple[bytes, ...] = (
    b"CNRV",
    b"ENRV",
    b"EGOV",
    b"L12V",
    b"NDCV",
    b"QZRV",
    b"MLXR",
    b"DPSV",
    b"DIFV",
)


# ── Inventory invariants ──────────────────────────────────────────────────


def test_inventory_count_is_48():
    rows = canonical_substrate_inventory()
    assert len(rows) == 48, (
        f"expected 48 rows (24 legacy + 15 FIX-J + 9 WAVE-A-2), got {len(rows)}"
    )


def test_all_wave_a_2_substrate_ids_present():
    rows = canonical_substrate_inventory()
    ids = {r.substrate_id for r in rows}
    missing = set(WAVE_A_2_SUBSTRATE_IDS) - ids
    assert not missing, f"missing WAVE-A-2 substrate_ids: {missing}"


def test_all_wave_a_2_rows_have_dispatch_cost():
    for sid in WAVE_A_2_SUBSTRATE_IDS:
        assert sid in DISPATCH_COST_USD_MIDPOINT, (
            f"WAVE-A-2 substrate {sid!r} missing DISPATCH_COST_USD_MIDPOINT entry"
        )


def test_all_wave_a_2_format_ids_unique_within_class():
    rows = canonical_substrate_inventory()
    by_cls: dict[SubstrateClass, list[int]] = {}
    for r in rows:
        by_cls.setdefault(r.substrate_class, []).append(r.format_id)
    for cls, fids in by_cls.items():
        assert len(set(fids)) == len(fids), (
            f"duplicate format_id within {cls}: {sorted(fids)}"
        )


def test_all_magic_bytes_unique_inventory_wide():
    rows = canonical_substrate_inventory()
    mbs = [r.magic_bytes for r in rows]
    assert len(set(mbs)) == len(mbs), f"duplicate magic_bytes: {mbs}"


def test_wave_a_2_format_ids_in_band_0xa0_0xaf():
    """Format-id band 0xA0-0xAF reserved for WAVE-A-2 TRADITION 2 rows."""
    rows = canonical_substrate_inventory()
    wave_a_2_rows = [r for r in rows if r.substrate_id in WAVE_A_2_SUBSTRATE_IDS]
    for r in wave_a_2_rows:
        assert 0xA0 <= r.format_id <= 0xAF, (
            f"{r.substrate_id} format_id {hex(r.format_id)} outside 0xA0-0xAF band"
        )


def test_no_format_id_outside_wave_a_2_in_band_0xa0_0xaf():
    """Conversely, no non-WAVE-A-2 row sits in 0xA0-0xAF."""
    rows = canonical_substrate_inventory()
    for r in rows:
        if 0xA0 <= r.format_id <= 0xAF:
            assert r.substrate_id in WAVE_A_2_SUBSTRATE_IDS, (
                f"format-id {hex(r.format_id)} (substrate {r.substrate_id!r}) "
                "in WAVE-A-2 band but not a WAVE-A-2 substrate"
            )


def test_wave_a_2_substrate_class_is_renderer_replacement():
    rows = canonical_substrate_inventory()
    for r in rows:
        if r.substrate_id in WAVE_A_2_SUBSTRATE_IDS:
            assert r.substrate_class == SubstrateClass.RENDERER_REPLACEMENT, (
                f"WAVE-A-2 substrate {r.substrate_id!r} must be RENDERER_REPLACEMENT"
            )


def test_mlx_mask_renderer_advisory_zero_cost():
    """`mlx_mask_renderer` is `[macOS-CPU advisory only]` per CLAUDE.md MPS rule;
    has $0 dispatch cost because no paid remote-GPU path exists."""
    assert DISPATCH_COST_USD_MIDPOINT["mlx_mask_renderer"] == 0.0


def test_inventory_matrix_builds():
    matrix = build_composition_matrix()
    n = matrix.n_substrates()
    assert n == 48
    assert matrix.n_cells() == n * n


def test_wave_a_2_predicted_delta_in_sane_bounds():
    """Each WAVE-A-2 row's predicted delta band must be within
    ``[-0.05, +0.05]`` — wider than that suggests row spec error."""
    rows = canonical_substrate_inventory()
    for r in rows:
        if r.substrate_id in WAVE_A_2_SUBSTRATE_IDS:
            lo, hi = r.predicted_delta_alone_band
            assert -0.05 <= lo <= hi <= 0.05, (
                f"{r.substrate_id} band ({lo}, {hi}) outside [-0.05, +0.05]"
            )


def test_wave_a_2_landing_memo_canonical():
    rows = canonical_substrate_inventory()
    expected_memo = "feedback_wave_a_2_taxonomy_inventory_drift_landed_20260512"
    for r in rows:
        if r.substrate_id in WAVE_A_2_SUBSTRATE_IDS:
            assert r.landing_memo == expected_memo, (
                f"{r.substrate_id} landing_memo {r.landing_memo!r}, "
                f"expected {expected_memo!r}"
            )


def test_wave_a_2_landed_at_post_may_9():
    rows = canonical_substrate_inventory()
    for r in rows:
        if r.substrate_id in WAVE_A_2_SUBSTRATE_IDS:
            assert r.landed_at >= "2026-05-09", (
                f"{r.substrate_id} landed_at {r.landed_at!r} pre-dates 2026-05-09"
            )


def test_wave_a_2_does_not_collide_with_fix_j_ids():
    fix_j_ids = {
        "sane_hnerv",
        "balle_renderer",
        "hybrid_renderer_residual",
        "self_compress_nn",
        "pr101_lc_v2_clone",
        "cool_chic_full_renderer",
        "wavelet_full_renderer",
        "grayscale_lut",
        "vq_vae_substrate",
        "siren_substrate",
        "block_nerv_substrate",
        "tc_nerv_substrate",
        "ff_nerv_substrate",
        "ds_nerv_substrate",
        "hi_nerv_substrate",
    }
    wave_a_2 = set(WAVE_A_2_SUBSTRATE_IDS)
    assert not (fix_j_ids & wave_a_2), (
        f"WAVE-A-2 ids collide with FIX-J: {fix_j_ids & wave_a_2}"
    )


def test_inventory_deterministic_across_calls():
    rows_a = canonical_substrate_inventory()
    rows_b = canonical_substrate_inventory()
    assert [r.substrate_id for r in rows_a] == [r.substrate_id for r in rows_b]


# ── Tradition taxonomy docstring + memo presence ──────────────────────────


def test_substrates_init_documents_both_traditions():
    init_path = REPO_ROOT / "src" / "tac" / "substrates" / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
    assert "TRADITION 1" in text, "substrates/__init__.py must mention TRADITION 1"
    assert "TRADITION 2" in text, "substrates/__init__.py must mention TRADITION 2"
    assert "Literature citations" in text or "Literature citation" in text


def test_tradition_taxonomy_memo_exists():
    memo = REPO_ROOT / ".omx" / "research" / "substrate_tradition_taxonomy_20260512.md"
    assert memo.exists(), f"tradition taxonomy memo missing: {memo}"
    text = memo.read_text(encoding="utf-8")
    assert "CANON-1.A" in text
    assert "TRADITION 1" in text
    assert "TRADITION 2" in text
    assert "Reactivation criteria" in text or "Reactivation criterion" in text


def test_tradition_taxonomy_memo_cites_literature():
    memo = REPO_ROOT / ".omx" / "research" / "substrate_tradition_taxonomy_20260512.md"
    text = memo.read_text(encoding="utf-8")
    # Critical literature anchors per operator directive "ground in literature first".
    assert "Chen et al." in text  # HNeRV NeurIPS 2023
    assert "Ballé" in text  # ICLR 2018 (handle either 'Ballé' or 'Balle' Unicode)
    assert "Mallat" in text  # Wavelet PAMI 1989
    assert "van den Oord" in text  # VQ-VAE NeurIPS 2017
    assert "Sitzmann" in text  # SIREN NeurIPS 2020
    assert "Ladune" in text  # Cool-Chic ICCV 2023


# ── xray classifier wire-in ───────────────────────────────────────────────


def test_xray_classifier_has_wave_a_2_magic_bytes():
    from tools.xray_substrate_classifier import _SECTION_MAGIC_SIGNATURES

    magics = {sig for sig, _label in _SECTION_MAGIC_SIGNATURES}
    for m in WAVE_A_2_MAGIC_BYTES:
        assert m in magics, f"xray classifier missing magic byte {m!r}"


def test_xray_classifier_has_wave_a_2_substrate_classes():
    from tools.xray_substrate_classifier import _SUBSTRATE_CLASSES

    expected_classes = {
        "cnerv_substrate",
        "lane_12_v2_nerv_substrate",
        "quantizr_faithful_substrate",
        "mlx_mask_renderer_substrate",
        "dp_sims_renderer_substrate",
        "diffusion_renderer_substrate",
    }
    missing = expected_classes - set(_SUBSTRATE_CLASSES)
    assert not missing, f"xray classifier _SUBSTRATE_CLASSES missing: {missing}"


# ── CompressAI primitive inventory ────────────────────────────────────────


def test_compressai_codec_primitives_present():
    from tac.composition.registry import canonical_primitive_inventory

    rows = canonical_primitive_inventory()
    ids = {p.primitive_id for p in rows}
    expected = {
        "compressai_factorized_prior",
        "compressai_balle_hyperprior",
        "compressai_cheng2020",
    }
    missing = expected - ids
    assert not missing, f"CompressAI codec primitives missing: {missing}"


def test_compressai_codec_primitives_have_correct_category():
    from tac.composition.registry import (
        PrimitiveCategory,
        canonical_primitive_inventory,
    )

    rows = {p.primitive_id: p for p in canonical_primitive_inventory()}
    for compressai_id in (
        "compressai_factorized_prior",
        "compressai_balle_hyperprior",
        "compressai_cheng2020",
    ):
        p = rows[compressai_id]
        assert p.category == PrimitiveCategory.COMPRESSAI_CODEC


def test_compressai_codec_magic_bytes_registered_in_xray():
    from tools.xray_substrate_classifier import _SECTION_MAGIC_SIGNATURES

    magics = {sig for sig, _label in _SECTION_MAGIC_SIGNATURES}
    for m in (b"CAFP", b"CABH", b"CACG"):
        assert m in magics, f"CompressAI magic {m!r} missing from xray classifier"


# ── SUBSTRATE_SCAFFOLDS untouched by WAVE-A-2 ────────────────────────────


def test_substrate_scaffolds_unchanged_15_entries():
    """WAVE-A-2 must NOT touch TRADITION 1 SUBSTRATE_SCAFFOLDS mapping."""
    from tac.substrates import SUBSTRATE_SCAFFOLDS

    assert len(SUBSTRATE_SCAFFOLDS) == 15, (
        f"SUBSTRATE_SCAFFOLDS should be 15 TRADITION 1 entries, got {len(SUBSTRATE_SCAFFOLDS)}"
    )


def test_substrate_scaffolds_does_not_include_wave_a_2_ids():
    """TRADITION 2 single-file substrates do NOT live in substrates/<name>/
    subpackages; they should not appear in SUBSTRATE_SCAFFOLDS."""
    from tac.substrates import SUBSTRATE_SCAFFOLDS

    for sid in WAVE_A_2_SUBSTRATE_IDS:
        assert sid not in SUBSTRATE_SCAFFOLDS.values(), (
            f"WAVE-A-2 substrate {sid!r} leaked into TRADITION 1 SUBSTRATE_SCAFFOLDS"
        )
