# SPDX-License-Identifier: MIT
"""Tests for the SAFE-COMPILE determinism-first mx.compile layer (#252).

Pure-logic tests (partitioner classification, manifest schema, default-OFF
byte-identity, activation resolution, known-unsafe FAILS certification) run
everywhere with no MLX. Empirical certification tests (bit-equality + speedup
on real MLX) are guarded by ``importorskip``.
"""
from __future__ import annotations

import pytest

from tac import mlx_safe_compile as sc


# --------------------------------------------------------------------------- #
# 1-6. Partitioner classification (falling-rule order)
# --------------------------------------------------------------------------- #


def test_classify_unary_elementwise_is_safe_no_fma():
    v = sc.classify_op_kinds({"tanh", "sin", "mul"})
    assert v.region_class is sc.RegionClass.SAFE_ELEMENTWISE
    assert v.compile_eligible is True
    assert v.fma_contraction_possible is False  # mul present but no add


def test_classify_mul_add_flags_fma_but_still_safe():
    v = sc.classify_op_kinds({"mul", "add"})
    assert v.region_class is sc.RegionClass.SAFE_ELEMENTWISE
    assert v.compile_eligible is True
    assert v.fma_contraction_possible is True  # FMA-contraction-eligible


def test_classify_reduction_is_unsafe_reduction():
    v = sc.classify_op_kinds({"logsumexp", "sum", "mean", "mul"})
    assert v.region_class is sc.RegionClass.UNSAFE_REDUCTION
    assert v.compile_eligible is False


def test_classify_contraction_dominates_reduction():
    # matmul present alongside a reduction -> contraction wins (first falling rule)
    v = sc.classify_op_kinds({"matmul", "sum"})
    assert v.region_class is sc.RegionClass.UNSAFE_CONTRACTION
    assert v.compile_eligible is False


def test_classify_unknown_op_is_unsafe_unknown():
    v = sc.classify_op_kinds({"tanh", "some_exotic_op"})
    assert v.region_class is sc.RegionClass.UNSAFE_UNKNOWN
    assert v.compile_eligible is False


def test_classify_empty_is_unsafe_unknown():
    v = sc.classify_op_kinds(set())
    assert v.region_class is sc.RegionClass.UNSAFE_UNKNOWN
    assert v.compile_eligible is False


# --------------------------------------------------------------------------- #
# 7-8. Verdict + certificate value objects
# --------------------------------------------------------------------------- #


def test_opkind_verdict_as_dict_roundtrips_fields():
    v = sc.classify_op_kinds({"mul", "add"})
    d = v.as_dict()
    assert d["region_class"] == "safe_elementwise"
    assert d["compile_eligible"] is True
    assert d["fma_contraction_possible"] is True


def test_certificate_cross_process_determinism_predicate():
    full = sc.RegionCertificate("r", "safe_elementwise", "CERTIFIED", True, 0.0,
                                5, 5, 1.3, 2.0, 1.5, 32)
    partial = sc.RegionCertificate("r", "safe_elementwise", "FAILED", True, 0.0,
                                   3, 5, 1.3, 2.0, 1.5, 32)
    none = sc.RegionCertificate("r", "safe_elementwise", "FAILED", True, 0.0,
                                0, 0, 1.3, 2.0, 1.5, 32)
    assert full.cross_process_deterministic is True
    assert partial.cross_process_deterministic is False
    assert none.cross_process_deterministic is False


# --------------------------------------------------------------------------- #
# 9-10. Manifest schema + persistence
# --------------------------------------------------------------------------- #


def test_manifest_add_and_certified_ids():
    man = sc.CertificationManifest(device="cpu")
    man.add(sc.RegionCertificate("good", "safe_elementwise", "CERTIFIED", True, 0.0,
                                 5, 5, 1.4, 2.0, 1.4, 32))
    man.add(sc.RegionCertificate("bad", "safe_elementwise", "FAILED", False, 4.8e-3,
                                 5, 5, 0.9, 2.0, 2.2, 32))
    assert man.certified_ids() == ["good"]
    assert man.is_certified("good") is True
    assert man.is_certified("bad") is False
    assert man.is_certified("missing") is False


def test_manifest_save_load_roundtrip(tmp_path):
    man = sc.CertificationManifest(device="gpu", created_utc="2026-07-08T00:00:00Z")
    man.add(sc.RegionCertificate(
        "hosc", "safe_elementwise", "CERTIFIED", True, 0.0, 5, 5, 1.41, 0.19, 0.13, 32,
        input_shapes=["(256, 96)"], fma_contraction_possible=False, input_source="synthetic"))
    man.add(sc.RegionCertificate(
        "film", "safe_elementwise", "FAILED", False, 4.8e-3, 5, 5, 1.0, 0.2, 0.2, 32,
        fma_contraction_possible=True,
        first_divergence={"input_index": 0, "output_leaf": 0, "abs_delta": 4.8e-3}))
    p = str(tmp_path / "manifest.json")
    man.save(p)
    loaded = sc.CertificationManifest.load(p)
    assert loaded.schema_version == sc.SCHEMA_VERSION
    assert loaded.certified_ids() == ["hosc"]
    h = loaded.rows["hosc"]
    assert h.bit_equal is True and h.determinism_n_pass == 5 and h.determinism_n == 5
    assert h.speedup == pytest.approx(1.41)
    f = loaded.rows["film"]
    assert f.verdict == "FAILED" and f.first_divergence["abs_delta"] == pytest.approx(4.8e-3)


def test_manifest_as_dict_has_schema_version_and_certified_list():
    man = sc.CertificationManifest()
    man.add(sc.RegionCertificate("x", "safe_elementwise", "CERTIFIED", True, 0.0,
                                 5, 5, 1.0, 1.0, 1.0, 32))
    d = man.as_dict()
    assert d["schema_version"] == sc.SCHEMA_VERSION
    assert d["certified_ids"] == ["x"]
    assert "x" in d["rows"]


# --------------------------------------------------------------------------- #
# 11-13. Activation API — default-OFF byte-identity + fail-closed resolution
# --------------------------------------------------------------------------- #


def test_safe_compile_default_off_returns_fn_unchanged():
    def f(x):
        return x

    man = sc.CertificationManifest()
    man.add(sc.RegionCertificate("r", "safe_elementwise", "CERTIFIED", True, 0.0,
                                 5, 5, 1.0, 1.0, 1.0, 32))
    # enabled=False -> byte-identical (fn itself), even though certified
    assert sc.safe_compile(f, region_id="r", manifest=man, enabled=False) is f


def test_safe_compile_enabled_but_uncertified_is_fail_closed():
    def f(x):
        return x

    man = sc.CertificationManifest()  # r NOT certified
    assert sc.safe_compile(f, region_id="r", manifest=man, enabled=True) is f
    assert sc.safe_compile(f, region_id="r", manifest=None, enabled=True) is f


def test_resolve_enabled_regions_variants():
    man = sc.CertificationManifest()
    man.add(sc.RegionCertificate("a", "safe_elementwise", "CERTIFIED", True, 0.0,
                                 5, 5, 1.0, 1.0, 1.0, 32))
    man.add(sc.RegionCertificate("b", "safe_elementwise", "FAILED", False, 1e-3,
                                 5, 5, 1.0, 1.0, 1.0, 32))
    assert sc.resolve_enabled_regions(None, man) == frozenset()
    assert sc.resolve_enabled_regions("none", man) == frozenset()
    assert sc.resolve_enabled_regions("off", man) == frozenset()
    assert sc.resolve_enabled_regions("all-certified", man) == frozenset({"a"})
    # explicit list intersected with certified: 'a' in, 'b' (failed) out, 'c' (absent) out
    assert sc.resolve_enabled_regions("a,b,c", man) == frozenset({"a"})


def test_maybe_safe_compile_not_enabled_returns_fn():
    def f(x):
        return x

    assert sc.maybe_safe_compile(f, region_id="a", enabled_regions=frozenset()) is f
    assert sc.maybe_safe_compile(f, region_id="a", enabled_regions=frozenset({"b"})) is f


# --------------------------------------------------------------------------- #
# 14-15. Registry + canonical regions
# --------------------------------------------------------------------------- #


def test_register_and_get_region():
    reg = sc.SafeCompileRegion("unit_test_region", lambda: (None, None),
                               frozenset({"tanh"}), notes="unit")
    sc.register_region(reg)
    assert sc.get_region("unit_test_region") is reg
    with pytest.raises(KeyError):
        sc.get_region("definitely_not_a_region")


def test_canonical_regions_registered():
    for rid in ("hosc_activation", "sigmoid_scale", "film_modulate", "ce_reduction"):
        assert rid in sc.REGION_BUILDERS
    # the reduction region classifies UNSAFE_REDUCTION; the elementwise ones SAFE
    assert sc.get_region("ce_reduction").classify().region_class is sc.RegionClass.UNSAFE_REDUCTION
    assert sc.get_region("hosc_activation").classify().compile_eligible is True


# --------------------------------------------------------------------------- #
# 16. Known-unsafe region correctly FAILS certification (no GPU needed —
#     the classifier gate refuses a reduction before any compile is attempted).
# --------------------------------------------------------------------------- #


def test_certify_region_on_reduction_fails_not_compile_eligible():
    cert = sc.certify_region("ce_reduction", cross_process=False)
    assert cert.verdict == "FAILED"
    assert cert.region_class == "unsafe_reduction"
    assert "not compile-eligible" in cert.reason
    # never enabled: the manifest would never activate it under mx.compile
    man = sc.CertificationManifest().add(cert)
    assert man.is_certified("ce_reduction") is False


# --------------------------------------------------------------------------- #
# 17-19. Empirical certification (MLX-gated)
# --------------------------------------------------------------------------- #


def test_certify_hosc_activation_bit_equal_in_process():
    pytest.importorskip("mlx.core")
    cert = sc.certify_region("hosc_activation", n_inputs=8, cross_process=False, reps=5)
    assert cert.verdict == "CERTIFIED"
    assert cert.bit_equal is True
    assert cert.max_abs_delta == 0.0
    assert cert.input_coverage == 8
    assert cert.uncompiled_ms > 0.0 and cert.compiled_ms > 0.0


def test_certify_region_bit_equality_detects_divergence_if_any():
    # A region whose compiled form is bit-equal must report max_abs_delta == 0;
    # if a future GPU contracts an FMA, the harness records the first divergence.
    pytest.importorskip("mlx.core")
    cert = sc.certify_region("film_modulate", n_inputs=8, cross_process=False, reps=5)
    assert cert.verdict in {"CERTIFIED", "FAILED"}
    if cert.verdict == "FAILED":
        assert cert.first_divergence is not None
        assert cert.max_abs_delta > 0.0
    else:
        assert cert.bit_equal is True and cert.max_abs_delta == 0.0


def test_fixed_order_reduce_matches_native_sum_shape_and_close():
    mx = pytest.importorskip("mlx.core")
    import numpy as np

    x = mx.array(np.random.default_rng(0).standard_normal((32, 17)).astype(np.float32))
    fixed = sc.fixed_order_reduce_mlx(x, axis=-1)
    native = mx.sum(x, axis=-1)
    mx.eval(fixed, native)
    assert tuple(fixed.shape) == tuple(native.shape) == (32,)
    # fixed-order left-fold vs native reduction agree to fp32 tolerance
    assert float(mx.max(mx.abs(fixed - native))) < 1e-4


def test_certify_fixed_point_reduction_skips_compile():
    # certify_fixed_point_reduction never runs mx.compile; without cross-process it
    # returns a CERTIFIED determinism-only certificate for the reduction region.
    cert = sc.certify_fixed_point_reduction("ce_reduction", cross_process=False)
    assert cert.fixed_point_routed is True
    assert cert.reference == "fixed_order_reduce_mlx"
    assert cert.verdict == "CERTIFIED"
