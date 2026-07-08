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


# =========================================================================== #
# v2 — AUTO-DISCOVERY (AST + trace) partitioner
# =========================================================================== #


def test_ast_op_kinds_detects_matmul_operator():
    # the '@' operator dispatches to array.__matmul__ — invisible to a runtime
    # mx-namespace trace; ONLY the AST scan catches it (the safety net).
    def f(a, b):
        return a @ b

    ops = sc.ast_op_kinds(f)
    assert "matmul" in ops
    assert sc.classify_op_kinds(ops).region_class is sc.RegionClass.UNSAFE_CONTRACTION


def test_ast_op_kinds_detects_mul_add_operators():
    def f(x, s, b):
        return x * s + b

    ops = sc.ast_op_kinds(f)
    assert "mul" in ops and "add" in ops
    assert sc.classify_op_kinds(ops).fma_contraction_possible is True


def test_ast_op_kinds_detects_named_mx_calls_and_reduction():
    def f(x):
        import mlx.core as mx

        return mx.sum(mx.tanh(x))

    ops = sc.ast_op_kinds(f)
    assert "tanh" in ops and "sum" in ops
    assert sc.classify_op_kinds(ops).region_class is sc.RegionClass.UNSAFE_REDUCTION


def test_ast_op_kinds_linear_construction_flags_matmul():
    # an nn.Linear(...) / mx.matmul(...) call NAME is flagged a contraction. (A call
    # through a variable bound to a Module — layer(h) — is NOT name-detectable; that
    # honest limitation is why the certified candidate builders are pure math, and
    # MIXED render fns are represented by their elementwise sub-stages, not scanned.)
    def f(x, w):
        import mlx.core as mx
        from mlx import nn

        lin = nn.Linear(4, 4)
        return mx.matmul(lin(x), w)

    ops = sc.ast_op_kinds(f)
    assert "matmul" in ops  # both mx.matmul and nn.Linear construction flag contraction
    assert sc.classify_op_kinds(ops).region_class is sc.RegionClass.UNSAFE_CONTRACTION


def test_ast_op_kinds_bad_source_returns_empty():
    # a builtin has no python source -> empty set, never raises.
    assert sc.ast_op_kinds(len) == frozenset()


def test_structural_ops_classify_safe():
    # concatenate/reshape/broadcast are layout ops (no arithmetic) -> SAFE.
    v = sc.classify_op_kinds({"mul", "add", "concatenate", "reshape"})
    assert v.region_class is sc.RegionClass.SAFE_ELEMENTWISE
    assert v.compile_eligible is True


def test_take_gather_stays_unknown_vjp_hazard():
    # 'take' (gather) is deliberately EXCLUDED (its VJP is the #348 scatter hazard).
    v = sc.classify_op_kinds({"mul", "take"})
    assert v.region_class is sc.RegionClass.UNSAFE_UNKNOWN


def test_discover_candidate_row_shape_and_union():
    row = sc.discover_candidate("hosc_activation", trace=False)
    assert row.region_id == "hosc_activation"
    assert row.compile_eligible is True
    # declared op-kinds feed the union even without a trace
    assert {"tanh", "sin"} <= row.union_ops
    d = row.as_dict()
    assert d["region_class"] == "safe_elementwise"
    assert d["compile_eligible"] is True


def test_discover_candidates_classifies_reduction_and_elementwise():
    rows = {r.region_id: r for r in sc.discover_candidates(trace=False)}
    assert rows["ce_reduction"].compile_eligible is False
    assert rows["ce_reduction"].verdict.region_class is sc.RegionClass.UNSAFE_REDUCTION
    for rid in ("hosc_activation", "siren_activation", "wire_activation",
                "film_affine_tail", "compose_rgb_tail", "achromatic_luma"):
        assert rows[rid].compile_eligible is True


def test_trace_named_ops_records_mx_calls():
    pytest.importorskip("mlx.core")
    seen = sc.trace_named_ops(sc.get_region("hosc_activation").build, device="cpu")
    assert "tanh" in seen and "sin" in seen


# =========================================================================== #
# v2 — PER-CHIP TRUST CLOSURE (fingerprint)
# =========================================================================== #


def test_host_fingerprint_has_three_keys():
    fp = sc.host_fingerprint()
    assert set(fp) == {"chip", "macos_build", "mlx_version"}


def test_fingerprint_matches_empty_and_mismatch():
    host = {"chip": "Apple M5 Max", "macos_build": "25E246", "mlx_version": "0.31.2"}
    assert sc.fingerprint_matches({}, host) is True          # empty recorded -> match
    assert sc.fingerprint_matches(None, host) is True
    assert sc.fingerprint_matches(host, host) is True
    stale = {**host, "chip": "Apple M3 Max"}
    assert sc.fingerprint_matches(stale, host) is False       # chip mismatch -> stale
    # a host field the machine cannot read ("") does NOT veto
    assert sc.fingerprint_matches(host, {**host, "chip": ""}) is True


def test_manifest_fingerprint_ok_and_stale():
    host = {"chip": "Apple M5 Max", "macos_build": "25E246", "mlx_version": "0.31.2"}
    fresh = sc.CertificationManifest(fingerprint=host)
    ok, _ = sc.manifest_fingerprint_ok(fresh, host)
    assert ok is True
    stale = sc.CertificationManifest(fingerprint={**host, "chip": "Apple M1"})
    ok2, reason = sc.manifest_fingerprint_ok(stale, host)
    assert ok2 is False and "STALE" in reason
    # legacy manifest (no fingerprint) is OK by construction
    ok3, _ = sc.manifest_fingerprint_ok(sc.CertificationManifest(), host)
    assert ok3 is True


def test_resolve_enabled_regions_refuses_stale_fingerprint():
    host = {"chip": "Apple M5 Max", "macos_build": "25E246", "mlx_version": "0.31.2"}
    man = sc.CertificationManifest(fingerprint={**host, "chip": "Apple M1 Pro"})
    man.add(sc.RegionCertificate("a", "safe_elementwise", "CERTIFIED", True, 0.0,
                                 5, 5, 1.0, 1.0, 1.0, 32))
    # stale fingerprint -> fail-closed even though 'a' is certified
    assert sc.resolve_enabled_regions("all-certified", man, host=host) == frozenset()
    # fresh fingerprint -> 'a' enabled
    fresh = sc.CertificationManifest(fingerprint=host)
    fresh.add(sc.RegionCertificate("a", "safe_elementwise", "CERTIFIED", True, 0.0,
                                   5, 5, 1.0, 1.0, 1.0, 32))
    assert sc.resolve_enabled_regions("all-certified", fresh, host=host) == frozenset({"a"})
    # enforce_fingerprint=False bypasses the check
    assert sc.resolve_enabled_regions("all-certified", man, enforce_fingerprint=False,
                                      host=host) == frozenset({"a"})


def test_resolve_enabled_regions_refuses_device_mismatch():
    host = sc.host_fingerprint()
    man = sc.CertificationManifest(device="cpu", fingerprint=host)  # CPU-measured cert
    man.add(sc.RegionCertificate("a", "safe_elementwise", "CERTIFIED", True, 0.0,
                                 5, 5, 1.0, 1.0, 1.0, 32))
    # a GPU run must NOT activate a CPU-measured certificate (fp-contraction is device-specific)
    assert sc.resolve_enabled_regions("all-certified", man, host=host, run_device="gpu") == frozenset()
    # same device -> allowed
    assert sc.resolve_enabled_regions("all-certified", man, host=host, run_device="cpu") == frozenset({"a"})
    # run_device unspecified -> no device check (back-compat)
    assert sc.resolve_enabled_regions("all-certified", man, host=host) == frozenset({"a"})


def test_manifest_fingerprint_save_load_roundtrip(tmp_path):
    host = {"chip": "Apple M5 Max", "macos_build": "25E246", "mlx_version": "0.31.2"}
    man = sc.CertificationManifest(device="gpu", fingerprint=host)
    man.add(sc.RegionCertificate("hosc_activation", "safe_elementwise", "CERTIFIED",
                                 True, 0.0, 5, 5, 1.41, 0.19, 0.13, 32))
    p = str(tmp_path / "m.json")
    man.save(p)
    loaded = sc.CertificationManifest.load(p)
    assert loaded.fingerprint == host
    assert loaded.schema_version == sc.SCHEMA_VERSION
    # as_dict carries the kernel-candidate list
    assert "kernel_candidates" in loaded.as_dict()


# =========================================================================== #
# v2 — FAILED-CERT -> KERNEL-CANDIDATE pipeline (D16 feed; dormant at 0 failures)
# =========================================================================== #


def test_emit_kernel_candidates_empty_when_no_failures():
    man = sc.CertificationManifest()
    man.add(sc.RegionCertificate("a", "safe_elementwise", "CERTIFIED", True, 0.0,
                                 5, 5, 1.0, 1.0, 1.0, 32))
    assert sc.emit_kernel_candidates(man) == []


def test_emit_kernel_candidates_ranks_fp_contraction_first():
    man = sc.CertificationManifest()
    # a genuine fp-contraction failure (bit_equal False, finite delta, elementwise)
    man.add(sc.RegionCertificate("film", "safe_elementwise", "FAILED", False, 4.8e-3,
                                 5, 5, 1.6, 2.0, 1.25, 32, fma_contraction_possible=True,
                                 first_divergence={"abs_delta": 4.8e-3}))
    # a harness error (lower rank)
    man.add(sc.RegionCertificate("broke", "safe_elementwise", "ERROR", False,
                                 float("nan"), 0, 0, 0.0, 0.0, 0.0, 0))
    cands = sc.emit_kernel_candidates(man)
    assert len(cands) == 2
    assert cands[0]["region_id"] == "film"
    assert cands[0]["candidate_kind"] == "fp_contraction"
    assert cands[0]["rank"] == 0
    assert cands[1]["candidate_kind"] == "harness_error"


# =========================================================================== #
# v2 — HOT-LOOP WIRING (install_safe_compiled_regions + flag-flip byte-identity)
# =========================================================================== #


class _FakeModel:
    """Minimal stand-in for the witness model's activation contract: the plain
    path is python-float scalars; the compiled slot (when installed) is used."""

    def __init__(self):
        self.activation = "hosc"
        self.hosc_beta = 2.5
        self.hosc_omega = 1.0
        self._compiled_act = None
        self.compiled_calls = 0

    def act(self, u):
        import mlx.core as mx

        if self.activation == "hosc":
            ca = self._compiled_act
            if ca is not None:
                self.compiled_calls += 1
                return ca(u, mx.array(self.hosc_beta, dtype=u.dtype),
                          mx.array(self.hosc_omega, dtype=u.dtype))
            return mx.tanh(self.hosc_beta * mx.sin(self.hosc_omega * u))
        return u


def test_install_safe_compiled_regions_empty_is_noop():
    m = _FakeModel()
    installed = sc.install_safe_compiled_regions(m, frozenset())
    assert installed == []
    assert m._compiled_act is None  # byte-identical default path preserved


def test_install_safe_compiled_regions_uncertified_fail_closed():
    m = _FakeModel()
    man = sc.CertificationManifest()  # hosc NOT certified
    installed = sc.install_safe_compiled_regions(m, frozenset({"hosc_activation"}), man)
    assert installed == []
    assert m._compiled_act is None  # nothing installed -> plain path


def test_install_safe_compiled_regions_installs_when_certified():
    m = _FakeModel()
    man = sc.CertificationManifest(fingerprint=sc.host_fingerprint())
    man.add(sc.RegionCertificate("hosc_activation", "safe_elementwise", "CERTIFIED",
                                 True, 0.0, 5, 5, 1.41, 0.19, 0.13, 32))
    installed = sc.install_safe_compiled_regions(m, frozenset({"hosc_activation"}), man)
    assert installed == ["hosc_activation"]
    assert m._compiled_act is not None  # compiled slot set -> flag-flip complete


def test_flag_flip_off_byte_identical_on_uses_compiled_path():
    mx = pytest.importorskip("mlx.core")
    import numpy as np

    u = mx.array(np.random.default_rng(0).standard_normal((32, 8)).astype(np.float32))
    # OFF: plain path, spy counter stays 0
    m_off = _FakeModel()
    y_off = m_off.act(u)
    mx.eval(y_off)
    assert m_off.compiled_calls == 0

    # ON: install certified compiled hook -> compiled path used (spy fires) AND
    # the result is bit-identical to the plain path (max|Δ| == 0).
    m_on = _FakeModel()
    man = sc.CertificationManifest(fingerprint=sc.host_fingerprint())
    man.add(sc.RegionCertificate("hosc_activation", "safe_elementwise", "CERTIFIED",
                                 True, 0.0, 5, 5, 1.41, 0.19, 0.13, 32))
    sc.install_safe_compiled_regions(m_on, frozenset({"hosc_activation"}), man)
    y_on = m_on.act(u)
    mx.eval(y_on)
    assert m_on.compiled_calls == 1
    assert float(mx.max(mx.abs(y_off - y_on))) == 0.0  # certified bit-identity holds when wired


# =========================================================================== #
# v2 — reduction-site inventory
# =========================================================================== #


def test_step_reduction_sites_all_native_or_fused():
    sites = {s["site"]: s for s in sc.STEP_REDUCTION_SITES}
    assert "ce_loss" in sites and "R_operator_reductions" in sites
    for s in sc.STEP_REDUCTION_SITES:
        assert s["routing"] in {"native", "native_fused_r"}
        assert s["reason"]  # every site documents WHY it is (un)routed
