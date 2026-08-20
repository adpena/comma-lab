# SPDX-License-Identifier: MIT
"""ddm_rr8 - tests for the C free-corrector port and its stager.

WHAT EACH LAYER IS FOR, because they prove different things and only one of them is the gate:

* ``TestStagerAnchors``      the recorded rewrites still match the LIVE base tree exactly
                             once, and refuse when it drifts.  This is what stops a stale
                             transformation from silently patching text that no longer means
                             what it did.
* ``TestConfigDriftGuard``   the binding REFUSES a corrector configuration the C was not
                             compiled for.  A silent mismatch desynchronises the arithmetic
                             decoder (``ddm_rr2``, S = 27.83) and reads as a model failure.
* ``TestDifferentialParity`` the C and the shipped Python corrector agree BIT-FOR-BIT on
                             randomised-but-structurally-valid groups, tables included.

The parity test's inputs are randomised, not real, and that is deliberate: randomisation
walks the clamps, the negative gradients and the cold/warm boundaries that one real clip may
never visit.  It is NOT the port's validation -- ``experiments/ddm_rr8_corrector_parity.py``
replays the REAL captured decoder trace for that, and the full n600 identity run against the
four published anchors is the actual gate.  These tests keep the port honest between gates.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
BASE_TREE = REPO / "submissions" / "robust_current" / "jg5_sub015_runtime" / "runtime"
SOURCE_DIR = REPO / "runtime-rs" / "native" / "f26-corrector"
STAGER_PATH = REPO / "experiments" / "ddm_rr8_stage_native_corrector_runtime.py"
PLANE = 384 * 512
NUM_CLASSES = 5


def _load_stager():
    spec = importlib.util.spec_from_file_location("ddm_rr8_stager", STAGER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stager():
    return _load_stager()


# --- the recorded rewrites --------------------------------------------------------------


class TestStagerAnchors:
    def test_every_python_anchor_matches_the_live_base_exactly_once(self, stager):
        """A count of 0 means the base drifted; a count of 2 means the rewrite is ambiguous."""
        if not (BASE_TREE / stager.PY_TARGET).is_file():
            pytest.skip("jg5 base tree not present in this checkout")
        text = (BASE_TREE / stager.PY_TARGET).read_text()
        for label, old, _ in stager.PY_REWRITES:
            assert text.count(old) == 1, f"{label}: anchor count {text.count(old)} != 1"

    def test_shell_anchor_matches_the_live_base_exactly_once(self, stager):
        if not (BASE_TREE / stager.SH_TARGET).is_file():
            pytest.skip("jg5 base tree not present in this checkout")
        text = (BASE_TREE / stager.SH_TARGET).read_text()
        for label, old, _ in stager.SH_REWRITES:
            assert text.count(old) == 1, f"{label}: anchor count {text.count(old)} != 1"

    def test_apply_refuses_a_missing_anchor(self, stager):
        with pytest.raises(stager.StagingError, match="found 0"):
            stager._apply("nothing here", "absent", "new", "probe")

    def test_apply_refuses_an_ambiguous_anchor(self, stager):
        with pytest.raises(stager.StagingError, match="found 2"):
            stager._apply("xx", "x", "y", "probe")

    def test_receipt_anchor_survives_a_cd1_style_instrumented_report(self, stager):
        """The receipt rewrite must compose onto ``ddm_cd1``'s instrumented tree.

        cd1 rewrites the same report block to add its breakdown; anchoring on the whole
        block would go stale against it, so the anchor is the bit-position line alone.  That
        is a claim about cd1's output, so it is tested against cd1's actual replacement text
        rather than against a hand-typed approximation of it.
        """
        cd1 = _load_stager_module(REPO / "experiments" / "ddm_cd1_stage_instrumented_runtime.py")
        receipt_old = stager.RECEIPT_OLD
        assert cd1.REPORT_NEW.count(receipt_old) == 1
        assert cd1.REPORT_OLD.count(receipt_old) == 1

    def test_helper_anchor_survives_cd1(self, stager):
        """cd1 re-emits the ``decode_production_tokens`` signature it anchors on, so this
        stager's helper anchor must still appear exactly once afterwards."""
        cd1 = _load_stager_module(REPO / "experiments" / "ddm_cd1_stage_instrumented_runtime.py")
        assert cd1.HELPER_NEW.count(stager.HELPER_ANCHOR) == 1

    def test_added_files_exist_in_the_repo(self, stager):
        assert (SOURCE_DIR / "f26_corrector_native.c").is_file()
        assert (SOURCE_DIR / "native_free_corrector.py").is_file()


def _load_stager_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestExpectedDelta:
    """``_assert_expected_delta`` is the invariant that keeps the diff honest."""

    def _trees(self, tmp_path: Path) -> tuple[Path, Path]:
        base = tmp_path / "base"
        out = tmp_path / "out"
        for root in (base, out):
            (root / "runtime").mkdir(parents=True)
            (root / "inflate.sh").write_text("#!/bin/bash\n")
            (root / "runtime" / "residual_archive.py").write_text("x = 1\n")
        return base, out

    def test_accepts_exactly_the_declared_delta(self, stager, tmp_path):
        base, out = self._trees(tmp_path)
        (out / "inflate.sh").write_text("#!/bin/bash\n# changed\n")
        (out / "runtime" / "residual_archive.py").write_text("x = 2\n")
        for name in stager.ADDED:
            (out / name).write_text("added\n")
        delta = stager._assert_expected_delta(base, out)
        assert delta["added"] == sorted(stager.ADDED)
        assert delta["changed"] == sorted((stager.PY_TARGET, stager.SH_TARGET))

    def test_refuses_an_undeclared_addition(self, stager, tmp_path):
        base, out = self._trees(tmp_path)
        (out / "inflate.sh").write_text("#!/bin/bash\n# changed\n")
        (out / "runtime" / "residual_archive.py").write_text("x = 2\n")
        for name in stager.ADDED:
            (out / name).write_text("added\n")
        (out / "runtime" / "surprise.py").write_text("nope\n")
        with pytest.raises(stager.StagingError, match="added"):
            stager._assert_expected_delta(base, out)

    def test_refuses_a_removal(self, stager, tmp_path):
        base, out = self._trees(tmp_path)
        (base / "runtime" / "extra.py").write_text("keep me\n")
        with pytest.raises(stager.StagingError, match="removed"):
            stager._assert_expected_delta(base, out)

    def test_refuses_an_unexpected_modification(self, stager, tmp_path):
        base, out = self._trees(tmp_path)
        for name in stager.ADDED:
            (out / name).write_text("added\n")
        (out / "inflate.sh").write_text("#!/bin/bash\n# changed\n")
        # residual_archive.py deliberately NOT changed -> changed set is wrong
        with pytest.raises(stager.StagingError, match="changed"):
            stager._assert_expected_delta(base, out)


# --- the binding ------------------------------------------------------------------------


def _import_runtime_tree():
    if not (BASE_TREE / "runtime" / "free_corrector.py").is_file():
        pytest.skip("jg5 base tree not present in this checkout")
    if str(BASE_TREE) not in sys.path:
        sys.path.insert(0, str(BASE_TREE))
    if str(SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(SOURCE_DIR))
    import native_free_corrector
    import runtime.free_corrector as free_corrector

    return free_corrector, native_free_corrector


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    """Compile the port with the SHIPPED flags.

    ``-ffp-contract=off`` is not hygiene here: FMA contraction fuses a multiply and an add
    into one rounding step, so a build without it would emit different probabilities.  The
    test compiles the way ``inflate.sh`` compiles, or it is testing a different binary.
    """
    source = SOURCE_DIR / "f26_corrector_native.c"
    if not source.is_file():
        pytest.skip("native corrector source missing")
    out = tmp_path_factory.mktemp("rr8") / "f26_corrector_native.so"
    result = subprocess.run(
        [
            "cc", "-O3", "-std=c11", "-shared", "-fPIC",
            "-ffp-contract=off", "-fno-fast-math",
            str(source), "-lm", "-o", str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"no working C toolchain: {result.stderr[:400]}")
    return out


class TestConfigDriftGuard:
    def test_accepts_the_live_shipped_config(self):
        free_corrector, binding = _import_runtime_tree()
        binding.assert_config_matches()  # must not raise against the shipped tree

    def test_refuses_a_drifted_config(self, monkeypatch):
        """A config the C was not compiled for must REFUSE, not silently run."""
        free_corrector, binding = _import_runtime_tree()
        drifted = dict(free_corrector.SHIPPED_CONFIG)
        drifted["mixer_context"] = "cls_boundary_agree_ubin8"
        monkeypatch.setattr(free_corrector, "SHIPPED_CONFIG", drifted)
        with pytest.raises(binding.NativeCorrectorError, match="mixer_context"):
            binding.assert_config_matches()

    def test_refuses_a_drifted_constant(self, monkeypatch):
        """Constants reach the mixer as defaults, not through SHIPPED_CONFIG, so they need
        their own check or a change would slip past the config comparison entirely."""
        free_corrector, binding = _import_runtime_tree()
        import runtime.fx1_logistic_mixer_corrector as fx1

        monkeypatch.setattr(fx1, "POWER_BITS", 7)
        with pytest.raises(binding.NativeCorrectorError, match="POWER_BITS"):
            binding.assert_config_matches()

    def test_load_returns_none_when_no_library_is_named(self, monkeypatch):
        """The fail-closed path: no build -> Python corrector, never an exception."""
        _, binding = _import_runtime_tree()
        monkeypatch.delenv("F26_CORRECTOR_NATIVE_LIBRARY", raising=False)
        assert binding.load_native_corrector(PLANE) is None

    def test_load_raises_when_a_named_library_is_missing(self, monkeypatch, tmp_path):
        """Operator misconfiguration is a DIFFERENT class from a hostile toolchain."""
        _, binding = _import_runtime_tree()
        monkeypatch.setenv("F26_CORRECTOR_NATIVE_LIBRARY", str(tmp_path / "absent.so"))
        with pytest.raises(binding.NativeCorrectorError, match="missing"):
            binding.load_native_corrector(PLANE)


# --- differential parity ------------------------------------------------------------------


def _synthetic_frame(rng: np.random.Generator, groups: int, per_group: int):
    """Structurally valid decoder inputs: a causal wavefront over a real-sized plane.

    Positions are drawn WITHOUT replacement across the frame so no pixel is decoded twice,
    which is the invariant the causal-neighbour gather depends on.  Probabilities are
    normalised float32 rows, because that is what ``_probability_table`` emits.
    """
    total = groups * per_group
    positions = rng.choice(PLANE, size=total, replace=False).astype(np.int64)
    logits = rng.normal(0.0, 3.0, size=(total, NUM_CLASSES))
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability = (exponent / exponent.sum(axis=1, keepdims=True)).astype(np.float32)
    predicted = probability.argmax(axis=1).astype(np.int64)
    # Symbols agree with the prediction most of the time and miss otherwise, so both the hit
    # and the within-miss paths are exercised rather than only the common one.
    miss = rng.random(total) < 0.25
    symbols = predicted.copy()
    symbols[miss] = (predicted[miss] + 1 + rng.integers(0, NUM_CLASSES - 1, miss.sum())) % NUM_CLASSES
    return positions, probability, predicted, symbols.astype(np.int64)


class TestDifferentialParity:
    def test_c_matches_python_bit_for_bit_including_state(self, native_library):
        free_corrector, binding = _import_runtime_tree()
        python = free_corrector.FreeCorrector(PLANE)
        native = binding.NativeFreeCorrector(PLANE, native_library)

        rng = np.random.default_rng(20260820)
        groups, per_group = 24, 512
        try:
            for frame in range(4):
                boundary = rng.integers(0, 5, size=PLANE).astype(np.uint8)
                python.begin_frame(boundary)
                native.begin_frame(boundary)

                positions, probability, predicted, symbols = _synthetic_frame(
                    rng, groups, per_group
                )
                tokens = np.zeros(PLANE, dtype=np.uint8)
                for group in range(groups):
                    lo, hi = group * per_group, (group + 1) * per_group
                    py_state = python.group_state(
                        probability[lo:hi], predicted[lo:hi], positions[lo:hi]
                    )
                    nat_state = native.group_state(
                        probability[lo:hi], predicted[lo:hi], positions[lo:hi]
                    )
                    py_row = python.coding_row(py_state)
                    nat_row = native.coding_row(nat_state)
                    # BYTES.  The RC64 backend turns a row into an integer frequency with
                    # ``(uint64_t)(value * 2**31)``, so one float32 ULP moves a frequency by
                    # up to 128 counts and desynchronises the decoder from there on.
                    assert py_row.tobytes() == nat_row.tobytes(), (
                        f"frame {frame} group {group}: coding_row differs"
                    )
                    python.observe(py_state, symbols[lo:hi])
                    native.observe(nat_state, symbols[lo:hi])
                    tokens[positions[lo:hi]] = symbols[lo:hi].astype(np.uint8)

                python.end_frame(tokens)
                native.end_frame(tokens)

                # State, not only output: a cold cell emits exactly 1.0 on both sides, so an
                # output-only comparison can pass long after the tables have diverged.
                for position in range(len(python.families)):
                    family = python.families[position]
                    for which, expected in (
                        (0, family.counts), (1, family.hits), (2, family.phat_q)
                    ):
                        assert np.array_equal(
                            native.table(which, position), expected
                        ), f"frame {frame}: family {family.name} table {which} diverged"
                assert np.array_equal(native.table(3), python.weights.reshape(-1))
                assert np.array_equal(native.table(4), python._miss_counts.reshape(-1))
                assert np.array_equal(native.table(5), python._miss_expect.reshape(-1))
                assert np.array_equal(native.table(6), python._miss_seen)
                assert np.array_equal(native.table(7), python.run)
        finally:
            native.close()

    def test_the_learner_actually_moved_the_weights(self, native_library):
        """A parity test over an INERT mechanism proves nothing.

        If the mixer weights never left their initial value, ``dyadic_power`` would be
        exercised only on the integer path and the whole radical/gradient surface would be
        untested while every assertion above still passed.  So the fixture is checked for
        having driven the thing it claims to compare.
        """
        free_corrector, binding = _import_runtime_tree()
        python = free_corrector.FreeCorrector(PLANE)
        native = binding.NativeFreeCorrector(PLANE, native_library)
        rng = np.random.default_rng(7)
        initial = python.weights.copy()
        try:
            for _ in range(3):
                boundary = rng.integers(0, 5, size=PLANE).astype(np.uint8)
                python.begin_frame(boundary)
                native.begin_frame(boundary)
                positions, probability, predicted, symbols = _synthetic_frame(rng, 8, 700)
                tokens = np.zeros(PLANE, dtype=np.uint8)
                for group in range(8):
                    lo, hi = group * 700, (group + 1) * 700
                    py_state = python.group_state(
                        probability[lo:hi], predicted[lo:hi], positions[lo:hi]
                    )
                    nat_state = native.group_state(
                        probability[lo:hi], predicted[lo:hi], positions[lo:hi]
                    )
                    assert python.coding_row(py_state).tobytes() == (
                        native.coding_row(nat_state).tobytes()
                    )
                    python.observe(py_state, symbols[lo:hi])
                    native.observe(nat_state, symbols[lo:hi])
                    tokens[positions[lo:hi]] = symbols[lo:hi].astype(np.uint8)
                python.end_frame(tokens)
                native.end_frame(tokens)

            assert not np.array_equal(python.weights, initial), (
                "the mixer learner never moved; this fixture does not exercise the port"
            )
            assert np.array_equal(native.table(3), python.weights.reshape(-1))
            assert python._miss_seen.sum() > 0, "the within-miss law was never reached"
        finally:
            native.close()
