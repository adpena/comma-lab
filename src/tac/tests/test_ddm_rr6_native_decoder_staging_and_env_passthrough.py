# SPDX-License-Identifier: MIT
"""ddm_rr6 - behaviour tests for the two surfaces that put the native token
decoder into a submission.

These assert BEHAVIOUR, not constants.  Each test is written so that replacing
the function body with a canonical-looking return value fails it:

* the env passthrough is checked by the values it REFUSES, not by the presence
  of a flag;
* the inflate.sh rewrites are checked by composing them onto a synthetic base
  and reading the resulting shell control flow, not by comparing literals to
  themselves.

Why these two surfaces are worth a regression guard:

1. ``fire_local_advisory`` carries PATH and PYTHONDONTWRITEBYTECODE because
   omitting either produced a measured launch failure (ck1 rc=2, V7 refusal at
   t=5s).  A passthrough that could overwrite them would reopen both classes,
   so the collision refusal is load-bearing and not hygiene.
2. ``F26_TOKEN_DECODER`` defaults decide whether the accelerator runs AT ALL on
   the contest runner, which invokes ``inflate.sh`` with no environment.  A
   default of ``python`` ships the port dark; a build failure that aborts
   instead of degrading turns a wall-clock WARN into a zero.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def firer():
    return _load(REPO / "tools" / "fire_local_advisory.py", "_rr6_fire_local_advisory")


@pytest.fixture(scope="module")
def stager():
    return _load(
        REPO / "experiments" / "ddm_wc2c_stage_native_split_runtime.py",
        "_rr6_stage_native_split_runtime",
    )


# --------------------------------------------------------------------------
# 1. env passthrough
# --------------------------------------------------------------------------


def test_passthrough_carries_a_runtime_knob(firer):
    assert firer.parse_passthrough_env(["F26_TOKEN_DECODER=native-hpac"]) == {
        "F26_TOKEN_DECODER": "native-hpac"
    }


def test_passthrough_allows_an_empty_value(firer):
    """An empty value is a legitimate way to clear a knob, not a malformed pair."""
    assert firer.parse_passthrough_env(["F26_HPAC_THREADS="]) == {"F26_HPAC_THREADS": ""}


def test_passthrough_keeps_an_equals_sign_inside_the_value(firer):
    assert firer.parse_passthrough_env(["A=b=c"]) == {"A": "b=c"}


@pytest.mark.parametrize("key", sorted(("PATH", "PYTHONDONTWRITEBYTECODE")))
def test_passthrough_refuses_every_carried_key(firer, key):
    """The refusal is the point: overriding a carried key reopens ck1/V7."""
    with pytest.raises(ValueError, match="carries it"):
        firer.parse_passthrough_env([f"{key}=anything"])


def test_carried_keys_are_exactly_the_keys_the_launcher_composes(firer):
    """Guards against a future key being carried but not protected.

    Reads the composed env out of the module rather than restating it, so a
    third carried key added to ``main`` without adding it to the refusal set
    fails here instead of silently becoming overridable.
    """
    source = (REPO / "tools" / "fire_local_advisory.py").read_text()
    body = source.split("env_pairs = {", 1)[1].split("}", 1)[0]
    composed = {
        line.split(":", 1)[0].strip().strip('"')
        for line in body.splitlines()
        if ":" in line
    }
    assert composed == set(firer.CARRIED_ENV_KEYS)


def test_passthrough_refuses_a_malformed_pair(firer):
    with pytest.raises(ValueError, match="KEY=VALUE"):
        firer.parse_passthrough_env(["no_equals_sign"])


def test_passthrough_refuses_an_empty_key(firer):
    with pytest.raises(ValueError, match="KEY=VALUE"):
        firer.parse_passthrough_env(["=value"])


def test_passthrough_refuses_a_duplicate_key(firer):
    """Order-dependent effective values are a confound, not a convenience."""
    with pytest.raises(ValueError, match="twice"):
        firer.parse_passthrough_env(["A=1", "A=2"])


# --------------------------------------------------------------------------
# 2. inflate.sh rewrites
# --------------------------------------------------------------------------


def test_rewrites_compose_onto_a_synthetic_base_in_stage_order(stager):
    """End-to-end on text, not literals: the composed script must select native
    by default, and its build failure branch must fall back rather than exit."""
    base = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + stager.DEFAULT_OLD
        + '  else\n'
        + stager.BUILD_OLD
        + stager.EXPORT_OLD
    )
    text = stager._apply(base, stager.BUILD_OLD, stager.BUILD_NEW, "build")
    text = stager._apply(text, stager.EXPORT_OLD, stager.EXPORT_NEW, "export")
    text = stager._apply(text, stager.DEFAULT_OLD, stager.DEFAULT_NEW, "default")

    assert 'F26_TOKEN_DECODER:-native-hpac' in text
    assert 'F26_TOKEN_DECODER:-python' not in text
    # exactly one export, and it is downstream of the success `then`
    assert text.count("export F26_HPAC_NATIVE_LIBRARY") == 1
    assert text.index("\n    then") < text.index("export F26_HPAC_NATIVE_LIBRARY")
    assert text.index("export F26_HPAC_NATIVE_LIBRARY") < text.index("export F26_TOKEN_DECODER=python")


def test_apply_refuses_when_the_base_text_has_drifted(stager):
    """A stale transformation must refuse rather than patch text it misreads."""
    with pytest.raises(stager.StagingError, match="drifted"):
        stager._apply("nothing to match here", "absent", "replacement", "label")


def test_apply_refuses_an_ambiguous_multiple_match(stager):
    with pytest.raises(stager.StagingError, match="found 2"):
        stager._apply("xx", "x", "y", "label")


def test_default_decoder_is_native_on_a_bare_invocation(stager):
    """upstream/evaluate.sh runs inflate.sh with no env; the default IS the ship."""
    assert 'F26_TOKEN_DECODER:-native-hpac' in stager.DEFAULT_NEW
    assert 'F26_TOKEN_DECODER:-python' not in stager.DEFAULT_NEW


def test_build_failure_degrades_to_python_instead_of_aborting(stager):
    """A compiler failure must lose speed, never the whole inflate."""
    new = stager.BUILD_NEW
    assert "export F26_TOKEN_DECODER=python" in new
    assert "exit" not in new.split("else", 1)[1]


def test_both_build_attempts_sit_inside_the_if_condition(stager):
    """`set -e` does not apply inside an `if` condition; outside it, it aborts."""
    new = stager.BUILD_NEW
    condition = new.split("\n    then", 1)[0]
    assert condition.count('"${CC:-cc}"') == 2
    assert "||" in condition


def _build_commands(stager) -> str:
    """BUILD_NEW with comment lines stripped.

    The flags must be on the COMMAND lines; prose that merely mentions a flag
    compiles nothing.  Counting over the raw string would let a comment satisfy
    the assertion, which is the tests-verify-constants failure mode.
    """
    return "\n".join(
        line for line in stager.BUILD_NEW.splitlines()
        if not line.lstrip().startswith("#")
    )


def test_shipped_build_pins_the_intrinsic_free_twin(stager):
    """Unexecuted AVX2 kernels must not be in the shipped binary."""
    assert _build_commands(stager).count("-DF26_FORCE_SCALAR=1") == 2


def test_ieee_flags_survive_every_build_attempt(stager):
    """FMA contraction and reassociation both desynchronise the decoder."""
    commands = _build_commands(stager)
    assert commands.count("-ffp-contract=off") == 2
    assert commands.count("-fno-fast-math") == 2
    assert "-ffast-math" not in commands.replace("-fno-fast-math", "")


def test_library_export_only_happens_on_a_successful_build(stager):
    """Exporting a path the build never produced would fail closed at `-f`."""
    assert "F26_HPAC_NATIVE_LIBRARY" not in stager.EXPORT_NEW
    success_branch = stager.BUILD_NEW.split("\n    then", 1)[1].split("else", 1)[0]
    assert "export F26_HPAC_NATIVE_LIBRARY" in success_branch


def test_an_explicitly_named_missing_library_still_fails_loud(stager):
    """Operator misconfiguration is a different class from a hostile toolchain."""
    base = (REPO / "submissions" / "robust_current" / "jg5_sub015_runtime"
            / "runtime" / "inflate.sh")
    if not base.is_file():
        pytest.skip("sealed runtime tree not present")
    text = base.read_text()
    assert 'missing F26 native library' in text and 'exit 69' in text
