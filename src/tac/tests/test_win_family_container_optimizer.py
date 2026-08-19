# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.win_families.container_optimizer` (win family F3).

The identity control is EXECUTED against a synthetic compiler with a known sha, and the
anti-laundering rules (sealed space, tie-to-incumbent, parse-back-before-win) are each
tested as behaviour rather than asserted about in a docstring.
"""

from __future__ import annotations

import hashlib

import pytest

from tac.win_families import container_optimizer as co

# The ddm_up3 measurement this family's law rests on.
UP3_PAYLOAD_DELTA_BITS = 7
UP3_ARCHIVE_DELTA_BYTES = 48


# --- score arithmetic ---------------------------------------------------------


def test_bytes_to_score_uses_the_upstream_rate_weight_and_denominator():
    assert co.bytes_to_score(37_545_489) == pytest.approx(25.0)


def test_saving_bytes_is_a_negative_score_delta():
    assert co.bytes_to_score(-762) < 0


def test_the_family_total_matches_the_measured_762_bytes():
    """ck2 (-657 B) + to1 (-105 B) = -762 B = -5.07e-04 S."""
    assert co.bytes_to_score(-(657 + 105)) == pytest.approx(-5.074e-04, rel=1e-3)


# --- ContainerConfig ----------------------------------------------------------


def test_config_defaults_are_the_shipped_shape():
    config = co.ContainerConfig()
    assert config.interleave is True
    assert (config.brotli_quality, config.brotli_lgwin) == (11, 24)


def test_config_refuses_out_of_range_quality():
    with pytest.raises(co.ContainerOptimizerError, match="quality"):
        co.ContainerConfig(brotli_quality=12)


def test_config_refuses_out_of_range_lgwin():
    with pytest.raises(co.ContainerOptimizerError, match="lgwin"):
        co.ContainerConfig(brotli_lgwin=9)


def test_config_refuses_a_repeated_section():
    with pytest.raises(co.ContainerOptimizerError, match="repeats a section"):
        co.ContainerConfig(section_order=("a", "b", "a"))


def test_config_describe_is_readable():
    assert "interleave=off" in co.ContainerConfig(interleave=False).describe()


# --- ContainerSpace: the seal ---------------------------------------------------


def test_space_refuses_to_be_empty():
    with pytest.raises(co.ContainerOptimizerError, match="at least one"):
        co.ContainerSpace([])


def test_space_refuses_an_out_of_range_incumbent():
    with pytest.raises(co.ContainerOptimizerError, match="outside"):
        co.ContainerSpace([co.ContainerConfig()], incumbent_index=3)


def test_space_refuses_duplicate_configs():
    """A duplicate silently doubles a config's chance of winning a tie-break."""
    with pytest.raises(co.ContainerOptimizerError, match="duplicate"):
        co.ContainerSpace([co.ContainerConfig(), co.ContainerConfig()])


def test_seal_digest_is_stable_for_the_same_declared_space():
    first = co.ContainerSpace(co.UP3_DECLARED_OPTIONS, name="s")
    second = co.ContainerSpace(co.UP3_DECLARED_OPTIONS, name="s")
    assert first.seal_digest == second.seal_digest


def test_growing_the_space_changes_the_seal():
    """The anti-laundering property: a larger search cannot pose as the same experiment."""
    small = co.ContainerSpace(co.UP3_DECLARED_OPTIONS, name="s")
    grown = co.ContainerSpace(
        (*co.UP3_DECLARED_OPTIONS, co.ContainerConfig(False, 8, 16)), name="s"
    )
    assert small.seal_digest != grown.seal_digest


def test_renaming_the_space_changes_the_seal():
    a = co.ContainerSpace(co.UP3_DECLARED_OPTIONS, name="a")
    b = co.ContainerSpace(co.UP3_DECLARED_OPTIONS, name="b")
    assert a.seal_digest != b.seal_digest


def test_up3_declared_space_has_the_shipped_shape_at_index_zero():
    space = co.ContainerSpace(co.UP3_DECLARED_OPTIONS)
    assert space.incumbent.interleave is True
    assert (space.incumbent.brotli_quality, space.incumbent.brotli_lgwin) == (11, 24)


def test_space_len_and_json_agree():
    space = co.ContainerSpace(co.UP3_DECLARED_OPTIONS)
    assert len(space) == space.to_json()["size"] == 8


# --- the search ------------------------------------------------------------------


def _space(n: int = 3) -> co.ContainerSpace:
    configs = [
        co.ContainerConfig(True, 11, 24, label="shipped"),
        co.ContainerConfig(False, 10, 16, label="alt1"),
        co.ContainerConfig(False, 9, 16, label="alt2"),
    ][:n]
    return co.ContainerSpace(configs, name="test_space")


def _sized_compiler(sizes: dict[str, int]):
    def compile_fn(config: co.ContainerConfig) -> bytes:
        return (config.label.encode() + b"\x00" * 64)[: sizes[config.label]]

    return compile_fn


def test_search_selects_the_smallest_admissible_candidate():
    result = co.search_container_space(
        _space(),
        compile_fn=_sized_compiler({"shipped": 40, "alt1": 30, "alt2": 35}),
        parse_back_fn=lambda blob: "payload",
        expected_payload="payload",
    )
    assert result.winner.config.label == "alt1"
    assert result.delta_archive_bytes == -10


def test_search_reports_a_negative_score_for_a_saving():
    result = co.search_container_space(
        _space(),
        compile_fn=_sized_compiler({"shipped": 40, "alt1": 30, "alt2": 35}),
        parse_back_fn=lambda blob: "payload",
        expected_payload="payload",
    )
    assert result.delta_score < 0


def test_ties_go_to_the_incumbent():
    """Noise must never manufacture a win over the object that already ships."""
    result = co.search_container_space(
        _space(),
        compile_fn=_sized_compiler({"shipped": 30, "alt1": 30, "alt2": 30}),
        parse_back_fn=lambda blob: "payload",
        expected_payload="payload",
    )
    assert result.winner.index == result.incumbent_index
    assert result.delta_archive_bytes == 0


def test_a_candidate_that_does_not_parse_back_cannot_win():
    """A container that loses information is not a smaller archive, it is a broken one."""

    def parse_back(blob: bytes) -> str:
        return "payload" if len(blob) >= 35 else "corrupt"

    result = co.search_container_space(
        _space(),
        compile_fn=_sized_compiler({"shipped": 40, "alt1": 30, "alt2": 35}),
        parse_back_fn=parse_back,
        expected_payload="payload",
    )
    assert result.winner.config.label == "alt2"
    assert result.admissible_count == 2


def test_a_failing_config_is_recorded_not_raised():
    def compile_fn(config: co.ContainerConfig) -> bytes:
        if config.label == "alt1":
            raise ValueError("brotli exploded")
        return b"\x00" * 40

    result = co.search_container_space(
        _space(),
        compile_fn=compile_fn,
        parse_back_fn=lambda blob: "payload",
        expected_payload="payload",
    )
    failed = next(c for c in result.candidates if c.config.label == "alt1")
    assert failed.admissible is False
    assert "brotli exploded" in failed.error


def test_a_failing_incumbent_is_fatal():
    """Without a working incumbent there is no honest baseline."""

    def compile_fn(config: co.ContainerConfig) -> bytes:
        if config.label == "shipped":
            raise ValueError("cannot rebuild the shipped body")
        return b"\x00" * 10

    with pytest.raises(co.ContainerOptimizerError, match="no honest baseline"):
        co.search_container_space(
            _space(),
            compile_fn=compile_fn,
            parse_back_fn=lambda blob: "payload",
            expected_payload="payload",
        )


def test_incumbent_that_does_not_round_trip_is_fatal():
    with pytest.raises(co.ContainerOptimizerError, match="no honest baseline"):
        co.search_container_space(
            _space(),
            compile_fn=_sized_compiler({"shipped": 40, "alt1": 30, "alt2": 35}),
            parse_back_fn=lambda blob: "always wrong",
            expected_payload="payload",
        )


def test_compile_returning_non_bytes_is_recorded_as_inadmissible():
    def compile_fn(config: co.ContainerConfig):
        return "not bytes" if config.label == "alt1" else b"\x00" * 40

    result = co.search_container_space(
        _space(),
        compile_fn=compile_fn,
        parse_back_fn=lambda blob: "payload",
        expected_payload="payload",
    )
    assert not next(c for c in result.candidates if c.config.label == "alt1").admissible


def test_result_carries_the_seal_of_the_space_that_produced_it():
    space = _space()
    result = co.search_container_space(
        space,
        compile_fn=_sized_compiler({"shipped": 40, "alt1": 30, "alt2": 35}),
        parse_back_fn=lambda blob: "payload",
        expected_payload="payload",
    )
    assert result.seal_digest == space.seal_digest


def test_candidate_sha_is_the_sha_of_the_compiled_bytes():
    compile_fn = _sized_compiler({"shipped": 40, "alt1": 30, "alt2": 35})
    result = co.search_container_space(
        _space(),
        compile_fn=compile_fn,
        parse_back_fn=lambda blob: "payload",
        expected_payload="payload",
    )
    expected = hashlib.sha256(compile_fn(result.winner.config)).hexdigest()
    assert result.winner.archive_sha256 == expected


def test_result_json_is_never_a_score_claim():
    result = co.search_container_space(
        _space(),
        compile_fn=_sized_compiler({"shipped": 40, "alt1": 30, "alt2": 35}),
        parse_back_fn=lambda blob: "payload",
        expected_payload="payload",
    )
    assert result.to_json()["score_claim"] is False


def test_custom_payload_equality_is_honoured():
    """``==`` on arrays returns an array, not a verdict; callers must be able to override."""
    calls = {"n": 0}

    def equal(a, b):
        calls["n"] += 1
        return True

    co.search_container_space(
        _space(),
        compile_fn=_sized_compiler({"shipped": 40, "alt1": 30, "alt2": 35}),
        parse_back_fn=lambda blob: object(),
        expected_payload=object(),
        payload_equal=equal,
    )
    assert calls["n"] == 3


# --- the archive-vs-payload law ---------------------------------------------------


def test_up3_anchor_shows_archive_delta_is_not_payload_delta():
    """EXECUTED: +7 payload bits cost +48 archive bytes on the real body."""
    report = co.archive_delta_report(
        payload_delta_bits=UP3_PAYLOAD_DELTA_BITS,
        archive_delta_bytes=UP3_ARCHIVE_DELTA_BYTES,
    )
    assert report.payload_delta_bytes_ceiling == 1
    assert report.container_attributable_bytes == 47
    assert report.law_holds is True


def test_zero_container_term_when_the_deltas_agree():
    report = co.archive_delta_report(payload_delta_bits=8, archive_delta_bytes=1)
    assert report.container_attributable_bytes == 0
    assert report.law_holds is False


def test_negative_payload_delta_rounds_away_from_zero():
    report = co.archive_delta_report(payload_delta_bits=-7, archive_delta_bytes=-1)
    assert report.payload_delta_bytes_ceiling == -1


def test_delta_report_json_carries_the_law():
    payload = co.archive_delta_report(
        payload_delta_bits=7, archive_delta_bytes=48
    ).to_json()
    assert "archive delta is not payload delta" in payload["law"]


# --- the controls -------------------------------------------------------------------


def test_identity_control_passes_on_a_reproducing_builder():
    blob = b"shipped bytes"
    control = co.identity_control(
        compile_fn=lambda config: blob,
        shipped_config=co.ContainerConfig(),
        expected_sha256=hashlib.sha256(blob).hexdigest(),
    )
    assert control.passed is True


def test_identity_control_fails_on_a_drifting_builder():
    control = co.identity_control(
        compile_fn=lambda config: b"different",
        shipped_config=co.ContainerConfig(),
        expected_sha256=hashlib.sha256(b"shipped").hexdigest(),
    )
    assert control.passed is False


def test_determinism_control_passes_on_a_stable_builder():
    control = co.determinism_control(
        compile_fn=lambda config: b"stable", config=co.ContainerConfig()
    )
    assert control.passed is True


def test_determinism_control_catches_a_nondeterministic_builder():
    counter = {"n": 0}

    def unstable(config):
        counter["n"] += 1
        return f"run{counter['n']}".encode()

    control = co.determinism_control(compile_fn=unstable, config=co.ContainerConfig())
    assert control.passed is False


def test_control_json_carries_the_control_name():
    control = co.determinism_control(
        compile_fn=lambda config: b"x", config=co.ContainerConfig()
    )
    assert control.to_json()["control"] == "double_compile_determinism"
