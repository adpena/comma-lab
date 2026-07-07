"""Regression test for the round-2 CRITICAL: the durable-daemon wraps a training arm in safe_run
for a per-arm RSS cap, then runs its OWN authoritative system admission gate. Before the fix, the
wrapped safe_run re-ran the gate a SECOND time from --rss-mb/1024 (the conservative CAP, e.g.
87.9GiB) instead of the accurate projection (~70GiB) — which on an armed-enforce box could
false-refuse an already-admitted governed #205 launch. The wrap must tell the inner safe_run to
SKIP its gate (the daemon is the single gate point) and forward the accurate projection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import spawn_durable_daemon as d  # noqa: E402


def _ns(**kw):
    base = dict(rss_cap_mb=90000, walltime_cap_s=None, label="t", projected_peak_gib=70.0)
    base.update(kw)
    return argparse.Namespace(**base)


def test_wrap_skips_inner_gate_and_forwards_accurate_projection():
    w = d._maybe_wrap_safe_run(["bash", "launch.sh"], _ns())
    assert "--skip-admission-gate" in w              # daemon is the single authoritative gate
    i = w.index("--projected-gib")
    assert w[i + 1] == "70.0"                        # accurate projection, NOT 90000/1024=87.9
    assert w[w.index("--rss-mb") + 1] == "90000"     # per-arm RSS cap still applied


def test_no_cap_infra_stays_unwrapped():
    assert d._maybe_wrap_safe_run(["x"], _ns(rss_cap_mb=None, walltime_cap_s=None)) == ["x"]


def test_missing_projection_still_skips_inner_gate():
    # even with no projection, the inner gate must be skipped (daemon gates authoritatively);
    # --projected-gib is simply omitted rather than defaulting to the inflated rss/1024.
    w = d._maybe_wrap_safe_run(["bash", "l.sh"], _ns(projected_peak_gib=None))
    assert "--skip-admission-gate" in w and "--projected-gib" not in w


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
