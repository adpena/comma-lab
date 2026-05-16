# L5 v2 Probe Gate Verdict Hardening - 2026-05-16

## Summary

Hardened the L5 v2 `c1_z5_tt5l_probe_disambiguator` gate so a metadata stub
cannot satisfy it. The gate artifact must now embed the actual probe verdict,
not just schema/tool/candidate labels.

## Failure Class

- `l5_v2_gate_artifact_semantics_missing:c1_z5_tt5l_probe_disambiguator:probe_verdict`
- `l5_v2_gate_artifact_semantics_invalid:c1_z5_tt5l_probe_disambiguator:architecture_lock_allowed`
- `l5_v2_gate_artifact_semantics_invalid:c1_z5_tt5l_probe_disambiguator:probe_blockers_nonempty`
- `l5_v2_gate_artifact_semantics_missing:c1_z5_tt5l_probe_disambiguator:eligible_observations:*`

Before this patch, a hand-written artifact with the right candidate IDs and
`paired_exact_axes_required=true` could satisfy the disambiguator gate without
showing that `evaluate_l5_v2_probe()` had allowed architecture lock-in. The
artifact now must carry a verdict with:

- `architecture_lock_allowed=true`;
- `blockers=[]`;
- all required candidates;
- `contest_cpu` and `contest_cuda` required axes;
- eligible observations for every C1/Z5/TT5L candidate.

## Code Surfaces

- `src/tac/optimization/l5_staircase_v2.py`
- `src/tac/tests/test_l5_staircase_v2.py`

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py -q
# 33 passed
```

## Interpretation

This is a custody and rigor hardening. It does not make L5 v2 dispatch-ready:
the prediction band remains rank-blocked until baseline and empirical anchors
exist. It prevents stale metadata from unlocking the gate-probe path.
