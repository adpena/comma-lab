# L5 v2 TT5L Move-Level Structural Proof

Date: 2026-05-16
Author: Codex
Scope: TT5L / L5 v2 staircase move-level feasibility

## Summary

The L5 v2 staircase no longer relies on a hand-written
`.omx/state/tt5l_move_level_feasibility.json` shape. This landing adds and
runs a structural proof producer:

- `tools/prove_tt5l_move_level_feasibility.py`
- `.omx/research/tt5l_move_level_structural_proof_20260516_codex.json`

The canonical builder
`tools/build_tt5l_move_level_feasibility_artifact.py` now requires proof
tool provenance, tool SHA-256, score-axis sanity SHA-256, mechanism records,
and witness variables before emitting the local state artifact.

## Proof Scope

This is a structural mechanism proof only:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

It proves the implemented TT5L mechanism surface is present and
gradient/roundtrip-reachable for the required constraint IDs. It does not
claim score movement.

## Artifact Hashes

- Structural proof artifact:
  `.omx/research/tt5l_move_level_structural_proof_20260516_codex.json`
  `sha256=7c64142fc1614903324a5191183c2a359bfa6d09496a29bf130fa151a01f7f1a`
- Local state artifact:
  `.omx/state/tt5l_move_level_feasibility.json`
  `sha256=c3d80096a067d3ea43e08d9584209050423d5b5a014ea6563c4ce41463920162`
- Proof tool:
  `tools/prove_tt5l_move_level_feasibility.py`
  `sha256=fc96ca6bd89d772ee004c9c8222d842b698b6b812c6d75b647c1ccf4e55bfbf0`
- Canonical builder:
  `tools/build_tt5l_move_level_feasibility_artifact.py`
  `sha256=eb6fcfa8979c6752bec50eae6fc9c2cba3b1872de4cf8039cf03cf06c0e84f55`
- Dykstra score-axis sanity input:
  `.omx/state/dykstra_feasibility_time_traveler_l5.json`
  `sha256=226c227c1c08b25ea7208c6ee774f7621b25c25929870c28535a1f8896504b60`

## Commands

```bash
.venv/bin/python tools/prove_tt5l_move_level_feasibility.py \
  --repo-root . \
  --score-axis-sanity-artifact .omx/state/dykstra_feasibility_time_traveler_l5.json \
  --output-json .omx/research/tt5l_move_level_structural_proof_20260516_codex.json
```

Result: `predicate_passed=true`.

```bash
.venv/bin/python tools/build_tt5l_move_level_feasibility_artifact.py \
  --repo-root . \
  --proof-artifact .omx/research/tt5l_move_level_structural_proof_20260516_codex.json \
  --proof-command-argv-json '[".venv/bin/python","tools/prove_tt5l_move_level_feasibility.py","--repo-root",".","--score-axis-sanity-artifact",".omx/state/dykstra_feasibility_time_traveler_l5.json","--output-json",".omx/research/tt5l_move_level_structural_proof_20260516_codex.json"]' \
  --output-json .omx/state/tt5l_move_level_feasibility.json
```

Result: `artifact_valid=true`.

## Readiness After Landing

Current TT5L readiness:

```text
dykstra_valid=true
move_level_valid=true
sideinfo_valid=false
sideinfo_effect_curve_allowed=false
timing_artifact_valid=false
next_action=materialize_tt5l_contest_full_frame_sideinfo_consumption_proof
```

The staircase has advanced past move-level feasibility custody. The next
artifact-producing blocker is the contest full-frame temporal side-info
consumption proof with hardened inflate provenance.

## Verification

```bash
.venv/bin/python -m ruff check \
  src/tac/optimization/l5_staircase_v2.py \
  tools/build_tt5l_move_level_feasibility_artifact.py \
  tools/prove_tt5l_move_level_feasibility.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_build_tt5l_move_level_feasibility_artifact.py \
  src/tac/tests/test_prove_tt5l_move_level_feasibility.py
```

Result: all checks passed.

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_build_tt5l_move_level_feasibility_artifact.py \
  src/tac/tests/test_prove_tt5l_move_level_feasibility.py -q
```

Result: 97 passed.
