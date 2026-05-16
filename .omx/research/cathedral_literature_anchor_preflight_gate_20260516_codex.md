# Cathedral Literature Anchor Preflight Gate

Date: 2026-05-16
Owner: codex
Lane: lane_cathedral_literature_anchor_preflight_gate_20260516

## Problem

The Cathedral/autopilot composition matrix now carries literature anchors for
long-burn rows such as Ballé hyperprior, cooperative-receiver, predictive
coding, world-model, Time-Traveler L5, C1, and C6. The source-fidelity hardening
added four required fields:

- `source_supports`
- `paper_claim_scope`
- `pact_must_prove`
- `decode_complexity_evidence`

Without a preflight gate, a future row could keep `literature_anchor` and a
predicted score band while dropping the source-scope boundary. That would turn
paper citations into false score authority rather than hypothesis provenance.

## Landing

Catalog #293 adds
`check_cathedral_literature_anchors_have_source_scope` to `src/tac/preflight.py`.
The gate validates both canonical substrate inventory rows and serialized
Pareto rows consumed by autopilot/reporting surfaces.

Required invariant:

```text
literature_anchor != "" implies all source-scope fields are non-placeholder
```

The check is wired into `preflight_all(strict=True)` because live canonical rows
are already clean.

## Verification

Focused test target:

```bash
pytest src/tac/tests/test_check_293_cathedral_literature_scope.py -q
```

Related regression target:

```bash
pytest src/tac/tests/test_substrate_composition_matrix.py \
  src/tac/tests/test_cathedral_autopilot_substrate_composition_wire.py \
  src/tac/tests/test_build_composition_ranking_json.py -q
```

## Source-Fidelity Contract

This gate does not decide which paper is correct or whether a substrate will
lower score. It only forces the planning surface to preserve the distinction
between:

- what the cited source actually supports;
- what the source does not claim for this contest;
- what Pact still must prove empirically on byte-closed archives; and
- whether decode/runtime complexity has been measured on compliant hardware.

The row remains planning-only until exact contest evidence lands.
