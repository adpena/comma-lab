# Lane12 Geometry-Gate Repair Atoms - 2026-05-03

Scope: local-only Lane 12 geometry-unlock planning. No remote job was launched,
no retraining was started, no clearance packet was written, no archive was
built, and no score or promotion claim is made.

## Tool Added

`experiments/plan_lane12_geometry_gate_repair_atoms.py` emits deterministic
atom-level repair policies from existing Lane 12 geometry evidence:

- Alpha-Geo geometry JSON with `diagnostic=alpha_geo_0_nerv_geometry`;
- optional `alpha_geo_primitive_contract_v1` ranked boxes/transition pairs;
- optional predecoded `masks.mkv` and `masks.nrv` tensor cache for local crop
  disagreement and row-run cost estimates.

Default output:

```text
experiments/results/lane12_geometry_gate_repair_atoms_20260503/lane12_geometry_gate_repair_atoms.json
```

The planner records `score_claim=false`, `promotion_eligible=false`,
`exact_eval_claim=false`, `remote_jobs_dispatched=false`, and
`byte_closed_exact_eval_candidate_created=false`. Byte costs are deterministic
planning estimates only; they are not measured archive deltas.

## Current Blocker Targeted

The current Lane 12 readiness artifacts under
`experiments/results/lane12_l2_unblock_readiness_20260502` show decoded-baseline
contract and runtime closure passing, but L2 unblock remains blocked by missing
promotion-threshold Alpha-Geo geometry evidence and the missing clearance packet.
The repair-atom planner attacks the geometry side of that blocker by ranking
local residual-region, critical-component-box, and transition-pair correction
atoms that can later be converted into a charged builder.

## Verification

Focused verification target:

```bash
.venv/bin/python -m py_compile \
  experiments/plan_lane12_geometry_gate_repair_atoms.py \
  src/tac/tests/test_plan_lane12_geometry_gate_repair_atoms.py

.venv/bin/python -m pytest \
  src/tac/tests/test_plan_lane12_geometry_gate_repair_atoms.py -q
```

Promotion remains blocked until a future byte-closed candidate remeasures
geometry below promotion gates and then passes exact CUDA auth eval on the
canonical `archive.zip -> inflate.sh -> upstream/evaluate.py` path.
