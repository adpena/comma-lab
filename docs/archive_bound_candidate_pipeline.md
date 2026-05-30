# Archive-Bound Candidate Pipeline

**Status**: canonical TAC API for archive/entropy/substrate candidate emitters.

This pipeline keeps byte-producing work composable. A materializer, public
frontier intake, repair family, or new substrate adapter emits ordinary
candidate rows once; TAC normalizes them into
`tac_archive_bound_candidate_contract.v1` and every downstream consumer reads
that contract instead of duplicate readiness fields.

## Contract Boundary

Use `tac.optimization.archive_bound_candidate_contract` for all candidate
normalization:

- `archive_bound_candidate_contract_fields_for_row(row, repo_root=...)`
- `build_archive_bound_candidate_contract_surface(...)`
- `entropy_position_label_for_transform_kind(...)`

The contract records archive custody, source archive custody, receiver proof
state, runtime adapter state, entropy-stage position, substrate tags, byte
deltas, blockers, and false-authority fields. It never grants score authority,
promotion authority, budget spend authority, or exact dispatch authority.

## Adapter Spine

New substrates should implement one Protocol:

```python
class MyAdapter:
    adapter_id = "my_adapter"
    candidate_family = "my_family"

    def emit_archive_bound_candidate_rows(self, context):
        return [candidate_row]
```

Then call:

```python
from tac.optimization.archive_bound_candidate_adapter_spine import (
    build_archive_bound_candidate_adapter_package,
)

package = build_archive_bound_candidate_adapter_package(
    MyAdapter(),
    context={},
    repo_root=repo_root,
)
```

The package emits:

- archive-bound contract surfaces;
- deterministic replay bundles;
- MLX-local advisory triage requests;
- receiver-proof gates;
- exact-axis blockers;
- posterior-update hooks.

That package can be passed directly to
`build_cross_family_candidate_portfolio(..., archive_contract_surfaces=[package])`.

## Producer Rules

- Emit byte-closed archive paths and SHA-256s whenever bytes exist.
- Attach receiver proof and runtime adapter custody when available.
- Mark MLX rows as advisory only; they can route budget, not claim score.
- Prefer entropy-stage labels by transform kind: before coder, at coder, after
  coder, or unknown when the transform cannot be classified yet.
- Do not add a parallel readiness schema. Add fields to the shared contract or
  adapter spine.

## Consumer Rules

- Acquisition consumes only contract surfaces plus posterior ledgers.
- Negative outcomes, entropy-stage misses, byte-credit exhaustion, and
  anti-pattern blockers are routing penalties, not prose-only findings.
- Exact CPU/CUDA handoff remains fail-closed until archive custody, receiver
  proof, runtime custody, lane preclaim, and auth-axis payload custody are all
  present.
