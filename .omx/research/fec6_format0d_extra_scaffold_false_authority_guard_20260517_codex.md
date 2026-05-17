# FEC6 Format0D-EXTRA Scaffold False-Authority Guard - 2026-05-17

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_provider_dispatch=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`

No provider dispatch was launched and no lane claim was opened by this
follow-up.

## Bug Class

The `fec6+format0d-EXTRA` WIP was directionally useful for Rule #6/FEC6
stacking, but it carried a false-authority contradiction:

- the design memo and builder prose described the packet as dispatch-ready;
- the generated `inflate.py` intentionally raised `NotImplementedError`
  because the canonical fec6 base inflate path was not wired yet.

That means the archive grammar scaffold was real, but provider or exact-eval
dispatch from the emitted runtime would have been a known runtime failure, not
score evidence.

## Fix

The builder now emits explicit scaffold-only custody in `build_manifest.json`:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_provider_dispatch=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`
- `runtime_scaffold_only=true`
- `runtime_consumption_proof=false`
- `byte_consumption_proof=false`
- `dispatch_blockers=[inflate_py_phase1_scaffold_raises_NotImplementedError, fec6_base_inflate_path_not_wired, no_runtime_consumption_proof]`

The design memo now classifies the lane as scaffold-only until Phase 2 wires
the generated `inflate.py` to canonical fec6 base inflate, applies the
format0D-EXTRA latent correction before FES1 frame-0 RGB selector correction,
and proves byte consumption plus full-frame inflate success.

## Test Coverage

`src/tac/tests/test_fec6_format0d_extra.py` now builds a tiny deterministic
packet and asserts the manifest remains dispatch-ineligible while the generated
runtime still contains `NotImplementedError`.

This is intentionally a blocker-preserving test. It prevents the scaffold from
silently becoming a provider-dispatch authority before the runtime effect is
real.

## Routing Consequence

`fec6+format0d-EXTRA` remains a viable Rule #6/FEC6 bolt-on candidate, but the
next score-moving object is Phase 2 runtime consumption:

1. wire fec6 base inflate into the generated runtime;
2. apply the extra latent correction before FES1 selector correction;
3. mutate the extra bytes and prove full-frame output changes;
4. only then produce a paired CPU/CUDA exact-eval dispatch packet.

Until then, this lane is not rank/kill/promote eligible and should not consume
provider budget.

## Verification

Run before commit:

```bash
.venv/bin/python -m pytest src/tac/tests/test_fec6_format0d_extra.py
.venv/bin/python -m ruff check src/tac/fec6_format0d_extra.py src/tac/tests/test_fec6_format0d_extra.py tools/build_fec6_plus_format0d_extra_packet.py
.venv/bin/python -m py_compile src/tac/fec6_format0d_extra.py src/tac/tests/test_fec6_format0d_extra.py tools/build_fec6_plus_format0d_extra_packet.py
git diff --check -- src/tac/fec6_format0d_extra.py src/tac/tests/test_fec6_format0d_extra.py tools/build_fec6_plus_format0d_extra_packet.py .omx/research/fec6_plus_format0d_extra_design_20260517.md .omx/research/fec6_format0d_extra_scaffold_false_authority_guard_20260517_codex.md
```
