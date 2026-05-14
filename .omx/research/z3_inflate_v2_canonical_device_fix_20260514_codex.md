# Z3 inflate_v2 canonical device-selector fix - 2026-05-14

## Summary

R1 recursive review flagged that `z3_balle_hyperprior_bolton/inflate_v2.py`
duplicated local `select_inflate_device` logic instead of using the canonical
shared inflate runtime helper.

This landing keeps the existing Z3 v2 public API (`torch.device` return value)
but delegates policy to `tac.substrates._shared.inflate_runtime.select_inflate_device`.
It preserves the explicit `PACT_INFLATE_DEVICE=mps` refusal message documented
by the Z3 v2 tests.

## Evidence

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/substrates/z3_balle_hyperprior_bolton/tests/test_z3_v2_substrate.py -q
# 31 passed
```

Classification: hardening fix only; no score claim and no archive bytes changed.
