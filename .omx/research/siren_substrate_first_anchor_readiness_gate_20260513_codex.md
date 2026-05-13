# SIREN substrate first-anchor readiness gate (2026-05-13)

Scope: no GPU spend, no remote dispatch, no score claim, no proxy promotion.

This landing adds a fail-closed local readiness gate for
`lane_substrate_siren_20260512`:

```bash
.venv/bin/python tools/audit_siren_substrate_readiness.py --json --fail-if-not-ready
```

The gate checks the reusable local surfaces needed before a real SIREN
first-anchor training dispatch can be considered:

- `experiments/train_substrate_siren.py` exists, uses the shared trainer
  skeleton, declares Tier-1 operator flags, has smoke/full entrypoints, emits a
  runtime, builds `archive.zip`, and keeps promotion/exact-dispatch false.
- `.omx/operator_authorize_recipes/substrate_siren_modal_a100_dispatch.yaml`
  names the SIREN lane, declares target modes, points at the trainer, and
  advertises this readiness gate.
- `src/tac/substrates/siren/archive.py` declares SRV1 monolithic `0.bin`
  grammar with pack/parse functions.
- `src/tac/substrates/siren/inflate.py` consumes the parser/model, has no
  scorer imports, and avoids undeclared PIL runtime dependency by using a
  stdlib PNG writer.
- SIREN roundtrip and score-aware-loss tests are present.

The manifest remains deliberately fail-closed for paid execution:
`ready_for_remote_dispatch=false`, `ready_for_exact_eval_dispatch=false`,
`score_claim=false`, and `promotion_eligible=false`. A real first anchor still
requires operator authorization, an active `tools/claim_lane_dispatch.py` claim,
CUDA-capable provider execution, and harvested auth-eval custody.

This is a readiness gate, not a result ledger.
