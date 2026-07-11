# Costate-DSL consumer-leg disposition (2026-07-11) — OWED-to-#427, not silently dropped

**Trigger:** triality drift-detector (consumer leg) on commit `813543bd6` — it added public
controller-side DSL surface to `src/tac/witness_dsl/costate_agent_dsl.py` (+71 lines:
`AcquisitionSpec`, `RoutingSpec.interpretable_head` hard-req, PowerPlay/Gödel/prototype specs)
without touching any consumer surface.

**Verified state (this disposition, MEASURED):**
- No consumer imports `costate_agent_dsl`: `schedule_readback.py`, `tools/dashboard_server.py`,
  `tools/costate_digest.py`, `src/tac/witness_control/producer_bridge.py` — none reference it.
- `costate_agent_dsl` exposes only `.compile()` (→ `CompiledCostateOrgan`); it has NO generic
  `describe()` / `to_dict()` / registry-render surface → `[consumers-generic]` would be **dishonest**
  (there is no generic surface for the digest/dashboard to introspect). Not asserted.

**Disposition: the consumer leg is OWED and OWNED by the in-flight #427 fresh-Fable seal.** That
charter explicitly includes "reconcile triality (DAG FEED + costate_agent_dsl + equations/ledger) +
wire the sealed organ into its standing ADVISORY role (costate_digest / dashboard — score-neutral)."
The proper close is: (a) add a generic `describe()` / render surface to the costate DSL so the digest
+ dashboard render it by introspection (the same generic pattern the witness DSL uses), AND/OR
(b) wire `costate_digest.py` + the dashboard to surface the controller state (which prototypes fired,
mixture weights, uncertainty, acquisition queue — the OBSERVATORY per the Rudin/max-observability
directive). The Fable seal does the real wiring; I do NOT edit those files now (the seal agent is
active on the costate organ — avoid a moving-target collision).

**Why not do it in this turn:** the fresh-Fable adversarial-review + seal (#427) is live on the
costate organ (`tac.witness_control/*` + `costate_agent_dsl`); a main-loop edit to the DSL or its
consumers now would collide with the seal's own reconciliation pass. Recorded here so the drift is
TRACKED, not silent (the hook's purpose), and the seal clears it properly.

Pointer 0.19108282 [contest-CPU] UNMOVED — apparatus/consumer-leg disposition, moves no score.
