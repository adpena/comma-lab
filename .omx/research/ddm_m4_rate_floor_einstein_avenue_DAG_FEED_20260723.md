# FEED — DDM M4 rate-floor / Einstein Avenue

`research_only=true` · `$0` · no launch · no new scorer run · `score_claim=false` ·
pointer unchanged · MAIN landing review required.

## Executable evidence DAG

```text
#575 exact n600 archive + report ─┐
Q-axis n600 int8/int7/int6 ──────┤
historical exact-C1 receipt ─────┤
DDM v19b receiver row ───────────┤
                                 ├─> derive_ddm_m4_rate_floor.py
#604 rule-118 / U2 memo ─────────┤       │
#580 ker(A) receipt ─────────────┤       ├─> exact score/cap law
#574 xi-temporal receipt ────────┤       ├─> audited-set receiver minimum
#602 MDL member receipt ─────────┘       ├─> FREE/NULL/COUNTED + pool law
                                         └─> deterministic JSON receipt
                                                   │
                                                   ├─> focused tests
                                                   ├─> adversarial review
                                                   └─> MAIN merge-boundary review
```

Every source receipt is SHA-bound in
`.omx/research/ddm_m4_rate_floor_einstein_avenue_20260723_receipt.json`.
The #575 archive itself is rehashed and its physical 177,169-byte size is rechecked. No score is
re-run: this FEED changes the custody/decision surface, not the frontier pointer.

## Triality

- **DAG:** this file; exact inputs, admission gates, receipt, tests, and MAIN review edge.
- **DSL:** N/A with rationale. This lane adds no trainer lever, flag, launch, config, or curriculum
  state. Inventing a DSL knob would be false wiring.
- **Equations:** `tac.canonical_equations.ddm_m4_rate_floor_20260723` owns:
  strict fixed-distortion archive cap, exact score terms, typed receiver-row admission,
  seven-lever pool partition, and the scoped uint8 scheduled-debt law.

Canonical equations:

```text
S(d_seg,d_pose,B) = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489

B_cap(d_seg,d_pose; S*) =
    max { B in Z_{\ge 0} : S(d_seg,d_pose,B) < S* }

B_audit(Ds,Dp; R) =
    min { row.B : row in R,
                  row.n_pairs=600,
                  row.receiver_closed,
                  row.d_seg<=Ds,
                  row.d_pose<=Dp }

Delta_S_unrecovered =
    Delta_S_scheduled * (1 - realized_fraction)
```

At exact C1 and `S*=0.15`, `B_cap=154,524`. For the explicit audited relaxed-box set,
`B_audit=177,169`; for the explicit exact-C1 set, `B_audit=409,526,925`. Neither audited minimum is
a universal lower bound.

## Unified-stack wire-in disposition

No empirical result is left as chat-only state:

1. **Sensitivity map:** no new pixel/pair sensitivity was measured; existing #580/#574/v19b
   anchors are consumed by hash, so no new sensitivity row is authorized.
2. **Pareto constraint:** the exact `B_cap=154,524` and `B_audit=177,169` become the hard rate gap
   `22,645 B`; no proxy byte delta may enter.
3. **Bit allocator:** the pool map forbids adding same-pool singleton savings; allocator credit is
   conditional on one final receiver replay.
4. **Cathedral/autopilot:** no dispatch hook is enabled because no <=154,524-byte receiver row
   exists; MAIN review is the only successor edge.
5. **Continual learning:** the dated Codex findings memo records the receiver-box distinction,
   zero rule-118/ker(A) byte credits, and lattice debt.
6. **Probe disambiguator:** the deterministic derivation command arbitrates “relaxed box” versus
   “exact C1” and refuses to report one fake global floor.

## Verdict and blocker

`SUB015_NOT_REACHED_22645_BYTE_GAP_NOT_RULED_OUT`.

Exact blocker: no current same-artifact n600 receiver row simultaneously has
`B<=154,524`, `d_seg<=0.00116`, and `d_pose<=0.00161`; retaining exact C1 distortion at that rate
is even stronger and also absent. This is an explicit-audited-set and current-construction
blocker—not a family-dead or information-theoretic impossibility verdict.

`verdict_scope=EXPLICIT_AUDITED_SET:#575_EXACT_N600 + QAXIS_N600_INT8_INT7_INT6 +
HISTORICAL_EXACT_C1 + DDM_V19B; NO_GLOBAL_MDL_OPTIMUM; NO_NEW_SCORE; NO_PROMOTION`.
