# ddm_tw1 pre-registration — additivity sign (written BEFORE the 29-cell result)

Written at: 2026-08-02T00:55:17Z — while `marginal.log` still showed 30/145,
i.e. before ANY additivity row for the 29-cell sample had been computed or printed.

## Observed so far (2-cell smoke, receipt .../smoke/tw1_state_dependence_receipt.json)

Per-cell marginal byte saving RISES monotonically with drop depth:
- cell 583 (hood): k0 776 B -> kneeA 872 -> kneeB 916  (+18.0%)
- cell 438 (road): k0 863 B -> kneeA 923 -> kneeB 964  (+11.7%)

## PREDICTION (falsifiable)

If the per-unit price genuinely rises as neighbours are dropped, then dropping a SET
jointly must save MORE than the sum of its singleton marginals measured from the same
base state, because each member of the set is priced in the presence of the others.

  PREDICT: additivity defect = sum_of_singleton_marginals - joint_measured_saving  <  0
           (i.e. SUPERadditive) at every state, and |defect| grows with state depth.

## KILL CRITERION

If defect > 0 (subadditive) at k0, the state-dependence reading above is wrong or the
single-cell and set effects have different mechanisms, and the "rising price" law must
be restated as an artifact of wr1's drop ORDERING rather than a coder-context effect.

Note the confound this measurement controls for: wr1's ordering is flip-risk ascending,
tie-broken by residual-mass DESCENDING, so it drops the FATTEST safe cells first. That
ordering predicts per-cell savings should DECREASE along the descent. We observe the
opposite on FIXED cells, so the context effect must be strong enough to reverse it.
