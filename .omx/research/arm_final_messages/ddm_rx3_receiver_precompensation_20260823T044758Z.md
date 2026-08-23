RX3’s authorized scorer-free unit is complete and committed as `85eec433e1`.

- Reproduced all 28,602 gross breaks: 11,685 repaired and 16,917 persistent.
- Native scorer margin was the strongest discriminator, AUC 0.826963. Legal native luma gradient reached AUC 0.590403.
- Materialized four real, executable candidates costing +12 to +13 archive bytes, with fitted parameters stored inside each archive.
- Verified both frame roles apply precompensation before final rounding, while preserving the exact 113,777-byte DX2 token stream.
- Retained every archive, deterministic repeat, parameter payload, runtime tree, and the superseded post-round runtimes.
- Did not run an advisory or scorer because AP1 still owns the single full-n600 lane. The frontier remains DX2 at 0.14821987563243377.
- Passed Ruff, compilation, self-test, runtime verification v3, two tracked review passes, serialized commit hooks, and the developer preflight with its extended 60-second diagnostic budget.

Artifacts: [evidence memo](/Users/adpena/Projects/pact/.omx/research/ddm_rx3_receiver_precompensation_20260823.md) and [experiment/queued scorer consumer](/Users/adpena/Projects/pact/experiments/ddm_rx3_receiver_precompensation.py).

## NEXT_IF_RESUMED

- **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** `ddm_rx3_receiver_precompensation`. **Consumer store:** `.omx/tmp/arm_receipts_local/ddm_rx3_receiver_precompensation/advisory_and_dali_scorer/`. **Fire trigger:** `main_hot_state` releases AP1’s scorer grant, AP1’s latest queue becomes terminal, and RX3 records a non-conflicting scorer claim. Then fire the four candidates sequentially in `FIRE_ORDER.json` order and aggregate their full-n600 DALI Seg/Pose rows.

## LIVE-HYPOTHESES

- `local_highpass_regression` may reproduce part of the natural receiver repair. It has the largest scorer-free repaired-position alignment, although its selectivity is modest.
- `gradient_band_repair_mean` may reduce collateral damage better than the global treatment. Native gradient is the strongest legal observable discriminator, and this candidate has the best repair selectivity among the fitted rungs.
- A small global decode-side bias may transfer from L28 to current DX2. The repaired population is strongly margin-selected, but only full-n600 Seg/Pose scoring can establish transfer.

## DEAD-ENDS

- Residual magnitude alone is not a useful repair selector: total residual L2 AUC is 0.517405 and uint8-only L2 AUC is 0.507318.
- GT-boundary distance does not distinguish repaired from persistent positions: AUC is 0.500594.
- The generation-1 post-round runtimes are invalid mechanism tests. They were preserved as superseded evidence and must not be scored.
- Downstream scorer margins cannot key a legal receiver transform; they are outcome measurements unavailable to the receiver.
- Prefix scoring and `native-hpac` are unavailable for RX3 and must not be retried.