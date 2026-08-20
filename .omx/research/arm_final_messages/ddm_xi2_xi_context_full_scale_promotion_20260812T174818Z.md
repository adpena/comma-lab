XI2 is **READY_TO_FIRE** and committed as `c2ecc575b9d1f822cb869d80ed9f0d22f383d096`.

The full handoff is in [ddm_xi2_xi_context_full_scale_promotion_20260812.md](/Users/adpena/Projects/pact/.omx/research/ddm_xi2_xi_context_full_scale_promotion_20260812.md). The implementation is [run_ddm_xi2_xi_context_full_scale.py](/Users/adpena/Projects/pact/tools/run_ddm_xi2_xi_context_full_scale.py).

Measured and verified:

- Two retained n600 xi-context arrays are byte-identical: 117,964,928 B each, SHA-256 `4280603a…a44`.
- Banked CL1 control remains 116,716 real Range bytes; it was not retrained.
- Five CPU tests passed, including causal-context identity and real PR130 pack/unpack exact logits.
- Ruff, formatting, compilation, two review passes, and the strict payload-retention gate passed.
- [BUILD_RECEIPT.json](/Volumes/APDataStore/pact/ddm_xi2_20260812/BUILD_RECEIPT.json) and [READY_TO_FIRE.json](/Volumes/APDataStore/pact/ddm_xi2_20260812/READY_TO_FIRE.json) are linked and SHA-verified.
- Expected runtime is 82.4104 minutes, derived from matched CL1 receipts; xi warp overhead remains unmeasured.

Not measured: XI2 training, full-scale Range bytes, scorer output, archive bytes, or an exact score. No Metal/MPS or scorer was run.

Pinned fire command:

```bash
PYTHONHASHSEED=0 TAC_ADMISSION_ENFORCE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python tools/safe_run.py --rss-mb 4096 --projected-gib 4 --timeout 7200 --label ddm_xi2_xi_context_n600 --status-receipt /Volumes/APDataStore/pact/ddm_xi2_20260812/run/main.safe_run.json -- .venv/bin/python tools/run_ddm_xi2_xi_context_full_scale.py --leg all --resume-from auto
```

The preregistered falsifier is `xi/control < 0.98`: **114,381 B passes; 114,382 B fails**. Failure closes this full-scale formulation. The XI1 14.6× n120 ratio cannot be transferred.

The implementation-routing skill’s two external Codex dispatch attempts were blocked by sandbox app-server permissions; its documented two-strike escape hatch authorized the completed local implementation.

## NEXT_IF_RESUMED

- **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN Metal executor. **Consumer store:** `/Volumes/APDataStore/pact/ddm_xi2_20260812/FULL_SCALE_RESULT.json`. **Fire trigger:** MPS is available, the local-Metal lane is free, and all source, storage, and memory-governor preflights pass; then execute the pinned command above.

## LIVE-HYPOTHESES

- Full-scale learned xi context may beat the banked control by more than 2% because XI1 showed a large matched-capacity n120 reduction and the prior n600 transport negative never tested a learned boundary-context coder. This remains untested; the weak-baseline projection itself predicts 121,100 B and grants no promotion.

## DEAD-ENDS

- Transferring XI1’s 14.6× ratio is closed because its n120 zero-plane comparator is much weaker than the banked full-scale CL1 control.
- Using XI1’s zero-plane “spatial” row as the comparator is closed; CL1 actually consumes the unwarped previous decoded partition.
- Retraining the CL1 control is closed because its exact full-scale Range, model, decode, configuration, and receipts are already banked and pinned.
- Treating the old raster/zlib temporal-transport negative as a family-level refutation is closed because that verdict was formulation-scoped and explicitly left learned boundary context open.
- Claiming score progress from this build is closed: effective CP135 remains `S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`; own-vehicle frontier remains LC2 `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, adjudicated, n600]`.