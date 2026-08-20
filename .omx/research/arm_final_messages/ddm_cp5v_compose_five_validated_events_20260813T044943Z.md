# READY_TO_FIRE

CP5V completed and landed as commit `2e61c5c337`.

- Archive: `/Volumes/VertigoDataTier/pact/ddm_cp5v_20260812/retained/candidates/validated_five/primary/objects/archive.zip`
- SHA-256: `1c66e434ba60c6ff0be5f8634742eff3a85332bab89a12540050df852ea7a986`
- Size: **186,252 B**, exactly **+0 B** versus CP135
- Independent HP3/RC64 repeat: byte-identical
- Canonical adapted-runtime decode: 465.939 s, token SHA `9ab877de7e63d064624040c994368f83eb70da15a5ccd3e42f6a4364828340a5`
- Token diff: exactly the five requested cells and nowhere else
- Projected additive score: `0.16195005201522092`
- Exact score: **not measured**
- No scorer, Modal dispatch, MPS, or Metal ran

Artifacts: [receipt](/Users/adpena/Projects/pact/.omx/research/ddm_cp5v_compose_five_validated_events_20260812.md), [fire recipe](/Users/adpena/Projects/pact/.omx/research/ddm_cp5v_compose_five_validated_events_20260812_t4_recipe.json), [driver](/Users/adpena/Projects/pact/experiments/ddm_cp5v_compose_five_validated_events.py). The retained SSD tree is 2.7 GiB; all payloads remain present. Verification passed: 11 tests, Ruff, compilation, payload-retention gate, two review passes, and serializer post-commit hashes.

Canonical claim and dispatch:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id lane_ddm_cp5v_validated_five_contest_cuda_20260812 --platform modal --instance-job-id ddm_cp5v_validated_five_t4_20260812 --agent MAIN --status active_exact_eval_spawning --notes 'CP5V five n600-validated EC1 events; sole contest-CUDA row; validate 1c66e434ba60c6ff0be5f8634742eff3a85332bab89a12540050df852ea7a986 at 186252 B'
```

```bash
.venv/bin/modal run --detach experiments/modal_auth_eval.py::main --archive /Volumes/VertigoDataTier/pact/ddm_cp5v_20260812/retained/candidates/validated_five/primary/objects/archive.zip --output-dir /Volumes/VertigoDataTier/pact/ddm_cp5v_20260812/main_t4 --expected-archive-sha256 1c66e434ba60c6ff0be5f8634742eff3a85332bab89a12540050df852ea7a986 --submission-dir /Volumes/VertigoDataTier/pact/ddm_cp5v_20260812/retained/candidates/validated_five/primary/adapted_runtime --inflate-sh inflate.sh --gpu T4 --scorer-device cuda --expected-runtime-tree-sha256 auto --lane-id lane_ddm_cp5v_validated_five_contest_cuda_20260812 --instance-job-id ddm_cp5v_validated_five_t4_20260812 --claim-agent MAIN --claim-policy require_active --single-axis-waiver-reason 'CP135 F26 family is CUDA-locked; contest-CPU refused by vehicle precedent' --detach --provider-detach-ack
```

Harvest:

```bash
.venv/bin/python tools/recover_modal_auth_eval.py --output-dir /Volumes/VertigoDataTier/pact/ddm_cp5v_20260812/main_t4
```

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN exact contest-CUDA scorer owner. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_cp5v_20260812/main_t4`. Fire trigger: every live Modal/scorer claim is terminal, MAIN owns the sole scorer lane, and the archive SHA, size, and archive-pinned runtime still match; then execute the commands above exactly.**

The effective floor remains CP135 at **S = 0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]**. The own-vehicle frontier remains LC2 at **S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]**.

## LIVE-HYPOTHESES

- The five disjoint, n600-positive cells may retain most or all six Seg flips jointly; receiver identity makes this plausible, but the exact row is required.
- Joint Pose damage may remain negligible because the singleton global deltas sum to only `6.539362330040836e-09`.
- CP5V may improve CP135 without any rate penalty because the complete container has exactly the same byte count.

## DEAD-ENDS

- Widening the `+3 B` allowance is closed: the real compose costs 0 B.
- Treating `0.16195005201522092` as an exact score is closed. JO1’s six-event row already showed singleton interactions can reverse an optimistic projection.
- Shipping a separate EC1 sidecar or reusing the unchanged CP135 token stream is closed; the five cells are carried by a freshly exported HP3/RC64 probability object.
- Treating the constructed token plane as receiver evidence is closed; both the shipped RC64 backend and adapted canonical reader independently decoded the exact archive.