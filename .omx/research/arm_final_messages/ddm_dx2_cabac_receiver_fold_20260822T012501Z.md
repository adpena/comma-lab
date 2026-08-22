Implemented, retained, tested, and committed as `8175310960`.

Measured `[macOS-CPU scorer-free exact byte/receiver parse-back]`:

- Candidate: [candidate_dx2_cabac.zip](/Volumes/APDataStore/pact/ddm_dx2/r7/retained/candidate_dx2_cabac.zip), **180,368 B**, SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`.
- Exact improvement: **−18 B**, rate ΔS `−1.1985461156199085e-05`.
- CABAC payload: 9,811 B, exact DX1 winner SHA `b93131a52674abb4ada677e1b6cf08eebc6afb94381136d23d010e70a287e210`.
- Deterministic repeat: byte-identical.
- Real receiver parse-back: all nine consumed fields identical; corrupted-payload negative control refused.
- Validation: 12 tests passed, Ruff clean, zero payload-retention findings, two review passes per Python file.

Full evidence and commands are in [ddm_dx2_cabac_receiver_fold_20260821.md](/Users/adpena/Projects/pact/.omx/research/ddm_dx2_cabac_receiver_fold_20260821.md:1).

The fresh 3,662,409,600-byte raw decode was not launched because JO4 still owns the governed heavy-local slot. Therefore the target seal `/Volumes/APDataStore/pact/ddm_dx2/r7/CANDIDATE_SEAL_dx2_fx5_cabac.json` does not yet exist, and no `SEAL_VALID`, `READY`, score, or frontier movement is claimed.

After raw identity and seal validation, MAIN’s exact fire command is:

```bash
.venv/bin/python tools/fire_modal_auth_eval.py \
  --seal /Volumes/APDataStore/pact/ddm_dx2/r7/CANDIDATE_SEAL_dx2_fx5_cabac.json \
  --output-dir /Volumes/APDataStore/pact/ddm_dx2/r7/t4_row_r1 \
  --lane-id lane_ddm_dx2_fx5_cabac_cuda_20260822 \
  --instance-job-id ddm_dx2_fx5_cabac_t4_r1 \
  --axis cuda
```

Own-vehicle frontier remains **S 0.14823186109359 @ 180,386 B `[contest-CUDA T4, n600]`**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER:** owner `MAIN/local decode custodian`; consumer store `/Volumes/APDataStore/pact/ddm_dx2/r7/decode_r1`; fire trigger: JO4’s local claim is terminal/released and storage preflight admits the raw output. Run the exact detached decode command in the memo.
- **QUEUED-BEHIND-RAW:** owner `MAIN`; consumer store `/Volumes/APDataStore/pact/ddm_dx2/r7/CANDIDATE_SEAL_dx2_fx5_cabac.json`; fire trigger: local `0.raw` is 3,662,409,600 B with SHA `7246a4ff…f5f2de7`.
- **QUEUED-BEHIND-SEAL:** owner `MAIN`; consumer store `/Volumes/APDataStore/pact/ddm_dx2/r7/t4_row_r1`; fire trigger: `SEAL_VALID` plus a non-duplicated CUDA lane claim. Run the fire command above.

## LIVE-HYPOTHESES

- The full raw output will match fx5 because every receiver-consumed field already reconstructs identically; only the required fresh-process render remains untested.
- Contest CUDA will preserve both distortion components, making the exact −18-byte rate term the only score change.
- A new rate representation remains necessary for sub-0.12 because DX2 exhausts the last measured 18 B of this body’s composable rate ceiling.

## DEAD-ENDS

- Re-running the DX1 coder/fruit search: the optimal-form 16-coder race already selected this exact winner.
- Treating RR5 and DX2 as colliding: they operate on disjoint basis and coefficient streams and compose losslessly.
- Reusing OV1’s withdrawn −0.001190 ceiling: it priced the wrong object.
- Device-dependent decoding: the shipped path uses integer-only probability and range updates.
- Sealing from parse-back alone or launching alongside JO4: both violate the charter’s raw-identity and lane-custody requirements.