READY_TO_FIRE. EC2 is landed as a real scorer-free, resumable sparse-event HPAC implementation.

- Runner: [run_ddm_ec2_sparse_event_hpac_conditioning.py](/Users/adpena/Projects/pact/tools/run_ddm_ec2_sparse_event_hpac_conditioning.py)
- Memo: [ddm_ec2_sparse_event_hpac_20260812.md](/Users/adpena/Projects/pact/.omx/research/ddm_ec2_sparse_event_hpac_20260812.md)
- Commits: `d05b803022`, verification addendum `55ac5485f7`
- Tests: 6 passed; Ruff and compilation passed; two review passes completed; payload-discard findings: 0.
- Custody: 200/200 JS5 proposals verified; 234 unique sites.
- Counted coordinates: 447 B raw, 413 B Brotli-q11 selected, 473 B XZ. All candidates and repeats retained.
- Admission: exact receiver-consumed `model.xz + tokens.range + coordinates + 20 B framing < 116,716 B`; otherwise `FORMULATION_CLOSED_FULL_SCALE`.
- Memory cap: 6,144 MiB, derived from XI2’s measured 4,879.953 MiB peak plus 25.90% margin.
- Banked CL1 control remained untouched.
- No Metal training, scorer, candidate package result, or frontier movement was claimed.

Full developer preflight was 17/25 green; its eight red gates were unrelated baseline findings and named no EC2 file. They were neither waived nor modified.

Pinned command:

```bash
PYTHONHASHSEED=0 TAC_ADMISSION_ENFORCE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python tools/safe_run.py --rss-mb 6144 --projected-gib 6 --timeout 7200 --label ddm_ec2_sparse_event_hpac_n600 --status-receipt /Volumes/VertigoDataTier/pact/ddm_ec2/run/main.safe_run.json -- .venv/bin/python tools/run_ddm_ec2_sparse_event_hpac_conditioning.py --leg all --resume-from auto
```

Own-vehicle frontier remains **LC2 S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]**.

## NEXT_IF_RESUMED

- **Disposition:** QUEUED-WITH-A-FIRE-ORDER · **Owner:** MAIN Metal executor · **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_ec2/FULL_SCALE_RESULT.json` · **Fire trigger:** Metal lane free, MPS available, custody/storage pins valid, and governed 6 GiB admission passes; then execute the pinned command above.

## LIVE-HYPOTHESES

- The 234 receiver-effective topology sites may carry disproportionate HPAC surprise, allowing the learned 3×3 prior to amortize its model and 413 B coordinate cost.
- Zero-start additive conditioning may preserve CL1 globally while specializing only near marked events.
- Typed source/target/event channels may outperform occupancy alone, but constitute a separate counted formulation.

## DEAD-ENDS

- Free EC1 coordinates: closed because they are video-derived and not decoder-derived.
- Full 44,410 B SP1 sidecar or dense/global raster prior: closed by existing price and temporal evidence.
- Assuming learned priors win from conditional loss: closed by XI1’s exact CAP1-beats-CPR1 counterexample.
- Reusing the 4,096 MiB cap: closed by XI2’s measured OOM and 4,879.953 MiB successful peak.
- Retraining or altering the banked CL1 control: forbidden and not done.