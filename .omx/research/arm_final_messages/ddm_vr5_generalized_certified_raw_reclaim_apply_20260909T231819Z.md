Plan ready: **17/20 raws admitted, 57.985 GiB**. WD2, HV1, and JG4 remain blocked by consumer references. **0 bytes reclaimed; no SSD apply attempted.**

38 tests passed, ruff clean, two review passes completed. Git writes were denied; verified fallback commit `43c4a084639a7ad96c097b857e47a0b510aaf613` is **not landed**.

[Memo](/Users/adpena/Projects/pact/.omx/research/ddm_vr5_generalized_certified_raw_reclaim_20260909.md) · [Exact apply command](/Users/adpena/Projects/pact/.omx/research/ddm_vr5_20260909/main_apply_command.txt)

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — MAIN: land the [verified bundle](/Users/adpena/Projects/pact/.omx/research/ddm_vr5_20260909/serializer/20260909T231659.675679Z-86018/intended-commit.bundle); consumer: main Git; trigger: Git-writable session.
- **QUEUED-WITH-A-FIRE-ORDER** — MAIN: execute the pinned command; consumer: VR5 plan/apply journal; trigger: bundle landed and process visibility available outside the sandbox.
- **QUEUED-WITH-A-FIRE-ORDER** — MAIN and consumer owners: recertify the three blocked rows; consumer: VR5 ledger; trigger: documented retirement or rebinding of every named consumer.

## LIVE-HYPOTHESES

- The admitted raws can release 57.985 GiB: all hashes and reproduction chains match; live clearance and actual free-space gain remain untested.

## DEAD-ENDS

- All-20 admission: three unresolved consumer references prevent it.
- Empty sandbox `lsof` as clearance: independent process visibility is required.

Own-vehicle frontier unchanged: **S 0.13791730003757818 @ 180,388 B [contest-CUDA T4 n600]**.