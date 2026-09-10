Implemented the timing seal leg, inheritance checks, and fire-tool refusal. **175 tests and Ruff pass.** Payloads are retained.

| Receiver | Local CPU proxy, n600 | Actual T4 inflate |
|---|---:|---:|
| Move40 | 1,243.975 s | 990.054 s |
| Move41/tc3 | 1,242.385 s | 1,336.669 s |
| tc4 | 1,456.543 s | ≥1,800 s; timeout |

The admission limit is **1,260 seconds**. Move41 and tc4 fail. Move40 has T4 margin, but unavailable local process-count evidence blocks a passing calibration. These proxies establish no contest-CPU readiness.

Measured map costs: tc3 **0.518 s/million tokens**; tc4 additional maps **0.271 s/million**. These exclude probability mixing and rendering.

Shared Git writes were denied. Both serializer commits are preserved in the verified [HEAD bundle](/Users/adpena/Projects/pact/.omx/research/ddm_dwc1_20260910/landing.bundle) and [patch](/Users/adpena/Projects/pact/.omx/research/ddm_dwc1_20260910/landing.patch); shared index unchanged. Full findings: [memo](/Users/adpena/Projects/pact/.omx/research/ddm_dwc1_decode_wall_clock_seal_leg_20260910.md).

No new score. Frontier: **S=0.13763861019288715 @180,233 B [contest-CUDA T4 n600]**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — MAIN:** integrate source then enforcement; consumer `ddm_dwc1_20260910/MAIN_LANDING_ORDER.json`; trigger: verified bundle harvest.
- **QUEUED-WITH-A-FIRE-ORDER — MAIN:** obtain admissible move40 concurrency evidence; consumer `ddm_dwc1_20260910/MAIN_MARGIN_DECISION.json`; trigger: timing-backfill harvest.
- **QUEUED-WITH-A-FIRE-ORDER — MAIN/rlc1:** check inheritance eligibility; consumer `ddm_dwc1_20260910/RLC1_INHERITANCE_COORDINATION.json`; trigger: receiver freeze before sealing.

**LIVE-HYPOTHESES**

- Move40 may admit a quiesced calibration: its actual T4 decode has 269.946 seconds of margin.
- Probability mixing, host/device overhead, or contention may explain tc4’s remaining cost; measured maps alone do not establish the cause.

**DEAD-ENDS**

- Move41 margin PASS: actual T4 decode exceeds the limit.
- Overriding tc4’s timeout with a projection: actual failure vetoes it.
- Treating unknown concurrency as zero, or inheriting a failed leg: explicitly refused.