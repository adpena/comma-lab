# Current Focus — 2026-04-22

## Context: 11 Days Remaining. MEASUREMENT DISASTER RECOVERED. TWO 4090s RUNNING.

**Deadline**: May 3, 2026 (11 days)
**TRUE score**: 2.01 [contest-compliant] (was falsely reporting 0.87)
**Target**: < 0.40 (beat Quantizr at 0.33 is stretch)

## What Happened (2026-04-21/22)

1. Found 13 CRITICAL measurement bugs — ALL prior scores were wrong
2. Masks.mkv at 48x64 was destroying score (103.27 → 2.01 with correct masks)
3. Mapped full CRF Pareto frontier: CRF50=421KB, CRF56=280KB
4. Deobfuscated Quantizr: 88K params, 293KB archive, FiLM+DSConv, 5-stage QAT
5. Verified scorer architectures: SegNet=EfficientNet-B2, PoseNet=FastViT-T12
6. Discovered Yousfi was Fridrich's PhD student → inverse steganalysis framework
7. Implemented 4 Fridrich inverse steganalysis losses
8. Built and debugged QAT fine-tuning script (4 crashes, all fixed)
9. QAT on MPS BLOCKED (nn.utils.parametrize backward incompatibility)
10. Two 4090s provisioned and running

## Running Now

| Instance | IP | Task | Status | Cost |
|----------|-----|------|--------|------|
| 35404589 | ssh2.vast.ai:14588 | Small renderer training | Phase 1 ep375 loss 0.57 | $0.245/hr |
| 35409315 | ssh6.vast.ai:19314 | QAT on current renderer | Loading... | $0.248/hr |

## Next Steps

1. Instance 2 comes online → sync code → launch QAT for current renderer
2. Instance 1 Phase 1 finishes → kill → restart Phase 2 (1000 epochs + Fridrich)
3. Download results → build archives → full e2e eval
4. Best of two paths → submission candidate

## Score Paths

- Path 1 (current renderer + QAT): archive 607KB, projected ~0.80
- Path 2 (small renderer + QAT): archive 348KB, projected ~0.43
